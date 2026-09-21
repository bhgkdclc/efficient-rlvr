from types import SimpleNamespace

import pytest

from scripts.train_grpo import (
    add_rollout_token_counts,
    aggregate_difficulty_round_metadata,
    build_parser,
    empty_rollout_token_counts,
    ground_truth_from_answer,
    load_dataset_and_format_qa,
    rollout_group_token_counts,
    rollout_token_counts,
    resolve_sampling_strategy,
)


def test_raw_gsm8k_path_skips_sft_artifact():
    args = build_parser().parse_args(["--output-path", "unused"])
    train_data, test_data = load_dataset_and_format_qa(args)

    assert len(train_data) == 7473
    assert len(test_data) == 1319
    assert "Natalia sold clips" in train_data[0]["prompt"]
    assert ground_truth_from_answer(train_data[0]["answer"]) == "72"


def test_parser_accepts_dynamic_sampling():
    args = build_parser().parse_args(
        ["--output-path", "unused", "--sampling-strategy", "dynamic"]
    )

    assert args.sampling_strategy == "dynamic"


def test_parser_accepts_difficulty_dynamic_sampling():
    args = build_parser().parse_args(
        [
            "--output-path",
            "unused",
            "--sampling-strategy",
            "difficulty_dynamic",
            "--difficulty-ema-beta",
            "0.5",
        ]
    )

    assert args.sampling_strategy == "difficulty_dynamic"
    assert args.difficulty_ema_beta == 0.5


def test_parser_can_disable_final_checkpoint():
    args = build_parser().parse_args(
        ["--output-path", "unused", "--no-save-final-checkpoint"]
    )

    assert args.save_final_checkpoint is False


def test_parser_accepts_difficulty_sampling_configuration():
    args = build_parser().parse_args(
        [
            "--output-path",
            "unused",
            "--sampling-strategy",
            "difficulty",
            "--difficulty-ema-beta",
            "0.8",
            "--sampling-uniform-epsilon",
            "0.2",
            "--difficulty-warmup-groups",
            "64",
            "--difficulty-coverage-weight",
            "0.3",
        ]
    )

    assert args.sampling_strategy == "difficulty"
    assert args.difficulty_ema_beta == 0.8
    assert args.sampling_uniform_epsilon == 0.2
    assert args.difficulty_warmup_groups == 64
    assert args.difficulty_coverage_weight == 0.3


def test_parser_accepts_prompt_importance_correction():
    args = build_parser().parse_args(
        [
            "--output-path",
            "unused",
            "--sampling-strategy",
            "difficulty",
            "--prompt-importance-correction",
            "--importance-weight-clip-min",
            "0.25",
            "--importance-weight-clip-max",
            "4.0",
        ]
    )

    assert args.prompt_importance_correction is True
    assert args.importance_weight_clip_min == 0.25
    assert args.importance_weight_clip_max == 4.0


def test_parser_accepts_difficulty_then_random_schedule():
    args = build_parser().parse_args(
        [
            "--output-path",
            "unused",
            "--sampling-strategy",
            "difficulty_then_random",
            "--difficulty-switch-rollout-tokens",
            "2000000",
        ]
    )

    assert args.sampling_strategy == "difficulty_then_random"
    assert args.difficulty_switch_rollout_tokens == 2_000_000
    assert args.eval_at_sampling_switch is True


def test_difficulty_then_random_switches_at_token_boundary():
    assert resolve_sampling_strategy(
        "difficulty_then_random", 1_999_999, 2_000_000
    ) == "difficulty"
    assert resolve_sampling_strategy(
        "difficulty_then_random", 2_000_000, 2_000_000
    ) == "random"
    assert resolve_sampling_strategy("random", 2_000_000, 0) == "random"


def test_difficulty_then_random_requires_positive_switch_budget():
    with pytest.raises(ValueError, match="must be positive"):
        resolve_sampling_strategy("difficulty_then_random", 0, 0)


def test_parser_accepts_periodic_random_refresh_schedule():
    args = build_parser().parse_args(
        [
            "--output-path",
            "unused",
            "--sampling-strategy",
            "difficulty_periodic_random",
            "--difficulty-refresh-cycle-tokens",
            "1000000",
            "--difficulty-refresh-random-tokens",
            "200000",
        ]
    )

    assert args.sampling_strategy == "difficulty_periodic_random"
    assert args.difficulty_refresh_cycle_tokens == 1_000_000
    assert args.difficulty_refresh_random_tokens == 200_000


@pytest.mark.parametrize(
    ("rollout_tokens", "expected"),
    [
        (0, "random"),
        (199_999, "random"),
        (200_000, "difficulty"),
        (999_999, "difficulty"),
        (1_000_000, "random"),
        (1_199_999, "random"),
        (1_200_000, "difficulty"),
        (3_999_999, "difficulty"),
    ],
)
def test_periodic_random_refresh_resolves_cycle_phase(
    rollout_tokens,
    expected,
):
    assert resolve_sampling_strategy(
        "difficulty_periodic_random",
        rollout_tokens,
        0,
        1_000_000,
        200_000,
    ) == expected


@pytest.mark.parametrize(
    ("cycle_tokens", "random_tokens"),
    [(0, 200_000), (1_000_000, 0), (1_000_000, 1_000_000)],
)
def test_periodic_random_refresh_rejects_invalid_schedule(
    cycle_tokens,
    random_tokens,
):
    with pytest.raises(ValueError, match="difficulty_refresh"):
        resolve_sampling_strategy(
            "difficulty_periodic_random",
            0,
            0,
            cycle_tokens,
            random_tokens,
        )


def test_parser_accepts_hybrid_sampling_configuration():
    args = build_parser().parse_args(
        [
            "--output-path",
            "unused",
            "--sampling-strategy",
            "hybrid",
            "--hybrid-uniform-fraction",
            "0.5",
        ]
    )

    assert args.sampling_strategy == "hybrid"
    assert args.hybrid_uniform_fraction == 0.5


def test_rollout_costs_include_length_truncation():
    outputs = [
        SimpleNamespace(
            prompt_token_ids=[1, 2, 3],
            outputs=[
                SimpleNamespace(token_ids=[4, 5], finish_reason="stop"),
                SimpleNamespace(token_ids=[6, 7, 8], finish_reason="length"),
            ],
        )
    ]

    counts = rollout_token_counts(outputs)
    group_counts = rollout_group_token_counts(outputs[0])
    totals = empty_rollout_token_counts()
    add_rollout_token_counts(totals, group_counts, response_count=2)

    assert counts["generated_response_tokens"] == 5
    assert counts["prompt_tokens"] == 6
    assert counts["length_truncated_responses"] == 1
    assert counts["length_truncated_response_ratio"] == 0.5
    assert totals["length_truncated_responses"] == 1
    assert totals["length_truncated_response_ratio"] == 0.5


def test_difficulty_metadata_is_weighted_across_resampling_rounds():
    def metadata(
        *,
        warmup,
        observed_before,
        observed_after,
        seen_ratio,
        ema_accuracy,
        group_accuracy,
    ):
        return {
            "warmup_active": warmup,
            "observed_prompts_before": observed_before,
            "observed_prompt_ratio_before": observed_before / 10,
            "observed_prompts_after": observed_after,
            "observed_prompt_ratio_after": observed_after / 10,
            "selected_seen_ratio": seen_ratio,
            "selected_unseen_ratio": 1.0 - seen_ratio,
            "selected_sample_count_mean_before": 1.0,
            "selected_coverage_score_mean": 0.5,
            "selected_ema_accuracy_mean": ema_accuracy,
            "selected_boundary_score_mean": 0.75,
            "global_seen_ema_accuracy_mean": ema_accuracy,
            "selected_group_accuracy_mean": group_accuracy,
            "selected_seen_group_accuracy_mean": group_accuracy,
            "selected_seen_effective_group_ratio": 0.5,
            "selected_sample_count_mean_after": 2.0,
        }

    combined = aggregate_difficulty_round_metadata(
        [
            (
                2,
                metadata(
                    warmup=True,
                    observed_before=2,
                    observed_after=4,
                    seen_ratio=0.5,
                    ema_accuracy=0.4,
                    group_accuracy=0.5,
                ),
            ),
            (
                1,
                metadata(
                    warmup=False,
                    observed_before=4,
                    observed_after=5,
                    seen_ratio=1.0,
                    ema_accuracy=0.8,
                    group_accuracy=1.0,
                ),
            ),
        ]
    )

    assert combined["warmup_active"] is True
    assert combined["observed_prompts_before"] == 2
    assert combined["observed_prompts_after"] == 5
    assert combined["selected_seen_ratio"] == pytest.approx(2 / 3)
    assert combined["selected_ema_accuracy_mean"] == pytest.approx(0.6)
    assert combined["selected_group_accuracy_mean"] == pytest.approx(2 / 3)
