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
