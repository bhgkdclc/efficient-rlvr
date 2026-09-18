from types import SimpleNamespace

from scripts.train_grpo import (
    add_rollout_token_counts,
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
