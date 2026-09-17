from scripts.train_grpo import (
    build_parser,
    ground_truth_from_answer,
    load_dataset_and_format_qa,
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
