"""Analyze a prompt-matched Vanilla/Fast-EMA rollout-token budget curve."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_MILESTONES = [1_000_000, 2_000_000, 3_000_000, 4_000_000]


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def prompt_hash(indices: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(indices, separators=(",", ":")).encode()
    ).hexdigest()


def cumulative_rollout_summary(
    rollouts: list[dict[str, Any]],
    actual_tokens: int,
) -> dict[str, float | int | None]:
    selected = [
        record
        for record in rollouts
        if int(record["rollout/cumulative_rollout_tokens"]) <= actual_tokens
    ]
    if not selected:
        raise ValueError(f"no rollouts found at token budget {actual_tokens}")

    attempted_groups = sum(
        int(record["sampling/attempted_groups"]) for record in selected
    )

    def group_count(metric: str) -> float:
        return sum(
            float(record[metric]) * int(record["sampling/attempted_groups"])
            for record in selected
        )

    effective_groups = group_count("rollout/effective_group_ratio")
    all_correct_groups = group_count("rollout/all_correct_group_ratio")
    all_wrong_groups = group_count("rollout/all_wrong_group_ratio")
    final_rollout = selected[-1]
    coverage = final_rollout.get("difficulty/observed_prompt_ratio_after")
    return {
        "rollout_batches": len(selected),
        "actual_rollout_tokens": actual_tokens,
        "attempted_groups": attempted_groups,
        "effective_groups": effective_groups,
        "effective_group_ratio": effective_groups / attempted_groups,
        "zero_variance_group_ratio": 1.0
        - effective_groups / attempted_groups,
        "all_correct_group_ratio": all_correct_groups / attempted_groups,
        "all_wrong_group_ratio": all_wrong_groups / attempted_groups,
        "effective_groups_per_million_tokens": (
            effective_groups / actual_tokens * 1_000_000
        ),
        "prompt_coverage": float(coverage) if coverage is not None else None,
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    records = load_records(run_dir)
    rollouts = [record for record in records if record.get("event") == "rollout"]
    evaluations = [
        record
        for record in records
        if record.get("event") == "evaluation"
        and record.get("eval/stage") == "rollout_token_milestone"
    ]
    if not rollouts:
        raise ValueError(f"no rollout records in {run_dir}")

    evaluations_by_milestone = {
        int(record["eval/requested_rollout_token_milestone"]): record
        for record in evaluations
    }
    missing = sorted(set(EXPECTED_MILESTONES) - set(evaluations_by_milestone))
    if missing:
        raise ValueError(f"missing full evaluations at milestones {missing}: {run_dir}")

    warmup_groups = int(config["difficulty_warmup_groups"])
    recorded_prompt_indices = [
        int(index)
        for record in rollouts
        for index in record.get("sampling/prompt_indices", [])
    ]
    warmup_prompt_indices = recorded_prompt_indices[:warmup_groups]

    milestones: dict[str, Any] = {}
    for requested in EXPECTED_MILESTONES:
        evaluation = evaluations_by_milestone[requested]
        actual = int(evaluation["rollout/cumulative_rollout_tokens"])
        milestones[str(requested)] = {
            "requested_rollout_tokens": requested,
            "actual_rollout_tokens": actual,
            "overshoot_tokens": actual - requested,
            "model_step": int(evaluation["model_step"]),
            "accuracy": float(evaluation["eval/accuracy"]),
            "correct": int(evaluation["eval/correct"]),
            "wrong_answer": int(evaluation["eval/wrong_answer"]),
            "wrong_format": int(evaluation["eval/wrong_format"]),
            "eval_count": int(evaluation["eval/count"]),
            **cumulative_rollout_summary(rollouts, actual),
        }

    return {
        "run_dir": str(run_dir),
        "seed": int(config["seed"]),
        "git_commit": config.get("git_commit"),
        "git_is_dirty": config.get("git_is_dirty"),
        "sampling_strategy": config["sampling_strategy"],
        "matched_random_warmup": bool(
            config.get("matched_random_warmup", False)
        ),
        "warmup_groups": warmup_groups,
        "warmup_prompt_count_recorded": len(warmup_prompt_indices),
        "warmup_prompt_hash": (
            prompt_hash(warmup_prompt_indices)
            if len(warmup_prompt_indices) == warmup_groups
            else None
        ),
        "milestones": milestones,
    }


def compare_milestone(
    vanilla: dict[str, Any],
    difficulty: dict[str, Any],
    milestone: int,
) -> dict[str, Any]:
    vanilla_point = vanilla["milestones"][str(milestone)]
    difficulty_point = difficulty["milestones"][str(milestone)]
    vanilla_efficiency = float(
        vanilla_point["effective_groups_per_million_tokens"]
    )
    difficulty_efficiency = float(
        difficulty_point["effective_groups_per_million_tokens"]
    )
    return {
        "requested_rollout_tokens": milestone,
        "vanilla": vanilla_point,
        "fast_ema": difficulty_point,
        "budget_difference_ratio": abs(
            int(difficulty_point["actual_rollout_tokens"])
            - int(vanilla_point["actual_rollout_tokens"])
        )
        / max(
            int(difficulty_point["actual_rollout_tokens"]),
            int(vanilla_point["actual_rollout_tokens"]),
        ),
        "fast_ema_accuracy_delta_points": (
            float(difficulty_point["accuracy"])
            - float(vanilla_point["accuracy"])
        )
        * 100,
        "fast_ema_relative_effective_group_efficiency_gain": (
            difficulty_efficiency / vanilla_efficiency - 1.0
        ),
        "fast_ema_zero_variance_delta_points": (
            float(difficulty_point["zero_variance_group_ratio"])
            - float(vanilla_point["zero_variance_group_ratio"])
        )
        * 100,
        "fast_ema_prompt_coverage_delta_points": (
            float(difficulty_point["prompt_coverage"])
            - float(vanilla_point["prompt_coverage"])
        )
        * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vanilla_run", type=Path)
    parser.add_argument("difficulty_run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    vanilla = summarize_run(args.vanilla_run)
    difficulty = summarize_run(args.difficulty_run)
    if vanilla["seed"] != difficulty["seed"]:
        raise ValueError("paired runs must use the same seed")

    comparisons = {
        str(milestone): compare_milestone(vanilla, difficulty, milestone)
        for milestone in EXPECTED_MILESTONES
    }
    summary = {
        "experiment_status": "held_out_seed_budget_curve",
        "seed": vanilla["seed"],
        "vanilla": vanilla,
        "fast_ema": difficulty,
        "comparison_by_milestone": comparisons,
        "screen": {
            "matched_warmup_prompts": (
                vanilla["warmup_prompt_count_recorded"]
                == vanilla["warmup_groups"]
                == difficulty["warmup_prompt_count_recorded"]
                == difficulty["warmup_groups"]
                and vanilla["warmup_prompt_hash"]
                == difficulty["warmup_prompt_hash"]
            ),
            "all_four_full_evaluations": all(
                point["vanilla"]["eval_count"] == 1319
                and point["fast_ema"]["eval_count"] == 1319
                for point in comparisons.values()
            ),
            "all_budget_differences_within_three_percent": all(
                point["budget_difference_ratio"] <= 0.03
                for point in comparisons.values()
            ),
        },
    }

    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
