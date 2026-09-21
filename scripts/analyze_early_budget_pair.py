"""Compare a paired Vanilla/Fast-EMA run at the same rollout-token budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def summarize(run_dir: Path) -> dict[str, Any]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    records = load_records(run_dir)
    rollouts = [record for record in records if record.get("event") == "rollout"]
    evaluations = [
        record for record in records if record.get("event") == "evaluation"
    ]
    if not rollouts or not evaluations:
        raise ValueError(f"incomplete run: {run_dir}")

    total_tokens = int(rollouts[-1]["rollout/cumulative_rollout_tokens"])
    attempted_groups = sum(
        int(record["sampling/attempted_groups"]) for record in rollouts
    )
    effective_groups = sum(
        float(record["rollout/effective_group_ratio"])
        * int(record["sampling/attempted_groups"])
        for record in rollouts
    )
    full_count = max(int(record["eval/count"]) for record in evaluations)
    final_eval = [
        record
        for record in evaluations
        if int(record["eval/count"]) == full_count
    ][-1]

    return {
        "run_dir": str(run_dir),
        "seed": int(config["seed"]),
        "sampling_strategy": config["sampling_strategy"],
        "total_rollout_tokens": total_tokens,
        "attempted_groups": attempted_groups,
        "effective_groups": effective_groups,
        "zero_variance_group_ratio": 1.0
        - effective_groups / attempted_groups,
        "effective_groups_per_million_tokens": (
            effective_groups / total_tokens * 1_000_000
        ),
        "full_evaluation": {
            "accuracy": float(final_eval["eval/accuracy"]),
            "correct": int(final_eval["eval/correct"]),
            "wrong_answer": int(final_eval["eval/wrong_answer"]),
            "wrong_format": int(final_eval["eval/wrong_format"]),
            "count": full_count,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vanilla_run", type=Path)
    parser.add_argument("difficulty_run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    vanilla = summarize(args.vanilla_run)
    difficulty = summarize(args.difficulty_run)
    if vanilla["seed"] != difficulty["seed"]:
        raise ValueError("paired runs must use the same seed")

    vanilla_eval = vanilla["full_evaluation"]
    difficulty_eval = difficulty["full_evaluation"]
    accuracy_delta = (
        difficulty_eval["accuracy"] - vanilla_eval["accuracy"]
    ) * 100
    efficiency_gain = (
        difficulty["effective_groups_per_million_tokens"]
        / vanilla["effective_groups_per_million_tokens"]
        - 1.0
    )

    summary = {
        "experiment_status": "exploratory_follow_up",
        "seed": vanilla["seed"],
        "vanilla": vanilla,
        "fast_ema": difficulty,
        "comparison": {
            "fast_ema_accuracy_delta_points": accuracy_delta,
            "fast_ema_relative_effective_group_efficiency_gain": (
                efficiency_gain
            ),
            "fast_ema_zero_variance_delta_points": (
                difficulty["zero_variance_group_ratio"]
                - vanilla["zero_variance_group_ratio"]
            )
            * 100,
        },
        "screen": {
            "both_full_evaluations": (
                vanilla_eval["count"] == 1319
                and difficulty_eval["count"] == 1319
            ),
            "matched_budget_within_one_percent": (
                abs(
                    vanilla["total_rollout_tokens"]
                    - difficulty["total_rollout_tokens"]
                )
                / max(
                    vanilla["total_rollout_tokens"],
                    difficulty["total_rollout_tokens"],
                )
                <= 0.01
            ),
            "fast_ema_higher_accuracy": accuracy_delta > 0.0,
            "fast_ema_higher_effective_group_efficiency": efficiency_gain > 0.0,
        },
    }
    summary["screen"]["passes_seed_screen"] = all(
        summary["screen"].values()
    )

    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
