"""Summarize an importance-corrected difficulty run without pandas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VANILLA_SEED42_ACCURACY = 0.844579226686884
MIN_ACCURACY = VANILLA_SEED42_ACCURACY - 0.005
MIN_EFFECTIVE_GROUPS_PER_MILLION = 165.0
MAX_WRONG_FORMAT = 30


def load_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = load_records(args.run_dir / "metrics.jsonl")
    rollouts = [record for record in records if record.get("event") == "rollout"]
    evaluations = [
        record for record in records if record.get("event") == "evaluation"
    ]
    if not rollouts:
        raise ValueError("run contains no rollout records")

    total_tokens = int(rollouts[-1]["rollout/cumulative_rollout_tokens"])
    attempted_groups = sum(
        int(record["sampling/attempted_groups"]) for record in rollouts
    )
    effective_groups = sum(
        float(record["rollout/effective_group_ratio"])
        * int(record["sampling/attempted_groups"])
        for record in rollouts
    )
    weighted_effective_groups = sum(
        float(record["importance/weighted_effective_groups"])
        for record in rollouts
    )

    full_count = max(
        (int(record["eval/count"]) for record in evaluations),
        default=0,
    )
    full_evaluations = [
        record
        for record in evaluations
        if int(record["eval/count"]) == full_count
    ]
    final_eval = full_evaluations[-1]

    def weighted_batch_mean(key: str) -> float:
        return sum(
            float(record[key]) * int(record["sampling/accepted_groups"])
            for record in rollouts
        ) / sum(int(record["sampling/accepted_groups"]) for record in rollouts)

    total_ess = sum(
        float(record["importance/effective_sample_size"])
        for record in rollouts
    )
    final_coverage = float(
        rollouts[-1]["difficulty/observed_prompt_ratio_after"]
    )
    efficiency = effective_groups / total_tokens * 1_000_000
    weighted_efficiency = (
        weighted_effective_groups / total_tokens * 1_000_000
    )
    accuracy = float(final_eval["eval/accuracy"])
    wrong_format = int(final_eval["eval/wrong_format"])

    screen = {
        "thresholds": {
            "min_effective_groups_per_million_tokens": (
                MIN_EFFECTIVE_GROUPS_PER_MILLION
            ),
            "min_accuracy": MIN_ACCURACY,
            "max_wrong_format": MAX_WRONG_FORMAT,
        },
        "passes_efficiency": efficiency >= MIN_EFFECTIVE_GROUPS_PER_MILLION,
        "passes_accuracy": accuracy >= MIN_ACCURACY,
        "passes_format": wrong_format <= MAX_WRONG_FORMAT,
    }
    screen["passes_all"] = all(
        value
        for key, value in screen.items()
        if key.startswith("passes_") and key != "passes_all"
    )

    summary = {
        "run_dir": str(args.run_dir),
        "total_rollout_tokens": total_tokens,
        "groups": {
            "attempted": attempted_groups,
            "effective": effective_groups,
            "effective_group_ratio": effective_groups / attempted_groups,
            "zero_variance_group_ratio": 1.0 - effective_groups / attempted_groups,
            "effective_groups_per_million_tokens": efficiency,
            "importance_weighted_effective_groups": weighted_effective_groups,
            "importance_weighted_effective_groups_per_million_tokens": (
                weighted_efficiency
            ),
        },
        "importance": {
            "raw_mean": weighted_batch_mean("importance/raw_mean"),
            "raw_min": min(
                float(record["importance/raw_min"]) for record in rollouts
            ),
            "raw_max": max(
                float(record["importance/raw_max"]) for record in rollouts
            ),
            "clipped_fraction": weighted_batch_mean(
                "importance/clipped_fraction"
            ),
            "mean_batch_ess_ratio": weighted_batch_mean(
                "importance/effective_sample_size_ratio"
            ),
            "aggregate_batch_ess_ratio": total_ess / attempted_groups,
        },
        "ending_prompt_coverage": final_coverage,
        "final_full_evaluation": {
            "model_step": int(final_eval["model_step"]),
            "cumulative_rollout_tokens": total_tokens,
            "accuracy": accuracy,
            "correct": int(final_eval["eval/correct"]),
            "wrong_answer": int(final_eval["eval/wrong_answer"]),
            "wrong_format": wrong_format,
            "count": int(final_eval["eval/count"]),
        },
        "screen": screen,
    }
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
