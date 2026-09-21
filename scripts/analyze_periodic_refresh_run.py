"""Summarize a periodic random-refresh run without optional dependencies."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


VANILLA_SEED42_ACCURACY = 0.844579226686884
MIN_ACCURACY = VANILLA_SEED42_ACCURACY - 0.005
MIN_EFFECTIVE_GROUPS_PER_MILLION = 165.0
MIN_PROMPT_COVERAGE = 0.05
MAX_WRONG_FORMAT = 30


def load_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_rollouts(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    rollout_tokens = sum(
        int(record["rollout/total_rollout_tokens"]) for record in rollouts
    )
    attempted_groups = sum(
        int(record["sampling/attempted_groups"]) for record in rollouts
    )

    def group_count(ratio_key: str) -> float:
        return sum(
            float(record[ratio_key])
            * int(record["sampling/attempted_groups"])
            for record in rollouts
        )

    effective_groups = group_count("rollout/effective_group_ratio")
    all_correct_groups = group_count("rollout/all_correct_group_ratio")
    all_wrong_groups = group_count("rollout/all_wrong_group_ratio")
    return {
        "rollout_batches": len(rollouts),
        "rollout_tokens": rollout_tokens,
        "attempted_groups": attempted_groups,
        "effective_groups": effective_groups,
        "effective_group_ratio": (
            effective_groups / attempted_groups if attempted_groups else 0.0
        ),
        "zero_variance_group_ratio": (
            1.0 - effective_groups / attempted_groups
            if attempted_groups
            else 0.0
        ),
        "all_correct_group_ratio": (
            all_correct_groups / attempted_groups if attempted_groups else 0.0
        ),
        "all_wrong_group_ratio": (
            all_wrong_groups / attempted_groups if attempted_groups else 0.0
        ),
        "effective_groups_per_million_tokens": (
            effective_groups / rollout_tokens * 1_000_000
            if rollout_tokens
            else 0.0
        ),
        "ending_observed_prompts": rollouts[-1].get(
            "difficulty/observed_prompts_after"
        ),
        "ending_prompt_coverage": rollouts[-1].get(
            "difficulty/observed_prompt_ratio_after"
        ),
    }


def evaluation_summary(
    record: dict[str, Any] | None,
    total_rollout_tokens: int,
) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "model_step": record["model_step"],
        "cumulative_rollout_tokens": record.get(
            "rollout/cumulative_rollout_tokens",
            total_rollout_tokens,
        ),
        "accuracy": record["eval/accuracy"],
        "correct": record["eval/correct"],
        "wrong_answer": record["eval/wrong_answer"],
        "wrong_format": record["eval/wrong_format"],
        "count": record["eval/count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = load_records(args.run_dir / "metrics.jsonl")
    rollouts = [record for record in records if record.get("event") == "rollout"]
    if not rollouts:
        raise ValueError("run contains no rollout records")

    total_rollout_tokens = int(
        rollouts[-1]["rollout/cumulative_rollout_tokens"]
    )
    overall = summarize_rollouts(rollouts)

    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cycle: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in rollouts:
        by_phase[str(record["sampling/active_strategy"])].append(record)
        by_cycle[int(record["sampling/schedule_cycle_index"])].append(record)

    evaluations = [
        record for record in records if record.get("event") == "evaluation"
    ]
    full_count = max(
        (int(record["eval/count"]) for record in evaluations),
        default=0,
    )
    full_evaluations = [
        record
        for record in evaluations
        if int(record["eval/count"]) == full_count
    ]
    final_evaluation = full_evaluations[-1] if full_evaluations else None
    final_summary = evaluation_summary(
        final_evaluation,
        total_rollout_tokens,
    )
    final_coverage = float(
        overall.get("ending_prompt_coverage") or 0.0
    )

    screen = None
    if final_summary is not None:
        screen = {
            "thresholds": {
                "min_effective_groups_per_million_tokens": (
                    MIN_EFFECTIVE_GROUPS_PER_MILLION
                ),
                "min_accuracy": MIN_ACCURACY,
                "min_prompt_coverage": MIN_PROMPT_COVERAGE,
                "max_wrong_format": MAX_WRONG_FORMAT,
            },
            "passes_efficiency": (
                overall["effective_groups_per_million_tokens"]
                >= MIN_EFFECTIVE_GROUPS_PER_MILLION
            ),
            "passes_accuracy": final_summary["accuracy"] >= MIN_ACCURACY,
            "passes_coverage": final_coverage >= MIN_PROMPT_COVERAGE,
            "passes_format": final_summary["wrong_format"] <= MAX_WRONG_FORMAT,
        }
        screen["passes_all"] = all(
            value
            for key, value in screen.items()
            if key.startswith("passes_") and key != "passes_all"
        )

    summary = {
        "run_dir": str(args.run_dir),
        "total_rollout_tokens": total_rollout_tokens,
        "overall": overall,
        "phases": {
            phase: summarize_rollouts(phase_rollouts)
            for phase, phase_rollouts in by_phase.items()
        },
        "cycles": {
            str(cycle): {
                "overall": summarize_rollouts(cycle_rollouts),
                "phases": {
                    phase: summarize_rollouts(
                        [
                            record
                            for record in cycle_rollouts
                            if record["sampling/active_strategy"] == phase
                        ]
                    )
                    for phase in sorted(
                        {
                            str(record["sampling/active_strategy"])
                            for record in cycle_rollouts
                        }
                    )
                },
            }
            for cycle, cycle_rollouts in sorted(by_cycle.items())
        },
        "phase_switches": [
            record
            for record in records
            if record.get("event") == "sampling_phase_switch"
        ],
        "final_full_evaluation": final_summary,
        "screen": screen,
    }
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
