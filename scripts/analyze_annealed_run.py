"""Summarize a difficulty-to-random run without optional data dependencies."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_phase(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    rollout_tokens = sum(
        int(record["rollout/total_rollout_tokens"]) for record in rollouts
    )
    attempted_groups = sum(
        int(record["sampling/attempted_groups"]) for record in rollouts
    )
    effective_groups = sum(
        float(record["rollout/effective_group_ratio"])
        * int(record["sampling/attempted_groups"])
        for record in rollouts
    )
    return {
        "rollout_batches": len(rollouts),
        "rollout_tokens": rollout_tokens,
        "attempted_groups": attempted_groups,
        "effective_groups": effective_groups,
        "effective_group_ratio": (
            effective_groups / attempted_groups if attempted_groups else 0.0
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


def evaluation_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "model_step": record["model_step"],
        "cumulative_rollout_tokens": record.get(
            "rollout/cumulative_rollout_tokens"
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

    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rollouts:
        by_phase[str(record["sampling/active_strategy"])].append(record)

    switches = [
        record for record in records if record.get("event") == "sampling_switch"
    ]
    evaluations = [
        record for record in records if record.get("event") == "evaluation"
    ]
    switch_evaluation = next(
        (
            record
            for record in evaluations
            if record.get("eval/stage") == "sampling_switch"
        ),
        None,
    )
    full_count = max((int(record["eval/count"]) for record in evaluations), default=0)
    full_evaluations = [
        record for record in evaluations if int(record["eval/count"]) == full_count
    ]
    final_evaluation = full_evaluations[-1] if full_evaluations else None

    summary = {
        "run_dir": str(args.run_dir),
        "total_rollout_tokens": int(
            rollouts[-1]["rollout/cumulative_rollout_tokens"]
        ),
        "switch": switches[-1] if switches else None,
        "phases": {
            phase: summarize_phase(phase_rollouts)
            for phase, phase_rollouts in by_phase.items()
        },
        "switch_full_evaluation": evaluation_summary(switch_evaluation),
        "final_full_evaluation": evaluation_summary(final_evaluation),
    }
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
