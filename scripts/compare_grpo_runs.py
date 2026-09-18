"""Compare GRPO sampling strategies under a rollout-token budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla-dir", type=Path, required=True)
    parser.add_argument("--dynamic-dir", type=Path, required=True)
    parser.add_argument("--difficulty-dir", type=Path)
    parser.add_argument("--difficulty-beta05-dir", type=Path)
    parser.add_argument("--difficulty-coverage-dir", type=Path)
    parser.add_argument("--difficulty-frontloaded-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_run(path: Path) -> tuple[dict, pd.DataFrame]:
    with (path / "summary.json").open(encoding="utf-8") as file:
        summary = json.load(file)
    evaluations = pd.read_csv(path / "evaluation_metrics.csv")
    return summary, evaluations


def fixed_evaluations(frame: pd.DataFrame) -> pd.DataFrame:
    sample_count = int(frame["eval/count"].value_counts().index[0])
    return frame[frame["eval/count"] == sample_count].copy()


def plot_accuracy(
    runs: dict[str, tuple[dict, pd.DataFrame]], output_dir: Path
) -> None:
    fig, ax = plt.subplots(
        figsize=(8.8, 5.2) if len(runs) > 5 else (7.4, 4.6)
    )
    colors = {
        "Vanilla": "#1f77b4",
        "Dynamic": "#d95f02",
        "Difficulty-Aware": "#2ca02c",
        "Difficulty beta=0.9": "#2ca02c",
        "Difficulty beta=0.5": "#9467bd",
        "Difficulty coverage=0.3": "#8c564b",
        "Front-loaded 512": "#e377c2",
    }
    for label, (summary, evaluations) in runs.items():
        fixed = fixed_evaluations(evaluations)
        ax.plot(
            fixed["cumulative_rollout_tokens"] / 1e6,
            fixed["eval/accuracy"] * 100,
            marker="o",
            linewidth=2,
            markersize=4,
            color=colors[label],
            label=f"{label} (fixed n=256)",
        )
        full = summary["final_full_evaluation"]
        total_tokens = summary["rollout_cost"]["total_rollout_tokens"]
        ax.scatter(
            total_tokens / 1e6,
            full["accuracy"] * 100,
            marker="X",
            s=90,
            color=colors[label],
            zorder=3,
            label=f"{label} (full n={full['sample_count']})",
        )
    ax.set_xlabel("Cumulative rollout tokens (millions)")
    ax.set_ylabel("Pass@1 / accuracy (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=3 if len(runs) > 5 else 2)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_rollout_tokens.png", dpi=180)
    plt.close(fig)


def plot_efficiency(runs: dict[str, tuple[dict, pd.DataFrame]], output_dir: Path) -> None:
    labels = list(runs)
    effective_per_million = [
        runs[label][0]["groups"]["effective_groups_per_million_rollout_tokens"]
        for label in labels
    ]
    final_accuracy = [
        runs[label][0]["final_full_evaluation"]["accuracy"] * 100
        for label in labels
    ]
    color_map = {
        "Vanilla": "#1f77b4",
        "Dynamic": "#d95f02",
        "Difficulty-Aware": "#2ca02c",
        "Difficulty beta=0.9": "#2ca02c",
        "Difficulty beta=0.5": "#9467bd",
        "Difficulty coverage=0.3": "#8c564b",
        "Front-loaded 512": "#e377c2",
    }
    colors = [color_map[label] for label in labels]
    fig, axes = plt.subplots(
        1, 2, figsize=(12.2, 4.8) if len(runs) > 5 else (10.5, 4.5)
    )
    axes[0].bar(labels, effective_per_million, color=colors)
    axes[0].set_ylabel("Generated effective groups / 1M tokens")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, final_accuracy, color=colors)
    axes[1].set_ylabel("Final full-set Pass@1 (%)")
    axes[1].set_ylim(70, 84)
    axes[1].grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=12)
    fig.tight_layout()
    fig.savefig(output_dir / "efficiency_comparison.png", dpi=180)
    plt.close(fig)


def build_comparison(summaries: dict[str, dict]) -> dict:
    target_comparison = {}
    target_sets = [
        set(summary["comparable_evaluation"]["tokens_to_target_accuracy"])
        for summary in summaries.values()
    ]
    vanilla_targets = summaries["vanilla"]["comparable_evaluation"][
        "tokens_to_target_accuracy"
    ]
    for target in sorted(set().union(*target_sets)):
        vanilla_result = vanilla_targets.get(target)
        target_comparison[target] = {}
        for strategy, summary in summaries.items():
            result = summary["comparable_evaluation"][
                "tokens_to_target_accuracy"
            ].get(target)
            target_comparison[target][strategy] = result
            target_comparison[target][f"{strategy}_to_vanilla_token_ratio"] = (
                result["cumulative_rollout_tokens"]
                / vanilla_result["cumulative_rollout_tokens"]
                if result and vanilla_result
                else None
            )

    vanilla_accuracy = summaries["vanilla"]["final_full_evaluation"]["accuracy"]
    return {
        "budget": {
            strategy: summary["rollout_cost"]["total_rollout_tokens"]
            for strategy, summary in summaries.items()
        },
        "optimization": {
            strategy: {
                "steps": summary["run"]["completed_grpo_steps"],
                "elapsed_seconds": summary["run"]["elapsed_train_seconds"],
            }
            for strategy, summary in summaries.items()
        },
        "sampling": {
            strategy: {
                "attempted_groups": summary["groups"]["attempted_groups"],
                "zero_variance_group_ratio": summary["groups"][
                    "zero_variance_group_ratio"
                ],
                "effective_groups": summary["groups"]["effective_groups"],
                "optimized_effective_groups": summary["groups"][
                    "optimized_effective_groups"
                ],
                "optimized_effective_groups_per_million_tokens": summary[
                    "groups"
                ]["optimized_effective_groups_per_million_rollout_tokens"],
                "relative_effective_groups_per_million_vs_vanilla": (
                    summary["groups"][
                        "optimized_effective_groups_per_million_rollout_tokens"
                    ]
                    / summaries["vanilla"]["groups"][
                        "optimized_effective_groups_per_million_rollout_tokens"
                    ]
                    - 1
                ),
                "discarded_groups": summary["groups"]["discarded_groups"],
                "discarded_rollout_tokens": summary["rollout_cost"][
                    "exact_discarded_rollout_tokens"
                ],
            }
            for strategy, summary in summaries.items()
        },
        "performance": {
            strategy: {
                "final_full_accuracy": summary["final_full_evaluation"][
                    "accuracy"
                ],
                "accuracy_points_vs_vanilla": (
                    summary["final_full_evaluation"]["accuracy"]
                    - vanilla_accuracy
                )
                * 100,
            }
            for strategy, summary in summaries.items()
        }
        | {
            "tokens_to_target_accuracy": target_comparison,
        },
    }


def main() -> None:
    args = parse_args()
    vanilla = load_run(args.vanilla_dir)
    dynamic = load_run(args.dynamic_dir)
    runs = {"Vanilla": vanilla, "Dynamic": dynamic}
    summaries = {"vanilla": vanilla[0], "dynamic": dynamic[0]}
    if args.difficulty_dir:
        difficulty = load_run(args.difficulty_dir)
        label = (
            "Difficulty beta=0.9"
            if args.difficulty_beta05_dir
            else "Difficulty-Aware"
        )
        runs[label] = difficulty
        summaries["difficulty"] = difficulty[0]
    if args.difficulty_beta05_dir:
        difficulty_beta05 = load_run(args.difficulty_beta05_dir)
        runs["Difficulty beta=0.5"] = difficulty_beta05
        summaries["difficulty_beta05"] = difficulty_beta05[0]
    if args.difficulty_coverage_dir:
        difficulty_coverage = load_run(args.difficulty_coverage_dir)
        runs["Difficulty coverage=0.3"] = difficulty_coverage
        summaries["difficulty_coverage"] = difficulty_coverage[0]
    if args.difficulty_frontloaded_dir:
        difficulty_frontloaded = load_run(args.difficulty_frontloaded_dir)
        runs["Front-loaded 512"] = difficulty_frontloaded
        summaries["difficulty_frontloaded"] = difficulty_frontloaded[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluations = []
    for label, (_, frame) in runs.items():
        frame = frame.copy()
        frame.insert(0, "strategy", label.lower())
        evaluations.append(frame)
    pd.concat(evaluations, ignore_index=True).to_csv(
        args.output_dir / "evaluation_comparison.csv", index=False
    )

    comparison = build_comparison(summaries)
    with (args.output_dir / "comparison.json").open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2, ensure_ascii=False)
    plot_accuracy(runs, args.output_dir)
    plot_efficiency(runs, args.output_dir)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
