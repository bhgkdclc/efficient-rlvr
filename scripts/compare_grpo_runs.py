"""Compare Vanilla and Dynamic GRPO runs under a rollout-token budget."""

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
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    colors = {"Vanilla": "#1f77b4", "Dynamic": "#d95f02"}
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
    ax.legend(frameon=False, fontsize=8)
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
    colors = ["#1f77b4", "#d95f02"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    axes[0].bar(labels, effective_per_million, color=colors)
    axes[0].set_ylabel("Generated effective groups / 1M tokens")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, final_accuracy, color=colors)
    axes[1].set_ylabel("Final full-set Pass@1 (%)")
    axes[1].set_ylim(70, 84)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "efficiency_comparison.png", dpi=180)
    plt.close(fig)


def build_comparison(vanilla: dict, dynamic: dict) -> dict:
    target_comparison = {}
    vanilla_targets = vanilla["comparable_evaluation"]["tokens_to_target_accuracy"]
    dynamic_targets = dynamic["comparable_evaluation"]["tokens_to_target_accuracy"]
    for target in sorted(set(vanilla_targets) | set(dynamic_targets)):
        vanilla_result = vanilla_targets.get(target)
        dynamic_result = dynamic_targets.get(target)
        target_comparison[target] = {
            "vanilla": vanilla_result,
            "dynamic": dynamic_result,
            "dynamic_to_vanilla_token_ratio": (
                dynamic_result["cumulative_rollout_tokens"]
                / vanilla_result["cumulative_rollout_tokens"]
                if vanilla_result and dynamic_result
                else None
            ),
        }

    vanilla_accuracy = vanilla["final_full_evaluation"]["accuracy"]
    dynamic_accuracy = dynamic["final_full_evaluation"]["accuracy"]
    vanilla_effective = vanilla["groups"][
        "optimized_effective_groups_per_million_rollout_tokens"
    ]
    dynamic_effective = dynamic["groups"][
        "optimized_effective_groups_per_million_rollout_tokens"
    ]
    return {
        "budget": {
            "vanilla_rollout_tokens": vanilla["rollout_cost"][
                "total_rollout_tokens"
            ],
            "dynamic_rollout_tokens": dynamic["rollout_cost"][
                "total_rollout_tokens"
            ],
        },
        "optimization": {
            "vanilla_steps": vanilla["run"]["completed_grpo_steps"],
            "dynamic_steps": dynamic["run"]["completed_grpo_steps"],
            "vanilla_elapsed_seconds": vanilla["run"]["elapsed_train_seconds"],
            "dynamic_elapsed_seconds": dynamic["run"]["elapsed_train_seconds"],
        },
        "sampling": {
            "vanilla_attempted_groups": vanilla["groups"]["attempted_groups"],
            "vanilla_effective_groups": vanilla["groups"]["effective_groups"],
            "vanilla_optimized_effective_groups": vanilla["groups"][
                "optimized_effective_groups"
            ],
            "dynamic_attempted_groups": dynamic["groups"]["attempted_groups"],
            "dynamic_effective_groups": dynamic["groups"]["effective_groups"],
            "dynamic_optimized_groups": dynamic["groups"]["optimized_groups"],
            "dynamic_optimized_effective_groups": dynamic["groups"][
                "optimized_effective_groups"
            ],
            "dynamic_discarded_groups": dynamic["groups"]["discarded_groups"],
            "dynamic_discarded_rollout_tokens": dynamic["rollout_cost"][
                "exact_discarded_rollout_tokens"
            ],
            "dynamic_discarded_rollout_token_ratio": dynamic["rollout_cost"][
                "exact_discarded_rollout_token_ratio"
            ],
            "vanilla_optimized_effective_groups_per_million_tokens": (
                vanilla_effective
            ),
            "dynamic_optimized_effective_groups_per_million_tokens": (
                dynamic_effective
            ),
            "optimized_effective_groups_per_million_relative_change": (
                dynamic_effective / vanilla_effective - 1
            ),
        },
        "performance": {
            "vanilla_final_full_accuracy": vanilla_accuracy,
            "dynamic_final_full_accuracy": dynamic_accuracy,
            "dynamic_minus_vanilla_accuracy_points": (
                dynamic_accuracy - vanilla_accuracy
            )
            * 100,
            "tokens_to_target_accuracy": target_comparison,
        },
    }


def main() -> None:
    args = parse_args()
    vanilla = load_run(args.vanilla_dir)
    dynamic = load_run(args.dynamic_dir)
    runs = {"Vanilla": vanilla, "Dynamic": dynamic}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluations = []
    for label, (_, frame) in runs.items():
        frame = frame.copy()
        frame.insert(0, "strategy", label.lower())
        evaluations.append(frame)
    pd.concat(evaluations, ignore_index=True).to_csv(
        args.output_dir / "evaluation_comparison.csv", index=False
    )

    comparison = build_comparison(vanilla[0], dynamic[0])
    with (args.output_dir / "comparison.json").open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2, ensure_ascii=False)
    plot_accuracy(runs, args.output_dir)
    plot_efficiency(runs, args.output_dir)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
