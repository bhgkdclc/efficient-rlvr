"""Summarize and plot one GRPO experiment from its metrics.jsonl file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROLLOUT_KEYS = {
    "zero_variance_group_ratio": "rollout/zero_variance_group_ratio",
    "effective_group_ratio": "rollout/effective_group_ratio",
    "all_correct_group_ratio": "rollout/all_correct_group_ratio",
    "all_wrong_group_ratio": "rollout/all_wrong_group_ratio",
    "mixed_group_ratio": "rollout/mixed_group_ratio",
    "format_reward_mean": "rollout/format_reward_mean",
    "answer_reward_mean": "rollout/answer_reward_mean",
    "reward_mean": "rollout/reward_mean",
    "reward_std": "rollout/reward_std",
    "group_reward_variance_mean": "rollout/group_reward_variance_mean",
    "advantage_mean": "rollout/advantage_mean",
    "advantage_std": "rollout/advantage_std",
    "generated_response_tokens": "rollout/generated_response_tokens",
    "total_rollout_tokens": "rollout/total_rollout_tokens",
    "average_response_length": "rollout/average_response_length",
    "cumulative_generated_response_tokens": (
        "rollout/cumulative_generated_response_tokens"
    ),
    "cumulative_rollout_tokens": "rollout/cumulative_rollout_tokens",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path, help="Path to metrics.jsonl")
    parser.add_argument("--config", type=Path, help="Optional config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rolling-window", type=int, default=10)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def dataframe_for_event(records: list[dict], event: str) -> pd.DataFrame:
    rows = [record for record in records if record.get("event") == event]
    if not rows:
        raise ValueError(f"no {event!r} records in metrics file")
    return pd.DataFrame(rows)


def mean_dict(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    return {column: float(frame[column].mean()) for column in columns}


def add_rollout_token_axis(
    evaluations: pd.DataFrame, rollouts: pd.DataFrame
) -> pd.DataFrame:
    evaluations = evaluations.copy()
    steps = rollouts["grpo_step"].astype(int).to_numpy()
    tokens = rollouts["cumulative_rollout_tokens"].astype(int).to_numpy()

    def tokens_at_step(model_step: int) -> int:
        if model_step <= 0:
            return 0
        eligible = np.flatnonzero(steps <= model_step)
        return int(tokens[eligible[-1]]) if len(eligible) else 0

    evaluations["cumulative_rollout_tokens"] = [
        tokens_at_step(int(step)) for step in evaluations["model_step"]
    ]
    return evaluations


def save_accuracy_plots(
    evaluations: pd.DataFrame, output_dir: Path, comparable_count: int
) -> None:
    comparable = evaluations[evaluations["eval/count"] == comparable_count]
    other = evaluations[evaluations["eval/count"] != comparable_count]

    for x_column, filename, xlabel, scale in (
        ("model_step", "accuracy_vs_step.png", "GRPO step", 1.0),
        (
            "cumulative_rollout_tokens",
            "accuracy_vs_rollout_tokens.png",
            "Cumulative rollout tokens (millions)",
            1e6,
        ),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(
            comparable[x_column] / scale,
            comparable["eval/accuracy"] * 100,
            marker="o",
            linewidth=2,
            markersize=4,
            label=f"Fixed evaluation subset (n={comparable_count})",
        )
        if not other.empty:
            ax.scatter(
                other[x_column] / scale,
                other["eval/accuracy"] * 100,
                marker="X",
                s=80,
                color="#d95f02",
                label="Full final evaluation",
                zorder=3,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Pass@1 / accuracy (%)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def save_group_plot(
    rollouts: pd.DataFrame, output_dir: Path, rolling_window: int
) -> None:
    step = rollouts["grpo_step"]
    smooth = rollouts.select_dtypes(include=[np.number]).rolling(
        rolling_window, min_periods=1
    ).mean()
    all_correct = smooth["all_correct_group_ratio"] * 100
    all_wrong = smooth["all_wrong_group_ratio"] * 100
    mixed = smooth["mixed_group_ratio"] * 100

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.stackplot(
        step,
        all_correct,
        all_wrong,
        mixed,
        labels=("All correct", "All wrong", "Mixed"),
        colors=("#4daf4a", "#e41a1c", "#377eb8"),
        alpha=0.72,
    )
    ax.plot(
        step,
        smooth["zero_variance_group_ratio"] * 100,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Zero variance (weighted reward)",
    )
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("Group ratio (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Group composition ({rolling_window}-step trailing mean)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(output_dir / "group_composition_vs_step.png", dpi=180)
    plt.close(fig)


def save_dynamics_plot(
    rollouts: pd.DataFrame,
    optimizer: pd.DataFrame,
    output_dir: Path,
    rolling_window: int,
) -> None:
    rollout_smooth = rollouts.select_dtypes(include=[np.number]).rolling(
        rolling_window, min_periods=1
    ).mean()
    optimizer_smooth = optimizer.select_dtypes(include=[np.number]).rolling(
        rolling_window, min_periods=1
    ).mean()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes[0, 0].plot(optimizer["grpo_step"], optimizer_smooth["train/policy_entropy"])
    axes[0, 0].set_ylabel("Policy entropy")
    axes[0, 1].plot(
        rollouts["grpo_step"], rollout_smooth["average_response_length"]
    )
    axes[0, 1].set_ylabel("Average response tokens")
    axes[1, 0].plot(rollouts["grpo_step"], rollout_smooth["reward_mean"])
    axes[1, 0].set_ylabel("Mean weighted reward")
    axes[1, 1].plot(
        rollouts["grpo_step"],
        rollout_smooth["group_reward_variance_mean"],
    )
    axes[1, 1].set_ylabel("Mean group reward variance")
    for ax in axes.flat:
        ax.set_xlabel("GRPO step")
        ax.grid(alpha=0.25)
    fig.suptitle(f"Training dynamics ({rolling_window}-step trailing mean)")
    fig.tight_layout()
    fig.savefig(output_dir / "training_dynamics.png", dpi=180)
    plt.close(fig)


def summarize(
    rollouts: pd.DataFrame,
    optimizer: pd.DataFrame,
    evaluations: pd.DataFrame,
    config: dict,
) -> dict:
    comparable_count = Counter(evaluations["eval/count"].astype(int)).most_common(1)[0][0]
    comparable = evaluations[evaluations["eval/count"] == comparable_count].sort_values(
        "model_step"
    )
    other = evaluations[evaluations["eval/count"] != comparable_count].sort_values(
        "model_step"
    )
    initial = comparable.iloc[0]
    last_comparable = comparable.iloc[-1]

    rollout_batch_size = int(config.get("rollout_batch_size", 0))
    group_size = int(config.get("group_size", 0))
    groups_per_step = rollout_batch_size // group_size
    total_groups = len(rollouts) * groups_per_step
    zero_groups = int(
        round((rollouts["zero_variance_group_ratio"] * groups_per_step).sum())
    )

    aggregate_columns = [
        "zero_variance_group_ratio",
        "effective_group_ratio",
        "all_correct_group_ratio",
        "all_wrong_group_ratio",
        "mixed_group_ratio",
        "format_reward_mean",
        "answer_reward_mean",
        "reward_mean",
        "reward_std",
        "group_reward_variance_mean",
        "advantage_std",
        "average_response_length",
    ]
    windows = {
        "first_20_steps": rollouts.head(20),
        "middle_20_steps": rollouts.iloc[
            max(0, len(rollouts) // 2 - 10) : len(rollouts) // 2 + 10
        ],
        "last_20_steps": rollouts.tail(20),
        "all_steps": rollouts,
    }
    window_summary = {
        name: mean_dict(frame, aggregate_columns) for name, frame in windows.items()
    }
    window_summary["first_20_steps"]["policy_entropy"] = float(
        optimizer.head(20)["train/policy_entropy"].mean()
    )
    window_summary["middle_20_steps"]["policy_entropy"] = float(
        optimizer.iloc[
            max(0, len(optimizer) // 2 - 10) : len(optimizer) // 2 + 10
        ]["train/policy_entropy"].mean()
    )
    window_summary["last_20_steps"]["policy_entropy"] = float(
        optimizer.tail(20)["train/policy_entropy"].mean()
    )
    window_summary["all_steps"]["policy_entropy"] = float(
        optimizer["train/policy_entropy"].mean()
    )

    targets = {}
    for target in (0.65, 0.70, 0.75, 0.79):
        reached = comparable[comparable["eval/accuracy"] >= target]
        if not reached.empty:
            row = reached.iloc[0]
            targets[f"{target:.2f}"] = {
                "model_step": int(row["model_step"]),
                "cumulative_rollout_tokens": int(row["cumulative_rollout_tokens"]),
                "accuracy": float(row["eval/accuracy"]),
            }

    accuracy_gain_points = (
        float(last_comparable["eval/accuracy"] - initial["eval/accuracy"]) * 100
    )
    total_rollout_tokens = int(rollouts.iloc[-1]["cumulative_rollout_tokens"])
    estimated_ineffective_tokens = int(
        round(
            (
                rollouts["total_rollout_tokens"]
                * rollouts["zero_variance_group_ratio"]
            ).sum()
        )
    )
    zero_trend_correlation = float(
        np.corrcoef(
            rollouts["grpo_step"], rollouts["zero_variance_group_ratio"]
        )[0, 1]
    )

    summary = {
        "run": {
            "git_commit": config.get("git_commit"),
            "seed": config.get("seed"),
            "model_name_or_path": config.get("model_name_or_path"),
            "dataset": config.get("train_data_path"),
            "sampling_strategy": config.get("sampling_strategy"),
            "group_size": group_size,
            "rollout_batch_size": rollout_batch_size,
            "completed_grpo_steps": int(len(rollouts)),
            "elapsed_train_seconds": float(optimizer.iloc[-1]["train/elapsed_time"]),
        },
        "rollout_cost": {
            "total_rollout_tokens": total_rollout_tokens,
            "generated_response_tokens": int(
                rollouts.iloc[-1]["cumulative_generated_response_tokens"]
            ),
            "estimated_zero_variance_rollout_tokens": estimated_ineffective_tokens,
            "estimated_zero_variance_token_ratio": (
                estimated_ineffective_tokens / total_rollout_tokens
            ),
        },
        "groups": {
            "total_groups": total_groups,
            "zero_variance_groups": zero_groups,
            "effective_groups": total_groups - zero_groups,
            "zero_variance_group_ratio": zero_groups / total_groups,
        },
        "comparable_evaluation": {
            "sample_count": comparable_count,
            "initial_step": int(initial["model_step"]),
            "initial_accuracy": float(initial["eval/accuracy"]),
            "last_step": int(last_comparable["model_step"]),
            "last_accuracy": float(last_comparable["eval/accuracy"]),
            "accuracy_gain_points": accuracy_gain_points,
            "rollout_tokens_per_accuracy_point": (
                int(last_comparable["cumulative_rollout_tokens"])
                / accuracy_gain_points
                if accuracy_gain_points > 0
                else None
            ),
            "tokens_to_target_accuracy": targets,
        },
        "final_full_evaluation": (
            {
                "model_step": int(other.iloc[-1]["model_step"]),
                "sample_count": int(other.iloc[-1]["eval/count"]),
                "correct": int(other.iloc[-1]["eval/correct"]),
                "accuracy": float(other.iloc[-1]["eval/accuracy"]),
                "wrong_answer": int(other.iloc[-1]["eval/wrong_answer"]),
                "wrong_format": int(other.iloc[-1]["eval/wrong_format"]),
            }
            if not other.empty
            else None
        ),
        "trend": {
            "step_vs_zero_variance_pearson_r": zero_trend_correlation,
            "window_means": window_summary,
        },
        "notes": [
            "Intermediate and initial accuracy use the fixed evaluation subset; the final full evaluation is not directly comparable.",
            "Estimated zero-variance token cost weights each batch token count by its zero-variance group ratio; per-group token lengths were not logged.",
        ],
    }
    return summary


def main() -> None:
    args = parse_args()
    if args.rolling_window <= 0:
        raise ValueError("--rolling-window must be positive")
    records = load_records(args.metrics)
    rollouts = dataframe_for_event(records, "rollout").sort_values("grpo_step")
    optimizer = dataframe_for_event(records, "optimizer_step").sort_values("grpo_step")
    evaluations = dataframe_for_event(records, "evaluation").sort_values("model_step")

    rollouts = rollouts.rename(columns={value: key for key, value in ROLLOUT_KEYS.items()})
    evaluations = add_rollout_token_axis(evaluations, rollouts)
    config = {}
    if args.config:
        with args.config.open(encoding="utf-8") as file:
            config = json.load(file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollouts.to_csv(args.output_dir / "rollout_metrics.csv", index=False)
    optimizer.to_csv(args.output_dir / "optimizer_metrics.csv", index=False)
    evaluations.to_csv(args.output_dir / "evaluation_metrics.csv", index=False)

    summary = summarize(rollouts, optimizer, evaluations, config)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    comparable_count = int(summary["comparable_evaluation"]["sample_count"])
    save_accuracy_plots(evaluations, args.output_dir, comparable_count)
    save_group_plot(rollouts, args.output_dir, args.rolling_window)
    save_dynamics_plot(rollouts, optimizer, args.output_dir, args.rolling_window)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
