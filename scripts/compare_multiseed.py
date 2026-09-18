"""Aggregate paired Vanilla and Difficulty-Aware GRPO runs across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGETS = ("0.65", "0.70", "0.75", "0.79")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--difficulty-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_run(path: Path, strategy: str) -> dict:
    with (path / "summary.json").open(encoding="utf-8") as file:
        summary = json.load(file)
    evaluations = pd.read_csv(path / "evaluation_metrics.csv")
    return {
        "path": path,
        "strategy": strategy,
        "seed": int(summary["run"]["seed"]),
        "summary": summary,
        "evaluations": evaluations,
    }


def fixed_evaluations(run: dict) -> pd.DataFrame:
    frame = run["evaluations"]
    count = int(run["summary"]["comparable_evaluation"]["sample_count"])
    return frame[frame["eval/count"] == count].sort_values(
        "cumulative_rollout_tokens"
    )


def per_seed_row(run: dict) -> dict[str, float | int | str]:
    summary = run["summary"]
    difficulty = summary.get("difficulty_sampler")
    return {
        "strategy": run["strategy"],
        "seed": run["seed"],
        "total_rollout_tokens": summary["rollout_cost"]["total_rollout_tokens"],
        "attempted_groups": summary["groups"]["attempted_groups"],
        "zero_variance_group_ratio": summary["groups"][
            "zero_variance_group_ratio"
        ],
        "effective_groups": summary["groups"]["effective_groups"],
        "effective_groups_per_million_tokens": summary["groups"][
            "optimized_effective_groups_per_million_rollout_tokens"
        ],
        "last_20_zero_variance_group_ratio": summary["trend"]["window_means"][
            "last_20_steps"
        ]["zero_variance_group_ratio"],
        "average_response_length": summary["trend"]["window_means"][
            "all_steps"
        ]["average_response_length"],
        "fixed_last_accuracy": summary["comparable_evaluation"]["last_accuracy"],
        "full_accuracy": summary["final_full_evaluation"]["accuracy"],
        "full_correct": summary["final_full_evaluation"]["correct"],
        "full_wrong_answer": summary["final_full_evaluation"]["wrong_answer"],
        "full_wrong_format": summary["final_full_evaluation"]["wrong_format"],
        "calibration_gap": (
            difficulty["post_warmup_accuracy_calibration_gap"]
            if difficulty
            else np.nan
        ),
        "observed_prompt_coverage": (
            difficulty["dataset_prompt_coverage"] if difficulty else np.nan
        ),
    }


def aggregate_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in per_seed.columns
        if column not in {"strategy", "seed"}
    ]
    aggregate = per_seed.groupby("strategy")[numeric].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    return aggregate.reset_index()


def paired_differences(per_seed: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in per_seed.columns
        if column not in {"strategy", "seed", "calibration_gap", "observed_prompt_coverage"}
    ]
    rows = []
    for seed in sorted(per_seed["seed"].unique()):
        paired = per_seed[per_seed["seed"] == seed].set_index("strategy")
        if set(paired.index) != {"vanilla", "difficulty_beta05"}:
            raise ValueError(f"seed {seed} does not have one run per strategy")
        row: dict[str, float | int] = {"seed": int(seed)}
        for metric in numeric:
            row[f"{metric}_difference"] = float(
                paired.loc["difficulty_beta05", metric]
                - paired.loc["vanilla", metric]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def tokens_to_target_rows(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        targets = run["summary"]["comparable_evaluation"][
            "tokens_to_target_accuracy"
        ]
        for target in TARGETS:
            result = targets.get(target)
            rows.append(
                {
                    "strategy": run["strategy"],
                    "seed": run["seed"],
                    "target_accuracy": float(target),
                    "reached": result is not None,
                    "rollout_tokens": (
                        result["cumulative_rollout_tokens"] if result else np.nan
                    ),
                    "model_step": result["model_step"] if result else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_targets(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, target), frame in targets.groupby(
        ["strategy", "target_accuracy"]
    ):
        reached = frame[frame["reached"]]
        rows.append(
            {
                "strategy": strategy,
                "target_accuracy": target,
                "reached_seeds": int(len(reached)),
                "total_seeds": int(len(frame)),
                "reach_rate": float(frame["reached"].mean()),
                "rollout_tokens_mean_among_reached": (
                    float(reached["rollout_tokens"].mean())
                    if len(reached)
                    else np.nan
                ),
                "rollout_tokens_std_among_reached": (
                    float(reached["rollout_tokens"].std(ddof=1))
                    if len(reached) > 1
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_accuracy_curves(runs: list[dict], output_dir: Path) -> None:
    maximum = min(
        float(fixed_evaluations(run)["cumulative_rollout_tokens"].max())
        for run in runs
    )
    grid = np.linspace(0, maximum, 160)
    colors = {"vanilla": "#1f77b4", "difficulty_beta05": "#9467bd"}
    labels = {"vanilla": "Vanilla", "difficulty_beta05": "Difficulty beta=0.5"}

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for strategy in ("vanilla", "difficulty_beta05"):
        selected = [run for run in runs if run["strategy"] == strategy]
        interpolated = []
        for run in selected:
            frame = fixed_evaluations(run)
            interpolated.append(
                np.interp(
                    grid,
                    frame["cumulative_rollout_tokens"],
                    frame["eval/accuracy"],
                )
            )
        values = np.vstack(interpolated) * 100
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        ax.plot(grid / 1e6, mean, color=colors[strategy], label=labels[strategy])
        ax.fill_between(
            grid / 1e6,
            mean - std,
            mean + std,
            color=colors[strategy],
            alpha=0.18,
            linewidth=0,
        )

        full_accuracy = np.array(
            [run["summary"]["final_full_evaluation"]["accuracy"] for run in selected]
        )
        total_tokens = np.array(
            [run["summary"]["rollout_cost"]["total_rollout_tokens"] for run in selected]
        )
        ax.errorbar(
            total_tokens.mean() / 1e6,
            full_accuracy.mean() * 100,
            yerr=full_accuracy.std(ddof=1) * 100,
            marker="X",
            markersize=9,
            capsize=4,
            color=colors[strategy],
            linestyle="none",
            label=f"{labels[strategy]} full-test",
        )

    ax.set_xlabel("Cumulative rollout tokens (millions)")
    ax.set_ylabel("Pass@1 / accuracy (%)")
    ax.set_title("Three-seed learning curves (mean +/- sample std)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "multiseed_accuracy_vs_tokens.png", dpi=180)
    plt.close(fig)


def plot_summary_bars(per_seed: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("zero_variance_group_ratio", "Zero-variance groups (%)", 100),
        (
            "effective_groups_per_million_tokens",
            "Effective groups / 1M tokens",
            1,
        ),
        ("full_accuracy", "Full GSM8K Pass@1 (%)", 100),
    ]
    labels = ["Vanilla", "Difficulty beta=0.5"]
    strategies = ["vanilla", "difficulty_beta05"]
    colors = ["#1f77b4", "#9467bd"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    for axis, (metric, ylabel, scale) in zip(axes, metrics):
        means = [
            per_seed.loc[per_seed["strategy"] == strategy, metric].mean() * scale
            for strategy in strategies
        ]
        stds = [
            per_seed.loc[per_seed["strategy"] == strategy, metric].std(ddof=1)
            * scale
            for strategy in strategies
        ]
        axis.bar(labels, means, yerr=stds, capsize=4, color=colors)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelrotation=10)
    axes[2].set_ylim(74, 84)
    fig.tight_layout()
    fig.savefig(output_dir / "multiseed_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = [load_run(path, "vanilla") for path in args.vanilla_dirs]
    runs += [
        load_run(path, "difficulty_beta05") for path in args.difficulty_dirs
    ]
    vanilla_seeds = {run["seed"] for run in runs if run["strategy"] == "vanilla"}
    difficulty_seeds = {
        run["seed"] for run in runs if run["strategy"] == "difficulty_beta05"
    }
    if vanilla_seeds != difficulty_seeds:
        raise ValueError(
            f"paired seeds required, got vanilla={vanilla_seeds}, "
            f"difficulty={difficulty_seeds}"
        )
    if len(vanilla_seeds) < 2:
        raise ValueError("at least two paired seeds are required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame([per_seed_row(run) for run in runs]).sort_values(
        ["seed", "strategy"]
    )
    aggregate = aggregate_metrics(per_seed)
    paired = paired_differences(per_seed)
    targets = tokens_to_target_rows(runs)
    target_summary = summarize_targets(targets)

    per_seed.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "paired_differences.csv", index=False)
    targets.to_csv(args.output_dir / "tokens_to_target.csv", index=False)
    target_summary.to_csv(
        args.output_dir / "tokens_to_target_summary.csv", index=False
    )

    summary = {
        "seeds": sorted(vanilla_seeds),
        "aggregate": json.loads(aggregate.to_json(orient="records")),
        "paired_difference_mean": {
            column: float(paired[column].mean())
            for column in paired.columns
            if column != "seed"
        },
        "target_summary": json.loads(target_summary.to_json(orient="records")),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, allow_nan=False)

    plot_accuracy_curves(runs, args.output_dir)
    plot_summary_bars(per_seed, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
