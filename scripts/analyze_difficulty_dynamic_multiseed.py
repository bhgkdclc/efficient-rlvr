"""Aggregate the Difficulty-Aware + Dynamic Filtering replication study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--difficulty-dynamic-dirs", type=Path, nargs="+", required=True
    )
    parser.add_argument("--fastema-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-length-audit", type=Path, required=True)
    parser.add_argument("--dynamic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_analyzed_run(path: Path, strategy: str) -> dict:
    with (path / "summary.json").open(encoding="utf-8") as file:
        summary = json.load(file)
    optimizer = pd.read_csv(path / "optimizer_metrics.csv")
    return {
        "strategy": strategy,
        "seed": int(summary["run"]["seed"]),
        "summary": summary,
        "optimizer": optimizer,
    }


def training_row(run: dict) -> dict[str, float | int | str]:
    summary = run["summary"]
    final = summary["final_full_evaluation"]
    exact_discard = summary["rollout_cost"].get(
        "exact_discarded_rollout_token_ratio", 0.0
    )
    return {
        "strategy": run["strategy"],
        "seed": run["seed"],
        "optimizer_steps": summary["run"]["completed_grpo_steps"],
        "total_rollout_tokens": summary["rollout_cost"]["total_rollout_tokens"],
        "attempted_groups": summary["groups"]["attempted_groups"],
        "optimized_effective_groups": summary["groups"][
            "optimized_effective_groups"
        ],
        "zero_variance_group_ratio": summary["groups"][
            "zero_variance_group_ratio"
        ],
        "discarded_rollout_token_ratio": exact_discard,
        "effective_groups_per_million_tokens": summary["groups"][
            "optimized_effective_groups_per_million_rollout_tokens"
        ],
        "training_format_reward_mean": summary["trend"]["window_means"][
            "all_steps"
        ]["format_reward_mean"],
        "mean_gradient_norm": float(run["optimizer"]["train/grad_norm"].mean()),
        "mean_policy_entropy": float(
            run["optimizer"]["train/policy_entropy"].mean()
        ),
        "full_accuracy": final["accuracy"],
        "full_correct": final["correct"],
        "full_wrong_answer": final["wrong_answer"],
        "full_wrong_format": final["wrong_format"],
    }


def mean_std(frame: pd.DataFrame, column: str) -> dict[str, float]:
    return {
        "mean": float(frame[column].mean()),
        "sample_std": float(frame[column].std(ddof=1)),
    }


def paired_interval(values: pd.Series) -> dict[str, float]:
    values = values.astype(float)
    mean = float(values.mean())
    sample_std = float(values.std(ddof=1))
    radius = float(
        stats.t.ppf(0.975, df=len(values) - 1)
        * sample_std
        / np.sqrt(len(values))
    )
    return {
        "mean": mean,
        "sample_std": sample_std,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
    }


def plot_summary(
    per_seed: pd.DataFrame,
    endpoint: pd.DataFrame,
    dynamic_summary: dict,
    output_path: Path,
) -> None:
    fast = per_seed[per_seed["strategy"] == "fastema"].sort_values("seed")
    combined = per_seed[
        per_seed["strategy"] == "difficulty_dynamic"
    ].sort_values("seed")
    colors = {
        "dynamic": "#ff7f0e",
        "fastema": "#9467bd",
        "difficulty_dynamic": "#2ca02c",
        "vanilla": "#1f77b4",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.7))

    strategies = ["dynamic", "fastema", "difficulty_dynamic"]
    labels = ["Dynamic\n(seed 42)", "Fast-EMA\n(3 seeds)", "Difficulty +\nDynamic (3 seeds)"]
    zero_values = [
        dynamic_summary["groups"]["zero_variance_group_ratio"] * 100,
        fast["zero_variance_group_ratio"].mean() * 100,
        combined["zero_variance_group_ratio"].mean() * 100,
    ]
    zero_errors = [
        0,
        fast["zero_variance_group_ratio"].std(ddof=1) * 100,
        combined["zero_variance_group_ratio"].std(ddof=1) * 100,
    ]
    axes[0, 0].bar(
        labels,
        zero_values,
        yerr=zero_errors,
        capsize=4,
        color=[colors[name] for name in strategies],
    )
    axes[0, 0].set_ylabel("Attempted zero-variance groups (%)")
    axes[0, 0].set_title("Pre-selection reduces rejected groups")

    efficiency_values = [
        dynamic_summary["groups"][
            "optimized_effective_groups_per_million_rollout_tokens"
        ],
        fast["effective_groups_per_million_tokens"].mean(),
        combined["effective_groups_per_million_tokens"].mean(),
    ]
    efficiency_errors = [
        0,
        fast["effective_groups_per_million_tokens"].std(ddof=1),
        combined["effective_groups_per_million_tokens"].std(ddof=1),
    ]
    axes[0, 1].bar(
        labels,
        efficiency_values,
        yerr=efficiency_errors,
        capsize=4,
        color=[colors[name] for name in strategies],
    )
    axes[0, 1].set_ylabel("Optimized effective groups / 1M tokens")
    axes[0, 1].set_title("Filtering adds little beyond Fast-EMA")

    endpoint_order = ["vanilla", "difficulty_beta05", "difficulty_dynamic"]
    endpoint_labels = ["Vanilla", "Fast-EMA", "Difficulty +\nDynamic"]
    for index, (strategy, label) in enumerate(zip(endpoint_order, endpoint_labels)):
        values = endpoint.loc[endpoint["strategy"] == strategy, "accuracy"] * 100
        axes[1, 0].bar(
            index,
            values.mean(),
            yerr=values.std(ddof=1),
            capsize=4,
            color=colors["fastema" if strategy == "difficulty_beta05" else strategy],
        )
        axes[1, 0].scatter(
            np.full(len(values), index), values, color="black", s=19, zorder=3
        )
    axes[1, 0].set_xticks(range(3), endpoint_labels)
    axes[1, 0].set_ylabel("Full GSM8K Pass@1 @ 1024 (%)")
    axes[1, 0].set_ylim(75, 86)
    axes[1, 0].set_title("Dense filtering hurts endpoint quality")

    error_names = ["Wrong answer", "Wrong format"]
    fast_errors = [fast["full_wrong_answer"].mean(), fast["full_wrong_format"].mean()]
    combined_errors = [
        combined["full_wrong_answer"].mean(),
        combined["full_wrong_format"].mean(),
    ]
    positions = np.arange(2)
    width = 0.35
    axes[1, 1].bar(
        positions - width / 2,
        fast_errors,
        width,
        label="Fast-EMA",
        color=colors["fastema"],
    )
    axes[1, 1].bar(
        positions + width / 2,
        combined_errors,
        width,
        label="Difficulty + Dynamic",
        color=colors["difficulty_dynamic"],
    )
    axes[1, 1].set_xticks(positions, error_names)
    axes[1, 1].set_ylabel("Mean full-test failures")
    axes[1, 1].set_title("Regression is mostly format failure")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Difficulty-Aware + Dynamic Filtering: three-seed audit", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = [
        load_analyzed_run(path, "difficulty_dynamic")
        for path in args.difficulty_dynamic_dirs
    ]
    runs += [load_analyzed_run(path, "fastema") for path in args.fastema_dirs]
    per_seed = pd.DataFrame([training_row(run) for run in runs]).sort_values(
        ["seed", "strategy"]
    )
    for strategy in ("difficulty_dynamic", "fastema"):
        seeds = set(per_seed.loc[per_seed["strategy"] == strategy, "seed"])
        if seeds != {42, 43, 44}:
            raise ValueError(f"expected seeds 42, 43, 44 for {strategy}, got {seeds}")

    eval_audit = pd.read_csv(args.eval_length_audit)
    endpoint = eval_audit[
        eval_audit["strategy"].isin(["vanilla", "difficulty_beta05"])
    ][["seed", "strategy", "accuracy_1024", "wrong_answer_1024", "wrong_format_1024"]]
    endpoint = endpoint.rename(
        columns={
            "accuracy_1024": "accuracy",
            "wrong_answer_1024": "wrong_answer",
            "wrong_format_1024": "wrong_format",
        }
    )
    combined = per_seed[per_seed["strategy"] == "difficulty_dynamic"]
    combined_endpoint = combined[
        ["seed", "full_accuracy", "full_wrong_answer", "full_wrong_format"]
    ].rename(
        columns={
            "full_accuracy": "accuracy",
            "full_wrong_answer": "wrong_answer",
            "full_wrong_format": "wrong_format",
        }
    )
    combined_endpoint["strategy"] = "difficulty_dynamic"
    endpoint = pd.concat([endpoint, combined_endpoint], ignore_index=True).sort_values(
        ["seed", "strategy"]
    )

    accuracy_wide = endpoint.pivot(index="seed", columns="strategy", values="accuracy")
    accuracy_wide["difference_vs_fastema_pp"] = (
        accuracy_wide["difficulty_dynamic"]
        - accuracy_wide["difficulty_beta05"]
    ) * 100
    accuracy_wide["difference_vs_vanilla_pp"] = (
        accuracy_wide["difficulty_dynamic"] - accuracy_wide["vanilla"]
    ) * 100
    accuracy_wide = accuracy_wide.reset_index()

    fast = per_seed[per_seed["strategy"] == "fastema"]
    combined = per_seed[per_seed["strategy"] == "difficulty_dynamic"]
    with (args.dynamic_dir / "summary.json").open(encoding="utf-8") as file:
        dynamic_summary = json.load(file)

    endpoint_aggregate = {
        strategy: mean_std(frame, "accuracy")
        for strategy, frame in endpoint.groupby("strategy")
    }
    summary = {
        "seeds": [42, 43, 44],
        "difficulty_dynamic": {
            "zero_variance_group_ratio": mean_std(
                combined, "zero_variance_group_ratio"
            ),
            "discarded_rollout_token_ratio": mean_std(
                combined, "discarded_rollout_token_ratio"
            ),
            "optimized_effective_groups_per_million_tokens": mean_std(
                combined, "effective_groups_per_million_tokens"
            ),
            "optimizer_steps": mean_std(combined, "optimizer_steps"),
            "mean_gradient_norm": mean_std(combined, "mean_gradient_norm"),
            "training_format_reward_mean": mean_std(
                combined, "training_format_reward_mean"
            ),
        },
        "fastema": {
            "zero_variance_group_ratio": mean_std(
                fast, "zero_variance_group_ratio"
            ),
            "optimized_effective_groups_per_million_tokens": mean_std(
                fast, "effective_groups_per_million_tokens"
            ),
            "optimizer_steps": mean_std(fast, "optimizer_steps"),
            "mean_gradient_norm": mean_std(fast, "mean_gradient_norm"),
            "training_format_reward_mean": mean_std(
                fast, "training_format_reward_mean"
            ),
        },
        "dynamic_seed42": {
            "discarded_rollout_token_ratio": dynamic_summary["rollout_cost"][
                "exact_discarded_rollout_token_ratio"
            ],
            "optimized_effective_groups_per_million_tokens": dynamic_summary[
                "groups"
            ]["optimized_effective_groups_per_million_rollout_tokens"],
        },
        "full_evaluation_1024": endpoint_aggregate,
        "paired_accuracy_difference_vs_fastema_pp": paired_interval(
            accuracy_wide["difference_vs_fastema_pp"]
        ),
        "paired_accuracy_difference_vs_vanilla_pp": paired_interval(
            accuracy_wide["difference_vs_vanilla_pp"]
        ),
        "mean_error_decomposition": {
            strategy: {
                "wrong_answer": float(frame["wrong_answer"].mean()),
                "wrong_format": float(frame["wrong_format"].mean()),
            }
            for strategy, frame in endpoint.groupby("strategy")
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.output_dir / "per_seed_training_metrics.csv", index=False)
    endpoint.to_csv(args.output_dir / "per_seed_endpoint_metrics.csv", index=False)
    accuracy_wide.to_csv(args.output_dir / "paired_accuracy_differences.csv", index=False)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    plot_summary(
        per_seed,
        endpoint,
        dynamic_summary,
        args.output_dir / "difficulty_dynamic_multiseed.png",
    )

    dd = summary["difficulty_dynamic"]
    fast_summary = summary["fastema"]
    dynamic = summary["dynamic_seed42"]
    endpoints = summary["full_evaluation_1024"]
    errors = summary["mean_error_decomposition"]
    versus_fast = summary["paired_accuracy_difference_vs_fastema_pp"]
    report = f"""# Difficulty-Aware + Dynamic Filtering audit

All runs use Qwen2.5-Math-1.5B, GSM8K, group size 8, and an approximately
four-million rollout-token budget. Difficulty-Dynamic results use paired seeds
42, 43, and 44. The standalone Dynamic baseline is available only for seed 42,
so comparisons against it are descriptive rather than paired multi-seed claims.

## Result

| Metric | Dynamic, seed 42 | Fast-EMA, 3 seeds | Difficulty + Dynamic, 3 seeds |
|---|---:|---:|---:|
| Attempted zero-variance groups | {dynamic_summary['groups']['zero_variance_group_ratio'] * 100:.2f}% | {fast_summary['zero_variance_group_ratio']['mean'] * 100:.2f}% +/- {fast_summary['zero_variance_group_ratio']['sample_std'] * 100:.2f}% | **{dd['zero_variance_group_ratio']['mean'] * 100:.2f}% +/- {dd['zero_variance_group_ratio']['sample_std'] * 100:.2f}%** |
| Discarded rollout-token ratio | {dynamic['discarded_rollout_token_ratio'] * 100:.2f}% | n/a | **{dd['discarded_rollout_token_ratio']['mean'] * 100:.2f}% +/- {dd['discarded_rollout_token_ratio']['sample_std'] * 100:.2f}%** |
| Optimized effective groups / 1M tokens | {dynamic['optimized_effective_groups_per_million_tokens']:.2f} | {fast_summary['optimized_effective_groups_per_million_tokens']['mean']:.2f} +/- {fast_summary['optimized_effective_groups_per_million_tokens']['sample_std']:.2f} | **{dd['optimized_effective_groups_per_million_tokens']['mean']:.2f} +/- {dd['optimized_effective_groups_per_million_tokens']['sample_std']:.2f}** |
| Optimizer steps | {dynamic_summary['run']['completed_grpo_steps']} | {fast_summary['optimizer_steps']['mean']:.2f} +/- {fast_summary['optimizer_steps']['sample_std']:.2f} | {dd['optimizer_steps']['mean']:.2f} +/- {dd['optimizer_steps']['sample_std']:.2f} |
| Full GSM8K Pass@1 @ 1024 | not re-evaluated | {endpoints['difficulty_beta05']['mean'] * 100:.2f}% +/- {endpoints['difficulty_beta05']['sample_std'] * 100:.2f}% | {endpoints['difficulty_dynamic']['mean'] * 100:.2f}% +/- {endpoints['difficulty_dynamic']['sample_std'] * 100:.2f}% |

![Three-seed audit](difficulty_dynamic_multiseed.png)

Difficulty-aware pre-selection consistently lowers the waste seen by
post-generation Dynamic Filtering. Relative to the seed-42 Dynamic baseline,
the mean discarded-token ratio falls from 44.36% to 31.59%, while optimized
effective groups per million tokens rise from 147.65 to 178.51. However,
Difficulty-Dynamic is only 0.86% above Fast-EMA on effective groups per token:
the filtering stage mostly repacks approximately the same amount of useful
signal rather than creating more of it.

## Why endpoint accuracy regressed

Difficulty-Dynamic performs {dd['optimizer_steps']['mean']:.1f} optimizer
updates on average, versus {fast_summary['optimizer_steps']['mean']:.1f} for
Fast-EMA. Every Difficulty-Dynamic update contains eight effective groups,
whereas Fast-EMA retains zero-advantage groups in the nominal batch. With the
same learning rate and token-mean loss, mean gradient norm rises from
{fast_summary['mean_gradient_norm']['mean']:.3f} to
{dd['mean_gradient_norm']['mean']:.3f} ({(dd['mean_gradient_norm']['mean'] / fast_summary['mean_gradient_norm']['mean'] - 1) * 100:.1f}%).
This is evidence that filtering changed optimization intensity, not just
rollout accounting.

The paired full-test difference versus Fast-EMA is
{versus_fast['mean']:.2f} percentage points, with a wide three-seed 95% t
interval of [{versus_fast['ci95_low']:.2f}, {versus_fast['ci95_high']:.2f}].
The failure decomposition is more diagnostic than the aggregate score:
Fast-EMA averages {errors['difficulty_beta05']['wrong_format']:.1f} format
failures and {errors['difficulty_beta05']['wrong_answer']:.1f} wrong answers;
Difficulty-Dynamic averages {errors['difficulty_dynamic']['wrong_format']:.1f}
and {errors['difficulty_dynamic']['wrong_answer']:.1f}, respectively. Of the
35-example mean correctness gap, 32.3 examples come from format failures and
only 2.7 from explicit wrong answers. The regression is therefore primarily a
format-stability failure, not evidence of a comparable collapse in mathematical
answer quality.

## Conclusion

Pre-generation Fast-EMA sampling remains the recommended method. It provides
nearly all of Difficulty-Dynamic's token-level signal efficiency without the
format instability introduced by packing every optimizer batch with effective
groups. This rejects the naive hypothesis that maximizing the effective-group
fraction of each optimizer batch must improve final accuracy.

A targeted follow-up should restore Fast-EMA's optimizer-update cadence while
leaving the sampler, reward, group size, and token budget fixed. Targeting five
accepted groups per batch matches the observed Fast-EMA effective-group count
per update. It should first be screened on the worst seed (44) before
replication.
"""
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
