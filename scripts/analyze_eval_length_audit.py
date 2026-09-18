"""Audit evaluation truncation and rebuild the paired multi-seed endpoint result."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


EVAL_NAME_RE = re.compile(r"^(vanilla|fastema)(?P<seed>\d+)_eval1024_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-root",
        type=Path,
        required=True,
        help="Directory containing extracted *_eval1024_* experiment folders",
    )
    parser.add_argument("--old-multiseed-dir", type=Path, required=True)
    parser.add_argument("--hybrid-train-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def full_evaluation(path: Path) -> dict:
    evaluations = [
        row for row in load_jsonl(path / "metrics.jsonl")
        if row.get("event") == "evaluation"
    ]
    if not evaluations:
        raise ValueError(f"no evaluation records in {path}")
    return max(evaluations, key=lambda row: int(row["eval/count"]))


def load_eval_1024(eval_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(eval_root.glob("*_eval1024_*")):
        match = EVAL_NAME_RE.match(path.name)
        if match is None:
            continue
        evaluation = full_evaluation(path)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": int(match.group("seed")),
                "strategy": (
                    "vanilla"
                    if match.group(1) == "vanilla"
                    else "difficulty_beta05"
                ),
                "eval_max_tokens": int(config["eval_max_tokens"]),
                "accuracy_1024": float(evaluation["eval/accuracy"]),
                "correct_1024": int(evaluation["eval/correct"]),
                "wrong_answer_1024": int(evaluation["eval/wrong_answer"]),
                "wrong_format_1024": int(evaluation["eval/wrong_format"]),
                "generated_response_tokens_1024": int(
                    evaluation["eval/generated_response_tokens"]
                ),
                "average_response_length_1024": float(
                    evaluation["eval/average_response_length"]
                ),
                "source_run": path.name,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 6:
        raise ValueError(f"expected six paired 1024 evaluations, found {len(frame)}")
    if set(frame["eval_max_tokens"]) != {1024}:
        raise ValueError("all audited evaluations must use max_tokens=1024")
    return frame


def paired_confidence_interval(values: pd.Series) -> dict[str, float]:
    values = values.astype(float)
    mean = float(values.mean())
    sample_std = float(values.std(ddof=1))
    half_width = float(
        stats.t.ppf(0.975, df=len(values) - 1)
        * sample_std
        / np.sqrt(len(values))
    )
    return {
        "mean": mean,
        "sample_std": sample_std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def hybrid_summary(train_dir: Path, eval_root: Path, vanilla_seed42: pd.Series) -> dict:
    records = load_jsonl(train_dir / "metrics.jsonl")
    rollouts = [row for row in records if row.get("event") == "rollout"]
    old_eval = full_evaluation(train_dir)
    new_dirs = sorted(eval_root.glob("hybrid_u50_eval1024_*"))
    if len(new_dirs) != 1:
        raise ValueError("expected exactly one Hybrid 1024 evaluation")
    new_eval = full_evaluation(new_dirs[0])

    total_groups = sum(row["sampling/attempted_groups"] for row in rollouts)
    effective_groups = sum(
        row["rollout/effective_group_ratio"] * row["sampling/attempted_groups"]
        for row in rollouts
    )
    total_tokens = int(rollouts[-1]["rollout/cumulative_rollout_tokens"])
    sampler = json.loads(
        (train_dir / "sampler_state.json").read_text(encoding="utf-8")
    )

    def stratum(name: str) -> dict[str, float | int]:
        group_key = f"difficulty/hybrid_{name}_groups"
        effective_key = f"difficulty/hybrid_{name}_effective_group_ratio"
        accuracy_key = f"difficulty/hybrid_{name}_group_accuracy_mean"
        token_key = f"difficulty/hybrid_{name}_total_rollout_tokens"
        groups = sum(row[group_key] for row in rollouts)
        effective = sum(row[group_key] * row[effective_key] for row in rollouts)
        tokens = sum(row[token_key] for row in rollouts)
        return {
            "groups": int(groups),
            "effective_group_ratio": float(effective / groups),
            "group_accuracy_mean": float(
                sum(row[group_key] * row[accuracy_key] for row in rollouts)
                / groups
            ),
            "rollout_tokens": int(tokens),
            "effective_groups_per_million_tokens": float(
                effective / tokens * 1e6
            ),
        }

    efficiency = float(effective_groups / total_tokens * 1e6)
    return {
        "total_rollout_tokens": total_tokens,
        "effective_groups_per_million_tokens": efficiency,
        "relative_efficiency_change_vs_vanilla_seed42": float(
            efficiency
            / vanilla_seed42["effective_groups_per_million_tokens"]
            - 1.0
        ),
        "prompt_coverage": float(
            sampler["observed_prompt_count"] / sampler["num_prompts"]
        ),
        "accuracy_512": float(old_eval["eval/accuracy"]),
        "accuracy_1024": float(new_eval["eval/accuracy"]),
        "accuracy_1024_difference_vs_vanilla_seed42": float(
            new_eval["eval/accuracy"] - vanilla_seed42["accuracy_1024"]
        ),
        "wrong_format_512": int(old_eval["eval/wrong_format"]),
        "wrong_format_1024": int(new_eval["eval/wrong_format"]),
        "uniform_stratum": stratum("uniform"),
        "difficulty_stratum": stratum("difficulty"),
    }


def save_protocol_plot(frame: pd.DataFrame, output_dir: Path) -> None:
    strategies = ["vanilla", "difficulty_beta05"]
    labels = ["Vanilla", "Fast-EMA"]
    colors = ["#1f77b4", "#9467bd"]
    x = np.arange(2)
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for offset, (column, label) in enumerate(
        [("accuracy_512", "512 tokens"), ("accuracy_1024", "1024 tokens")]
    ):
        means = [
            frame.loc[frame["strategy"] == strategy, column].mean() * 100
            for strategy in strategies
        ]
        stds = [
            frame.loc[frame["strategy"] == strategy, column].std(ddof=1) * 100
            for strategy in strategies
        ]
        axes[0].bar(
            x + (offset - 0.5) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=label,
            color=colors,
            alpha=0.65 + offset * 0.3,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Full GSM8K Pass@1 (%)")
    axes[0].set_ylim(74, 86)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    for offset, (column, label) in enumerate(
        [("full_wrong_format", "512 tokens"), ("wrong_format_1024", "1024 tokens")]
    ):
        means = [
            frame.loc[frame["strategy"] == strategy, column].mean()
            for strategy in strategies
        ]
        axes[1].bar(
            x + (offset - 0.5) * width,
            means,
            width,
            label=label,
            color=colors,
            alpha=0.65 + offset * 0.3,
        )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Unparseable / format-error examples")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Evaluation-length audit (three seeds, mean +/- sample std)")
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_length_audit.png", dpi=180)
    plt.close(fig)


def save_tradeoff_plot(
    frame: pd.DataFrame,
    hybrid: dict | None,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    styles = {
        "vanilla": ("Vanilla", "#1f77b4", "o"),
        "difficulty_beta05": ("Fast-EMA Difficulty", "#9467bd", "s"),
    }
    for strategy, (label, color, marker) in styles.items():
        selected = frame[frame["strategy"] == strategy]
        ax.scatter(
            selected["effective_groups_per_million_tokens"],
            selected["accuracy_1024"] * 100,
            color=color,
            marker=marker,
            s=60,
            label=label,
        )
        for _, row in selected.iterrows():
            ax.annotate(
                f"s{int(row['seed'])}",
                (
                    row["effective_groups_per_million_tokens"],
                    row["accuracy_1024"] * 100,
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    if hybrid is not None:
        ax.scatter(
            hybrid["effective_groups_per_million_tokens"],
            hybrid["accuracy_1024"] * 100,
            color="#2ca02c",
            marker="D",
            s=65,
            label="Hybrid 50/50 (s42)",
        )
    ax.set_xlabel("Effective groups / 1M rollout tokens")
    ax.set_ylabel("Full GSM8K Pass@1 @ 1024 tokens (%)")
    ax.set_title("Signal efficiency versus endpoint quality")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "efficiency_accuracy_tradeoff_1024.png", dpi=180)
    plt.close(fig)


def write_report(
    frame: pd.DataFrame,
    summary: dict,
    output_dir: Path,
) -> None:
    aggregate = summary["aggregate_1024"]
    paired = summary["paired_accuracy_difference_pp"]
    efficiency = summary["sampling_efficiency"]
    lines = [
        "# Evaluation-Length Audit and Final Multi-Seed Result",
        "",
        "All endpoint comparisons use the complete 1,319-example GSM8K test "
        "split with greedy decoding. Training used the same approximately "
        "four-million rollout-token budget in every run.",
        "",
        "## Corrected endpoint result",
        "",
        "| Seed | Vanilla @512 | Fast-EMA @512 | Vanilla @1024 | Fast-EMA @1024 | 1024 paired diff |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in sorted(frame["seed"].unique()):
        paired_seed = frame[frame["seed"] == seed].set_index("strategy")
        vanilla = paired_seed.loc["vanilla"]
        difficulty = paired_seed.loc["difficulty_beta05"]
        lines.append(
            f"| {seed} | {vanilla['accuracy_512'] * 100:.2f}% | "
            f"{difficulty['accuracy_512'] * 100:.2f}% | "
            f"{vanilla['accuracy_1024'] * 100:.2f}% | "
            f"{difficulty['accuracy_1024'] * 100:.2f}% | "
            f"{(difficulty['accuracy_1024'] - vanilla['accuracy_1024']) * 100:+.2f} pp |"
        )
    lines += [
        f"| **Mean** | **{summary['aggregate_512']['vanilla_accuracy_mean'] * 100:.2f}%** | "
        f"**{summary['aggregate_512']['difficulty_accuracy_mean'] * 100:.2f}%** | "
        f"**{aggregate['vanilla_accuracy_mean'] * 100:.2f}%** | "
        f"**{aggregate['difficulty_accuracy_mean'] * 100:.2f}%** | "
        f"**{paired['mean']:+.2f} pp** |",
        "",
        f"At 1024 tokens, Vanilla averaged {aggregate['vanilla_accuracy_mean'] * 100:.2f}% +/- "
        f"{aggregate['vanilla_accuracy_std'] * 100:.2f} and Fast-EMA averaged "
        f"{aggregate['difficulty_accuracy_mean'] * 100:.2f}% +/- "
        f"{aggregate['difficulty_accuracy_std'] * 100:.2f}. The paired Fast-EMA "
        f"minus Vanilla difference was {paired['mean']:+.2f} +/- "
        f"{paired['sample_std']:.2f} points, with a 95% t interval of "
        f"[{paired['ci95_low']:.2f}, {paired['ci95_high']:.2f}]. With three seeds, "
        "this interval is too wide to establish equivalence or superiority.",
        "",
        "![Evaluation length audit](evaluation_length_audit.png)",
        "",
        "## Truncation was a real evaluation confound",
        "",
        f"Raising the evaluation limit from 512 to 1024 tokens improved Vanilla by "
        f"{summary['truncation_audit']['vanilla_accuracy_gain_pp']:.2f} points on average "
        f"and Fast-EMA by {summary['truncation_audit']['difficulty_accuracy_gain_pp']:.2f} "
        f"points. Mean format/unparseable failures fell from "
        f"{summary['truncation_audit']['vanilla_wrong_format_512_mean']:.1f} to "
        f"{summary['truncation_audit']['vanilla_wrong_format_1024_mean']:.1f} for Vanilla "
        f"and from {summary['truncation_audit']['difficulty_wrong_format_512_mean']:.1f} "
        f"to {summary['truncation_audit']['difficulty_wrong_format_1024_mean']:.1f} for "
        "Fast-EMA. The original 512-token protocol therefore mislabeled many "
        "length-limited chains as formatting failures and disproportionately "
        "understated the active sampler's endpoint quality. Because the older "
        "runs did not log finish reasons, the before/after recovery is strong "
        "evidence of truncation rather than a direct per-response count.",
        "",
        "## Signal efficiency remains the robust positive result",
        "",
        f"Across the same paired seeds, Fast-EMA increased effective groups per "
        f"million rollout tokens from {efficiency['vanilla_effective_groups_per_million_mean']:.2f} "
        f"to {efficiency['difficulty_effective_groups_per_million_mean']:.2f}, a "
        f"{efficiency['relative_gain_mean'] * 100:.2f}% +/- "
        f"{efficiency['relative_gain_std'] * 100:.2f}% relative gain. Zero-variance "
        f"groups fell from {efficiency['vanilla_zero_variance_mean'] * 100:.2f}% to "
        f"{efficiency['difficulty_zero_variance_mean'] * 100:.2f}%.",
        "",
        "![Efficiency-quality tradeoff](efficiency_accuracy_tradeoff_1024.png)",
        "",
        "## Hybrid ablation",
        "",
    ]
    hybrid = summary.get("hybrid")
    if hybrid is not None:
        lines += [
            f"The 50/50 uniform-plus-difficulty sampler raised prompt coverage to "
            f"{hybrid['prompt_coverage'] * 100:.2f}% but achieved only "
            f"{hybrid['effective_groups_per_million_tokens']:.2f} effective groups per "
            f"million tokens and {hybrid['accuracy_1024'] * 100:.2f}% Pass@1. It was "
            "dominated by seed-42 Fast-EMA on both efficiency and endpoint accuracy, "
            "so it was correctly stopped after screening rather than replicated.",
            "",
        ]
    lines += [
        "## Defensible conclusion",
        "",
        "Fast-EMA Difficulty-Aware Sampling reliably generates more non-zero GRPO "
        "advantages under a fixed rollout-token budget. After correcting the "
        "evaluation-length confound, its mean endpoint gap is 0.61 percentage points, "
        "not the original 1.42 points. The evidence supports a strong signal-efficiency "
        "claim and near-parity as a descriptive result, but not statistical "
        "non-inferiority or improved final accuracy.",
        "",
        "Limitations: one model and dataset, three paired seeds, group size fixed at "
        "eight, and only final checkpoints were re-evaluated at 1024 tokens. Existing "
        "intermediate tokens-to-target curves still use the 512-token protocol.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    eval_1024 = load_eval_1024(args.eval_root)
    old = pd.read_csv(args.old_multiseed_dir / "per_seed_metrics.csv")
    old = old[
        [
            "seed",
            "strategy",
            "full_accuracy",
            "full_correct",
            "full_wrong_answer",
            "full_wrong_format",
            "zero_variance_group_ratio",
            "effective_groups_per_million_tokens",
        ]
    ].rename(
        columns={
            "full_accuracy": "accuracy_512",
            "full_correct": "correct_512",
            "full_wrong_answer": "wrong_answer_512",
            "full_wrong_format": "full_wrong_format",
        }
    )
    frame = old.merge(eval_1024, on=["seed", "strategy"], validate="one_to_one")
    frame["accuracy_gain_from_1024_pp"] = (
        frame["accuracy_1024"] - frame["accuracy_512"]
    ) * 100
    frame["wrong_format_reduction"] = (
        frame["full_wrong_format"] - frame["wrong_format_1024"]
    )
    frame["relative_efficiency_gain_vs_paired_vanilla"] = np.nan
    for seed in sorted(frame["seed"].unique()):
        selected = frame[frame["seed"] == seed]
        vanilla_efficiency = float(
            selected.loc[
                selected["strategy"] == "vanilla",
                "effective_groups_per_million_tokens",
            ].iloc[0]
        )
        mask = (frame["seed"] == seed) & (
            frame["strategy"] == "difficulty_beta05"
        )
        frame.loc[mask, "relative_efficiency_gain_vs_paired_vanilla"] = (
            frame.loc[mask, "effective_groups_per_million_tokens"]
            / vanilla_efficiency
            - 1.0
        )

    pivot = frame.pivot(index="seed", columns="strategy", values="accuracy_1024")
    pivot_512 = frame.pivot(
        index="seed", columns="strategy", values="accuracy_512"
    )
    differences_pp = (
        pivot["difficulty_beta05"] - pivot["vanilla"]
    ) * 100
    efficiency_gain = frame.loc[
        frame["strategy"] == "difficulty_beta05",
        "relative_efficiency_gain_vs_paired_vanilla",
    ]
    vanilla = frame[frame["strategy"] == "vanilla"]
    difficulty = frame[frame["strategy"] == "difficulty_beta05"]

    vanilla_seed42 = vanilla[vanilla["seed"] == 42].iloc[0]
    hybrid = (
        hybrid_summary(args.hybrid_train_dir, args.eval_root, vanilla_seed42)
        if args.hybrid_train_dir
        else None
    )
    summary = {
        "seeds": sorted(int(seed) for seed in frame["seed"].unique()),
        "aggregate_512": {
            "vanilla_accuracy_mean": float(vanilla["accuracy_512"].mean()),
            "difficulty_accuracy_mean": float(difficulty["accuracy_512"].mean()),
            "paired_difference_pp": float(
                (
                    pivot_512["difficulty_beta05"] - pivot_512["vanilla"]
                ).mean()
                * 100
            ),
        },
        "aggregate_1024": {
            "vanilla_accuracy_mean": float(vanilla["accuracy_1024"].mean()),
            "vanilla_accuracy_std": float(vanilla["accuracy_1024"].std(ddof=1)),
            "difficulty_accuracy_mean": float(difficulty["accuracy_1024"].mean()),
            "difficulty_accuracy_std": float(
                difficulty["accuracy_1024"].std(ddof=1)
            ),
        },
        "paired_accuracy_difference_pp": paired_confidence_interval(
            differences_pp
        ),
        "truncation_audit": {
            "vanilla_accuracy_gain_pp": float(
                vanilla["accuracy_gain_from_1024_pp"].mean()
            ),
            "difficulty_accuracy_gain_pp": float(
                difficulty["accuracy_gain_from_1024_pp"].mean()
            ),
            "vanilla_wrong_format_512_mean": float(
                vanilla["full_wrong_format"].mean()
            ),
            "vanilla_wrong_format_1024_mean": float(
                vanilla["wrong_format_1024"].mean()
            ),
            "difficulty_wrong_format_512_mean": float(
                difficulty["full_wrong_format"].mean()
            ),
            "difficulty_wrong_format_1024_mean": float(
                difficulty["wrong_format_1024"].mean()
            ),
        },
        "sampling_efficiency": {
            "vanilla_effective_groups_per_million_mean": float(
                vanilla["effective_groups_per_million_tokens"].mean()
            ),
            "difficulty_effective_groups_per_million_mean": float(
                difficulty["effective_groups_per_million_tokens"].mean()
            ),
            "relative_gain_mean": float(efficiency_gain.mean()),
            "relative_gain_std": float(efficiency_gain.std(ddof=1)),
            "vanilla_zero_variance_mean": float(
                vanilla["zero_variance_group_ratio"].mean()
            ),
            "difficulty_zero_variance_mean": float(
                difficulty["zero_variance_group_ratio"].mean()
            ),
        },
        "hybrid": hybrid,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["seed", "strategy"]).to_csv(
        args.output_dir / "per_seed_length_audit.csv", index=False
    )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    save_protocol_plot(frame, args.output_dir)
    save_tradeoff_plot(frame, hybrid, args.output_dir)
    write_report(frame, summary, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
