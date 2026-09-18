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
    "length_truncated_responses": "rollout/length_truncated_responses",
    "length_truncated_response_ratio": (
        "rollout/length_truncated_response_ratio"
    ),
    "cumulative_generated_response_tokens": (
        "rollout/cumulative_generated_response_tokens"
    ),
    "cumulative_rollout_tokens": "rollout/cumulative_rollout_tokens",
    "sampling_strategy": "sampling/strategy",
    "batch_complete": "sampling/batch_complete",
    "attempted_groups": "sampling/attempted_groups",
    "accepted_groups": "sampling/accepted_groups",
    "discarded_zero_variance_groups": (
        "sampling/discarded_zero_variance_groups"
    ),
    "resample_rounds": "sampling/resample_rounds",
    "accepted_total_rollout_tokens": "sampling/accepted_total_rollout_tokens",
    "discarded_total_rollout_tokens": "sampling/discarded_total_rollout_tokens",
    "cumulative_discarded_total_rollout_tokens": (
        "sampling/cumulative_discarded_total_rollout_tokens"
    ),
    "difficulty_warmup_active": "difficulty/warmup_active",
    "difficulty_observed_prompts_before": "difficulty/observed_prompts_before",
    "difficulty_observed_prompts_after": "difficulty/observed_prompts_after",
    "difficulty_observed_prompt_ratio_after": (
        "difficulty/observed_prompt_ratio_after"
    ),
    "difficulty_selected_seen_ratio": "difficulty/selected_seen_ratio",
    "difficulty_selected_unseen_ratio": "difficulty/selected_unseen_ratio",
    "difficulty_selected_sample_count_mean_before": (
        "difficulty/selected_sample_count_mean_before"
    ),
    "difficulty_selected_coverage_score_mean": (
        "difficulty/selected_coverage_score_mean"
    ),
    "difficulty_selected_ema_accuracy_mean": (
        "difficulty/selected_ema_accuracy_mean"
    ),
    "difficulty_selected_boundary_score_mean": (
        "difficulty/selected_boundary_score_mean"
    ),
    "difficulty_global_seen_ema_accuracy_mean": (
        "difficulty/global_seen_ema_accuracy_mean"
    ),
    "difficulty_selected_group_accuracy_mean": (
        "difficulty/selected_group_accuracy_mean"
    ),
    "difficulty_selected_seen_group_accuracy_mean": (
        "difficulty/selected_seen_group_accuracy_mean"
    ),
    "difficulty_selected_seen_effective_group_ratio": (
        "difficulty/selected_seen_effective_group_ratio"
    ),
    "difficulty_selected_sample_count_mean_after": (
        "difficulty/selected_sample_count_mean_after"
    ),
    "hybrid_uniform_fraction": "difficulty/hybrid_uniform_fraction",
    "hybrid_uniform_effective_group_ratio": (
        "difficulty/hybrid_uniform_effective_group_ratio"
    ),
    "hybrid_difficulty_effective_group_ratio": (
        "difficulty/hybrid_difficulty_effective_group_ratio"
    ),
    "hybrid_uniform_group_accuracy_mean": (
        "difficulty/hybrid_uniform_group_accuracy_mean"
    ),
    "hybrid_difficulty_group_accuracy_mean": (
        "difficulty/hybrid_difficulty_group_accuracy_mean"
    ),
    "hybrid_uniform_total_rollout_tokens": (
        "difficulty/hybrid_uniform_total_rollout_tokens"
    ),
    "hybrid_difficulty_total_rollout_tokens": (
        "difficulty/hybrid_difficulty_total_rollout_tokens"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path, help="Path to metrics.jsonl")
    parser.add_argument("--config", type=Path, help="Optional config.json")
    parser.add_argument(
        "--sampler-state",
        type=Path,
        help="Optional DifficultyAwareSampler state JSON",
    )
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


def mean_dict(
    frame: pd.DataFrame,
    columns: list[str],
    weight_column: str | None = None,
) -> dict[str, float]:
    if weight_column is None or weight_column not in frame:
        return {column: float(frame[column].mean()) for column in columns}
    weights = frame[weight_column].astype(float)
    return {
        column: float(np.average(frame[column].astype(float), weights=weights))
        for column in columns
    }


def add_rollout_token_axis(
    evaluations: pd.DataFrame, rollouts: pd.DataFrame
) -> pd.DataFrame:
    evaluations = evaluations.copy()
    rollout_timestamps = rollouts["timestamp"].to_numpy()
    tokens = rollouts["cumulative_rollout_tokens"].astype(int).to_numpy()

    def tokens_at_evaluation(timestamp: float) -> int:
        # Timestamp alignment includes a final incomplete dynamic collection,
        # whose rollout tokens were spent even though the model was not updated.
        eligible = np.flatnonzero(rollout_timestamps <= timestamp)
        return int(tokens[eligible[-1]]) if len(eligible) else 0

    evaluations["cumulative_rollout_tokens"] = [
        tokens_at_evaluation(float(timestamp))
        for timestamp in evaluations["timestamp"]
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


def save_difficulty_calibration_plot(
    rollouts: pd.DataFrame,
    output_dir: Path,
    rolling_window: int,
) -> None:
    if (
        "difficulty_selected_ema_accuracy_mean" not in rollouts
        or rollouts["difficulty_selected_ema_accuracy_mean"].isna().all()
    ):
        return
    has_seen_accuracy = (
        "difficulty_selected_seen_group_accuracy_mean" in rollouts
        and rollouts["difficulty_selected_seen_group_accuracy_mean"].notna().any()
    )
    observed_key = (
        "difficulty_selected_seen_group_accuracy_mean"
        if has_seen_accuracy
        else "difficulty_selected_group_accuracy_mean"
    )
    has_seen_effective_ratio = (
        "difficulty_selected_seen_effective_group_ratio" in rollouts
        and rollouts["difficulty_selected_seen_effective_group_ratio"]
        .notna()
        .any()
    )
    effective_key = (
        "difficulty_selected_seen_effective_group_ratio"
        if has_seen_effective_ratio
        else "effective_group_ratio"
    )
    valid = rollouts[
        ~rollouts["difficulty_warmup_active"].astype(bool)
        & rollouts["difficulty_selected_ema_accuracy_mean"].notna()
        & rollouts[observed_key].notna()
        & rollouts[effective_key].notna()
    ]
    if not (has_seen_accuracy and has_seen_effective_ratio):
        valid = valid[valid["difficulty_selected_seen_ratio"] == 1.0]
    if len(valid) < 2:
        (output_dir / "difficulty_calibration.png").unlink(missing_ok=True)
        return
    smooth = valid.select_dtypes(include=[np.number]).rolling(
        rolling_window, min_periods=1
    ).mean()
    step = valid["grpo_step"]
    warmup_end = int(valid.iloc[0]["grpo_step"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    axes[0].plot(
        step,
        smooth["difficulty_selected_ema_accuracy_mean"],
        label="Historical EMA prediction",
    )
    axes[0].plot(
        step,
        smooth[observed_key],
        label="Observed group accuracy",
    )
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        step,
        smooth["difficulty_selected_boundary_score_mean"],
        label="Predicted boundary score",
    )
    axes[1].plot(
        step,
        smooth[effective_key],
        label="Observed effective ratio",
    )
    axes[1].set_ylabel("Score / ratio")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.axvline(
            warmup_end,
            color="black",
            linestyle="--",
            alpha=0.5,
            label="Warm-up end",
        )
        axis.set_xlabel("GRPO step")
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.25)
    fig.suptitle(f"Difficulty sampler calibration ({rolling_window}-step mean)")
    fig.tight_layout()
    fig.savefig(output_dir / "difficulty_calibration.png", dpi=180)
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
    has_sampling_counts = (
        "attempted_groups" in rollouts
        and rollouts["attempted_groups"].notna().all()
    )
    total_groups = (
        int(rollouts["attempted_groups"].sum())
        if has_sampling_counts
        else len(rollouts) * groups_per_step
    )
    group_weights = (
        rollouts["attempted_groups"]
        if has_sampling_counts
        else pd.Series(groups_per_step, index=rollouts.index)
    )
    zero_groups = int(
        round((rollouts["zero_variance_group_ratio"] * group_weights).sum())
    )
    accepted_groups = (
        int(rollouts["accepted_groups"].sum())
        if has_sampling_counts
        else total_groups
    )
    discarded_groups = (
        int(rollouts["discarded_zero_variance_groups"].sum())
        if has_sampling_counts
        else 0
    )
    if has_sampling_counts and "batch_complete" in rollouts:
        complete_mask = rollouts["batch_complete"].astype(bool)
        optimized_groups = int(
            rollouts.loc[complete_mask, "accepted_groups"].sum()
        )
        optimized_effective_groups = int(
            round(
                (
                    rollouts.loc[complete_mask, "effective_group_ratio"]
                    * rollouts.loc[complete_mask, "attempted_groups"]
                ).sum()
            )
        )
    else:
        optimized_groups = total_groups
        optimized_effective_groups = total_groups - zero_groups

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
    if "length_truncated_response_ratio" in rollouts:
        aggregate_columns.append("length_truncated_response_ratio")
    windows = {
        "first_20_steps": rollouts.head(20),
        "middle_20_steps": rollouts.iloc[
            max(0, len(rollouts) // 2 - 10) : len(rollouts) // 2 + 10
        ],
        "last_20_steps": rollouts.tail(20),
        "all_steps": rollouts,
    }
    window_summary = {
        name: mean_dict(
            frame,
            aggregate_columns,
            "attempted_groups" if has_sampling_counts else None,
        )
        for name, frame in windows.items()
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
    exact_discarded_tokens = (
        int(rollouts["discarded_total_rollout_tokens"].sum())
        if has_sampling_counts
        else None
    )
    zero_trend_correlation = float(
        np.corrcoef(
            rollouts["grpo_step"], rollouts["zero_variance_group_ratio"]
        )[0, 1]
    )

    difficulty_summary = None
    if config.get("sampling_strategy") in {
        "difficulty",
        "difficulty_dynamic",
        "hybrid",
    }:
        def summarize_phase(frame: pd.DataFrame) -> dict | None:
            if frame.empty:
                return None
            phase_weights = frame["attempted_groups"].astype(float)
            phase_groups = int(phase_weights.sum())
            phase_tokens = int(frame["total_rollout_tokens"].sum())
            phase_effective_groups = int(
                round((frame["effective_group_ratio"] * phase_weights).sum())
            )
            phase_zero_tokens = float(
                (
                    frame["total_rollout_tokens"]
                    * frame["zero_variance_group_ratio"]
                ).sum()
            )
            return {
                "steps": int(len(frame)),
                "groups": phase_groups,
                "rollout_tokens": phase_tokens,
                "zero_variance_group_ratio": float(
                    np.average(
                        frame["zero_variance_group_ratio"],
                        weights=phase_weights,
                    )
                ),
                "effective_groups": phase_effective_groups,
                "effective_groups_per_million_rollout_tokens": (
                    phase_effective_groups / phase_tokens * 1e6
                ),
                "estimated_zero_variance_token_ratio": (
                    phase_zero_tokens / phase_tokens
                ),
                "answer_reward_mean": float(
                    np.average(frame["answer_reward_mean"], weights=phase_weights)
                ),
                "format_reward_mean": float(
                    np.average(frame["format_reward_mean"], weights=phase_weights)
                ),
                "average_response_length": float(
                    np.average(
                        frame["average_response_length"], weights=phase_weights
                    )
                ),
                "length_truncated_response_ratio": (
                    float(
                        np.average(
                            frame["length_truncated_response_ratio"],
                            weights=phase_weights,
                        )
                    )
                    if "length_truncated_response_ratio" in frame
                    else None
                ),
            }

        warmup = rollouts[rollouts["difficulty_warmup_active"].astype(bool)]
        post_warmup = rollouts[~rollouts["difficulty_warmup_active"].astype(bool)]
        has_seen_accuracy = (
            "difficulty_selected_seen_group_accuracy_mean" in post_warmup
            and post_warmup[
                "difficulty_selected_seen_group_accuracy_mean"
            ].notna().any()
        )
        observed_accuracy_key = (
            "difficulty_selected_seen_group_accuracy_mean"
            if has_seen_accuracy
            else "difficulty_selected_group_accuracy_mean"
        )
        has_seen_effective_ratio = (
            "difficulty_selected_seen_effective_group_ratio" in post_warmup
            and post_warmup[
                "difficulty_selected_seen_effective_group_ratio"
            ].notna().any()
        )
        observed_effective_key = (
            "difficulty_selected_seen_effective_group_ratio"
            if has_seen_effective_ratio
            else "effective_group_ratio"
        )
        calibrated = post_warmup[
            post_warmup["difficulty_selected_ema_accuracy_mean"].notna()
            & post_warmup[observed_accuracy_key].notna()
            & post_warmup[observed_effective_key].notna()
        ]
        if not (has_seen_accuracy and has_seen_effective_ratio):
            calibrated = calibrated[
                calibrated["difficulty_selected_seen_ratio"] == 1.0
            ]
        predicted_accuracy = calibrated["difficulty_selected_ema_accuracy_mean"]
        enough_calibration = len(calibrated) >= 2
        boundary_scores = calibrated[
            "difficulty_selected_boundary_score_mean"
        ]
        effective_ratios = calibrated[observed_effective_key]
        correlation_available = (
            enough_calibration
            and boundary_scores.nunique() > 1
            and effective_ratios.nunique() > 1
        )
        difficulty_summary = {
            "warmup_groups": int(config.get("difficulty_warmup_groups", 0)),
            "ema_beta": float(config.get("difficulty_ema_beta", 0.0)),
            "uniform_epsilon": float(
                config.get("sampling_uniform_epsilon", 0.0)
            ),
            "coverage_weight": float(
                config.get("difficulty_coverage_weight", 0.0)
            ),
            "phase_metrics": {
                "warmup": summarize_phase(warmup),
                "post_warmup": summarize_phase(post_warmup),
            },
            "observed_prompts": int(
                rollouts.iloc[-1]["difficulty_observed_prompts_after"]
            ),
            "dataset_prompt_coverage": (
                float(
                    rollouts.iloc[-1]["difficulty_observed_prompts_after"]
                    / int(config["train_samples"])
                )
                if int(config.get("train_samples", 0)) > 0
                else None
            ),
            "post_warmup_selected_seen_ratio": float(
                post_warmup["difficulty_selected_seen_ratio"].mean()
            ),
            "calibration_batch_count": int(len(calibrated)),
            "post_warmup_predicted_accuracy_mean": (
                float(predicted_accuracy.mean()) if enough_calibration else None
            ),
            "post_warmup_observed_group_accuracy_mean": (
                float(calibrated[observed_accuracy_key].mean())
                if enough_calibration
                else None
            ),
            "post_warmup_accuracy_calibration_gap": (
                float(
                    (
                        calibrated[observed_accuracy_key]
                        - calibrated["difficulty_selected_ema_accuracy_mean"]
                    ).mean()
                )
                if enough_calibration
                else None
            ),
            "boundary_score_vs_effective_ratio_pearson_r": (
                float(
                    np.corrcoef(
                        calibrated["difficulty_selected_boundary_score_mean"],
                        calibrated[observed_effective_key],
                    )[0, 1]
                )
                if correlation_available
                else None
            ),
            "last_20_predicted_accuracy_mean": (
                float(predicted_accuracy.tail(20).mean())
                if enough_calibration
                else None
            ),
            "last_20_observed_group_accuracy_mean": (
                float(calibrated[observed_accuracy_key].tail(20).mean())
                if enough_calibration
                else None
            ),
        }
        if config.get("sampling_strategy") == "hybrid":
            uniform_groups = int(
                round(
                    (
                        rollouts["attempted_groups"]
                        * rollouts["hybrid_uniform_fraction"]
                    ).sum()
                )
            )
            difficulty_groups = int(total_groups - uniform_groups)
            uniform_tokens = int(
                rollouts["hybrid_uniform_total_rollout_tokens"].sum()
            )
            difficulty_tokens = int(
                rollouts["hybrid_difficulty_total_rollout_tokens"].sum()
            )
            difficulty_summary["hybrid"] = {
                "configured_uniform_fraction": float(
                    config.get("hybrid_uniform_fraction", 0.5)
                ),
                "uniform_groups": uniform_groups,
                "difficulty_groups": difficulty_groups,
                "uniform_effective_group_ratio": float(
                    np.average(
                        rollouts["hybrid_uniform_effective_group_ratio"],
                        weights=(
                            rollouts["attempted_groups"]
                            * rollouts["hybrid_uniform_fraction"]
                        ),
                    )
                ),
                "difficulty_effective_group_ratio": float(
                    np.average(
                        rollouts["hybrid_difficulty_effective_group_ratio"],
                        weights=(
                            rollouts["attempted_groups"]
                            * (1.0 - rollouts["hybrid_uniform_fraction"])
                        ),
                    )
                ),
                "uniform_group_accuracy_mean": float(
                    rollouts["hybrid_uniform_group_accuracy_mean"].mean()
                ),
                "difficulty_group_accuracy_mean": float(
                    rollouts["hybrid_difficulty_group_accuracy_mean"].mean()
                ),
                "uniform_rollout_tokens": uniform_tokens,
                "difficulty_rollout_tokens": difficulty_tokens,
                "uniform_effective_groups_per_million_rollout_tokens": (
                    uniform_groups
                    * float(
                        np.average(
                            rollouts["hybrid_uniform_effective_group_ratio"],
                            weights=(
                                rollouts["attempted_groups"]
                                * rollouts["hybrid_uniform_fraction"]
                            ),
                        )
                    )
                    / uniform_tokens
                    * 1e6
                ),
                "difficulty_effective_groups_per_million_rollout_tokens": (
                    difficulty_groups
                    * float(
                        np.average(
                            rollouts[
                                "hybrid_difficulty_effective_group_ratio"
                            ],
                            weights=(
                                rollouts["attempted_groups"]
                                * (1.0 - rollouts["hybrid_uniform_fraction"])
                            ),
                        )
                    )
                    / difficulty_tokens
                    * 1e6
                ),
            }

    summary = {
        "run": {
            "git_commit": config.get("git_commit"),
            "seed": config.get("seed"),
            "model_name_or_path": config.get("model_name_or_path"),
            "dataset": config.get("train_data_path"),
            "sampling_strategy": config.get("sampling_strategy"),
            "group_size": group_size,
            "rollout_batch_size": rollout_batch_size,
            "completed_grpo_steps": int(optimizer["grpo_step"].nunique()),
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
            "exact_discarded_rollout_tokens": exact_discarded_tokens,
            "exact_discarded_rollout_token_ratio": (
                exact_discarded_tokens / total_rollout_tokens
                if exact_discarded_tokens is not None
                else None
            ),
        },
        "groups": {
            "total_groups": total_groups,
            "attempted_groups": total_groups,
            "zero_variance_groups": zero_groups,
            "effective_groups": total_groups - zero_groups,
            "zero_variance_group_ratio": zero_groups / total_groups,
            "accepted_groups": accepted_groups,
            "discarded_groups": discarded_groups,
            "optimized_groups": optimized_groups,
            "optimized_effective_groups": optimized_effective_groups,
            "optimizer_batch_effective_group_ratio": (
                optimized_effective_groups / optimized_groups
            ),
            "effective_groups_per_million_rollout_tokens": (
                (total_groups - zero_groups) / (total_rollout_tokens / 1e6)
            ),
            "optimized_effective_groups_per_million_rollout_tokens": (
                optimized_effective_groups / (total_rollout_tokens / 1e6)
            ),
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
                "average_response_length": float(
                    other.iloc[-1]["eval/average_response_length"]
                ),
                "length_truncated_responses": (
                    int(other.iloc[-1]["eval/length_truncated_responses"])
                    if "eval/length_truncated_responses" in other
                    else None
                ),
                "length_truncated_response_ratio": (
                    float(
                        other.iloc[-1][
                            "eval/length_truncated_response_ratio"
                        ]
                    )
                    if "eval/length_truncated_response_ratio" in other
                    else None
                ),
            }
            if not other.empty
            else None
        ),
        "trend": {
            "step_vs_zero_variance_pearson_r": zero_trend_correlation,
            "window_means": window_summary,
        },
        "difficulty_sampler": difficulty_summary,
        "notes": [
            "Intermediate and initial accuracy use the fixed evaluation subset; the final full evaluation is not directly comparable.",
            (
                "Exact discarded token cost comes from per-group dynamic-filtering logs."
                if config.get("sampling_strategy")
                in {"dynamic", "difficulty_dynamic"}
                else "Estimated zero-variance token cost weights each batch token count by its zero-variance group ratio; per-group token lengths were not logged."
            ),
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
    if args.sampler_state:
        with args.sampler_state.open(encoding="utf-8") as file:
            sampler_state = json.load(file)
        counts = np.array(
            [item["sample_count"] for item in sampler_state["prompt_states"]]
        )
        ema_accuracy = np.array(
            [item["ema_accuracy"] for item in sampler_state["prompt_states"]]
        )
        summary["difficulty_sampler"].update(
            {
                "dataset_prompt_coverage": (
                    sampler_state["observed_prompt_count"]
                    / sampler_state["num_prompts"]
                ),
                "total_prompt_group_selections": int(counts.sum()),
                "sample_count_mean": float(counts.mean()),
                "sample_count_median": float(np.median(counts)),
                "sample_count_p90": float(np.quantile(counts, 0.9)),
                "sample_count_max": int(counts.max()),
                "final_prompt_ema_accuracy_mean": float(ema_accuracy.mean()),
                "prompts_with_zero_ema_accuracy": int((ema_accuracy == 0).sum()),
                "prompts_with_one_ema_accuracy": int((ema_accuracy == 1).sum()),
            }
        )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    comparable_count = int(summary["comparable_evaluation"]["sample_count"])
    save_accuracy_plots(evaluations, args.output_dir, comparable_count)
    save_group_plot(rollouts, args.output_dir, args.rolling_window)
    save_dynamics_plot(rollouts, optimizer, args.output_dir, args.rolling_window)
    save_difficulty_calibration_plot(
        rollouts, args.output_dir, args.rolling_window
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
