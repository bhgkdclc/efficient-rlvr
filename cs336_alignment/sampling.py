"""Small, testable helpers shared by GRPO sampling strategies."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

import torch


class DifficultyAwareSampler:
    """Sample previously mixed prompts while retaining uniform exploration."""

    def __init__(
        self,
        num_prompts: int,
        ema_beta: float,
        uniform_epsilon: float,
        warmup_groups: int,
        seed: int,
        coverage_weight: float = 0.0,
    ) -> None:
        if num_prompts <= 0:
            raise ValueError("num_prompts must be positive")
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1)")
        if not 0.0 <= uniform_epsilon <= 1.0:
            raise ValueError("uniform_epsilon must be in [0, 1]")
        if not 0 <= warmup_groups <= num_prompts:
            raise ValueError("warmup_groups must be between 0 and num_prompts")
        if not 0.0 <= coverage_weight <= 1.0:
            raise ValueError("coverage_weight must be in [0, 1]")

        self.num_prompts = num_prompts
        self.ema_beta = ema_beta
        self.uniform_epsilon = uniform_epsilon
        self.warmup_groups = warmup_groups
        self.coverage_weight = coverage_weight
        self.rng = random.Random(seed)
        self.ema_accuracy: list[float | None] = [None] * num_prompts
        self.sample_counts = [0] * num_prompts
        self.last_sampled_steps = [-1] * num_prompts

    @staticmethod
    def boundary_score(success_rate: float) -> float:
        return 4.0 * success_rate * (1.0 - success_rate)

    @staticmethod
    def coverage_score(sample_count: int) -> float:
        """Favor unseen and under-sampled prompts without hard filtering."""
        return 1.0 / math.sqrt(sample_count + 1.0)

    @property
    def observed_prompt_count(self) -> int:
        return sum(count > 0 for count in self.sample_counts)

    def _weighted_sample_without_replacement(
        self,
        candidates: list[int],
        weights: list[float],
        sample_count: int,
    ) -> list[int]:
        selected, _ = self._weighted_sample_without_replacement_with_importance(
            candidates,
            weights,
            sample_count,
        )
        return selected

    def _weighted_sample_without_replacement_with_importance(
        self,
        candidates: list[int],
        weights: list[float],
        sample_count: int,
    ) -> tuple[list[int], list[float]]:
        """Sample and return approximate uniform-target importance ratios.

        The target distribution is uniform over the candidates remaining at
        each draw.  Because batches are tiny relative to the prompt pool, the
        per-draw ratio is a low-variance approximation to full inclusion-
        probability correction for weighted sampling without replacement.
        """
        selected: list[int] = []
        importance_weights: list[float] = []
        candidates = candidates.copy()
        weights = weights.copy()
        for _ in range(sample_count):
            total_weight = sum(weights)
            if total_weight <= 0.0:
                selected_position = self.rng.randrange(len(candidates))
                behavior_probability = 1.0 / len(candidates)
            else:
                threshold = self.rng.random() * total_weight
                cumulative = 0.0
                selected_position = len(candidates) - 1
                for position, weight in enumerate(weights):
                    cumulative += weight
                    if cumulative > threshold:
                        selected_position = position
                        break
                behavior_probability = weights[selected_position] / total_weight
            target_probability = 1.0 / len(candidates)
            selected.append(candidates.pop(selected_position))
            importance_weights.append(
                target_probability / behavior_probability
            )
            weights.pop(selected_position)
        return selected, importance_weights

    def _difficulty_sample(
        self,
        sample_count: int,
        excluded: set[int],
    ) -> list[int]:
        selected, _ = self._difficulty_sample_with_importance(
            sample_count,
            excluded,
        )
        return selected

    def _difficulty_sample_with_importance(
        self,
        sample_count: int,
        excluded: set[int],
    ) -> tuple[list[int], list[float]]:
        candidates = [
            index for index in range(self.num_prompts) if index not in excluded
        ]
        boundary_scores = [
            (
                self.boundary_score(self.ema_accuracy[index])
                if self.ema_accuracy[index] is not None
                else 0.0
            )
            for index in candidates
        ]
        boundary_total = sum(boundary_scores)
        uniform_probability = 1.0 / len(candidates)
        boundary_probabilities = (
            [score / boundary_total for score in boundary_scores]
            if boundary_total > 0.0
            else [uniform_probability] * len(candidates)
        )
        coverage_scores = [
            self.coverage_score(self.sample_counts[index]) for index in candidates
        ]
        coverage_total = sum(coverage_scores)
        coverage_probabilities = [
            score / coverage_total for score in coverage_scores
        ]
        weights = [
            (1.0 - self.uniform_epsilon)
            * (
                (1.0 - self.coverage_weight) * boundary_probability
                + self.coverage_weight * coverage_probability
            )
            + self.uniform_epsilon * uniform_probability
            for boundary_probability, coverage_probability in zip(
                boundary_probabilities, coverage_probabilities
            )
        ]
        return self._weighted_sample_without_replacement_with_importance(
            candidates, weights, sample_count
        )

    def _selection_metadata(
        self,
        selected: list[int],
        observed_before: int,
    ) -> dict[str, float | int | bool | None]:
        seen_selected = [
            index for index in selected if self.ema_accuracy[index] is not None
        ]
        selected_accuracies = [
            float(self.ema_accuracy[index]) for index in seen_selected
        ]
        selected_scores = [
            self.boundary_score(accuracy) for accuracy in selected_accuracies
        ]
        selected_counts = [self.sample_counts[index] for index in selected]
        selected_coverage_scores = [
            self.coverage_score(count) for count in selected_counts
        ]
        all_accuracies = [
            float(accuracy)
            for accuracy in self.ema_accuracy
            if accuracy is not None
        ]
        return {
            "warmup_active": observed_before < self.warmup_groups,
            "observed_prompts_before": observed_before,
            "observed_prompt_ratio_before": observed_before / self.num_prompts,
            "selected_seen_ratio": len(seen_selected) / len(selected),
            "selected_unseen_ratio": 1.0 - len(seen_selected) / len(selected),
            "selected_sample_count_mean_before": fmean(selected_counts),
            "selected_coverage_score_mean": fmean(selected_coverage_scores),
            "selected_ema_accuracy_mean": (
                fmean(selected_accuracies) if selected_accuracies else None
            ),
            "selected_boundary_score_mean": (
                fmean(selected_scores) if selected_scores else None
            ),
            "global_seen_ema_accuracy_mean": (
                fmean(all_accuracies) if all_accuracies else None
            ),
        }

    def sample(
        self,
        sample_count: int,
        excluded: set[int] | None = None,
    ) -> tuple[list[int], dict[str, float | int | bool | None]]:
        selected, metadata, _ = self.sample_with_importance_weights(
            sample_count,
            excluded,
        )
        return selected, metadata

    def sample_with_importance_weights(
        self,
        sample_count: int,
        excluded: set[int] | None = None,
    ) -> tuple[
        list[int],
        dict[str, float | int | bool | None],
        list[float],
    ]:
        """Sample prompts and estimate uniform-target importance weights.

        Initial unseen-prompt warmup draws receive unit weight.  Once the
        difficulty distribution is active, each returned weight is the ratio
        between the uniform and behavior probabilities at that draw.
        """
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        excluded = set() if excluded is None else set(excluded)
        if any(index < 0 or index >= self.num_prompts for index in excluded):
            raise IndexError("excluded prompt index out of range")
        if sample_count > self.num_prompts - len(excluded):
            raise ValueError("sample_count cannot exceed available prompts")

        observed_before = self.observed_prompt_count
        selected: list[int] = []
        importance_weights: list[float] = []
        warmup_needed = min(
            sample_count,
            max(0, self.warmup_groups - observed_before),
        )
        if warmup_needed:
            unseen = [
                index
                for index, count in enumerate(self.sample_counts)
                if count == 0 and index not in excluded
            ]
            warmup_needed = min(warmup_needed, len(unseen))
            selected.extend(self.rng.sample(unseen, warmup_needed))
            importance_weights.extend([1.0] * warmup_needed)

        remaining = sample_count - len(selected)
        if remaining:
            difficulty_selected, difficulty_importance_weights = (
                self._difficulty_sample_with_importance(
                    remaining,
                    excluded | set(selected),
                )
            )
            selected.extend(difficulty_selected)
            importance_weights.extend(difficulty_importance_weights)

        return (
            selected,
            self._selection_metadata(selected, observed_before),
            importance_weights,
        )

    def sample_uniform(
        self,
        sample_count: int,
        excluded: set[int] | None = None,
    ) -> tuple[list[int], dict[str, float | int | bool | None]]:
        """Sample prompts uniformly while retaining difficulty diagnostics.

        This is used by scheduled samplers after they switch away from
        difficulty-weighted selection.  Keeping the sampler state live during
        the uniform phase lets us measure prompt coverage and update EMA values
        without allowing those values to influence selection.
        """
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        excluded = set() if excluded is None else set(excluded)
        if any(index < 0 or index >= self.num_prompts for index in excluded):
            raise IndexError("excluded prompt index out of range")
        candidates = [
            index for index in range(self.num_prompts) if index not in excluded
        ]
        if sample_count > len(candidates):
            raise ValueError("sample_count cannot exceed available prompts")

        observed_before = self.observed_prompt_count
        selected = self.rng.sample(candidates, sample_count)
        return selected, self._selection_metadata(selected, observed_before)

    def sample_hybrid(
        self,
        sample_count: int,
        uniform_fraction: float,
    ) -> tuple[
        list[int],
        dict[str, float | int | bool | None],
        int,
    ]:
        """Draw an explicit uniform stratum followed by a difficulty stratum."""
        if sample_count <= 1:
            raise ValueError("hybrid sampling requires at least two prompt groups")
        if sample_count > self.num_prompts:
            raise ValueError("sample_count cannot exceed num_prompts")
        if not 0.0 < uniform_fraction < 1.0:
            raise ValueError("uniform_fraction must be strictly between 0 and 1")

        observed_before = self.observed_prompt_count
        uniform_count = min(
            sample_count - 1,
            max(1, round(sample_count * uniform_fraction)),
        )
        uniform_indices = self.rng.sample(
            range(self.num_prompts), uniform_count
        )
        difficulty_indices, _ = self.sample(
            sample_count - uniform_count,
            excluded=set(uniform_indices),
        )
        selected = uniform_indices + difficulty_indices
        metadata = self._selection_metadata(selected, observed_before)
        metadata.update(
            {
                "hybrid_uniform_fraction": uniform_count / sample_count,
                "hybrid_uniform_groups": uniform_count,
                "hybrid_difficulty_groups": sample_count - uniform_count,
            }
        )
        return selected, metadata, uniform_count

    def update(
        self,
        prompt_indices: Sequence[int],
        group_accuracies: Sequence[float],
        step: int,
    ) -> None:
        if len(prompt_indices) != len(group_accuracies):
            raise ValueError("prompt_indices and group_accuracies must have equal length")
        if len(set(prompt_indices)) != len(prompt_indices):
            raise ValueError("prompt_indices must be unique within one update")
        for prompt_index, accuracy in zip(prompt_indices, group_accuracies):
            if not 0 <= prompt_index < self.num_prompts:
                raise IndexError("prompt index out of range")
            if not 0.0 <= accuracy <= 1.0:
                raise ValueError("group accuracy must be in [0, 1]")
            previous = self.ema_accuracy[prompt_index]
            self.ema_accuracy[prompt_index] = (
                float(accuracy)
                if previous is None
                else self.ema_beta * previous + (1.0 - self.ema_beta) * accuracy
            )
            self.sample_counts[prompt_index] += 1
            self.last_sampled_steps[prompt_index] = step

    def state_dict(self) -> dict:
        prompt_states = [
            {
                "prompt_index": index,
                "ema_accuracy": self.ema_accuracy[index],
                "sample_count": self.sample_counts[index],
                "last_sampled_step": self.last_sampled_steps[index],
            }
            for index in range(self.num_prompts)
            if self.sample_counts[index] > 0
        ]
        return {
            "num_prompts": self.num_prompts,
            "ema_beta": self.ema_beta,
            "uniform_epsilon": self.uniform_epsilon,
            "coverage_weight": self.coverage_weight,
            "warmup_groups": self.warmup_groups,
            "observed_prompt_count": self.observed_prompt_count,
            "prompt_states": prompt_states,
        }

    def save(self, path: Path) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(self.state_dict(), indent=2), encoding="utf-8"
        )
        temporary_path.replace(path)


def normalize_clipped_importance_weights(
    raw_weights: torch.Tensor | Sequence[float],
    clip_min: float,
    clip_max: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clip prompt-level importance ratios and normalize them to mean one."""
    if clip_min <= 0.0:
        raise ValueError("clip_min must be positive")
    if clip_max < clip_min:
        raise ValueError("clip_max must be greater than or equal to clip_min")

    weights = torch.as_tensor(raw_weights, dtype=torch.float32).flatten()
    if weights.numel() == 0:
        raise ValueError("raw_weights must not be empty")
    if not torch.isfinite(weights).all() or not torch.all(weights > 0.0):
        raise ValueError("raw_weights must be finite and positive")

    clipped = weights.clamp(min=clip_min, max=clip_max)
    normalized = clipped / clipped.mean()
    effective_sample_size = (
        normalized.sum().square() / normalized.square().sum()
    ).item()
    metadata = {
        "raw_mean": weights.mean().item(),
        "raw_min": weights.min().item(),
        "raw_max": weights.max().item(),
        "clipped_fraction": (weights != clipped).float().mean().item(),
        "normalized_mean": normalized.mean().item(),
        "normalized_std": normalized.std(correction=0).item(),
        "normalized_min": normalized.min().item(),
        "normalized_max": normalized.max().item(),
        "effective_sample_size": effective_sample_size,
        "effective_sample_size_ratio": effective_sample_size / weights.numel(),
    }
    return normalized, metadata


def split_group_indices_by_reward_variance(
    rewards: torch.Tensor | Sequence[float],
    group_size: int,
    zero_variance_epsilon: float,
) -> tuple[list[int], list[int]]:
    """Return (effective, zero-variance) group indices.

    The population standard deviation matches the zero-variance diagnostic used
    by the trainer. Groups with ``std < epsilon`` are considered ineffective.
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if zero_variance_epsilon < 0:
        raise ValueError("zero_variance_epsilon must be non-negative")

    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32).flatten()
    if reward_tensor.numel() == 0:
        raise ValueError("rewards must not be empty")
    if reward_tensor.numel() % group_size != 0:
        raise ValueError("number of rewards must be divisible by group_size")

    reward_groups = reward_tensor.reshape(-1, group_size)
    group_stds = reward_groups.var(dim=-1, correction=0).sqrt()
    effective_mask = group_stds >= zero_variance_epsilon
    effective = effective_mask.nonzero(as_tuple=False).flatten().tolist()
    zero_variance = (~effective_mask).nonzero(as_tuple=False).flatten().tolist()
    return effective, zero_variance
