"""Small, testable helpers shared by GRPO sampling strategies."""

from __future__ import annotations

import json
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
    ) -> None:
        if num_prompts <= 0:
            raise ValueError("num_prompts must be positive")
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1)")
        if not 0.0 <= uniform_epsilon <= 1.0:
            raise ValueError("uniform_epsilon must be in [0, 1]")
        if not 0 <= warmup_groups <= num_prompts:
            raise ValueError("warmup_groups must be between 0 and num_prompts")

        self.num_prompts = num_prompts
        self.ema_beta = ema_beta
        self.uniform_epsilon = uniform_epsilon
        self.warmup_groups = warmup_groups
        self.rng = random.Random(seed)
        self.ema_accuracy: list[float | None] = [None] * num_prompts
        self.sample_counts = [0] * num_prompts
        self.last_sampled_steps = [-1] * num_prompts

    @staticmethod
    def boundary_score(success_rate: float) -> float:
        return 4.0 * success_rate * (1.0 - success_rate)

    @property
    def observed_prompt_count(self) -> int:
        return sum(count > 0 for count in self.sample_counts)

    def _weighted_sample_without_replacement(
        self,
        candidates: list[int],
        weights: list[float],
        sample_count: int,
    ) -> list[int]:
        selected: list[int] = []
        candidates = candidates.copy()
        weights = weights.copy()
        for _ in range(sample_count):
            total_weight = sum(weights)
            if total_weight <= 0.0:
                selected_position = self.rng.randrange(len(candidates))
            else:
                threshold = self.rng.random() * total_weight
                cumulative = 0.0
                selected_position = len(candidates) - 1
                for position, weight in enumerate(weights):
                    cumulative += weight
                    if cumulative > threshold:
                        selected_position = position
                        break
            selected.append(candidates.pop(selected_position))
            weights.pop(selected_position)
        return selected

    def _difficulty_sample(
        self,
        sample_count: int,
        excluded: set[int],
    ) -> list[int]:
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
        if boundary_total <= 0.0:
            weights = [1.0] * len(candidates)
        else:
            uniform_probability = 1.0 / len(candidates)
            weights = [
                (1.0 - self.uniform_epsilon) * score / boundary_total
                + self.uniform_epsilon * uniform_probability
                for score in boundary_scores
            ]
        return self._weighted_sample_without_replacement(
            candidates, weights, sample_count
        )

    def sample(
        self, sample_count: int
    ) -> tuple[list[int], dict[str, float | int | bool | None]]:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if sample_count > self.num_prompts:
            raise ValueError("sample_count cannot exceed num_prompts")

        observed_before = self.observed_prompt_count
        selected: list[int] = []
        warmup_needed = min(
            sample_count,
            max(0, self.warmup_groups - observed_before),
        )
        if warmup_needed:
            unseen = [
                index
                for index, count in enumerate(self.sample_counts)
                if count == 0
            ]
            selected.extend(self.rng.sample(unseen, warmup_needed))

        remaining = sample_count - len(selected)
        if remaining:
            selected.extend(self._difficulty_sample(remaining, set(selected)))

        seen_selected = [
            index for index in selected if self.ema_accuracy[index] is not None
        ]
        selected_accuracies = [
            float(self.ema_accuracy[index]) for index in seen_selected
        ]
        selected_scores = [
            self.boundary_score(accuracy) for accuracy in selected_accuracies
        ]
        all_accuracies = [
            float(accuracy)
            for accuracy in self.ema_accuracy
            if accuracy is not None
        ]
        metadata: dict[str, float | int | bool | None] = {
            "warmup_active": observed_before < self.warmup_groups,
            "observed_prompts_before": observed_before,
            "selected_seen_ratio": len(seen_selected) / len(selected),
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
        return selected, metadata

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
