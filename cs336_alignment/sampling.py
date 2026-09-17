"""Small, testable helpers shared by GRPO sampling strategies."""

from __future__ import annotations

from collections.abc import Sequence

import torch


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
