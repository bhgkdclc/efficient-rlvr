import torch


def compute_group_diagnostics(
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    group_size: int,
    zero_variance_epsilon: float,
    format_rewards: torch.Tensor | None = None,
    answer_rewards: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute inexpensive rollout-group diagnostics on CPU tensors."""
    reward_groups = rewards.reshape(-1, group_size)
    group_variances = reward_groups.var(dim=-1, correction=0)
    zero_variance = group_variances.sqrt() < zero_variance_epsilon

    metrics = {
        "mean": rewards.mean(),
        "std": rewards.std(),
        "max": rewards.max(),
        "min": rewards.min(),
        "reward_mean": rewards.mean(),
        "reward_std": rewards.std(),
        "group_reward_variance_mean": group_variances.mean(),
        "advantage_mean": advantages.mean(),
        "advantage_std": advantages.std(),
        "zero_variance_group_ratio": zero_variance.float().mean(),
        "effective_group_ratio": (~zero_variance).float().mean(),
        # Every rollout in a group is effective/ineffective together for relative advantage.
        "effective_rollout_ratio": (~zero_variance).float().mean(),
    }

    if format_rewards is not None:
        metrics["format_reward_mean"] = format_rewards.mean()

    if answer_rewards is not None:
        answer_groups = answer_rewards.reshape(-1, group_size)
        all_correct = (answer_groups >= 1.0 - zero_variance_epsilon).all(dim=-1)
        all_wrong = (answer_groups <= zero_variance_epsilon).all(dim=-1)
        mixed = ~(all_correct | all_wrong)
        answer_variances = answer_groups.var(dim=-1, correction=0)
        metrics.update(
            {
                "answer_reward_mean": answer_rewards.mean(),
                "answer_zero_variance_group_ratio": (
                    answer_variances.sqrt() < zero_variance_epsilon
                ).float().mean(),
                "all_correct_group_ratio": all_correct.float().mean(),
                "all_wrong_group_ratio": all_wrong.float().mean(),
                "mixed_group_ratio": mixed.float().mean(),
            }
        )

    return metrics
