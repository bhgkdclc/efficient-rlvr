from collections.abc import Callable


RewardFn = Callable[[str, str], dict[str, float]]


def make_weighted_reward_fn(
    component_reward_fn: RewardFn,
    format_weight: float,
    answer_weight: float,
) -> RewardFn:
    """Combine separately logged format and answer rewards into the training reward."""
    if format_weight < 0 or answer_weight < 0:
        raise ValueError("reward weights must be non-negative")
    if format_weight == 0 and answer_weight == 0:
        raise ValueError("at least one reward weight must be positive")

    def weighted_reward_fn(response: str, ground_truth: str) -> dict[str, float]:
        components = dict(component_reward_fn(response, ground_truth))
        format_reward = float(components.get("format_reward", 0.0))
        answer_reward = float(components.get("answer_reward", 0.0))
        components["format_reward"] = format_reward
        components["answer_reward"] = answer_reward
        components["reward"] = (
            format_weight * format_reward + answer_weight * answer_reward
        )
        return components

    return weighted_reward_fn
