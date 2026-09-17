import pytest

from cs336_alignment.sampling import split_group_indices_by_reward_variance


def test_split_group_indices_by_reward_variance():
    effective, zero_variance = split_group_indices_by_reward_variance(
        rewards=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        group_size=4,
        zero_variance_epsilon=1e-8,
    )

    assert effective == [1]
    assert zero_variance == [0, 2]


def test_variance_threshold_is_strictly_less_than_epsilon():
    effective, zero_variance = split_group_indices_by_reward_variance(
        rewards=[0.0, 2.0],
        group_size=2,
        zero_variance_epsilon=1.0,
    )

    assert effective == [0]
    assert zero_variance == []


@pytest.mark.parametrize(
    ("rewards", "group_size", "epsilon", "message"),
    [
        ([], 4, 1e-8, "must not be empty"),
        ([0.0, 1.0, 0.0], 2, 1e-8, "must be divisible"),
        ([0.0], 0, 1e-8, "must be positive"),
        ([0.0], 1, -1.0, "must be non-negative"),
    ],
)
def test_split_group_indices_rejects_invalid_inputs(
    rewards, group_size, epsilon, message
):
    with pytest.raises(ValueError, match=message):
        split_group_indices_by_reward_variance(rewards, group_size, epsilon)
