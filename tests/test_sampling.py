import pytest

from cs336_alignment.sampling import (
    DifficultyAwareSampler,
    split_group_indices_by_reward_variance,
)


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


def test_difficulty_sampler_warms_up_with_unique_prompts():
    sampler = DifficultyAwareSampler(
        num_prompts=20,
        ema_beta=0.9,
        uniform_epsilon=0.1,
        warmup_groups=8,
        seed=42,
    )

    first, first_metadata = sampler.sample(4)
    sampler.update(first, [0.0, 0.25, 0.5, 1.0], step=0)
    second, second_metadata = sampler.sample(4)

    assert len(set(first + second)) == 8
    assert first_metadata["warmup_active"] is True
    assert first_metadata["selected_seen_ratio"] == 0.0
    assert second_metadata["observed_prompts_before"] == 4


def test_difficulty_sampler_updates_ema_accuracy():
    sampler = DifficultyAwareSampler(
        num_prompts=4,
        ema_beta=0.5,
        uniform_epsilon=0.0,
        warmup_groups=0,
        seed=42,
    )

    sampler.update([2], [0.25], step=0)
    sampler.update([2], [0.75], step=1)

    assert sampler.ema_accuracy[2] == pytest.approx(0.5)
    assert sampler.sample_counts[2] == 2
    assert sampler.last_sampled_steps[2] == 1
    assert sampler.boundary_score(sampler.ema_accuracy[2]) == pytest.approx(1.0)


def test_difficulty_sampler_prefers_model_boundary_without_exploration():
    sampler = DifficultyAwareSampler(
        num_prompts=3,
        ema_beta=0.9,
        uniform_epsilon=0.0,
        warmup_groups=0,
        seed=42,
    )
    sampler.update([0, 1, 2], [0.0, 0.5, 1.0], step=0)

    selected, metadata = sampler.sample(1)

    assert selected == [1]
    assert metadata["selected_ema_accuracy_mean"] == pytest.approx(0.5)
    assert metadata["selected_boundary_score_mean"] == pytest.approx(1.0)


def test_difficulty_sampler_excludes_prompts_already_attempted_in_batch():
    sampler = DifficultyAwareSampler(
        num_prompts=4,
        ema_beta=0.5,
        uniform_epsilon=0.0,
        warmup_groups=0,
        seed=42,
    )
    sampler.update([0, 1, 2, 3], [0.5, 0.5, 0.5, 0.5], step=0)

    selected, _ = sampler.sample(2, excluded={0, 1})

    assert set(selected) == {2, 3}


def test_difficulty_sampler_can_sample_uniformly_with_metadata():
    sampler = DifficultyAwareSampler(
        num_prompts=8,
        ema_beta=0.5,
        uniform_epsilon=0.1,
        warmup_groups=0,
        seed=42,
    )
    sampler.update([0, 1], [0.25, 0.75], step=0)

    selected, metadata = sampler.sample_uniform(4, excluded={7})

    assert len(selected) == 4
    assert len(set(selected)) == 4
    assert 7 not in selected
    assert metadata["observed_prompts_before"] == 2
    assert metadata["selected_seen_ratio"] in {0.0, 0.25, 0.5}


def test_hybrid_sampler_uses_explicit_uniform_and_difficulty_strata():
    sampler = DifficultyAwareSampler(
        num_prompts=20,
        ema_beta=0.5,
        uniform_epsilon=0.0,
        warmup_groups=0,
        seed=42,
    )
    sampler.update([0, 1, 2], [0.0, 0.5, 1.0], step=0)

    selected, metadata, uniform_count = sampler.sample_hybrid(
        sample_count=8,
        uniform_fraction=0.5,
    )

    assert len(selected) == 8
    assert len(set(selected)) == 8
    assert uniform_count == 4
    assert metadata["hybrid_uniform_groups"] == 4
    assert metadata["hybrid_difficulty_groups"] == 4
    assert metadata["hybrid_uniform_fraction"] == pytest.approx(0.5)


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_hybrid_sampler_rejects_degenerate_fraction(fraction):
    sampler = DifficultyAwareSampler(
        num_prompts=20,
        ema_beta=0.5,
        uniform_epsilon=0.1,
        warmup_groups=0,
        seed=42,
    )

    with pytest.raises(ValueError, match="uniform_fraction"):
        sampler.sample_hybrid(8, fraction)


def test_coverage_weight_prefers_under_sampled_prompts():
    sampler = DifficultyAwareSampler(
        num_prompts=3,
        ema_beta=0.5,
        uniform_epsilon=0.0,
        warmup_groups=0,
        seed=42,
        coverage_weight=1.0,
    )
    for step in range(10):
        sampler.update([0], [0.5], step=step)
    sampler.update([1], [0.5], step=10)

    selected, metadata = sampler.sample(1)

    assert selected == [2]
    assert metadata["selected_unseen_ratio"] == 1.0
    assert metadata["selected_sample_count_mean_before"] == 0.0


def test_difficulty_sampler_rejects_invalid_coverage_weight():
    with pytest.raises(ValueError, match="coverage_weight"):
        DifficultyAwareSampler(
            num_prompts=3,
            ema_beta=0.5,
            uniform_epsilon=0.1,
            warmup_groups=0,
            seed=42,
            coverage_weight=1.1,
        )


def test_difficulty_sampler_state_only_contains_observed_prompts(tmp_path):
    sampler = DifficultyAwareSampler(
        num_prompts=10,
        ema_beta=0.9,
        uniform_epsilon=0.1,
        warmup_groups=2,
        seed=42,
    )
    sampler.update([3, 7], [0.25, 0.75], step=4)
    state_path = tmp_path / "sampler_state.json"

    sampler.save(state_path)

    state = sampler.state_dict()
    assert state["observed_prompt_count"] == 2
    assert state["coverage_weight"] == 0.0
    assert [item["prompt_index"] for item in state["prompt_states"]] == [3, 7]
    assert state_path.exists()
