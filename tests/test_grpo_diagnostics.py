import pytest

from cs336_alignment.drgrpo_grader import question_only_reward_fn
from cs336_alignment.grpo import compute_group_normalized_rewards
from cs336_alignment.rewards import make_weighted_reward_fn


def test_group_diagnostics_distinguish_all_correct_wrong_and_mixed():
    rewards = [0.0] * 4 + [0.0, 1.0, 0.0, 1.0] + [1.0] * 4

    def reward_fn(response, _ground_truth):
        reward = rewards[int(response)]
        return {
            "reward": reward,
            "format_reward": 1.0,
            "answer_reward": reward,
        }

    _, _, metadata = compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=[str(i) for i in range(len(rewards))],
        repeated_ground_truths=[""] * len(rewards),
        group_size=4,
        advantage_eps=1e-6,
        normalize_by_std=True,
        zero_variance_epsilon=1e-8,
    )

    assert metadata["zero_variance_group_ratio"].item() == pytest.approx(2 / 3)
    assert metadata["effective_group_ratio"].item() == pytest.approx(1 / 3)
    assert metadata["all_correct_group_ratio"].item() == pytest.approx(1 / 3)
    assert metadata["all_wrong_group_ratio"].item() == pytest.approx(1 / 3)
    assert metadata["mixed_group_ratio"].item() == pytest.approx(1 / 3)


def test_weighted_reward_preserves_components():
    def component_reward_fn(_response, _ground_truth):
        return {"reward": 0.0, "format_reward": 1.0, "answer_reward": 0.5}

    reward = make_weighted_reward_fn(component_reward_fn, 0.1, 1.0)("", "")

    assert reward == {
        "reward": pytest.approx(0.6),
        "format_reward": 1.0,
        "answer_reward": 0.5,
    }


def test_question_only_reward_supports_training_without_sft_format():
    reward_fn = make_weighted_reward_fn(question_only_reward_fn, 0.1, 1.0)

    correct = reward_fn(r"The result is \boxed{72}.", "72")
    wrong = reward_fn(r"The result is \boxed{71}.", "72")

    assert correct == {
        "reward": pytest.approx(1.1),
        "format_reward": 1.0,
        "answer_reward": 1.0,
    }
    assert wrong == {
        "reward": pytest.approx(0.1),
        "format_reward": 1.0,
        "answer_reward": 0.0,
    }
