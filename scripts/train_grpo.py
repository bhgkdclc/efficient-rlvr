"""Train the existing CS336 GRPO implementation with reproducible experiment logs.

The default path deliberately skips SFT and starts from Qwen2.5-Math-1.5B on
raw GSM8K using the repository's question-only prompt/reward implementation.
"""

import argparse
import gc
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.grpo import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
)
from cs336_alignment.rewards import make_weighted_reward_fn
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output


logger = logging.getLogger(__name__)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion: str) -> str | None:
    match = GSM_RE.search(completion)
    if match is None:
        return None
    return match.group(1).strip().replace(",", "")


def to_float(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return value.detach().float().cpu().tolist()
        return value.detach().float().item()
    if isinstance(value, Path):
        return str(value)
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return to_float(value)


class ExperimentLogger:
    """Write every metric locally and optionally mirror it to Weights & Biases."""

    def __init__(
        self,
        output_path: Path,
        config: dict[str, Any],
        run_name: str,
        wandb_mode: str,
        wandb_project: str,
    ) -> None:
        self.metrics_path = output_path / "metrics.jsonl"
        self.wandb_run = None
        if wandb_mode != "disabled":
            import wandb

            self.wandb_run = wandb.init(
                project=wandb_project,
                name=run_name,
                mode=wandb_mode,
                config=config,
            )

    def log(self, metrics: dict[str, Any]) -> None:
        record = jsonable({"timestamp": time.time(), **metrics})
        with self.metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log(record)

    def finish(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()


def get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_is_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float,
):
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(
    policy: PreTrainedModel,
    llm,
    vllm_device: str,
) -> None:
    policy.eval()
    policy.tie_weights()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(policy.state_dict().items())
    torch.cuda.synchronize(torch.device(vllm_device))
    policy.train()


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def format_raw_qa(
    rows: list[dict[str, Any]],
    prompt_path: str,
) -> list[dict[str, str]]:
    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt_template = file.read()
    return [
        {
            "prompt": prompt_template.format(question=row["question"]),
            "answer": row["answer"],
        }
        for row in rows
    ]


def load_generated_qa(file_path: str) -> list[dict[str, str]]:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return [
        {"prompt": item["prompt"], "answer": item["answer"]}
        for item in data["results"]
    ]


def load_dataset_and_format_qa(args: argparse.Namespace):
    if args.train_data_format == "raw_gsm8k":
        train_data = format_raw_qa(load_jsonl(args.train_data_path), args.prompt_path)
    else:
        train_data = load_generated_qa(args.train_data_path)

    if args.train_samples > 0 and args.train_samples < len(train_data):
        train_data = random.sample(train_data, args.train_samples)

    test_data = format_raw_qa(load_jsonl(args.test_data_path), args.prompt_path)
    return train_data, test_data


def ground_truth_from_answer(answer: str) -> str:
    return extract_answer(answer) or answer


def rollout_token_counts(outputs) -> dict[str, float | int]:
    response_lengths = [
        len(completion.token_ids)
        for output in outputs
        for completion in output.outputs
    ]
    generated_response_tokens = sum(response_lengths)
    prompt_tokens = sum(
        len(output.prompt_token_ids or []) * len(output.outputs) for output in outputs
    )
    return {
        "generated_response_tokens": generated_response_tokens,
        "prompt_tokens": prompt_tokens,
        "total_rollout_tokens": prompt_tokens + generated_response_tokens,
        "average_response_length": (
            generated_response_tokens / len(response_lengths) if response_lengths else 0.0
        ),
    }


def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    answers: list[str],
    eval_sampling_params,
) -> tuple[dict[str, int], dict[str, float | int]]:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    overview = {
        "correct": 0,
        "format_wrong": 0,
        "answer_wrong": 0,
        "count": 0,
    }
    for output, answer in zip(outputs, answers):
        reward = reward_fn(output.outputs[0].text, ground_truth_from_answer(answer))
        overview["count"] += 1
        if reward["answer_reward"] == 1:
            overview["correct"] += 1
        elif reward["format_reward"] == 1:
            overview["answer_wrong"] += 1
        else:
            overview["format_wrong"] += 1
    return overview, rollout_token_counts(outputs)


def select_eval_data(
    test_data: list[dict[str, str]],
    sample_count: int,
) -> list[dict[str, str]]:
    return test_data if sample_count <= 0 else test_data[:sample_count]


def save_checkpoint(model, tokenizer, output_path: Path, model_step: int) -> None:
    checkpoint_path = output_path / "checkpoints" / f"step_{model_step:06d}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)


def run_evaluation(
    *,
    model,
    tokenizer,
    vllm,
    vllm_device: str,
    reward_fn,
    test_data: list[dict[str, str]],
    sample_count: int,
    eval_sampling_params,
    experiment_logger: ExperimentLogger,
    model_step: int,
    output_path: Path,
    save: bool,
) -> None:
    eval_data = select_eval_data(test_data, sample_count)
    load_policy_into_vllm_instance(model, vllm, vllm_device)
    overview, token_counts = evaluate_vllm(
        vllm,
        reward_fn,
        [item["prompt"] for item in eval_data],
        [item["answer"] for item in eval_data],
        eval_sampling_params,
    )
    accuracy = overview["correct"] / max(overview["count"], 1)
    experiment_logger.log(
        {
            "event": "evaluation",
            "model_step": model_step,
            "eval/accuracy": accuracy,
            "eval/pass_at_1": accuracy,
            "eval/correct": overview["correct"],
            "eval/wrong_answer": overview["answer_wrong"],
            "eval/wrong_format": overview["format_wrong"],
            "eval/count": overview["count"],
            **{f"eval/{key}": value for key, value in token_counts.items()},
        }
    )
    logger.info("eval model_step=%d accuracy=%.4f", model_step, accuracy)
    if save:
        save_checkpoint(model, tokenizer, output_path, model_step)


def train_grpo_experiment(
    model,
    tokenizer,
    vllm,
    args: argparse.Namespace,
    reward_fn,
    experiment_logger: ExperimentLogger,
) -> None:
    from vllm import SamplingParams

    if args.sampling_strategy != "random":
        raise ValueError("Only random sampling is implemented in the vanilla baseline")
    if args.rollout_batch_size % args.group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size")
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError(
            "train_batch_size must be divisible by gradient_accumulation_steps"
        )
    if args.rollout_batch_size % args.train_batch_size != 0:
        raise ValueError("rollout_batch_size must be divisible by train_batch_size")

    micro_train_batch_size = (
        args.train_batch_size // args.gradient_accumulation_steps
    )
    prompts_per_rollout_batch = args.rollout_batch_size // args.group_size
    optimizer_steps_per_rollout = (
        args.epochs_per_rollout_batch
        * args.rollout_batch_size
        // args.train_batch_size
    )
    total_optimizer_steps = args.n_grpo_steps * optimizer_steps_per_rollout

    train_data, test_data = load_dataset_and_format_qa(args)
    if len(train_data) < prompts_per_rollout_batch:
        raise ValueError(
            f"need at least {prompts_per_rollout_batch} train prompts, got {len(train_data)}"
        )

    stop_strings = ["</answer>"] if args.reward_mode == "r1_zero" else None
    rollout_kwargs = {
        "temperature": args.sampling_temperature,
        "top_p": 1.0,
        "max_tokens": args.sampling_max_tokens,
        "min_tokens": args.sampling_min_tokens,
        "n": args.group_size,
        "seed": args.seed,
    }
    eval_kwargs = {
        "temperature": args.eval_temperature,
        "top_p": 1.0,
        "max_tokens": args.eval_max_tokens,
        "min_tokens": args.sampling_min_tokens,
        "n": 1,
        "seed": args.seed,
    }
    if stop_strings is not None:
        rollout_kwargs.update(
            {"stop": stop_strings, "include_stop_str_in_output": True}
        )
        eval_kwargs.update(
            {"stop": stop_strings, "include_stop_str_in_output": True}
        )
    rollout_sampling_params = SamplingParams(**rollout_kwargs)
    eval_sampling_params = SamplingParams(**eval_kwargs)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    model.zero_grad(set_to_none=True)

    if args.eval_before_training:
        run_evaluation(
            model=model,
            tokenizer=tokenizer,
            vllm=vllm,
            vllm_device=args.vllm_device,
            reward_fn=reward_fn,
            test_data=test_data,
            sample_count=args.eval_samples,
            eval_sampling_params=eval_sampling_params,
            experiment_logger=experiment_logger,
            model_step=0,
            output_path=Path(args.output_path),
            save=False,
        )

    train_step = 0
    cumulative_response_tokens = 0
    cumulative_rollout_tokens = 0
    smooth_train_loss = 0.0
    training_start = time.time()
    last_evaluated_model_step = 0 if args.eval_before_training else -1
    completed_grpo_steps = 0

    for grpo_step in range(args.n_grpo_steps):
        if (
            args.max_rollout_tokens > 0
            and cumulative_rollout_tokens >= args.max_rollout_tokens
        ):
            logger.info("reached total rollout-token budget")
            break

        rollout_dataset = random.sample(train_data, prompts_per_rollout_batch)
        rollout_prompts = [item["prompt"] for item in rollout_dataset]
        rollout_answers = [item["answer"] for item in rollout_dataset]

        load_policy_into_vllm_instance(model, vllm, args.vllm_device)
        outputs = vllm.generate(rollout_prompts, rollout_sampling_params)
        token_counts = rollout_token_counts(outputs)
        cumulative_response_tokens += int(token_counts["generated_response_tokens"])
        cumulative_rollout_tokens += int(token_counts["total_rollout_tokens"])

        repeated_ground_truths = []
        responses = []
        prompts = []
        for output, prompt, answer in zip(
            outputs, rollout_prompts, rollout_answers
        ):
            ground_truth = ground_truth_from_answer(answer)
            if len(output.outputs) != args.group_size:
                raise RuntimeError(
                    f"vLLM returned {len(output.outputs)} samples; expected {args.group_size}"
                )
            for completion in output.outputs:
                repeated_ground_truths.append(ground_truth)
                prompts.append(prompt)
                responses.append(completion.text)

        advantages, raw_rewards, rollout_metadata = (
            compute_group_normalized_rewards(
                reward_fn,
                responses,
                repeated_ground_truths,
                args.group_size,
                args.advantage_eps,
                args.use_std_normalization,
                args.zero_variance_epsilon,
            )
        )
        advantages = advantages.unsqueeze(1)
        raw_rewards = raw_rewards.unsqueeze(1)

        experiment_logger.log(
            {
                "event": "rollout",
                "step": train_step,
                "grpo_step": grpo_step + 1,
                **{
                    f"rollout/{key}": value
                    for key, value in rollout_metadata.items()
                },
                **{f"rollout/{key}": value for key, value in token_counts.items()},
                "rollout/cumulative_generated_response_tokens": cumulative_response_tokens,
                "rollout/cumulative_rollout_tokens": cumulative_rollout_tokens,
            }
        )
        logger.info(
            "rollout %d zero_var=%.3f all_correct=%.3f all_wrong=%.3f mixed=%.3f response_tokens=%d",
            grpo_step + 1,
            to_float(rollout_metadata["zero_variance_group_ratio"]),
            to_float(rollout_metadata["all_correct_group_ratio"]),
            to_float(rollout_metadata["all_wrong_group_ratio"]),
            to_float(rollout_metadata["mixed_group_ratio"]),
            token_counts["generated_response_tokens"],
        )

        tokenized = tokenize_prompt_and_output(
            prompt_strs=prompts,
            output_strs=responses,
            tokenizer=tokenizer,
        )
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        response_mask = tokenized["response_mask"]

        old_log_probs = None
        if args.loss_type == "grpo_clip":
            old_log_prob_batches = []
            model.eval()
            with torch.inference_mode():
                for start in range(0, args.rollout_batch_size, micro_train_batch_size):
                    end = start + micro_train_batch_size
                    batch_input_ids = input_ids[start:end].to(args.train_device)
                    batch_labels = labels[start:end].to(args.train_device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = get_response_log_probs(
                            model=model,
                            input_ids=batch_input_ids,
                            labels=batch_labels,
                            return_token_entropy=False,
                        )
                    old_log_prob_batches.append(output["log_probs"].detach().cpu())
            old_log_probs = torch.cat(old_log_prob_batches, dim=0)

        model.train()
        for _epoch in range(args.epochs_per_rollout_batch):
            permutation = torch.randperm(args.rollout_batch_size)
            input_ids_epoch = input_ids[permutation]
            labels_epoch = labels[permutation]
            response_mask_epoch = response_mask[permutation]
            raw_rewards_epoch = raw_rewards[permutation]
            advantages_epoch = advantages[permutation]
            old_log_probs_epoch = (
                old_log_probs[permutation] if old_log_probs is not None else None
            )

            window_loss = 0.0
            window_entropy_sum = 0.0
            window_token_count = 0
            window_clipped_count = 0
            step_start = time.time()

            for batch_index, start in enumerate(
                range(0, args.rollout_batch_size, micro_train_batch_size)
            ):
                end = start + micro_train_batch_size
                batch_input_ids = input_ids_epoch[start:end].to(args.train_device)
                batch_labels = labels_epoch[start:end].to(args.train_device)
                batch_response_mask = response_mask_epoch[start:end].to(args.train_device)
                batch_raw_rewards = raw_rewards_epoch[start:end].to(args.train_device)
                batch_advantages = advantages_epoch[start:end].to(args.train_device)
                batch_old_log_probs = (
                    old_log_probs_epoch[start:end].to(args.train_device)
                    if old_log_probs_epoch is not None
                    else None
                )

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    log_prob_output = get_response_log_probs(
                        model=model,
                        input_ids=batch_input_ids,
                        labels=batch_labels,
                        return_token_entropy=True,
                    )
                    loss, loss_metadata = grpo_microbatch_train_step(
                        policy_log_probs=log_prob_output["log_probs"],
                        response_mask=batch_response_mask,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        loss_type=args.loss_type,
                        raw_rewards=(
                            batch_raw_rewards
                            if args.loss_type == "no_baseline"
                            else None
                        ),
                        advantages=(
                            batch_advantages
                            if args.loss_type != "no_baseline"
                            else None
                        ),
                        old_log_probs=batch_old_log_probs,
                        cliprange=(
                            args.cliprange
                            if args.loss_type == "grpo_clip"
                            else None
                        ),
                    )

                mask_token_count = int(batch_response_mask.sum().item())
                window_loss += loss.detach().float().item()
                window_entropy_sum += (
                    log_prob_output["token_entropy"][batch_response_mask]
                    .detach()
                    .float()
                    .sum()
                    .item()
                )
                window_token_count += mask_token_count
                if args.loss_type == "grpo_clip":
                    window_clipped_count += int(
                        loss_metadata["clipped"][batch_response_mask].sum().item()
                    )

                if (batch_index + 1) % args.gradient_accumulation_steps != 0:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                ).item()
                optimizer.step()
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize(torch.device(args.train_device))

                step_time = time.time() - step_start
                smooth_train_loss = (
                    args.log_ema_beta * smooth_train_loss
                    + (1 - args.log_ema_beta) * window_loss
                )
                debiased_loss = smooth_train_loss / (
                    1 - args.log_ema_beta ** (train_step + 1)
                )
                metrics = {
                    "event": "optimizer_step",
                    "step": train_step,
                    "grpo_step": grpo_step + 1,
                    "train/loss": debiased_loss,
                    "train/grad_norm": grad_norm,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/policy_entropy": window_entropy_sum
                    / max(window_token_count, 1),
                    "train/response_tokens_in_step": window_token_count,
                    "train/step_time": step_time,
                    "train/elapsed_time": time.time() - training_start,
                    "rollout/cumulative_generated_response_tokens": cumulative_response_tokens,
                    "rollout/cumulative_rollout_tokens": cumulative_rollout_tokens,
                }
                if args.loss_type == "grpo_clip":
                    metrics["train/clip_fraction"] = window_clipped_count / max(
                        window_token_count, 1
                    )
                if train_step % args.log_steps == 0:
                    experiment_logger.log(metrics)
                logger.info(
                    "step %d/%d loss=%.6f grad_norm=%.4f entropy=%.4f dt=%.2fs",
                    train_step,
                    total_optimizer_steps,
                    debiased_loss,
                    grad_norm,
                    metrics["train/policy_entropy"],
                    step_time,
                )
                train_step += 1
                window_loss = 0.0
                window_entropy_sum = 0.0
                window_token_count = 0
                window_clipped_count = 0
                step_start = time.time()

        completed_grpo_steps = grpo_step + 1
        if completed_grpo_steps % args.eval_steps == 0:
            run_evaluation(
                model=model,
                tokenizer=tokenizer,
                vllm=vllm,
                vllm_device=args.vllm_device,
                reward_fn=reward_fn,
                test_data=test_data,
                sample_count=args.eval_samples,
                eval_sampling_params=eval_sampling_params,
                experiment_logger=experiment_logger,
                model_step=completed_grpo_steps,
                output_path=Path(args.output_path),
                save=(
                    args.checkpoint_steps > 0
                    and completed_grpo_steps % args.checkpoint_steps == 0
                ),
            )
            last_evaluated_model_step = completed_grpo_steps

    if (
        completed_grpo_steps != last_evaluated_model_step
        or args.final_eval_samples != args.eval_samples
    ):
        run_evaluation(
            model=model,
            tokenizer=tokenizer,
            vllm=vllm,
            vllm_device=args.vllm_device,
            reward_fn=reward_fn,
            test_data=test_data,
            sample_count=args.final_eval_samples,
            eval_sampling_params=eval_sampling_params,
            experiment_logger=experiment_logger,
            model_step=completed_grpo_steps,
            output_path=Path(args.output_path),
            save=True,
        )

    logger.info(
        "peak allocated CUDA memory: %.2f GiB",
        torch.cuda.max_memory_allocated(torch.device(args.train_device)) / 2**30,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Hugging Face model ID or local checkpoint path",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-data-path", default="data/gsm8k/train.jsonl")
    parser.add_argument("--test-data-path", default="data/gsm8k/test.jsonl")
    parser.add_argument(
        "--train-data-format",
        choices=["raw_gsm8k", "generated"],
        default="raw_gsm8k",
    )
    parser.add_argument(
        "--prompt-path", default="cs336_alignment/prompts/question_only.prompt"
    )
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--final-eval-samples", type=int, default=0)

    parser.add_argument(
        "--reward-mode", choices=["question_only", "r1_zero"], default="question_only"
    )
    parser.add_argument("--format-reward-weight", type=float, default=0.1)
    parser.add_argument("--answer-reward-weight", type=float, default=1.0)
    parser.add_argument("--zero-variance-epsilon", type=float, default=1e-8)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument(
        "--use-std-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--sampling-strategy", choices=["random"], default="random"
    )
    parser.add_argument("--n-grpo-steps", type=int, default=200)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-min-tokens", type=int, default=4)
    parser.add_argument("--sampling-max-tokens", type=int, default=1024)
    parser.add_argument("--max-rollout-tokens", type=int, default=0)

    parser.add_argument(
        "--loss-type",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        default="reinforce_with_baseline",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=256)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--eval-steps", type=int, default=16)
    parser.add_argument("--checkpoint-steps", type=int, default=50)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--eval-before-training",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-steps", type=int, default=1)
    parser.add_argument("--log-ema-beta", type=float, default=0.9)

    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--vllm-device", default="cuda:0")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument(
        "--attn-implementation",
        choices=["flash_attention_2", "sdpa", "eager"],
        default="sdpa",
    )

    parser.add_argument(
        "--wandb-mode", choices=["disabled", "offline", "online"], default="disabled"
    )
    parser.add_argument("--wandb-project", default="efficient-rlvr")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the vLLM GRPO training entrypoint")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    args.run_name = args.run_name or (
        f"vanilla_grpo_{args.sampling_strategy}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    config = vars(args).copy()
    config.update(
        {
            "git_commit": get_git_commit(),
            "git_is_dirty": git_is_dirty(),
            "command": " ".join(sys.argv),
            "cuda_device_name": torch.cuda.get_device_name(
                torch.device(args.train_device)
            ),
        }
    )
    with (output_path / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    experiment_logger = ExperimentLogger(
        output_path=output_path,
        config=config,
        run_name=args.run_name,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
    )

    component_reward_fn = {
        "question_only": question_only_reward_fn,
        "r1_zero": r1_zero_reward_fn,
    }[args.reward_mode]
    reward_fn = make_weighted_reward_fn(
        component_reward_fn,
        format_weight=args.format_reward_weight,
        answer_weight=args.answer_reward_weight,
    )

    logger.info("initializing vLLM from %s", args.model_name_or_path)
    vllm = init_vllm(
        model_id=args.model_name_or_path,
        device=args.vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    logger.info("loading trainable policy on %s", args.train_device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(args.train_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    try:
        train_grpo_experiment(
            model=model,
            tokenizer=tokenizer,
            vllm=vllm,
            args=args,
            reward_fn=reward_fn,
            experiment_logger=experiment_logger,
        )
    finally:
        experiment_logger.finish()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
