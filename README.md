# Efficient RLVR

**Diagnosing ineffective GRPO rollouts and testing difficulty-aware sampling
for mathematical reasoning.**

This project extends Stanford CS336 Assignment 5 with a fixed-cost study of a
common GRPO failure mode. For each prompt, GRPO samples a group of responses.
If every response receives the same reward, the group has zero reward variance
and contributes essentially no relative-advantage signal, despite consuming
rollout compute.

The experiments compare random sampling, post-generation Dynamic Sampling,
pre-generation Difficulty-Aware Sampling, and coverage-aware/hybrid ablations
using Qwen2.5-Math-1.5B on GSM8K. Every main run uses the same approximately
four-million rollout-token budget.

## Main result

Across paired seeds 42, 43, and 44, the best Difficulty-Aware sampler uses a
fast EMA of per-prompt rollout accuracy and samples according to
`4p(1-p)` with 10% uniform exploration.

| Metric | Vanilla GRPO | Fast-EMA Difficulty | Change |
|---|---:|---:|---:|
| Zero-variance groups | 52.68% +/- 0.65% | **36.88% +/- 2.51%** | **-15.81 pp** |
| Effective groups / 1M rollout tokens | 143.19 +/- 1.13 | **176.99 +/- 3.42** | **+23.60% +/- 1.81%** |
| Full GSM8K Pass@1, 1024-token eval | **83.93% +/- 0.50%** | 83.32% +/- 0.73% | -0.61 pp |

The paired accuracy difference has a wide 95% t interval of
`[-2.83, +1.61]` percentage points with only three seeds. The evidence supports
a robust signal-efficiency improvement and descriptive near-parity, not a
claim of statistical equivalence or higher final accuracy.

![Efficiency-quality tradeoff](results/eval_length_audit_20260918/efficiency_accuracy_tradeoff_1024.png)

## What the experiments found

1. **Vanilla GRPO increasingly wastes rollouts.** Zero-variance groups average
   52.68% across training and rise to 67.92% in the last 20 steps. The late
   failures are primarily all-correct groups.
2. **Post-generation filtering is expensive.** Dynamic Sampling guarantees an
   effective optimizer batch, but the seed-42 run spent 44.36% of rollout
   tokens on groups discarded after generation.
3. **Pre-generation difficulty prediction works.** Fast-EMA Difficulty
   Sampling improves effective groups per token in all three paired seeds.
4. **More advantage signal does not automatically imply better
   generalization.** Coverage and 50/50 Hybrid ablations increase prompt
   diversity but do not beat Fast-EMA or Vanilla at the endpoint.
5. **Generation length was an evaluation confound.** Raising the greedy
   evaluation limit from 512 to 1024 tokens improves Vanilla by 3.21 points
   and Fast-EMA by 4.02 points on average. Mean unparseable outputs fall from
   92.7 to 24.0 for Vanilla and from 109.3 to 28.3 for Fast-EMA.

![Evaluation-length audit](results/eval_length_audit_20260918/evaluation_length_audit.png)

The length audit was discovered after the Hybrid run lost accuracy mostly
through parser/format failures rather than explicit wrong answers. New runs log
the number and ratio of responses whose vLLM finish reason is `length`.

## Experimental setup

- Model: Qwen2.5-Math-1.5B, trained directly with RLVR (no project-specific
  SFT stage).
- Data: GSM8K train/test split.
- Group size: 8 responses per prompt; rollout batch: 64 responses.
- Reward: `1.0 * answer_reward + 0.1 * format_reward`, where a parseable
  `\boxed{...}` answer earns the format component.
- Budget: approximately 4M prompt-plus-response rollout tokens per run.
- Evaluation: full 1,319-example test set, greedy decoding, 1024-token limit.
- Hardware: one NVIDIA A800 80GB PCIe GPU.

Rollout cost includes tokens from rejected Dynamic Sampling groups. Evaluation
tokens are reported separately and are not counted in the training budget.

## Reproduce

The cloud scripts default to a local model cache under `PERSIST_ROOT`:

```bash
source .venv/bin/activate
export PERSIST_ROOT=/root/autodl-tmp/efficient-rlvr-cache

bash scripts/run_vanilla_smoke.sh
bash scripts/run_vanilla_baseline.sh
bash scripts/run_difficulty_fast_ema.sh
```

Run the paired replications with:

```bash
bash scripts/run_replication_seeds.sh 43 44
```

Install plotting dependencies and analyze one run with:

```bash
uv pip install --python .venv/bin/python -r requirements-analysis.txt

python scripts/analyze_grpo_run.py \
  experiments/<run>/metrics.jsonl \
  --config experiments/<run>/config.json \
  --sampler-state experiments/<run>/sampler_state.json \
  --output-dir results/<run>
```

The consolidated reports are:

- [Three-seed sampling analysis](results/multiseed_vanilla_vs_fast_ema_20260918/report.md)
- [Evaluation-length audit and corrected endpoints](results/eval_length_audit_20260918/report.md)
- [Coverage-schedule ablations](results/coverage_schedule_ablation_seed42_20260918/report.md)

## Repository layout

- `scripts/train_grpo.py`: existing GRPO path plus sampling flags and logging.
- `cs336_alignment/sampling.py`: Dynamic and Difficulty-Aware samplers.
- `scripts/analyze_grpo_run.py`: per-run summaries and plots.
- `scripts/compare_multiseed.py`: paired multi-seed aggregation.
- `scripts/analyze_eval_length_audit.py`: corrected endpoint and truncation audit.
- `experiments/README.md`: experiment protocol and pre-declared screening rules.
- `results/`: compact CSV/JSON summaries, plots, and reports.

## Scope and limitations

The current evidence covers one model, one dataset, three paired seeds, and a
fixed group size of eight. Only final checkpoints were re-evaluated with the
corrected 1024-token protocol, so historical intermediate tokens-to-target
curves still use 512-token evaluation. The project therefore reports improved
rollout-signal efficiency separately from downstream accuracy.

This repository is derived from Stanford CS336 Spring 2025 Assignment 5 and
retains the original course implementation as its training foundation.
