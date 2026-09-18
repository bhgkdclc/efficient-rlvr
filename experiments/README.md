# Experiment outputs

Each run writes to its own directory under `experiments/` and contains:

- `config.json`: complete CLI config, command, Git commit, seed, and GPU name.
- `metrics.jsonl`: rollout, optimizer, and evaluation records.
- `checkpoints/step_*/`: Hugging Face model/tokenizer checkpoints.

Generated run directories are ignored by Git. The first-stage scripts are:

- `scripts/run_vanilla_smoke.sh`
- `scripts/run_vanilla_baseline.sh`
- `scripts/run_dynamic_smoke.sh`
- `scripts/run_dynamic_baseline.sh`
- `scripts/run_difficulty_smoke.sh`
- `scripts/run_difficulty_baseline.sh`
- `scripts/run_difficulty_fast_ema.sh`

Dynamic sampling discards zero-variance groups after generation and resamples
until the optimizer batch contains the requested number of effective groups.
Its logs include the token cost of every discarded group. Difficulty-aware
sampling uses an EMA of per-prompt answer accuracy, prioritizes prompts using
`4p(1-p)`, and retains configurable uniform exploration. It does not apply
post-rollout filtering, so it is directly comparable with the random sampler.

`run_difficulty_fast_ema.sh` is the focused follow-up to the first
Difficulty-Aware run. It changes only `difficulty_ema_beta` from 0.9 to 0.5
to test whether stale prompt histories caused the measured calibration lag.

After a run, generate standardized summaries, CSV files, and plots with
`scripts/analyze_grpo_run.py`. Reviewable outputs belong under `results/`;
large checkpoints and raw logs remain under this ignored directory.
