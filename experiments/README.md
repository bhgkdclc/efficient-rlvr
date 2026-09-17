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

Dynamic sampling discards zero-variance groups after generation and resamples
until the optimizer batch contains the requested number of effective groups.
Its logs include the token cost of every discarded group. Difficulty-aware
sampling remains out of scope until the dynamic baseline curve is collected.

After a run, generate standardized summaries, CSV files, and plots with
`scripts/analyze_grpo_run.py`. Reviewable outputs belong under `results/`;
large checkpoints and raw logs remain under this ignored directory.
