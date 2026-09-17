# Experiment outputs

Each run writes to its own directory under `experiments/` and contains:

- `config.json`: complete CLI config, command, Git commit, seed, and GPU name.
- `metrics.jsonl`: rollout, optimizer, and evaluation records.
- `checkpoints/step_*/`: Hugging Face model/tokenizer checkpoints.

Generated run directories are ignored by Git. The first-stage scripts are:

- `scripts/run_vanilla_smoke.sh`
- `scripts/run_vanilla_baseline.sh`

Both use random prompt sampling. Dynamic and difficulty-aware sampling are not
implemented until the vanilla curve has been collected.
