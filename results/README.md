# Results

Small, reviewable experiment summaries and plots live here. Raw checkpoints,
console logs, and per-step JSONL files remain under `experiments/` and are not
committed.

Generate a run report with:

```bash
python scripts/analyze_grpo_run.py \
  experiments/<run>/metrics.jsonl \
  --config experiments/<run>/config.json \
  --output-dir results/<run>
```

The analysis script requires `pandas` and `matplotlib`; keep these optional
plotting packages out of the training environment if minimizing setup time.

Generate a two-run fixed-budget comparison with:

```bash
python scripts/compare_grpo_runs.py \
  --vanilla-dir results/<vanilla-run> \
  --dynamic-dir results/<dynamic-run> \
  --output-dir results/<comparison>
```

Current reports:

- `vanilla_random_20260917_220541/report.md`
- `dynamic_20260917_230635/report.md`
- `vanilla_vs_dynamic_20260917/report.md`
