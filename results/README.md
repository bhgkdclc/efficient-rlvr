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

For Difficulty-Aware runs, also pass
`--sampler-state experiments/<run>/sampler_state.json`.

The analysis scripts require the optional plotting dependencies. Install them
without changing the training environment with:

```bash
uv pip install --python .venv/bin/python -r requirements-analysis.txt
```

Generate a two-run fixed-budget comparison with:

```bash
python scripts/compare_grpo_runs.py \
  --vanilla-dir results/<vanilla-run> \
  --dynamic-dir results/<dynamic-run> \
  --difficulty-dir results/<difficulty-run> \
  --difficulty-beta05-dir results/<difficulty-beta05-run> \
  --difficulty-coverage-dir results/<difficulty-coverage-run> \
  --difficulty-frontloaded-dir results/<difficulty-frontloaded-run> \
  --output-dir results/<comparison>
```

Generate a paired multi-seed comparison with:

```bash
python scripts/compare_multiseed.py \
  --vanilla-dirs results/<vanilla-seed-42> results/<vanilla-seed-43> results/<vanilla-seed-44> \
  --difficulty-dirs results/<difficulty-seed-42> results/<difficulty-seed-43> results/<difficulty-seed-44> \
  --output-dir results/<multiseed-comparison>
```

Rebuild the corrected 1024-token endpoint audit from extracted evaluation
runs with:

```bash
python scripts/analyze_eval_length_audit.py \
  --eval-root experiments/final_evaluations \
  --old-multiseed-dir results/multiseed_vanilla_vs_fast_ema_20260918 \
  --hybrid-train-dir experiments/hybrid_u50_seed42_<timestamp> \
  --output-dir results/eval_length_audit_20260918
```

Current reports:

- `vanilla_random_20260917_220541/report.md`
- `dynamic_20260917_230635/report.md`
- `vanilla_vs_dynamic_20260917/report.md`
- `difficulty_20260918_090256/report.md`
- `three_way_20260918/report.md`
- `difficulty_beta05_20260918_094940/report.md`
- `four_way_20260918/report.md`
- `multiseed_vanilla_vs_fast_ema_20260918/report.md`
- `difficulty_coverage_w03_20260918_135914/report.md`
- `coverage_ablation_seed42_20260918/report.md`
- `difficulty_frontloaded_coverage_seed42_20260918_144542/report.md`
- `coverage_schedule_ablation_seed42_20260918/report.md`
- `eval_length_audit_20260918/report.md`
