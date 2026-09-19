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
- `scripts/run_difficulty_dynamic_smoke.sh`
- `scripts/run_difficulty_dynamic_baseline.sh`
- `scripts/run_difficulty_dynamic_cadence_matched.sh`
- `scripts/run_difficulty_coverage_smoke.sh`
- `scripts/run_difficulty_coverage_baseline.sh`
- `scripts/run_difficulty_frontloaded_coverage.sh`
- `scripts/run_replication_seeds.sh`
- `scripts/run_base_full_eval.sh`
- `scripts/run_hybrid_smoke.sh`
- `scripts/run_hybrid_baseline.sh`

Dynamic sampling discards zero-variance groups after generation and resamples
until the optimizer batch contains the requested number of effective groups.
Its logs include the token cost of every discarded group. Difficulty-aware
sampling uses an EMA of per-prompt answer accuracy, prioritizes prompts using
`4p(1-p)`, and retains configurable uniform exploration. It does not apply
post-rollout filtering, so it is directly comparable with the random sampler.

Difficulty-Dynamic sampling composes these mechanisms without changing the
loss: Fast-EMA selects prompts before generation, and zero-variance groups are
then rejected until a complete effective optimizer batch is available. Prompt
indices are not reused within one optimizer batch, and every attempted group
still counts toward the rollout-token budget. Its seed-42 screening criteria
are registered before running: versus Dynamic, reduce the discarded-token
ratio by at least 15% relatively and improve optimized effective groups per
million rollout tokens by at least 10%; final Pass@1 should be no more than one
point below the paired seed-42 Fast-EMA run. Only a passing efficiency screen
is replicated on seeds 43 and 44.

The replications confirmed the efficiency result but failed the accuracy
guardrail on seeds 43 and 44. Across three seeds, Difficulty-Dynamic packed
roughly the same number of effective groups as Fast-EMA into about 89 rather
than 141 optimizer updates; 32.3 of the mean 35-example endpoint gap came from
format failures. `run_difficulty_dynamic_cadence_matched.sh` is the registered
single-variable follow-up: it targets five accepted prompt groups (40
responses) per optimizer batch, matching Fast-EMA's observed effective-group
update cadence while leaving the sampler, reward, group size, and 4M-token
budget unchanged. Screen seed 44 first. It passes if full-test Pass@1 is at
least 82.85% (within one point of Fast-EMA seed 44), wrong-format outputs are
at most 50, and optimized effective groups per million rollout tokens remain
at least 170. Only a passing seed-44 run is replicated on seed 43.

The seed-44 control restored optimizer steps from 87 to 139 and preserved
173.12 optimized effective groups per million tokens. Pass@1 improved from
78.39% to 80.36%, while wrong-format outputs fell from 101 to 72. It therefore
passed only the efficiency criterion, not the 82.85% accuracy or 50-format-
failure guardrails. This branch is closed without a seed-43 replication;
Fast-EMA remains the selected sampler.

`run_difficulty_fast_ema.sh` is the focused follow-up to the first
Difficulty-Aware run. It changes only `difficulty_ema_beta` from 0.9 to 0.5
to test whether stale prompt histories caused the measured calibration lag.
It disables intermediate checkpoints by default while retaining the final
checkpoint. Set `SEED` to run replications without editing the script.

`run_replication_seeds.sh` sequentially runs paired Vanilla and Fast-EMA
replications. With no arguments it runs seeds 43 and 44, writes a separate
console log for each run, and keeps only each run's final checkpoint.

Coverage-aware Difficulty Sampling adds a normalized inverse-square-root
sample-count score to the boundary score. A zero coverage weight exactly
recovers the previous sampler. The default coverage experiment uses weight
0.3 with beta 0.5 to test the measured efficiency-versus-diversity tradeoff.

`run_difficulty_frontloaded_coverage.sh` is the controlled follow-up to that
fixed-weight ablation. It samples 512 unique groups during warm-up, then sets
the coverage mixture to zero and uses the unchanged beta=0.5 boundary sampler.
This tests an early-coverage/late-focus schedule without changing reward,
loss, rollout budget, or evaluation settings.

Hybrid sampling is the targeted follow-up to the observed efficiency-versus-
accuracy tradeoff. Each optimizer batch contains an explicit uniform stratum
and a disjoint difficulty-aware stratum (50/50 by default). The uniform half
anchors training to the original data distribution; the difficulty half seeks
more non-zero-variance groups. Metrics separately report each stratum's group
accuracy, effective-group ratio, and rollout-token cost. The loss, reward,
group size, and total rollout-token budget remain unchanged.

This is a pre-declared screening experiment, not a hyperparameter sweep. For
seed 42 it passes only if (1) effective groups per million rollout tokens are
at least 10% above the three-seed Vanilla mean, (2) full-test Pass@1 is no more
than one percentage point below the paired Vanilla result, and (3) prompt
coverage exceeds Fast-EMA Difficulty Sampling. Only a passing seed-42 run is
replicated on seeds 43 and 44. A failure ends this sampling branch and is
reported as evidence that maximizing group variance alone is not sufficient.
The observed Hybrid run failed the screen: it reached 153.92 effective groups
per million tokens and 82.87% full-test Pass@1 at the corrected 1024-token
limit, so this branch was not replicated.

`run_base_full_eval.sh` evaluates the untouched checkpoint on the complete
held-out split with exactly the same deterministic decoding settings as final
training evaluation. It closes the earlier protocol gap where the initial
checkpoint had only been evaluated on the first 256 examples.

Baseline scripts now use a 1024-token final-evaluation limit and log vLLM
`finish_reason == "length"` counts. Historical intermediate curves use the
original 512-token limit; final-checkpoint corrections and the associated
evaluation audit live under `results/eval_length_audit_20260918/`.

After a run, generate standardized summaries, CSV files, and plots with
`scripts/analyze_grpo_run.py`. Reviewable outputs belong under `results/`;
large checkpoints and raw logs remain under this ignored directory.
