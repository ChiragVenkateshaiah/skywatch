# SkyWatch — Model 1 Transformer variant

## 1. Why

The MLOps arc (`docs/MLOPS_PLAN.md`, all 8 tracks) is done and published (video, LinkedIn post,
`docs/SKYWATCH_MLOPS_ARTICLE.md`). `docs/ML_ROADMAP.md` §14 left one Model 1 decision open:
*"also build the sequence-model (Transformer) variant to show that path, or leave it
documented?"* Decided (2026-09-24): build it, staying entirely on Free Edition.

The goal isn't "beat LightGBM." It's giving the Transformer a **real, fair evaluation through
the exact promotion-gate infrastructure the MLOps arc just built** — same registered model, same
gate, same audit log. Whether it wins or holds is a legitimate, documented outcome either way,
consistent with the project's "verify live, don't design on paper" pattern from the MLOps arc.

Checked live before starting:
- **Model 3 label volume** (the roadmap's other open item) is still nowhere near learnable
  scale: emergency 8 segments/8 aircraft, holding 160 segments/15 aircraft (was 159 two weeks
  earlier — effectively flat), go-around 0. Rule-based stays correct; not this arc.
- **`gold_arrival_tracks` volume**: 479,624 rows, 25,084 arrival segments, average 19.1
  reports/segment (min 1, max 40) — enough for a small sequence model, small enough to train on
  CPU in a serverless notebook.

## 2. The two problems that shaped the design

`score_eta.py` and `promote_eta.py` both call `mlflow.lightgbm.load_model(...)` directly, not
generic `mlflow.pyfunc.load_model` — deliberately, per `score_eta.py`'s own comment: LightGBM
needs `phase` as a pandas `Categorical`, which pyfunc's schema enforcement (declared `string`)
rejects. A blanket switch to pyfunc would have broken the *existing, working* champion.

The gate and scoring code also call `model.predict(X)` with one flat row of `ETA_FEATURES` per
prediction — no trajectory. A sequence model needs the last N reports, not one row, and that has
to be solved **upstream** (a wider flat DataFrame, one row per query point, columns = flattened
history) rather than by changing the predict() calling convention itself.

Solved together with one small pattern: a `model_flavor` version tag (`"lightgbm"` default /
`"pytorch_transformer"`), read by `promote_eta.py`, branching (a) the loader and (b) which
feature-frame builder to use. The existing LightGBM path is untouched. The gate math
(`evaluate_gate`, `by_band`, `audit_row`) needed zero changes — it only ever sees predictions +
actuals + distance.

## 3. What shipped

| File | Role |
|---|---|
| `src/lib/sequences.py` | Pure numpy: `causal_windows` (last `SEQ_LEN=24` reports ending at each row, left-padded, never future rows), `flatten_windows`/`unflatten_windows` (round-trip to/from a flat DataFrame row), `phase_to_index`. Unit tested, `tests/test_sequences.py` (9 cases: shape, padding alignment, truncation, no-future-leakage, flatten round-trip, phase mapping). |
| `src/eta_sequences.py` | `%run` shared module (mirrors `eta_features.py`'s one-definition pattern) — Spark/pandas glue only: groups `gold_arrival_tracks`-shaped input by `seg_id` ordered by `snapshot_ts`, calls into `lib/sequences.py`, returns the wide per-row frame. |
| `src/train_eta_transformer.py` | New notebook. Same confirmed-touchdown filter, same time-based train/val/test split, same distance-band MAE report as `train_eta.py`. Model: 2-layer Transformer encoder (4 heads, d_model 64) over the window, `phase` via `nn.Embedding` instead of LightGBM's native categorical split, regression head on the window's last (current-timestep) position. Two-stage fit mirrors `train_eta.py`'s LightGBM calibration: stage 1 early-stops on the validation day, stage 2 refits on train+val for `best_epoch * 1.15` epochs (same adjustment factor), test day touched once. CPU only. Logged as a custom `mlflow.pyfunc.PythonModel` (state dict via `torch.save`/`context.artifacts`, MLflow's documented pattern — not cloudpickled inline), tagged `model_flavor=pytorch_transformer`, registered to the **same** `skywatch.ml.eta_touchdown` model, set as `@challenger`. |
| `src/promote_eta.py` (modified) | Reads `model_flavor` off both `@challenger` and `@champion` (default `lightgbm` for versions that predate the tag); branches the loader and the eval-frame builder. `evaluate_gate`/`audit_row` unchanged. |
| `resources/skywatch.traintransformer.job.yml` | Serverless on-demand job, mirrors `skywatch.train.job.yml`. |

## 4. Scope boundary

This arc trains, registers, and gates the Transformer **for real** — real `@challenger`, real
gate verdict, real `promotion_audit` row. It does **not** extend live serving
(`score_eta.py`'s in-app click-to-predict, which scores currently-airborne aircraft) to support
a sequence model. Building live sequence construction from in-flight data is a separate,
materially sized piece of work, worth doing only if the Transformer actually wins the gate — if
it holds instead, that's a complete, honest result on its own (mirrors the M2 HOLD story from
the MLOps arc). Live-serving support is a fast-follow only if warranted, not built speculatively.

## 5. Next queued arc (documented, not started)

Per the same planning conversation: an **LLM ops-briefing** — a natural-language shift summary
using Databricks' Free Edition Foundation Model API (limited endpoints, no GPU, no provisioned
throughput per `docs/ML_ROADMAP.md` §13), with a documented fallback if quota/capacity gates it.
This is a new user-facing capability rather than more model depth, and it's next after this arc
closes.

## 6. Progress log

- **2026-09-24 — code complete, local tests pass.** `src/lib/sequences.py` +
  `tests/test_sequences.py` (9/9 pass locally). `src/eta_sequences.py`, `src/train_eta_transformer.py`,
  `promote_eta.py`'s `model_flavor` branch, and the new job resource all written per §3 above.
  Also ran a local synthetic smoke test (no Spark/Databricks) of the Transformer forward/backward
  pass and the flatten/unflatten window round-trip — shapes, masking, and one training step all
  correct before spending live compute. PR #39 open on `ml/m1-transformer-variant`, not merged.
  Live verification (train job run, promote gate run against the real workspace) next.

- **2026-09-24/25 — live verification attempt #1: failed, platform-side.** `bundle deploy -t dev`
  succeeded (new job `skywatch_train_eta_transformer` created). `bundle run skywatch_train_eta_transformer
  -t dev` executed for real — task ran ~100 minutes (cluster setup was 4s, not the bottleneck),
  then terminated `INTERNAL_ERROR`: *"The service encountered a transient issue. Retry the run,
  or contact Databricks support if the issue persists."* No notebook stdout was captured via the
  Jobs API, so it's unknown whether this hit near the end (e.g. during `mlflow.pyfunc.log_model`'s
  artifact upload) or earlier. Databricks run id `956016922336320` / task run id
  `1006766208315228`, job id `816953901790321`, if re-checking the UI directly.
  **Not yet resolved whether this was a genuine one-off platform blip or the training loop is
  just slow on serverless CPU** — 100 min for ~90 epochs total (two-stage fit) isn't absurd for a
  constrained shared CPU, but it's slower than expected. Worth watching on the next attempt.

- **2026-10-07 — live verification attempt #2: blocked, not a code issue.** Retried per plan
  (`databricks auth login`'s refresh token had gone stale after ~12 idle days — the same
  known Free Edition OAuth lifetime issue hit during the original MLOps arc). `bundle run` failed
  immediately with `invalid_grant: Refresh token is invalid` — never reached the job at all.
  **Needs a human to run `databricks auth login --profile skywatch` interactively** (browser
  login) before any further live step on this model can proceed — not something resolvable from
  a non-interactive session on any machine.

  **Development is moving to a second machine at this point.** Nothing is uncommitted or
  unpushed — `ml/m1-transformer-variant` (commit `8ebaf52`) is fully pushed to origin, PR #39 is
  open, and this doc is the checkpoint. **Exact next step on whichever machine continues:**
  1. `databricks auth login --profile skywatch` (interactive, one-time per machine — CLI auth
     profiles are local and are *not* carried by git; see this repo's README/`docs/RUNBOOKS.md`
     for profile setup if the profile itself doesn't exist yet on the new machine).
  2. `databricks bundle run skywatch_train_eta_transformer -t dev --profile skywatch` — watch
     whether it completes faster/cleanly this time; if it hits `INTERNAL_ERROR` again, that
     stops looking like a one-off and starts looking like a real performance problem worth
     profiling (candidate suspects: `src/eta_sequences.py`'s per-segment Python loop via
     `lib.sequences.causal_windows`, or just CPU-constrained `nn.TransformerEncoder` on
     serverless — neither has been profiled yet).
  3. On success: `databricks bundle run skywatch_promote_eta -t dev --profile skywatch` (dry
     run, default `apply=false`) — confirms the `model_flavor` branch loads both flavors
     correctly and produces a real PASS/HOLD verdict + `promotion_audit` row.
  4. Only then: decide whether to merge PR #39 (a HOLD verdict is a legitimate, complete result
     on its own per §4's scope boundary — it doesn't block merging, since the deliverable is the
     real evaluation, not a win).
