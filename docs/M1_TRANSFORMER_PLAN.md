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

- **2026-10-11/12 — live verification attempt #3: timed out, root cause found (our code).**
  Re-authed on the new machine, ran `bundle run skywatch_train_eta_transformer -t dev` (run id
  `165882139007389`). Hit the job's 2h `timeout_seconds` → `TIMEDOUT`. The only stdout line was
  the opening `train set: ...` print — the next print, right after `build_sequence_frame()`, never
  appeared, so **training never started**: the whole 2h went into building the input frame. Cause:
  `src/eta_sequences.py` assigned the 456 `seq_*` columns onto a pandas chunk per segment
  (`chunk[seq_cols] = flat`, 25,084 segments → ~11M single-column inserts; 52 `PerformanceWarning:
  DataFrame is highly fragmented` lines in the log). Almost certainly attempt #1's ~100 min failure
  too — it was never "slow Transformer on serverless CPU". (Jobs API reports attempt #1's run as
  `CANCELED`, not `INTERNAL_ERROR` as logged above; doesn't change the diagnosis.)

  **Fix:** windows built per segment in numpy over contiguous `seg_id` slices, wide frame assembled
  once. Local benchmark on synthetic data at real volume (25,084 segments, ~20 reports each): old
  ≈157 min (extrapolated from 300 segments — over the 120 min limit), new 42 s; outputs identical
  on the 300-segment sample. 81/81 tests pass. `promote_eta.py` uses the same builder, so the gate
  step benefited too. **Next:** redeploy, attempt #4 — first time real training time is observed.

- **2026-10-12 — live verification attempt #4: data build fixed, training NaN from epoch 1.**
  Redeployed with the frame-build fix (`5a055ae`), run id `797709381635769`. The frame step now
  completes (461,052 rows / 22,665 arrivals; split train 438,410 / val 61 / test 22,581 rows),
  but every train and val MAE was `nan` from epoch 1; the run then failed after 68 min with
  `ValueError: Input contains NaN` in the test-day `by_band`. Cause: the Transformer took raw
  `ETA_FEATURES` with no null handling and no scaling. Counted on `gold_arrival_tracks` (479,624
  rows): `turn_rate_dps` 24,882 nulls, `closure_kt` 24,351, `sel_altitude_ft` 5,165, a few more in
  `heading_err_deg`/`gs_kt`/`vrate_fpm`. LightGBM handles NaN natively, so the champion never hit
  this; in PyTorch one NaN batch poisons every weight on the first `optimizer.step()`.

  **Fix:** `lib.sequences.fit_feature_scaler` / `scale_features` (numpy, unit tested — 4 new
  cases) fit per-feature mean/std on the **train split only** (last window position = each row's
  own report, NaN-aware, std→1 for constant/all-NaN columns). `EtaTransformer` holds them as
  `register_buffer`s — saved in the state_dict, so the pyfunc wrapper and `promote_eta.py` need no
  change — and `forward()` standardises then maps NaN/inf → 0 (= the train mean). `train_model`
  now raises on a non-finite train MAE so a bad run fails in minutes. Local smoke test of the
  notebook's own model/training cells on synthetic data with ~5% NaNs and real-like magnitudes:
  finite, falling loss; finite predictions; buffers round-trip through the state_dict.

  **Open, not changed here:** the validation day (2026-09-14) is only 61 rows / 5 arrivals — a
  thin early-stopping signal for stage 1's epoch calibration. Same split scheme as
  `train_eta.py`, so the gate comparison stays like-for-like; revisit if the epoch count looks
  noisy. **Next:** redeploy, attempt #5.

- **2026-10-12 — live verification attempt #5: trained, registered, gated → HOLD.** Redeployed
  with the scaler fix (`61c4a4a`), run id `152281981271006`: `SUCCESS` in ~41 min end to end.
  Registered as `skywatch.ml.eta_touchdown` **v11**, set `@challenger` (`@champion` stays v9,
  LightGBM). MLflow run `e01ffc1c601e4b38b02758d8cdf8b03d`.

  | Test MAE (min) | 0–20nm | 20–40nm | 40–70nm | 70–100nm | ALL |
  |---|---|---|---|---|---|
  | Transformer v11 | 0.87 | 1.10 | 1.45 | 1.71 | 1.19 |
  | LightGBM v9 (champion) | 0.78 | 1.04 | 1.41 | 1.73 | 1.14 |

  Gate dry run (`skywatch_promote_eta`, `apply=false`, run `1057526847745553`) loaded both
  flavors through the `model_flavor` branch and wrote a real `promotion_audit` row:
  **HOLD** — *"overall MAE 1.190 vs champion 1.139 (-4.5%) | FAILS: worse than champion by more
  than the 1.0% noise band and short of the 2.0% improvement bar"*. The end-to-end path (§3) is
  now verified live.

  **Caveat on the result:** `n_epochs_final = 1`. Stage 1's best validation MAE came at epoch 1
  on the 61-row / 5-arrival validation day (the open item from attempt #4), so stage 2 refit for
  `max(1, round(1 * 1.15)) = 1` epoch. The HOLD is real, but it is a one-epoch Transformer
  against a tuned LightGBM — not yet a fair test of the architecture. Also noted:
  `train_model` keeps `model.state_dict()` by reference (`best_state` tracks the live weights,
  not a snapshot). Harmless today — stage 1's state is discarded and stage 2 wants the last
  state — but wrong if the returned state is ever used. **Next:** decide whether to give the
  Transformer a fairer epoch budget before closing the arc, or merge PR #39 with HOLD as the result.

- **2026-10-12 — live verification attempt #6: 10-epoch floor → HOLD, and worse than 1 epoch.**
  Changes: `train_model` now snapshots the best state with `copy.deepcopy(model.state_dict())`;
  new `min_epochs` widget (default 10) floors the stage-2 budget at
  `max(MIN_EPOCHS, round(best_epoch * 1.15))`; per-epoch `stage{1,2}_train_mae`,
  `stage1_val_mae` and `stage{1,2}_epoch_seconds` go to MLflow (the Jobs API returns no
  serverless notebook stdout, so this is the only per-epoch record); job timeout 2h → 3h.
  Run id `663449468559033`, `SUCCESS` in ~89 min. Registered **v12**, now `@challenger`.
  MLflow run `2e9b23d218ff408193058a6ff200933f`.

  | Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | stage 1 train MAE | 1.63 | 1.24 | 1.20 | 1.17 | 1.15 | 1.14 | | | | |
  | stage 1 val MAE (61 rows) | **1.00** | 1.24 | 1.38 | 1.30 | 1.18 | 1.09 | | | | |
  | stage 2 train MAE | 1.69 | 1.24 | 1.20 | 1.17 | 1.15 | 1.13 | 1.12 | 1.11 | 1.10 | 1.09 |

  ~5.3 min/epoch on serverless CPU (310–349 s), flat across epochs — 16 epochs ≈ 85 min.

  | Test MAE (min) | 0–20nm | 20–40nm | 40–70nm | 70–100nm | ALL |
  |---|---|---|---|---|---|
  | Transformer v12 (10 epochs) | 0.93 | 1.16 | 1.55 | 1.81 | 1.27 |
  | Transformer v11 (1 epoch) | 0.87 | 1.10 | 1.45 | 1.71 | 1.19 |
  | LightGBM v9 (champion) | 0.78 | 1.04 | 1.41 | 1.73 | 1.14 |

  Gate dry run (run `252049550373023`): **HOLD** — *"overall MAE 1.268 vs champion 1.139
  (-11.3%) | FAILS: worse than champion by more than the 1.0% noise band and short of the 2.0%
  improvement bar"*.

  **Reading:** the attempt-#5 caveat ("a 1-epoch Transformer loses") is answered. Training
  longer lowered train MAE (1.69 → 1.09) but *raised* test-day MAE (1.19 → 1.27), in every band
  except 70–100nm. That is overfitting to the training days, not under-training — the
  architecture, at this data scale and with no regularisation tuning, does not generalise to the
  held-out day as well as LightGBM. One seed per run, so the 1.19 vs 1.27 gap is indicative, not
  measured variance; the direction against the champion (both runs HOLD, by 4.5% and 11.3%) is
  consistent. The HOLD stands as the arc's result. `@champion` stays v9; nothing was applied.
