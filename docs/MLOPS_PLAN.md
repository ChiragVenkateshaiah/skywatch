# SkyWatch — Production MLOps plan

**Status: planned, not started.** Picked up *after* Model 3 is trained and the LinkedIn video +
post are out (the project having matured from "SkyWatch Lite" to "SkyWatch"). This document is
the agreed scope and division of labour so we can start cleanly when the time comes.

The goal is **hands-on practice of production MLOps** on a real project: every change tested in
CI, promoted through environments by an automated gate with a human approval, monitored in
production, retrained on a trigger, released by tag with a rollback path.

---

## 1. Where SkyWatch is now

| Capability | Have | Gap |
|---|---|---|
| IaC (Asset Bundle) | ✅ all resources bundled | one target (`dev`); no staging/prod |
| Experiment tracking | ✅ MLflow + UC registry, `@champion`/`@challenger` | promotion is inline in `train_eta.py`, not a gated step |
| Versioned pipeline code | ✅ git, feature-branch + PR | no CI — nothing runs on a PR |
| Batch scoring → Delta | ✅ 3 scoring jobs (`score_eta`, `score_demand`, `score_irregularities`) | deployed PAUSED, no enforced cadence |
| Testing | ❌ | zero automated tests; notebooks aren't importable |
| Environments | ❌ | dev only; `skywatch.stream` referenced across ~6 files |
| Monitoring | ❌ | `predictions_scored` computes error but nothing watches it; no drift, no freshness, no alerts |
| Retraining | ❌ manual | no trigger, no schedule, no automation |
| Release management | ❌ | no tags, no changelog, no rollback runbook |
| Governance | partial | roadmap doc exists; no model cards, no promotion audit |

Current level: **"reproducible notebooks in git with a registry"** — solid Level 1. Target:
Level 3–4.

---

## 2. The eight tracks

### Track 1 — Test harness
`eta_features.py`, `demand_lib.py`, `build_gold.py` are `# Databricks notebook source` files
with `%run` / `dbutils` — not importable, not testable.

- Extract pure logic into plain `src/lib/` modules (`features.py`, `forecasting.py`,
  `geometry.py`, `labels.py`); keep thin notebook wrappers that `%run` them.
- `pytest` suite: feature dtype/category handling, `mae`/`mase`/`wql` math, haversine/bearing,
  `seg_id` determinism (already verified ad-hoc — make it a real test), touchdown acceptance
  logic.
- A tiny ADS-B fixture + golden-output test for the gold transforms.

### Track 2 — CI pipeline (GitHub Actions)
- **PR** → `pytest` + `ruff` + `databricks bundle validate -t dev`
- **merge to `main`** → `bundle deploy -t staging`
- **release tag `v*`** → `bundle deploy -t prod`, behind a GitHub Environment with a required
  reviewer
- Auth: service-principal OAuth (M2M) — `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` in
  GH Secrets; scoped PAT as the documented fallback if Free Edition blocks SP OAuth.

### Track 3 — Environments (catalog isolation, one workspace)
| env | catalog | mode | deploys | data |
|---|---|---|---|---|
| **dev** | `skywatch_dev` | `development` | you, from laptop | reads `skywatch` silver/gold **read-only**; writes own `.ml` / `.stream` |
| **staging** | `skywatch_staging` | `production` (run-as SP) | CI on merge to `main` | same read-only pattern |
| **prod** | `skywatch` (existing) | `production` | CI only, on release tag, behind approval | the real poller + medallion + backfill land here; real `@champion` |

Data lives **once** (prod). Lower envs read a copy/sample read-only — realistic, and avoids
running ingestion three times on Free Edition quota. Document how the catalog split maps to a
real multi-workspace setup.

- Code change: audit the ~6 files referencing `skywatch.stream`, make every one take it from a
  bundle var.
- Add `targets: {staging, prod}` to `databricks.yml` with `mode: production`, `run_as` an SP,
  a `permissions` block, per-env variable files.

### Track 4 — Promotion gate
Pull alias-setting out of `train_eta.py` into `src/promote_eta.py` + a `skywatch_promote_eta`
job:

- load `@challenger`, evaluate on a **pinned holdout** (Delta `VERSION AS OF`)
- compare to `@champion` on the metric contract — M1: MAE by distance band; M2: MASE; gate =
  "≥ 2 % better or within noise **and** no band regression"
- move the `@champion` alias **only if** the gate passes **and** a human approves (GitHub
  deployment approval, or a Databricks job manual task)

### Track 5 — Monitoring (DIY — Lakehouse Monitoring is paid)
`src/monitor.py` + a **live, low-frequency** `skywatch_monitor` job → `skywatch.ml.model_health`:

- **feature drift** — PSI / KS on `ETA_FEATURES` between the training snapshot and the last N
  days of `predictions`
- **performance** — rolling MAE from `predictions_scored` vs the champion's registered backtest
  number
- **freshness / volume** — poller landing rows; row-count anomaly vs 7-day median

Dashboard tile + a **Databricks SQL Alert** on threshold breach → notification.

### Track 6 — Retraining automation
`skywatch_retrain` workflow: `build_gold mode=full` → train → register `@challenger` → run the
Track 4 gate. Triggers: (a) weekly schedule, (b) a drift alert from Track 5, (c) "N new backfill
days landed". **Built + deployed PAUSED**; demonstrated once end-to-end; not left to fire
unattended (would eventually lock out a day's Free Edition quota).

### Track 7 — Release management
Semver tags, `CHANGELOG.md`, tag → prod deploy, documented rollback (redeploy previous tag +
`@champion` alias rollback to the prior model version).

### Track 8 — Runbooks & governance
Runbooks (deploy, rollback, incident, retrain), model cards for M1 / M2 / M3, a promotions audit
log, an MLOps architecture doc.

---

## 3. Free Edition reality check

| Need | Free Edition | Practice substitute |
|---|---|---|
| Multi-workspace envs | ❌ one workspace | catalog isolation + `mode: production` targets — same workflow, same commands |
| Model serving endpoints + inference tables | ❌ | batch scoring → Delta (already the design); monitor those tables |
| Lakehouse Monitoring | ❌ | DIY `monitor.py` → `model_health` + SQL Alert |
| Scheduled jobs 24/7 | ⚠️ quota | build schedules; monitor live + low-freq; retrain/scoring PAUSED, run on demand |
| Service principals | limited | try SP OAuth; scoped PAT fallback |
| CI compute | GH Actions free tier | fine |

Everything is practiceable with these substitutions. Paid-only pieces are built as the
substitute **and** documented as "production swaps X for Y" — itself a portfolio artifact.

---

## 4. Division of labour

| Track | **Your hands-on part** | **My part** |
|---|---|---|
| 1 Tests | run `pytest` locally, add cases, wire coverage | refactor notebooks → `src/lib/`, write the initial suite + fixtures |
| 2 CI | author `.github/workflows/*.yml`, create the SP/PAT, set GH Environments + required reviewers + branch protection | exact commands CI runs, auth setup checklist, review your YAML |
| 3 Environments | create `skywatch_staging` / `skywatch_prod` (or the read-only grants), first staging deploy | `targets` block + per-env var files + `run_as` / `permissions`, finish the schema-var audit |
| 4 Promotion gate | run the promote job, make the promote/hold call, approve the deployment | `promote_*.py` + metric contract + the gate job |
| 5 Monitoring | configure the SQL Alert + notification destination, read the health dashboard, act on a drift signal | `monitor.py` (PSI / rolling-MAE / freshness), schedule, dashboard tile |
| 6 Retraining | set the schedule, trigger a drift-driven retrain once, review the challenger PR | wire the retrain workflow + the three triggers |
| 7 Release | cut a tagged release, write a changelog entry, practice one rollback | semver + release-notes convention, tag → prod path, rollback doc |
| 8 Runbooks | write each runbook the first time you perform that operation | model cards, MLOps architecture doc, review runbooks |

**Sequence:** Track 1 + Track 3 (parallel) → Track 2 (CI) → Track 4 → Track 5 → Track 6 →
Track 7 / 8. Estimated 2–3 weeks part-time.

---

## 5. Decisions made (2026-09-07)

1. **Environments:** catalog isolation in the one Free workspace; data lives once in prod; lower
   envs read prod data read-only. Not a second free account.
2. **CI:** GitHub Actions. SP OAuth (M2M) first, scoped PAT fallback. Prod deploy behind a
   GitHub Environment with a required reviewer.
3. **Notifications:** email for job failures (built-in `email_notifications.on_failure`); Slack
   incoming webhook for model alerts **if a Slack workspace is available** — otherwise email for
   everything. *(open — see §6)*
4. **How live:** hybrid — CI + promotion gate fully live (GH Actions, no Databricks quota);
   monitor job live but low-frequency; retrain + scoring jobs built and PAUSED, run on demand.
5. **Timing:** the **LinkedIn video ships first** on the current build (M1 + M2 + rule-based M3
   + App + dashboard). MLOps is a separate arc with its own post
   ("took my portfolio project to production-grade MLOps"). Not bundled.
6. **Ordering:** finish the 15-month data ingest → gold `full` rebuild → one **manual** retrain
   of M1 / M2 + build the M3 hold-risk classifier **now**. MLOps pipeline after. The *next*
   retrain cycle is the first one through the gate.
7. **Repo layout:** one `skywatch` repo — `.github/`, `src/`, `src/lib/`, `tests/`,
   `resources/`, `docs/`. No infra/CI split.

---

## 6. Open items to resolve at kickoff

1. **Slack workspace** for model alerts? If yes, an incoming webhook; if no, email for
   everything.
2. **M3 classifier scope** — once the 60 s backfill is ingested and we can count events: binary
   **hold-risk** only, or also attempt a **go-around** class? The ingest decides.
3. **Prerequisite check** — Model 3 trained ✔, LinkedIn video out ✔, LinkedIn post out ✔ before
   Track 1 starts.

---

## 7. Relationship to the roadmap

`docs/ML_ROADMAP.md` §11 (Monitoring & MLOps) and §12 Phase 6 (Production hardening) are the
one-paragraph version; this document is the executable plan for the Free-Edition-practiceable
slice of Phase 6. The paid-workspace items in Phase 6 (real serving endpoints, online feature
tables, continuous streaming, Mosaic AI model training) stay out of scope here.
