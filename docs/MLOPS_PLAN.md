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
| Experiment tracking | ✅ MLflow + UC registry, `@champion`/`@challenger`, gated promotion (M1) | M2's promotion still inline in `forecast_demand.py` — not yet extracted |
| Versioned pipeline code | ✅ git, feature-branch + PR, CI runs lint/test/validate on every PR | environment gap only — see Environments row |
| Batch scoring → Delta | ✅ 3 scoring jobs (`score_eta`, `score_demand`, `score_irregularities`) | deployed PAUSED, no enforced cadence |
| Testing | 🟡 started | 39 pytest tests (geometry, ETA features, demand-forecast lib) — see §6a. `build_gold.py` transforms + CI wiring still open |
| Environments | ✅ dev (live) + staging (CI-deployed, isolated catalog) | no literal `prod` tier — by design, see Track 3 |
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

**Done, 2026-09-16 — see §6a.** Revised scope to match Track 3's dev/staging (no literal
`prod` target — see that section for why):
- **PR** (`.github/workflows/pr-checks.yml`) → `pytest` + `ruff` (no credentials needed) →
  `databricks bundle validate` against **both** bundles (root `-t dev`, `serving_staging -t
  staging`) — needs the `skywatch-ci` SP.
- **merge to `main`** (`.github/workflows/deploy-staging.yml`) → `bundle deploy -t staging`
  only. Deliberately does **not** touch the root `databricks.yml` / `dev` target — promoting a
  change to the live `dev` environment stays a manual, human-run deploy. That's a Track 4
  (promotion gate) decision, not CI plumbing; revisit whether it deserves its own
  tag-triggered, approval-gated workflow once Track 4 exists.
- Auth: reuses the **same `skywatch-ci` service principal Track 3 already created** — no
  separate CI identity needed. OAuth M2M (`DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET`
  / `DATABRICKS_HOST`) stored as GitHub repository secrets. PAT fallback not needed — SP OAuth
  worked on the first try on this Free Edition workspace.
- **Public-repo note:** GitHub withholds secrets from workflows triggered by fork PRs (a
  built-in protection) — the `validate` job will fail there. Expected, not a bug; this project
  has one contributor.

### Track 3 — Environments (catalog isolation, one workspace)

**Revised 2026-09-16, after hitting two real DAB constraints (below) — this supersedes the
original dev/staging/prod table.** Scope trimmed to **dev (unchanged) + one new staging**, not
a literal three-tier split: for a single-operator portfolio project, `dev` already **is**
prod in practice (it's the live app your demo/LinkedIn post point at), and migrating it to a
differently-named `prod` target would recreate the App under a new identity — new URL, breaking
the one already shared. Not worth it. Staging is the piece that actually teaches the
CI/promotion-gate workflow; a third tier can be revisited later if this project ever needs it.

| env | catalog | mode | deploys | data |
|---|---|---|---|---|
| **dev** (existing, untouched) | `skywatch` | `development` | you, from laptop, root `databricks.yml` | the real poller + medallion + backfill land here — unchanged from today |
| **staging** (new) | `skywatch_staging` | `production`, `run_as` an SP | CI on merge to `main`, a **second, separate** bundle | reads `skywatch` silver/gold **read-only**; writes its own `.ml` / `.stream` |

**Constraint 1 — confirmed empirically, not assumed:** DAB has no way to exclude a resource
from one target. Tested `resources.pipelines.skywatch_medallion: null` under a scratch target;
`bundle validate` accepted it silently but the resolved resource graph still had the pipeline,
fully intact. A target deploys *every* declared resource or none.

**Why that's a hard blocker, not just untidy:** Free Edition allows exactly **one active
Lakeflow pipeline per workspace**. If `staging` shared `databricks.yml` with `dev`, deploying
`-t staging` would try to create a *second* pipeline object under staging's independent
deployment state, and that fails outright — Free Edition rejects it. Same practical problem
(not platform-blocked, just wasteful and pointless — staging has no poller feeding it) for the
poller/backfill/gold jobs.

**The fix: two separate bundle projects**, each with its own `databricks.yml`, deployed by two
separate `bundle deploy` commands:

1. **Root `databricks.yml` (existing) — unchanged.** Poller, pipeline, backfill, gold, the
   legacy `skywatch_lite` job, **and** the existing `dev`-target App + train/score jobs stay
   exactly where they are. Moving them to a new bundle would change their Databricks-side
   identity (state is tracked by bundle name + target) — Databricks Apps and Jobs are
   workspace-uniquely-named, so a differently-named bundle deploying "the same" resource for
   the first time would try to *create* it, colliding with the one that already exists. This is
   the same reasoning that ruled out a literal `prod` migration above, applied one level down.
2. **New `serving_staging/databricks.yml` — staging-only, no `dev` target inside it** (dev's
   serving resources already exist in the root bundle; this bundle only ever adds the staging
   copies). Contains:
   - `mode: production`, an explicit `workspace.root_path` (DAB requires this for
     `mode: production` — confirmed via `bundle validate`, it errors without one), `run_as` a
     service principal (Track 2 prerequisite — this bundle can't deploy until that SP exists).
   - Its own **copies** (not moves) of `skywatch.train.job.yml`, `skywatch.forecast.job.yml`,
     `skywatch.holdrisk.job.yml`, `skywatch.score.job.yml`, `skywatch.scoredemand.job.yml`,
     `skywatch.irregularity.job.yml`, `skywatch.app.yml`, `skywatch.appkeepalive.job.yml`, each
     with resource keys/names suffixed (e.g. `skywatch-arrival-manager-staging`) so nothing
     collides workspace-globally with the `dev` originals. Notebook/source paths point at the
     *same* shared `../src/*.py` and `../app/` — staging runs identical code, isolated data.
   - Its own `variables:` block. DAB doesn't let two separate `databricks.yml` files share
     variable definitions — this means duplicating the subset of vars these resources use
     (`catalog`, `stream_schema`, `warehouse_id`, `apt_icao`/`apt_lat`/`apt_lon`/
     `apt_aar_per_hour`, model names, the `eta_*`/`demand_*`/`holdrisk_*` tuning vars). Real
     cost of the two-bundle split; bounded and worth it for correctness. Keep the two files'
     shared var *defaults* in sync by hand — call this out in both files' header comments.

**Constraint 2 — a read/write catalog split the code doesn't have yet.** Staging must *read*
prod's real `gold_*` tables (it can never run its own ingestion — constraint 1) but *write* its
own `predictions` / `demand_forecast` / `irregularity_flags` / registered models. Today,
`score_eta.py` / `score_demand.py` / `score_irregularities.py` / `train_eta.py` /
`forecast_demand.py` all take a single `stream_schema` widget used for *both* reading Gold and
writing scoring output — there's no way to point those at two different catalogs at once.
**Code change needed** (not started): add a second widget (`read_stream_schema`, defaulting to
the same value as `stream_schema` so `dev`'s behavior is byte-identical) to each of those five
notebooks, and split every `SELECT ... FROM {STREAM}.gold_*` from every
`.saveAsTable(f"{STREAM}...")` / model-registration call — reads use `READ_STREAM`, writes use
`STREAM`. `build_gold.py` itself doesn't need this (staging never runs it — constraint 1).

**Your hands-on part (per the original division of labour), now concrete:**
- Create the `skywatch_staging` catalog + `ml` / `stream` schemas.
- Create a service principal for CI (this is also a Track 2 prerequisite — the two tracks are
  now coupled at this one point); grant it `USE CATALOG` + `SELECT` on `skywatch.stream` /
  `skywatch.ml` (read-only, prod) and full read-write on `skywatch_staging.*`.
- First `bundle deploy` of `serving_staging/` (manual, before CI exists to do it).

### Track 4 — Promotion gate

**M1 done, 2026-09-16 — see §6a.** `src/promote_eta.py` + `skywatch_promote_eta` job (dev) /
`skywatch_promote_eta_staging` (staging):

- load `@challenger`, evaluate on a **pinned holdout** (`gold_arrival_tracks VERSION AS OF`,
  blank = current) — same day/filters `train_eta.py` uses, so the two are comparable
- compare to `@champion` (if one exists — bootstrap auto-passes) on MAE **by distance band**,
  not just the headline number — the gate decision itself is pure Python
  (`src/lib/promotion.py`, unit-tested) called from the notebook
- gate = "≥ 2% better or within noise **and** no individual band regression" — both
  thresholds are bundle vars, not hardcoded
- **The human-approval mechanism**: the notebook takes an `apply` parameter, default `false`
  (dry run — reports the recommendation, touches nothing). A passing gate is only ever applied
  by a second, deliberate run with `apply=true` — that re-trigger, by a human who read the
  report, *is* the approval. A failing gate is never applied regardless of `apply`. No GitHub
  Environment / manual-approval-task machinery needed — simpler, and matches what Databricks
  Jobs actually support today.
- **M2's promotion gate (MASE-based) is not yet built** — `forecast_demand.py` still
  auto-promotes inline, same pattern M1 had before this. Fast-follow using the same
  `evaluate_gate` function once there's a reason to prioritize it.

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
| 3 Environments | create `skywatch_staging` catalog + schemas, create the CI service principal (shared prerequisite with Track 2), the read-only prod grants, first `serving_staging/` deploy | `serving_staging/` second bundle (§3), `read_stream_schema` split across the 5 train/score notebooks |
| 4 Promotion gate | run `promote_eta` with `apply=true` when you want to act on a passing recommendation (that's the approval — see §2 Track 4) | done for M1 (`promote_eta.py` + gate job, both envs); M2's `promote_demand.py` not started |
| 5 Monitoring | configure the SQL Alert + notification destination, read the health dashboard, act on a drift signal | `monitor.py` (PSI / rolling-MAE / freshness), schedule, dashboard tile |
| 6 Retraining | set the schedule, trigger a drift-driven retrain once, review the challenger PR | wire the retrain workflow + the three triggers |
| 7 Release | cut a tagged release, write a changelog entry, practice one rollback | semver + release-notes convention, tag → prod path, rollback doc |
| 8 Runbooks | write each runbook the first time you perform that operation | model cards, MLOps architecture doc, review runbooks |

**Sequence:** Track 1 (done, §6a) → **Track 3 and Track 2 are now coupled at the service
principal** (Track 3's `serving_staging/` `run_as` needs the same SP Track 2's CI auth needs) —
create the SP once, satisfies both. Practically: Track 3's code (the bundle split +
`read_stream_schema` change) can be written and validated without the SP; the SP is only needed
to actually *deploy* `serving_staging/`, and to wire CI. → Track 4 → Track 5 → Track 6 →
Track 7 / 8. Estimated 2–3 weeks part-time (revised up slightly from the original estimate —
the two-bundle split wasn't in the original scope).

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

## 6. Open items — resolved at kickoff (2026-09-15)

1. **Slack workspace?** No. Email for everything, including model-health alerts (Track 5) —
   the built-in `email_notifications.on_failure` plus a plain SQL Alert notification, no
   webhook to manage.
2. **M3 classifier scope** — resolved harder than the question assumed. The 24-day ingest
   showed **0 of 159** detected holds ever landed (all persistent loiterers; KATL doesn't hold
   arrivals in clear weather, and clear-weather days are all the archive has). Not "binary vs
   multiclass" — **no learned classifier at all**. Model 3 stays rule-based permanently
   (`train_hold_risk.py` stays dormant, kept as ready code per PR #15).
3. **Prerequisite check** — Model 3 trained ✔, LinkedIn video out ✔, LinkedIn post out ✔.
   Track 1 started 2026-09-15.

---

## 6a. Progress log

- **2026-09-15 — Track 1 kicked off, first slice merged.** `src/lib/geometry.py`
  (`haversine_nm` / `initial_bearing_deg` / `angular_diff_deg` — extracted, faithfully, from
  `pipeline_medallion.py`'s inline versions) + `tests/` (pytest, a local-Spark `conftest.py`
  fixture, 39 tests covering geometry, `eta_features.py`, `demand_lib.py`). Verified locally
  (portable JDK 17, no root) before pushing — all 39 pass.
  - **Deliberately NOT done in this slice:** `pipeline_medallion.py` / `build_gold.py` do
    **not** yet import from `src/lib/` — they still run their own duplicated copy. `dev` is
    currently the only bundle target and points at the live `skywatch` catalog (Track 3 not
    started), so swapping the import in the notebook that feeds the live demo/dashboard
    wasn't worth the risk without an isolated catalog to test against first. Tracked as a
    Track 1/Track 3 joint follow-up: migrate the import once a `dev` catalog exists, verify
    with a real `bundle deploy` + pipeline run there, *then* delete the duplicate.
  - `eta_features.py` and `demand_lib.py` turned out to already be plain-importable (no
    `dbutils`/`spark` reference at module level) — tested in place rather than moved into
    `src/lib/`.
  - Remaining Track 1 scope: `build_gold.py`'s touchdown-acceptance / `seg_id` logic (a
    golden-output test against a tiny ADS-B fixture, per §2 Track 1) — not started yet.

- **2026-09-16 — Track 3 code done: read/write split + `serving_staging/` bundle skeleton,
  validated but not deployable yet (needs the SP + catalog, your hands-on part).**
  - `read_stream_schema` widget added to `score_eta.py`, `score_demand.py`,
    `score_irregularities.py`, `train_eta.py`, `forecast_demand.py`, **and**
    `train_hold_risk.py` (dormant, but made consistent while touching the others). Blank
    default falls back to `stream_schema` — verified live against `dev`
    (`skywatch_score_irregularities` run, fresh row landed) that this is byte-identical to the
    old behavior.
  - `serving_staging/` created: its own `databricks.yml` + `resources/*.yml` (copies of the 6
    train/score jobs + the App + its keepalive, all resource keys/names `_staging`-suffixed).
    Two more real DAB constraints surfaced while building it, both resolved:
    - **Databricks Apps require the deploying identity to equal `run_as` exactly** — this is
      stricter than jobs (which support true impersonation: deployer ≠ runner). Confirmed by
      testing: a human deploying with a placeholder SP in `run_as` is rejected outright
      ("apps do not support a setting a run_as user that is different from the owner");
      swapping in the human's own matching identity validates fine. **Practical implication**:
      the first `serving_staging` deploy must genuinely authenticate the CLI *as* the service
      principal (SP OAuth client-credentials profile), not a human passing the SP's ID via
      `--var` — revises the "manual first deploy" hands-on step below.
    - **A bundle's file sync defaults to its own directory and rejects `notebook_path`/
      `source_code_path` references outside it** ("not contained in sync root path") —
      `serving_staging/resources/*.yml`'s `../../src/*.py` / `../../app` references need
      `sync.paths: [../src, ../app]` in `serving_staging/databricks.yml` to be allowed at all.
      Added; `bundle validate -t staging` now resolves every notebook path and the app's
      `source_code_path` correctly under `serving_staging`'s own workspace deployment path.
  - Verified via `bundle validate -t staging --output json`: `score_eta_staging`'s resolved
    `base_parameters` show `read_stream_schema=skywatch.stream` (prod) and
    `stream_schema=skywatch_staging.stream` / `model_name=skywatch_staging.ml.eta_touchdown`
    (staging's own), and the App's resolved `config.env` all point at `skywatch_staging`. Not
    deployed — `staging_run_as` has no default on purpose and blocks deploy until real.
  - **Update, same day: staging is live and verified end-to-end.** Created the
    `skywatch_staging` catalog + `stream`/`ml` schemas, and a service principal
    (`skywatch-ci`, app id `7bdaf533-8c9c-492c-ba0f-16df4ed969f2`) with an OAuth M2M secret
    (90-day lifetime) stored as a local `skywatch-ci` CLI profile — granted it read-only on
    `skywatch.stream`/`skywatch.ml` (prod) and full read-write on `skywatch_staging.*`, per the
    bundle's own header comment.
    - Deployed `serving_staging/` authenticated *as* that SP (`--profile skywatch-ci`), not a
      human passing its ID via `--var` — confirms the `run_as`-must-match-deployer constraint
      from earlier applies exactly as understood.
    - Two more issues found only by actually deploying (neither visible from `validate`):
      (1) `root_path` was still hardcoded to a personal `/Workspace/Users/<human>/` folder from
      an earlier scratch test — the SP can't write there. Fixed to `/Workspace/Shared/...` +
      an explicit `permissions: CAN_MANAGE for group_name: users` block (acknowledging the
      shared-path warning rather than ignoring it — fine on this single-user workspace).
      (2) Databricks App names cap at 30 characters; `skywatch-arrival-manager-staging` is 32.
      Renamed to `skywatch-arrmgr-staging` (app resource + the keepalive job's `app_name` param).
    - **Ran `skywatch_score_irregularities_staging` for real**: TERMINATED SUCCESS, and queried
      `skywatch_staging.stream.irregularity_flags` (as the SP — as the deploying human I have
      no grant on it, by design, since the SP owns what it creates) — one fresh row, proving
      the read-from-prod/write-to-staging split works against real data, not just resolved
      config.
    - **Started the App itself**: also needed its own grants — Databricks Apps auto-create a
      *separate* per-app service principal (distinct from the deploying `run_as` SP), which
      needed the same catalog/schema grants issued against `skywatch_staging` (run as the
      catalog owner, not the CI SP — the CI SP has no `MANAGE` privilege to grant on a catalog
      it doesn't own). Confirmed `RUNNING` / `ACTIVE` via the Apps API.
    - **Remaining for Track 3 / Track 2**: nothing blocking left for staging itself. Track 2
      (CI) can now reuse the same `skywatch-ci` SP for GitHub Actions once that track starts.

- **2026-09-16 — Track 2 done.** `.github/workflows/pr-checks.yml` (lint + test + validate
  both bundles on every PR) and `deploy-staging.yml` (redeploy `serving_staging` on every merge
  to `main`), both authenticated as the same `skywatch-ci` SP from Track 3 — no separate CI
  identity needed.
  - Added `pyproject.toml` for `ruff` (scoped to `E9`/`F` — real bugs and syntax errors, not
    style): `dbutils`/`spark`/`display`/`dlt`/`sc` declared as `builtins` (Databricks-injected
    runtime globals, not undefined names), and a `per-file-ignores` for `F821` specifically on
    the four notebooks that `%run` a shared module (`score_eta.py`, `train_eta.py`,
    `score_demand.py`, `forecast_demand.py`) — `%run` injects names at runtime in a way ruff
    can't see statically. Ran it for real first: found one genuine issue (an unused
    `timedelta` import in `backfill.py`), fixed; everything else was the expected
    notebook-global noise, now silenced correctly rather than papered over with a blanket
    ignore.
  - GitHub repository secrets set (`DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`,
    `DATABRICKS_CLIENT_SECRET`) from the `skywatch-ci` SP's own OAuth credentials.
  - **Not done**: any path that deploys to `dev` / the live environment. Deliberately left
    manual — see the Track 2 section above for why.

- **2026-09-16 — Track 4 done (M1).** `train_eta.py` no longer auto-promotes to `@champion`
  on beating the naive baseline (that was never a real bar) — it now only ever registers
  `@challenger`. `src/promote_eta.py` + `skywatch_promote_eta` / `_staging` jobs own the actual
  decision, evaluated by `src/lib/promotion.py::evaluate_gate` (pure Python, 13 pytest cases —
  bootstrap, clear pass/fail, noise-band boundaries, band-regression-blocks-an-overall-win,
  missing-band tolerance).
  - **Verified fully live, not just deployed**: deliberately retrained with `eta_max_evals=3`
    (undertuned on purpose) to get a genuinely different challenger (v8, MAE 1.154) against the
    existing tuned champion (v7, MAE 1.144). Dry run (`apply=false`, the deployed default)
    correctly reported `PASS` (0.87% worse is inside the 1% noise band) and left `@champion`
    at v7, confirmed via the registry API. Redeployed with `apply=true` baked in (`bundle run
    --var` does **not** override job parameters — a known gotcha from earlier in this project;
    only `bundle deploy --var` does), re-ran, confirmed `@champion` actually flipped to v8, then
    redeployed the safe `apply=false` default back and reverted `@champion` to v7 (v8 was a
    deliberately undertuned test artifact, not worth keeping as the live model backing
    click-to-predict — and the README's published MAE number refers to v7).
  - **Resolves a Track 1 open question**: `from lib.promotion import evaluate_gate` works
    correctly from a live Databricks Job notebook task (confirmed — the run would have failed
    with an ImportError otherwise, it didn't). This was previously unverified and cited as the
    reason `pipeline_medallion.py` / `build_gold.py` weren't migrated to import from `src/lib/`
    yet. Narrower than it sounds, though: this confirms it for a **Job notebook task**
    specifically, not a **Lakeflow/DLT pipeline task** (`pipeline_medallion.py`'s case) — DLT
    may resolve libraries differently. `build_gold.py` (also a plain Job notebook) is now
    unblocked by this; `pipeline_medallion.py` is not, yet.

---

## 7. Relationship to the roadmap

`docs/ML_ROADMAP.md` §11 (Monitoring & MLOps) and §12 Phase 6 (Production hardening) are the
one-paragraph version; this document is the executable plan for the Free-Edition-practiceable
slice of Phase 6. The paid-workspace items in Phase 6 (real serving endpoints, online feature
tables, continuous streaming, Mosaic AI model training) stay out of scope here.
