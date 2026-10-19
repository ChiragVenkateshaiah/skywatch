# SkyWatch — pre-interview refresh (roadmap DATA-1)

## 1. Why

Demos are recorded for interviews (`docs/CAPABILITIES_ROADMAP.md` §1). The live pipeline stays
deployed `PAUSED` for Free Edition quota, so before an interview the App's "now" can be weeks
old. DATA-1 is one on-demand job that turns that into **today's** picture in ≤ 10 min, so the
demo can say "this is today's ingestion as of HH:MM UTC; the models were trained on 50 days of
history".

Decided 2026-10-18: **keep history, reset only the live picture**. `silver_positions` (50 days,
5.0 M rows) and `gold_*` are the models' training data, and the M2 champion is a climatology
computed from them.

## 2. Measured before designing

The first manual pass (`docs/OPS_BRIEFING_PLAN.md` §6, 2026-10-18) took ~25 min. Per task:
poller 5.3 min, pipeline ~2.3, gold 1.1, briefing 0.4 — and the three scoring jobs, started in
parallel, **645–705 s of setup each** for 43–68 s of execution. `score_irregularities.py` has no
`%pip`, so it wasn't package installs. Re-run alone (`770714143016534`): **setup 4 s**, exec 59 s
— same work. Concurrent serverless starts wait for each other on this workspace, so the refresh
runs every task strictly in sequence.

## 3. Design

`resources/skywatch.demorefresh.job.yml` → `skywatch_demo_refresh`, no schedule, 25-min hard
timeout, 9 tasks in sequence:

| Task | What | Notes |
|---|---|---|
| `prepare` | `src/demo_refresh_prepare.py` — preflight, then reset | Preflight: history tables exist, `eta_touchdown@champion` and `demand_forecast@champion` resolve, the exported click-to-predict model file is on the Volume, the briefing endpoint is `READY`. **Any failure raises before anything is deleted.** Reset: `DELETE` from `predictions`, `demand_forecast`, `irregularity_flags`, printing the `RESTORE TABLE … TO VERSION AS OF n` line for each. |
| `poll` | `src/poller.py` | `var.refresh_poll_seconds` = 150 s burst (vs 270 s on the schedule) |
| `medallion` | `pipeline_task` on `skywatch_medallion` | incremental, not full refresh |
| `gold` | `src/build_gold.py` | `mode: serving` hard-coded (the job default `full` rebuilds all history) |
| `score_eta` → `score_demand` → `score_irregularities` | existing notebooks | same parameters as their own jobs |
| `brief` | `src/brief_ops.py` | `var.briefing_endpoint` |
| `check` | `src/demo_refresh_check.py` | `run_if: ALL_DONE` — reports a broken refresh too. Reads its own run via the Jobs API (`{{job.run_id}}`), freshness of `gold_tracks` / `demand_forecast` / `ops_briefings`, appends one row to `refresh_audit`, raises on `NOT_READY` |

`src/lib/refresh.py` (pure, `tests/test_refresh.py`, 11 cases): `step_timings` (Jobs API JSON →
per-task setup/exec/total) and `readiness` → `READY` / `READY_SLOW` (over budget) / `NOT_READY`
(failed task, or any source missing or > 15 min old). Ingestion freshness is judged on
`gold_tracks`, not `predictions`: a quiet night sky can give zero inbounds, and that is a
warning, not a failed refresh.

Kept, by design: `ops_briefings`, `promotion_audit`, `model_health` (audit logs). Side effect:
`monitor.py`'s drift and rolling-MAE checks read `predictions` history, so they start again
from the refresh.

## 4. Progress log

- **2026-10-18 — first run: `READY` in 8.53 min** (job run `1065288548212290`).

  | Task | setup s | exec s | total s |
  |---|---|---|---|
  | prepare | 8 | 27 | 35 |
  | poll | 2 | 157 | 160 |
  | medallion | 0 | 132 | 132 |
  | gold | 3 | 46 | 49 |
  | score_eta | 1 | 57 | 59 |
  | score_demand | 1 | 25 | 27 |
  | score_irregularities | 1 | 24 | 26 |
  | brief | 2 | 14 | 17 |

  Every task started in ≤ 8 s (vs 645–705 s in parallel). 91 KATL inbounds, 1 holding flag
  (N422V, 20 nm), briefing source `llm`, check passed, `stale = false`, data age 1 min. The
  briefing matches its facts. Two facts checked by hand: "all 91 within 30 min" is correct (max
  ETA 27.1 min — the 120 nm scoring radius caps it); see the open item below for the other.

  **Open — `arrivals_so_far_today` is "seen by SkyWatch", not the airport's count.**
  `score_demand.py` builds today's context from `gold_touchdowns` and fills every 15-min slot
  without data with 0. After one short burst it reported 1 arrival at 15:39 UTC, and the App's
  KPI tile shows "Arrivals so far today: 1". Correct for what was observed, misleading in a
  demo. Not changed in DATA-1.

- **2026-10-19 — ops agent (roadmap DATA-2a): `scripts/demo_ready.py`.** One local command
  starts the refresh, prints each task as it starts and ends, reads the job's own
  `refresh_audit` verdict, and exits 0 (READY / READY_SLOW) or 1. A failed task is classified
  (`lib.refresh.classify_failure`: transient / auth / data / code); only an all-transient failure
  is retried, once, as a Jobs **repair** run (re-runs the failed task and later ones, not the
  preflight/reset). Anything else gets an LLM diagnosis. Shared REST client moved to
  `scripts/dbx.py` (also used by `eval_briefing.py`).

  **Agent finding, worth keeping:** the first version gave Maverick tools to fetch the evidence
  itself. Tested on a real past failure (M1 Transformer attempt #4, run `797709381635769`,
  `ValueError: Input contains NaN`), it wrote a fake tool call as plain text and **invented a
  `java.io.FileNotFoundException` that was nowhere in the run**. Also found: the run JSON lists
  every retry attempt of a task under the same key (`train` twice). Fixes: `latest_attempts`;
  ANSI colour codes stripped from traces; the evidence goes straight into the prompt (no tool
  loop); and the answer must quote an error line that really appears in it
  (`lib.refresh.quote_is_grounded`) — otherwise the raw error is printed instead. Re-tested on two
  real failures: both answers quoted the real error ("Input contains NaN.", "Run timed out").
  Limit: the agent only knows what the error says — for the 2 h timeout it suggested a longer
  timeout, while the real cause (the slow frame build) isn't in that error at all.

  **End-to-end run** `1057941818921701`: **READY_SLOW, 10.1 min** — 113 inbound, 6 flagged,
  LLM briefing, data as of 14:56 UTC. Over budget by 6 s: `prepare` exec 65 s (27 s on the first
  run) and `medallion` 163 s (132 s). `refresh_poll_seconds` 150 → 120 for headroom.
