# SkyWatch — LLM ops briefing

## 1. Why

Queued as the next arc in `docs/M1_TRANSFORMER_PLAN.md` §5, after the M1 Transformer arc closed
(HOLD, PR #39). The roadmap's App table (`docs/ML_ROADMAP.md` §9) has had an *"Ops briefing
(later) — LLM-generated paragraph summarising the arrival picture and risks"* row since day one.
This is a new user-facing capability, not more model depth: one short, plain-English paragraph
an arrival coordinator can read in ten seconds, built from the numbers the three models already
publish.

Same rule as the MLOps and M1 arcs: verify live before designing, and evaluate for real rather
than demo once.

## 2. What was checked live first (2026-10-18)

- **Foundation Model API on Free Edition:** 12 pay-per-token endpoints are `READY` on the
  workspace, 9 of them chat (`llm/v1/chat`): `databricks-meta-llama-3-3-70b-instruct`,
  `databricks-gpt-oss-120b`, `databricks-gpt-oss-20b`, `databricks-qwen3-next-80b-a3b-instruct`,
  `databricks-qwen35-122b-a10b`, `databricks-llama-4-maverick`, `databricks-gemma-3-12b`,
  `databricks-meta-llama-3-1-8b-instruct`, `databricks-deepseek-v4-flash-0731`. So the roadmap
  §13 caveat ("limited endpoints") doesn't block this arc; the template fallback (§3) is still
  needed for quota/outage.
- **One-prompt probe** (same 3 facts, `temperature=0`) to three of them: all answered in
  2.4–3.5 s. Llama 3.3 70B and gpt-oss-120b were correct. Qwen3-next wrote *"42 aircraft
  **arrived**"* for 42 **inbound**. One prompt, but it is the exact failure mode a briefing must
  not have, so the design checks the output, not just the prompt.
- **Source-table freshness:** the live pipeline deploys `PAUSED` (`poller_pause_status`, Free
  Edition quota). Newest rows right now: `predictions` 2026-09-18 12:03, `demand_forecast`
  2026-09-14 01:53, `irregularity_flags` 2026-09-16 00:46 — weeks old, and **days apart from each
  other**. A briefing that silently blends them would read like one picture of "now". So data age
  and cross-source skew are first-class facts, and the age banner is rendered by code, not left to
  the model.

## 3. Design

```
predictions ─┐
demand_forecast ─┼─► build_facts() ─► facts (JSON, all numbers pre-rounded)
irregularity_flags ┘         │
                             ├─► LLM (FM API endpoint) ─► check_briefing(text, facts)
                             │                               │ pass → source = "llm"
                             │                               │ fail ─┐
                             └─► template_briefing(facts) ◄──────────┘ source = "template"
                                          │
                                          ▼
                              skywatch.stream.ops_briefings  (one row per run, every run)
                                          │
                                          ▼
                                 App "Ops briefing" panel
```

- **Batch job, not an in-app call.** `src/brief_ops.py` runs on the same schedule pattern as the
  scoring jobs (deployed `PAUSED`) and appends one row per run to `ops_briefings`. Matches the
  Free Edition serving pattern (`docs/ML_ROADMAP.md` §8: jobs write Delta, the App reads it),
  needs no new App permission (the App SP already has `SELECT` on the schema, and no
  `serving_endpoint` grant), and every briefing the App ever showed is auditable afterwards —
  same idea as `promotion_audit`.
- **Facts first, prose second.** `lib/briefing.py:build_facts` turns the latest rows into a small
  dict: inbound count and near-term arrivals, next-hour and peak-bin demand vs AAR, surge bins,
  current flags, and per-source `as_of` / age / skew. Numbers are rounded once there, so the
  check can compare exactly.
- **Grounding check, deterministic.** `lib/briefing.py:check_briefing` rejects text whose numbers,
  `HH:MM` times, or callsign-shaped tokens don't appear in the facts, plus a length cap. It does
  **not** catch semantic errors like "arrived" vs "inbound" with a correct number — that is
  measured by the evaluation (§4), not claimed as solved.
- **Template fallback.** `lib/briefing.py:template_briefing` writes the same picture from the
  facts with plain f-strings. Used when the endpoint errors, times out, or the check fails. The
  App always has something correct to show, and the row records which path produced it and why.
- **Data age is code, not prose.** The App renders "data as of … (N min old)" and a stale warning
  from the row's columns. The model is told the age but is not trusted to state it.

## 4. Evaluation (model choice)

`scripts/eval_briefing.py` (local; `urllib` + `databricks auth token`, no new dependency) runs a
fixed set of fact sheets — every historical scoring snapshot plus synthetic edge cases (surge,
emergency, empty sky, stale/skewed sources) — through the candidate endpoints, a few samples
each, and reports per endpoint: grounding-check pass rate, latency p50/p95, completion tokens, and
the failing texts for manual review. The job's default endpoint is chosen from that table and the
table goes in this doc.

**Run 2026-10-18** — 14 fact sheets (7 historical snapshots, 7 synthetic) × 6 endpoints × 3
samples, `temperature=0.2`, `max_tokens=1024`, 2 workers:

| endpoint | check pass | call errors | latency p50 s | p95 s | completion tokens p50 |
|---|---|---|---|---|---|
| `databricks-llama-4-maverick` | **42/42** | 0 | 2.5 | 4.2 | 82 |
| `databricks-meta-llama-3-3-70b-instruct` | 41/42 | 0 | 1.9 | 3.9 | 54 |
| `databricks-qwen3-next-80b-a3b-instruct` | 41/42 | 0 | 4.9 | 7.1 | 99 |
| `databricks-gpt-oss-120b` | 38/42 | 0 | 4.4 | 6.9 | 720 |
| `databricks-gemma-3-12b` | 35/42 | 0 | 2.6 | 3.6 | 109 |
| `databricks-gpt-oss-20b` | 24/42 | 0 | 5.8 | 9.4 | 1023 |

What the failures were:
- **gpt-oss (both):** every failure is `empty text` — reasoning used the whole `max_tokens`
  budget before any answer (p50 720 / 1023 completion tokens for a ~60-word paragraph). Wrong
  tool for a short summary at this latency; not retried with a bigger budget.
- **Gemma 3 12B:** invented end-of-bin times (`17:30`) and split callsigns (`DAL 100` → the
  check sees a stray `100`). Also the clearest *semantic* error in the manual read, which the
  check passes: "nine bins exceed the AAR upper limit of 25.3" (25.3 is the forecast's upper
  estimate, not an AAR).
- **Llama 3.3 70B / Qwen3-next:** one each, both an invented number ("both within 2 minutes",
  "within the next 11 minutes") — exactly what the check is for.

Manual read of passing texts (one sample per sheet, the two top endpoints): no wrong-meaning
errors found for Maverick — one suspected one ("52 inbound within 30 minutes") checked against
the table and was correct (all 52 had ETA ≤ 30 min). Llama 3.3 is shorter and vaguer ("there is
an emergency" without the callsign). **Default: `databricks-llama-4-maverick`**
(`var.briefing_endpoint`). Llama 3.3 70B is the drop-in alternative if Maverick is retired.

Also from the first, smaller run: **`REQUEST_LIMIT_EXCEEDED`** (workspace QPS limit on
pay-per-token endpoints) at 4 parallel workers — the script and the job both back off and retry
on it. And a script bug worth knowing about: `scored_at` has microseconds, the Statement API's
JSON timestamps only milliseconds, so equality filters must use the value cast to string.

## 5. Scope boundary

In: facts, prompt, check, fallback, job, `ops_briefings` table, App panel, evaluation, live run.
Out: chat/Q&A over the data (the Genie space already covers it), fine-tuning (production-only on
Free Edition, roadmap §13), and turning the live pipeline on permanently — a short live window
for the final verification is a separate, explicit decision because of quota.

## 6. Progress log

- **2026-10-18 — arc started.** Live checks in §2. Branch `ml/ops-briefing`.
- **2026-10-18 — built and verified live.** `src/lib/briefing.py` + `tests/test_briefing.py`
  (36 cases: facts, prompt, response parsing, check, and "the template always passes its own
  check" across normal / surge / near-capacity / emergency+holding+go-around / empty sky /
  missing sources). `scripts/eval_briefing.py` → §4 table, default endpoint Maverick.
  `src/brief_ops.py` + `resources/skywatch.briefing.job.yml` (`skywatch_brief_ops`, every 10 min
  at :01, deployed `PAUSED`), `var.briefing_endpoint`, App "Ops briefing" panel
  (`app/data.py:latest_briefing`).

  Found while building: `score_irregularities.py` writes **no rows** on a run that flags
  nothing, so "newest flags" can be days older than the newest predictions and still look
  current. `build_facts` drops flags older than 15 min before the predictions run (same
  10-min schedule) — tested. The App's existing Model 3 panel had the same blind spot — fixed
  with the same rule in a follow-up (`fix/app-flags-and-sklearn`).

  Live runs on the real (stale) tables:
  - `387998633168606` — `SUCCESS`, source `llm`, check passed, 1.2 s, 605 prompt / 79 completion
    tokens. Every statement matches the facts (48 inbound, FFT2539 first at 2.6 min, 26 next
    hour, peak 8.7 vs 22.5). The 2026-09-16 holding flag was correctly dropped (older than the
    2026-09-18 predictions). Row: `stale = true`, `max_age_min = 49206`, `source_skew_min = 6370`.
  - `475221092241260` — fallback test with `endpoint=databricks-no-such-endpoint`: `SUCCESS`,
    source `template`, `check_reasons` = the `ResourceDoesNotExist` error, same picture in
    template wording.
  - App redeployed; panel rendered headless with Streamlit `AppTest` against the real warehouse:
    stale warning ("34 days old, … 4 days apart … not a picture of now"), the briefing, caption
    "LLM `databricks-llama-4-maverick`, grounding check passed", facts expander.

  **Open:** a fresh-data run needs the live pipeline on for a window (poller → gold → scoring →
  briefing), which is a quota decision, not done here.

- **2026-10-18 — follow-up fixes (App).** (1) Model 3 panel: flags older than 15 min before the
  current Model 1 run are no longer shown as current; a caption gives the last flag's age
  instead. (2) Click-to-predict was broken: `app/requirements.txt` had no scikit-learn, and the
  App build log shows only `joblib`, `lightgbm`, `scipy` installed. `LGBMRegressor` unpickled
  without scikit-learn gets a stub base class, so `predict()` raised `'super' object has no
  attribute 'get_params'`. Pinned `scikit-learn==1.6.1`, the champion's training env (its MLflow
  `requirements.txt`: `lightgbm==4.7.0`, `scikit-learn==1.6.1`). Verified headless with `AppTest`
  against the real warehouse: the whole page renders, click-to-predict returns 2.6 min (matches
  the predictions table), the Model 3 panel shows "Last flag was 32 days ago … not shown as
  current". Ages over 48 h now read in days everywhere in the App.

- **2026-10-18 — fresh-data run (closes the §6 open item).** One manual pass instead of
  unpausing the schedules (quota): poller burst `460541681492515` (5.3 min) → `skywatch_medallion`
  update `12aa367c` → gold with `notebook_params.mode=serving` (`963599045278190`, 1.1 min — `run-now`
  params do override, unlike `bundle run --var`) → `score_eta` / `score_demand` /
  `score_irregularities` in parallel (12.9 / 12.5 / 11.7 min) → `skywatch_brief_ops`
  `953757917752291`. Briefing: source `llm`, check passed, 1.7 s, **`stale = false`, data age
  1 min, skew 0** — "1 aircraft inbound within 30 minutes … N208LS … 22.8 minutes … peak at 09:45
  is 4.2 … headroom 18.3 … no emergencies or aircraft holding"; every statement matches the facts.
  Night at KATL (03:25 local) and only one burst of today's data, so `arrivals_so_far_today = 1`
  and a near-empty sky are expected, not a bug. App redeployed from `main` (PR #42 build installs
  `scikit-learn-1.6.1`); headless render on the fresh data: no stale warning, click-to-predict
  22.8 min (matches the briefing). All schedules left `PAUSED`.
