# SkyWatch — Capabilities roadmap

The single prioritised list of what SkyWatch does next, and **when** each item is picked up.
Each arc still gets its own plan doc (`docs/*_PLAN.md`) when it starts; this file only holds the
ranking, the triggers, and the version each item ships in. Re-ranked at the end of every arc.

Started 2026-10-18, after the ops-briefing arc (`docs/OPS_BRIEFING_PLAN.md`, PRs #41–#43).

## 1. Decisions this roadmap is built on (2026-10-18)

- **Audience = a recorded demo** (interviews). Databricks Apps need a workspace login, so there
  is no public hosting work. The demo is recorded from the live App.
- **Fresh data once, right before an interview — keep history, reset only the live picture.**
  `silver_positions` holds 50 days (5.0 M rows) of training history for all three models; the
  M2 champion *is* a climatology computed from it. Deleting it would stop M2 and make retraining
  impossible. The refresh clears the serving tables and rebuilds them from one fresh burst.
  Target: **≤ 10 min end to end**, data stays at "a few GB" at most (today: ~0.8 GB in total).
- **An agent may run tasks around the models and the refresh** — a Databricks-provided agent if
  Free Edition offers a usable one (to verify live), otherwise our own tool-calling agent on the
  Foundation Model API.
- **Chat v1 includes `ask_genie`** (the existing Genie space) for open SQL questions; deeper
  Genie grounding is a v2 item.

## 2. Constraints (measured, not assumed)

| Constraint | Value | Source |
|---|---|---|
| Apps per workspace | 3; 2 in use (`skywatch-arrival-manager`, `skywatch-arrmgr-staging`) | `databricks apps list` |
| App access | workspace login only | Databricks Apps |
| Chat-capable endpoints | 9 pay-per-token; **tool calling verified on `databricks-llama-4-maverick`** | live probe 2026-10-18 |
| Pay-per-token QPS | `REQUEST_LIMIT_EXCEEDED` at 4 parallel calls | `docs/OPS_BRIEFING_PLAN.md` §4 |
| Embeddings | `databricks-gte-large-en`, `databricks-bge-large-en` `READY` | endpoint list |
| Vector Search | API answers, **no endpoints** — not relied on | `GET /api/2.0/vector-search/endpoints` |
| Genie | 2 SkyWatch spaces exist ("KATL Arrival Manager", "Aviation Assistant") | `GET /api/2.0/genie/spaces` |
| **Databricks-managed MCP** | **works**: Genie (`/api/2.0/mcp/genie/<space>` → `query_space_…` tool) and UC functions (`/api/2.0/mcp/functions/skywatch/ml`) answer `tools/list` | live probe 2026-10-19 |
| `ai_query()` in SQL | works on the warehouse (`databricks-llama-4-maverick`) | live probe 2026-10-19 |
| Agent hosting | Agent Framework deploy needs custom model serving — not on Free Edition (roadmap §13); Agent Bricks not verified. Agents run as our own code. | — |
| Live pipeline | deployed `PAUSED` (quota) — data only grows when we run it | `databricks.yml` |
| Data volume | `gold_tracks` 305 MB, `silver_positions` 261 MB, `bronze_aircraft` 176 MB, rest < 30 MB | `DESCRIBE DETAIL` |
| Refresh timing (2026-10-18) | poller 5.3 min, pipeline ~2.3 min, gold 1.1 min, brief 0.4 min; **scoring ×3 in parallel: 645–705 s *setup* each, 43–68 s execution** | Jobs API |

The scoring setup time is the one number that breaks the 10-min budget. Working hypothesis (not
yet verified): Free Edition limits concurrent serverless starts, so the three parallel jobs
waited for each other. A sequential run estimates at ~9 min.

## 3. Priority list

Priority: **P0** now · **P1** next · **P2** when its trigger fires · **P3** parked.
Effort: S ≤ 1 day · M 2–4 days · L ≥ 1 week.

| ID | Capability | Pri | Value | Effort | Depends on | Pick-up trigger | Ships in |
|---|---|---|---|---|---|---|---|
| **DATA-1** | Pre-interview refresh job `skywatch_demo_refresh`: preflight → reset live tables → short poller burst → pipeline → gold serving → scoring (sequential) → briefing → readiness check; per-step timings logged | P0 | High — every demo depends on it | M | — | now | Platform v1.1 |
| **DATA-2a** | Local ops agent `scripts/demo_ready.py`: starts DATA-1, prints each task state change, reads the `refresh_audit` verdict, retries a *transient* failure once (Jobs repair), else an LLM diagnosis that must quote the run's real error | P0 | High — one command before an interview | M | DATA-1 | ✅ built 2026-10-19 | Platform v1.2 |
| **DATA-2b** | In-app "Get SkyWatch ready": a sidebar button with a confirmation (a human action — the model gets only a read-only `refresh_status` tool); App resource `job` CAN_MANAGE_RUN | P1 | Medium — demo moment | S–M | CHAT-1, DATA-2a | ✅ built 2026-10-19 | App v2.0 |
| **CHAT-1** | "Chat with SkyWatch" page in the existing App: Maverick + curated tools (`get_arrival_picture`, `get_aircraft`, `explain_eta`, `get_flags`, `get_briefing`, `get_model_status`, `ask_genie`); grounding check on numeric answers; data-age label from tool results; "how I answered" panel. **Tools are UC functions served through Databricks-managed MCP** (decided 2026-10-19): written once, used by the chat, the ops agent and Genie; `ask_genie` = the managed Genie MCP tool | P0 | High — the headline feature | L | App SP: `serving_endpoint` CAN_QUERY, `genie_space` access, `EXECUTE` on the UC functions; UC Python functions on this warehouse (verify first) | ✅ done 2026-10-19 (runs 38/37/38 of 38) | App v2.0 |
| **CHAT-EVAL-1** | `chat_audit` table (question, tools, answer, check, tokens, latency, prompt version, 👍/👎) + golden set of 50–100 questions (tool-selection accuracy, grounding pass rate, out-of-scope refusals) | P0 | High — "measure, then tune" | M | CHAT-1 | with CHAT-1 | App v2.0 |
| **CHAT-2** | Knowledge about SkyWatch and its models: docs chunked + embedded (`gte-large-en`) into Delta by a local script (not a job — decided 2026-10-21), in-memory index in the App, `search_docs` tool, answers cite `file §section` (checked) | P1 | High — "why/how" questions | M | CHAT-1 | ✅ done 2026-10-21 (PR #54) — 207 passages; golden set 43/44, 43/44, 42/44 (CHAT-1 items 38/38 each); SELFCHECK 15/15; a cited answer checked in a browser (`docs/CHAT_KNOWLEDGE_PLAN.md`) | App v2.1 |
| **CHAT-3** | Genie v2: ground-check Genie answers (show its SQL + rows), route ambiguous questions to curated tools first | P1 | Medium | M | CHAT-1 | `ask_genie` share of turns ≥ 20% or its 👎 rate ≥ 2× curated tools | App v2.2 |
| **OPT-1** | Latency: a prefetched live snapshot so common questions take one model round (prompt size measured *not* to matter; streaming rejected — an answer must pass its check before it is shown) | P1 | Medium | S–M | CHAT-EVAL-1 (numbers to tune against) | ✅ done 2026-10-19 — *rounds, not tokens*: a prefetched live snapshot cut p50 5.2 → 2.1–2.6 s at 38/38 (`docs/CHAT_PLAN.md` §6) | App v2.x |
| **OPT-2** | MLflow tracing per chat turn and per refresh step | P1 | Medium — explains each slow or wrong answer | S | CHAT-1 | with OPT-1 | App v2.x |
| **OPT-3** | Model fallback order for chat: Maverick → Llama 3.3 70B → tools-only template answer | P0 | Medium — demo safety | S | CHAT-1 | ✅ built with CHAT-1 2026-10-19 | App v2.0 |
| **DEMO-1** | Fixed demo script + warm-up at the end of `demo_ready.py` (warehouse, chat tools, Genie — first MCP call was 37 s cold); the script shows the guards working (trap question, Transformer HOLD) | P0 | High — what the interview actually sees | S | CHAT-1 | ✅ built 2026-10-19 (`docs/DEMO_SCRIPT.md`) | App v2.0 |
| **DOC-1** | ✅ `docs/LESSONS.md` (2026-10-19) — "Lessons" write-up: the real engineering stories with evidence (2 h frame build → 42 s, NaN training, Transformer overfit, parallel-setup wait, the agent's invented error, golden-set variance) | P1 | High — interview answers | S | — | after CHAT-1 (2026-10-19) | Docs |
| **DATA-3** | More history: monthly `readsb-hist` backfill (`scripts/backfill_local.py`, off-platform) | P2 | High for models | S per month | — | each new 1st-of-month file; target ≥ 14 then ≥ 60 days | Data |
| **M1-WX** | Weather features for M1 (and M2 covariate): METAR from aviationweather.gov (free, no key) joined by time | P2 | Medium–High | M | DATA-3 helps | METAR history join verified on the 50 days | M1 vNext |
| **M2-FT** | Fine-tune Chronos-Bolt; try a covariate-aware open time-series model with weather | P2 | Medium | M | DATA-3 | ≥ 14 backfill days (roadmap §14) | M2 v2 |
| **M3-LBL** | Label more holding / go-around events with the open-source `traffic` library's detectors; compare with the current rules | P2 | Medium | M | DATA-3 | always useful; then learned M3 when labels reach classifier scale | M3 v2 (rules+) |
| **AGENT-ML** | Experiment agent: proposes a feature/hyper-parameter change, runs the train job, runs the gate, writes the audit summary — a human approves any `apply=true` | P2 | Medium | M | M1-WX or M2-FT exists to try | first P2 model item starts | Platform v1.3 |
| **M1-TX** | Revisit the Transformer with regularisation and a larger validation window | P3 | Low now | M | DATA-3 | ≥ 60 days of arrivals (it overfit at today's size, `docs/M1_TRANSFORMER_PLAN.md`) | M1 vNext |
| **M3-LEARN** | Learned Model 3 (one-vs-rest) | P3 | Medium later | L | M3-LBL | label count reaches classifier scale (roadmap §7) | M3 v3 |
| **APP-1** | Minimal dashboard (decided 2026-10-19): freshness pill, 4 KPIs (inbound, next 3 h vs capacity, peak headroom, alerts), the briefing card, **one map** (alerts as rings), one demand chart, and "Ask SkyWatch" buttons that open the chat with the question asked. Removed — the chat answers them: click-to-predict, arrival-sequence table, surge list, second map + Model 3 history, the red/orange alert boxes that read as app errors | P0 | High — what an interviewer sees first | S | CHAT-1 | ✅ deployed 2026-10-21 (SELFCHECK 14/14, checked in a browser) | App v2.2 |
| **OPT-4** | Quota-aware evaluation: the golden set is the biggest single compute cost (~40 tool SQL calls + ~60 model calls per run, 15+ runs on 2026-10-19). Run it only on prompt / tool / loop changes; cache tool results per run | P1 | Medium — keeps Free Edition usable | S | CHAT-EVAL-1 | ✅ done 2026-10-21 — `scripts/eval_chat.py` skips a full run whose inputs (loop, grounding, tool SQL, golden set, endpoint) are unchanged (`--force`), caches tool results per data version (`--fresh-tools`) and prints tool SQL / cache hits / model calls; a re-run on frozen data: 7 → 1 SQL calls per item (`src/lib/eval_cache.py`) | Platform |
| **SMALL-1** | KPI tile headroom (+18.2, unrounded peak) vs briefing (18.3, rounded peak) — use one rounding | P3 | Low | S | — | next time `app/app.py` KPIs are touched | App v2.x |

Every model item goes through the existing promotion gate and writes a `promotion_audit` row. A
HOLD is a complete, documented result (as in the M1 Transformer arc).

## 4. Execution order

1. **DATA-1** — and test the parallel-setup hypothesis first (it decides the job's shape).
2. **DATA-2a** — ✅ live agent check done (MCP + `ai_query` work, agent hosting doesn't); local ops agent built.
3. **CHAT-1 + CHAT-EVAL-1** — one arc; the golden set is written before the prompt is tuned.
4. **CHAT-2**, then **OPT-1/2/3** against the CHAT-EVAL-1 numbers, then **CHAT-3**.
5. **P2 model items** as their triggers fire; **AGENT-ML** with the first of them.

## 5. Versioning

| Thing | Scheme | Where it is recorded |
|---|---|---|
| App | `v1.x` dashboard + briefing (today) · `v2.0` chat · `v2.1` knowledge · `v2.2` Genie v2 | App caption + `CHANGELOG.md` |
| Platform jobs | `v1.1` refresh · `v1.2` ops agent (local) · `v1.3` experiment agent | `CHANGELOG.md` |
| Models | UC registry versions + `@champion` / `@challenger`, promotion gate | `promotion_audit` |
| Prompts / tool sets | `prompt_version` string (e.g. `chat-v2.0.1`) | every `chat_audit` / `ops_briefings` row |

## 6. When to re-rank

- At the end of every arc (as part of its last PR).
- When a trigger in §3 fires — the item moves to P1.
- Before an interview: DATA-1/DATA-2 must be green; nothing else blocks a demo.

## 7. Lessons that apply to every agent item

- **Code decides, the model explains.** In DATA-2a the retry decision is a deterministic
  classifier; the LLM only writes the diagnosis.
- **Evidence in, grounding checked out.** Given tools to fetch the evidence itself, Maverick once
  wrote a fake tool call as text and invented an error that wasn't in the run
  (`docs/DEMO_REFRESH_PLAN.md` §4). Every agent answer must be checked against the evidence it
  was given — numbers for the briefing, a quoted error line for the diagnosis — with a plain
  fallback when the check fails. CHAT-1's tool loop needs the same guard.

## 8. Where we stopped (checkpoint 2026-10-19)

Shipped to `main` and live in the App today: DATA-1 (refresh job), DATA-2a (local ops agent),
CHAT-1 + CHAT-EVAL-1 + DATA-2b + OPT-3 + DEMO-1 (PR #48), OPT-1 live snapshot + DOC-1 lessons
(PR #49), the Streamlit 1.38.0 pin that fixed the chat page crash (PR #50).

**Stopped by the Free Edition compute quota.** Late on 2026-10-19 the SQL warehouse
(`8fd481cbf45ac93e`) would not start: every query → "The request could not be processed by the
warehouse"; `databricks warehouses start` → "Cannot create the resource, please try again later."
The live App can't load data until it starts again. Most likely the daily quota after a heavy day
(three refreshes, many jobs, ~15 golden-set runs, several App restarts) → OPT-4.

**In progress — APP-1, branch `app/minimal-dashboard` (pushed, not merged, not deployed):**
`app/dashboard.py` rewritten; `app/chat.py` picks up a question handed over by the dashboard
(`pending_question`); unused `app/data.py` helpers removed. Tested headless on Streamlit 1.38.0:
no exceptions, no error/warning boxes, 4 KPIs, briefing, map, chart, 3 "Ask SkyWatch" buttons.

**Next, in order, once the warehouse starts:**
1. `databricks api post /api/2.0/sql/statements … "SELECT 1"` — one call to confirm it's back.
2. Re-run the hand-over test: chat page with `pending_question` preset must produce one checked
   turn (on 2026-10-19 it reached the page but failed on the warehouse, not on the code).
   `AppTest` 1.38 can't drive the `st.navigation` router, so test the two halves separately.
3. `bundle deploy` + `bundle run skywatch_arrival_manager`; `apps logs … | grep SELFCHECK` (expect
   14/14); open the App in a browser once — dashboard, an Ask button, the chat answer.
4. Open the APP-1 PR, merge, then OPT-4 before any further golden-set runs; then CHAT-2.

**Resumed 2026-10-21:** the warehouse started again (`SELECT 1` ok). Steps 1–3 done: the
hand-over test passed in two halves — the dashboard's Ask button sets `pending_question`; the
chat page with it preset gave one turn, Maverick, 1 round, grounding check passed, 5.4 s.
APP-1 deployed: SELFCHECK 14/14, dashboard → Ask → chat answer checked in a browser. The data
is still as of 2026-10-19 15:01 UTC (schedules paused) — run the refresh before a demo.
Step 4 done the same day: APP-1 (PR #52), OPT-4 (PR #53) and CHAT-2 (PR #54) merged and live.

Local test setup to remember: the App's Streamlit is **1.38.0** — test in a venv with exactly that
(`AppTest` there needs `PYTHONPATH=app`); the Spark tests need a JDK (CI has one).

