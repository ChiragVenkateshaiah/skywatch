# SkyWatch — Chat with SkyWatch (roadmap CHAT-1 + CHAT-EVAL-1, App v2.0)

## 1. Why

Roadmap P0 (`docs/CAPABILITIES_ROADMAP.md`): let a viewer of the recorded demo *ask* SkyWatch
about the live picture and its models, instead of reading panels. Same rules as the briefing
and ops-agent arcs: numbers come from tools, never from the model; every answer is checked
against the evidence it was built from; every turn is audited.

## 2. Measured before designing (2026-10-19)

| Check | Result |
|---|---|
| UC **SQL** table function on the warehouse | works (`skywatch.chat.probe_sql()` → 113 inbound) |
| UC **Python** scalar function on the warehouse | works (`probe_py(21)` → 42) |
| Databricks-managed MCP for `skywatch.chat` | lists both (UC `COMMENT` becomes the tool description), calls both |
| Same SQL function, **MCP vs direct SQL** (warm, 3 runs) | **MCP 3.3–4.3 s · direct 1.9–2.3 s**; first MCP call 37 s (cold) |
| Genie via managed MCP (`query_space_…` + `poll_response_…`) | works, **async: ~16 s to ask + ~2 s to poll ≈ 18 s**; answer carries its SQL and result rows |
| Tool calling on `databricks-llama-4-maverick` | works (2026-10-18) — but see the DATA-2a finding: it once wrote a fake tool call as text and invented evidence |

Probes dropped after the test; `skywatch.chat` schema kept as the tools' home.

## 3. Design

**Tools are written once, as UC functions in `skywatch.chat`** (decided 2026-10-19). The DDL
lives in `src/chat_tools.sql`, applied by a bundle `sql_task` job, so the tools are versioned in
the repo like everything else.

**Two doors to the same functions:**
- the App's chat calls them with **direct SQL** (`SELECT * FROM skywatch.chat.<fn>(…)`) through
  its existing warehouse resource — ~1.5 s faster per call than MCP, which matters at 1–3 tool
  calls per turn against a p50 < 4 s target;
- **Databricks-managed MCP** serves the very same functions to every other agent (ops agent,
  external MCP clients) — no second definition.

| Tool | Kind | Returns |
|---|---|---|
| `arrival_picture()` | UC SQL table fn | inbound count, within 10 / 30 min, Model 1 `scored_at` |
| `next_arrivals(n)` | UC SQL table fn | the next *n* aircraft by predicted touchdown: callsign, type, ETA min, distance |
| `aircraft(callsign)` | UC SQL table fn | one aircraft's latest prediction + key features |
| `demand_outlook(aar_per_hour)` | UC SQL table fn | the next 12 × 15-min bins: predicted / upper vs AAR, `scored_at` |
| `irregularities()` | UC SQL table fn | current flags, with the 15-min "same cycle as Model 1" rule built in |
| `latest_briefing()` | UC SQL table fn | newest `ops_briefings` row |
| `model_status(model)` | UC SQL table fn | latest `promotion_audit` rows + latest `model_health` checks |
| `explain_eta(callsign)` | **App-side** Python | per-feature contributions from the in-process champion (click-to-predict code) — the model lives in the App, not in UC |
| `ask_genie(question)` | Genie, managed MCP | Genie's answer + its SQL + rows; last resort for open questions (~18 s, UI shows progress) |

**The turn loop (App, `app/chat.py`):** Maverick with the tool schemas → run the requested tools
→ feed results back → at most 3 rounds → final answer. Guards, from the earlier arcs:
- A **text-only "tool call"** (the DATA-2a failure) is detected and rejected, not shown.
- **Grounding:** every number, `HH:MM` time and callsign in the answer must appear in this turn's
  tool results (generalising `lib.briefing.check_briefing`). On failure: one repair round ("your
  answer used X, which is not in the tool results"), then a plain fallback that shows the tool
  rows themselves.
- **Data age** is shown under the answer from the tools' `scored_at`, not from the model's text.
- **"How I answered"** expander: tools called, arguments, rows.
- **Out of scope** ("what's the weather in Paris?", "book me a flight"): a short refusal that says
  what SkyWatch *can* answer.

**Audit (CHAT-EVAL-1):** one `chat_audit` row per turn — question, tools + args, answer, check
result and reasons, rounds, tokens, latency per stage, `prompt_version`, 👍/👎. Needs the App SP to
`INSERT` into one table (grant to verify).

**Evaluation (CHAT-EVAL-1, written before the prompt is tuned):** `tests/chat_golden.yaml` —
questions with the tools they should call and what the answer must (or must not) contain, run by
`scripts/eval_chat.py` against the live endpoint like `eval_briefing.py`: tool-selection
accuracy, grounding pass rate, refusal correctness, rounds, latency p50/p95, tokens.

## 4. Permissions to verify (App service principal)

`USE SCHEMA` + `EXECUTE` on `skywatch.chat`; a `serving_endpoint` resource with `CAN_QUERY` on
the chat endpoint; access to the Genie space (a `genie_space` App resource); `MODIFY` on
`chat_audit`. Each is checked live before the code that needs it is written.

## 5. Order of work

1. `src/chat_tools.sql` + setup job; call each function live.
2. `tests/chat_golden.yaml` + `scripts/eval_chat.py` (tool loop runnable locally, before the UI).
3. Grounding for tool results (`lib/chat.py`, unit tested).
4. App SP grants + App resources, verified live.
5. `app/chat.py` page: loop, guards, "how I answered", data-age label, audit, 👍/👎.
6. Golden-set run → tune → numbers in this doc. DATA-2b ("Get SkyWatch ready") last.

## 6. Progress log

- **2026-10-19 — arc started.** Live checks in §2. Branch `chat/v1`.
