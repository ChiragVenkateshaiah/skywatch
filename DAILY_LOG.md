# SkyWatch — daily log

The chain of working-day checkpoints, written by `/start-day` and `/end-day`
(`.claude/commands/`). Append only. Each entry starts with a machine-readable marker; there must be
no gap: every START follows an END, and anything done between them is recorded in the next START.

<!-- checkpoint type=END ts=2026-10-19T17:46Z -->
## 2026-10-19 · END · 17:46 UTC

- **Since:** first entry of the log — no START recorded. Previous checkpoint:
  `docs/CAPABILITIES_ROADMAP.md` §8 ("Where we stopped").
- **Done (merged to `main`):**
  - #45 pre-interview refresh job (DATA-1) — READY in 8.5 min; parallel serverless starts waited
    645–705 s, so tasks run in sequence
  - #46 "Arrivals seen today" relabel
  - #44 capabilities roadmap
  - #47 local ops agent `scripts/demo_ready.py` (DATA-2a) — the agent once invented an error;
    evidence-in-prompt + quote check fixed it
  - #48 Chat with SkyWatch (CHAT-1, App v2.0) — 9 UC tool functions, grounding loop, golden set
    38/37/38 of 38, SP self-check 14/14, Get SkyWatch ready, demo warm-up + `docs/DEMO_SCRIPT.md`
  - #49 live snapshot (OPT-1) — chat p50 5.2 s → 2.1–2.6 s at 38/38; `docs/LESSONS.md` (DOC-1)
  - #50 Streamlit pinned to the App's 1.38.0 — fixed the chat page crash (`width=` TypeError)
- **In progress:**
  - `app/minimal-dashboard` (APP-1) — pushed, **not merged, not deployed**: the one-screen
    dashboard (freshness pill, 4 KPIs, briefing card, one map with alert rings, one chart, "Ask
    SkyWatch" buttons → chat). Tested headless on Streamlit 1.38.0; the dashboard → chat hand-over
    reached the chat but failed on the warehouse, so it is still unverified.
  - `chore/day-commands` — `/start-day` and `/end-day` (this log). PR to be opened.
- **Decisions:** keep history and reset only the live picture before interviews; recorded demo;
  Maverick primary / Llama 3.3 70B fallback; three-run ≥ 90% ship rule; refresh is a human action
  (the model only reads `refresh_status`); minimal dashboard keeps the map and drops
  click-to-predict.
- **Blockers / risks:** the SQL warehouse won't start — "Cannot create the resource, please try
  again later" (user: Free Edition quota). The App's compute is now STOPPED too. Golden-set runs
  were the biggest compute cost today (~15 runs) → roadmap OPT-4.
- **Live state:** App UNAVAILABLE / compute STOPPED; warehouse STOPPED (won't start); schedules
  PAUSED; CLI auth OK.
- **Next:**
  1. Check the warehouse state read-only (`databricks warehouses get 8fd481cbf45ac93e`); only if
     it can run, one `SELECT 1`. Start the App (`bundle run skywatch_arrival_manager`) once it does.
  2. Branch `app/minimal-dashboard`: re-run the hand-over test (chat page with `pending_question`
     preset → one checked turn; test with Streamlit 1.38.0, `PYTHONPATH=app`).
  3. Deploy; `apps logs … | grep SELFCHECK` (expect 14/14); open the App once in a browser —
     dashboard, one Ask button, the chat answer. Then open + merge the APP-1 PR.
  4. Merge the `chore/day-commands` PR if not done.
  5. OPT-4 (quota-aware evaluation) before any further golden-set run; then CHAT-2.

<!-- checkpoint type=START ts=2026-10-21T06:09Z -->
## 2026-10-21 · START · 06:09 UTC

- **Previous checkpoint:** 2026-10-19 END 17:46 UTC ✓
- **Between days:** #51 `chore/day-commands` (`/start-day`, `/end-day`) merged to `main` at
  2026-10-19 17:49 UTC. No other commits or PRs since.
- **State:** branch `main`, in sync with origin; no uncommitted changes; no open PRs;
  `app/minimal-dashboard` has 1 unmerged commit (pushed, no PR); CLI auth OK; warehouse STOPPED;
  App UNAVAILABLE / compute STOPPED; last refresh (`skywatch-demo-refresh`) SUCCESS, started
  2026-10-19 14:52 UTC.
- **Blockers carried over:** the SQL warehouse would not start on 2026-10-19 (Free Edition quota)
  — not re-tested yet; the App depends on it. Golden-set runs need OPT-4 first.
- **Plan for today:**
  1. Check the warehouse state read-only; only if it can run, one `SELECT 1`. Start the App
     (`bundle run skywatch_arrival_manager`) once it does.
  2. Branch `app/minimal-dashboard`: re-run the hand-over test (chat page with
     `pending_question` preset → one checked turn; Streamlit 1.38.0, `PYTHONPATH=app`).
  3. Deploy; `apps logs … | grep SELFCHECK` (expect 14/14); open the App once in a browser —
     dashboard, one Ask button, the chat answer. Then open + merge the APP-1 PR.
  4. ~~Merge the `chore/day-commands` PR~~ — done (#51).
  5. OPT-4 (quota-aware evaluation) before any further golden-set run; then CHAT-2.
