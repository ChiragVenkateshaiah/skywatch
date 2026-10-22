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

<!-- checkpoint type=END ts=2026-10-21T09:12Z -->
## 2026-10-21 · END · 09:12 UTC

- **Since:** START 06:09 UTC
- **Done (merged to `main`):**
  - #52 minimal one-screen dashboard with Ask SkyWatch hand-over (APP-1, App v2.2)
  - #53 quota-aware golden-set eval (OPT-4) — skips unchanged full runs (`--force`), caches tool
    results per data version (`--fresh-tools`), prints tool SQL / cache hits / model calls
  - #54 `search_docs` — the chat answers how/why from the docs, cited and checked (CHAT-2)
  - #55 roadmap: CHAT-2 marked done
- **From the session (not in git):**
  - Warehouse started again (user ran `warehouses start`; RUNNING, `SELECT 1` ok) — the
    2026-10-19 quota blocker is gone. App restarted (user ran `bundle run`).
  - APP-1 hand-over test passed in two halves (Streamlit 1.38.0); deployed 06:30, SELFCHECK
    14/14; user checked dashboard → Ask → chat answer in a browser.
  - OPT-4 live check, one item twice: tool SQL 7 → 1 on frozen data.
  - CHAT-2: index built with `scripts/index_docs.py` — 207 passages in `skywatch.chat.doc_chunks`.
    Hit the embedding QPS limit (batch 16 → REQUEST_LIMIT_EXCEEDED) → batches of 4 + pause.
    A 256-word passage measured 550 tokens (gte reads 512) → a 1,500-char cap per passage.
    `GRANT SELECT ON TABLE skywatch.chat.doc_chunks` to the App SP. Prompt `chat-v2.2.1`.
  - Golden set 44 items, three runs: 43/44 (44/44 rescored), 43/44, 42/44 — ship rule met; the
    38 CHAT-1 items 38/38 in all three; ~210 model calls, 9 tool SQL calls (cache).
  - Deployed 08:01 from `chat/knowledge`: SELFCHECK 15/15 (new `docs` line: 207 passages).
    User checked "How does Model 1 predict landing times?" in a browser — cited
    `[ML_ROADMAP.md §Model 1 — Time-to-touchdown (ETA)]`.
- **In progress:** nothing. No branch with commits outside `main`; no open PRs; `main` clean.
  Uncommitted: none.
- **Decisions:**
  - CHAT-2 index: a local script → Delta, not a bundle job (the bundle doesn't sync `*.md`;
    serverless job starts waited ~11 min).
  - CHAT-2 corpus: README + 12 docs; without `DEMO_SCRIPT.md` and `SKYWATCH_MLOPS_ARTICLE.md`.
  - Golden items doc-02 / doc-03 / doc-05 widened where the answers were right (doc-05 had
    required a key name the grounding check forbids).
- **Blockers / risks:**
  - Data is still as of 2026-10-19 15:01 UTC (last refresh SUCCESS 2026-10-19 14:52) — run the
    refresh before any demo.
  - The App's compute is still ACTIVE (it uses quota while it runs).
  - doc-05 / doc-06 are flaky: the model sometimes skips search_docs or doesn't cite — the guard
    then falls back to the tool rows.
  - `index_docs.py` must be re-run after a doc change in the corpus, then the App restarted.
- **Live state:** App RUNNING / compute ACTIVE (deployment `01f1cd25b5e5`, 08:02 UTC);
  warehouse STOPPED (auto-stop); refresh job has no schedule (other schedules not re-checked
  today — PAUSED on 2026-10-19); CLI auth OK (token expiry 10:13 UTC, refreshes on use).
- **Next:**
  1. Decide whether to stop the App's compute to save quota until the next demo
     (`databricks apps stop skywatch-arrival-manager --profile skywatch`).
  2. Before a demo: run the refresh (App → *Get SkyWatch ready*, or `scripts/demo_ready.py`),
     then check the freshness pill.
  3. Pick the next roadmap item: CHAT-3 (Genie v2) waits for its trigger (`ask_genie` share
     ≥ 20% or 👎 rate ≥ 2×); else OPT-2 or the doc-05 / doc-06 tool-choice weakness.
  4. Golden-set runs only after a prompt / tool / loop change; OPT-4 skips the rest.

<!-- checkpoint type=START ts=2026-10-22T05:02Z -->
## 2026-10-22 · START · 05:02 UTC

- **Previous checkpoint:** 2026-10-21 END 09:12 UTC ✓
- **Between days:**
  - App stopped by the user at 2026-10-21 09:39 UTC (`databricks apps stop`) — the END entry
    still said RUNNING / ACTIVE.
  - OPT-2 built on `opt/tracing` (`e5cfe32`): one MLflow trace per chat turn (App, tagged with
    the `chat_audit` turn_id) and per refresh run (`demo_ready.py`, real timestamps, no compute);
    bundle experiment `skywatch-traces` + App resource `traces-experiment` (CAN_EDIT). Live
    tests without warehouse / model compute into a probe experiment (deleted afterwards). Same
    PR fixed stale roadmap rows (DATA-1, CHAT-EVAL-1 done; CHAT-3 trigger not measured),
    re-ranked after CHAT-2 (no order change), §8 rewritten as the 2026-10-21 checkpoint.
  - #56 merged 2026-10-22 05:01 UTC.
- **State:** branch `main`, in sync with origin, clean; no open PRs; no unmerged feature
  branches; CLI auth OK; warehouse STOPPED; App UNAVAILABLE / compute STOPPED (still holds the
  CHAT-2 code — OPT-2 not deployed); last refresh SUCCESS 2026-10-19 14:52 UTC.
- **Blockers carried over:** data as of 2026-10-19 15:01 UTC; OPT-2 unverified in the App;
  doc-05 / doc-06 flaky (search_docs skipped or not cited).
- **Plan for today:**
  1. ~~Decide whether to stop the App's compute~~ — done (stopped 2026-10-21 09:39 UTC).
  2. Before a demo: run the refresh, then check the freshness pill.
  3. With the next App start: `bundle deploy` + `bundle run`, one chat turn, find its trace in
     `skywatch-traces` by `turn_id` (`docs/TRACING_PLAN.md` §4 step 4).
  4. Next roadmap item: CHAT-3 waits for its trigger (measure from `chat_audit`); else the
     doc-05 / doc-06 tool-choice weakness, using the new traces.
  5. Golden-set runs only after a prompt / tool / loop change; OPT-4 skips the rest.
