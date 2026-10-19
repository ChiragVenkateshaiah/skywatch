---
description: Start the SkyWatch working day — close any gap since the last checkpoint, snapshot the state, and set today's plan in the daily log
argument-hint: "[optional note or focus for today]"
allowed-tools: Bash(git:*), Bash(gh pr list:*), Bash(gh pr view:*), Bash(gh pr checks:*), Bash(date:*), Bash(databricks auth token:*), Bash(databricks warehouses get:*), Bash(databricks apps get:*), Bash(databricks jobs list-runs:*), Read, Edit, Write
---

Start the working day. The daily log is the chain of checkpoints between `/end-day` and
`/start-day`; there must be **no gap** in it. Follow these steps in order.

## The log

- File `DAILY_LOG.md` on the dedicated branch `daily-log`, checked out as a git worktree at
  `.worktrees/daily-log` (never in a feature branch, so it can't scatter or end up in a PR).
- Setup, only if the worktree is missing:
  - `git fetch origin daily-log` — if the branch exists on origin:
    `git worktree add .worktrees/daily-log daily-log`
  - otherwise create it: `git worktree add --orphan -b daily-log .worktrees/daily-log`, write a
    `# SkyWatch — daily log` header, commit, `git -C .worktrees/daily-log push -u origin daily-log`.
  - Make sure `.worktrees/` is in `.git/info/exclude`.
- Otherwise: `git -C .worktrees/daily-log pull --ff-only`.
- Every entry starts with a machine-readable marker line, then a heading:
  `<!-- checkpoint type=START|END ts=<UTC ISO, e.g. 2026-10-20T08:12Z> -->`
  `## 2026-10-20 · START · 08:12 UTC`
  Append only; never edit or delete earlier entries.

## Steps

1. **Time:** `date -u +%Y-%m-%dT%H:%MZ` (and the local time, for the heading's context).

2. **Gap check** — read the **last** marker in the log:
   - `type=END` → the chain is intact. Its `ts` is the base for step 3.
   - `type=START` (a day with no `/end-day`) → first append a **reconstructed END** for that day,
     built only from evidence — commits on all branches and PRs between that START's `ts` and
     now (step 3 commands) — headed `## <that date> · END (reconstructed by /start-day) · <now>`,
     with a line that it is reconstructed and anything that can't be known from git/GitHub
     (deploys, live state) marked *unknown*. Its marker's `ts` is the START's own date's last
     commit time if there was one, else the START's `ts`. Then continue as if it were an END.
   - No log entries yet → this is the first entry; use the newest checkpoint in
     `docs/CAPABILITIES_ROADMAP.md` ("Where we stopped") as the previous state.

3. **Since the last entry** (base `ts`), on every branch:
   - `git fetch --all --prune -q`
   - `git log --all --since=<base ts> --format='%h %ad %D %s' --date=format:'%Y-%m-%d %H:%M'`
   - `gh pr list --state all --search "updated:>=<base date>" --json number,title,state,mergedAt,headRefName`
   If anything happened after the last END — work done without a `/start-day` — it goes into
   today's START entry under **Between days**, so nothing is lost.

4. **State snapshot — read-only, no compute.** Do **not** run SQL, jobs, evaluations, or start the
   warehouse / App (the Free Edition quota stopped work on 2026-10-19):
   - git: current branch, `git status --short`, ahead/behind origin, branches with unmerged commits
   - `gh pr list --state open` (+ `gh pr checks <n>` for each)
   - `databricks auth token --profile skywatch` — OK or expired (expired → tell the user to run
     `! databricks auth login --profile skywatch`; continue with what doesn't need it)
   - `databricks warehouses get 8fd481cbf45ac93e --profile skywatch` → state only
   - `databricks apps get skywatch-arrival-manager --profile skywatch` → app + compute state
   - optionally `databricks jobs list-runs` for `skywatch_demo_refresh` (limit 1) → last refresh

5. **Today's plan:** the **Next** list of the last END (carried over in its order), then the
   user's note if any: $ARGUMENTS

6. **Append the START entry:**
   ```
   <!-- checkpoint type=START ts=<now> -->
   ## <date> · START · <HH:MM> UTC
   - **Previous checkpoint:** <date> END <HH:MM> UTC ✓   (or: reconstructed END added above / first entry)
   - **Between days:** <commits and PRs after the last END, or "nothing">
   - **State:** branch …; uncommitted …; open PRs …; CLI auth …; warehouse …; App …; last refresh …
   - **Blockers carried over:** …
   - **Plan for today:** 1. … 2. …
   ```
   Commit only the log: `git -C .worktrees/daily-log add DAILY_LOG.md && git -C .worktrees/daily-log commit -m "log: <date> START"`
   and push it (`git -C .worktrees/daily-log push`). Never touch feature branches here.

7. **Reply** to the user with the plan for today, the gap-check result, and anything that needs
   their action first (expired login, warehouse still blocked, failing CI).
