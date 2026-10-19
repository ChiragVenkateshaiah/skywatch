---
description: End the SkyWatch working day — record everything since today's start, what's in progress, and the next steps the next /start-day carries over
argument-hint: "[commit] [optional note]  — 'commit' also commits+pushes uncommitted work on the current feature branch"
allowed-tools: Bash(git:*), Bash(gh pr list:*), Bash(gh pr view:*), Bash(gh pr checks:*), Bash(date:*), Bash(databricks auth token:*), Bash(databricks warehouses get:*), Bash(databricks apps get:*), Bash(databricks jobs list-runs:*), Read, Edit, Write
---

End the working day. Write the END checkpoint the next `/start-day` resumes from, so there is
**no gap** in the daily log. Same log as `/start-day`: `DAILY_LOG.md` on branch `daily-log`,
worktree `.worktrees/daily-log` (set it up as `/start-day` describes if it's missing). Append only.

## Steps

1. **Time:** `date -u +%Y-%m-%dT%H:%MZ`. `git -C .worktrees/daily-log pull --ff-only`.

2. **Find today's base** — the last marker in the log:
   - `type=START` → normal; its `ts` is the base.
   - `type=END` (no `/start-day` since) → base = that END's `ts`; say in the entry that no START
     was recorded, so the chain still has no gap.

3. **Everything since the base**, on every branch, from evidence:
   - `git fetch --all --prune -q`
   - `git log --all --since=<base ts> --format='%h %ad %D %s' --date=format:'%Y-%m-%d %H:%M'`
   - `gh pr list --state all --search "updated:>=<base date>" --json number,title,state,mergedAt,headRefName`
   - What git can't show — deploys, jobs run, measurements, decisions the user made, problems hit —
     comes from this session's conversation. Say so where it matters; never invent.

4. **In progress:** branches with commits not on `main` (`git branch -vv`, `git log main..<branch>`),
   open PRs and their CI, and `git status --short` on the current branch.
   - Uncommitted changes: list them. Commit them **only** if the arguments include `commit`
     ($ARGUMENTS) — then commit on the current branch with a `wip:` message and push it; **never
     commit on `main`** (if on `main`, stop and ask). Without `commit`, leave them and say so in
     the entry, because uncommitted work is lost on another machine.

5. **Live state — read-only, no compute** (no SQL, jobs, evaluations, warehouse/App starts):
   `databricks auth token --profile skywatch` (expiry), `databricks warehouses get
   8fd481cbf45ac93e --profile skywatch`, `databricks apps get skywatch-arrival-manager --profile
   skywatch`, and the last `skywatch_demo_refresh` run if useful.

6. **Next steps:** an ordered list, concrete enough to start cold tomorrow (what to check first,
   which branch, which PR, which command). Add the user's note if any: $ARGUMENTS

7. **Append the END entry:**
   ```
   <!-- checkpoint type=END ts=<now> -->
   ## <date> · END · <HH:MM> UTC
   - **Since:** START <HH:MM> UTC   (or: no START recorded — since END <date> <HH:MM>)
   - **Done:** merged PRs (#n title), commits by branch, deploys / runs / measurements from the session
   - **In progress:** branch … (pushed? PR?); uncommitted: … (committed by /end-day? yes/no)
   - **Decisions:** …
   - **Blockers / risks:** …
   - **Live state:** App …; warehouse …; schedules …; CLI auth …
   - **Next:** 1. … 2. … 3. …
   ```
   `git -C .worktrees/daily-log add DAILY_LOG.md && git -C .worktrees/daily-log commit -m "log: <date> END"`
   and `git -C .worktrees/daily-log push`.

8. If `docs/CAPABILITIES_ROADMAP.md` has a "Where we stopped" section that is now out of date,
   tell the user (don't edit it from here — it belongs to the feature branch that changes it).

9. **Reply** with a short summary: done today, in progress, and tomorrow's first step.
