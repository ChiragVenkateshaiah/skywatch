# SkyWatch — Runbooks

Every procedure here is the exact sequence actually run and verified live during this
project's MLOps arc (`docs/MLOPS_PLAN.md`) — not a theoretical write-up. Where a step has a
known gotcha, it's noted inline (these cost real debugging time once each; they shouldn't cost
it again).

---

## Deploy

**`dev`** — the live environment. Always manual, always from a human's machine (never CI —
see `docs/MLOPS_PLAN.md` Track 2/3 for why: `dev` is what the demo/LinkedIn post point at, and
CI auto-deploying to it risked silently breaking something no one reviewed).

```bash
databricks bundle validate -t dev --profile skywatch
databricks bundle deploy   -t dev --profile skywatch
```

Multiple profiles matching the same host (`skywatch`, `skywatch-ci`) make the CLI refuse to
guess — always pass `--profile skywatch` for `dev` work.

If only a job's *parameters* changed (a bundle variable), `bundle deploy` updates the job
definition, but **`bundle run --var=...` does not override parameters at run time** — only
`bundle deploy --var=...` bakes a value in. This bit us twice (Track 4's `apply=true`
verification, Track 6's `eta_max_evals` test) before it was written down here.

**`staging`** — deploys automatically on every merge to `main` (`.github/workflows/
deploy-staging.yml`), authenticated as the `skywatch-ci` service principal. A manual deploy
looks like:

```bash
cd serving_staging
databricks bundle deploy -t staging --var="staging_run_as=<skywatch-ci application id>" \
  --profile skywatch-ci
```

Two gotchas specific to this bundle, found only by actually deploying (invisible to
`validate`): `root_path` must be a shared path (`/Workspace/Shared/...`), not a personal
`/Workspace/Users/<human>/` folder — the deploying SP can't write there. And Databricks App
names cap at 30 characters (`skywatch-arrmgr-staging`, not the longer literal name).

**After any deploy that touches the App**, source code still needs a separate push:

```bash
databricks apps deploy skywatch-arrival-manager \
  --source-code-path /Workspace/Users/<you>/.bundle/skywatch-lite/dev/files/app
```

`bundle deploy` updates the App *resource* (warehouse binding, env vars); this pushes the
*code*. Both are needed after an `app/` change.

---

## Rollback

**Code**: find the last good tag (`git tag -l`, or the GitHub Releases list), then
`git revert` the bad commit(s) — preferred over a hard reset, since it keeps history honest —
and redeploy exactly as in the Deploy section above. There is no automated rollback path; a
bad `dev` deploy is fixed the same manual way a good one is made.

**Model** (`@champion` alias) — practiced live during Track 7, not just written down:

```bash
databricks api get "/api/2.1/unity-catalog/models/<catalog>.ml.<model>/aliases/champion" \
  --profile skywatch
# ... note the current version, then:
databricks api put "/api/2.1/unity-catalog/models/<catalog>.ml.<model>/aliases/champion" \
  --json '{"version_num": <prior_good_version>}' --profile skywatch
```

After a `@champion` change, `skywatch_score_eta` (or `skywatch_score_demand`) needs to run
once for the live predictions table *and* the App's click-to-predict Volume export (only
refreshed on a scoring run) to actually reflect the new champion:

```bash
databricks bundle run skywatch_score_eta -t dev --profile skywatch
```

Every promote job run — pass or hold, applied or not — is recorded in
`<catalog>.ml.promotion_audit` (Track 8), so "what was `@champion` before, and why did it
change" is always answerable without digging through job run history.

---

## Incident

There hasn't been a real production incident yet — this is the shape a response takes,
built from the actual bugs found and fixed live during this arc (a `KeyError` in `monitor.py`
from reading un-transformed columns, a freshness query silently returning `NULL` instead of a
real staleness signal, a missed CI trigger with zero explanation available from GitHub). None
of those were "true" incidents (nothing was ever live-broken for an end user), but the
diagnostic order below is exactly how each was actually found:

1. **Check `skywatch.ml.model_health`** (Track 5) for a recent `significant` row — drift,
   performance regression, or a stale/low-volume feed. This is the fastest signal for
   "something about the models or the data is off."
2. **Check the failing job's run page directly** (`bundle run` prints the Run URL; `jobs
   get-run` / `get-run-output` via the API otherwise) — read the actual traceback, don't guess
   from the job name. Every real bug in this project was found this way, not by inspection.
3. **Check `skywatch.ml.promotion_audit`** if the question is model-version-shaped ("did we
   promote something bad?").
4. **Check the App directly** (`databricks apps get <name>` for `RUNNING`/`ACTIVE`, then load
   the URL) if the symptom is user-facing.
5. **Fix, verify the fix against the real live thing** (not just that it validates or looks
   right in code review), *then* write down what happened — in this file if it's a new class
   of failure, or as a one-off note in the relevant PR if it's already covered here.

---

## Retrain

`skywatch_retrain` (Track 6) chains `gold_full` → `{train_eta, forecast_demand}` →
`{promote_eta, promote_demand}` — both gates always dry-run (`apply=false` is hardcoded in the
job, not a variable — a retrain workflow that could auto-promote would defeat the entire
point of Track 4).

```bash
databricks bundle run skywatch_retrain -t dev --profile skywatch
```

Takes ~9 minutes end to end (verified live). After it finishes:

1. Read the `gate_verdict` on the new challenger version(s):
   ```bash
   databricks api get "/api/2.1/unity-catalog/models/<catalog>.ml.<model>/versions" \
     --profile skywatch
   ```
2. **Decide** — this is the human step the whole gate design exists to protect. A `PASS`
   verdict is a recommendation, not an instruction.
3. **If promoting**: deploy with `apply=true` baked in (not `bundle run --var`, see the Deploy
   section), run the relevant `promote_*` job, confirm the alias flipped, then redeploy the
   safe `apply=false` default back so a future accidental run can't auto-promote.
4. **Refresh serving state**: run `skywatch_score_eta` / `skywatch_score_demand` so live
   predictions and click-to-predict pick up the new champion (see Rollback above — same step).

The weekly schedule exists but is deployed **PAUSED** — an unattended full gold rebuild plus
two full trainings every week would eventually burn a day's Free Edition quota for nothing if
no new data had landed. The other two triggers (a Track 5 drift alert, new backfill days
landed) are both the same manual command above — by design, a human decides when to act on a
signal, not a fully automated closed loop.
