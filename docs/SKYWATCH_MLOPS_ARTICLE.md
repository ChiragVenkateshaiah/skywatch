# Three Models Aren't a Product

*The eight-track arc that turned a working demo into a system I'd trust unattended — gates, an isolated environment, monitoring, rollback, all verified against a live Databricks workspace, not designed on paper.*

**3 models · 8 MLOps tracks · 72 unit tests · 2 environments · ~9 min live retrain**

---

A few weeks ago I posted three working ML models — a time-to-touchdown predictor, a demand forecaster, a rule-based irregularity detector — running live against real ADS-B flight-transponder data. It felt like the finish line. It wasn't. It was a demo with good instincts: every retrain silently overwrote what was in production if it beat a weak baseline, there was nowhere to test a change without touching the live app, and if something went wrong at 2am, nothing would tell me and nothing would tell me how to undo it.

"Production" turned out to mean eight distinct pieces of discipline, each one forced into existence by something that broke the moment I actually tried it — not by reading about best practice. This is the record of what those eight pieces were, what they cost to build on a free-tier Databricks workspace with no serving endpoints and no paid monitoring, and the specific bugs that only showed up once real infrastructure was involved.

## Track 4 — The Gate

This is the piece everything else exists to protect, so it goes first. Before this track, `train_eta.py` registered a new model and, in the same breath, decided whether to put it in front of users — the bar was simply "beats a naive distance-over-speed baseline," which had nothing to do with whether it was better than whatever was already live. There was no comparison to production at all. That's not a gate. That's a coin flip with a diploma.

I pulled the decision apart from the training. Now a training run only ever produces a *challenger* — a candidate, registered but inert. A separate notebook, `promote_eta.py`, loads the challenger and the current champion, scores both on the same held-out data, and compares them by distance band (arrivals 0–20nm out matter more to a coordinator than 70–100nm out, so an overall win that hides a close-in regression should still fail):

```python
# the decision itself — pure Python, unit-tested, no Spark or MLflow
def evaluate_gate(challenger_bands, champion_bands,
                  min_improvement_pct, noise_band_pct, max_band_regression):
    # PASS if the challenger's overall error improves enough, OR is within
    # a noise band of the champion — AND no individual band got worse than
    # the champion by more than the allowed tolerance. A big overall win
    # can't hide a regression in the band that matters most.
```

And the part I care about most: **a passing gate is a recommendation, never an instruction.** Both promotion notebooks default to a dry run. The only way a model actually goes live is a second, separate, deliberate command with `apply=true` — and that command only ever does anything if the gate *also* passed. There's no path from "trained" to "shipped" that doesn't go through a human reading a report first.

**Proving it both directions.** I didn't trust this until I'd watched it fail correctly. First, on Model 2 (the demand forecaster), I pointed the challenger alias at an older, genuinely worse version and ran the gate against real production data:

> **promote_demand.py — live run against real registry data**
> - Challenger v2 vs. champion v3: MASE regression, real
> - Gate verdict: **HOLD**
> - `@champion` after the run: unchanged — v3

It refused. No override, no special case — the same code path a legitimate improvement runs through simply declined to promote a regression.

Then the real thing happened. A scheduled retrain of Model 1 produced a challenger that was genuinely, if narrowly, better than what was live:

> **promote_eta.py — the first real production promotion**
> - Champion (before), v7: MAE 1.144 min
> - Challenger, v9: MAE 1.139 min
> - Gate verdict: **PASS**
> - Applied: yes — by a second, explicit command

*A machine recommends. A person still decides.*

## Tracks 1–3 — Tests, CI, and Somewhere Safe to Fail

None of the gate matters if the code around it is untested or if there's only one environment to run it in. Seventy-two `pytest` cases now cover the gate math, the feature transforms, and a PSI drift function — pure Python, no Spark, so they run in under thirty seconds. Every pull request runs those tests, `ruff`, and a real `bundle validate` against live Databricks credentials before it's mergeable. Merges to `main` auto-deploy to a second environment.

Building that second environment is where the platform pushed back hardest, and where the story got more interesting than "add a staging slot":

**Databricks Asset Bundles can't exclude a resource from one target.** I confirmed this by testing it, not by reading a limitation list — setting a resource to `null` under a scratch target validated cleanly, but the resolved deployment plan still had it, completely intact. Combined with Free Edition's hard cap of one active pipeline per workspace, that ruled out sharing one bundle definition between the live environment and a new one: deploying a second target would try to spin up a second pipeline and simply fail. The fix was a second, physically separate bundle — not a third target — sharing the same source code via a `sync.paths` reference, so both environments run identical logic on isolated data.

**Apps enforce a stricter identity rule than jobs.** Jobs support real impersonation: you can deploy as yourself and still tell a job to run as a service principal. Databricks Apps don't — the deploying identity and the app's `run_as` identity have to match exactly, confirmed by watching it reject a mismatch outright. Which meant the CI service principal actually had to authenticate and deploy the new environment itself, not have a human fake it with a flag.

Two smaller landmines only surfaced on a real deploy attempt: app names cap at 30 characters (the descriptive name I wanted was 32 — Databricks doesn't warn, it just rejects), and a bundle deployed by a service principal needs a shared workspace path, not a personal folder the SP has no access to.

## Track 5 — Monitoring That Told the Truth About Itself

Lakehouse Monitoring isn't available on Free Edition, so the substitute is a scheduled notebook computing Population Stability Index against each feature, rolling error against the registered baseline, and a freshness/volume check against the poller's own bursty collection pattern — three signals into one `model_health` table, a dashboard tile, and a SQL alert.

The first real run threw **seventeen "significant" drift flags.**

Every one was noise. The comparison population was 148 rows spread across ten statistical bins — roughly fifteen samples per bin, nowhere near enough for a stable PSI reading. The detector was working exactly as coded; the code just hadn't accounted for how little live data a bursty, on-demand poller actually accumulates. I raised the minimum sample size for both the drift and performance checks by an order of magnitude, and re-ran it: the honest answer became "insufficient data," not seventeen false alarms.

That felt like the more important engineering decision than building the detector in the first place — recognizing that a system's own output wasn't trustworthy yet, and fixing the threshold instead of shipping something that would cry wolf on every run.

Two more bugs, both found by actually running the thing rather than reading the code: a reference query pulled raw columns straight from the training table, missing several features that only exist after a transform step — a silent `KeyError` the moment real data hit it. And a freshness check that filtered to "today" specifically, which meant it reported nothing at all the day the poller didn't run, instead of correctly saying "227 hours stale."

## Tracks 6–7 — Retraining, Release, and a Rehearsed Rollback

A single Databricks Job now chains a full data rebuild, both models' training, and both promotion gates — five tasks, dry-run gates by design, so an unattended weekly schedule can never silently ship anything. Verified end to end against the real workspace: **8 minutes 42 seconds**, every task green, both gates producing real verdicts on real new candidates.

Release management got the same "prove it, don't just write it" treatment. A tag push now extracts that version's changelog section and publishes a GitHub Release automatically — no deploy step, since the live environment stays a deliberate, human-run deploy on purpose. And rollback isn't a paragraph in a document I hope holds up during an incident; I rehearsed it, live, twice — rolled a model back to a prior version, confirmed the registry reflected it, restored the correct one:

```bash
# the entire rollback mechanism, in practice
databricks api put .../aliases/champion --json '{"version_num": 7}'
# confirm, then, if it was a rehearsal:
databricks api put .../aliases/champion --json '{"version_num": 9}'
```

It's deliberately this simple. An incident response that has to wait on a gate isn't a rollback.

## Track 8 — An Audit Trail, Not a Memory

The last piece was governance: runbooks written from the exact commands used throughout this arc rather than idealized ones, model cards for all three models including their honest failure modes, an architecture reference — and one real table. Every promotion decision, gate pass or hold, applied or not, now writes itself automatically:

> **`skywatch.ml.promotion_audit` — one row, every run**
> - Model: eta_touchdown
> - Challenger → prior champion: v10 → v9
> - Verdict: **PASS**
> - Applied: false

"Why is this the model that's live" is a query away now, not something I have to remember or reconstruct from job logs.

## The Whole Thing, End to End

None of the eight tracks changed what the models predict. All of them changed whether I'd trust the system running unattended — which is the actual bar for "production," not "it works when I personally run it." And the honest pattern across every track was the same one:

1. A platform constraint (**no per-target resource exclusion**) — found by testing it, not reading docs
2. A silent **`KeyError`** in the drift monitor — invisible until real data hit an untransformed column
3. A freshness check that returned **nothing instead of "stale"** — a query bug only real gaps in data exposed
4. **Seventeen false drift alerts** from a genuinely correct detector fed too little data
5. A CI run that **never fired at all** — zero explanation available even from the platform's own status history
6. Two dashboard regressions from an **AI authoring tool overwriting its own prior output**

Every one of those was found by actually running things against real infrastructure — not by design review, not by reading the plan back to myself and nodding. Production discipline, it turns out, isn't mostly about writing more code. It's refusing to trust anything you haven't personally watched fail and recover.

> A model in a notebook is a hypothesis. A model with a gate, a rollback, and an audit trail is a system.

---

SkyWatch runs entirely on Databricks Free Edition — no paid serving endpoints, no paid monitoring, no infrastructure outside a free-tier workspace and GitHub Actions' free minutes.

**Repo:** https://github.com/ChiragVenkateshaiah/skywatch
**Full MLOps log:** https://github.com/ChiragVenkateshaiah/skywatch/blob/main/docs/MLOPS_PLAN.md
**Releases:** https://github.com/ChiragVenkateshaiah/skywatch/releases

#MLOps #MachineLearning #Databricks #DataEngineering #ProductionML
