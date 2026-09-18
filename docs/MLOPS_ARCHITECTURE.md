# SkyWatch — MLOps Architecture

The end-to-end picture of `docs/MLOPS_PLAN.md`'s eight tracks, once they're actually built and
verified live rather than just planned. See that document for the track-by-track history and
the real bugs/gotchas found along the way; this is the synthesized reference.

## The whole system

```mermaid
flowchart TD
    subgraph GH[" GitHub — CI/CD (Track 2) "]
        PR["PR opened<br/>pytest + ruff + validate both bundles"]
        MERGE["merge to main<br/>auto-deploy serving_staging"]
        TAG["git tag v*<br/>release.yml -> GitHub Release"]
    end

    SP[("skywatch-ci service principal<br/>OAuth M2M — shared by CI + staging")]

    subgraph DEV[" Root bundle — dev (Track 3, live, manual deploy only) "]
        POLLER["poller + Lakeflow medallion<br/>+ gold (full/serving)"]
        DEVTRAIN["train_eta.py / forecast_demand.py<br/>register @challenger"]
        DEVGATE["promote_eta.py / promote_demand.py<br/>dry run by default"]
        DEVAPP["Databricks App + AI/BI dashboard<br/>skywatch-arrival-manager"]
    end

    subgraph STAGING[" serving_staging bundle — staging (Track 3, CI-deployed) "]
        STGTRAIN["train_eta.py / forecast_demand.py<br/>read prod Gold, read-only"]
        STGGATE["promote_eta.py / promote_demand.py<br/>write skywatch_staging.ml"]
        STGAPP["Databricks App<br/>skywatch-arrmgr-staging"]
    end

    subgraph GATE[" Promotion gate (Track 4) "]
        EVALGATE["src/lib/promotion.py::evaluate_gate<br/>MAE/MASE by band, unit-tested"]
        AUDIT[("promotion_audit table<br/>Track 8 — every run, pass/hold, applied or not")]
    end

    subgraph RETRAIN[" skywatch_retrain (Track 6, deployed PAUSED) "]
        RTCHAIN["gold_full -> train -> gate (dry run)<br/>weekly schedule, or manual on drift/new data"]
    end

    subgraph MON[" skywatch_monitor (Track 5, dev-only) "]
        DRIFT["src/lib/drift.py::psi<br/>feature drift, unit-tested"]
        HEALTH[("model_health table")]
        ALERT["SQL Alert -> email<br/>(user's hands-on part)"]
    end

    PR -->|"validate"| SP
    MERGE -->|"deploy"| SP
    SP --> STAGING

    POLLER --> DEVTRAIN --> DEVGATE
    DEVGATE -->|"human apply=true only"| DEVAPP
    DEVGATE --> EVALGATE
    STGTRAIN --> STGGATE --> EVALGATE
    EVALGATE --> AUDIT

    RTCHAIN -.->|"chains existing tasks"| DEVTRAIN
    RTCHAIN -.-> DEVGATE

    DEVAPP --> HEALTH
    HEALTH --> DRIFT
    HEALTH --> ALERT
    ALERT -.->|"human decides"| RTCHAIN

    TAG -.->|"marks a point in history<br/>NOT a deploy trigger"| DEV

    classDef ci fill:#dbeafe,stroke:#2563eb,color:#1e293b
    classDef dev fill:#dcfce7,stroke:#16a34a,color:#1e293b
    classDef staging fill:#fef9c3,stroke:#ca8a04,color:#1e293b
    classDef gate fill:#fee2e2,stroke:#dc2626,color:#1e293b
    classDef mon fill:#ede9fe,stroke:#7c3aed,color:#1e293b
    classDef store fill:#f1f5f9,stroke:#475569,color:#1e293b

    class PR,MERGE,TAG ci
    class POLLER,DEVTRAIN,DEVGATE,DEVAPP dev
    class STGTRAIN,STGGATE,STGAPP staging
    class EVALGATE,RTCHAIN gate
    class DRIFT,ALERT mon
    class SP,AUDIT,HEALTH store
```

## Design decisions that shaped this, and why

**No literal `prod` target.** The original plan assumed a classic dev/staging/prod ladder.
Two things ruled that out once actually built (`docs/MLOPS_PLAN.md` Track 3): `dev` already
*is* production in every way that matters for this project (it's what the live demo and the
LinkedIn post point at), and Databricks Asset Bundles has no way to exclude a resource from a
target — confirmed empirically, not assumed — combined with Free Edition's one-active-pipeline
limit, which rules out a `staging` target sharing `dev`'s `databricks.yml` at all. The fix was
a **second, separate bundle** (`serving_staging/`), not a third target.

**A human always applies the promotion, never a workflow.** Both `promote_eta.py` and
`promote_demand.py` default to `apply=false`; `skywatch_retrain`'s gate steps hardcode
`apply="false"` — not a variable, so it can't be accidentally flipped at deploy time the way a
shared variable could. A passing gate is a recommendation. This is the one invariant every
other piece of this architecture is built to respect, including the audit log (Track 8),
which records `applied` as its own column precisely so "the gate passed" and "we shipped it"
are never conflated after the fact.

**CI deploys `staging`, never `dev`.** The same reasoning as the target decision, one level
down: automating a deploy to the live environment risked shipping something no human reviewed
straight to what the world sees. `staging` exists specifically to be the thing CI can safely
touch on every merge.

**Monitoring and retraining are `dev`-only.** `staging` never runs its own poller or gold
rebuild (nothing to monitor freshness/volume on, no ingestion to retrain against) — it reads
prod's real Gold tables read-only and only ever produces its own scoring/model artifacts.

## Where to find things

| Question | Look at |
|---|---|
| "How do I do X?" (deploy, rollback, respond to a signal, retrain) | `docs/RUNBOOKS.md` |
| "What is this model, and what are its limits?" | `docs/MODEL_CARDS.md` |
| "What happened, when, and what did we learn?" | `docs/MLOPS_PLAN.md` §6a (the full progress log) |
| "What's the product this all serves?" | `pitch.md`, then the root `README.md` |
