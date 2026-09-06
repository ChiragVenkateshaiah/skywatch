# SkyWatch — Arrival Manager App

Streamlit app that serves the two models to an arrival coordinator:

| Panel | Model | Source table |
|---|---|---|
| Predicted arrival sequence | Model 1 — time-to-touchdown | `skywatch.stream.predictions` (latest `scored_at`) |
| Demand forecast — next 3 h vs AAR | Model 2 — 15-min demand forecast | `skywatch.stream.demand_forecast` (latest `scored_at`) |
| Surge alerts | derived | forecast bins where `predicted_q50` > AAR/bin |
| Typical-hour reference line | — | `skywatch.stream.gold_demand_15m` |

Free Edition has no model-serving endpoints, so the panels read Delta tables the
batch jobs (`skywatch_score_eta`, `skywatch_score_demand`) write. Data freshness =
the scoring cadence (~10 min).

## Files

- `app.py` — the Streamlit UI
- `data.py` — SQL Statement Execution helpers (app service-principal auth, no PAT)
- `app.yaml` — entry point + env (`SKYWATCH_*`, warehouse id injected from the resource)
- `requirements.txt` — extra deps beyond the Apps base image

## Deploy

```bash
databricks bundle deploy -t dev
databricks bundle run skywatch_arrival_manager -t dev     # start / restart (beats the 24 h auto-stop)
```

After the first deploy, grant the app's service principal read access (name is shown
on the app's page in the workspace), as metastore admin in a SQL editor:

```sql
GRANT USE CATALOG ON CATALOG skywatch TO `<app-service-principal>`;
GRANT USE SCHEMA, SELECT ON SCHEMA skywatch.stream TO `<app-service-principal>`;
```

## Run locally

```bash
cd app
pip install -r requirements.txt
export SKYWATCH_WAREHOUSE_ID=8fd481cbf45ac93e
export SKYWATCH_CATALOG=skywatch SKYWATCH_SCHEMA=stream SKYWATCH_APT_ICAO=KATL SKYWATCH_AAR_PER_HOUR=90
# auth: `databricks auth login` profile picked up by the SDK, or DATABRICKS_* env vars
streamlit run app.py
```

## Not in v1 (next PRs)

- deck.gl live map of current inbounds coloured by ETA band
- click-to-predict: in-process `eta_touchdown@champion` load + per-feature contributions
- scheduled restart job to keep the app up during demo windows
