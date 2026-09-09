# SkyWatch — Arrival Manager App

Streamlit app that serves all three models to an arrival coordinator:

| Panel | Model | Source |
|---|---|---|
| KPI row | 1 + 2 | `predictions`, `demand_forecast` |
| **Inbound aircraft map** (pydeck) | Model 1 | `predictions` — `lat`/`lon`, coloured by predicted-ETA band |
| Demand forecast — next 3 h vs AAR + surge alerts | Model 2 | `demand_forecast` (latest `scored_at`), `gold_demand_15m` |
| Predicted arrival sequence | Model 1 | `predictions` (latest `scored_at`) |
| **Click-to-predict** | Model 1 | `eta_touchdown@champion` loaded in-process — pick an aircraft, tweak groundspeed/distance, see the re-prediction + per-feature contributions (`pred_contrib`) |
| Irregularity flags | Model 3 (rules) | `irregularity_flags` (latest `scored_at`) |

Free Edition has no model-serving endpoints, so the panels read Delta tables the batch
jobs (`skywatch_score_eta`, `skywatch_score_demand`, `skywatch_score_irregularities`)
write. Data freshness = the scoring cadence (~10 min). Click-to-predict is the one
exception — it loads the model into the app process and calls it directly.

## Files

- `app.py` — the Streamlit UI
- `data.py` — SQL Statement Execution helpers (app service-principal auth, no PAT) + `ETA_FEATURES`
- `model.py` — in-process `eta_touchdown@champion` load + `predict_eta` / `contributions`
- `app.yaml` — entry point + env (`SKYWATCH_*`; warehouse id injected from the resource)
- `requirements.txt` — deps beyond the Apps base image (adds `pydeck`, `mlflow`, `lightgbm`)

## Deploy

```bash
databricks bundle deploy -t dev
databricks bundle run skywatch_arrival_manager -t dev      # start / restart (beats the 24 h auto-stop)
```

`skywatch_app_keepalive` (deployed PAUSED) restarts the app on a schedule during a demo window.

After the first deploy, grant the app's service principal (name is on the app's page), as
metastore admin in a SQL editor:

```sql
GRANT USE CATALOG ON CATALOG skywatch TO `<app-service-principal>`;
GRANT USE SCHEMA, SELECT ON SCHEMA skywatch.stream TO `<app-service-principal>`;
GRANT USE SCHEMA ON SCHEMA skywatch.ml TO `<app-service-principal>`;
GRANT EXECUTE ON MODEL skywatch.ml.eta_touchdown TO `<app-service-principal>`;   -- click-to-predict
```

The app degrades gracefully without the last two grants (every panel except click-to-predict
still works).

## Run locally

```bash
cd app
pip install -r requirements.txt
export SKYWATCH_WAREHOUSE_ID=8fd481cbf45ac93e
export SKYWATCH_CATALOG=skywatch SKYWATCH_SCHEMA=stream SKYWATCH_APT_ICAO=KATL SKYWATCH_AAR_PER_HOUR=90
# auth: `databricks auth login` profile picked up by the SDK, or DATABRICKS_* env vars
streamlit run app.py
```

## Not yet

- predicted track polyline on the map (next ~5 min extrapolation)
- bind the AI/BI dashboard as an IaC `dashboards` resource
