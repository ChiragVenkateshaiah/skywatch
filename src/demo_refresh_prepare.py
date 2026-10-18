# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — pre-interview refresh: preflight + reset the live picture
# MAGIC
# MAGIC Task 1 of `skywatch_demo_refresh` (docs/DEMO_REFRESH_PLAN.md). **Preflight first, and
# MAGIC nothing is deleted unless every check passes** — a refresh that clears the live tables and
# MAGIC then can't rebuild them is worse than no refresh.
# MAGIC
# MAGIC Reset = clear the serving tables the App shows as "now" (`predictions`, `demand_forecast`,
# MAGIC `irregularity_flags`); the rest of the job rebuilds them from one fresh burst. **History
# MAGIC is kept**: `silver_positions` / `gold_*` are the models' training data (and the M2
# MAGIC champion's climatology), and `ops_briefings`, `promotion_audit`, `model_health` are audit
# MAGIC logs. Cleared rows stay recoverable with Delta time travel (`RESTORE TABLE … TO VERSION AS
# MAGIC OF …`) for the retention period. Side effect, by design: `monitor.py`'s drift / rolling-MAE
# MAGIC checks start again from the refresh, since they read `predictions` history.

# COMMAND ----------
try:
    dbutils.widgets.text("catalog", "skywatch")
    dbutils.widgets.text("stream_schema", "skywatch.stream")
    dbutils.widgets.text("briefing_endpoint", "databricks-llama-4-maverick")
    dbutils.widgets.text("reset", "true")
    CATALOG = dbutils.widgets.get("catalog")
    STREAM = dbutils.widgets.get("stream_schema")
    ENDPOINT = dbutils.widgets.get("briefing_endpoint").strip()
    RESET = dbutils.widgets.get("reset").strip().lower() == "true"
except Exception:
    CATALOG, STREAM, ENDPOINT, RESET = "skywatch", "skywatch.stream", "databricks-llama-4-maverick", True

LIVE_TABLES = ["predictions", "demand_forecast", "irregularity_flags"]
HISTORY_TABLES = ["silver_positions", "gold_tracks", "gold_arrival_tracks", "gold_touchdowns", "gold_demand_15m"]
CHAMPIONS = [f"{CATALOG}.ml.eta_touchdown", f"{CATALOG}.ml.demand_forecast"]
MODEL_FILE = f"/Volumes/{CATALOG}/ml/models/eta_touchdown_champion.joblib"

print(f"preflight for {STREAM} | reset={RESET}")

# COMMAND ----------
# MAGIC %md ## 1. Preflight — fail before touching anything

# COMMAND ----------
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
problems = []

for t in HISTORY_TABLES:
    if not spark.catalog.tableExists(f"{STREAM}.{t}"):
        problems.append(f"history table missing: {STREAM}.{t}")

for name in CHAMPIONS:
    try:
        v = w.model_versions.get_by_alias(name, "champion")
        print(f"ok  {name}@champion = v{v.version}")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"{name}@champion not found: {type(exc).__name__}")

try:
    w.files.get_metadata(MODEL_FILE)
    print(f"ok  {MODEL_FILE}")
except Exception as exc:  # noqa: BLE001 — click-to-predict reads this file
    problems.append(f"exported model file missing: {MODEL_FILE} ({type(exc).__name__})")

if ENDPOINT:
    try:
        ready = w.serving_endpoints.get(ENDPOINT).state.ready
        ready = getattr(ready, "value", ready)
        if ready != "READY":
            problems.append(f"briefing endpoint {ENDPOINT} is {ready}")
        else:
            print(f"ok  {ENDPOINT} READY")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"briefing endpoint {ENDPOINT}: {type(exc).__name__}")

if problems:
    raise RuntimeError("preflight failed — nothing was reset:\n  " + "\n  ".join(problems))
print("preflight passed")

# COMMAND ----------
# MAGIC %md ## 2. Reset the live picture

# COMMAND ----------
if not RESET:
    dbutils.notebook.exit("preflight passed; reset=false")

for t in LIVE_TABLES:
    name = f"{STREAM}.{t}"
    if not spark.catalog.tableExists(name):
        print(f"skip {name} (doesn't exist yet)")
        continue
    version = spark.sql(f"DESCRIBE HISTORY {name} LIMIT 1").first()["version"]
    n = spark.table(name).count()
    spark.sql(f"DELETE FROM {name}")
    print(f"cleared {name}: {n} rows (restore: RESTORE TABLE {name} TO VERSION AS OF {version})")
