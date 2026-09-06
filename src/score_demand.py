# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — Model 2 batch scoring (arrival demand forecast)
# MAGIC
# MAGIC Computes the running arrival total for the scoring day up to the "as-of" bin, calls
# MAGIC `skywatch.ml.demand_forecast@champion`, and writes the next `horizon_bins` × 15-min
# MAGIC forecast to `skywatch.stream.demand_forecast`.
# MAGIC
# MAGIC `score_date` / `as_of_bin` default to **now** (production behaviour), but can be pinned to
# MAGIC a real historical day/time for validation — the live poller runs on a schedule, so "today"
# MAGIC often has little or no context yet, and pinning lets the whole `predict()` path be
# MAGIC exercised against real data instead of waiting for a fresh collection window.
# MAGIC
# MAGIC **`chronos-forecasting` is installed even though the champion is climatology mode.**
# MAGIC The logged `DemandForecastModel` pickle captured a live reference to `demand_lib.py`'s
# MAGIC module-level Chronos pipeline cache (populated during the training backtest's zero-shot
# MAGIC baseline calls), so unpickling *any* version of this model currently needs `chronos`/
# MAGIC `torch` importable regardless of which mode it runs in. Follow-up, no retrain needed
# MAGIC today: give `DemandForecastModel` a `__getstate__`/`__setstate__` that only pickles its
# MAGIC plain-data attributes (`mode`, `horizon`, `profile`, `chronos_model_id`), then a future
# MAGIC retrain will log a lighter, cache-free model and this install line can drop back to `scipy`.

# COMMAND ----------
# MAGIC %pip install -q scipy chronos-forecasting
# MAGIC %restart_python

# COMMAND ----------
# MAGIC %run ./demand_lib

# COMMAND ----------
try:
    dbutils.widgets.text("stream_schema", "skywatch.stream")
    dbutils.widgets.text("model_name", "skywatch.ml.demand_forecast")
    dbutils.widgets.text("model_alias", "champion")
    dbutils.widgets.text("apt_icao", "KATL")
    dbutils.widgets.text("horizon_bins", "12")
    dbutils.widgets.text("score_date", "")     # blank = today (UTC)
    dbutils.widgets.text("as_of_bin", "")      # blank = current 15-min bin from the wall clock
    STREAM = dbutils.widgets.get("stream_schema")
    MODEL_NAME = dbutils.widgets.get("model_name")
    ALIAS = dbutils.widgets.get("model_alias")
    APT_ICAO = dbutils.widgets.get("apt_icao")
    H = int(dbutils.widgets.get("horizon_bins"))
    SCORE_DATE = dbutils.widgets.get("score_date").strip()
    AS_OF_BIN = dbutils.widgets.get("as_of_bin").strip()
except Exception:
    STREAM, MODEL_NAME, ALIAS, APT_ICAO, H, SCORE_DATE, AS_OF_BIN = (
        "skywatch.stream", "skywatch.ml.demand_forecast", "champion", "KATL", 12, "", "",
    )

# COMMAND ----------
# MAGIC %md ## 1. Load the champion model

# COMMAND ----------
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
MODEL_VERSION = int(mv.version)
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
print(f"loaded {MODEL_NAME}@{ALIAS} = v{MODEL_VERSION}")

# COMMAND ----------
# MAGIC %md ## 2. Running context — arrivals so far on the scoring day

# COMMAND ----------
import datetime as dt

import pandas as pd

now = dt.datetime.now(dt.timezone.utc)
score_date = dt.date.fromisoformat(SCORE_DATE) if SCORE_DATE else now.date()
if AS_OF_BIN:
    current_slot = int(AS_OF_BIN)
else:
    current_slot = (now.hour * 60 + now.minute) // 15
# Spark dayofweek(): 1=Sunday..7=Saturday — match gold_demand_15m / demand_lib's convention
dow = (dt.date(score_date.year, score_date.month, score_date.day).isoweekday() % 7) + 1

print(f"scoring {APT_ICAO} for {score_date} (dow={dow}) as of bin {current_slot}/{BINS_PER_DAY}"
      + (f"  [pinned: score_date={SCORE_DATE!r} as_of_bin={AS_OF_BIN!r}]"
         if SCORE_DATE or AS_OF_BIN else ""))

today_counts = spark.sql(f"""
  SELECT floor((hour(touchdown_ts) * 60 + minute(touchdown_ts)) / 15) AS slot, count(*) AS n
  FROM {STREAM}.gold_touchdowns
  WHERE apt_icao = '{APT_ICAO}' AND to_date(touchdown_ts) = '{score_date.isoformat()}'
  GROUP BY 1
""").toPandas().set_index("slot")["n"].to_dict()

context = [float(today_counts.get(s, 0)) for s in range(current_slot)]
print(f"{len(context)} context bins, {sum(context):.0f} arrivals so far")

# COMMAND ----------
# MAGIC %md ## 3. Predict + append to `demand_forecast`

# COMMAND ----------
if len(context) < 4:
    print("fewer than 4 context bins (no poller collection covering this window) — "
          "nothing meaningful to score")
    dbutils.notebook.exit("empty")

req = pd.DataFrame([{"context": context, "dow": dow, "horizon": H}])
pred = model.predict(req).iloc[0]

day_start = dt.datetime.combine(score_date, dt.time(0, 0), tzinfo=dt.timezone.utc)
rows = [{
    "scored_at": now, "model_version": MODEL_VERSION, "apt_icao": APT_ICAO,
    "bin_start_ts": day_start + dt.timedelta(minutes=15 * (current_slot + i)),
    "horizon_step": i, "arrivals_so_far": float(sum(context)),
    "predicted_q10": float(pred["q10"][i]), "predicted_q50": float(pred["q50"][i]),
    "predicted_q90": float(pred["q90"][i]), "predicted_mean": float(pred["mean"][i]),
} for i in range(H)]

out = spark.createDataFrame(pd.DataFrame(rows))
(out.write.mode("append").option("mergeSchema", "true").saveAsTable(f"{STREAM}.demand_forecast"))
print(f"wrote {len(rows)} forecast bins")
display(out.orderBy("bin_start_ts"))
