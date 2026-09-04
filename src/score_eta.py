# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — Model 1 batch scoring
# MAGIC
# MAGIC Free Edition has no model serving, so this is how predictions reach the dashboard / App:
# MAGIC a scheduled job loads `skywatch.ml.eta_touchdown@champion`, scores every aircraft that is
# MAGIC **currently inbound** to KATL, and appends to `skywatch.stream.predictions`.
# MAGIC
# MAGIC - Feature engineering is `%run ./eta_features` — **the same module `train_eta.py` uses**,
# MAGIC   plus a hard assert that the columns match the model's logged input signature.
# MAGIC - `predictions` is append-only (`scored_at` per run) so accuracy can be measured later.
# MAGIC - `predictions_scored` joins past predictions to actual `gold_touchdowns` — the numbers
# MAGIC   behind the dashboard's accuracy tile.

# COMMAND ----------
# MAGIC %pip install -q lightgbm mlflow
# MAGIC %restart_python

# COMMAND ----------
# MAGIC %run ./eta_features

# COMMAND ----------
try:
    dbutils.widgets.text("stream_schema", "skywatch.stream")
    dbutils.widgets.text("model_name", "skywatch.ml.eta_touchdown")
    dbutils.widgets.text("model_alias", "champion")
    dbutils.widgets.text("max_dist_nm", "120")
    dbutils.widgets.text("freshness_min", "20")
    STREAM = dbutils.widgets.get("stream_schema")
    MODEL_NAME = dbutils.widgets.get("model_name")
    ALIAS = dbutils.widgets.get("model_alias")
    MAX_DIST_NM = float(dbutils.widgets.get("max_dist_nm"))
    FRESHNESS_MIN = int(dbutils.widgets.get("freshness_min"))
except Exception:
    STREAM, MODEL_NAME, ALIAS, MAX_DIST_NM, FRESHNESS_MIN = (
        "skywatch.stream", "skywatch.ml.eta_touchdown", "champion", 120.0, 20,
    )
print(f"model {MODEL_NAME}@{ALIAS} | inbound <= {MAX_DIST_NM} nm | last {FRESHNESS_MIN} min")

# COMMAND ----------
# MAGIC %md ## 1. Load the champion model + parity check

# COMMAND ----------
import mlflow
import pandas as pd
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
MODEL_VERSION = int(mv.version)

sig_cols = [c["name"] for c in model.metadata.get_input_schema().to_dict()]
if sig_cols != ETA_FEATURES:
    raise ValueError(
        f"train/serve feature drift:\n  model expects {sig_cols}\n  eta_features gives {ETA_FEATURES}"
    )
print(f"loaded v{MODEL_VERSION}; {len(ETA_FEATURES)} features match the model signature")

# COMMAND ----------
# MAGIC %md ## 2. Current inbound aircraft

# COMMAND ----------
from pyspark.sql import functions as F

scoring_sdf = add_eta_features(spark.sql(f"""
  WITH latest AS (
    SELECT *, row_number() OVER (PARTITION BY icao ORDER BY snapshot_ts DESC) AS rn
    FROM {STREAM}.gold_tracks
    WHERE snapshot_ts >= (SELECT max(snapshot_ts) FROM {STREAM}.gold_tracks)
                         - INTERVAL {FRESHNESS_MIN} MINUTES
  )
  SELECT * FROM latest
  WHERE rn = 1 AND inbound_flag AND NOT is_grounded
    AND dist_to_apt_nm IS NOT NULL AND dist_to_apt_nm <= {MAX_DIST_NM}
    AND gs_kt IS NOT NULL AND gs_kt > 40
"""))

score_pdf = eta_pandas(
    scoring_sdf.select(
        "icao", "callsign", "ac_type", "apt_icao", "snapshot_ts",
        "dist_to_apt_nm", "gs_kt", "alt_ft", "heading_err_deg", *ETA_FEATURES
    ).toPandas()
)
print(f"{len(score_pdf)} inbound aircraft to score "
      f"(as of {score_pdf['snapshot_ts'].max() if len(score_pdf) else 'n/a'})")

# COMMAND ----------
# MAGIC %md ## 3. Predict + append to `predictions`

# COMMAND ----------
import datetime as dt

if len(score_pdf) == 0:
    print("no current inbound aircraft — nothing to score")
    dbutils.notebook.exit("empty")

score_pdf["predicted_eta_min"] = model.predict(score_pdf[ETA_FEATURES]).clip(min=0)
score_pdf["predicted_touchdown_ts"] = (
    score_pdf["snapshot_ts"] + pd.to_timedelta(score_pdf["predicted_eta_min"], unit="m")
)
score_pdf["model_version"] = MODEL_VERSION
score_pdf["scored_at"] = dt.datetime.now(dt.timezone.utc)

out_cols = [
    "scored_at", "model_version", "icao", "callsign", "ac_type", "apt_icao",
    "snapshot_ts", "dist_to_apt_nm", "gs_kt", "alt_ft", "heading_err_deg",
    "predicted_eta_min", "predicted_touchdown_ts",
]
out = spark.createDataFrame(score_pdf[out_cols])
(out.write.mode("append").option("mergeSchema", "true")
    .saveAsTable(f"{STREAM}.predictions"))

print(f"appended {out.count()} predictions")
display(out.orderBy("predicted_touchdown_ts")
        .select("callsign", "ac_type", F.round("dist_to_apt_nm", 0).alias("dist_nm"),
                F.round("predicted_eta_min", 1).alias("eta_min"), "predicted_touchdown_ts"))

# COMMAND ----------
# MAGIC %md ## 4. Score past predictions against actual touchdowns

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE TABLE {STREAM}.predictions_scored AS
SELECT
  p.scored_at, p.model_version, p.icao, p.callsign, p.ac_type, p.apt_icao,
  p.snapshot_ts, p.dist_to_apt_nm, p.predicted_eta_min, p.predicted_touchdown_ts,
  td.touchdown_ts                                                                 AS actual_touchdown_ts,
  (unix_timestamp(td.touchdown_ts) - unix_timestamp(p.snapshot_ts)) / 60.0        AS actual_eta_min,
  p.predicted_eta_min
    - (unix_timestamp(td.touchdown_ts) - unix_timestamp(p.snapshot_ts)) / 60.0    AS error_min
FROM {STREAM}.predictions p
JOIN {STREAM}.gold_touchdowns td
  ON td.icao = p.icao
 AND td.touchdown_ts BETWEEN p.snapshot_ts AND p.snapshot_ts + INTERVAL 90 MINUTES
""")

sc = spark.table(f"{STREAM}.predictions_scored")
n = sc.count()
if n:
    agg = sc.selectExpr(
        "count(*) n", "round(avg(abs(error_min)), 2) mae_min",
        "round(percentile(abs(error_min), 0.9), 2) p90_min", "round(avg(error_min), 2) bias_min",
    ).first()
    print(f"predictions_scored: {agg['n']} matched | MAE {agg['mae_min']} min | "
          f"P90 {agg['p90_min']} | bias {agg['bias_min']}")
else:
    print("predictions_scored: 0 rows yet (need predictions whose aircraft later landed)")
