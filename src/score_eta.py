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
# MAGIC - Also exports the champion model to `/Volumes/{catalog}/ml/models/` every run — how the
# MAGIC   App's click-to-predict panel loads it (see step 1b; UC's per-model `EXECUTE` grant isn't
# MAGIC   available on this metastore).

# COMMAND ----------
# MAGIC %pip install -q lightgbm mlflow joblib
# MAGIC %restart_python

# COMMAND ----------
# MAGIC %run ./eta_features

# COMMAND ----------
try:
    dbutils.widgets.text("stream_schema", "skywatch.stream")
    # blank = same as stream_schema (today's dev behaviour, byte-identical). A staging
    # deploy points this at prod's real Gold tables while stream_schema stays staging's own
    # write catalog — see docs/MLOPS_PLAN.md Track 3.
    dbutils.widgets.text("read_stream_schema", "")
    dbutils.widgets.text("model_name", "skywatch.ml.eta_touchdown")
    dbutils.widgets.text("model_alias", "champion")
    dbutils.widgets.text("max_dist_nm", "120")
    dbutils.widgets.text("freshness_min", "20")
    STREAM = dbutils.widgets.get("stream_schema")
    READ_STREAM = dbutils.widgets.get("read_stream_schema").strip() or STREAM
    MODEL_NAME = dbutils.widgets.get("model_name")
    ALIAS = dbutils.widgets.get("model_alias")
    MAX_DIST_NM = float(dbutils.widgets.get("max_dist_nm"))
    FRESHNESS_MIN = int(dbutils.widgets.get("freshness_min"))
except Exception:
    STREAM, MODEL_NAME, ALIAS, MAX_DIST_NM, FRESHNESS_MIN = (
        "skywatch.stream", "skywatch.ml.eta_touchdown", "champion", 120.0, 20,
    )
    READ_STREAM = STREAM
print(f"model {MODEL_NAME}@{ALIAS} | inbound <= {MAX_DIST_NM} nm | last {FRESHNESS_MIN} min "
      f"| read {READ_STREAM} | write {STREAM}")

# COMMAND ----------
# MAGIC %md ## 1. Load the champion model + parity check

# COMMAND ----------
import mlflow
import pandas as pd
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
MODEL_VERSION = int(mv.version)

# parity check against the logged signature ...
sig_cols = [s["name"] for s in mlflow.models.get_model_info(model_uri).signature.inputs.to_dict()]
if sig_cols != ETA_FEATURES:
    raise ValueError(
        f"train/serve feature drift:\n  model expects {sig_cols}\n  eta_features gives {ETA_FEATURES}"
    )

# ... but predict on the raw LGBMRegressor: it needs `phase` as a category dtype, which the
# pyfunc wrapper's schema enforcement (signature says string) rejects.
model = mlflow.lightgbm.load_model(model_uri)
print(f"loaded v{MODEL_VERSION}; {len(ETA_FEATURES)} features match the model signature")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1b. Export the champion to a Volume — this is how the App's click-to-predict works
# MAGIC
# MAGIC The App loads a model in-process for interactive prediction (Free Edition has no serving
# MAGIC endpoint), which normally means granting its service principal `EXECUTE` on the UC
# MAGIC registered model. **That grant isn't available on this metastore** —
# MAGIC `GRANT EXECUTE ON MODEL` and the Grants API both fail with `REGISTERED_MODEL is not
# MAGIC enabled`, confirmed against the real workspace. A Volume is a plain, always-grantable
# MAGIC securable, so instead: export the exact model object this run just loaded to a Volume file
# MAGIC every run, and the App reads it from there (`READ VOLUME`, not a model-registry grant).
# MAGIC Runs before the early-exit below, so the App's copy stays current even on a quiet burst
# MAGIC with no aircraft to score.

# COMMAND ----------
import datetime as _dt
import json

import joblib

CATALOG = STREAM.split(".")[0]
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.ml.models")
VOL_DIR = f"/Volumes/{CATALOG}/ml/models"
joblib.dump(model, f"{VOL_DIR}/eta_touchdown_champion.joblib")
with open(f"{VOL_DIR}/eta_touchdown_champion.meta.json", "w") as f:
    json.dump({"model_name": MODEL_NAME, "model_version": MODEL_VERSION,
              "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat()}, f)
print(f"exported champion v{MODEL_VERSION} -> {VOL_DIR}/eta_touchdown_champion.joblib")

# COMMAND ----------
# MAGIC %md ## 2. Current inbound aircraft

# COMMAND ----------
from pyspark.sql import functions as F

# `airport_inbound_count` is joined here exactly as gold_arrival_tracks does it in
# build_gold.py — keep the two in sync.
scoring_sdf = add_eta_features(spark.sql(f"""
  WITH latest AS (
    SELECT *, row_number() OVER (PARTITION BY icao ORDER BY snapshot_ts DESC) AS rn
    FROM {READ_STREAM}.gold_tracks
    WHERE snapshot_ts >= (SELECT max(snapshot_ts) FROM {READ_STREAM}.gold_tracks)
                         - INTERVAL {FRESHNESS_MIN} MINUTES
  ),
  inbound_ct AS (
    -- keep identical to gold_arrival_tracks in build_gold.py: only the rings the backfill
    -- (100 nm) and the live poller (250 nm) both cover
    SELECT minute_ts, apt_icao, sum(n_inbound) AS n_inbound_common_rings
    FROM {READ_STREAM}.gold_congestion
    WHERE ring IN ('00-40', '40-100')
    GROUP BY 1, 2
  )
  SELECT /*+ BROADCAST(c) */ l.*, coalesce(c.n_inbound_common_rings, 0) AS airport_inbound_count
  FROM latest l
  LEFT JOIN inbound_ct c
    ON c.minute_ts = date_trunc('MINUTE', l.snapshot_ts) AND c.apt_icao = l.apt_icao
  WHERE l.rn = 1 AND l.inbound_flag AND NOT l.is_grounded
    AND l.dist_to_apt_nm IS NOT NULL AND l.dist_to_apt_nm <= {MAX_DIST_NM}
    AND l.gs_kt IS NOT NULL AND l.gs_kt > 40
"""))

# dist_to_apt_nm / gs_kt / alt_ft / heading_err_deg are already in ETA_FEATURES — don't
# re-list them (duplicate pandas columns break MLflow schema enforcement)
score_pdf = eta_pandas(
    scoring_sdf.select(
        "icao", "callsign", "ac_type", "apt_icao", "snapshot_ts", "lat", "lon", *ETA_FEATURES
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

# eta_pandas() has already pinned `phase` to the fixed PHASE_CATEGORIES, so the frame matches
# what the model saw at fit time.
score_pdf["predicted_eta_min"] = model.predict(score_pdf[ETA_FEATURES]).clip(min=0)
score_pdf["predicted_touchdown_ts"] = (
    score_pdf["snapshot_ts"] + pd.to_timedelta(score_pdf["predicted_eta_min"], unit="m")
)
score_pdf["model_version"] = MODEL_VERSION
score_pdf["scored_at"] = dt.datetime.now(dt.timezone.utc)

# write the full feature vector + lat/lon too — the Arrival Manager App reads it for the
# live map and for in-process "click-to-predict" (re-run the model on a tweaked aircraft).
score_pdf = score_pdf.copy()
score_pdf["phase"] = score_pdf["phase"].astype(str)   # Categorical -> str for Spark
out_cols = [
    "scored_at", "model_version", "icao", "callsign", "ac_type", "apt_icao", "snapshot_ts",
    "lat", "lon", "predicted_eta_min", "predicted_touchdown_ts", *ETA_FEATURES,
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
JOIN {READ_STREAM}.gold_touchdowns td
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
