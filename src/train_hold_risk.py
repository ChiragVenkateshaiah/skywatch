# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — Model 3: hold-risk classifier (investigation)
# MAGIC
# MAGIC **Question:** for an aircraft inbound to KATL and still 30–90 nm out, will it be put into a
# MAGIC holding pattern before it lands?
# MAGIC
# MAGIC ## Finding (2026-09-07): not viable — the phenomenon isn't in the data
# MAGIC
# MAGIC We expanded the backfill to **24 days at 60 s cadence** specifically to get enough holding
# MAGIC labels for this model (9 days gave 25, 24 days gave 159 `gold_irregularities` "holding"
# MAGIC segments). But when we link each detected hold to *that same aircraft landing at KATL
# MAGIC within 90 minutes*: **0 of 159**. Every detected "hold" is a persistent loiterer — the same
# MAGIC handful of ICAOs orbiting near KATL repeatedly at 40 nm / 9,000 ft (survey, traffic-watch,
# MAGIC law-enforcement, airwork), never landing.
# MAGIC
# MAGIC That is the correct answer for KATL: the archive only has the 1st of each month, which are
# MAGIC historically clear-weather days, and **KATL does not hold arrivals in good weather** — it
# MAGIC absorbs demand with in-trail spacing and speed control. Arrival holding stacks form in
# MAGIC weather / major disruption, which these days don't contain.
# MAGIC
# MAGIC So **Model 3 ships rule-based only** (`gold_irregularities` + `score_irregularities.py` —
# MAGIC emergency, go-around, live racetrack). A learned hold-risk model needs either weather-affected
# MAGIC days or a continuous-collection dataset that captures real stacks. This notebook is kept
# MAGIC ready: it builds the genuine-arrival-hold label and **exits early** until there are enough
# MAGIC real positives (`min_pos_segs`), then trains a grouped-CV class-weighted LightGBM and
# MAGIC registers `@challenger` (promotion to `@champion` stays a manual, documented decision).

# COMMAND ----------
# MAGIC %pip install -q lightgbm scikit-learn
# MAGIC %restart_python

# COMMAND ----------
try:
    dbutils.widgets.text("stream_schema", "skywatch.stream")
    dbutils.widgets.text("model_name", "skywatch.ml.hold_risk")
    dbutils.widgets.text("band_nm", "30,90")          # decision band: min,max distance
    dbutils.widgets.text("land_within_min", "90")     # hold -> same-icao KATL landing within this
    dbutils.widgets.text("max_same_day_holds", "2")   # icao with more holds/day = loiterer, excluded
    dbutils.widgets.text("min_pos_segs", "40")        # need this many genuine arrival-hold segments to train
    STREAM = dbutils.widgets.get("stream_schema")
    MODEL_NAME = dbutils.widgets.get("model_name")
    BAND = [float(x) for x in dbutils.widgets.get("band_nm").split(",")]
    LAND_WITHIN = int(dbutils.widgets.get("land_within_min"))
    MAX_DAY_HOLDS = int(dbutils.widgets.get("max_same_day_holds"))
    MIN_POS = int(dbutils.widgets.get("min_pos_segs"))
except Exception:
    STREAM, MODEL_NAME, BAND, LAND_WITHIN, MAX_DAY_HOLDS, MIN_POS = (
        "skywatch.stream", "skywatch.ml.hold_risk", [30.0, 90.0], 90, 2, 40,
    )
print(f"band {BAND} nm | land within {LAND_WITHIN} min | max {MAX_DAY_HOLDS} holds/icao/day | need {MIN_POS} positives")

# COMMAND ----------
# MAGIC %md ## 1. Genuine arrival-hold label
# MAGIC A `gold_irregularities` holding segment counts as a real arrival hold only if the **same
# MAGIC ICAO lands at KATL within `land_within_min`** of the hold ending, and that ICAO is not a
# MAGIC loiterer that day (`> max_same_day_holds` holding segments).

# COMMAND ----------
import pyspark.sql.functions as F

held_segs = spark.sql(f"""
WITH hold_end AS (
  SELECT i.seg_id, i.icao, to_date(i.event_ts) AS d, max(g.snapshot_ts) AS t_end
  FROM {STREAM}.gold_irregularities i
  JOIN {STREAM}.gold_tracks g ON g.seg_id = i.seg_id
  WHERE i.kind = 'holding'
  GROUP BY 1, 2, 3
),
loiter AS (
  SELECT icao, to_date(event_ts) AS d, count(*) AS n
  FROM {STREAM}.gold_irregularities WHERE kind = 'holding' GROUP BY 1, 2
),
linked AS (
  SELECT he.seg_id, coalesce(l.n, 1) AS same_day_holds, min(td.touchdown_ts) AS td_ts
  FROM hold_end he
  LEFT JOIN loiter l ON l.icao = he.icao AND l.d = he.d
  LEFT JOIN {STREAM}.gold_touchdowns td ON td.icao = he.icao
       AND td.touchdown_ts > he.t_end
       AND td.touchdown_ts < he.t_end + make_interval(0, 0, 0, 0, 0, {LAND_WITHIN}, 0)
  GROUP BY 1, 2
)
SELECT seg_id FROM linked WHERE td_ts IS NOT NULL AND same_day_holds <= {MAX_DAY_HOLDS}
""")
n_pos_segs = held_segs.count()
n_raw_holds = spark.table(f"{STREAM}.gold_irregularities").where(F.col("kind") == "holding").count()
print(f"{n_raw_holds} detected 'holding' segments -> {n_pos_segs} genuine arrival holds "
      f"(held then landed at KATL within {LAND_WITHIN} min, not a loiterer)")

if n_pos_segs < MIN_POS:
    print(f"\nFINDING: {n_pos_segs} genuine arrival-hold segments (need {MIN_POS}). "
          f"KATL does not hold arrivals on the clear-weather archive days — see the notebook "
          f"header. Model 3 remains rule-based. Exiting without training.")
    dbutils.notebook.exit(f"insufficient_positives:{n_pos_segs}")

# COMMAND ----------
# MAGIC %md ## 2. Build the decision-band dataset

# COMMAND ----------
import numpy as np
import pandas as pd

FEATURES = [
    "dist_to_apt_nm", "alt_ft", "gs_kt", "vrate_fpm", "heading_err_deg",
    "closure_kt", "turn_rate_dps", "bearing_to_apt", "naive_eta_min",
    "hour_utc", "dow", "airport_inbound_count",
]

at = (spark.table(f"{STREAM}.gold_arrival_tracks")
      .where((F.col("dist_to_apt_nm") >= BAND[0]) & (F.col("dist_to_apt_nm") <= BAND[1]))
      .where(F.col("gs_kt") > 60)
      .withColumn("naive_eta_min", F.col("dist_to_apt_nm") / F.col("gs_kt") * 60.0)
      .join(held_segs.withColumn("is_held", F.lit(1)), "seg_id", "left")
      .withColumn("is_held", F.coalesce(F.col("is_held"), F.lit(0))))

pdf = at.select("seg_id", F.to_date("snapshot_ts").alias("day"), "is_held", *FEATURES).toPandas()
for c in FEATURES:
    pdf[c] = pd.to_numeric(pdf[c], errors="coerce").astype("float64")
pdf = pdf.dropna(subset=FEATURES)
base_rate = pdf["is_held"].mean()
print(f"{len(pdf)} band rows | {pdf['is_held'].sum()} positive ({base_rate:.4%}) | "
      f"{pdf['seg_id'].nunique()} segments")

# COMMAND ----------
# MAGIC %md ## 3. Grouped-CV class-weighted LightGBM

# COMMAND ----------
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.03, num_leaves=31,
              min_child_samples=40, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbosity=-1)


def fit(X, y):
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    m = lgb.LGBMClassifier(**PARAMS, scale_pos_weight=spw)
    m.fit(X, y)
    return m


groups = pdf["seg_id"].values
aps, rocs = [], []
for tr_i, va_i in GroupKFold(n_splits=5).split(pdf[FEATURES], pdf["is_held"], groups):
    tr, va = pdf.iloc[tr_i], pdf.iloc[va_i]
    if va["is_held"].sum() == 0:
        continue
    m = fit(tr[FEATURES], tr["is_held"])
    p = m.predict_proba(va[FEATURES])[:, 1]
    aps.append(average_precision_score(va["is_held"], p))
    rocs.append(roc_auc_score(va["is_held"], p))

cv_ap, cv_roc = float(np.mean(aps)), float(np.mean(rocs))
lift = cv_ap / base_rate if base_rate > 0 else float("nan")
print(f"GroupKFold-5: AP {cv_ap:.3f} ± {np.std(aps):.3f} (base {base_rate:.4%}, lift {lift:.1f}x) | "
      f"ROC-AUC {cv_roc:.3f}")

model = fit(pdf[FEATURES], pdf["is_held"])
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("top features:", ", ".join(f"{k} {v}" for k, v in imp.head(6).items()))

# COMMAND ----------
# MAGIC %md ## 4. Register `@challenger` (promotion to `@champion` is a manual decision)

# COMMAND ----------
import mlflow
from mlflow.models import infer_signature
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
example = pdf[FEATURES].head(3)
sig = infer_signature(example, model.predict_proba(example)[:, 1])

with mlflow.start_run(run_name="hold_risk") as run:
    mlflow.log_params({"band_nm": BAND, "land_within_min": LAND_WITHIN, "n_pos_segs": n_pos_segs,
                       "n_rows": len(pdf), "base_rate": float(base_rate)})
    mlflow.log_metrics({"cv_ap": cv_ap, "cv_roc_auc": cv_roc, "cv_ap_lift": float(lift)})
    mlflow.lightgbm.log_model(model, "model", signature=sig, input_example=example,
                              pip_requirements=["lightgbm", "scikit-learn", "pandas", "numpy"])
    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)

client = MlflowClient()
client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
client.set_model_version_tag(MODEL_NAME, mv.version, "cv_ap", f"{cv_ap:.3f}")
client.set_model_version_tag(MODEL_NAME, mv.version, "cv_ap_lift", f"{lift:.1f}")
print(f"registered {MODEL_NAME} v{mv.version} @challenger (CV AP {cv_ap:.3f}, {lift:.1f}x base). "
      f"Promote to @champion manually after reviewing the lift on genuine holds.")
