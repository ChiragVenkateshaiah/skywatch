# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — Model 1 promotion gate
# MAGIC
# MAGIC `train_eta.py` only ever registers `@challenger` — it does not decide whether a candidate
# MAGIC ships. This notebook is that decision, separated out (docs/MLOPS_PLAN.md Track 4):
# MAGIC
# MAGIC 1. Load `@challenger` and (if one exists yet) `@champion`.
# MAGIC 2. Score both on the **same pinned holdout** — `gold_arrival_tracks VERSION AS OF
# MAGIC    holdout_table_version` (blank = current), so the comparison is apples-to-apples even
# MAGIC    as the table keeps growing, and reproducible if re-run later.
# MAGIC 3. Compare MAE **by distance band** (matching `train_eta.py`'s own report) — a candidate
# MAGIC    that wins overall but regresses 0-20 nm (the band that matters most for arrival
# MAGIC    sequencing) should not ship un-flagged.
# MAGIC 4. **Gate**: promote if overall MAE improves by >= `min_improvement_pct` OR is within
# MAGIC    `noise_band_pct` of the champion (not meaningfully worse), **and** no individual band
# MAGIC    regresses by more than `max_band_regression_min` minutes.
# MAGIC 5. **This notebook never flips `@champion` on its own** — a passing gate only prints a
# MAGIC    recommendation. Flipping the alias needs a second, explicit run with `apply=true` —
# MAGIC    that deliberate re-trigger, by a human who read the report, *is* the approval. If no
# MAGIC    `@champion` exists yet (first run ever for this model), the gate auto-passes: there is
# MAGIC    nothing to regress against.

# COMMAND ----------
# MAGIC %pip install -q lightgbm
# MAGIC %restart_python

# COMMAND ----------
try:
    dbutils.widgets.text("stream_schema", "skywatch.stream")
    dbutils.widgets.text("read_stream_schema", "")
    dbutils.widgets.text("model_name", "skywatch.ml.eta_touchdown")
    dbutils.widgets.text("test_date", "2026-09-01")
    dbutils.widgets.text("holdout_table_version", "")   # blank = current table state
    dbutils.widgets.text("min_improvement_pct", "2.0")
    dbutils.widgets.text("noise_band_pct", "1.0")
    dbutils.widgets.text("max_band_regression_min", "0.15")
    dbutils.widgets.text("apply", "false")
    STREAM = dbutils.widgets.get("stream_schema")
    READ_STREAM = dbutils.widgets.get("read_stream_schema").strip() or STREAM
    MODEL_NAME = dbutils.widgets.get("model_name")
    TEST_DATE = dbutils.widgets.get("test_date")
    HOLDOUT_VERSION = dbutils.widgets.get("holdout_table_version").strip()
    MIN_IMPROVEMENT_PCT = float(dbutils.widgets.get("min_improvement_pct"))
    NOISE_BAND_PCT = float(dbutils.widgets.get("noise_band_pct"))
    MAX_BAND_REGRESSION_MIN = float(dbutils.widgets.get("max_band_regression_min"))
    APPLY = dbutils.widgets.get("apply").lower() == "true"
except Exception:
    STREAM, MODEL_NAME, TEST_DATE, HOLDOUT_VERSION = (
        "skywatch.stream", "skywatch.ml.eta_touchdown", "2026-09-01", "",
    )
    READ_STREAM = STREAM
    MIN_IMPROVEMENT_PCT, NOISE_BAND_PCT, MAX_BAND_REGRESSION_MIN, APPLY = 2.0, 1.0, 0.15, False

print(f"{MODEL_NAME} | holdout day {TEST_DATE} | table version "
      f"{HOLDOUT_VERSION or 'current'} | apply={APPLY}")

# COMMAND ----------
# MAGIC %run ./eta_features

# COMMAND ----------
# MAGIC %md ## 1. Load challenger + champion (if one exists)

# COMMAND ----------
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

challenger_mv = client.get_model_version_by_alias(MODEL_NAME, "challenger")
challenger = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}@challenger")
print(f"challenger: v{challenger_mv.version}")

try:
    champion_mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    champion = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}@champion")
    print(f"champion:   v{champion_mv.version}")
except Exception:  # noqa: BLE001 — no @champion alias set yet (first run for this model)
    champion_mv, champion = None, None
    print("champion:   none yet — this will be a bootstrap promotion if the challenger scores at all")

if challenger_mv.version == getattr(champion_mv, "version", None):
    print("challenger IS the current champion — nothing to evaluate")
    dbutils.notebook.exit("already_champion")

# COMMAND ----------
# MAGIC %md ## 2. Build the held-out evaluation frame (same filters as train_eta.py)

# COMMAND ----------
from pyspark.sql import functions as F

version_clause = f"VERSION AS OF {HOLDOUT_VERSION}" if HOLDOUT_VERSION else ""

sdf = add_eta_features(
    spark.sql(f"SELECT * FROM {READ_STREAM}.gold_arrival_tracks {version_clause}")
    .join(
        spark.table(f"{READ_STREAM}.gold_touchdowns")
        .where(F.col("touchdown_confidence") == "confirmed")
        .select("seg_id"),
        "seg_id",
    )
    .where(F.col(ETA_TARGET).between(0.5, 40))
    .where(F.col("dist_to_apt_nm").isNotNull() & F.col("alt_ft").isNotNull())
    .where(F.col("gs_kt") > 60)
    .withColumn("obs_date", F.to_date("snapshot_ts"))
    .where(F.col("obs_date") == TEST_DATE)
)
pdf = eta_pandas(sdf.select("seg_id", ETA_TARGET, *ETA_FEATURES).toPandas())
print(f"{len(pdf):,} held-out rows, {pdf.seg_id.nunique():,} arrivals on {TEST_DATE}")

if len(pdf) == 0:
    print(f"no held-out data for {TEST_DATE} at this READ_STREAM — nothing to gate on")
    dbutils.notebook.exit("empty_holdout")

X, y = pdf[ETA_FEATURES], pdf[ETA_TARGET]

# COMMAND ----------
# MAGIC %md ## 3. Score both models, by distance band

# COMMAND ----------
import pandas as pd
from sklearn.metrics import mean_absolute_error


def by_band(pred, actual, dist):
    bands = pd.cut(dist, [0, 20, 40, 70, 101], include_lowest=True,
                   labels=["0-20nm", "20-40nm", "40-70nm", "70-100nm"])
    out = {}
    for band in bands.cat.categories:
        m = (bands == band).to_numpy()
        if m.sum():
            out[band] = mean_absolute_error(actual[m], pred[m])
    out["ALL"] = mean_absolute_error(actual, pred)
    return out


challenger_bands = by_band(challenger.predict(X), y.values, pdf["dist_to_apt_nm"])
champion_bands = (
    by_band(champion.predict(X), y.values, pdf["dist_to_apt_nm"]) if champion is not None else None
)

report = pd.DataFrame({"challenger_mae": challenger_bands})
if champion_bands:
    report["champion_mae"] = pd.Series(champion_bands)
    report["delta_min"] = report["challenger_mae"] - report["champion_mae"]
display(report.round(3))

# COMMAND ----------
# MAGIC %md ## 4. Gate
# MAGIC The decision itself is pure Python — `src/lib/promotion.py`, unit-tested in
# MAGIC `tests/test_promotion.py` rather than only ever exercised by a live job run.

# COMMAND ----------
from lib.promotion import evaluate_gate

gate = evaluate_gate(
    challenger_bands, champion_bands, MIN_IMPROVEMENT_PCT, NOISE_BAND_PCT, MAX_BAND_REGRESSION_MIN
)
passed, reasons = gate.passed, gate.reasons

verdict = "PASS" if passed else "HOLD"
print(f"\nGATE: {verdict}")
for r in reasons:
    print(f"  - {r}")

# COMMAND ----------
# MAGIC %md ## 5. Apply (only on a passing gate, and only if explicitly asked)

# COMMAND ----------
client.set_model_version_tag(MODEL_NAME, challenger_mv.version, "gate_verdict", verdict)
client.set_model_version_tag(MODEL_NAME, challenger_mv.version, "gate_holdout_date", TEST_DATE)

if passed and APPLY:
    client.set_registered_model_alias(MODEL_NAME, "champion", challenger_mv.version)
    print(f"\nPROMOTED {MODEL_NAME} v{challenger_mv.version} to @champion")
elif passed:
    print(f"\nGate passed — re-run with apply=true to actually promote v{challenger_mv.version}")
else:
    print(f"\nHolding @champion at v{getattr(champion_mv, 'version', '(none)')} — challenger not promoted")
    if APPLY:
        print("(apply=true was set, but a failing gate is never applied)")
