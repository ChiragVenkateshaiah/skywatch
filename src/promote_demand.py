# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — Model 2 promotion gate
# MAGIC
# MAGIC `forecast_demand.py` only ever registers `@challenger` (mirrors the `train_eta.py` /
# MAGIC `promote_eta.py` split from Track 4 — see that notebook's header for the full rationale).
# MAGIC This is the actual promote/hold decision for Model 2.
# MAGIC
# MAGIC **Honest methodology note**: unlike Model 1 (a true unseen test day), M2's champion is
# MAGIC fit on *all* collected days (its own backtest is leave-one-day-out, but the final
# MAGIC registered model's climatology profile is built from every day). There is no strictly
# MAGIC held-out day to gate on here — this evaluates both models on the same pinned snapshot of
# MAGIC `gold_demand_15m` instead, which is still meaningful (it's exactly what the LODO backtest
# MAGIC itself does) but isn't a true generalization test the way M1's holdout is.
# MAGIC
# MAGIC 1. Load `@challenger` and (if one exists) `@champion`.
# MAGIC 2. For every (day, cut-bin) window — same `cut_bins` convention as `forecast_demand.py`'s
# MAGIC    own backtest — score both models' `.predict()` and compare to a seasonal-naive
# MAGIC    baseline built from the *other* days, via MASE (the metric contract for M2, per
# MAGIC    docs/MLOPS_PLAN.md Track 4).
# MAGIC 3. Average MASE per **horizon bucket** (0-1h / 1-2h / 2-3h) across all windows — the M2
# MAGIC    analogue of M1's "distance band": a win on the 3-hour aggregate shouldn't hide a
# MAGIC    regression in the next-hour forecast, which is what a coordinator acts on soonest.
# MAGIC 4. Same gate + same human-approval mechanism as `promote_eta.py`:
# MAGIC    `src/lib/promotion.py::evaluate_gate`, and `apply=true` only ever applies a passing
# MAGIC    gate on a second, deliberate run.

# COMMAND ----------
# MAGIC %pip install -q scipy
# MAGIC %restart_python

# COMMAND ----------
try:
    dbutils.widgets.text("stream_schema", "skywatch.stream")
    dbutils.widgets.text("read_stream_schema", "")
    dbutils.widgets.text("model_name", "skywatch.ml.demand_forecast")
    dbutils.widgets.text("holdout_table_version", "")   # blank = current table state
    dbutils.widgets.text("cut_bins", "32,44,56,68,80")
    dbutils.widgets.text("min_improvement_pct", "2.0")
    dbutils.widgets.text("noise_band_pct", "1.0")
    dbutils.widgets.text("max_band_regression_mase", "0.05")
    dbutils.widgets.text("apply", "false")
    STREAM = dbutils.widgets.get("stream_schema")
    READ_STREAM = dbutils.widgets.get("read_stream_schema").strip() or STREAM
    MODEL_NAME = dbutils.widgets.get("model_name")
    HOLDOUT_VERSION = dbutils.widgets.get("holdout_table_version").strip()
    CUTS = [int(x) for x in dbutils.widgets.get("cut_bins").split(",")]
    MIN_IMPROVEMENT_PCT = float(dbutils.widgets.get("min_improvement_pct"))
    NOISE_BAND_PCT = float(dbutils.widgets.get("noise_band_pct"))
    MAX_BAND_REGRESSION_MASE = float(dbutils.widgets.get("max_band_regression_mase"))
    APPLY = dbutils.widgets.get("apply").lower() == "true"
except Exception:
    STREAM, MODEL_NAME, HOLDOUT_VERSION, CUTS = (
        "skywatch.stream", "skywatch.ml.demand_forecast", "", [32, 44, 56, 68, 80],
    )
    READ_STREAM = STREAM
    MIN_IMPROVEMENT_PCT, NOISE_BAND_PCT, MAX_BAND_REGRESSION_MASE, APPLY = 2.0, 1.0, 0.05, False

print(f"{MODEL_NAME} | table version {HOLDOUT_VERSION or 'current'} | cuts {CUTS} | apply={APPLY}")

# COMMAND ----------
# MAGIC %run ./demand_lib

# COMMAND ----------
# MAGIC %md ## 1. Load challenger + champion (if one exists)

# COMMAND ----------
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

challenger_mv = client.get_model_version_by_alias(MODEL_NAME, "challenger")
challenger = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@challenger")
print(f"challenger: v{challenger_mv.version}")

try:
    champion_mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    champion = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
    print(f"champion:   v{champion_mv.version}")
except Exception:  # noqa: BLE001 — no @champion alias set yet (first run for this model)
    champion_mv, champion = None, None
    print("champion:   none yet — this will be a bootstrap promotion if the challenger scores at all")

if challenger_mv.version == getattr(champion_mv, "version", None):
    print("challenger IS the current champion — nothing to evaluate")
    dbutils.notebook.exit("already_champion")

# COMMAND ----------
# MAGIC %md ## 2. Score both models over every (day, cut) window

# COMMAND ----------
import pandas as pd

version_clause = f"VERSION AS OF {HOLDOUT_VERSION}" if HOLDOUT_VERSION else ""
demand = spark.sql(f"SELECT * FROM {READ_STREAM}.gold_demand_15m {version_clause}").toPandas()
demand["arrivals"] = demand["arrivals"].astype(float)
frames = to_day_frames(demand)
days = sorted(frames)
print(f"{len(days)} days available: {[str(d) for d in days]}")

if len(days) < 2:
    print("need at least 2 days to compute a seasonal-naive baseline — nothing to gate on")
    dbutils.notebook.exit("insufficient_days")

BUCKET_BINS = 4  # 4 x 15-min bins = 1 h per bucket
BUCKET_LABELS = ["0-1h", "1-2h", "2-3h"]


def bucket_mase(actual, pred, naive):
    out = {}
    for i, label in enumerate(BUCKET_LABELS):
        sl = slice(i * BUCKET_BINS, (i + 1) * BUCKET_BINS)
        if sl.stop <= len(actual):
            out[label] = mase(actual[sl], pred[sl], naive[sl])
    out["ALL"] = mase(actual, pred, naive)
    return out


rows = {"challenger": [], "champion": []}
for d in days:
    ref = [x for x in days if x != d]
    y = frames[d].set_index("slot")["arrivals"].to_numpy(float)
    dow = int(frames[d]["dow"].iloc[0])
    for T in CUTS:
        H = 12
        if T + H > BINS_PER_DAY:
            continue
        slots = list(range(T, T + H))
        actual = y[slots]
        naive = seasonal_naive(frames, ref, slots)
        req = pd.DataFrame([{"context": y[:T].tolist(), "dow": dow, "horizon": H}])

        for name, model in [("challenger", challenger), ("champion", champion)]:
            if model is None:
                continue
            pred = model.predict(req).iloc[0]["mean"]
            rows[name].append(bucket_mase(actual, pred, naive))

challenger_bands = pd.DataFrame(rows["challenger"]).mean().to_dict()
champion_bands = pd.DataFrame(rows["champion"]).mean().to_dict() if rows["champion"] else None

report = pd.DataFrame({"challenger_mase": challenger_bands})
if champion_bands:
    report["champion_mase"] = pd.Series(champion_bands)
    report["delta"] = report["challenger_mase"] - report["champion_mase"]
display(report.round(3))

# COMMAND ----------
# MAGIC %md ## 3. Gate — same decision function M1 uses (src/lib/promotion.py)

# COMMAND ----------
from lib.promotion import evaluate_gate

gate = evaluate_gate(
    challenger_bands, champion_bands,
    MIN_IMPROVEMENT_PCT, NOISE_BAND_PCT, MAX_BAND_REGRESSION_MASE,
)
passed, reasons = gate.passed, gate.reasons

verdict = "PASS" if passed else "HOLD"
print(f"\nGATE: {verdict}")
for r in reasons:
    print(f"  - {r}")

# COMMAND ----------
# MAGIC %md ## 4. Apply (only on a passing gate, and only if explicitly asked)

# COMMAND ----------
client.set_model_version_tag(MODEL_NAME, challenger_mv.version, "gate_verdict", verdict)
client.set_model_version_tag(MODEL_NAME, challenger_mv.version, "gate_holdout_version", HOLDOUT_VERSION or "current")

if passed and APPLY:
    client.set_registered_model_alias(MODEL_NAME, "champion", challenger_mv.version)
    print(f"\nPROMOTED {MODEL_NAME} v{challenger_mv.version} to @champion")
elif passed:
    print(f"\nGate passed — re-run with apply=true to actually promote v{challenger_mv.version}")
else:
    print(f"\nHolding @champion at v{getattr(champion_mv, 'version', '(none)')} — challenger not promoted")
    if APPLY:
        print("(apply=true was set, but a failing gate is never applied)")
