# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — model health monitoring (DIY substitute for Lakehouse Monitoring)
# MAGIC
# MAGIC Lakehouse Monitoring is a paid Free-Edition-unavailable feature (docs/MLOPS_PLAN.md
# MAGIC Track 5) — this is the substitute: a live, low-frequency job writing to
# MAGIC `{catalog}.ml.model_health`, which a dashboard tile and a Databricks SQL Alert both read.
# MAGIC
# MAGIC **Scoped to Model 1 (ETA) for this first version** — same "build M1 first, mirror later"
# MAGIC pattern as the rest of this project. M2 monitoring (MASE drift) is a fast-follow.
# MAGIC
# MAGIC | Check | Method |
# MAGIC |---|---|
# MAGIC | **Feature drift** | PSI (`src/lib/drift.py`) per `ETA_FEATURES` column, reference = the training population (`gold_arrival_tracks`, confirmed touchdowns, older than the drift window) vs comparison = live `predictions` scored inside the drift window |
# MAGIC | **Performance** | Rolling MAE from `predictions_scored` (current `@champion`'s version only) vs that version's own registered `test_mae_min` tag |
# MAGIC | **Freshness / volume** | Hours since the last live report; today's report count vs the trailing 7-day median (this project's poller runs in bursts, not continuously — an SLA-style "must be < N minutes old" would constantly false-alarm, so this is an anomaly check against the collection pattern this project actually has, not a uptime check) |
# MAGIC
# MAGIC This notebook is `dev`-only by design — monitoring "the live feed" is inherently about the
# MAGIC one environment that has one (staging has no poller of its own, see docs/MLOPS_PLAN.md
# MAGIC Track 3).

# COMMAND ----------
try:
    dbutils.widgets.text("stream_schema", "skywatch.stream")
    dbutils.widgets.text("read_stream_schema", "")
    dbutils.widgets.text("eta_model_name", "skywatch.ml.eta_touchdown")
    dbutils.widgets.text("apt_icao", "KATL")
    dbutils.widgets.text("drift_window_days", "7")
    # PSI on ~10 bins wants a few hundred rows before it's stable rather than sampling noise —
    # this project's live poller runs in short bursts, so early on this will often mean
    # "insufficient data" rather than a real reading, which is the honest outcome (see §2 below).
    dbutils.widgets.text("min_rows_for_drift", "200")
    dbutils.widgets.text("min_predictions_for_performance", "20")
    dbutils.widgets.text("mae_regression_pct", "20.0")
    dbutils.widgets.text("freshness_stale_hours", "24")
    STREAM = dbutils.widgets.get("stream_schema")
    READ_STREAM = dbutils.widgets.get("read_stream_schema").strip() or STREAM
    MODEL_NAME = dbutils.widgets.get("eta_model_name")
    APT_ICAO = dbutils.widgets.get("apt_icao")
    DRIFT_WINDOW_DAYS = int(dbutils.widgets.get("drift_window_days"))
    MIN_ROWS = int(dbutils.widgets.get("min_rows_for_drift"))
    MIN_PREDICTIONS = int(dbutils.widgets.get("min_predictions_for_performance"))
    MAE_REGRESSION_PCT = float(dbutils.widgets.get("mae_regression_pct"))
    FRESHNESS_STALE_HOURS = float(dbutils.widgets.get("freshness_stale_hours"))
except Exception:
    STREAM, MODEL_NAME, APT_ICAO = "skywatch.stream", "skywatch.ml.eta_touchdown", "KATL"
    READ_STREAM = STREAM
    DRIFT_WINDOW_DAYS, MIN_ROWS, MIN_PREDICTIONS = 7, 200, 20
    MAE_REGRESSION_PCT, FRESHNESS_STALE_HOURS = 20.0, 24.0

CATALOG = STREAM.split(".")[0]
HEALTH_TABLE = f"{CATALOG}.ml.model_health"
print(f"monitoring {MODEL_NAME} | drift window {DRIFT_WINDOW_DAYS}d | writing to {HEALTH_TABLE}")

# COMMAND ----------
# MAGIC %run ./eta_features

# COMMAND ----------
# MAGIC %md ## 1. Current champion

# COMMAND ----------
import datetime as dt
import math

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

try:
    champion_mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    champion_version = int(champion_mv.version)
    champion_test_mae = float(
        client.get_model_version(MODEL_NAME, champion_mv.version).tags.get("test_mae_min", "nan")
    )
    print(f"champion v{champion_version}, registered test MAE {champion_test_mae:.3f} min")
except Exception:  # noqa: BLE001 — no @champion yet; performance check will just skip
    champion_version, champion_test_mae = None, float("nan")
    print("no @champion registered yet — performance check will be skipped")

now = dt.datetime.now(dt.timezone.utc)
rows = []  # each: {checked_at, check_type, metric_name, value, severity, detail}

# COMMAND ----------
# MAGIC %md ## 2. Feature drift — PSI per feature, training population vs recent live predictions
# MAGIC **Honest note**: this project's live poller runs in short bursts, not continuously, so
# MAGIC the "recent live" comparison population is often small — a genuinely correct PSI
# MAGIC computation on too few rows is mostly sampling noise, not a real signal. `min_rows_for_drift`
# MAGIC exists to say "insufficient data" instead of confidently reporting noise as drift.

# COMMAND ----------
from lib.drift import psi, psi_severity

cutoff = (now - dt.timedelta(days=DRIFT_WINDOW_DAYS)).date().isoformat()

# gold_arrival_tracks only has the RAW columns (bearing_to_apt, ac_type, ...) — is_heavy,
# bearing_sin/cos, hour_sin/cos, dow_sin/cos, closure_geom_kt are derived by add_eta_features()
# at load time (same as train_eta.py). predictions already carries the full derived vector
# (score_eta.py writes it), so this must go through the same transform to compare apples-to-apples.
reference_sdf = add_eta_features(
    spark.sql(f"""
        SELECT t.* FROM {READ_STREAM}.gold_arrival_tracks t
        JOIN {READ_STREAM}.gold_touchdowns td ON td.seg_id = t.seg_id
        WHERE td.touchdown_confidence = 'confirmed'
          AND to_date(t.snapshot_ts) < '{cutoff}'
    """)
)
reference_pdf = reference_sdf.select(*ETA_FEATURES).toPandas()

comparison_pdf = spark.sql(f"""
    SELECT * FROM {STREAM}.predictions
    WHERE scored_at >= '{cutoff}' AND model_version = {champion_version if champion_version else -1}
""").toPandas() if champion_version else reference_pdf.iloc[0:0]

DRIFT_FEATURES = [f for f in ETA_FEATURES if f != "phase"]  # phase is categorical, skip for PSI

if len(reference_pdf) < MIN_ROWS or len(comparison_pdf) < MIN_ROWS:
    print(f"skipping drift — reference {len(reference_pdf)} rows, comparison {len(comparison_pdf)} "
          f"rows (need >= {MIN_ROWS} each)")
else:
    for feat in DRIFT_FEATURES:
        value = psi(
            reference_pdf[feat].astype(float).to_numpy(),
            comparison_pdf[feat].astype(float).to_numpy(),
        )
        severity = psi_severity(value)
        rows.append({
            "checked_at": now, "check_type": "feature_drift", "metric_name": feat,
            "value": value, "severity": severity,
            "detail": f"PSI {value:.3f} ({len(reference_pdf)} ref rows, {len(comparison_pdf)} live rows)",
        })
    n_flagged = sum(1 for r in rows if r["severity"] == "significant")
    print(f"drift: {len(DRIFT_FEATURES)} features checked, {n_flagged} significant")

# COMMAND ----------
# MAGIC %md ## 3. Performance — rolling MAE vs the champion's registered test MAE

# COMMAND ----------
if champion_version:
    perf = spark.sql(f"""
        SELECT count(*) AS n, avg(abs(error_min)) AS rolling_mae
        FROM {STREAM}.predictions_scored
        WHERE model_version = {champion_version} AND scored_at >= '{cutoff}'
    """).first()

    if perf["n"] and perf["n"] >= MIN_PREDICTIONS and not math.isnan(champion_test_mae):
        rolling_mae = float(perf["rolling_mae"])
        regression_pct = 100 * (rolling_mae - champion_test_mae) / champion_test_mae
        severity = "significant" if regression_pct > MAE_REGRESSION_PCT else "none"
        rows.append({
            "checked_at": now, "check_type": "performance", "metric_name": "rolling_mae_min",
            "value": rolling_mae, "severity": severity,
            "detail": f"rolling MAE {rolling_mae:.3f} vs registered {champion_test_mae:.3f} "
                      f"({regression_pct:+.1f}%, n={perf['n']}, last {DRIFT_WINDOW_DAYS}d)",
        })
        print(f"performance: rolling MAE {rolling_mae:.3f} vs {champion_test_mae:.3f} "
              f"({regression_pct:+.1f}%) — {severity}")
    else:
        print(f"performance: {perf['n'] or 0} matched predictions for v{champion_version} in "
              f"the last {DRIFT_WINDOW_DAYS}d (need >= {MIN_PREDICTIONS}) — insufficient data")
else:
    print("performance: skipped, no @champion")

# COMMAND ----------
# MAGIC %md ## 4. Freshness / volume
# MAGIC Anomaly check against this project's own bursty collection pattern, not an uptime SLA —
# MAGIC see the header note.

# COMMAND ----------
fresh = spark.sql(f"""
    SELECT max(snapshot_ts) AS latest
    FROM {READ_STREAM}.gold_tracks
    WHERE apt_icao = '{APT_ICAO}'
""").first()

today = spark.sql(f"""
    SELECT count(*) AS n_today
    FROM {READ_STREAM}.gold_tracks
    WHERE apt_icao = '{APT_ICAO}' AND to_date(snapshot_ts) = current_date()
""").first()

daily_counts = spark.sql(f"""
    SELECT to_date(snapshot_ts) AS d, count(*) AS n
    FROM {READ_STREAM}.gold_tracks
    WHERE apt_icao = '{APT_ICAO}'
      AND snapshot_ts >= current_date() - INTERVAL 7 DAYS
      AND snapshot_ts < current_date()
    GROUP BY 1
""").toPandas()

if fresh["latest"] is not None:
    hours_since = (now - fresh["latest"].replace(tzinfo=dt.timezone.utc)).total_seconds() / 3600
    stale = hours_since > FRESHNESS_STALE_HOURS
    rows.append({
        "checked_at": now, "check_type": "freshness", "metric_name": "hours_since_last_report",
        "value": hours_since, "severity": "significant" if stale else "none",
        "detail": f"latest report {hours_since:.1f}h ago (threshold {FRESHNESS_STALE_HOURS}h)",
    })
    print(f"freshness: latest report {hours_since:.1f}h ago")

if len(daily_counts) >= 3:
    median_n = float(daily_counts["n"].median())
    today_n = today["n_today"] or 0
    low_volume = median_n > 0 and today_n < 0.2 * median_n
    rows.append({
        "checked_at": now, "check_type": "volume", "metric_name": "reports_today",
        "value": float(today_n), "severity": "significant" if low_volume else "none",
        "detail": f"{today_n} reports today vs 7-day median {median_n:.0f}",
    })
    print(f"volume: {today_n} reports today vs 7-day median {median_n:.0f}")
else:
    print(f"volume: only {len(daily_counts)} prior days with data — skipping median comparison")

# COMMAND ----------
# MAGIC %md ## 5. Write to `model_health`

# COMMAND ----------
import pandas as pd

if rows:
    out = spark.createDataFrame(pd.DataFrame(rows))
    (out.write.mode("append").option("mergeSchema", "true").saveAsTable(HEALTH_TABLE))
    n_significant = sum(1 for r in rows if r["severity"] == "significant")
    print(f"\nwrote {len(rows)} health rows to {HEALTH_TABLE} ({n_significant} significant)")
    display(out.orderBy(out["severity"].desc()))
else:
    print("\nnothing to write this run (insufficient data for every check)")
