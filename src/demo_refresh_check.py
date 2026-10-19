# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — pre-interview refresh: readiness check
# MAGIC
# MAGIC Last task of `skywatch_demo_refresh` (docs/DEMO_REFRESH_PLAN.md). Runs even when an earlier
# MAGIC task failed (`run_if: ALL_DONE`), so a broken refresh still reports *which* step broke.
# MAGIC Appends one row to `refresh_audit` — status, reasons, per-task timings, data as of — then
# MAGIC fails the job if the verdict is `NOT_READY`, so the Jobs UI shows red too.

# COMMAND ----------
try:
    dbutils.widgets.text("stream_schema", "skywatch.stream")
    dbutils.widgets.text("apt_icao", "KATL")
    dbutils.widgets.text("job_run_id", "")          # {{job.run_id}} from the job definition
    dbutils.widgets.text("budget_min", "10")
    dbutils.widgets.text("max_age_min", "15")
    STREAM = dbutils.widgets.get("stream_schema")
    APT_ICAO = dbutils.widgets.get("apt_icao")
    JOB_RUN_ID = dbutils.widgets.get("job_run_id").strip()
    BUDGET = float(dbutils.widgets.get("budget_min"))
    MAX_AGE = float(dbutils.widgets.get("max_age_min"))
except Exception:
    STREAM, APT_ICAO, JOB_RUN_ID, BUDGET, MAX_AGE = "skywatch.stream", "KATL", "", 10.0, 15.0

# COMMAND ----------
import datetime as dt
import json

from databricks.sdk import WorkspaceClient

from lib.refresh import NOT_READY, readiness, step_timings

w = WorkspaceClient()
now = dt.datetime.now(dt.timezone.utc)

run = w.jobs.get_run(int(JOB_RUN_ID)).as_dict() if JOB_RUN_ID else {"tasks": []}
steps = step_timings(run)
start_ms = run.get("start_time")
elapsed_min = (now.timestamp() * 1000 - start_ms) / 60000 if start_ms else float("nan")


def one(sql: str):
    try:
        return spark.sql(sql).first()
    except Exception as exc:  # noqa: BLE001 — a missing table is a "no rows" reason, not a crash
        print(f"query failed: {type(exc).__name__}: {str(exc)[:120]}")
        return None


def age_min(ts):
    return None if ts is None else (now - ts.replace(tzinfo=dt.timezone.utc)).total_seconds() / 60


gold = one(f"SELECT max(snapshot_ts) AS ts FROM {STREAM}.gold_tracks WHERE apt_icao = '{APT_ICAO}'")
fc = one(f"SELECT max(scored_at) AS ts FROM {STREAM}.demand_forecast")
brief = one(f"""SELECT generated_at AS ts, source FROM {STREAM}.ops_briefings
                WHERE apt_icao = '{APT_ICAO}' ORDER BY generated_at DESC LIMIT 1""")
inb = one(f"""SELECT count(*) AS n FROM {STREAM}.predictions
              WHERE scored_at = (SELECT max(scored_at) FROM {STREAM}.predictions)""")
flags = one(f"""SELECT count(*) AS n FROM {STREAM}.irregularity_flags
                WHERE scored_at >= '{dt.datetime.fromtimestamp(start_ms / 1000, dt.timezone.utc):%Y-%m-%d %H:%M:%S}'"""
            if start_ms else "SELECT 0 AS n")

verdict = readiness(
    steps=steps,
    gold_age_min=age_min(gold["ts"]) if gold else None,
    forecast_age_min=age_min(fc["ts"]) if fc else None,
    briefing_age_min=age_min(brief["ts"]) if brief else None,
    briefing_source=brief["source"] if brief else None,
    n_inbound=int(inb["n"]) if inb else 0,
    elapsed_min=elapsed_min,
    max_age_min=MAX_AGE,
    budget_min=BUDGET,
)

print(f"{verdict.status} in {elapsed_min:.1f} min (budget {BUDGET:.0f})")
for s in steps:
    print(f"  {s['task']:<22} {str(s['state']):<8} setup {s['setup_s']:>4} s  exec {s['exec_s']:>4} s")
for r in verdict.reasons:
    print("  NOT READY:", r)
for x in verdict.warnings:
    print("  warning:", x)

# COMMAND ----------
row = (
    now, APT_ICAO, int(JOB_RUN_ID) if JOB_RUN_ID else None, verdict.status,
    " | ".join(verdict.reasons) or None, " | ".join(verdict.warnings) or None,
    gold["ts"] if gold else None, round(elapsed_min, 2) if elapsed_min == elapsed_min else None,
    int(inb["n"]) if inb else 0, int(flags["n"]) if flags else 0,
    brief["source"] if brief else None, json.dumps(steps),
)
schema = """checked_at TIMESTAMP, apt_icao STRING, job_run_id BIGINT, status STRING, reasons STRING,
            warnings STRING, data_as_of TIMESTAMP, elapsed_min DOUBLE, n_inbound INT, n_flags INT,
            briefing_source STRING, steps_json STRING"""
(spark.createDataFrame([row], schema=schema)
      .write.mode("append").option("mergeSchema", "true")
      .saveAsTable(f"{STREAM}.refresh_audit"))
print(f"appended to {STREAM}.refresh_audit")

if verdict.status == NOT_READY:
    raise RuntimeError("NOT READY: " + "; ".join(verdict.reasons))
