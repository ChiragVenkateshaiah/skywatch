"""SQL-warehouse access for the Arrival Manager app.

Free Edition has no model-serving endpoints, so every panel reads Delta tables that
the batch scoring jobs (`score_eta.py`, `score_demand.py`) write. Queries go through
the SQL Statement Execution API using the app service principal's injected
credentials — no PAT, no connection string.
"""

from __future__ import annotations

import os

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

WAREHOUSE_ID = os.environ["SKYWATCH_WAREHOUSE_ID"]
CATALOG = os.environ.get("SKYWATCH_CATALOG", "skywatch")
SCHEMA = os.environ.get("SKYWATCH_SCHEMA", "stream")

_w = WorkspaceClient()

# Statement Execution returns every value as a string; coerce by the manifest's
# column type so downstream charting / arithmetic just works.
_NUMERIC = {"INT", "SHORT", "LONG", "BYTE", "FLOAT", "DOUBLE", "DECIMAL"}
_TEMPORAL = {"TIMESTAMP", "TIMESTAMP_NTZ", "DATE"}


def query(sql: str) -> pd.DataFrame:
    """Run `sql` on the configured warehouse and return a typed DataFrame."""
    resp = _w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=sql,
        catalog=CATALOG,
        schema=SCHEMA,
        wait_timeout="50s",
    )
    state = resp.status.state
    if state != StatementState.SUCCEEDED:
        err = getattr(resp.status, "error", None)
        raise RuntimeError(f"query failed ({state}): {getattr(err, 'message', err)}")

    columns = resp.manifest.schema.columns or []
    names = [c.name for c in columns]
    rows = (resp.result.data_array if resp.result else None) or []
    df = pd.DataFrame(rows, columns=names)

    for col in columns:
        t = (col.type_name.value if hasattr(col.type_name, "value") else str(col.type_name)).upper()
        if t in _NUMERIC:
            df[col.name] = pd.to_numeric(df[col.name], errors="coerce")
        elif t in _TEMPORAL:
            df[col.name] = pd.to_datetime(df[col.name], errors="coerce", utc=True)
        elif t == "BOOLEAN":
            df[col.name] = df[col.name].map({"true": True, "false": False})
    return df


def latest_predictions() -> pd.DataFrame:
    """Most recent Model 1 scoring run — one row per currently-inbound aircraft."""
    return query(
        """
        SELECT callsign, ac_type, icao,
               dist_to_apt_nm, gs_kt, alt_ft,
               predicted_eta_min, predicted_touchdown_ts, snapshot_ts, scored_at
        FROM predictions
        WHERE scored_at = (SELECT max(scored_at) FROM predictions)
        ORDER BY predicted_touchdown_ts
        """
    )


def latest_demand_forecast() -> pd.DataFrame:
    """Most recent Model 2 scoring run — the next horizon of 15-min arrival bins."""
    return query(
        """
        SELECT bin_start_ts, horizon_step, arrivals_so_far,
               predicted_q10, predicted_q50, predicted_q90, predicted_mean,
               model_version, scored_at
        FROM demand_forecast
        WHERE scored_at = (SELECT max(scored_at) FROM demand_forecast)
        ORDER BY horizon_step
        """
    )


def historical_hourly_profile(apt_icao: str) -> pd.DataFrame:
    """Mean arrivals per 15-min bin by UTC hour, across every collected day."""
    return query(
        f"""
        SELECT hour_utc, avg(arrivals) AS avg_arrivals_per_bin
        FROM gold_demand_15m
        WHERE apt_icao = '{apt_icao}'
        GROUP BY hour_utc
        ORDER BY hour_utc
        """
    )
