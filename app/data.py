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
from databricks.sdk.service.sql import StatementParameterListItem, StatementState

WAREHOUSE_ID = os.environ["SKYWATCH_WAREHOUSE_ID"]
CATALOG = os.environ.get("SKYWATCH_CATALOG", "skywatch")
SCHEMA = os.environ.get("SKYWATCH_SCHEMA", "stream")

_w = WorkspaceClient()

# Statement Execution returns every value as a string; coerce by the manifest's
# column type so downstream charting / arithmetic just works.
_NUMERIC = {"INT", "SHORT", "LONG", "BYTE", "FLOAT", "DOUBLE", "DECIMAL"}
_TEMPORAL = {"TIMESTAMP", "TIMESTAMP_NTZ", "DATE"}


def _execute(sql: str, params: dict | None = None):
    """Statement Execution API call; `params` become named parameter markers (`:name`), so
    free text such as a chat question is never pasted into the SQL."""
    resp = _w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=sql,
        catalog=CATALOG,
        schema=SCHEMA,
        wait_timeout="50s",
        parameters=[StatementParameterListItem(name=k, value=None if v is None else str(v))
                    for k, v in (params or {}).items()] or None,
    )
    state = resp.status.state
    if state != StatementState.SUCCEEDED:
        err = getattr(resp.status, "error", None)
        raise RuntimeError(f"query failed ({state}): {getattr(err, 'message', err)}")
    return resp


def execute(sql: str, params: dict | None = None) -> None:
    """A statement with no result worth reading (INSERT / UPDATE)."""
    _execute(sql, params)


def query_raw(sql: str) -> tuple[list[str], list[list]]:
    """Column names + rows exactly as the warehouse returns them (strings) — what the chat's
    tools hand the model, and what its grounding check compares against."""
    resp = _execute(sql)
    names = [c.name for c in (resp.manifest.schema.columns or [])] if resp.manifest else []
    return names, (resp.result.data_array if resp.result else None) or []


def query(sql: str) -> pd.DataFrame:
    """Run `sql` on the configured warehouse and return a typed DataFrame."""
    resp = _execute(sql)

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


# the model's 18 features (order matters for pred_contrib); phase is categorical.
# Keep in sync with src/eta_features.py:ETA_FEATURES / PHASE_CATEGORIES.
ETA_FEATURES = [
    "dist_to_apt_nm", "heading_err_deg", "bearing_sin", "bearing_cos", "alt_ft", "gs_kt",
    "vrate_fpm", "closure_kt", "closure_geom_kt", "turn_rate_dps", "sel_altitude_ft",
    "airport_inbound_count", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_heavy", "phase",
]
PHASE_CATEGORIES = ["ground", "climb", "descent", "cruise", "level", "unknown"]


def latest_predictions() -> pd.DataFrame:
    """Most recent Model 1 scoring run — one row per currently-inbound aircraft,
    with lat/lon (for the map) and the full feature vector (for click-to-predict)."""
    cols = ("callsign, ac_type, icao, lat, lon, "
            "predicted_eta_min, predicted_touchdown_ts, snapshot_ts, scored_at, "
            + ", ".join(ETA_FEATURES))
    return query(
        f"""
        SELECT {cols}
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


def latest_irregularity_flags(apt_icao: str) -> pd.DataFrame:
    """Model 3 (rule-based) — aircraft flagged holding / go-around / emergency in the last run."""
    return query(
        f"""
        SELECT kind, severity, callsign, ac_type, icao, lat, lon,
               dist_to_apt_nm, alt_ft, gs_kt, detail, snapshot_ts, scored_at
        FROM irregularity_flags
        WHERE apt_icao = '{apt_icao}'
          AND scored_at = (SELECT max(scored_at) FROM irregularity_flags WHERE apt_icao = '{apt_icao}')
        ORDER BY severity DESC, dist_to_apt_nm
        """
    )


def latest_briefing(apt_icao: str) -> pd.DataFrame:
    """Newest LLM ops briefing (src/brief_ops.py) — one row: the text, which path wrote it
    (`llm` / `template`), and the data-age columns the panel renders itself."""
    return query(
        f"""
        SELECT generated_at, briefing, source, endpoint, check_reasons, facts_json,
               data_as_of, max_age_min, source_skew_min, stale
        FROM ops_briefings
        WHERE apt_icao = '{apt_icao}'
        ORDER BY generated_at DESC
        LIMIT 1
        """
    )
