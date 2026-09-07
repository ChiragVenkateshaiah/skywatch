# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — Gold batch build
# MAGIC
# MAGIC Reads `{catalog}.{schema}.silver_positions` and (re)builds the history-dependent Gold
# MAGIC tables. Plain SQL — runs on a serverless SQL warehouse or serverless notebook compute; it
# MAGIC is **not** part of the Lakeflow pipeline (that one is Bronze + Silver, streaming). Scheduled
# MAGIC by `resources/skywatch.gold.job.yml`; also runnable ad hoc.
# MAGIC
# MAGIC ## Two modes (`mode` widget)
# MAGIC
# MAGIC | mode | what it does | when |
# MAGIC |---|---|---|
# MAGIC | **`full`** (default) | full `CREATE OR REPLACE` of every table from all of history | before a model retrain; after a backfill; first run |
# MAGIC | **`serving`** | only the last `serving_hours` of `gold_tracks` / `gold_congestion` / `gold_touchdowns` / `gold_holding` / `gold_kpis`, via Delta `REPLACE WHERE`; **skips** the training tables | behind the live poller, every N min |
# MAGIC
# MAGIC `serving` mode never rebuilds `gold_arrival_tracks` (Model 1 set) or `gold_demand_15m`
# MAGIC (Model 2 series) — those only matter at retrain time, so re-deriving them every 10 min is
# MAGIC wasted compute. Run `mode=full` before retraining.
# MAGIC
# MAGIC ## Incrementality — how the history-dependent columns stay correct
# MAGIC
# MAGIC `gold_tracks` uses per-`icao` `LAG` and a sessionised `seg_id`. `serving` mode scans
# MAGIC `silver_positions` from `serving_hours + seg_lookback_hours` back so every `LAG` and every
# MAGIC segment boundary inside the write window is computed from real neighbours, then writes only
# MAGIC `snapshot_ts >= cutoff`. `seg_id` is `icao + '-' + <segment-start timestamp>` (not a running
# MAGIC count), so it is identical no matter how much history the scan covers — a segment keeps its
# MAGIC id across runs as long as its first report is within the lookback (true for every airborne
# MAGIC approach; a multi-hour *grounded* ping-blob can churn its id, but nothing consumes that).
# MAGIC
# MAGIC | Table | Grain | Cluster key |
# MAGIC |---|---|---|
# MAGIC | `gold_tracks` | one row per report | `icao, snapshot_ts` |
# MAGIC | `gold_congestion` | minute × ring | `apt_icao, minute_ts` |
# MAGIC | `gold_holding` | one row per circling segment | — (small) |
# MAGIC | `gold_touchdowns` | one row per landing | `apt_icao, touchdown_ts` |
# MAGIC | `gold_arrival_tracks` | one row per (aircraft, time) pre-touchdown — **Model 1** set | `apt_icao, snapshot_ts` |
# MAGIC | `gold_demand_15m` | one row per 15-min bin per active day — **Model 2** series | `apt_icao, bin_start_ts` |
# MAGIC | `gold_kpis` | one row | — |

# COMMAND ----------
# MAGIC %md ## Parameters

# COMMAND ----------
import datetime as dt

try:
    dbutils.widgets.text("catalog", "skywatch")
    dbutils.widgets.text("schema", "stream")
    dbutils.widgets.text("apt_elev_ft", "1026")            # KATL field elevation
    dbutils.widgets.dropdown("mode", "full", ["full", "serving"])
    dbutils.widgets.text("serving_hours", "6")             # serving: write window, hours back from the anchor
    dbutils.widgets.text("seg_lookback_hours", "2")        # serving: extra scan-back so LAG / seg_id are stable
    dbutils.widgets.text("serving_anchor_ts", "")          # serving: pin "now" for validation (blank = current_timestamp)
    CATALOG = dbutils.widgets.get("catalog")
    SCHEMA = dbutils.widgets.get("schema")
    APT_ELEV_FT = int(dbutils.widgets.get("apt_elev_ft"))
    MODE = dbutils.widgets.get("mode")
    SERVING_HOURS = float(dbutils.widgets.get("serving_hours"))
    SEG_LOOKBACK_HOURS = float(dbutils.widgets.get("seg_lookback_hours"))
    ANCHOR_RAW = dbutils.widgets.get("serving_anchor_ts").strip()
except Exception:
    CATALOG, SCHEMA, APT_ELEV_FT = "skywatch", "stream", 1026
    MODE, SERVING_HOURS, SEG_LOOKBACK_HOURS, ANCHOR_RAW = "full", 6.0, 2.0, ""

S = f"{CATALOG}.{SCHEMA}"
TBLPROPS = "delta.autoOptimize.optimizeWrite = true, delta.autoOptimize.autoCompact = true"

if MODE == "serving":
    anchor = (
        dt.datetime.fromisoformat(ANCHOR_RAW.replace("Z", "+00:00"))
        if ANCHOR_RAW else dt.datetime.now(dt.timezone.utc)
    ).replace(second=0, microsecond=0)   # minute-align so date_trunc('MINUTE', ..) buckets line up
    CUTOFF_TS = (anchor - dt.timedelta(hours=SERVING_HOURS)).strftime("%Y-%m-%d %H:%M:00")
    SCAN_FROM_TS = (
        anchor - dt.timedelta(hours=SERVING_HOURS + SEG_LOOKBACK_HOURS)
    ).strftime("%Y-%m-%d %H:%M:00")
    print(f"MODE=serving  write snapshot_ts >= {CUTOFF_TS}  (scan from {SCAN_FROM_TS})")
    SILVER_SCAN = f"has_position AND snapshot_ts >= timestamp('{SCAN_FROM_TS}')"
    TRACKS_SCAN_WIDE = f"snapshot_ts >= timestamp('{SCAN_FROM_TS}')"     # need whole segments
    TRACKS_SCAN_TIGHT = f"snapshot_ts >= timestamp('{CUTOFF_TS}')"       # exact write window
else:
    CUTOFF_TS = SCAN_FROM_TS = None
    print(f"MODE=full  rebuilding all of {S}.gold_*")
    SILVER_SCAN = "has_position"
    TRACKS_SCAN_WIDE = TRACKS_SCAN_TIGHT = "true"


def emit(name: str, select_body: str, cluster_by: str | None, time_col: str) -> None:
    """`full` -> CREATE OR REPLACE; `serving` -> INSERT ... REPLACE WHERE {time_col} >= cutoff."""
    if MODE == "full":
        clause = f"CLUSTER BY ({cluster_by})\n" if cluster_by else ""
        spark.sql(f"CREATE OR REPLACE TABLE {S}.{name}\n{clause}TBLPROPERTIES ({TBLPROPS}) AS\n{select_body}")
    else:
        if not spark.catalog.tableExists(f"{S}.{name}"):
            raise RuntimeError(f"{S}.{name} missing — run this notebook once with mode=full first")
        pred = f"{time_col} >= timestamp('{CUTOFF_TS}')"
        spark.sql(f"INSERT INTO {S}.{name} REPLACE WHERE {pred}\nSELECT * FROM (\n{select_body}\n) WHERE {pred}")
    n = spark.table(f"{S}.{name}").count()
    print(f"  {name}: {n} rows total")


print(f"building Gold in {S}  (field elevation {APT_ELEV_FT} ft)")

# COMMAND ----------
# MAGIC %md ## `gold_tracks` — per-report trajectory context

# COMMAND ----------
emit(
    "gold_tracks",
    f"""
WITH ordered AS (
  SELECT
    icao, callsign, ac_type, apt_icao, snapshot_ts,
    lat, lon, alt_ft, alt_geom_ft, gs_kt, track_deg, sel_altitude_ft,
    vertical_rate_fpm, vertical_rate_src,
    dist_to_apt_nm, bearing_to_apt, heading_err_deg, is_grounded, phase,
    lag(snapshot_ts)    OVER w AS prev_ts,
    lag(alt_ft)         OVER w AS prev_alt_ft,
    lag(dist_to_apt_nm) OVER w AS prev_dist_nm,
    lag(track_deg)      OVER w AS prev_track_deg
  FROM {S}.silver_positions
  WHERE {SILVER_SCAN}
  WINDOW w AS (PARTITION BY icao ORDER BY snapshot_ts)
),
stepped AS (SELECT *, timestampdiff(SECOND, prev_ts, snapshot_ts) AS dt_s FROM ordered),
flagged AS (
  -- a new track starts when there is no usable previous report (first report, or a gap
  -- outside 1..180 s — e.g. the same airframe seen on two disjoint backfill days)
  SELECT *, CASE WHEN dt_s IS NULL OR dt_s < 1 OR dt_s > 180 THEN 1 ELSE 0 END AS seg_break
  FROM stepped
),
segmented AS (
  -- seg_id keyed on the segment's START timestamp, not a running count: stable regardless of
  -- how far back the scan reaches, so an incremental window reproduces the same id. Every
  -- icao's first scanned row has dt_s IS NULL -> seg_break = 1, so the running max is defined.
  SELECT *,
    concat(icao, '-', date_format(
      max(CASE WHEN seg_break = 1 THEN snapshot_ts END)
        OVER (PARTITION BY icao ORDER BY snapshot_ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
      'yyyyMMddHHmmss')) AS seg_id
  FROM flagged
),
derived AS (
  SELECT *,
    -- vertical rate keeps the wider 1..600 s guard (conservative — the touchdown label set
    -- depends on descent_reports; keep it stable)
    CASE WHEN dt_s BETWEEN 1 AND 600 THEN (alt_ft - prev_alt_ft) / dt_s * 60.0 END AS derived_vrate_fpm,
    -- closure / turn / "closing" are gated on seg_break = 0 (dt_s in 1..180, same track) so a
    -- cross-day lag can never produce a value.
    -- closure_kt = a RATE (nm closed per hour = kt), so 60 s backfill and 15 s live cadence
    -- give the same number for the same approach. Clamp implausible values from bad fixes.
    CASE WHEN seg_break = 0
           AND abs((prev_dist_nm - dist_to_apt_nm) / dt_s * 3600) <= 700
         THEN (prev_dist_nm - dist_to_apt_nm) / dt_s * 3600 END AS closure_kt,
    CASE WHEN seg_break = 0 AND prev_track_deg IS NOT NULL
         THEN least(abs(track_deg - prev_track_deg), 360 - abs(track_deg - prev_track_deg)) END AS turn_deg,
    CASE WHEN seg_break = 0 AND prev_dist_nm > dist_to_apt_nm THEN 1 ELSE 0 END AS closing_step
  FROM segmented
)
SELECT
  seg_id, icao, callsign, ac_type, apt_icao, snapshot_ts,
  lat, lon, alt_ft, alt_geom_ft, gs_kt, track_deg, sel_altitude_ft,
  dist_to_apt_nm, bearing_to_apt, heading_err_deg, is_grounded, phase,
  CASE WHEN seg_break = 0 THEN dt_s END AS dt_s,   -- null at a segment boundary (no cross-day step)
  closure_kt, turn_deg,
  coalesce(vertical_rate_fpm, derived_vrate_fpm) AS vrate_fpm,
  coalesce(vertical_rate_src, CASE WHEN derived_vrate_fpm IS NOT NULL THEN 'delta' END) AS vrate_src,
  CASE WHEN turn_deg IS NOT NULL THEN turn_deg / dt_s END AS turn_rate_dps,
  count(*)         OVER (PARTITION BY seg_id) AS seg_n_reports,
  min(snapshot_ts) OVER (PARTITION BY seg_id) AS seg_start_ts,
  max(snapshot_ts) OVER (PARTITION BY seg_id) AS seg_end_ts,
  (heading_err_deg < 70
   AND alt_ft BETWEEN 500 AND 40000
   AND NOT is_grounded
   AND sum(closing_step)
         OVER (PARTITION BY seg_id ORDER BY snapshot_ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) >= 2
  ) AS inbound_flag
FROM derived
""",
    cluster_by="icao, snapshot_ts",
    time_col="snapshot_ts",
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## `gold_congestion` — minute × distance ring
# MAGIC
# MAGIC **Coverage boundary: 100 nm.** The historical backfill (`backfill_local.py`) only ever
# MAGIC captured aircraft within 100 nm. Nothing in the project uses distance beyond ~120 nm
# MAGIC (Model 1's evaluated bands top out at 100 nm, `score_eta.py`'s `max_dist_nm` default is
# MAGIC 120). The live poller captures out to 250 nm, so the `100+` ring is **populated only for
# MAGIC live snapshots** — one merged bucket, informational ("how far out can we see"), not a
# MAGIC modelling feature. `airport_inbound_count` in `gold_arrival_tracks` / `score_eta.py`
# MAGIC already restricts to `00-40`/`40-100` for exactly this reason.

# COMMAND ----------
emit(
    "gold_congestion",
    f"""
SELECT
  date_trunc('MINUTE', snapshot_ts) AS minute_ts,
  apt_icao,
  CASE WHEN dist_to_apt_nm < 40  THEN '00-40'
       WHEN dist_to_apt_nm < 100 THEN '40-100'
       ELSE '100+' END AS ring,   -- live-only / informational, see note above
  count(DISTINCT icao)                                     AS n_aircraft,
  count(DISTINCT CASE WHEN inbound_flag THEN icao END)     AS n_inbound,
  round(avg(alt_ft))                                       AS mean_alt_ft,
  round(avg(gs_kt))                                        AS mean_gs_kt,
  round(avg(CASE WHEN inbound_flag AND gs_kt > 60
                 THEN dist_to_apt_nm / gs_kt * 60 END), 1) AS mean_eta_min
FROM {S}.gold_tracks
WHERE NOT is_grounded AND {TRACKS_SCAN_TIGHT}
GROUP BY 1, 2, 3
""",
    cluster_by="apt_icao, minute_ts",
    time_col="minute_ts",
)

# COMMAND ----------
# MAGIC %md ## `gold_holding` — circling / racetrack detection
# MAGIC Circular variance of `track_deg` (0 = dead straight, 1 = every heading) inside a small
# MAGIC bounding box, **per `seg_id`** — one continuous track. Previously grouped by `icao` over all
# MAGIC of history, which merged an airframe's disjoint backfill days into one row whose bounding
# MAGIC box then blew past the `ns_nm <= 15` gate — silently *suppressing* real circling events.
# MAGIC `serving` mode replaces only rows whose track ended in the write window. Still flags any
# MAGIC circling aircraft — pattern-working light aircraft and military orbits included; restricting
# MAGIC to airline holds is a tuning task.

# COMMAND ----------
emit(
    "gold_holding",
    f"""
WITH agg AS (
  SELECT
    seg_id,
    any_value(icao) AS icao, any_value(callsign) AS callsign, any_value(ac_type) AS ac_type,
    any_value(apt_icao) AS apt_icao,
    count(*) AS n_reports, min(snapshot_ts) AS first_ts, max(snapshot_ts) AS last_ts,
    round(avg(alt_ft))            AS mean_alt_ft,
    round(avg(dist_to_apt_nm), 1) AS mean_dist_nm,
    round(avg(lat), 4)            AS approx_lat,
    round(avg(lon), 4)            AS approx_lon,
    round((max(lat) - min(lat)) * 60, 1)                          AS ns_nm,
    round((max(lon) - min(lon)) * 60 * cos(radians(avg(lat))), 1) AS ew_nm,
    round(1 - sqrt(pow(avg(cos(radians(track_deg))), 2)
                 + pow(avg(sin(radians(track_deg))), 2)), 2)      AS heading_spread
  FROM {S}.gold_tracks
  WHERE NOT is_grounded AND track_deg IS NOT NULL
    AND alt_ft BETWEEN 2000 AND 20000
    AND {TRACKS_SCAN_WIDE}
  GROUP BY seg_id
)
SELECT * FROM agg
WHERE n_reports >= 8 AND heading_spread >= 0.5
  AND ns_nm <= 15 AND ew_nm <= 15 AND mean_dist_nm <= 80
""",
    cluster_by=None,
    time_col="last_ts",
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## `gold_touchdowns` — detected landings
# MAGIC `touchdown_ts` = first on-ground report within 3 nm of the field, else the last airborne
# MAGIC short-final report. Wide candidate window + loose acceptance gate so a sparse-cadence
# MAGIC (180 s) track still lands ≥ 1 report inside it; `touchdown_confidence` records which gate a
# MAGIC row met — `confirmed` (saw it within 3 nm / +500 ft) vs `inferred` (only as close as the
# MAGIC wider acceptance). Use `confirmed` where label precision matters; `inferred` is fine for
# MAGIC 15-minute demand bucketing.

# COMMAND ----------
emit(
    "gold_touchdowns",
    f"""
WITH sf AS (
  SELECT seg_id, icao, callsign, ac_type, apt_icao, snapshot_ts,
         dist_to_apt_nm, alt_ft, gs_kt, is_grounded, vrate_fpm
  FROM {S}.gold_tracks
  WHERE dist_to_apt_nm < 8 AND alt_ft <= {APT_ELEV_FT} + 4000 AND gs_kt BETWEEN 25 AND 250
    AND {TRACKS_SCAN_WIDE}
),
seg AS (
  SELECT
    seg_id, icao,
    any_value(callsign) AS callsign, any_value(ac_type) AS ac_type, any_value(apt_icao) AS apt_icao,
    coalesce(
      min(CASE WHEN is_grounded AND dist_to_apt_nm < 3 THEN snapshot_ts END),
      max(CASE WHEN NOT is_grounded THEN snapshot_ts END)
    ) AS touchdown_ts,
    round(min(dist_to_apt_nm), 2) AS min_dist_nm,
    min(alt_ft)                   AS min_alt_ft,
    count_if(vrate_fpm < -200)    AS descent_reports,
    count(*)                      AS n_short_final
  FROM sf
  GROUP BY seg_id, icao
)
SELECT *,
  CASE WHEN min_dist_nm < 3 AND min_alt_ft <= {APT_ELEV_FT} + 500
       THEN 'confirmed' ELSE 'inferred' END AS touchdown_confidence
FROM seg
WHERE min_dist_nm < 6 AND min_alt_ft <= {APT_ELEV_FT} + 2000 AND descent_reports >= 1
  AND touchdown_ts IS NOT NULL
""",
    cluster_by="apt_icao, touchdown_ts",
    time_col="touchdown_ts",
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Training tables — `full` mode only
# MAGIC `gold_arrival_tracks` (Model 1) and `gold_demand_15m` (Model 2) are only consumed at
# MAGIC retrain time. `serving` mode leaves whatever a prior `full` run wrote in place.

# COMMAND ----------
if MODE != "full":
    print("MODE=serving — skipping gold_arrival_tracks / gold_demand_15m (run mode=full before retraining)")

# COMMAND ----------
# MAGIC %md
# MAGIC ### `gold_arrival_tracks` — the Model 1 training set
# MAGIC Every airborne report of a segment that ends in a detected touchdown, labelled with
# MAGIC `minutes_to_touchdown`. Segments have no internal gap > 3 min, so the label is trustworthy.
# MAGIC Reports 0.5–40 min before touchdown are kept.

# COMMAND ----------
if MODE == "full":
    spark.sql(f"""
CREATE OR REPLACE TABLE {S}.gold_arrival_tracks
CLUSTER BY (apt_icao, snapshot_ts)
TBLPROPERTIES ({TBLPROPS}) AS
WITH inbound_ct AS (
  -- only the rings BOTH data sources cover: backfill keeps aircraft within 100 nm,
  -- the live poller within 250 nm. Counting all rings makes the feature systematically
  -- larger at serving time than the model ever saw. Keep this identical to score_eta.py.
  SELECT minute_ts, apt_icao, sum(n_inbound) AS n_inbound_common_rings
  FROM {S}.gold_congestion
  WHERE ring IN ('00-40', '40-100')
  GROUP BY 1, 2
)
SELECT /*+ BROADCAST(c) */
  t.seg_id, t.icao, t.callsign, t.ac_type, t.apt_icao,
  t.snapshot_ts,
  td.touchdown_ts,
  cast((unix_timestamp(td.touchdown_ts) - unix_timestamp(t.snapshot_ts)) / 60.0 AS double)
                                                     AS minutes_to_touchdown,
  -- features
  t.dist_to_apt_nm, t.bearing_to_apt, t.heading_err_deg,
  t.alt_ft, t.alt_geom_ft, t.sel_altitude_ft,
  t.gs_kt, t.track_deg, t.vrate_fpm, t.turn_rate_dps, t.closure_kt, t.phase,
  hour(t.snapshot_ts)                                 AS hour_utc,
  dayofweek(t.snapshot_ts)                            AS dow,
  coalesce(c.n_inbound_common_rings, 0)               AS airport_inbound_count
FROM {S}.gold_tracks t
JOIN {S}.gold_touchdowns td ON td.seg_id = t.seg_id
LEFT JOIN inbound_ct c
  ON c.minute_ts = date_trunc('MINUTE', t.snapshot_ts) AND c.apt_icao = t.apt_icao
WHERE t.snapshot_ts < td.touchdown_ts
  AND NOT t.is_grounded
  AND t.dist_to_apt_nm IS NOT NULL
  AND (unix_timestamp(td.touchdown_ts) - unix_timestamp(t.snapshot_ts)) BETWEEN 30 AND 2400
""")
    _at = spark.table(f"{S}.gold_arrival_tracks")
    print(f"  gold_arrival_tracks: {_at.count()} rows | {_at.select('seg_id').distinct().count()} arrivals")

# COMMAND ----------
# MAGIC %md
# MAGIC ### `gold_demand_15m` — the Model 2 series
# MAGIC Touchdowns bucketed into 15-minute bins. A full 96-bin spine per date that clears
# MAGIC `MIN_TOUCHDOWNS_FOR_ACTIVE_DAY` (the archive only has the 1st of each month, so the series
# MAGIC is a set of independent full days). The day-boundary filter drops dates that got 1–2 stray
# MAGIC touchdowns from a snapshot whose `now` slipped seconds into the prior day.

# COMMAND ----------
if MODE == "full":
    MIN_TOUCHDOWNS_FOR_ACTIVE_DAY = 20
    spark.sql(f"""
CREATE OR REPLACE TABLE {S}.gold_demand_15m
CLUSTER BY (apt_icao, bin_start_ts)
TBLPROPERTIES ({TBLPROPS}) AS
WITH td AS (
  SELECT apt_icao,
         timestamp_seconds(floor(unix_timestamp(touchdown_ts) / 900) * 900) AS bin_start_ts
  FROM {S}.gold_touchdowns
  WHERE touchdown_ts IS NOT NULL
),
counts AS (SELECT apt_icao, bin_start_ts, count(*) AS arrivals FROM td GROUP BY 1, 2),
active_dates AS (
  SELECT apt_icao, to_date(bin_start_ts) AS d
  FROM td
  GROUP BY 1, 2
  HAVING count(*) >= {MIN_TOUCHDOWNS_FOR_ACTIVE_DAY}
),
spine AS (
  SELECT apt_icao,
         explode(sequence(to_timestamp(d),
                          to_timestamp(d) + INTERVAL 1 DAY - INTERVAL 15 MINUTES,
                          INTERVAL 15 MINUTES)) AS bin_start_ts
  FROM active_dates
)
SELECT
  sp.apt_icao, sp.bin_start_ts,
  coalesce(c.arrivals, 0)    AS arrivals,
  hour(sp.bin_start_ts)      AS hour_utc,
  dayofweek(sp.bin_start_ts) AS dow,
  to_date(sp.bin_start_ts)   AS bin_date
FROM spine sp
LEFT JOIN counts c ON c.apt_icao = sp.apt_icao AND c.bin_start_ts = sp.bin_start_ts
""")
    _d = spark.table(f"{S}.gold_demand_15m")
    print(f"  gold_demand_15m: {_d.count()} bins | {_d.selectExpr('sum(arrivals)').first()[0]} arrivals | "
          f"{_d.select('bin_date').distinct().count()} days")

# COMMAND ----------
# MAGIC %md ## `gold_kpis` — dashboard headline numbers (always rebuilt; one row)

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE TABLE {S}.gold_kpis
TBLPROPERTIES ({TBLPROPS}) AS
WITH s AS (
  SELECT max(snapshot_ts) mx, min(snapshot_ts) mn, count(DISTINCT icao) ac
  FROM {S}.silver_positions
),
last_snap AS (SELECT max(snapshot_ts) mx FROM {S}.gold_tracks)
SELECT
  (SELECT mx FROM s) AS as_of_ts,
  round((unix_timestamp((SELECT mx FROM s)) - unix_timestamp((SELECT mn FROM s))) / 60.0, 1) AS window_minutes,
  (SELECT ac FROM s) AS aircraft_seen,
  (SELECT count(DISTINCT icao) FROM {S}.gold_tracks
     WHERE inbound_flag AND snapshot_ts = (SELECT mx FROM last_snap)) AS inbound_now,
  (SELECT count(*) FROM {S}.gold_holding)     AS holding_now,
  (SELECT count(*) FROM {S}.gold_touchdowns)  AS touchdowns_in_window
""")

# COMMAND ----------
# MAGIC %md ## Validation

# COMMAND ----------
_tables = ["gold_tracks", "gold_congestion", "gold_holding", "gold_touchdowns", "gold_kpis"]
if MODE == "full":
    _tables += ["gold_arrival_tracks", "gold_demand_15m"]
for t in _tables:
    print(f"{t:20} {spark.table(f'{S}.{t}').count():>8} rows")
display(spark.table(f"{S}.gold_kpis"))
