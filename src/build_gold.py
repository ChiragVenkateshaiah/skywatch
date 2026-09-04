# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — Gold batch build
# MAGIC
# MAGIC Reads `{catalog}.{schema}.silver_positions` and (re)builds the history-dependent Gold
# MAGIC tables. Plain `CREATE OR REPLACE TABLE` — runs on a serverless SQL warehouse or serverless
# MAGIC notebook compute; it is **not** part of the Lakeflow pipeline (that one is Bronze + Silver,
# MAGIC streaming). Scheduled by `resources/skywatch.gold.job.yml`; also runnable ad hoc.
# MAGIC
# MAGIC | Table | Grain | Notes |
# MAGIC |---|---|---|
# MAGIC | `gold_tracks` | one row per report | + Δt, Δalt-derived vertical rate, along-track closure rate, turn rate, `seg_id` (gap > 3 min splits), `inbound_flag` |
# MAGIC | `gold_congestion` | minute × ring | inbound / total counts, mean alt / gs / ETA per ring |
# MAGIC | `gold_holding` | one row per circling aircraft | circular-variance heading spread in a small box — currently flags *any* circling (light a/c, military); airline-hold tuning is a TODO |
# MAGIC | `gold_touchdowns` | one row per landing | widened for sparse-cadence recall; `touchdown_confidence` = confirmed / inferred |
# MAGIC | `gold_arrival_tracks` | one row per (aircraft, time) pre-touchdown | the **Model 1** training set; label `minutes_to_touchdown` |
# MAGIC | `gold_demand_15m` | one row per 15-min bin per active day | the **Model 2** series; zero bins explicit |
# MAGIC | `gold_kpis` | one row | dashboard headline numbers |

# COMMAND ----------
# MAGIC %md ## Parameters

# COMMAND ----------
try:
    dbutils.widgets.text("catalog", "skywatch")
    dbutils.widgets.text("schema", "stream")
    dbutils.widgets.text("apt_elev_ft", "1026")        # KATL field elevation
    CATALOG = dbutils.widgets.get("catalog")
    SCHEMA = dbutils.widgets.get("schema")
    APT_ELEV_FT = int(dbutils.widgets.get("apt_elev_ft"))
except Exception:
    CATALOG, SCHEMA, APT_ELEV_FT = "skywatch", "stream", 1026

S = f"{CATALOG}.{SCHEMA}"
print(f"building Gold in {S}  (field elevation {APT_ELEV_FT} ft)")

# COMMAND ----------
# MAGIC %md ## `gold_tracks` — per-report trajectory context

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE TABLE {S}.gold_tracks AS
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
  WHERE has_position
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
  SELECT *,
    concat(icao, '-', cast(sum(seg_break) OVER (PARTITION BY icao ORDER BY snapshot_ts) AS string)) AS seg_id
  FROM flagged
),
derived AS (
  SELECT *,
    -- vertical rate keeps the wider 1..600 s guard (conservative — the touchdown label set
    -- depends on descent_reports; keep it stable across this PR)
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
""")
print(spark.table(f"{S}.gold_tracks").count(), "rows")

# COMMAND ----------
# MAGIC %md
# MAGIC ## `gold_congestion` — minute × distance ring
# MAGIC
# MAGIC **Coverage boundary: 100 nm.** The historical backfill (`backfill_local.py`) only ever
# MAGIC captured aircraft within 100 nm — pulling further out means re-downloading every source
# MAGIC file (the radius filter is applied *after* download, so it doesn't reduce transfer), which
# MAGIC isn't worth it: nothing in the project actually uses distance beyond ~120 nm (Model 1's
# MAGIC evaluated bands top out at 100 nm, `score_eta.py`'s `max_dist_nm` default is 120). The live
# MAGIC poller captures out to 250 nm, so the `100-200` / `200-250` rings below are **populated only
# MAGIC for live snapshots** — on every backfill day those two ring rows are simply absent (not
# MAGIC zero — `GROUP BY` never emits a ring with no matching aircraft). Do not build a model
# MAGIC feature or a cross-day aggregate on those two rings; `airport_inbound_count` in
# MAGIC `gold_arrival_tracks` / `score_eta.py` already restricts to `00-40`/`40-100` for exactly
# MAGIC this reason. The two outer buckets are informational (live "how far out can we see")
# MAGIC only. *(Simplification tracked for the PR 5 gold rebuild: merge them into one `100+`
# MAGIC bucket, or drop them, so the schema stops implying coverage it doesn't have.)*

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE TABLE {S}.gold_congestion AS
SELECT
  date_trunc('MINUTE', snapshot_ts) AS minute_ts,
  apt_icao,
  CASE WHEN dist_to_apt_nm < 40  THEN '00-40'
       WHEN dist_to_apt_nm < 100 THEN '40-100'
       WHEN dist_to_apt_nm < 200 THEN '100-200'   -- live-only, see note above
       ELSE '200-250' END AS ring,                -- live-only, see note above
  count(DISTINCT icao)                                     AS n_aircraft,
  count(DISTINCT CASE WHEN inbound_flag THEN icao END)     AS n_inbound,
  round(avg(alt_ft))                                       AS mean_alt_ft,
  round(avg(gs_kt))                                        AS mean_gs_kt,
  round(avg(CASE WHEN inbound_flag AND gs_kt > 60
                 THEN dist_to_apt_nm / gs_kt * 60 END), 1) AS mean_eta_min
FROM {S}.gold_tracks
WHERE NOT is_grounded
GROUP BY 1, 2, 3
""")
print(spark.table(f"{S}.gold_congestion").count(), "rows")

# COMMAND ----------
# MAGIC %md ## `gold_holding` — circling / racetrack detection
# MAGIC Circular variance of `track_deg` (0 = dead straight, 1 = every heading) inside a small
# MAGIC bounding box. **Currently flags any circling aircraft** — pattern-working light aircraft
# MAGIC and military orbits included. Restricting to airline holds is a tuning task.

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE TABLE {S}.gold_holding AS
WITH agg AS (
  SELECT
    icao, any_value(callsign) AS callsign, any_value(ac_type) AS ac_type,
    count(*) AS n_reports, min(snapshot_ts) AS first_ts, max(snapshot_ts) AS last_ts,
    round(avg(alt_ft))            AS mean_alt_ft,
    round(avg(dist_to_apt_nm), 1) AS mean_dist_nm,
    round(avg(lat), 4)            AS approx_lat,
    round(avg(lon), 4)            AS approx_lon,
    round((max(lat) - min(lat)) * 60, 1)                          AS ns_nm,
    round((max(lon) - min(lon)) * 60 * cos(radians(avg(lat))), 1) AS ew_nm,
    round(1 - sqrt(pow(avg(cos(radians(track_deg))), 2)
                 + pow(avg(sin(radians(track_deg))), 2)), 2)      AS heading_spread
  FROM {S}.silver_positions
  WHERE has_position AND NOT is_grounded AND track_deg IS NOT NULL
    AND alt_ft BETWEEN 2000 AND 20000
  GROUP BY icao
)
SELECT * FROM agg
WHERE n_reports >= 8 AND heading_spread >= 0.5
  AND ns_nm <= 15 AND ew_nm <= 15 AND mean_dist_nm <= 80
ORDER BY heading_spread DESC
""")
print(spark.table(f"{S}.gold_holding").count(), "rows")

# COMMAND ----------
# MAGIC %md
# MAGIC ## `gold_touchdowns` — detected landings
# MAGIC `touchdown_ts` = first on-ground report within 3 nm of the field, else the last airborne
# MAGIC short-final report. Two widths:
# MAGIC - **candidate window** (`sf`) — wide enough that a sparse-cadence track still lands at
# MAGIC   least one report inside it. At 180 s cadence and ~200 kt, consecutive reports are up to
# MAGIC   ~10 nm apart, so the original 4 nm / +1500 ft window missed most sparse-day arrivals
# MAGIC   entirely (~360/day observed vs ~1,070/day on the 60 s days — a detector problem, not a
# MAGIC   traffic difference).
# MAGIC - **acceptance gate** — looser than before for the same reason: requiring a report within
# MAGIC   3 nm / +500 ft (the original acceptance) is often simply never observed at 180 s cadence.
# MAGIC   `touchdown_confidence` records which gate a row actually met — `confirmed` (saw it that
# MAGIC   close) vs `inferred` (extrapolated from a report only as close as the wider acceptance
# MAGIC   allows). Use `confirmed` where label precision matters (e.g. tightening Model 1's
# MAGIC   labels further); `inferred` is fine for 15-minute demand bucketing.

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE TABLE {S}.gold_touchdowns AS
WITH sf AS (
  SELECT seg_id, icao, callsign, ac_type, apt_icao, snapshot_ts,
         dist_to_apt_nm, alt_ft, gs_kt, is_grounded, vrate_fpm
  FROM {S}.gold_tracks
  WHERE dist_to_apt_nm < 8 AND alt_ft <= {APT_ELEV_FT} + 4000 AND gs_kt BETWEEN 25 AND 250
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
""")
print(spark.table(f"{S}.gold_touchdowns").count(), "rows")

# COMMAND ----------
# MAGIC %md
# MAGIC ## `gold_arrival_tracks` — the Model 1 training set
# MAGIC Every airborne report of a segment that ends in a detected touchdown, labelled with
# MAGIC `minutes_to_touchdown`. One row = one (aircraft, time) training example. Segments have no
# MAGIC internal gap > 3 min (that splits `seg_id`), so the label is trustworthy. Reports 0.5–40 min
# MAGIC before touchdown are kept — beyond that the aircraft was usually not yet on approach.

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE TABLE {S}.gold_arrival_tracks AS
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
print(_at.count(), "training rows |",
      _at.select("seg_id").distinct().count(), "arrivals")

# COMMAND ----------
# MAGIC %md
# MAGIC ## `gold_demand_15m` — the Model 2 series
# MAGIC Touchdowns bucketed into 15-minute bins. A full 96-bin spine is generated for every date
# MAGIC that clears `min_touchdowns_for_active_day` arrivals, so zero-arrival bins are explicit
# MAGIC within a real day (the archive only has the 1st of each month, so the series is a set of
# MAGIC independent full days, not one continuous timeline).
# MAGIC
# MAGIC **Day-boundary filter.** A poll target of `HH:MM` on the 1st can land a snapshot whose
# MAGIC `now` is a few seconds into the prior day (e.g. `2025-08-31T23:59:59`), landing 1-2 stray
# MAGIC touchdowns on a date nobody actually collected. Left unfiltered, each such date got a full
# MAGIC spurious 96-bin spine. Real days have 200+ touchdowns; the threshold below is comfortably
# MAGIC between the two.

# COMMAND ----------
MIN_TOUCHDOWNS_FOR_ACTIVE_DAY = 20

spark.sql(f"""
CREATE OR REPLACE TABLE {S}.gold_demand_15m AS
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
print(_d.count(), "bins |", _d.selectExpr("sum(arrivals)").first()[0], "total arrivals |",
      _d.select("bin_date").distinct().count(), "days")

# COMMAND ----------
# MAGIC %md ## `gold_kpis` — dashboard headline numbers

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE TABLE {S}.gold_kpis AS
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
for t in ["gold_tracks", "gold_congestion", "gold_holding", "gold_touchdowns",
          "gold_arrival_tracks", "gold_demand_15m", "gold_kpis"]:
    print(f"{t:20} {spark.table(f'{S}.{t}').count():>8} rows")
display(spark.table(f"{S}.gold_kpis"))
