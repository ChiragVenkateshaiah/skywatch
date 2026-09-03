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
# MAGIC | `gold_tracks` | one row per report | + Δt, Δalt-derived vertical rate, along-track closure, turn rate, `seg_id` (gap > 3 min splits), `inbound_flag` |
# MAGIC | `gold_congestion` | minute × ring | inbound / total counts, mean alt / gs / ETA per ring |
# MAGIC | `gold_holding` | one row per circling aircraft | circular-variance heading spread in a small box — currently flags *any* circling (light a/c, military); airline-hold tuning is a TODO |
# MAGIC | `gold_touchdowns` | one row per landing | **first-cut thresholds** — tune against a full arrival wave |
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
derived AS (
  SELECT *,
    CASE WHEN dt_s BETWEEN 1 AND 600 THEN (alt_ft - prev_alt_ft) / dt_s * 60.0 END AS derived_vrate_fpm,
    (prev_dist_nm - dist_to_apt_nm) AS closure_nm,
    CASE WHEN prev_track_deg IS NOT NULL
         THEN least(abs(track_deg - prev_track_deg), 360 - abs(track_deg - prev_track_deg)) END AS turn_deg,
    CASE WHEN dt_s IS NULL OR dt_s > 180 THEN 1 ELSE 0 END AS seg_break
  FROM stepped
),
segmented AS (
  SELECT *,
    concat(icao, '-', cast(sum(seg_break) OVER (PARTITION BY icao ORDER BY snapshot_ts) AS string)) AS seg_id
  FROM derived
)
SELECT
  seg_id, icao, callsign, ac_type, apt_icao, snapshot_ts,
  lat, lon, alt_ft, alt_geom_ft, gs_kt, track_deg, sel_altitude_ft,
  dist_to_apt_nm, bearing_to_apt, heading_err_deg, is_grounded, phase,
  dt_s, closure_nm, turn_deg,
  coalesce(vertical_rate_fpm, derived_vrate_fpm) AS vrate_fpm,
  coalesce(vertical_rate_src, CASE WHEN derived_vrate_fpm IS NOT NULL THEN 'delta' END) AS vrate_src,
  CASE WHEN turn_deg IS NOT NULL AND dt_s BETWEEN 1 AND 600 THEN turn_deg / dt_s END AS turn_rate_dps,
  count(*)         OVER (PARTITION BY seg_id) AS seg_n_reports,
  min(snapshot_ts) OVER (PARTITION BY seg_id) AS seg_start_ts,
  max(snapshot_ts) OVER (PARTITION BY seg_id) AS seg_end_ts,
  (heading_err_deg < 70
   AND alt_ft BETWEEN 500 AND 40000
   AND NOT is_grounded
   AND sum(CASE WHEN (prev_dist_nm - dist_to_apt_nm) > 0 THEN 1 ELSE 0 END)
         OVER (PARTITION BY seg_id ORDER BY snapshot_ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) >= 2
  ) AS inbound_flag
FROM segmented
""")
print(spark.table(f"{S}.gold_tracks").count(), "rows")

# COMMAND ----------
# MAGIC %md ## `gold_congestion` — minute × distance ring

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE TABLE {S}.gold_congestion AS
SELECT
  date_trunc('MINUTE', snapshot_ts) AS minute_ts,
  apt_icao,
  CASE WHEN dist_to_apt_nm < 40  THEN '00-40'
       WHEN dist_to_apt_nm < 100 THEN '40-100'
       WHEN dist_to_apt_nm < 200 THEN '100-200'
       ELSE '200-250' END AS ring,
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
# MAGIC %md ## `gold_touchdowns` — detected landings
# MAGIC **First-cut thresholds.** `touchdown_ts` = first on-ground report within 3 nm of the field,
# MAGIC else the last airborne short-final report. Validate + tune against a full arrival wave.

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE TABLE {S}.gold_touchdowns AS
WITH sf AS (
  SELECT seg_id, icao, callsign, ac_type, apt_icao, snapshot_ts,
         dist_to_apt_nm, alt_ft, gs_kt, is_grounded, vrate_fpm
  FROM {S}.gold_tracks
  WHERE dist_to_apt_nm < 4 AND alt_ft <= {APT_ELEV_FT} + 1500 AND gs_kt BETWEEN 25 AND 190
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
SELECT * FROM seg
WHERE min_dist_nm < 3 AND min_alt_ft <= {APT_ELEV_FT} + 500 AND descent_reports >= 1
""")
print(spark.table(f"{S}.gold_touchdowns").count(), "rows")

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
for t in ["gold_tracks", "gold_congestion", "gold_holding", "gold_touchdowns", "gold_kpis"]:
    print(f"{t:18} {spark.table(f'{S}.{t}').count():>8} rows")
display(spark.table(f"{S}.gold_kpis"))
