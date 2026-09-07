# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — Model 3 live irregularity flags (rule-based)
# MAGIC
# MAGIC The early-warning serving path. No model load — Model 3 is **rules**, not a learned
# MAGIC classifier (the archive has ~4 emergency aircraft and ~25 real holds across 9 days, and
# MAGIC no cleanly-labelled go-arounds at 180 s cadence — roadmap §7). Runs a few minutes behind
# MAGIC the poller / gold-serving job and appends one row per (currently-airborne aircraft, kind)
# MAGIC that trips a rule right now, to `skywatch.stream.irregularity_flags`.
# MAGIC
# MAGIC | kind | live rule | severity |
# MAGIC |---|---|---|
# MAGIC | `emergency` | current `squawk ∈ {7500,7600,7700}` or `emergency` field set | 3 |
# MAGIC | `go_around` | had a low descending on-approach report in the last `lookback_min`, now above field+2000 ft and climbing > 400 fpm | 3 |
# MAGIC | `holding` | last `lookback_min` of track = racetrack (heading spread ≥ 0.5 in a ≤ 15 nm box), 2000–20000 ft, 5–60 nm out | 2 |
# MAGIC
# MAGIC The historical counterpart is `gold_irregularities` (per completed segment) in `build_gold.py`.

# COMMAND ----------
try:
    dbutils.widgets.text("stream_schema", "skywatch.stream")
    dbutils.widgets.text("apt_icao", "KATL")
    dbutils.widgets.text("apt_elev_ft", "1026")
    dbutils.widgets.text("freshness_min", "20")     # an aircraft counts as "current" if seen this recently
    dbutils.widgets.text("lookback_min", "30")      # how much recent track to judge geometry on
    STREAM = dbutils.widgets.get("stream_schema")
    APT_ICAO = dbutils.widgets.get("apt_icao")
    APT_ELEV_FT = int(dbutils.widgets.get("apt_elev_ft"))
    FRESH = int(dbutils.widgets.get("freshness_min"))
    LOOKBACK = int(dbutils.widgets.get("lookback_min"))
except Exception:
    STREAM, APT_ICAO, APT_ELEV_FT, FRESH, LOOKBACK = "skywatch.stream", "KATL", 1026, 20, 30

print(f"irregularity flags for {APT_ICAO}  (fresh {FRESH} min, lookback {LOOKBACK} min)")

# COMMAND ----------
flags = spark.sql(f"""
WITH latest AS (SELECT max(snapshot_ts) AS mx FROM {STREAM}.gold_tracks WHERE apt_icao = '{APT_ICAO}'),
win AS (
  SELECT g.* FROM {STREAM}.gold_tracks g, latest
  WHERE g.apt_icao = '{APT_ICAO}' AND g.snapshot_ts >= latest.mx - INTERVAL {LOOKBACK} MINUTES
),
cur AS (
  SELECT icao, max(snapshot_ts) AS last_ts
  FROM win
  GROUP BY icao
  HAVING max(snapshot_ts) >= (SELECT mx FROM latest) - INTERVAL {FRESH} MINUTES
),
cur_row AS (
  SELECT w.* FROM win w JOIN cur c ON c.icao = w.icao AND c.last_ts = w.snapshot_ts
),
geom AS (
  SELECT icao,
    count(*) AS n,
    round(1 - sqrt(pow(avg(cos(radians(track_deg))), 2)
                 + pow(avg(sin(radians(track_deg))), 2)), 2)          AS heading_spread,
    round((max(lat) - min(lat)) * 60, 1)                              AS ns_nm,
    round((max(lon) - min(lon)) * 60 * cos(radians(avg(lat))), 1)     AS ew_nm,
    avg(alt_ft)          AS mean_alt,
    avg(dist_to_apt_nm)  AS mean_dist,
    min(CASE WHEN dist_to_apt_nm < 5 AND alt_ft < {APT_ELEV_FT} + 2500 AND vrate_fpm < -200
             THEN snapshot_ts END)                                    AS appr_ts
  FROM win
  WHERE NOT is_grounded AND track_deg IS NOT NULL
  GROUP BY icao
)
SELECT current_timestamp() AS scored_at, r.apt_icao, r.icao, r.callsign, r.ac_type,
       r.snapshot_ts, round(r.dist_to_apt_nm, 1) AS dist_to_apt_nm, r.alt_ft, r.gs_kt,
       'emergency' AS kind, 3 AS severity,
       concat_ws(' ',
         nullif(concat('squawk=', r.squawk), 'squawk='),
         nullif(concat('emergency=', r.emergency), 'emergency=none')) AS detail
FROM cur_row r
WHERE r.squawk IN ('7500','7600','7700')
   OR (r.emergency IS NOT NULL AND lower(r.emergency) NOT IN ('none',''))

UNION ALL
SELECT current_timestamp(), r.apt_icao, r.icao, r.callsign, r.ac_type,
       r.snapshot_ts, round(r.dist_to_apt_nm, 1), r.alt_ft, r.gs_kt,
       'go_around', 3,
       'was low on final in the last window, now climbing away'
FROM cur_row r JOIN geom g ON g.icao = r.icao
WHERE g.appr_ts IS NOT NULL
  AND r.snapshot_ts > g.appr_ts
  AND r.alt_ft > {APT_ELEV_FT} + 2000
  AND r.vrate_fpm > 400

UNION ALL
SELECT current_timestamp(), r.apt_icao, r.icao, r.callsign, r.ac_type,
       r.snapshot_ts, round(r.dist_to_apt_nm, 1), r.alt_ft, r.gs_kt,
       'holding', 2,
       concat('racetrack geometry, heading spread ', cast(g.heading_spread AS string))
FROM cur_row r JOIN geom g ON g.icao = r.icao
WHERE g.n >= 6 AND g.heading_spread >= 0.5
  AND g.ns_nm <= 15 AND g.ew_nm <= 15
  AND g.mean_alt BETWEEN 2000 AND 20000
  AND g.mean_dist BETWEEN 5 AND 60
""")

n = flags.count()
print(f"{n} live flag(s)")
if n == 0:
    dbutils.notebook.exit("empty")

(flags.write.mode("append").option("mergeSchema", "true")
      .saveAsTable(f"{STREAM}.irregularity_flags"))
display(flags.orderBy(flags["severity"].desc(), "dist_to_apt_nm"))
