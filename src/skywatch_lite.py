# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch Lite
# MAGIC Historical airspace analytics on Databricks Free Edition.
# MAGIC
# MAGIC **Source:** ADS-B Exchange free sample archive (`samples.adsbexchange.com/readsb-hist`) — CC-BY-NC.
# MAGIC
# MAGIC **Pipeline:** land `.json.gz` snapshots -> Bronze -> Silver -> Gold -> AI/BI dashboard + Genie.
# MAGIC
# MAGIC Run cells top to bottom.

# COMMAND ----------
# MAGIC %md ## 1. Config  — edit `DATE_PATH` to a date that actually has data

# COMMAND ----------
CATALOG    = "skywatch"
SCHEMA     = "core"
VOLUME     = "landing"
START_HHMMSS = (12, 0, 0)   # time of day to start pulling snapshots from

# Parameters — supplied by the Asset Bundle job (base_parameters), with defaults for
# interactive runs. Verify DATE_PATH against https://samples.adsbexchange.com/readsb-hist/
try:
    dbutils.widgets.text("date_path", "2024/06/01")
    dbutils.widgets.text("n_files", "60")
    DATE_PATH = dbutils.widgets.get("date_path")
    N_FILES   = int(dbutils.widgets.get("n_files"))
except Exception:
    DATE_PATH, N_FILES = "2024/06/01", 60
print(f"DATE_PATH={DATE_PATH}  N_FILES={N_FILES}")

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA  IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME  IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"ready: {CATALOG}.{SCHEMA}, volume /Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/")

# COMMAND ----------
# MAGIC %md ## 2. Land raw snapshots into the Volume

# COMMAND ----------
import os, requests
from datetime import datetime, timedelta

base = f"https://samples.adsbexchange.com/readsb-hist/{DATE_PATH}/"
dest = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/"

t0 = datetime(2000, 1, 1, *START_HHMMSS)
candidates = [(t0 + timedelta(seconds=5 * i)).strftime("%H%M%SZ.json.gz") for i in range(4000)]

got, tried = 0, 0
for name in candidates:
    if got >= N_FILES:
        break
    # the archive serves these decompressed despite the .gz name -> save as plain .json
    local = os.path.join(dest, name.replace(".json.gz", ".json"))
    if os.path.exists(local):
        got += 1
        continue
    tried += 1
    try:
        r = requests.get(base + name, timeout=30)
    except Exception as e:
        print("err", name, e); continue
    if r.status_code == 200 and len(r.content) > 500:
        with open(local, "wb") as f:
            f.write(r.content)
        got += 1

print(f"{got} files in {dest}  (tried {tried})")
if got == 0:
    raise RuntimeError("No files downloaded. Check DATE_PATH against the directory listing, "
                       "or use the manual-upload fallback in the README.")
display(dbutils.fs.ls(dest))

# COMMAND ----------
# MAGIC %md ## 3. Bronze — read the gzipped JSON, explode the aircraft array

# COMMAND ----------
from pyspark.sql import functions as F

raw = spark.read.option("multiLine", True).json(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/*.json")
print("top-level columns:", raw.columns)

ac_col = "aircraft" if "aircraft" in raw.columns else "ac"
bronze = raw.select(F.col("now").alias("snapshot_epoch"), F.explode(ac_col).alias("ac"))
(bronze.write.mode("overwrite").option("overwriteSchema", "true")
       .saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_aircraft"))
print(spark.table(f"{CATALOG}.{SCHEMA}.bronze_aircraft").count(), "raw position rows")

# COMMAND ----------
# MAGIC %md ## 4. Silver — one clean, typed row per aircraft-position

# COMMAND ----------
b = spark.table(f"{CATALOG}.{SCHEMA}.bronze_aircraft")

silver = (b.select(
        F.to_timestamp("snapshot_epoch").alias("snapshot_ts"),
        F.col("ac.hex").alias("icao"),
        F.trim(F.col("ac.flight")).alias("callsign"),
        F.col("ac.r").alias("registration"),
        F.col("ac.t").alias("ac_type"),
        F.col("ac.lat").cast("double").alias("lat"),
        F.col("ac.lon").cast("double").alias("lon"),
        F.when(F.col("ac.alt_baro").cast("string") == "ground", F.lit(0))
         .otherwise(F.col("ac.alt_baro").cast("int")).alias("alt_ft"),
        F.col("ac.gs").cast("double").alias("gs_kt"),
        F.col("ac.track").cast("double").alias("track_deg"),
        F.col("ac.squawk").cast("string").alias("squawk"),
        F.coalesce(F.col("ac.emergency").cast("string"), F.lit("none")).alias("emergency"),
        F.col("ac.category").cast("string").alias("category"),
        F.col("ac.lat").isNotNull().alias("has_position"))
     .filter("icao IS NOT NULL"))
# keep position-less rows: an aircraft squawking an emergency but with no fix still matters.

(silver.write.mode("overwrite").option("overwriteSchema", "true")
       .saveAsTable(f"{CATALOG}.{SCHEMA}.silver_positions"))
print(spark.table(f"{CATALOG}.{SCHEMA}.silver_positions").count(), "position rows",
      "|", spark.table(f"{CATALOG}.{SCHEMA}.silver_positions").filter("has_position").count(), "with a fix")
display(spark.table(f"{CATALOG}.{SCHEMA}.silver_positions").limit(10))

# COMMAND ----------
# MAGIC %md ## 5. Gold — aggregate tables for the dashboard
# MAGIC (SQL below hardcodes `skywatch.core`; find/replace if you changed the names.)

# COMMAND ----------
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE skywatch.core.gold_kpis AS
# MAGIC SELECT
# MAGIC   count(DISTINCT CASE WHEN has_position THEN icao END)          AS aircraft_tracked,
# MAGIC   count(DISTINCT icao)                                          AS raw_contacts,
# MAGIC   count_if(has_position)                                        AS position_reports,
# MAGIC   count(DISTINCT left(callsign, 3))                             AS distinct_airlines,
# MAGIC   count(DISTINCT ac_type)                                       AS distinct_types,
# MAGIC   round(count_if(alt_ft > 0) / count_if(has_position) * 100, 1) AS pct_airborne,
# MAGIC   count(DISTINCT CASE WHEN emergency IN
# MAGIC        ('general','lifeguard','minfuel','nordo','unlawful','downed')
# MAGIC        THEN icao END)                                           AS emergency_aircraft,
# MAGIC   min(snapshot_ts)                                              AS window_start,
# MAGIC   max(snapshot_ts)                                              AS window_end
# MAGIC FROM skywatch.core.silver_positions;

# COMMAND ----------
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE skywatch.core.gold_h3_density AS
# MAGIC SELECT
# MAGIC   h3_longlatash3string(lon, lat, 3) AS h3_cell,
# MAGIC   count(*)             AS position_reports,
# MAGIC   count(DISTINCT icao) AS aircraft,
# MAGIC   round(avg(alt_ft))   AS avg_alt_ft
# MAGIC FROM skywatch.core.silver_positions
# MAGIC WHERE has_position AND lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180
# MAGIC GROUP BY 1;

# COMMAND ----------
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE skywatch.core.gold_special_squawks AS
# MAGIC SELECT
# MAGIC   icao, callsign, ac_type, registration, squawk, emergency,
# MAGIC   CASE
# MAGIC     WHEN squawk = '7500' THEN 'Unlawful interference / hijack (7500)'
# MAGIC     WHEN squawk = '7600' THEN 'Radio failure (7600)'
# MAGIC     WHEN squawk = '7700' THEN 'General emergency (7700)'
# MAGIC     WHEN emergency = 'lifeguard' THEN 'Lifeguard / medical flight'
# MAGIC     WHEN emergency = 'minfuel'   THEN 'Minimum fuel'
# MAGIC     WHEN emergency = 'nordo'     THEN 'No radio (NORDO)'
# MAGIC     WHEN emergency = 'unlawful'  THEN 'Unlawful interference'
# MAGIC     WHEN emergency = 'downed'    THEN 'Aircraft downed'
# MAGIC     WHEN emergency = 'general'   THEN 'General emergency (flag)'
# MAGIC     ELSE concat('Emergency: ', emergency) END       AS event_type,
# MAGIC   -- raw ADS-B is noisy: tier by how much corroborating data came with the emergency bit
# MAGIC   CASE
# MAGIC     WHEN bool_or(has_position) AND callsign IS NOT NULL AND icao NOT LIKE '~%' THEN 'high'
# MAGIC     WHEN bool_or(has_position) OR callsign IS NOT NULL                         THEN 'medium'
# MAGIC     ELSE 'low' END                                  AS confidence,
# MAGIC   bool_or(has_position)                             AS has_position,
# MAGIC   min(snapshot_ts)                                  AS first_seen,
# MAGIC   max(snapshot_ts)                                  AS last_seen,
# MAGIC   count(*)                                          AS pings,
# MAGIC   round(avg(lat), 3)                                AS approx_lat,
# MAGIC   round(avg(lon), 3)                                AS approx_lon
# MAGIC FROM skywatch.core.silver_positions
# MAGIC WHERE squawk IN ('7500', '7600', '7700')
# MAGIC    OR emergency IN ('general','lifeguard','minfuel','nordo','unlawful','downed')
# MAGIC GROUP BY icao, callsign, ac_type, registration, squawk, emergency
# MAGIC ORDER BY (confidence = 'high') DESC, (confidence = 'medium') DESC, first_seen;

# COMMAND ----------
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE skywatch.core.gold_airline_activity AS
# MAGIC SELECT
# MAGIC   left(callsign, 3)    AS airline_icao,
# MAGIC   count(DISTINCT icao) AS aircraft,
# MAGIC   count(*)             AS position_reports,
# MAGIC   round(avg(alt_ft))   AS avg_alt_ft,
# MAGIC   round(avg(gs_kt))    AS avg_ground_speed_kt
# MAGIC FROM skywatch.core.silver_positions
# MAGIC WHERE callsign RLIKE '^[A-Z]{3}[0-9]'
# MAGIC GROUP BY 1
# MAGIC ORDER BY aircraft DESC;

# COMMAND ----------
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE skywatch.core.gold_altitude_bands AS
# MAGIC SELECT
# MAGIC   CASE WHEN alt_ft <= 0 THEN 'On ground'
# MAGIC        ELSE concat(cast(floor(alt_ft / 5000) * 5 AS int), '-',
# MAGIC                    cast(floor(alt_ft / 5000) * 5 + 5 AS int), 'k ft') END AS altitude_band,
# MAGIC   floor(alt_ft / 5000) AS band_sort,
# MAGIC   count(*)             AS position_reports,
# MAGIC   count(DISTINCT icao) AS aircraft
# MAGIC FROM skywatch.core.silver_positions
# MAGIC WHERE alt_ft IS NOT NULL
# MAGIC GROUP BY 1, 2
# MAGIC ORDER BY band_sort;

# COMMAND ----------
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE skywatch.core.gold_type_mix AS
# MAGIC SELECT ac_type,
# MAGIC        count(DISTINCT icao) AS aircraft,
# MAGIC        count(*)             AS position_reports
# MAGIC FROM skywatch.core.silver_positions
# MAGIC WHERE ac_type IS NOT NULL AND ac_type <> ''
# MAGIC GROUP BY 1
# MAGIC ORDER BY aircraft DESC;

# COMMAND ----------
# MAGIC %md ## 6. Sanity check

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT * FROM skywatch.core.gold_kpis;

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT * FROM skywatch.core.gold_special_squawks;

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. STRETCH (only if time) — surveillance-orbit / holding detector
# MAGIC Circling = heading swept through many directions while the aircraft stayed inside a small box.
# MAGIC `heading_spread` is circular variance (0 = dead straight, 1 = headings all around the compass),
# MAGIC computed from sin/cos so it does not break at the 360->0 wrap.

# COMMAND ----------
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE skywatch.core.gold_orbits AS
# MAGIC WITH agg AS (
# MAGIC   SELECT icao,
# MAGIC     any_value(callsign) AS callsign,
# MAGIC     any_value(ac_type)  AS ac_type,
# MAGIC     count(*)            AS pings,
# MAGIC     round(avg(gs_kt))   AS avg_speed_kt,
# MAGIC     round(avg(alt_ft))  AS avg_alt_ft,
# MAGIC     round((max(lat) - min(lat)) * 111)                          AS ns_km,
# MAGIC     round((max(lon) - min(lon)) * 111 * cos(radians(avg(lat)))) AS ew_km,
# MAGIC     round(1 - sqrt(pow(avg(cos(radians(track_deg))), 2)
# MAGIC                  + pow(avg(sin(radians(track_deg))), 2)), 2)    AS heading_spread,
# MAGIC     round(avg(lat), 3) AS approx_lat,
# MAGIC     round(avg(lon), 3) AS approx_lon
# MAGIC   FROM skywatch.core.silver_positions
# MAGIC   WHERE has_position AND alt_ft BETWEEN 1000 AND 45000 AND track_deg IS NOT NULL
# MAGIC   GROUP BY icao
# MAGIC )
# MAGIC SELECT * FROM agg
# MAGIC WHERE pings >= 25            -- loitering for most of the window, not just a turn
# MAGIC   AND avg_speed_kt >= 120
# MAGIC   AND heading_spread >= 0.5
# MAGIC   AND ns_km <= 18 AND ew_km <= 18
# MAGIC ORDER BY heading_spread DESC;

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. STRETCH (only if time) — GenAI airspace briefing
# MAGIC If the model name errors, open **Serving** and copy an available Foundation Model endpoint name.

# COMMAND ----------
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE skywatch.core.gold_briefing AS
# MAGIC SELECT ai_query(
# MAGIC   'databricks-meta-llama-3-3-70b-instruct',
# MAGIC   concat(
# MAGIC     'You are an air-traffic analyst. In 3 punchy sentences, brief the global airspace picture. ',
# MAGIC     '"aircraft_tracked" is the confirmed count; "raw_contacts" also counts unverified targets. ',
# MAGIC     'KPIs: ', (SELECT to_json(struct(*)) FROM skywatch.core.gold_kpis),
# MAGIC     '  Notable circling / holding aircraft: ',
# MAGIC     (SELECT to_json(collect_list(struct(callsign, ac_type, avg_alt_ft, approx_lat, approx_lon)))
# MAGIC      FROM (SELECT * FROM skywatch.core.gold_orbits ORDER BY heading_spread DESC LIMIT 8)))
# MAGIC ) AS briefing;

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT briefing FROM skywatch.core.gold_briefing;
