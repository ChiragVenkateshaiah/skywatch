# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — medallion pipeline (thin slice)
# MAGIC
# MAGIC Lakeflow Declarative Pipeline. Auto Loader ingests the raw poller output from the landing
# MAGIC Volume and refines it Bronze -> Silver.
# MAGIC
# MAGIC **This is the thin slice** — Bronze + a deliberately minimal Silver (typed columns + the
# MAGIC two geometry features that are safe to compute without seeing the data: distance and
# MAGIC bearing to the airport). Phase-of-flight, vertical-rate reconciliation, and every Gold
# MAGIC table are added *after* the Genie Code EDA pass on real airplanes.live data
# MAGIC (see `docs/ML_ROADMAP.md` Phase 1).
# MAGIC
# MAGIC Pipeline configuration (set in `resources/skywatch.pipeline.yml`):
# MAGIC - `skywatch.landing_path`   — Volume dir the poller writes to
# MAGIC - `skywatch.schema_location` — Auto Loader schema tracking dir (separate prefix)

# COMMAND ----------
import dlt
from pyspark.sql import functions as F

LANDING_PATH = spark.conf.get(
    "skywatch.landing_path", "/Volumes/skywatch/core/landing/airplaneslive"
)
SCHEMA_LOCATION = spark.conf.get(
    "skywatch.schema_location", "/Volumes/skywatch/core/landing/_autoloader/medallion"
)

EARTH_RADIUS_NM = 3440.065


def _great_circle_nm(lat1, lon1, lat2, lon2):
    """Haversine distance in nautical miles, as a Column expression."""
    p1, p2 = F.radians(lat1), F.radians(lat2)
    dphi = F.radians(lat2 - lat1)
    dlam = F.radians(lon2 - lon1)
    a = F.sin(dphi / 2) ** 2 + F.cos(p1) * F.cos(p2) * F.sin(dlam / 2) ** 2
    return F.lit(EARTH_RADIUS_NM) * 2 * F.asin(F.sqrt(a))


def _initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from point 1 to point 2, degrees 0..360."""
    p1, p2 = F.radians(lat1), F.radians(lat2)
    dlam = F.radians(lon2 - lon1)
    y = F.sin(dlam) * F.cos(p2)
    x = F.cos(p1) * F.sin(p2) - F.sin(p1) * F.cos(p2) * F.cos(dlam)
    return (F.degrees(F.atan2(y, x)) + 360) % 360


# COMMAND ----------
# MAGIC %md ## Bronze — one row per aircraft report, raw `report` struct kept intact

# COMMAND ----------
@dlt.table(
    name="bronze_aircraft",
    comment="Raw airplanes.live /v2/point responses, aircraft array exploded. "
            "One row per aircraft per poll; full per-aircraft object kept as `report`.",
    table_properties={"quality": "bronze"},
)
def bronze_aircraft():
    raw = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaLocation", SCHEMA_LOCATION)
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .option("multiLine", "true")
        .load(LANDING_PATH)
    )
    return (
        raw.select(
            (F.col("now") / 1000).cast("timestamp").alias("snapshot_ts"),
            F.col("capture.apt_icao").alias("apt_icao"),
            F.col("capture.apt_lat").cast("double").alias("apt_lat"),
            F.col("capture.apt_lon").cast("double").alias("apt_lon"),
            F.col("capture.radius_nm").cast("int").alias("radius_nm"),
            F.col("capture.source").alias("source"),
            F.col("_metadata.file_path").alias("_ingest_file"),
            F.current_timestamp().alias("_ingest_ts"),
            F.explode("ac").alias("report"),
        )
    )


# COMMAND ----------
# MAGIC %md ## Silver — typed, deduped, + distance/bearing to the airport

# COMMAND ----------
@dlt.table(
    name="silver_positions",
    comment="One typed, deduplicated row per aircraft report. Thin slice: geometry features "
            "only (dist_to_apt_nm, bearing_to_apt). Position-less rows are kept.",
    table_properties={"quality": "silver"},
)
@dlt.expect_or_drop("has_icao", "icao IS NOT NULL")
@dlt.expect("plausible_altitude", "alt_ft IS NULL OR alt_ft BETWEEN -1500 AND 60000")
def silver_positions():
    b = spark.readStream.table("bronze_aircraft")

    projected = b.select(
        F.col("snapshot_ts"),
        F.col("apt_icao"),
        F.col("report.hex").alias("icao"),
        F.trim(F.col("report.flight")).alias("callsign"),
        F.col("report.r").alias("registration"),
        F.col("report.t").alias("ac_type"),
        F.col("report.category").cast("string").alias("category"),
        F.col("report.lat").cast("double").alias("lat"),
        F.col("report.lon").cast("double").alias("lon"),
        F.when(F.col("report.alt_baro").cast("string") == "ground", F.lit(0))
         .otherwise(F.col("report.alt_baro").cast("int")).alias("alt_ft"),
        F.col("report.gs").cast("double").alias("gs_kt"),
        F.col("report.track").cast("double").alias("track_deg"),
        F.col("report.baro_rate").cast("double").alias("baro_rate_fpm"),
        F.col("report.geom_rate").cast("double").alias("geom_rate_fpm"),
        F.col("report.squawk").cast("string").alias("squawk"),
        F.coalesce(F.col("report.emergency").cast("string"), F.lit("none")).alias("emergency"),
        F.col("report.seen_pos").cast("double").alias("seen_pos_s"),
        F.col("report.lat").isNotNull().alias("has_position"),
        F.col("apt_lat"),
        F.col("apt_lon"),
    )

    deduped = (
        projected
        .withWatermark("snapshot_ts", "15 minutes")
        .dropDuplicates(["icao", "snapshot_ts"])
    )

    return (
        deduped
        .withColumn(
            "dist_to_apt_nm",
            F.when(
                F.col("has_position"),
                _great_circle_nm("lat", "lon", "apt_lat", "apt_lon"),
            ),
        )
        .withColumn(
            "bearing_to_apt",
            F.when(
                F.col("has_position"),
                _initial_bearing_deg("lat", "lon", "apt_lat", "apt_lon"),
            ),
        )
        .drop("apt_lat", "apt_lon")
    )
