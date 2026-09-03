# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — medallion pipeline (Bronze + Silver)
# MAGIC
# MAGIC Lakeflow Declarative Pipeline. Auto Loader ingests the raw poller output from the landing
# MAGIC Volume and refines it Bronze -> Silver. Both flows are streaming; the pipeline runs
# MAGIC triggered (not continuous).
# MAGIC
# MAGIC Silver's column set is finalised from the Phase 1 EDA. Everything history-dependent —
# MAGIC trajectory assembly, delta-altitude vertical rate, touchdown detection, congestion and
# MAGIC holding, the demand series — lives in the batch **Gold** job (`src/build_gold.py`,
# MAGIC added once we have a real arrival wave to validate against). See `docs/ML_ROADMAP.md` §5.
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


def _cols(*names):
    return tuple(F.col(n) if isinstance(n, str) else n for n in names)


def _great_circle_nm(lat1, lon1, lat2, lon2):
    """Haversine distance in nautical miles, as a Column expression."""
    lat1, lon1, lat2, lon2 = _cols(lat1, lon1, lat2, lon2)
    p1, p2 = F.radians(lat1), F.radians(lat2)
    dphi = F.radians(lat2 - lat1)
    dlam = F.radians(lon2 - lon1)
    a = F.sin(dphi / 2) ** 2 + F.cos(p1) * F.cos(p2) * F.sin(dlam / 2) ** 2
    return F.lit(EARTH_RADIUS_NM) * 2 * F.asin(F.sqrt(a))


def _initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from point 1 to point 2, degrees 0..360."""
    lat1, lon1, lat2, lon2 = _cols(lat1, lon1, lat2, lon2)
    p1, p2 = F.radians(lat1), F.radians(lat2)
    dlam = F.radians(lon2 - lon1)
    y = F.sin(dlam) * F.cos(p2)
    x = F.cos(p1) * F.sin(p2) - F.sin(p1) * F.cos(p2) * F.cos(dlam)
    return (F.degrees(F.atan2(y, x)) + 360) % 360


def _angular_diff_deg(a, b):
    """Smallest absolute difference between two bearings, degrees 0..180."""
    a, b = _cols(a, b)
    d = F.abs((a - b) % 360)
    return F.least(d, 360 - d)


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
# MAGIC %md
# MAGIC ## Silver — one typed, deduped row per report
# MAGIC
# MAGIC Point-wise features only — nothing that needs the previous report. History-dependent
# MAGIC work (delta-altitude vertical rate for the ~5% with no `baro_rate`/`geom_rate`, trajectory
# MAGIC phase, "inbound over N snapshots", touchdown detection) is the batch Gold layer's job.
# MAGIC
# MAGIC Column set finalised from the Phase 1 EDA (8 552 rows / 523 aircraft):
# MAGIC promoted `alt_geom` (94.9%), `nav_altitude_mcp` → `sel_altitude_ft` (76.3%, descent-intent
# MAGIC signal), `nav_heading` → `sel_heading_deg` (54.9%), `nic`/`rc`/`seen` (100%);
# MAGIC skipped `mag_heading` (0.06%), `true_heading` (4.7%), `roll` (absent).

# COMMAND ----------
@dlt.table(
    name="silver_positions",
    comment="One typed, deduplicated row per aircraft report for the target airport. Point-wise "
            "features only. Position-less rows are kept — an emergency squawk with no fix matters.",
    table_properties={"quality": "silver"},
)
@dlt.expect_or_drop("has_icao", "icao IS NOT NULL")
@dlt.expect("plausible_altitude", "alt_ft IS NULL OR alt_ft BETWEEN -1500 AND 60000")
@dlt.expect("plausible_groundspeed", "gs_kt IS NULL OR gs_kt BETWEEN 0 AND 700")
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
        # alt_baro arrives as a string: numeric feet, or the literal "ground"
        F.when(F.lower(F.col("report.alt_baro").cast("string")) == "ground", F.lit(0))
         .otherwise(F.col("report.alt_baro").cast("int")).alias("alt_ft"),
        F.col("report.alt_geom").cast("int").alias("alt_geom_ft"),
        F.col("report.gs").cast("double").alias("gs_kt"),
        F.col("report.track").cast("double").alias("track_deg"),
        F.col("report.baro_rate").cast("double").alias("baro_rate_fpm"),
        F.col("report.geom_rate").cast("double").alias("geom_rate_fpm"),
        F.col("report.nav_altitude_mcp").cast("int").alias("sel_altitude_ft"),
        F.col("report.nav_heading").cast("double").alias("sel_heading_deg"),
        F.col("report.squawk").cast("string").alias("squawk"),
        F.coalesce(F.col("report.emergency").cast("string"), F.lit("none")).alias("emergency"),
        F.col("report.nic").cast("int").alias("nic"),
        F.col("report.rc").cast("int").alias("rc"),
        F.col("report.seen").cast("double").alias("seen_s"),
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

    vrate = F.coalesce(F.col("baro_rate_fpm"), F.col("geom_rate_fpm"))
    grounded = (F.col("alt_ft") <= 0) & (F.coalesce(F.col("gs_kt"), F.lit(0.0)) < 50)

    return (
        deduped
        .withColumn("dist_to_apt_nm",
                    F.when(F.col("has_position"),
                           _great_circle_nm("lat", "lon", "apt_lat", "apt_lon")))
        .withColumn("bearing_to_apt",
                    F.when(F.col("has_position"),
                           _initial_bearing_deg("lat", "lon", "apt_lat", "apt_lon")))
        # angle between heading and the direction to the field: ~0 = pointed at KATL
        .withColumn("heading_err_deg",
                    F.when(F.col("has_position") & F.col("track_deg").isNotNull(),
                           _angular_diff_deg("track_deg", "bearing_to_apt")))
        .withColumn("vertical_rate_fpm", vrate)
        .withColumn("vertical_rate_src",
                    F.when(F.col("baro_rate_fpm").isNotNull(), F.lit("baro"))
                     .when(F.col("geom_rate_fpm").isNotNull(), F.lit("geom")))
        .withColumn("is_grounded", grounded)
        # point-wise phase only; trajectory phases (approach, go-around) are a Gold job
        .withColumn(
            "phase",
            F.when(grounded, "ground")
             .when(vrate > 400, "climb")
             .when(vrate < -400, "descent")
             .when(F.col("alt_ft") >= 18000, "cruise")
             .when(vrate.isNotNull(), "level")
             .otherwise("unknown"),
        )
        .drop("apt_lat", "apt_lon")
    )
