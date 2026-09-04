# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — Model 1 shared feature engineering
# MAGIC
# MAGIC `%run` this from `train_eta.py` and `score_eta.py` so training and scoring can never
# MAGIC drift. One definition of the feature set, one transform.
# MAGIC
# MAGIC Input DataFrame must carry: `ac_type`, `snapshot_ts`, `bearing_to_apt`, and the raw
# MAGIC kinematics (`dist_to_apt_nm`, `heading_err_deg`, `alt_ft`, `gs_kt`, `vrate_fpm`,
# MAGIC `closure_nm`, `turn_rate_dps`, `sel_altitude_ft`, `airport_inbound_count`, `phase`).

# COMMAND ----------
import math

import pandas as pd
from pyspark.sql import functions as F

# The complete set of `phase` values produced by build_gold.py, in a fixed order — training
# and scoring MUST use the same list or LightGBM rejects the predict frame (a scoring batch
# rarely contains every phase).
PHASE_CATEGORIES = ["ground", "climb", "descent", "cruise", "level", "unknown"]

# Wide-body / heavy ICAO type codes seen at KATL (+ common freighters).
_HEAVY = {
    "B762", "B763", "B764", "B772", "B773", "B77L", "B77W", "B778", "B779",
    "B788", "B789", "B78X", "B742", "B744", "B748", "A332", "A333", "A338",
    "A339", "A342", "A343", "A345", "A346", "A359", "A35K", "A388", "MD11",
    "IL76", "A124", "C17", "C5M",
}

ETA_CAT = ["phase"]
ETA_FEATURES = [
    "dist_to_apt_nm", "heading_err_deg", "bearing_sin", "bearing_cos", "alt_ft", "gs_kt",
    "vrate_fpm", "closure_nm", "turn_rate_dps", "sel_altitude_ft", "airport_inbound_count",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_heavy", "phase",
]
ETA_TARGET = "minutes_to_touchdown"


def add_eta_features(df):
    """Add the ETA_FEATURES columns to a Spark DataFrame. Idempotent on `hour_utc`/`dow`."""
    two_pi = 2 * math.pi
    return (
        df
        .withColumn("is_heavy", F.col("ac_type").isin(list(_HEAVY)).cast("int"))
        .withColumn("hour_utc", F.hour("snapshot_ts"))
        .withColumn("dow", F.dayofweek("snapshot_ts"))
        .withColumn("hour_sin", F.sin(F.col("hour_utc") / 24 * two_pi))
        .withColumn("hour_cos", F.cos(F.col("hour_utc") / 24 * two_pi))
        .withColumn("dow_sin", F.sin((F.col("dow") - 1) / 7 * two_pi))
        .withColumn("dow_cos", F.cos((F.col("dow") - 1) / 7 * two_pi))
        .withColumn("bearing_sin", F.sin(F.radians("bearing_to_apt")))
        .withColumn("bearing_cos", F.cos(F.radians("bearing_to_apt")))
    )


def eta_pandas(pdf):
    """Post-`toPandas()` cleanup: Spark round() yields decimal.Decimal — force every numeric
    column to float; make `phase` a category so LightGBM treats it natively."""
    num = [c for c in ETA_FEATURES if c not in ETA_CAT] + [ETA_TARGET, "baseline_pred"]
    for c in num:
        if c in pdf.columns:
            pdf[c] = pdf[c].astype("float64")
    pdf["phase"] = pd.Categorical(
        pdf["phase"].fillna("unknown"), categories=PHASE_CATEGORIES
    )
    return pdf
