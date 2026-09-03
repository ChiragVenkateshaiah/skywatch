# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # SkyWatch EDA: Bronze & Silver Profiling (KATL)
# MAGIC
# MAGIC Read-only exploratory data analysis of SkyWatch ADS-B data for the KATL Arrival Manager. Profiles field coverage in `bronze_aircraft`, distributions in `silver_positions`, traffic classification, callsign patterns, data quality, and arrival trajectories. All queries are read-only — no tables are created or modified.

# COMMAND ----------

# DBTITLE 1,Setup: imports, table loads, basic counts
import builtins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pandas as pd

# Restore Python builtins shadowed by pyspark.sql.functions wildcard import
round = builtins.round

bronze = spark.table("skywatch.stream.bronze_aircraft")
silver = spark.table("skywatch.stream.silver_positions")

total_bronze = bronze.count()
total_silver = silver.count()
n_snapshots = silver.select("snapshot_ts").distinct().count()

print(f"Bronze rows:        {total_bronze}")
print(f"Silver rows:        {total_silver}")
print(f"Distinct snapshots: {n_snapshots}")

# COMMAND ----------

# DBTITLE 1,Section 1 — Field Coverage
# MAGIC %md
# MAGIC ## 1 — Field Coverage of `bronze_aircraft.report`
# MAGIC
# MAGIC For every field in the `report` struct: non-null count and percentage, plus min / median / max for numeric fields.

# COMMAND ----------

# DBTITLE 1,Field coverage table
# Build field-coverage table for every field in report struct
report_fields = bronze.schema["report"].dataType.fields
total = total_bronze

rows = []
for f in report_fields:
    fname = f.name
    dtype = f.dataType.simpleString()
    is_numeric = dtype in ("bigint", "long", "int", "double", "float")

    nn = bronze.agg(count(col(f"report.`{fname}`")).alias("nn")).collect()[0]["nn"]
    pct = round(nn / total * 100, 2) if total > 0 else 0.0

    min_v = med_v = max_v = None
    if is_numeric and nn > 0:
        stats = bronze.agg(
            min(col(f"report.`{fname}`")).alias("mn"),
            expr(f"percentile_approx(report.`{fname}`, 0.5)").alias("md"),
            max(col(f"report.`{fname}`")).alias("mx"),
        ).collect()[0]
        min_v = stats["mn"]
        med_v = stats["md"]
        max_v = stats["mx"]

    rows.append({
        "field_name": fname,
        "data_type": dtype,
        "non_null_count": int(nn),
        "non_null_pct": pct,
        "min_val": min_v,
        "median_val": med_v,
        "max_val": max_v,
    })

pdf_cov = pd.DataFrame(rows).sort_values("non_null_pct", ascending=False).reset_index(drop=True)
display(pdf_cov)

# COMMAND ----------

# DBTITLE 1,Section 1 recommendations
# MAGIC %md
# MAGIC ### Section 1 Recommendations — Fields to Promote into Silver
# MAGIC
# MAGIC Based on the coverage table above, here is the assessment for each candidate field:
# MAGIC
# MAGIC | Field | Type | Promote? | Rationale |
# MAGIC | --- | --- | --- | --- |
# MAGIC | `alt_geom` | bigint | 94.9 % | **Yes** | Geometric altitude; fallback when `alt_baro` (barometric) is missing or "ground". Essential for continuous altitude tracking during approach. |
# MAGIC | `nav_altitude_mcp` | bigint | 76.3 % | **Yes** | Selected altitude (Mode Control Panel). Directly indicates pilot descent intent — a strong signal for arrival detection. |
# MAGIC | `nav_heading` | double | 54.9 % | **Yes** | Selected heading; reveals intended flight direction, useful for approach pattern detection. |
# MAGIC | `mag_heading` | double | 0.06 % (5 rows) | **Low** | Nearly absent from this data source. Not worth promoting unless upstream coverage improves. |
# MAGIC | `true_heading` | double | 4.7 % (402 rows) | **Low** | Sparse; complements track where available but too few rows to rely on for Gold. |
# MAGIC | `nic` | bigint | 100 % | **Yes** | Navigation Integrity Category; quality indicator for position accuracy. Helps filter unreliable positions. |
# MAGIC | `rc` | bigint | 100 % | **Yes** | Radius of containment; position uncertainty radius. Quality flag for Gold-layer filtering. |
# MAGIC | `seen` | double | 100 % | **Yes** | Seconds since last message; staleness indicator — reports with high `seen` may be stale positions. |
# MAGIC | `seen_pos` | double | 100 % | **Yes** | Seconds since last position fix; already partially in Silver as `seen_pos_s`. Confirm it maps correctly. |
# MAGIC | `dst` | double | 100 % | **Maybe** | Distance from receiver (not airport). Less useful than computed `dist_to_apt_nm` but can flag receiver-side issues. |
# MAGIC | `dir` | double | 100 % | **Maybe** | Direction from receiver. Same caveat as `dst`. |
# MAGIC | `roll` | — | **N/A** | `roll` does **not exist** in the `report` struct from this data source. Bank-angle would need to be derived from heading changes over time. |
# MAGIC
# MAGIC **Already promoted to Silver:** `lat`, `lon`, `alt_baro`→`alt_ft`, `gs`→`gs_kt`, `track`→`track_deg`, `baro_rate`→`baro_rate_fpm`, `geom_rate`→`geom_rate_fpm`, `squawk`, `emergency`, `category`, `flight`→`callsign`, `hex`→`icao`.
# MAGIC
# MAGIC **Not recommended for Silver:** `rssi` (receiver signal strength, not arrival-relevant), `messages` (cumulative counter), `mlat`/`tisb`/`nav_modes` (array types, low analytical value for arrivals), `dbFlags`, `gva`, `sda`, `sil`, `sil_type`, `spi`, `alert`, `version`, `nac_p`, `nac_v`, `nic_baro`, `nav_altitude_fms`, `nav_qnh`, `calc_track`, `r`, `t`, `type`.

# COMMAND ----------

# DBTITLE 1,Section 2 — Distributions
# MAGIC %md
# MAGIC ## 2 — Distributions in `silver_positions`
# MAGIC
# MAGIC ### 2a — Banded distributions of alt_ft, gs_kt, baro_rate_fpm, dist_to_apt_nm, bearing_to_apt

# COMMAND ----------

# DBTITLE 1,2a banded distributions
# --- 2a: Banded distributions ---

# alt_ft bands
alt_bands = silver.select(
    when(col("alt_ft").isNull(), "null")
    .when(col("alt_ft") == 0, "0 (ground)")
    .when(col("alt_ft") < 5000, "0–5k")
    .when(col("alt_ft") < 10000, "5k–10k")
    .when(col("alt_ft") < 20000, "10k–20k")
    .when(col("alt_ft") < 30000, "20k–30k")
    .when(col("alt_ft") < 40000, "30k–40k")
    .otherwise("40k+").alias("alt_band")
).groupBy("alt_band").count().orderBy("alt_band")
print("=== alt_ft bands ===")
display(alt_bands)

# gs_kt bands
gs_bands = silver.select(
    when(col("gs_kt").isNull(), "null")
    .when(col("gs_kt") < 50, "0–50")
    .when(col("gs_kt") < 100, "50–100")
    .when(col("gs_kt") < 200, "100–200")
    .when(col("gs_kt") < 300, "200–300")
    .when(col("gs_kt") < 400, "300–400")
    .otherwise("400+").alias("gs_band")
).groupBy("gs_band").count().orderBy("gs_band")
print("=== gs_kt bands ===")
display(gs_bands)

# baro_rate_fpm bands
baro_bands = silver.select(
    when(col("baro_rate_fpm").isNull(), "null")
    .when(col("baro_rate_fpm") < -2000, "< -2000")
    .when(col("baro_rate_fpm") < -500, "-2000 to -500")
    .when(col("baro_rate_fpm") <= 500, "-500 to 500")
    .when(col("baro_rate_fpm") <= 2000, "500 to 2000")
    .otherwise("> 2000").alias("baro_band")
).groupBy("baro_band").count().orderBy("baro_band")
print("=== baro_rate_fpm bands ===")
display(baro_bands)

# dist_to_apt_nm bands
dist_bands = silver.select(
    when(col("dist_to_apt_nm").isNull(), "null")
    .when(col("dist_to_apt_nm") < 5, "0–5")
    .when(col("dist_to_apt_nm") < 10, "5–10")
    .when(col("dist_to_apt_nm") < 20, "10–20")
    .when(col("dist_to_apt_nm") < 50, "20–50")
    .when(col("dist_to_apt_nm") < 100, "50–100")
    .otherwise("100+").alias("dist_band")
).groupBy("dist_band").count().orderBy("dist_band")
print("=== dist_to_apt_nm bands ===")
display(dist_bands)

# bearing_to_apt bands (45-degree bins)
bearing_bands = silver.select(
    when(col("bearing_to_apt").isNull(), "null")
    .when(col("bearing_to_apt") < 45, "0–45")
    .when(col("bearing_to_apt") < 90, "45–90")
    .when(col("bearing_to_apt") < 135, "90–135")
    .when(col("bearing_to_apt") < 180, "135–180")
    .when(col("bearing_to_apt") < 225, "180–225")
    .when(col("bearing_to_apt") < 270, "225–270")
    .when(col("bearing_to_apt") < 315, "270–315")
    .otherwise("315–360").alias("bearing_band")
).groupBy("bearing_band").count().orderBy("bearing_band")
print("=== bearing_to_apt bands ===")
display(bearing_bands)

# COMMAND ----------

# DBTITLE 1,2a distribution charts
# Histograms for alt_ft, gs_kt, dist_to_apt_nm
pdf_silver = silver.select("alt_ft", "gs_kt", "dist_to_apt_nm").toPandas()

fig_alt = px.histogram(pdf_silver, x="alt_ft", title="Distribution of alt_ft", nbins=30)
fig_alt.show()

fig_gs = px.histogram(pdf_silver, x="gs_kt", title="Distribution of gs_kt", nbins=30)
fig_gs.show()

fig_dist = px.histogram(pdf_silver, x="dist_to_apt_nm", title="Distribution of dist_to_apt_nm", nbins=30)
fig_dist.show()

# COMMAND ----------

# DBTITLE 1,2b ground/airborne + reports per aircraft
# --- 2b: On-ground vs airborne, distinct aircraft, reports per aircraft ---

# Ground vs airborne
ground_airborne = silver.select(
    when(col("alt_ft") == 0, "on-ground").otherwise("airborne").alias("state")
).groupBy("state").count().orderBy("count", ascending=False)
print("=== On-ground vs airborne ===")
display(ground_airborne)

# Distinct aircraft
distinct_ac = silver.select("icao").distinct().count()
print(f"\nDistinct aircraft: {distinct_ac}")

# Reports per aircraft
rpc = silver.groupBy("icao").agg(count("*").alias("n_reports"))
rpc_stats = rpc.agg(
    min("n_reports").alias("min_reports"),
    expr("percentile_approx(n_reports, 0.5)").alias("median_reports"),
    max("n_reports").alias("max_reports"),
)
print("=== Reports per aircraft (min / median / max) ===")
display(rpc_stats)

# Distribution chart
rpc_pdf = rpc.toPandas()
fig_rpc = px.histogram(rpc_pdf, x="n_reports", title="Reports per Aircraft", nbins=20,
                       labels={"n_reports": "Number of reports"})
fig_rpc.show()

# COMMAND ----------

# DBTITLE 1,2c vertical rate analysis
# --- 2c: Vertical rate analysis ---
total = total_silver

vr = silver.agg(
    count("baro_rate_fpm").alias("baro_non_null"),
    count("geom_rate_fpm").alias("geom_non_null"),
    count(when(col("baro_rate_fpm").isNull() & col("geom_rate_fpm").isNull(), 1)).alias("both_null"),
).collect()[0]

vr_rows = [
    {"metric": "baro_rate_fpm non-null", "count": int(vr["baro_non_null"]), "pct": round(vr["baro_non_null"] / total * 100, 2)},
    {"metric": "geom_rate_fpm non-null", "count": int(vr["geom_non_null"]), "pct": round(vr["geom_non_null"] / total * 100, 2)},
    {"metric": "BOTH null (need delta-alt derivation)", "count": int(vr["both_null"]), "pct": round(vr["both_null"] / total * 100, 2)},
]
print("=== Vertical rate availability ===")
display(pd.DataFrame(vr_rows))

# COMMAND ----------

# DBTITLE 1,Section 3 — Classification
# MAGIC %md
# MAGIC ## 3 — Inbound / Outbound / Overflight Classification
# MAGIC
# MAGIC For airborne reports with non-null `track_deg` and `bearing_to_apt`:
# MAGIC - **Inbound**: absolute angular difference < 45° (flying toward the airport)
# MAGIC - **Outbound**: absolute angular difference > 135° (flying away)
# MAGIC - **Overflight**: otherwise (passing by)
# MAGIC
# MAGIC Also per-aircraft: is `dist_to_apt_nm` decreasing over the window? Is the aircraft descending (avg `baro_rate_fpm` < −250)?

# COMMAND ----------

# DBTITLE 1,3 classification + per-aircraft trend
# --- Section 3: Inbound / outbound / overflight classification ---

# Airborne with non-null track & bearing
classified = silver.filter(
    (col("alt_ft") > 0)
    & col("track_deg").isNotNull()
    & col("bearing_to_apt").isNotNull()
).withColumn(
    "raw_diff", abs(col("track_deg") - col("bearing_to_apt"))
).withColumn(
    "abs_angular_diff",
    when(col("raw_diff") <= 180, col("raw_diff"))
    .otherwise(lit(360) - col("raw_diff"))
).withColumn(
    "class",
    when(col("abs_angular_diff") < 45, "inbound")
    .when(col("abs_angular_diff") > 135, "outbound")
    .otherwise("overflight")
)

# Counts by class
class_counts = classified.groupBy("class").count().orderBy("count", ascending=False)
print("=== Classification counts ===")
display(class_counts)

# Per-aircraft trend: dist decreasing? descending?
w = Window.partitionBy("icao").orderBy("snapshot_ts").rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

ac_trend = (
    classified.withColumn("first_dist", first("dist_to_apt_nm").over(w))
    .withColumn("last_dist", last("dist_to_apt_nm").over(w))
    .groupBy("icao")
    .agg(
        max("first_dist").alias("first_dist"),
        max("last_dist").alias("last_dist"),
        avg("baro_rate_fpm").alias("avg_baro_rate"),
        count("*").alias("n_obs"),
    )
    .withColumn("dist_decreasing", col("first_dist") > col("last_dist"))
    .withColumn("descending", col("avg_baro_rate") < -250)
)

print("\n=== Per-aircraft trend (top 20 by distance decrease) ===")
display(ac_trend.orderBy(col("dist_decreasing").desc(), (col("first_dist") - col("last_dist")).desc()).limit(20))

# COMMAND ----------

# DBTITLE 1,3 map of last snapshot
# --- Section 3: Map of last snapshot coloured by class ---

last_ts = silver.agg(max("snapshot_ts")).collect()[0][0]
last_snap = classified.filter(col("snapshot_ts") == lit(last_ts))

# Airport coords from the bronze table (silver doesn't carry apt_lat/apt_lon)
apt_info = bronze.select("apt_lat", "apt_lon").limit(1).collect()[0]
apt_lat, apt_lon = apt_info["apt_lat"], apt_info["apt_lon"]

map_pdf = last_snap.select("lat", "lon", "class", "callsign", "alt_ft", "dist_to_apt_nm").toPandas()

fig = px.scatter_mapbox(
    map_pdf, lat="lat", lon="lon", color="class",
    hover_data=["callsign", "alt_ft", "dist_to_apt_nm"],
    title=f"Last snapshot ({last_ts}) — Airborne aircraft by class",
    zoom=7, height=500,
    color_discrete_map={"inbound": "green", "outbound": "red", "overflight": "blue"},
)
fig.add_trace(
    go.Scattermapbox(
        lat=[apt_lat], lon=[apt_lon],
        mode="markers", marker=dict(size=14, color="black", symbol="airport"),
        name="KATL",
    )
)
fig.update_layout(mapbox_style="carto-positron")
fig.show()

# COMMAND ----------

# DBTITLE 1,Section 4 — Callsign Shapes
# MAGIC %md
# MAGIC ## 4 — Callsign Shapes
# MAGIC
# MAGIC Classify callsigns as airline-style (3 letters then digits), GA tail number (starts with N), or other. List top 15 airline prefixes by distinct aircraft.

# COMMAND ----------

# DBTITLE 1,4 callsign classification
# --- Section 4: Callsign shapes ---

cs = silver.select("callsign", "icao").distinct()

cs_classified = cs.withColumn(
    "callsign_type",
    when(col("callsign").rlike(r"^[A-Z]{3}\d"), "airline")
    .when(col("callsign").rlike(r"^N\d"), "GA_tail")
    .otherwise("other")
)

cs_counts = cs_classified.groupBy("callsign_type").count().orderBy("count", ascending=False)
print("=== Callsign type counts (distinct aircraft-callsign pairs) ===")
display(cs_counts)

# Top 15 airline prefixes by distinct aircraft
airline_cs = cs_classified.filter(col("callsign_type") == "airline")
airline_prefix = (
    airline_cs.withColumn("prefix", substring(col("callsign"), 1, 3))
    .groupBy("prefix")
    .agg(countDistinct("icao").alias("distinct_aircraft"))
    .orderBy("distinct_aircraft", ascending=False)
    .limit(15)
)
print("\n=== Top 15 airline prefixes by distinct aircraft ===")
display(airline_prefix)

# COMMAND ----------

# DBTITLE 1,Section 5 — Data Quality
# MAGIC %md
# MAGIC ## 5 — Data-Quality Checks
# MAGIC
# MAGIC Checks for position/altitude/speed anomalies, emergency squawks, and duplicate dedup verification.

# COMMAND ----------

# DBTITLE 1,5 data quality checks
# --- Section 5: Data-quality checks ---
checks = []

# 1. has_position = true but lat or lon is null
c1 = silver.filter(col("has_position") == True).filter(col("lat").isNull() | col("lon").isNull()).count()
checks.append(("has_position=true & (lat OR lon) null", c1))

# 2. has_position = true but lat=0 AND lon=0
c2 = silver.filter(col("has_position") == True).filter((col("lat") == 0) & (col("lon") == 0)).count()
checks.append(("has_position=true & lat=0 & lon=0", c2))

# 3. alt_ft outside -1500..60000
c3 = silver.filter((col("alt_ft") < -1500) | (col("alt_ft") > 60000)).count()
checks.append(("alt_ft outside -1500..60000", c3))

# 4. gs_kt > 700
c4 = silver.filter(col("gs_kt") > 700).count()
checks.append(("gs_kt > 700", c4))

# 5. squawk in (7500, 7600, 7700)
c5 = silver.filter(col("squawk").isin("7500", "7600", "7700")).count()
checks.append(("squawk in (7500,7600,7700)", c5))

# 6. Duplicate (icao, snapshot_ts) pairs — should be 0 after dedup
dupes = silver.groupBy("icao", "snapshot_ts").agg(count("*").alias("cnt")).filter(col("cnt") > 1)
c6 = dupes.count()
checks.append(("duplicate (icao, snapshot_ts) pairs", c6))

print("=== Data-quality check results ===")
display(pd.DataFrame(checks, columns=["check_name", "count"]))

if c6 == 0:
    print("\n✓ Dedup verified: zero duplicate (icao, snapshot_ts) pairs.")
else:
    print(f"\n✗ Found {c6} duplicate (icao, snapshot_ts) pairs — dedup may have failed.")
    display(dupes.orderBy(col("cnt").desc()).limit(20))

# COMMAND ----------

# DBTITLE 1,Section 6 — What an Arrival Looks Like
# MAGIC %md
# MAGIC ## 6 — “What an Arrival Looks Like”
# MAGIC
# MAGIC The 5 aircraft with the largest decrease in `dist_to_apt_nm` over the window — plotted as distance and altitude vs `snapshot_ts`.

# COMMAND ----------

# DBTITLE 1,6 top 5 by distance decrease
# --- Section 6: Top 5 aircraft by largest distance decrease ---

w6 = Window.partitionBy("icao").orderBy("snapshot_ts").rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

arrival = (
    silver.filter(col("dist_to_apt_nm").isNotNull())
    .withColumn("first_dist", first("dist_to_apt_nm").over(w6))
    .withColumn("last_dist", last("dist_to_apt_nm").over(w6))
    .groupBy("icao")
    .agg(
        max("first_dist").alias("first_dist"),
        max("last_dist").alias("last_dist"),
        count("*").alias("n_obs"),
    )
    .withColumn("dist_decrease", col("first_dist") - col("last_dist"))
    .filter(
        (col("n_obs") >= 3)
        & col("first_dist").isNotNull()
        & col("last_dist").isNotNull()
    )
    .orderBy(col("dist_decrease").desc())
    .limit(5)
)

print("=== Top 5 aircraft by distance decrease ===")
display(arrival)

top5_icaos = [r["icao"] for r in arrival.collect()]
print(f"\nTop 5 ICAs: {top5_icaos}")

# COMMAND ----------

# DBTITLE 1,6 arrival plots
# --- Section 6: Plot dist_to_apt_nm & alt_ft vs snapshot_ts for each top-5 aircraft ---

top5_data = (
    silver.filter(col("icao").isin(top5_icaos))
    .select("icao", "callsign", "snapshot_ts", "dist_to_apt_nm", "alt_ft")
    .orderBy("icao", "snapshot_ts")
    .toPandas()
)

# Build subplot titles with callsign info
icao_to_cs = top5_data.groupby("icao")["callsign"].first().to_dict()
subplot_titles = [f"{icao} ({icao_to_cs.get(icao, '?')})" for icao in top5_icaos]

fig = make_subplots(
    rows=len(top5_icaos), cols=1,
    subplot_titles=subplot_titles,
    vertical_spacing=0.06,
    specs=[[{"secondary_y": True}]] * len(top5_icaos),
)

for i, icao in enumerate(top5_icaos, start=1):
    grp = top5_data[top5_data["icao"] == icao].sort_values("snapshot_ts")

    fig.add_trace(
        go.Scatter(x=grp["snapshot_ts"], y=grp["dist_to_apt_nm"],
                   name="dist_to_apt_nm", mode="lines+markers",
                   line=dict(color="blue")),
        row=i, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=grp["snapshot_ts"], y=grp["alt_ft"],
                   name="alt_ft", mode="lines+markers",
                   line=dict(color="red")),
        row=i, col=1, secondary_y=True,
    )

fig.update_yaxes(title_text="dist_to_apt_nm (nm)", secondary_y=False, row=1, col=1)
fig.update_yaxes(title_text="alt_ft", secondary_y=True, row=1, col=1)
fig.update_layout(
    height=200 * len(top5_icaos) + 100,
    title_text="Top 5 Arrivals — dist_to_apt_nm (left, blue) & alt_ft (right, red) over time",
    showlegend=False,
)
fig.show()

# COMMAND ----------

# DBTITLE 1,Final Summary
# MAGIC %md
# MAGIC ## Summary — Recommended Silver Column Additions & Data Issues
# MAGIC
# MAGIC ### Recommended Silver Column Additions
# MAGIC
# MAGIC Based on Section 1 field-coverage analysis (8 552 rows, 45 fields), these fields should be promoted from `bronze_aircraft.report` into `silver_positions`:
# MAGIC
# MAGIC | New Column | Source Field | Coverage | Priority | Rationale |
# MAGIC | --- | --- | --- | --- | --- |
# MAGIC | `alt_geom_ft` | `report.alt_geom` | 94.9 % | **High** | Geometric altitude fallback when barometric is missing/ground. Essential for continuous altitude tracking during approach. |
# MAGIC | `nav_altitude_mcp_ft` | `report.nav_altitude_mcp` | 76.3 % | **High** | Selected altitude (Mode Control Panel) — direct pilot descent-intent signal for arrival detection. |
# MAGIC | `nav_heading_deg` | `report.nav_heading` | 54.9 % | **High** | Selected heading — intended flight direction for approach pattern detection. |
# MAGIC | `nic` | `report.nic` | 100 % | **Medium** | Navigation Integrity Category — position quality flag for Gold filtering. |
# MAGIC | `rc` | `report.rc` | 100 % | **Medium** | Radius of containment — position uncertainty radius. |
# MAGIC | `seen_s` | `report.seen` | 100 % | **Medium** | Seconds since last message — staleness indicator. |
# MAGIC | `true_heading_deg` | `report.true_heading` | 4.7 % | **Low** | True heading; sparse coverage but complements track where available. |
# MAGIC | `mag_heading_deg` | `report.mag_heading` | 0.06 % | **Low** | Magnetic heading; nearly absent from this data source (5 / 8 552 rows). Not worth promoting unless coverage improves. |
# MAGIC | `seen_pos_s` | `report.seen_pos` | 100 % | **Confirm** | Already present in Silver — verify it maps correctly from `report.seen_pos`. |
# MAGIC
# MAGIC **Already promoted to Silver:** `lat`, `lon`, `alt_baro`→`alt_ft`, `gs`→`gs_kt`, `track`→`track_deg`, `baro_rate`→`baro_rate_fpm`, `geom_rate`→`geom_rate_fpm`, `squawk`, `emergency`, `category`, `flight`→`callsign`, `hex`→`icao`.
# MAGIC
# MAGIC **Not recommended:** `rssi`, `messages`, `mlat`, `tisb`, `nav_modes`, `dbFlags`, `gva`, `sda`, `sil`, `sil_type`, `spi`, `alert`, `version`, `nac_p`, `nac_v`, `nic_baro`, `nav_altitude_fms`, `nav_qnh`, `calc_track`, `r`, `t`, `type` — low analytical value for arrival detection.
# MAGIC
# MAGIC ### Data Issues Affecting Touchdown Detection
# MAGIC
# MAGIC 1. **Vertical rate gaps (Section 2c):** `baro_rate_fpm` is non-null 78.6 % of the time, `geom_rate_fpm` only 18.8 %, and **both are null for 5.2 % (442 rows)**. For those rows, descent rate must be derived from consecutive altitude deltas (`Δalt_ft / Δt`). This derivation should be built into the Gold layer.
# MAGIC
# MAGIC 2. **Track completeness (Section 2b):** 523 distinct aircraft across 19 snapshots. Reports-per-aircraft: **min = 1, median = 19, max = 19** — most aircraft appear in all snapshots, but some appear only once. Full arrival-wave detection requires hours of data; this 4.5-minute window validates schema and field quality only.
# MAGIC
# MAGIC 3. **Classification reliability (Section 3):** Inbound 2 338, outbound 2 402, overflight 3 258 airborne reports. The classification is a snapshot heuristic (track vs bearing). With only 4.5 minutes, the “distance decreasing” trend is based on first-vs-last only. The Gold layer should use a rolling window (3+ consecutive decreasing-distance observations) before labelling an aircraft as inbound.
# MAGIC
# MAGIC 4. **Ground detection (Section 2b):** 471 reports (5.5 %) have `alt_ft = 0` (on-ground). ADS-B ground altitude can be noisy; a secondary ground-speed check (`gs_kt < 50` with `alt_ft ≤ 0`) would make touchdown detection more robust.
# MAGIC
# MAGIC 5. **No `roll` field:** `roll` is **not present** in the `report` struct from this data source. Bank-angle would need to be derived from heading changes over time.
# MAGIC
# MAGIC 6. **Dedup verified (Section 5):** All six data-quality checks pass with **zero** anomalies — no null/zero lat-lon with `has_position`, no altitude outliers, no speed outliers, no emergency squawks, and **zero duplicate `(icao, snapshot_ts)` pairs** confirming the Silver dedup logic works correctly.
# MAGIC
# MAGIC 7. **Distance decrease is modest (Section 6):** The top-5 aircraft by distance decrease moved only ~35 nm closer over 4.5 minutes — consistent with cruise-speed approach from 60–200+ nm out. None appear to be in the final approach phase (within ~10 nm). Longer data windows are needed to observe full arrival trajectories through touchdown.
# MAGIC
# MAGIC 8. **`alt_baro` is a string in Bronze** (e.g., `"ground"` vs numeric). Silver correctly types it as `alt_ft` (int); the Bronze-to-Silver transformation handles the `"ground"` → 0 conversion correctly.