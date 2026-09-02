# SkyWatch Lite — dashboard build + demo

## A. Build the AI/BI dashboard (~12 min)

1. **SQL Editor** → confirm the 2X-Small SQL warehouse is running.
2. **Dashboards → Create dashboard**. Add these datasets (each is `SELECT * FROM <table>`):
   - `skywatch.core.gold_kpis`
   - `skywatch.core.gold_h3_density`
   - `skywatch.core.gold_special_squawks`
   - `skywatch.core.gold_airline_activity`
   - `skywatch.core.gold_altitude_bands`
   - `skywatch.core.gold_type_mix`
3. Tiles:
   | Tile | Dataset | Viz | Notes |
   |---|---|---|---|
   | Aircraft tracked / Position reports / % airborne | gold_kpis | Counter | one per metric |
   | Global traffic density | gold_h3_density | **Map (H3)** | H3 column = `h3_cell`, color by `position_reports` |
   | Emergency & special squawks | gold_special_squawks | Table | sort by `first_seen`; this is the headline |
   | Busiest airlines | gold_airline_activity | Bar | `airline_icao` × `aircraft`, top 15 |
   | Altitude distribution | gold_altitude_bands | Bar | `altitude_band` × `position_reports` |
   | Aircraft type mix | gold_type_mix | Bar | top 15 by `aircraft` |
4. Title it **SkyWatch Lite — Global Airspace Snapshot**, add a text tile:
   "Source: ADS-B Exchange sample archive (CC-BY-NC). Window: <window_start>–<window_end> UTC."
5. **Publish**.

## B. Genie space (~4 min)

1. **Genie → New**. Add tables: `silver_positions` + all `gold_*`.
2. Instructions: "Aircraft position snapshots. `icao` = airframe, `callsign` first 3 letters = airline
   ICAO, `alt_ft` barometric altitude, `gs_kt` ground speed knots, squawk 7500/7600/7700 = emergency."
3. Sample questions to save:
   - How many aircraft squawked an emergency code?
   - Which aircraft type was most common above 30,000 ft?
   - Show flights faster than 500 knots ground speed.
   - Which airline had the most aircraft airborne?

## C. Demo script (60–90 sec)

> "The brief was to consume ADS-B Exchange data. I pulled ~5 minutes of the entire planet's
> air traffic — their free sample archive, no API key — into a Databricks Free Edition lakehouse:
> bronze, silver, gold, all serverless.
>
> [Dashboard] ~11,000 aircraft in that window. This is the traffic heatmap built with Databricks'
> native H3 geospatial functions — you can see the North Atlantic track, the European core, the
> US hubs. Busiest airlines here, altitude distribution here.
>
> [Point at squawk table] And this is the payoff: every aircraft that broadcast an emergency or
> special transponder code in that window — [pick a row].
>
> [Genie] And anyone can interrogate it in plain English — watch: *'show me flights faster than
> 500 knots ground speed.'* [run] — chart, no SQL.
>
> Next steps would be live polling on a schedule, military-airframe enrichment, and an LLM briefing
> tile — the query's already written."

## Cut list if behind
Drop Genie → drop stretch cells → ship dashboard with just KPIs + H3 map + squawk table.
