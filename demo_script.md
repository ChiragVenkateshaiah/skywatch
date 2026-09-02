# SkyWatch Lite — dashboard build + demo

## A. Build the AI/BI dashboard (~12 min)

1. **SQL Editor** → confirm the 2X-Small SQL warehouse is running.
2. **Dashboards → Create dashboard**. Add these datasets (each is `SELECT * FROM <table>`):
   - `skywatch.core.gold_kpis`
   - `skywatch.core.gold_h3_density`
   - `skywatch.core.gold_orbits`
   - `skywatch.core.gold_special_squawks`
   - `skywatch.core.gold_airline_activity`
   - `skywatch.core.gold_altitude_bands`
   - `skywatch.core.gold_type_mix`
   - `skywatch.core.gold_briefing`  (stretch — the AI text tile)
3. Tiles:
   | Tile | Dataset | Viz | Notes |
   |---|---|---|---|
   | Aircraft tracked / Position reports / % airborne / Emergency aircraft | gold_kpis | Counter | one per metric |
   | Global traffic density | gold_h3_density | **Map (H3)** | H3 column = `h3_cell`, color by `position_reports` |
   | Circling / holding aircraft | gold_orbits | Table + Map | **headline** — `callsign, ac_type, avg_alt_ft, ns_km, ew_km, heading_spread`; map on `approx_lat`/`approx_lon` |
   | Emergency broadcasts (tiered) | gold_special_squawks | Table | filter `confidence <> 'low'` for the tile; show `event_type, callsign, confidence, approx_lat, approx_lon` |
   | Busiest airlines | gold_airline_activity | Bar | `airline_icao` × `aircraft`, top 15 |
   | Altitude distribution | gold_altitude_bands | Bar | `altitude_band` × `position_reports` |
   | Aircraft type mix | gold_type_mix | Bar | top 15 by `aircraft` |
   | AI airspace briefing | gold_briefing | Text / Markdown | stretch — bind the `briefing` field |
4. Title it **SkyWatch Lite — Global Airspace Snapshot**, add a text tile:
   "Source: ADS-B Exchange sample archive (CC-BY-NC). Window: <window_start>–<window_end> UTC."
5. **Publish**.

## B. Genie space (~4 min)

1. **Genie → New**. Add tables: `silver_positions` + all `gold_*`.
2. Instructions: "Aircraft position snapshots over a ~5-minute window. `icao` = airframe hex,
   first 3 letters of `callsign` = airline ICAO code, `alt_ft` = barometric altitude,
   `gs_kt` = ground speed in knots, `emergency` <> 'none' or squawk 7500/7600/7700 = emergency,
   `has_position` = whether the row had a lat/lon fix."
3. Sample questions to save:
   - How many distinct aircraft had an emergency status?
   - Which aircraft type was most common above 30,000 ft?
   - Show flights faster than 500 knots ground speed.
   - Which airline had the most aircraft airborne?

## C. Demo script (60–90 sec)

> "The brief was to consume ADS-B Exchange data. I pulled ~5 minutes of the entire planet's
> air traffic — their free sample archive, no API key — into a Databricks Free Edition lakehouse.
> Everything you see is deployed as code: a Databricks Asset Bundle — `databricks bundle deploy`
> creates the serverless job, `bundle run` runs bronze → silver → gold. No clicking.
>
> [Dashboard] ~12,800 aircraft in that window, 650k position reports. This is the traffic heatmap,
> built with Databricks' native H3 geospatial functions — North Atlantic track, the European core,
> the US hubs. Busiest airlines, altitude distribution.
>
> [Orbits tile] The payoff: these aircraft turned through the whole compass while staying inside a
> ~10 km box — holding stacks over London, a survey aircraft near Copenhagen, a trainer doing
> airwork over Saudi. Found with a circular-variance metric on the heading, so the 360→0 wrap
> doesn't fool it.
>
> [Squawk table] We also pull every emergency-status broadcast — but raw ADS-B is noisy, so each is
> tiered: this firefighting aircraft over the Sea of Marmara is high-confidence (full track,
> callsign), the rest are anonymized targets with just the emergency bit set.
>
> [Genie] Anyone can interrogate it in plain English — *'show me flights faster than 500 knots
> ground speed.'* [run] — chart, no SQL.
>
> Next: swap the batch source for scheduled live polling, add airframe-registry enrichment, and an
> LLM briefing tile — the `ai_query` is already in the notebook."

## Cut list if behind
Drop Genie → drop stretch cells → ship dashboard with just KPIs + H3 map + squawk table.
