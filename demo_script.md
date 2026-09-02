# SkyWatch Lite — dashboard build + demo

## A. Build the AI/BI dashboard with the assistant (~12 min)

### A0. Create + add datasets
1. Left nav **Dashboards → Create dashboard**. Rename (top-left) to
   **SkyWatch Lite — Global Airspace Snapshot**. Pick the **Serverless Starter Warehouse**.
2. **Data** tab → **Create from SQL** for each of these (name each dataset as shown):

   | dataset name | SQL |
   |---|---|
   | `kpis`        | `SELECT * FROM skywatch.core.gold_kpis` |
   | `traffic`     | `SELECT lat, lon, alt_ft, callsign FROM skywatch.core.silver_positions WHERE has_position QUALIFY row_number() OVER (PARTITION BY icao ORDER BY snapshot_ts DESC)=1` |
   | `h3`          | `SELECT * FROM skywatch.core.gold_h3_density` |
   | `orbits`      | `SELECT * FROM skywatch.core.gold_orbits ORDER BY heading_spread DESC` |
   | `emergencies` | `SELECT * FROM skywatch.core.gold_special_squawks` |
   | `airlines`    | `SELECT * FROM skywatch.core.gold_airline_activity LIMIT 15` |
   | `altitude`    | `SELECT * FROM skywatch.core.gold_altitude_bands` |
   | `types`       | `SELECT * FROM skywatch.core.gold_type_mix LIMIT 15` |
   | `briefing`    | `SELECT briefing FROM skywatch.core.gold_briefing` |

### A1. Add each widget via the assistant
**Canvas** tab → click **Add visualization** (or draw a box) → in the widget, pick the **dataset**,
then type the prompt into **"Ask the assistant"**. Prompts:

| Widget | Dataset | Assistant prompt | Fix up by hand |
|---|---|---|---|
| Counter — aircraft | `kpis` | `big number of aircraft_tracked, label "Aircraft tracked"` | — |
| Counter — reports | `kpis` | `big number of position_reports` | — |
| Counter — airborne | `kpis` | `counter of pct_airborne, show as percent` | — |
| Counter — emergencies | `kpis` | `counter of emergency_aircraft, label "Emergency contacts"` | — |
| **World traffic map** | `traffic` | `map with a point for every lat and lon` | Viz = **Symbol map**; size small, opacity ~0.4 |
| H3 density (flex) | `h3` | `map colored by position_reports using h3_cell` | if H3 not offered, skip — `traffic` map covers it |
| **Circling / holding** (headline) | `orbits` | `table of callsign, ac_type, avg_alt_ft, ns_km, ew_km, heading_spread` | sort `heading_spread` desc |
| Orbits map | `orbits` | `map of approx_lat and approx_lon, point size by pings` | Symbol map |
| Emergency broadcasts | `emergencies` | `table of event_type, callsign, confidence, approx_lat, approx_lon` | filter `confidence` is `high` or `medium`; sort by `confidence` |
| Busiest airlines | `airlines` | `horizontal bar of aircraft by airline_icao, descending` | — |
| Altitude distribution | `altitude` | `bar of position_reports by altitude_band, keep band order` | sort by `band_sort` |
| Aircraft type mix | `types` | `bar of aircraft by ac_type` | — |
| AI briefing | `briefing` | *(no chart)* use a **Text** widget → insert field `briefing` | stretch |

### A2. Finishing touches
- Top **Text** widget: `# SkyWatch Lite`  ·  second line
  `Source: ADS-B Exchange sample archive (CC-BY-NC) · window 2024-06-01 11:59–12:04 UTC · 12,809 aircraft`
- Layout: counters as a row across the top, world map big on the left, orbits table + map top-right,
  emergency table below, three bars along the bottom, briefing as a footer.
- **Publish** (top-right) → toggle **Embed credentials** so judges can view it live.

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
