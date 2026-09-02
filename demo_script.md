# SkyWatch Lite — dashboard build + demo

## A. Build the dashboard from a Genie space (~15 min)

### A0. Create the Genie space
1. Left nav **Genie** -> **New**. Name it **SkyWatch Lite**. Warehouse: **Serverless Starter Warehouse**.
2. **Data** -> add tables from `skywatch.core`:
   `silver_positions, gold_kpis, gold_h3_density, gold_orbits, gold_special_squawks,
   gold_airline_activity, gold_altitude_bands, gold_type_mix, gold_briefing`.
3. **Instructions** -> paste:

   > SkyWatch Lite: ~5 minutes of global ADS-B aircraft transponder snapshots from ADS-B Exchange
   > (2024-06-01 ~12:00 UTC), refined into a medallion model.
   > - `silver_positions`: one row per aircraft per snapshot. `icao` = airframe hex. First 3 letters
   >   of `callsign` = airline ICAO code. `alt_ft` = barometric altitude ft (0 = on ground).
   >   `gs_kt` = ground speed kt. `emergency` is 'none' normally, else general / minfuel / nordo /
   >   lifeguard / unlawful / downed. `has_position` = had a lat/lon fix.
   > - `gold_kpis`: one row. `aircraft_tracked` = confirmed (with a fix); `raw_contacts` also counts
   >   unverified targets.
   > - `gold_h3_density`: H3 resolution-3 cell counts; `h3_cell` is an H3 string id.
   > - `gold_orbits`: aircraft that circled / held. `heading_spread` 0..1 (1 = turned through every
   >   direction); `ns_km` / `ew_km` = bounding-box size in km.
   > - `gold_special_squawks`: emergency broadcasts; `confidence` = high / medium / low.
   > - `gold_airline_activity`; `gold_altitude_bands` (`band_sort` orders the bands); `gold_type_mix`.
   > - `gold_briefing`: one LLM-generated paragraph.
   > Answer from the `gold_*` tables whenever one matches; use `silver_positions` only for
   > per-aircraft detail.

### A1. Ask these one per message — after each answer click **Add to dashboard** -> a *new* dashboard "SkyWatch Lite"

| # | Paste into Genie | Expect |
|---|---|---|
| 1 | `From gold_kpis show aircraft_tracked, position_reports and pct_airborne as big-number counters.` | counters |
| 2 | `From gold_kpis, how many aircraft broadcast an emergency status? Show as a counter.` | counter |
| 3 | `Plot the latest position of every aircraft in silver_positions that has_position, as points on a map.` | world map |
| 4 | `Map air-traffic density from gold_h3_density using h3_cell, colored by position_reports.` | H3 map (skip if it errors) |
| 5 | `From gold_orbits list callsign, ac_type, avg_alt_ft, ns_km, ew_km, heading_spread sorted by heading_spread descending.` | table (headline) |
| 6 | `Show the gold_orbits aircraft on a map using approx_lat and approx_lon, point size by pings.` | map |
| 7 | `From gold_special_squawks where confidence is high or medium, show event_type, callsign, confidence, approx_lat, approx_lon.` | table |
| 8 | `From gold_airline_activity, bar chart of aircraft by airline_icao, top 15 descending.` | bar |
| 9 | `From gold_altitude_bands, bar chart of position_reports by altitude_band ordered by band_sort.` | bar |
| 10 | `From gold_type_mix, bar chart of the 15 aircraft types with the most airframes.` | bar |
| 11 | `Show the text in gold_briefing.` | text |

If Genie picks the wrong table, resend the prompt prefixed with `Use only <table>:`.

### A2. Open the dashboard -> tidy + publish
- Drag tiles: counters row on top; world map large on the left; orbits table + orbits map top-right;
  emergency table in the middle; the three bars along the bottom; briefing as a footer.
- Add a **Text** widget at the top:
  `# SkyWatch Lite  —  ADS-B Exchange sample archive (CC-BY-NC), 2024-06-01 11:59-12:04 UTC`
- **Publish** (top-right) -> keep **Embed credentials** on so judges can view it live.

### A3. Capture the dashboard back into the bundle (optional IaC finish)
```bash
export DATABRICKS_CONFIG_PROFILE=skywatch
databricks bundle generate dashboard --existing-dashboard-id <id-from-the-URL>
databricks bundle deploy -t dev
```

## B. Keep the Genie space for the live demo
The same space is your Q&A moment. Star these so they show as suggestions:
- How many distinct aircraft had an emergency status?
- Which aircraft type was most common above 30,000 ft?
- Show flights faster than 500 knots ground speed.
- Which airline had the most aircraft airborne?

## C. Demo script (60-90 sec)

> "The brief was to consume ADS-B Exchange data. I pulled ~5 minutes of the entire planet's
> air traffic — their free sample archive, no API key — into a Databricks Free Edition lakehouse.
> Everything is deployed as code: a Databricks Asset Bundle — `databricks bundle deploy` creates
> the serverless job, `bundle run` runs bronze -> silver -> gold. No clicking.
>
> [Dashboard] 12,809 confirmed aircraft in that window, 651k position reports. World traffic map,
> busiest airlines, altitude distribution.
>
> [Orbits tile] The payoff: these aircraft turned through the whole compass while staying inside a
> ~10 km box — holding stacks over London, a survey aircraft near Copenhagen, a trainer doing
> airwork over Saudi. Found with a circular-variance metric on the heading, so the 360->0 wrap
> doesn't fool it.
>
> [Emergency table] We also pull every emergency-status broadcast — but raw ADS-B is noisy, so each
> is tiered by confidence: this firefighting aircraft over the Sea of Marmara is high-confidence
> (full track, callsign), the rest are anonymized targets with just the emergency bit set.
>
> [Genie] Anyone can interrogate it in plain English — *'show me flights faster than 500 knots
> ground speed.'* [run] — chart, no SQL. And this footer paragraph is generated by `ai_query`
> against Llama 3.3 right in the SQL pipeline.
>
> Next: swap the batch source for scheduled live polling and add airframe-registry enrichment."

## Cut list if behind
Drop the Genie Q&A polish -> drop the H3 map (#4) and orbits map (#6) -> ship counters + world map
+ orbits table + emergency table.
