# SkyWatch Lite

Historical airspace analytics on **Databricks Free Edition**, built for the Databricks Builder Launchpad.

Takes a few minutes of global aircraft-transponder snapshots from the **ADS-B Exchange** free
sample archive and refines them through a medallion pipeline into a dashboard + Genie space that
answers "what was happening in the sky during this window?"

## Data source

`https://samples.adsbexchange.com/readsb-hist/<yyyy>/<mm>/<dd>/` — one global snapshot every
~5 seconds, named `HHMMSSZ.json.gz` but **served decompressed as plain JSON** (~6 MB each,
~13.5k aircraft). Free, no key. CC-BY-NC — credit "ADS-B Exchange" on the dashboard.

Verified working dates include `2024/06/01`, `2024/01/01`, `2023/01/01`, `2022/06/01`, `2016/07/01`.
Each aircraft record: `hex, flight, r, t, alt_baro, gs, track, squawk, emergency, category, lat, lon`.

## Architecture

```
samples.adsbexchange.com/.json.gz
        │  (notebook cell "Land": requests -> UC Volume)
        ▼
  Bronze  skywatch.core.bronze_aircraft     raw, aircraft array exploded
        ▼
  Silver  skywatch.core.silver_positions    one typed row per aircraft-position
        ▼
  Gold    skywatch.core.gold_*              kpis · h3_density · special_squawks
                                            airline_activity · altitude_bands · type_mix
        │
        ├── AI/BI Dashboard  (map on gold_h3_density, KPI tiles, bar charts, squawk table)
        └── Genie space      (silver_positions + all gold_* tables)
```

## Deploy via Databricks Asset Bundle (IaC)

```
databricks.yml              bundle definition + targets + variables
resources/skywatch.job.yml  serverless job that runs the pipeline notebook
src/skywatch_lite.py        the medallion pipeline (notebook source)
```

One-time setup:

```bash
# 1. Authenticate the CLI to your Free Edition workspace (opens a browser)
databricks auth login --host https://<your-workspace>.cloud.databricks.com --profile skywatch

# 2. Put that same host into databricks.yml -> targets.dev.workspace.host
```

Deploy + run:

```bash
export DATABRICKS_CONFIG_PROFILE=skywatch
databricks bundle validate -t dev
databricks bundle deploy   -t dev          # uploads notebook + creates the job
databricks bundle run skywatch_lite -t dev # runs the pipeline; streams logs

# change ingestion window without editing code:
databricks bundle run skywatch_lite -t dev --var="date_path=2023/01/01,n_files=90"
```

Then build the dashboard + Genie space manually (see `demo_script.md`). Once the dashboard
exists you can capture it back into the bundle:

```bash
databricks bundle generate dashboard --existing-dashboard-id <id>   # writes src/*.lvdash.json
```

## Fallback if the job can't reach the internet

Serverless usually has outbound HTTP. If the "Land" cell fails:

1. On your laptop, download ~40 files (saved as `.json` because they arrive decompressed):
   ```bash
   D=2024/06/01
   for i in $(seq 0 5 295); do
     s=$(printf '12%02d%02d' $((i/60)) $((i%60)))
     curl -sf "https://samples.adsbexchange.com/readsb-hist/$D/${s}Z.json.gz" -o "${s}Z.json"
   done
   ```
2. In Databricks: Catalog -> `skywatch.core.landing` volume -> **Upload to volume** -> drop the files.
3. Re-run the job starting from the Bronze cell (or just re-run; the download loop no-ops on
   files that already exist).

## Free Edition constraints this design respects

Serverless only · 1 SQL warehouse (2X-Small) · max 5 concurrent job tasks · batch not streaming ·
no app / no ML in MVP · daily usage quota — don't re-run the ingest repeatedly.
