# SkyWatch Lite

Historical airspace analytics on **Databricks Free Edition**, built for the Databricks Builder Launchpad.

Takes a few minutes of global aircraft-transponder snapshots from the **ADS-B Exchange** free
sample archive and refines them through a medallion pipeline into a dashboard + Genie space that
answers "what was happening in the sky during this window?"

**Where this is going:** [`docs/ML_ROADMAP.md`](docs/ML_ROADMAP.md) — the plan to grow this into
an **Arrival Manager** (per-flight touchdown-time prediction + arrival demand forecasting),
served through a Databricks App, exercising the full Databricks ML platform. Includes a
Free Edition vs Production capability matrix.

---

## What SkyWatch does — the Arrival Manager

*The one-page version. Full pitch: [`pitch.md`](pitch.md).*

At a busy hub like Atlanta, **arrival demand is spiky** — sometimes more aircraft want to land in
a 15-minute window than the runways can accept. Controllers absorb the overflow with holding and
vectoring, which burns fuel and pushes delay downstream, and it's hard to see coming. Real
systems that manage this (Eurocontrol AMAN, FAA TBFM) are expensive infrastructure. The signal
they run on — live aircraft position and speed — is exactly what **ADS-B transponder data gives
us for free**.

SkyWatch turns that free stream into an **Arrival Manager** built on two models:

| | **Model 1 — Time-to-touchdown** | **Model 2 — Arrival demand forecast** |
|---|---|---|
| **Question** | For *this* aircraft, right now, how many minutes until it lands at KATL? | How many aircraft will land in each future 15-minute bin, out to +3 h? |
| **ML shape** | Tabular **regression** — one row per position report | Univariate **time-series forecast** — one value per 15-min bin |
| **Label** | `minutes_to_touchdown`, self-supervised from detected landings on historical trajectories | count of landings per bin (same detection, aggregated) |
| **Algorithm** | LightGBM + Hyperopt, ~480k reports / 24 days | bake-off (seasonal-naive, AutoETS/ARIMA, Chronos-Bolt); **climatological mean** won |
| **Result** | **MAE ≈ 1.14 min** vs a 5.9 min "distance ÷ speed" baseline (~80% better) | **MASE 0.79** — beats seasonal-naive, and zero-shot Chronos by ~50% |
| **Powers** | the **predicted arrival sequence** (landing order + timing) | the **demand curve + surge alerts** vs the Airport Acceptance Rate |

**How they combine:** Model 1's per-flight ETAs cover **0–45 min out** (real airborne aircraft,
precise); Model 2's statistical forecast covers **45 min – 3 h** (aircraft not yet airborne or in
range). Stitched together they give one continuous "arrivals expected" curve against the runway
capacity line.

> **Analogy:** Model 1 is Google Maps ETA for each individual car heading toward a bridge.
> Model 2 is the rush-hour traffic forecast for that bridge over the next 3 hours. Together you
> know both *who arrives when* and *whether the total will exceed what the bridge can handle.*

These are two genuinely different ML problems — "predict an outcome from an object's current
state" vs. "predict a quantity's future from its history" — with different data shapes,
algorithms, and metrics. Both are served to a coordinator through a Streamlit **Databricks App**
that reads the batch-scored predictions from Delta (Free Edition has no model-serving endpoints).

A third strand — **Model 3, irregularity early-warning** (holding / go-around / emergency) —
ships as **geometry + squawk rules rather than a learned classifier**. Two rounds of
investigation (9 days, then 24 days at a finer sampling rate) showed the same thing: of 159
detected "holds", **zero** were aircraft that then landed — all persistent loiterers. KATL
doesn't hold arrivals in clear weather, and clear-weather days are all the archive has. Deciding
that on evidence rather than shipping a weak model is itself part of the story — see
[`pitch.md`](pitch.md).

---

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
  Silver  skywatch.core.silver_positions    one typed row per aircraft report
                                            (has_position flag; emergency rows kept even w/o a fix)
        ▼
  Gold    skywatch.core.gold_*              kpis · h3_density · special_squawks · orbits
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
