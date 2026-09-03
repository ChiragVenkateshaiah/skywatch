# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — live ADS-B poller
# MAGIC
# MAGIC Polls the **airplanes.live** REST API for every aircraft within `radius_nm` of the target
# MAGIC airport and drops one raw JSON file per poll into a UC Volume. Auto Loader (in the medallion
# MAGIC pipeline) ingests from there — this notebook only lands raw data, it does no parsing.
# MAGIC
# MAGIC - API: `GET https://api.airplanes.live/v2/point/<lat>/<lon>/<radius_nm>` (radius ≤ 250 nm,
# MAGIC   responses conform to the ADSB Exchange v2 schema, rate limit 1 req/s, non-commercial —
# MAGIC   credit "airplanes.live").
# MAGIC - One job run = one polling *burst*: it polls every `poll_interval_seconds` for
# MAGIC   `poll_seconds` total, then exits. The Asset Bundle schedules the run every few minutes.
# MAGIC   Set `poll_seconds=0` for a single poll per run.
# MAGIC
# MAGIC Output layout:
# MAGIC `/Volumes/<catalog>/<schema>/<volume>/<source_name>/dt=YYYY-MM-DD/hh=HH/<now_ms>.json`

# COMMAND ----------
# MAGIC %md ## 1. Parameters

# COMMAND ----------
import json, os, time
from datetime import datetime, timezone

import requests

_DEFAULTS = {
    "catalog": "skywatch",
    "schema": "core",
    "volume": "landing",
    "apt_icao": "KATL",
    "apt_lat": "33.6407",
    "apt_lon": "-84.4277",
    "radius_nm": "250",
    "poll_seconds": "270",          # poll for 4.5 min inside a run scheduled every 5 min
    "poll_interval_seconds": "15",  # >= 1 to respect the API rate limit
    "source_name": "airplaneslive",
}

try:
    for k, v in _DEFAULTS.items():
        dbutils.widgets.text(k, v)
    P = {k: dbutils.widgets.get(k) for k in _DEFAULTS}
except Exception:
    P = dict(_DEFAULTS)

CATALOG, SCHEMA, VOLUME = P["catalog"], P["schema"], P["volume"]
APT_ICAO = P["apt_icao"]
APT_LAT, APT_LON = float(P["apt_lat"]), float(P["apt_lon"])
RADIUS_NM = int(float(P["radius_nm"]))
POLL_SECONDS = int(float(P["poll_seconds"]))
POLL_INTERVAL = max(1, int(float(P["poll_interval_seconds"])))
SOURCE = P["source_name"]

print(f"{APT_ICAO} @ ({APT_LAT}, {APT_LON})  radius={RADIUS_NM} nm")
print(f"burst: {POLL_SECONDS}s total, every {POLL_INTERVAL}s  ->  {CATALOG}.{SCHEMA}.{VOLUME}/{SOURCE}/")

# COMMAND ----------
# MAGIC %md ## 2. Ensure catalog / schema / volume

# COMMAND ----------
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA  IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME  IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
VOL_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{SOURCE}"
print("landing at", VOL_ROOT)

# COMMAND ----------
# MAGIC %md ## 3. Poll

# COMMAND ----------
API = f"https://api.airplanes.live/v2/point/{APT_LAT}/{APT_LON}/{RADIUS_NM}"
HEADERS = {
    "User-Agent": "skywatch-portfolio/0.1 (+https://github.com/ChiragVenkateshaiah/skywatch)",
    "Accept": "application/json",
}


def one_poll(session: requests.Session) -> dict:
    """Fetch once; write the raw response envelope to the Volume. Returns a small status dict."""
    r = session.get(API, headers=HEADERS, timeout=30)
    r.raise_for_status()
    body = r.json()

    # The API stamps the snapshot time in `now` (epoch ms). Fall back to wall clock.
    now_ms = int(body.get("now") or time.time() * 1000)
    ts = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)

    # tag each record with the airport this poll was centred on, so a future multi-airport
    # setup keeps its provenance in the raw layer
    body["capture"] = {
        "apt_icao": APT_ICAO,
        "apt_lat": APT_LAT,
        "apt_lon": APT_LON,
        "radius_nm": RADIUS_NM,
        "source": SOURCE,
        "polled_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    out_dir = f"{VOL_ROOT}/dt={ts:%Y-%m-%d}/hh={ts:%H}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{now_ms}.json"
    with open(out_path, "w") as f:
        json.dump(body, f, separators=(",", ":"))

    return {"now_ms": now_ms, "aircraft": int(body.get("total") or len(body.get("ac", []))),
            "bytes": len(r.content), "path": out_path}


deadline = time.time() + POLL_SECONDS
results, errors = [], 0
with requests.Session() as s:
    while True:
        start = time.time()
        try:
            results.append(one_poll(s))
        except Exception as e:  # noqa: BLE001 — a transient poll failure must not kill the burst
            errors += 1
            print("poll error:", repr(e))
        if time.time() >= deadline:
            break
        time.sleep(max(0.0, POLL_INTERVAL - (time.time() - start)))

# COMMAND ----------
# MAGIC %md ## 4. Summary

# COMMAND ----------
ok = len(results)
print(f"{ok} polls written, {errors} errors")
if ok == 0:
    raise RuntimeError(
        "No polls succeeded. Check outbound HTTP from serverless and that "
        f"{API} is reachable."
    )

if results:
    import pandas as pd
    summary = pd.DataFrame(results)
    print(f"aircraft per poll: min={summary.aircraft.min()} "
          f"max={summary.aircraft.max()} mean={summary.aircraft.mean():.0f}")
    display(spark.createDataFrame(summary))
