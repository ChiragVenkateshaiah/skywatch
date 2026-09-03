# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — historical backfill
# MAGIC
# MAGIC Pulls **ADS-B Exchange `readsb-hist`** global snapshots for a range of past days, keeps only
# MAGIC the aircraft within `radius_nm` of the target airport, and writes each one wrapped in the
# MAGIC **same envelope the live poller uses** so it flows through Bronze → Silver → Gold unchanged
# MAGIC (the `capture.source` tag is `"backfill"` instead of `"adsblol"`).
# MAGIC
# MAGIC - Source: `https://samples.adsbexchange.com/readsb-hist/<yyyy>/<mm>/<dd>/HHMMSSZ.json.gz`
# MAGIC   (served decompressed, ~6 MB each, native cadence ~5 s, CC-BY-NC — credit "ADS-B Exchange").
# MAGIC - **Resumable**: one output file per target timestamp; existing outputs are skipped, so the
# MAGIC   job can be re-run in chunks across Free Edition quota windows.
# MAGIC - **Be a good citizen**: this hits a free community archive. Keep `interval_seconds` coarse
# MAGIC   (≥ 120), `radius_nm` tight (≤ 100), and `request_sleep_s` > 0. Each 6 MB global file is
# MAGIC   downloaded in full and filtered locally — the download volume, not the row count, is the
# MAGIC   cost. ~3 days × 120 s ≈ 2 000 files ≈ 13 GB.
# MAGIC
# MAGIC Output: `/Volumes/<catalog>/<schema>/<volume>/backfill/dt=YYYY-MM-DD/hh=HH/<yyyymmdd_hhmm>.json`

# COMMAND ----------
# MAGIC %md ## Parameters

# COMMAND ----------
import gzip
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone

import requests

_DEFAULTS = {
    "catalog": "skywatch",
    "schema": "core",
    "volume": "landing",
    "apt_lat": "33.6407",
    "apt_lon": "-84.4277",
    "apt_icao": "KATL",
    "radius_nm": "80",           # tight — backfill is for touchdown / demand, near the field
    "start_date": "",            # YYYY-MM-DD; blank = (today UTC - n_days)
    "n_days": "3",
    "interval_seconds": "120",   # target spacing between snapshots
    "hours": "",                 # "" = all; else inclusive UTC range e.g. "11-23"
    "max_files": "6000",         # hard safety cap on writes per run
    "request_sleep_s": "0.25",
}

try:
    for k, v in _DEFAULTS.items():
        dbutils.widgets.text(k, v)
    P = {k: dbutils.widgets.get(k) for k in _DEFAULTS}
except Exception:
    P = dict(_DEFAULTS)

CATALOG, SCHEMA, VOLUME = P["catalog"], P["schema"], P["volume"]
APT_LAT, APT_LON, APT_ICAO = float(P["apt_lat"]), float(P["apt_lon"]), P["apt_icao"]
RADIUS_NM = float(P["radius_nm"])
N_DAYS = int(P["n_days"])
INTERVAL_S = max(30, int(P["interval_seconds"]))
MAX_FILES = int(P["max_files"])
SLEEP_S = float(P["request_sleep_s"])

if P["start_date"].strip():
    start_date = datetime.strptime(P["start_date"].strip(), "%Y-%m-%d").date()
else:
    start_date = (datetime.now(timezone.utc) - timedelta(days=N_DAYS)).date()
dates = [start_date + timedelta(days=i) for i in range(N_DAYS)]

hour_lo, hour_hi = 0, 23
if P["hours"].strip():
    hour_lo, hour_hi = (int(x) for x in P["hours"].split("-"))

print(f"{APT_ICAO}: keep aircraft within {RADIUS_NM} nm")
print(f"dates {dates[0]} .. {dates[-1]}  |  every {INTERVAL_S}s  |  hours {hour_lo}-{hour_hi} UTC")

# COMMAND ----------
# MAGIC %md ## Setup

# COMMAND ----------
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA  IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME  IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
OUT_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/backfill"
BASE = "https://samples.adsbexchange.com/readsb-hist"
HEADERS = {"User-Agent": "skywatch-portfolio/0.1 (+https://github.com/ChiragVenkateshaiah/skywatch)"}
print("landing at", OUT_ROOT)

EARTH_NM = 3440.065


def hav_nm(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return EARTH_NM * 2 * math.asin(math.sqrt(a))


def fetch_json(url, session):
    """readsb-hist is served as plain JSON despite the .gz name; handle either."""
    r = session.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200 or len(r.content) < 500:
        return None
    raw = r.content
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)

# COMMAND ----------
# MAGIC %md ## Backfill loop

# COMMAND ----------
written = skipped = missing = errors = 0
t_start = time.time()

with requests.Session() as s:
    for d in dates:
        yyyy, mm, dd = f"{d.year:04d}", f"{d.month:02d}", f"{d.day:02d}"
        day_url = f"{BASE}/{yyyy}/{mm}/{dd}"

        for secs in range(0, 86400, INTERVAL_S):
            if written >= MAX_FILES:
                print(f"hit max_files={MAX_FILES}; stopping")
                break
            hh, rem = divmod(secs, 3600)
            mn = rem // 60
            if not (hour_lo <= hh <= hour_hi):
                continue

            tag = f"{yyyy}{mm}{dd}_{hh:02d}{mn:02d}"
            out_dir = f"{OUT_ROOT}/dt={yyyy}-{mm}-{dd}/hh={hh:02d}"
            out_path = f"{out_dir}/{tag}.json"
            if os.path.exists(out_path):
                skipped += 1
                continue

            # readsb-hist is every ~5 s; try a few seconds around the target minute
            body = None
            for sec in (0, 5, 10, 15, 20, 25):
                body = fetch_json(f"{day_url}/{hh:02d}{mn:02d}{sec:02d}Z.json.gz", s)
                time.sleep(SLEEP_S)
                if body is not None:
                    break
            if body is None:
                missing += 1
                continue

            try:
                now_ms = int(float(body["now"]) * 1000)   # readsb-hist `now` is epoch SECONDS
                aircraft = body.get("aircraft") or body.get("ac") or []
                kept = [
                    a for a in aircraft
                    if a.get("lat") is not None and a.get("lon") is not None
                    and hav_nm(a["lat"], a["lon"], APT_LAT, APT_LON) <= RADIUS_NM
                ]
                out = {
                    "ac": kept,
                    "now": now_ms,
                    "total": len(kept),
                    "capture": {
                        "apt_icao": APT_ICAO, "apt_lat": APT_LAT, "apt_lon": APT_LON,
                        "radius_nm": RADIUS_NM, "source": "backfill",
                        "src_date": f"{yyyy}-{mm}-{dd}", "target_tag": tag,
                        "backfilled_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
                os.makedirs(out_dir, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(out, f, separators=(",", ":"))
                written += 1
            except Exception as e:  # noqa: BLE001
                errors += 1
                print("write error", tag, repr(e))

            if written and written % 100 == 0:
                rate = written / (time.time() - t_start)
                print(f"  {written} written  ({rate:.1f}/s)  skipped={skipped} missing={missing}")

        if written >= MAX_FILES:
            break

# COMMAND ----------
# MAGIC %md ## Summary

# COMMAND ----------
elapsed = time.time() - t_start
print(f"written={written}  skipped(existing)={skipped}  missing(no source file)={missing}  errors={errors}")
print(f"elapsed {elapsed/60:.1f} min")
if written == 0 and skipped == 0:
    raise RuntimeError(
        "Nothing written. Check the date range has data at "
        "https://samples.adsbexchange.com/readsb-hist/ and that serverless has outbound HTTP."
    )

# quick peek at what landed
peek = (
    spark.read.option("multiLine", "true").json(f"{OUT_ROOT}/**/*.json")
    .selectExpr("explode(ac) AS a", "capture.src_date AS src_date")
)
print(peek.count(), "aircraft rows across the backfill files")
display(
    peek.groupBy("src_date").count().orderBy("src_date")
)
