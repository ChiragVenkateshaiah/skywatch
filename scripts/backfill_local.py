#!/usr/bin/env python3
"""SkyWatch historical backfill — run OFF Databricks.

Downloads ADS-B Exchange `readsb-hist` global snapshots, keeps only aircraft within a radius of
the target airport, and writes each one wrapped in the **same envelope the live poller uses**
(`capture.source = "backfill"`) so it flows through the existing Bronze -> Silver -> Gold path.

Why run it here and not as a Databricks job: the raw files are ~6 MB global snapshots and only
~1 % survives the spatial filter. Downloading 10s of GB on Free Edition serverless would blow
the fair-use quota. This does the heavy download on any machine with internet, then you upload
only the ~1 % that matters:

    python scripts/backfill_local.py --out ./_backfill --months 3 --interval 60 --radius-nm 100
    databricks fs cp -r ./_backfill/backfill dbfs:/Volumes/skywatch/core/landing/backfill
    databricks bundle run skywatch_medallion -t dev --full-refresh-all
    databricks bundle run skywatch_gold -t dev

The archive only has the **1st of each month** (2023-01 .. present), each a full 24 h at ~5 s
cadence. `--months N` takes the N most recent; `--dates` overrides with an explicit list.
Resumable: one output file per target timestamp, existing files are skipped.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BASE = "https://samples.adsbexchange.com/readsb-hist"
UA = "skywatch-portfolio/0.1 (+https://github.com/ChiragVenkateshaiah/skywatch)"
EARTH_NM = 3440.065


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return EARTH_NM * 2 * math.asin(math.sqrt(a))


def fetch(url: str, timeout: int = 40) -> bytes | None:
    try:
        with urlopen(Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
            if r.status != 200:
                return None
            raw = r.read()
    except (HTTPError, URLError, TimeoutError, ConnectionError):
        return None
    if len(raw) < 500:
        return None
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw


def recent_first_of_month(n: int) -> list[date]:
    today = datetime.now(timezone.utc).date()
    y, m = today.year, today.month
    out = []
    for _ in range(n):
        out.append(date(y, m, 1))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return list(reversed(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--months", type=int, default=3, help="how many recent 1st-of-month days (default 3)")
    ap.add_argument("--dates", default="", help="explicit YYYY-MM-DD list (comma-separated); overrides --months")
    ap.add_argument("--interval", type=int, default=60, help="target seconds between snapshots (default 60)")
    ap.add_argument("--radius-nm", type=float, default=100.0, help="keep aircraft within this many nm (default 100)")
    ap.add_argument("--apt-lat", type=float, default=33.6407)
    ap.add_argument("--apt-lon", type=float, default=-84.4277)
    ap.add_argument("--apt-icao", default="KATL")
    ap.add_argument("--hours", default="", help='UTC hour range e.g. "10-04" (wraps); blank = all')
    ap.add_argument("--sleep", type=float, default=0.15, help="pause between requests (be polite)")
    args = ap.parse_args()

    if args.dates.strip():
        dates = [datetime.strptime(d.strip(), "%Y-%m-%d").date() for d in args.dates.split(",")]
    else:
        dates = recent_first_of_month(args.months)

    hour_ok = lambda h: True
    if args.hours.strip():
        lo, hi = (int(x) for x in args.hours.split("-"))
        hour_ok = (lambda h: lo <= h <= hi) if lo <= hi else (lambda h: h >= lo or h <= hi)

    root = args.out / "backfill"
    print(f"{args.apt_icao}: keep within {args.radius_nm} nm | dates {[d.isoformat() for d in dates]}")
    print(f"every {args.interval}s | hours {args.hours or 'all'} | -> {root}")

    written = skipped = missing = errors = 0
    t0 = time.time()

    for d in dates:
        ymd = f"{d.year:04d}/{d.month:02d}/{d.day:02d}"
        for secs in range(0, 86400, args.interval):
            hh, rem = divmod(secs, 3600)
            mn = rem // 60
            if not hour_ok(hh):
                continue
            tag = f"{d.year:04d}{d.month:02d}{d.day:02d}_{hh:02d}{mn:02d}"
            out_dir = root / f"dt={d.isoformat()}" / f"hh={hh:02d}"
            out_path = out_dir / f"{tag}.json"
            if out_path.exists():
                skipped += 1
                continue

            raw = None
            for sec in (0, 5, 10, 15, 20, 25):
                raw = fetch(f"{BASE}/{ymd}/{hh:02d}{mn:02d}{sec:02d}Z.json.gz")
                time.sleep(args.sleep)
                if raw is not None:
                    break
            if raw is None:
                missing += 1
                continue

            try:
                body = json.loads(raw)
                now_ms = int(float(body["now"]) * 1000)  # readsb-hist `now` is epoch SECONDS
                ac = body.get("aircraft") or body.get("ac") or []
                kept = [
                    a for a in ac
                    if a.get("lat") is not None and a.get("lon") is not None
                    and haversine_nm(a["lat"], a["lon"], args.apt_lat, args.apt_lon) <= args.radius_nm
                ]
                out = {
                    "ac": kept, "now": now_ms, "total": len(kept),
                    "capture": {
                        "apt_icao": args.apt_icao, "apt_lat": args.apt_lat, "apt_lon": args.apt_lon,
                        "radius_nm": args.radius_nm, "source": "backfill",
                        "src_date": d.isoformat(), "target_tag": tag,
                        "backfilled_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(out, separators=(",", ":")))
                written += 1
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"  ! {tag}: {e!r}", file=sys.stderr)

            if written and written % 200 == 0:
                el = time.time() - t0
                print(f"  {written} written  ({written/el:.1f}/s)  skipped={skipped} missing={missing} err={errors}")

    el = time.time() - t0
    print(f"\ndone: written={written} skipped={skipped} missing={missing} errors={errors}  ({el/60:.1f} min)")
    total = sum(1 for _ in root.rglob("*.json")) if root.exists() else 0
    size_mb = sum(p.stat().st_size for p in root.rglob("*.json")) / 1e6 if root.exists() else 0
    print(f"staged: {total} files, {size_mb:.0f} MB under {root}")
    return 0 if (written or skipped) else 1


if __name__ == "__main__":
    raise SystemExit(main())
