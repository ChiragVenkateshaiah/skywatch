"""Ops-briefing model evaluation (docs/OPS_BRIEFING_PLAN.md §4) — run locally, off-platform.

Builds a fixed set of fact sheets — one per historical Model 1 scoring snapshot (paired with the
demand forecast and flags of the same moment) plus synthetic edge cases — and sends each through
every candidate Foundation Model API endpoint, `--samples` times. Reports per endpoint:
grounding-check pass rate, latency p50/p95 and completion tokens, and writes every call (facts
id, text, check reasons) to a JSONL file for manual review.

No new dependency: plain urllib, with host + token from the Databricks CLI profile.

    python scripts/eval_briefing.py --warehouse-id <id> [--profile skywatch] [--samples 3]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lib.briefing import build_facts, build_messages, check_briefing, extract_text  # noqa: E402

CANDIDATES = [
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-gpt-oss-120b",
    "databricks-gpt-oss-20b",
    "databricks-qwen3-next-80b-a3b-instruct",
    "databricks-llama-4-maverick",
    "databricks-gemma-3-12b",
]
APT, AAR = "KATL", 90
NUMERIC = {"predicted_eta_min", "horizon_step", "arrivals_so_far", "predicted_mean",
           "predicted_q90", "severity", "dist_to_apt_nm"}


class Workspace:
    def __init__(self, profile: str):
        env = json.loads(subprocess.check_output(["databricks", "auth", "env", "--profile", profile]))
        self.host = env.get("env", env)["DATABRICKS_HOST"].rstrip("/")
        self.profile = profile
        self._token, self._expiry = None, 0.0

    def token(self) -> str:
        if time.time() > self._expiry - 120:
            t = json.loads(subprocess.check_output(["databricks", "auth", "token", "--profile", self.profile]))
            self._token, self._expiry = t["access_token"], time.time() + int(t.get("expires_in", 3000))
        return self._token

    def post(self, path: str, body: dict, timeout: int = 90, retries: int = 5) -> dict:
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                self.host + path, data=json.dumps(body).encode(), method="POST",
                headers={"Authorization": f"Bearer {self.token()}", "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                # pay-per-token endpoints have a per-workspace QPS limit (REQUEST_LIMIT_EXCEEDED)
                if exc.code != 429 or attempt == retries:
                    raise
                time.sleep(2 ** attempt)

    def sql(self, warehouse_id: str, statement: str) -> pd.DataFrame:
        d = self.post("/api/2.0/sql/statements", {
            "warehouse_id": warehouse_id, "statement": statement, "wait_timeout": "50s",
            "catalog": "skywatch", "schema": "stream",
        })
        if d["status"]["state"] != "SUCCEEDED":
            raise RuntimeError(d["status"])
        cols = [c["name"] for c in d["manifest"]["schema"]["columns"]]
        df = pd.DataFrame(d.get("result", {}).get("data_array") or [], columns=cols)
        for c in df.columns:
            if c.endswith("_ts") or c == "scored_at":
                df[c] = pd.to_datetime(df[c], utc=True)
            elif c in NUMERIC:
                df[c] = pd.to_numeric(df[c])
        return df


# ------------------------------------------------------------------------- fact sheets
def historical_sheets(ws: Workspace, wh: str) -> dict[str, dict]:
    # as a string: scored_at has microseconds, the Statement API's JSON timestamps only ms,
    # so an equality filter built from the parsed value matches nothing
    runs = ws.sql(wh, "SELECT DISTINCT cast(scored_at AS string) AS t FROM predictions ORDER BY t")["t"]
    sheets = {}
    for t in runs:
        ts = pd.Timestamp(t, tz="UTC")
        preds = ws.sql(wh, f"SELECT callsign, icao, predicted_eta_min, scored_at FROM predictions "
                           f"WHERE scored_at = '{t}'")
        fc = ws.sql(wh, f"""SELECT bin_start_ts, horizon_step, arrivals_so_far, predicted_mean,
                            predicted_q90, scored_at FROM demand_forecast WHERE scored_at =
                            (SELECT max(scored_at) FROM demand_forecast WHERE scored_at <= '{t}' + INTERVAL 15 MINUTES)""")
        flags = ws.sql(wh, f"""SELECT kind, severity, callsign, icao, dist_to_apt_nm, scored_at
                               FROM irregularity_flags WHERE apt_icao = '{APT}' AND scored_at =
                               (SELECT max(scored_at) FROM irregularity_flags
                                WHERE apt_icao = '{APT}' AND scored_at <= '{t}' + INTERVAL 15 MINUTES)""")
        sheets[f"hist_{ts:%m%d_%H%M}"] = build_facts(
            preds, fc if not fc.empty else None, flags,
            apt_icao=APT, aar_per_hour=AAR, now=ts.to_pydatetime(),
        )
    return sheets


def _synthetic_frames(now, etas, means, uppers=None, flags=(), so_far=640):
    preds = pd.DataFrame({
        "callsign": [f"{a}{100 + 7 * i}" for i, a in zip(range(len(etas)), ["DAL", "SWA", "UAL", "AAL"] * 99)],
        "icao": [f"a{i:05x}" for i in range(len(etas))],
        "predicted_eta_min": etas, "scored_at": [now] * len(etas),
    })
    fc = None
    if means is not None:
        uppers = uppers or [m * 1.3 for m in means]
        fc = pd.DataFrame({
            "bin_start_ts": [now.replace(minute=0, second=0) + dt.timedelta(minutes=15 * (i + 1)) for i in range(len(means))],
            "horizon_step": range(len(means)), "arrivals_so_far": so_far,
            "predicted_mean": means, "predicted_q90": uppers, "scored_at": now,
        })
    fl = pd.DataFrame([(*r, now) for r in flags],
                      columns=["kind", "severity", "callsign", "icao", "dist_to_apt_nm", "scored_at"])
    return preds, fc, fl


def synthetic_sheets() -> dict[str, dict]:
    now = dt.datetime(2026, 10, 18, 17, 2, tzinfo=dt.timezone.utc)
    busy = [2.1, 3.4, 5.0, 6.2, 7.7, 9.3, 11.0, 12.5, 14.1, 16.8, 19.0, 22.4, 25.1, 28.3, 33.0, 38.5]
    cases = {
        "syn_surge": dict(etas=busy, means=[26.4, 25.1, 23.0, 21.7, 18.2, 16.0] * 2),
        "syn_near_cap": dict(etas=busy, means=[21.9, 20.4, 19.8, 17.5] * 3, uppers=[25.3, 24.0, 23.1, 20.0] * 3),
        "syn_emergency_holding": dict(etas=busy[:9], means=[14.0, 15.5, 13.2, 12.1] * 3, flags=[
            ("emergency", 3, "AAL1923", "c1", 41.6), ("holding", 2, "DAL2207", "c2", 22.9),
            ("holding", 2, "EDV5091", "c3", 25.4)]),
        "syn_go_around": dict(etas=busy[:6], means=[9.0, 8.4, 7.9, 6.0] * 3, flags=[
            ("go_around", 3, "SWA4410", "c4", 3.8)]),
        "syn_quiet_night": dict(etas=[12.4, 31.0], means=[1.2, 0.8, 0.4, 0.6] * 3, so_far=1012),
        "syn_empty_sky": dict(etas=[], means=[0.3, 0.2, 0.1, 0.2] * 3, so_far=1050),
        "syn_no_forecast": dict(etas=busy[:8], means=None),
    }
    out = {}
    for k, c in cases.items():
        preds, fc, fl = _synthetic_frames(now, **c)
        preds["predicted_eta_min"] = preds["predicted_eta_min"].astype(float)
        out[k] = build_facts(preds, fc, fl, apt_icao=APT, aar_per_hour=AAR, now=now)
    return out


# --------------------------------------------------------------------------------- run
def call(ws: Workspace, endpoint: str, facts: dict, temperature: float, max_tokens: int) -> dict:
    t0 = time.time()
    try:
        r = ws.post(f"/serving-endpoints/{endpoint}/invocations", {
            "messages": build_messages(facts), "temperature": temperature, "max_tokens": max_tokens,
        })
        text = extract_text(r)
        chk = check_briefing(text, facts)
        return {"ok": True, "text": text, "passed": chk.passed, "reasons": chk.reasons,
                "latency_s": time.time() - t0,
                "completion_tokens": (r.get("usage") or {}).get("completion_tokens")}
    except (urllib.error.URLError, KeyError, TimeoutError) as exc:
        detail = exc.read().decode()[:300] if isinstance(exc, urllib.error.HTTPError) else str(exc)
        return {"ok": False, "text": "", "passed": False, "reasons": [f"call failed: {detail}"],
                "latency_s": time.time() - t0, "completion_tokens": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warehouse-id", required=True)
    ap.add_argument("--profile", default="skywatch")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1024)  # reasoning models spend tokens before the answer
    ap.add_argument("--workers", type=int, default=2)  # 4 hit the workspace QPS limit
    ap.add_argument("--endpoints", nargs="*", default=CANDIDATES)
    ap.add_argument("--out", default="eval_briefing.jsonl")
    args = ap.parse_args()

    ws = Workspace(args.profile)
    sheets = {**historical_sheets(ws, args.warehouse_id), **synthetic_sheets()}
    print(f"{len(sheets)} fact sheets x {len(args.endpoints)} endpoints x {args.samples} samples")

    jobs = [(ep, sid, s) for ep in args.endpoints for sid in sheets for s in range(args.samples)]
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(
            lambda j: {"endpoint": j[0], "sheet": j[1], "sample": j[2],
                       **call(ws, j[0], sheets[j[1]], args.temperature, args.max_tokens)}, jobs))

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\n| endpoint | pass | call errors | latency p50 s | p95 s | completion tokens p50 |")
    print("|---|---|---|---|---|---|")
    for ep in args.endpoints:
        rs = [r for r in results if r["endpoint"] == ep]
        lat = sorted(r["latency_s"] for r in rs if r["ok"])
        tok = [r["completion_tokens"] for r in rs if r["completion_tokens"]]
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else float("nan")
        print(f"| `{ep}` | {sum(r['passed'] for r in rs)}/{len(rs)} | {sum(not r['ok'] for r in rs)} | "
              f"{statistics.median(lat) if lat else float('nan'):.1f} | {p95:.1f} | "
              f"{statistics.median(tok) if tok else '—'} |")
    print(f"\nall calls -> {args.out}")


if __name__ == "__main__":
    main()
