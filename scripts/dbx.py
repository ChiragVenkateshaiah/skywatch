"""Minimal Databricks REST client for the local scripts (eval_briefing.py, demo_ready.py).

No SDK dependency: urllib, the host from the CLI profile in ~/.databrickscfg, and a short-lived
token from `databricks auth token`. An expired CLI login surfaces as the CLI's own
`invalid_grant` error — fix with `databricks auth login --profile <profile>`.
"""

from __future__ import annotations

import configparser
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


class Workspace:
    def __init__(self, profile: str):
        cfg = configparser.ConfigParser()
        cfg.read(Path.home() / ".databrickscfg")
        if profile not in cfg or "host" not in cfg[profile]:
            raise SystemExit(f"profile [{profile}] with a host not found in ~/.databrickscfg")
        self.host = cfg[profile]["host"].rstrip("/")
        self.profile = profile
        self._token, self._expiry = None, 0.0

    def token(self) -> str:
        if time.time() > self._expiry - 120:
            try:
                out = subprocess.check_output(["databricks", "auth", "token", "--profile", self.profile],
                                              stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as exc:
                raise SystemExit(f"databricks auth token failed: {exc.stderr.decode().strip()[:300]}\n"
                                 f"-> run: databricks auth login --profile {self.profile}") from None
            t = json.loads(out)
            self._token, self._expiry = t["access_token"], time.time() + int(t.get("expires_in", 3000))
        return self._token

    def _call(self, method: str, path: str, body: dict | None, timeout: int, retries: int) -> dict:
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                self.host + path, method=method,
                data=json.dumps(body).encode() if body is not None else None,
                headers={"Authorization": f"Bearer {self.token()}", "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read() or b"{}")
            except urllib.error.HTTPError as exc:
                # pay-per-token endpoints have a per-workspace QPS limit (REQUEST_LIMIT_EXCEEDED)
                if exc.code != 429 or attempt == retries:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def post(self, path: str, body: dict, timeout: int = 90, retries: int = 5) -> dict:
        return self._call("POST", path, body, timeout, retries)

    def get(self, path: str, params: dict | None = None, timeout: int = 60, retries: int = 3) -> dict:
        q = f"?{urllib.parse.urlencode(params)}" if params else ""
        return self._call("GET", path + q, None, timeout, retries)

    def sql(self, warehouse_id: str, statement: str, numeric: set[str] = frozenset()) -> pd.DataFrame:
        """Statement Execution API -> DataFrame. `*_ts` / `scored_at` parsed as UTC timestamps,
        columns in `numeric` as numbers, the rest left as strings."""
        d = self.post("/api/2.0/sql/statements", {
            "warehouse_id": warehouse_id, "statement": statement, "wait_timeout": "50s",
            "catalog": "skywatch", "schema": "stream",
        })
        if d["status"]["state"] != "SUCCEEDED":
            raise RuntimeError(d["status"])
        cols = [c["name"] for c in d["manifest"]["schema"]["columns"]]
        df = pd.DataFrame(d.get("result", {}).get("data_array") or [], columns=cols)
        for c in df.columns:
            if c.endswith("_ts") or c in ("scored_at", "checked_at", "generated_at", "data_as_of"):
                df[c] = pd.to_datetime(df[c], utc=True)
            elif c in numeric:
                df[c] = pd.to_numeric(df[c])
        return df
