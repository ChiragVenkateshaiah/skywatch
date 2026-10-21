"""Quota-aware evaluation helpers for scripts/eval_chat.py (roadmap OPT-4).

The golden set was the biggest single compute cost on Free Edition (~40 tool SQL calls + ~60
model calls per run, 15+ runs on 2026-10-19, then the warehouse quota stopped work). Two guards:

  * `fingerprint` — a hash of everything that can change a golden-set result (the turn loop, the
    grounding check, the tool SQL, the golden set, the eval script, the endpoint). If it matches
    the last full run's stamp, the eval prints that run's result and stops before any warehouse or
    model call. Data changes alone don't re-run it: the set measures the prompt, tools and loop.
  * `ToolCache` — tool results kept on disk per data version (the live picture's `scored_at`)
    and tool-SQL hash. While the data doesn't move (schedules paused, a demo snapshot), a re-run
    pays only for model calls. Errors are never cached.

Pure: no network, no Databricks imports — unit-tested in tests/test_eval_cache.py.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable


def fingerprint(paths: list[Path], extra: list[str] = ()) -> str:
    """sha256 over the files' bytes (in order, path names included) and the extra strings."""
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.name).encode() + b"\0" + Path(p).read_bytes() + b"\0")
    for e in extra:
        h.update(str(e).encode() + b"\0")
    return h.hexdigest()[:16]


def load_stamp(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_stamp(path: Path, fp: str, summary: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"fingerprint": fp, "summary": summary}, indent=1))


def should_skip(stamp: dict | None, fp: str, *, force: bool, partial: bool) -> bool:
    """Skip a full run whose inputs match the last full run. `--only` runs are never stamped
    and never skipped (they are the cheap way to check one item)."""
    return bool(stamp) and stamp.get("fingerprint") == fp and not force and not partial


def tool_key(name: str, args: dict) -> str:
    return f"{name}|{json.dumps(args, sort_keys=True)}"


class ToolCache:
    """Tool results keyed by (name, args), valid for one (data_version, tools_hash) pair.
    Thread-safe; a second caller for the same key waits for the first instead of re-running it."""

    def __init__(self, path: Path | None, data_version: str, tools_hash: str, reuse: bool = True):
        self.path, self.version = path, {"data_version": data_version, "tools_hash": tools_hash}
        self.results: dict[str, Any] = {}
        self.hits = self.misses = 0
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        saved = load_stamp(path) if path and reuse else None
        self.reused = bool(saved) and all(saved.get(k) == v for k, v in self.version.items())
        if self.reused:
            self.results = saved.get("results", {})

    def get(self, name: str, args: dict, call: Callable[[], Any]) -> Any:
        key = tool_key(name, args)
        with self._lock:
            lock = self._key_locks.setdefault(key, threading.Lock())
        with lock:
            with self._lock:
                if key in self.results:
                    self.hits += 1
                    return self.results[key]
            result = call()           # raises → not cached
            with self._lock:
                self.misses += 1
                self.results[key] = result
            return result

    def save(self) -> None:
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.path).write_text(json.dumps({**self.version, "results": self.results}))
