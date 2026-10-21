"""Tests for src/lib/eval_cache.py — the quota guards of the golden-set eval (roadmap OPT-4).
Nothing touches the network."""

from __future__ import annotations

import threading
import time

import pytest

from lib.eval_cache import ToolCache, fingerprint, load_stamp, save_stamp, should_skip, tool_key


def test_fingerprint_changes_with_file_and_extra(tmp_path):
    a = tmp_path / "chat.py"
    a.write_text("PROMPT = 'v1'")
    fp1 = fingerprint([a], ["maverick"])
    assert fp1 == fingerprint([a], ["maverick"])
    assert fp1 != fingerprint([a], ["llama"])
    a.write_text("PROMPT = 'v2'")
    assert fp1 != fingerprint([a], ["maverick"])


def test_should_skip_only_full_unchanged_unforced_runs(tmp_path):
    stamp_file = tmp_path / "last_run.json"
    assert load_stamp(stamp_file) is None
    save_stamp(stamp_file, "abc", "| pass | 38/38 |")
    stamp = load_stamp(stamp_file)
    assert stamp["summary"] == "| pass | 38/38 |"
    assert should_skip(stamp, "abc", force=False, partial=False)
    assert not should_skip(stamp, "abd", force=False, partial=False)     # something changed
    assert not should_skip(stamp, "abc", force=True, partial=False)
    assert not should_skip(stamp, "abc", force=False, partial=True)      # --only
    assert not should_skip(None, "abc", force=False, partial=False)      # first run


def test_tool_key_ignores_arg_order():
    assert tool_key("t", {"a": 1, "b": 2}) == tool_key("t", {"b": 2, "a": 1})
    assert tool_key("t", {"n": 5}) != tool_key("t", {"n": 6})


def test_cache_hits_and_persists_per_version(tmp_path):
    path = tmp_path / "tools.json"
    calls = []
    c = ToolCache(path, "2026-10-19T15:01", "h1")
    for _ in range(3):
        assert c.get("next_arrivals", {"n": 5}, lambda: calls.append(1) or {"rows": [["DAL1"]]}) == {"rows": [["DAL1"]]}
    assert (len(calls), c.hits, c.misses, c.reused) == (1, 2, 1, False)
    c.save()

    same = ToolCache(path, "2026-10-19T15:01", "h1")
    assert same.reused
    assert same.get("next_arrivals", {"n": 5}, lambda: pytest.fail("should come from the cache"))
    assert not ToolCache(path, "2026-10-21T06:00", "h1").reused             # new data
    assert not ToolCache(path, "2026-10-19T15:01", "h2").reused             # tool SQL changed
    assert not ToolCache(path, "2026-10-19T15:01", "h1", reuse=False).reused  # --fresh-tools


def test_errors_are_not_cached(tmp_path):
    c = ToolCache(None, "v", "h")

    def boom():
        raise RuntimeError("warehouse down")
    with pytest.raises(RuntimeError):
        c.get("t", {}, boom)
    assert c.get("t", {}, lambda: "ok") == "ok"
    assert c.misses == 1
    c.save()   # no path: a no-op


def test_concurrent_callers_run_the_tool_once():
    c, calls = ToolCache(None, "v", "h"), []

    def slow():
        calls.append(1)
        time.sleep(0.05)
        return 42
    threads = [threading.Thread(target=lambda: c.get("t", {}, slow)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert (len(calls), c.hits) == (1, 3)
