"""Tests for src/lib/grounding.py — the check shared by the briefing and the chat."""

from __future__ import annotations

from lib.grounding import check_text, harvest, harvest_prose


def test_sql_string_cells_are_harvested():
    a = harvest({"rows": [["113", "12.7", "DAL3100", "14:57", "2026-10-19T15:01:20.067Z", "B739"]]})
    assert {113.0, 12.7, 13.0, 2026.0, 10.0, 19.0} <= a.numbers
    assert {"14:57", "15:01"} <= a.times
    assert "DAL3100" in a.callsigns


def test_column_names_add_no_numbers_but_free_text_cells_do():
    a = harvest({"columns": ["within_10_min"], "rows": [["squawk=7700 emergency=general", "spread 0.98"]]})
    assert 10.0 not in a.numbers
    assert {7700.0, 0.98} <= a.numbers


def test_prose_values_are_harvested():
    a = harvest_prose(["Show me the next 5 arrivals after 14:30", "DAL3100 lands in 1.3 minutes."])
    assert {5.0, 1.3} <= a.numbers and "14:30" in a.times and "DAL3100" in a.callsigns
    assert 3100.0 not in a.numbers  # callsign digits are not numbers


def test_full_timestamps_are_judged_by_hh_mm():
    allowed = harvest({"t": "2026-10-12T15:51:28.200Z"})
    assert check_text("Audited 2026-10-12T15:51:28.200Z.", allowed, 50).passed
    assert check_text("At 15:51:28 UTC.", allowed, 50).passed
    assert check_text("At 16:02:00 UTC.", allowed, 50).reasons == ["time 16:02:00 not in facts"]


def test_check_text_names_its_source():
    r = check_text("114 inbound", harvest({"n": "113"}), max_words=50, source="tool results")
    assert r.reasons == ["number 114 not in tool results"]
