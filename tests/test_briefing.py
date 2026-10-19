"""Tests for src/lib/briefing.py — the ops-briefing facts, grounding check and template fallback
(docs/OPS_BRIEFING_PLAN.md). Pure Python + pandas; the notebook (brief_ops.py) does the Spark
reads and the Foundation Model API call.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from lib.briefing import (
    build_facts,
    build_messages,
    check_briefing,
    extract_text,
    prompt_facts,
    template_briefing,
)

NOW = dt.datetime(2026, 10, 18, 14, 0, tzinfo=dt.timezone.utc)


def _preds(etas, scored_at=NOW - dt.timedelta(minutes=5)):
    return pd.DataFrame({
        "callsign": [f"DAL{100 + i}" for i in range(len(etas))],
        "icao": [f"a{i:05x}" for i in range(len(etas))],
        "predicted_eta_min": etas,
        "scored_at": [scored_at] * len(etas),
    })


def _fc(means, uppers=None, so_far=212, scored_at=NOW - dt.timedelta(minutes=6)):
    uppers = uppers or [m + 3 for m in means]
    start = dt.datetime(2026, 10, 18, 14, 0, tzinfo=dt.timezone.utc)
    return pd.DataFrame({
        "bin_start_ts": [start + dt.timedelta(minutes=15 * i) for i in range(len(means))],
        "horizon_step": list(range(len(means))),
        "arrivals_so_far": [so_far] * len(means),
        "predicted_mean": means,
        "predicted_q90": uppers,
        "scored_at": [scored_at] * len(means),
    })


def _flags(rows=(), scored_at=NOW - dt.timedelta(minutes=4)):
    cols = ["kind", "severity", "callsign", "icao", "dist_to_apt_nm", "scored_at"]
    return pd.DataFrame([(*r, scored_at) for r in rows], columns=cols)


_DEFAULT = object()


def _facts(preds=_DEFAULT, fc=_DEFAULT, flags=_DEFAULT, **kw):
    """Unset = a normal frame; an explicit `None` = that source unavailable."""
    return build_facts(
        _preds([2.6, 8.0, 14.2, 27.5, 41.0]) if preds is _DEFAULT else preds,
        _fc([18.4, 20.1, 16.0, 15.2] * 3) if fc is _DEFAULT else fc,
        _flags() if flags is _DEFAULT else flags,
        apt_icao="KATL", aar_per_hour=90, now=NOW, **kw,
    )


# ----------------------------------------------------------------------------- facts
class TestBuildFacts:
    def test_inbound_counts_and_next(self):
        f = _facts()
        assert f["inbound"]["count"] == 5
        assert f["inbound"]["within_10_min"] == 2
        assert f["inbound"]["within_30_min"] == 4
        assert f["inbound"]["next"][0] == {"callsign": "DAL100", "eta_min": 2.6}
        assert len(f["inbound"]["next"]) == 3

    def test_demand_peak_and_next_hour(self):
        f = _facts()
        d = f["demand"]
        assert d["predicted_next_hour"] == round(18.4 + 20.1 + 16.0 + 15.2)
        assert d["peak_15min_start_utc"] == "14:15"
        assert d["peak_15min_predicted"] == 20.1
        assert d["peak_15min_headroom"] == round(22.5 - 20.1, 1)
        assert d["horizon_hours"] == 3
        assert d["arrivals_seen_by_skywatch_today"] == 212

    def test_surge_bins_counted_against_aar_per_bin(self):
        f = _facts(fc=_fc([24.0, 23.0, 20.0, 10.0], uppers=[27, 26, 23, 12]))
        assert f["demand"]["bins_over_aar"] == 2      # 24 and 23 > 22.5
        assert f["demand"]["bins_upper_over_aar"] == 3

    def test_fresh_sources_are_not_stale(self):
        f = _facts()
        assert f["stale"] is False
        assert f["max_age_min"] == 6
        # 5 vs 6 min: an empty flags frame has no scored_at, so it adds no age of its own
        assert f["source_skew_min"] == 1

    def test_old_and_skewed_sources_are_flagged(self):
        f = _facts(fc=_fc([5.0] * 12, scored_at=NOW - dt.timedelta(days=4)))
        assert f["stale"] is True
        assert f["source_skew_min"] == 4 * 24 * 60 - 5

    def test_missing_sources_become_null(self):
        f = build_facts(None, None, None, apt_icao="KATL", aar_per_hour=90, now=NOW)
        assert f["inbound"] is None and f["demand"] is None and f["irregularities"] is None
        assert f["stale"] is True

    def test_empty_flags_means_zero_counts_not_unavailable(self):
        assert _facts()["irregularities"] == {"emergency": 0, "go_around": 0, "holding": 0, "flagged": []}

    def test_flags_ordered_by_severity_then_distance(self):
        f = _facts(flags=_flags([
            ("holding", 2, "UAL22", "b1", 31.4),
            ("emergency", 3, "AAL7", "b2", 48.0),
            ("holding", 2, "SWA9", "b3", 12.2),
        ]))
        assert [x["callsign"] for x in f["irregularities"]["flagged"]] == ["AAL7", "SWA9", "UAL22"]
        assert f["irregularities"]["holding"] == 2

    def test_flags_from_an_older_run_than_predictions_are_dropped(self):
        old = _flags([("holding", 2, "SWA9", "b3", 12.2)], scored_at=NOW - dt.timedelta(days=2))
        f = _facts(flags=old)
        assert f["irregularities"] == {"emergency": 0, "go_around": 0, "holding": 0, "flagged": []}

    def test_flags_from_the_same_cycle_are_kept(self):
        f = _facts(flags=_flags([("holding", 2, "SWA9", "b3", 12.2)]))
        assert f["irregularities"]["holding"] == 1

    def test_blank_callsign_falls_back_to_icao(self):
        p = _preds([3.0])
        p.loc[0, "callsign"] = "  "
        assert _facts(preds=p)["inbound"]["next"][0]["callsign"] == "a00000"


class TestPrompt:
    def test_prompt_leaves_out_data_age(self):
        pf = prompt_facts(_facts())
        assert "data" not in pf and "max_age_min" not in pf and "source_skew_min" not in pf
        assert "stale" in pf

    def test_messages_carry_the_facts_json(self):
        msgs = build_messages(_facts())
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert '"predicted_next_hour": 70' in msgs[1]["content"]


class TestExtractText:
    def test_plain_string_content(self):
        assert extract_text({"choices": [{"message": {"content": "  hi  "}}]}) == "hi"

    def test_reasoning_parts_are_dropped(self):
        resp = {"choices": [{"message": {"content": [
            {"type": "reasoning", "summary": [{"text": "think"}]},
            {"type": "text", "text": "the answer"},
        ]}}]}
        assert extract_text(resp) == "the answer"


# ----------------------------------------------------------------------------- check
class TestCheck:
    def test_numbers_from_facts_pass(self):
        r = check_briefing("70 arrivals are predicted next hour; the peak at 14:15 UTC is 20.1. "
                           "DAL100 lands in 2.6 minutes.", _facts())
        assert r.passed, r.reasons

    def test_rounded_fact_value_passes(self):
        assert check_briefing("About 20 aircraft at the peak.", _facts()).passed

    def test_invented_number_fails(self):
        r = check_briefing("73 arrivals are predicted next hour.", _facts())
        assert not r.passed and r.reasons == ["number 73 not in facts"]

    def test_computed_difference_fails(self):
        # 90 - 70 = 20 would happen to pass; 70 - 15.2 = 54.8 is a recomputation
        assert not check_briefing("54.8 below the last bin.", _facts()).passed

    def test_unknown_time_fails(self):
        assert check_briefing("The peak is at 15:45 UTC.", _facts()).reasons == ["time 15:45 not in facts"]

    def test_unknown_callsign_fails(self):
        assert check_briefing("UAL900 is holding.", _facts()).reasons == ["callsign UAL900 not in facts"]

    def test_callsign_digits_are_not_checked_as_numbers(self):
        assert check_briefing("DAL101 is next.", _facts()).passed

    def test_data_age_is_not_allowed(self):
        # max_age_min = 6 is not in the prompt facts, so the model can't state it
        assert not check_briefing("Data is 6 minutes old.", _facts()).passed

    def test_leaked_key_name_fails(self):
        r = check_briefing("The peak_15min_upper is 23.1.", _facts())
        assert r.reasons == ["key name peak_15min_upper in text"]

    def test_too_long_fails(self):
        assert not check_briefing("word " * 130, _facts()).passed

    def test_empty_fails(self):
        assert check_briefing("   ", _facts()).reasons == ["empty text"]


# -------------------------------------------------------------------------- template
SCENARIOS = {
    "normal": {},
    "surge": {"fc": _fc([24.0, 23.0, 20.0, 10.0] * 3, uppers=[27, 26, 23, 12] * 3)},
    "upper_only": {"fc": _fc([20.0] * 12, uppers=[24.0] * 12)},
    "emergency_and_holding": {"flags": _flags([
        ("emergency", 3, "AAL7", "b2", 48.0), ("holding", 2, "SWA9", "b3", 12.2),
        ("go_around", 3, "DAL55", "b4", 4.0),
    ])},
    "empty_sky": {"preds": _preds([]).astype({"predicted_eta_min": float})},
    "missing_forecast": {"fc": _fc([1.0]).iloc[0:0]},
    "missing_predictions": {"preds": None},
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_template_always_passes_its_own_check(name):
    facts = _facts(**SCENARIOS[name])
    text = template_briefing(facts)
    r = check_briefing(text, facts)
    assert r.passed, (text, r.reasons)


def test_template_with_every_source_missing_passes():
    facts = build_facts(None, None, None, apt_icao="KATL", aar_per_hour=90, now=NOW)
    text = template_briefing(facts)
    assert "unavailable" in text
    assert check_briefing(text, facts).passed


def test_template_leads_with_emergency_then_surge():
    facts = _facts(flags=_flags([("emergency", 3, "AAL7", "b2", 48.0)]),
                   fc=_fc([24.0] * 12))
    text = template_briefing(facts)
    assert text.startswith("Emergency: AAL7")
    assert text.index("Surge:") < text.index("inbound")


def test_template_empty_sky_wording():
    text = template_briefing(_facts(**SCENARIOS["empty_sky"]))
    assert "No aircraft are inbound to KATL." in text
