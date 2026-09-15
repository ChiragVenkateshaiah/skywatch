"""Tests for src/demand_lib.py — the shared Model 2 library (`%run` from forecast_demand.py
and score_demand.py). Already plain-importable (numpy/pandas only; mlflow/chronos/
statsforecast are all optional, guarded imports) so it's tested in place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand_lib import (
    BINS_PER_DAY,
    QUANTILES,
    blended,
    climatological_mean,
    context_mean,
    mae,
    mase,
    seasonal_naive,
    to_day_frames,
    wql,
)


class TestMetrics:
    def test_mae_known_values(self):
        assert mae([1, 2, 3], [1, 2, 4]) == pytest.approx(1 / 3)

    def test_mae_zero_for_perfect_prediction(self):
        assert mae([1, 2, 3], [1, 2, 3]) == 0.0

    def test_mase_ratio_against_naive(self):
        actual, pred, naive = [1, 2, 3], [1, 2, 4], [0, 0, 0]
        expected = mae(actual, pred) / mae(actual, naive)
        assert mase(actual, pred, naive) == pytest.approx(expected)

    def test_mase_nan_when_naive_is_perfect(self):
        # naive_pred == actual -> mae(actual, naive) ~ 0 -> division guarded to NaN, not inf.
        actual = [1, 2, 3]
        assert np.isnan(mase(actual, [1, 2, 4], actual))

    def test_wql_zero_for_perfect_quantile_forecast(self):
        actual = np.array([10.0])
        quantile_pred = np.tile(10.0, (1, len(QUANTILES)))
        assert wql(actual, quantile_pred) == pytest.approx(0.0, abs=1e-9)

    def test_wql_known_uniform_underforecast(self):
        # Every quantile column predicts a constant 8 against actual=10 (e=2>0), so
        # max(q*e, (q-1)*e) == q*e for every quantile in (0,1) -> hand-derivable.
        actual = np.array([10.0])
        quantile_pred = np.tile(8.0, (1, len(QUANTILES)))
        expected = 2 * (2 * np.mean(QUANTILES)) / 10.0
        assert wql(actual, quantile_pred) == pytest.approx(expected, abs=1e-9)


def _day(slot_values: dict[int, float], dow: int) -> pd.DataFrame:
    arrivals = np.full(BINS_PER_DAY, np.nan)
    for slot, v in slot_values.items():
        arrivals[slot] = v
    return pd.DataFrame({"slot": range(BINS_PER_DAY), "arrivals": arrivals, "dow": dow})


class TestBaselines:
    def test_climatological_mean_averages_ref_days(self):
        day_frames = {
            "d1": _day({0: 2.0, 1: 4.0}, dow=1),
            "d2": _day({0: 6.0, 1: 8.0}, dow=2),
        }
        out = climatological_mean(day_frames, ["d1", "d2"], [0, 1])
        assert out.tolist() == pytest.approx([4.0, 6.0])

    def test_climatological_mean_filters_by_dow_when_available(self):
        day_frames = {
            "d1": _day({0: 2.0}, dow=1),
            "d2": _day({0: 6.0}, dow=1),
            "d3": _day({0: 100.0}, dow=9),  # different dow, should be excluded
        }
        out = climatological_mean(day_frames, ["d1", "d2", "d3"], [0], dow=1)
        assert out.tolist() == pytest.approx([4.0])

    def test_climatological_mean_falls_back_when_no_dow_match(self):
        day_frames = {"d1": _day({0: 2.0}, dow=1), "d2": _day({0: 6.0}, dow=1)}
        out = climatological_mean(day_frames, ["d1", "d2"], [0], dow=99)
        assert out.tolist() == pytest.approx([4.0])

    def test_seasonal_naive_uses_most_recent_ref_day(self):
        day_frames = {"d1": _day({0: 1.0}, dow=1), "d2": _day({0: 9.0}, dow=1)}
        out = seasonal_naive(day_frames, ["d1", "d2"], [0])
        assert out.tolist() == pytest.approx([9.0])

    def test_context_mean_uses_tail_only(self):
        context = [1.0, 1.0, 1.0, 5.0, 5.0]
        out = context_mean(context, slots=[0, 1], tail=2)
        assert out.tolist() == pytest.approx([5.0, 5.0])

    def test_blended_matches_climatology_far_out_in_horizon(self):
        # The blend weight is indexed by *position in the requested slots list*, not by slot
        # number, and is only nonzero for the first 3 positions — so with 4 slots requested,
        # position 3 (the 4th) gets weight 0 and should equal plain climatology exactly.
        day_frames = {"d1": _day({0: 2.0, 1: 2.0, 2: 2.0, 3: 8.0}, dow=1)}
        slots = [0, 1, 2, 3]
        clim = climatological_mean(day_frames, ["d1"], slots)
        out = blended(day_frames, ["d1"], context=[2.0, 2.0], slots=slots)
        assert out[3] == pytest.approx(clim[3])
        # positions 0-2 DO get pulled toward the context mean, so they need not match clim.
        assert out[3] == pytest.approx(8.0)


class TestToDayFrames:
    def test_reshapes_into_96_bin_slots(self):
        pdf = pd.DataFrame(
            {
                "bin_start_ts": pd.to_datetime(["2026-01-01T00:00:00", "2026-01-01T00:15:00"]),
                "hour_utc": [0, 0],
                "bin_date": ["2026-01-01", "2026-01-01"],
                "arrivals": [2, 5],
                "dow": [5, 5],
            }
        )
        out = to_day_frames(pdf)
        assert set(out.keys()) == {"2026-01-01"}
        day = out["2026-01-01"]
        assert len(day) == BINS_PER_DAY
        assert day.loc[day["slot"] == 0, "arrivals"].iloc[0] == 2
        assert day.loc[day["slot"] == 1, "arrivals"].iloc[0] == 5
        assert day["dow"].iloc[0] == 5
        # unfilled slots are explicit NaN, not dropped
        assert day.loc[day["slot"] == 50, "arrivals"].isna().all()
