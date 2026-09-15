"""Tests for src/eta_features.py — the shared Model 1 feature module `%run` from both
train_eta.py and score_eta.py. This file is already plain-importable (no dbutils/spark
reference at module level), so it's tested in place rather than moved into src/lib/.
"""

from __future__ import annotations

import datetime
import math

import pandas as pd
import pytest

from eta_features import ETA_FEATURES, PHASE_CATEGORIES, add_eta_features, eta_pandas


def _base_row(**overrides):
    row = {
        "ac_type": "B738",
        "snapshot_ts": datetime.datetime(2026, 1, 1, 5, 0, 0),
        "bearing_to_apt": 0.0,
        "dist_to_apt_nm": 40.0,
        "heading_err_deg": 0.0,
        "alt_ft": 10000.0,
        "gs_kt": 300.0,
        "vrate_fpm": -500.0,
        "closure_kt": 250.0,
        "turn_rate_dps": 0.0,
        "sel_altitude_ft": 8000.0,
        "airport_inbound_count": 5,
        "phase": "descent",
    }
    row.update(overrides)
    return row


class TestAddEtaFeatures:
    def test_is_heavy_flag(self, spark):
        df = spark.createDataFrame([_base_row(ac_type="B772"), _base_row(ac_type="B738")])
        out = add_eta_features(df).select("ac_type", "is_heavy").collect()
        flags = {r["ac_type"]: r["is_heavy"] for r in out}
        assert flags["B772"] == 1
        assert flags["B738"] == 0

    @pytest.mark.parametrize(
        "heading_err_deg,gs_kt,expected",
        [(0.0, 300.0, 300.0), (90.0, 300.0, 0.0), (180.0, 300.0, -300.0)],
    )
    def test_closure_geom_kt_projects_groundspeed(self, spark, heading_err_deg, gs_kt, expected):
        df = spark.createDataFrame([_base_row(heading_err_deg=heading_err_deg, gs_kt=gs_kt)])
        v = add_eta_features(df).select("closure_geom_kt").first()["closure_geom_kt"]
        assert v == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        "bearing,exp_sin,exp_cos", [(0.0, 0.0, 1.0), (90.0, 1.0, 0.0), (180.0, 0.0, -1.0)]
    )
    def test_bearing_sin_cos(self, spark, bearing, exp_sin, exp_cos):
        df = spark.createDataFrame([_base_row(bearing_to_apt=bearing)])
        r = add_eta_features(df).select("bearing_sin", "bearing_cos").first()
        assert r["bearing_sin"] == pytest.approx(exp_sin, abs=1e-6)
        assert r["bearing_cos"] == pytest.approx(exp_cos, abs=1e-6)

    def test_hour_cyclical_encoding(self, spark):
        ts = datetime.datetime(2026, 1, 1, 5, 0, 0)  # hour = 5
        df = spark.createDataFrame([_base_row(snapshot_ts=ts)])
        r = add_eta_features(df).select("hour_sin", "hour_cos").first()
        expected_sin = math.sin(5 / 24 * 2 * math.pi)
        expected_cos = math.cos(5 / 24 * 2 * math.pi)
        assert r["hour_sin"] == pytest.approx(expected_sin, abs=1e-6)
        assert r["hour_cos"] == pytest.approx(expected_cos, abs=1e-6)

    def test_dow_cyclical_encoding_matches_spark_convention(self, spark):
        # Spark's dayofweek(): 1=Sunday .. 7=Saturday. Derive it from Python's isoweekday()
        # (1=Monday..7=Sunday) rather than hardcoding a calendar fact about this date.
        d = datetime.date(2026, 1, 1)
        spark_dow = (d.isoweekday() % 7) + 1
        ts = datetime.datetime(d.year, d.month, d.day, 12, 0, 0)
        df = spark.createDataFrame([_base_row(snapshot_ts=ts)])
        r = add_eta_features(df).select("dow_sin", "dow_cos").first()
        expected_sin = math.sin((spark_dow - 1) / 7 * 2 * math.pi)
        expected_cos = math.cos((spark_dow - 1) / 7 * 2 * math.pi)
        assert r["dow_sin"] == pytest.approx(expected_sin, abs=1e-6)
        assert r["dow_cos"] == pytest.approx(expected_cos, abs=1e-6)

    def test_output_has_all_eta_features_except_precomputed_inputs(self, spark):
        # add_eta_features derives is_heavy/hour_*/dow_*/bearing_*/closure_geom_kt; the rest
        # of ETA_FEATURES are expected to already be present on the input frame.
        df = spark.createDataFrame([_base_row()])
        out_cols = set(add_eta_features(df).columns)
        assert set(ETA_FEATURES) <= out_cols


class TestEtaPandas:
    def test_phase_is_pinned_categorical(self):
        pdf = pd.DataFrame({"phase": ["descent", "cruise"], "dist_to_apt_nm": [10.0, 50.0]})
        for col in ETA_FEATURES:
            if col not in pdf.columns and col != "phase":
                pdf[col] = 0.0
        out = eta_pandas(pdf)
        assert list(out["phase"].cat.categories) == PHASE_CATEGORIES

    def test_unseen_phase_value_becomes_nan_category(self):
        # A phase value outside PHASE_CATEGORIES silently becomes NaN, not an error — worth
        # pinning down explicitly since it's easy to trip over with a future new phase.
        pdf = pd.DataFrame({"phase": ["descent", "totally_unknown_phase"]})
        for col in ETA_FEATURES:
            if col not in pdf.columns and col != "phase":
                pdf[col] = 0.0
        out = eta_pandas(pdf)
        assert out["phase"].isna().tolist() == [False, True]

    def test_missing_phase_becomes_unknown(self):
        pdf = pd.DataFrame({"phase": [None, "cruise"]})
        for col in ETA_FEATURES:
            if col not in pdf.columns and col != "phase":
                pdf[col] = 0.0
        out = eta_pandas(pdf)
        assert out["phase"].tolist() == ["unknown", "cruise"]

    def test_decimal_columns_cast_to_float64(self):
        import decimal

        pdf = pd.DataFrame({"phase": ["cruise"], "dist_to_apt_nm": [decimal.Decimal("12.5")]})
        for col in ETA_FEATURES:
            if col not in pdf.columns and col != "phase":
                pdf[col] = decimal.Decimal("0")
        out = eta_pandas(pdf)
        assert out["dist_to_apt_nm"].dtype == "float64"
        assert out["dist_to_apt_nm"].iloc[0] == pytest.approx(12.5)
