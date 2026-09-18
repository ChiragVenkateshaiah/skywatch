"""Tests for src/lib/promotion.py — the promotion-gate decision (docs/MLOPS_PLAN.md Track 4).
Pure Python, no Spark/mlflow needed — the notebook (promote_eta.py) handles scoring the
actual models and only calls evaluate_gate() with the resulting per-band MAE dicts.
"""

from __future__ import annotations

import datetime as dt

import pytest

from lib.promotion import GateResult, audit_row, evaluate_gate

BANDS = ["0-20nm", "20-40nm", "40-70nm", "70-100nm", "ALL"]


def _bands(all_mae: float, **overrides: float) -> dict[str, float]:
    d = {b: 1.0 for b in BANDS}
    d["ALL"] = all_mae
    d.update(overrides)
    return d


class TestBootstrap:
    def test_passes_when_no_champion_exists(self):
        result = evaluate_gate(_bands(1.0), None, 2.0, 1.0, 0.15)
        assert result.passed
        assert "bootstrap" in result.reasons[0]


class TestOverallMetric:
    def test_passes_on_clear_improvement(self):
        # 10% better clears the 2% bar easily.
        result = evaluate_gate(_bands(0.9), _bands(1.0), 2.0, 1.0, 0.15)
        assert result.passed

    def test_passes_within_noise_band_even_without_clearing_the_bar(self):
        # 0.5% worse is within a 1% noise band — not a "clear win" but not a regression either.
        result = evaluate_gate(_bands(1.005), _bands(1.0), 2.0, 1.0, 0.15)
        assert result.passed

    def test_fails_when_worse_than_noise_band_and_short_of_improvement_bar(self):
        # 5% worse is well outside a 1% noise band.
        result = evaluate_gate(_bands(1.05), _bands(1.0), 2.0, 1.0, 0.15)
        assert not result.passed
        assert any("FAILS" in r for r in result.reasons)

    def test_boundary_improvement_pct_exactly_at_bar_passes(self):
        # Exactly 2% better should pass (>=, not >).
        result = evaluate_gate(_bands(0.98), _bands(1.0), 2.0, 1.0, 0.15)
        assert result.passed

    def test_just_inside_noise_band_passes(self):
        # 0.9% worse, comfortably inside a 1% noise band (avoids asserting on an exact
        # floating-point boundary, which 100 * (1.0 - 1.01) / 1.0 != -1.0 exactly would be).
        result = evaluate_gate(_bands(1.009), _bands(1.0), 2.0, 1.0, 0.15)
        assert result.passed

    def test_just_outside_noise_band_fails(self):
        # 1.1% worse, just past a 1% noise band.
        result = evaluate_gate(_bands(1.011), _bands(1.0), 2.0, 1.0, 0.15)
        assert not result.passed


class TestBandRegression:
    def test_fails_on_band_regression_even_if_overall_improves(self):
        # Overall a big win (10% better), but 0-20nm (the band that matters most for
        # sequencing) regressed by 0.3 min — a real regression a single overall number hides.
        challenger = _bands(0.9, **{"0-20nm": 1.3})
        champion = _bands(1.0, **{"0-20nm": 1.0})
        result = evaluate_gate(challenger, champion, 2.0, 1.0, 0.15)
        assert not result.passed
        assert any("band regression" in r for r in result.reasons)

    def test_band_improvement_never_blocks(self):
        challenger = _bands(0.9, **{"0-20nm": 0.5})  # much better, not worse
        champion = _bands(1.0, **{"0-20nm": 1.0})
        result = evaluate_gate(challenger, champion, 2.0, 1.0, 0.15)
        assert result.passed

    def test_band_regression_within_tolerance_passes(self):
        # 0.1 min regression, tolerance is 0.15 — should not trip the gate.
        challenger = _bands(0.9, **{"0-20nm": 1.1})
        champion = _bands(1.0, **{"0-20nm": 1.0})
        result = evaluate_gate(challenger, champion, 2.0, 1.0, 0.15)
        assert result.passed

    @pytest.mark.parametrize("missing_band", ["20-40nm", "40-70nm", "70-100nm"])
    def test_tolerates_a_band_missing_from_one_side(self, missing_band):
        # A band with zero holdout rows in it might be absent from one dict — must not crash.
        challenger = _bands(0.9)
        champion = _bands(1.0)
        del champion[missing_band]
        result = evaluate_gate(challenger, champion, 2.0, 1.0, 0.15)
        assert result.passed


class TestAuditRow:
    def test_pass_and_applied(self):
        row = audit_row("skywatch.ml.eta_touchdown", 9, 7, GateResult(True, ["ok"]), applied=True)
        assert row["model_name"] == "skywatch.ml.eta_touchdown"
        assert row["challenger_version"] == 9
        assert row["champion_version_before"] == 7
        assert row["gate_verdict"] == "PASS"
        assert row["applied"] is True
        assert row["reasons"] == "ok"
        assert isinstance(row["audited_at"], dt.datetime)
        assert row["audited_at"].tzinfo is not None

    def test_pass_but_not_applied_is_a_valid_combination(self):
        # A dry run: the gate passed, but apply=false, so nothing was actually promoted.
        row = audit_row("m", 2, 1, GateResult(True, ["ok"]), applied=False)
        assert row["gate_verdict"] == "PASS"
        assert row["applied"] is False

    def test_hold_forces_applied_false_in_practice_but_the_function_itself_is_honest(self):
        # audit_row records whatever `applied` it's given — the notebooks are responsible for
        # never passing applied=True alongside a failing gate. Verified here so a future
        # refactor that breaks that invariant fails a test, not silently corrupts the audit log.
        row = audit_row("m", 2, 1, GateResult(False, ["FAILS: worse"]), applied=False)
        assert row["gate_verdict"] == "HOLD"
        assert row["applied"] is False
        assert "FAILS" in row["reasons"]

    def test_bootstrap_has_no_prior_champion(self):
        row = audit_row("m", 1, None, GateResult(True, ["no @champion exists yet"]), applied=True)
        assert row["champion_version_before"] is None

    def test_multiple_reasons_joined(self):
        row = audit_row("m", 2, 1, GateResult(False, ["reason one", "reason two"]), applied=False)
        assert row["reasons"] == "reason one | reason two"
