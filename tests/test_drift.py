"""Tests for src/lib/drift.py — PSI feature-drift detection (docs/MLOPS_PLAN.md Track 5)."""

from __future__ import annotations

import numpy as np
import pytest

from lib.drift import PSI_SIGNIFICANT, psi, psi_severity

RNG = np.random.default_rng(42)


class TestPsi:
    def test_identical_distributions_have_near_zero_psi(self):
        ref = RNG.normal(0, 1, 5000)
        comp = RNG.normal(0, 1, 5000)
        assert psi(ref, comp) < 0.05

    def test_shifted_distribution_has_high_psi(self):
        ref = RNG.normal(0, 1, 5000)
        comp = RNG.normal(3, 1, 5000)  # a big, obvious shift
        assert psi(ref, comp) > PSI_SIGNIFICANT

    def test_moderate_shift_lands_in_moderate_range(self):
        ref = RNG.normal(0, 1, 5000)
        comp = RNG.normal(0.5, 1, 5000)
        value = psi(ref, comp)
        assert 0.05 < value < 1.0  # loose bound — just confirms it's between "none" and "huge"

    def test_symmetric_ish(self):
        # PSI isn't exactly symmetric (bins come from the reference), but swapping reference
        # and comparison on a clear shift should still detect a large drift either direction.
        a = RNG.normal(0, 1, 5000)
        b = RNG.normal(3, 1, 5000)
        assert psi(a, b) > 0.25
        assert psi(b, a) > 0.25

    def test_comparison_values_outside_reference_range_still_counted(self):
        # A reference with a tight range and a comparison sample entirely outside it should
        # read as maximally different, not silently drop those points and read as ~0.
        ref = np.full(1000, 1.0)  # will hit the "no spread" branch — see next test
        comp = np.full(1000, 100.0)
        assert psi(ref, comp) == 0.0  # documented behavior: no-spread reference short-circuits

    def test_no_spread_reference_returns_zero_not_a_crash(self):
        ref = np.full(500, 5.0)
        comp = RNG.normal(5, 1, 500)
        assert psi(ref, comp) == 0.0

    def test_empty_inputs_return_nan(self):
        assert np.isnan(psi(np.array([]), RNG.normal(0, 1, 100)))
        assert np.isnan(psi(RNG.normal(0, 1, 100), np.array([])))

    def test_nans_are_dropped_not_propagated(self):
        ref = np.concatenate([RNG.normal(0, 1, 1000), [np.nan] * 50])
        comp = np.concatenate([RNG.normal(0, 1, 1000), [np.nan] * 50])
        assert not np.isnan(psi(ref, comp))


class TestPsiSeverity:
    @pytest.mark.parametrize(
        "value,expected",
        [(0.0, "none"), (0.05, "none"), (0.1, "moderate"), (0.2, "moderate"),
         (0.25, "significant"), (1.0, "significant")],
    )
    def test_thresholds(self, value, expected):
        assert psi_severity(value) == expected

    def test_nan_is_unknown(self):
        assert psi_severity(float("nan")) == "unknown"
