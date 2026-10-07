"""Tests for src/lib/sequences.py — causal windowing for the M1 Transformer variant
(docs/M1_TRANSFORMER_PLAN.md)."""

from __future__ import annotations

import numpy as np
import pytest

from lib.sequences import (
    SEQ_LEN,
    causal_windows,
    flatten_windows,
    phase_to_index,
    unflatten_windows,
)


class TestCausalWindows:
    def test_shape(self):
        feats = np.arange(10 * 3, dtype=float).reshape(10, 3)
        windows, mask = causal_windows(feats)
        assert windows.shape == (10, SEQ_LEN, 3)
        assert mask.shape == (10, SEQ_LEN)

    def test_short_segment_is_right_aligned_and_padded(self):
        # a 3-report arrival: every window's real reports sit at the end, zero-padded before it.
        feats = np.array([[1.0], [2.0], [3.0]])
        windows, mask = causal_windows(feats)
        # row 0: only report 0 is real, at the last position.
        assert mask[0].sum() == 1
        assert windows[0, -1, 0] == 1.0
        assert np.all(windows[0, :-1] == 0.0)
        # row 2 (the last report): all 3 real reports present, most recent last.
        assert mask[2].sum() == 3
        np.testing.assert_array_equal(windows[2, -3:, 0], [1.0, 2.0, 3.0])

    def test_current_report_always_at_last_index(self):
        feats = np.arange(40 * 2, dtype=float).reshape(40, 2)
        windows, mask = causal_windows(feats)
        for i in range(40):
            assert mask[i, -1]  # always real — "current timestep" is a fixed position
            np.testing.assert_array_equal(windows[i, -1], feats[i])

    def test_long_segment_truncates_to_seq_len(self):
        feats = np.arange(40, dtype=float).reshape(40, 1)
        windows, mask = causal_windows(feats)
        assert mask[-1].sum() == SEQ_LEN
        # the last window holds exactly the most recent SEQ_LEN reports, oldest first.
        expected = feats[40 - SEQ_LEN : 40, 0]
        np.testing.assert_array_equal(windows[-1, :, 0], expected)

    def test_no_future_leakage(self):
        # every value in windows[i] must come from reports <= i.
        feats = np.arange(15, dtype=float).reshape(15, 1)
        windows, mask = causal_windows(feats)
        for i in range(15):
            real_values = windows[i][mask[i]]
            assert np.all(real_values <= i)

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError):
            causal_windows(np.arange(10))


class TestFlattenRoundTrip:
    def test_flatten_unflatten_is_lossless(self):
        feats = np.arange(8 * 4, dtype=float).reshape(8, 4)
        windows, mask = causal_windows(feats)
        flat = flatten_windows(windows, mask)
        assert flat.shape == (8, SEQ_LEN * (4 + 1))

        out_windows, out_mask = unflatten_windows(flat, n_features=4)
        np.testing.assert_allclose(out_windows, windows)
        np.testing.assert_array_equal(out_mask, mask)


class TestPhaseToIndex:
    CATEGORIES = ["ground", "climb", "descent", "cruise", "level", "unknown"]

    def test_known_phases_map_to_their_position(self):
        idx = phase_to_index(["ground", "descent", "cruise"], self.CATEGORIES)
        np.testing.assert_array_equal(idx, [0, 2, 3])

    def test_unknown_or_missing_maps_to_unknown_slot(self):
        idx = phase_to_index(["bogus", None, "unknown"], self.CATEGORIES)
        unknown_idx = self.CATEGORIES.index("unknown")
        assert (idx == unknown_idx).all()
