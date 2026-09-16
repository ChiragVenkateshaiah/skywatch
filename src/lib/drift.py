"""Feature-drift math for src/monitor.py (docs/MLOPS_PLAN.md Track 5) — pure numpy, no Spark,
so it has a real pytest suite (tests/test_drift.py) instead of only ever being exercised by a
live job run. Lakehouse Monitoring is a paid feature; this is the DIY substitute.
"""

from __future__ import annotations

import numpy as np

# Standard PSI severity cutoffs (Population Stability Index — a common, simple drift metric).
PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25


def psi(reference: np.ndarray, comparison: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two 1-D numeric samples. Bins are the reference
    distribution's quantile edges (so the reference is ~uniform across bins by construction);
    the outer edges extend to +/-inf so comparison values outside the reference's range still
    land in a bin rather than being silently dropped. NaNs are dropped from both samples
    first. Returns NaN if either sample is empty after dropping NaNs.
    """
    ref = np.asarray(reference, dtype=float)
    comp = np.asarray(comparison, dtype=float)
    ref = ref[~np.isnan(ref)]
    comp = comp[~np.isnan(comp)]
    if len(ref) == 0 or len(comp) == 0:
        return float("nan")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return 0.0  # reference has no spread (all one value) — nothing meaningful to compare
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    comp_counts, _ = np.histogram(comp, bins=edges)
    # clip to avoid log(0) / div-by-0 on a bin either sample has zero mass in
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    comp_pct = np.clip(comp_counts / comp_counts.sum(), 1e-6, None)
    return float(np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct)))


def psi_severity(value: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value < PSI_MODERATE:
        return "none"
    if value < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"
