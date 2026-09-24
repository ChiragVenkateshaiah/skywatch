"""Causal sequence-windowing for the Model 1 Transformer variant (docs/M1_TRANSFORMER_PLAN.md)
— pure numpy, no Spark, so it has a real pytest suite (tests/test_sequences.py) instead of only
ever being exercised by a live job run. `src/eta_sequences.py` is the Spark glue that groups
`gold_arrival_tracks` by `seg_id` and calls into this module; keeping the windowing math here
means the same array-in/array-out functions are used by training, the promotion gate, and (if
this variant ever needs it) live scoring — no train/serve drift, same pattern as
`src/lib/promotion.py` and `src/lib/drift.py`.
"""

from __future__ import annotations

import numpy as np

SEQ_LEN = 24  # covers the mean (~19) and the large majority of the 25,084 arrival segments;
              # longer segments are truncated to their most recent SEQ_LEN reports — the near
              # history matters most for a short-horizon ETA, and this matches what live scoring
              # would actually have (it never sees an aircraft's full trajectory either).


def causal_windows(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One arrival's ordered feature matrix -> a causal window ending at every row.

    `features`: (T, F) array, one row per report, already ordered by `snapshot_ts`.

    Returns `(windows, mask)`:
    - `windows`: (T, SEQ_LEN, F) float array. `windows[i]` holds report `i`'s trailing history
      — reports `max(0, i-SEQ_LEN+1)..i` — left-padded with zeros to a fixed length. Row `i`
      never contains any report `j > i`: this mirrors live scoring, which only ever has an
      aircraft's past, never its future.
    - `mask`: (T, SEQ_LEN) bool array, `True` where `windows` holds a real (non-padding) report.
      The most recent report in every window is always at index `SEQ_LEN - 1` (real, never
      padding) — a fixed "current timestep" position a model can always read from.
    """
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError(f"expected a 2-D (T, F) array, got shape {feats.shape}")
    t, f = feats.shape

    windows = np.zeros((t, SEQ_LEN, f), dtype=np.float32)
    mask = np.zeros((t, SEQ_LEN), dtype=bool)
    for i in range(t):
        start = max(0, i - SEQ_LEN + 1)
        span = i - start + 1  # number of real reports in this window, always <= SEQ_LEN
        windows[i, SEQ_LEN - span:, :] = feats[start : i + 1]
        mask[i, SEQ_LEN - span:] = True
    return windows, mask


def flatten_windows(windows: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(T, SEQ_LEN, F) windows + (T, SEQ_LEN) mask -> one flat (T, SEQ_LEN*(F+1)) array, mask
    appended as an extra per-timestep column. This is the "wide" row format that travels through
    a plain pandas DataFrame (and an MLflow pyfunc signature) the same way `ETA_FEATURES` does
    for the LightGBM model — one row per prediction, no external state needed at predict time.
    """
    t, seq_len, f = windows.shape
    with_mask = np.concatenate([windows, mask.astype(np.float32)[:, :, None]], axis=2)
    return with_mask.reshape(t, seq_len * (f + 1))


def unflatten_windows(flat: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `flatten_windows` — used inside the pyfunc wrapper's `predict()` to rebuild
    `(windows, mask)` from the flat DataFrame columns MLflow hands it.
    """
    flat = np.asarray(flat, dtype=np.float32)
    n, width = flat.shape
    seq_len = width // (n_features + 1)
    reshaped = flat.reshape(n, seq_len, n_features + 1)
    return reshaped[:, :, :n_features], reshaped[:, :, n_features].astype(bool)


def phase_to_index(phases, categories: list[str]) -> np.ndarray:
    """Map `phase` strings to a fixed integer index (for `nn.Embedding`), unknown/missing values
    to `categories`'s "unknown" slot. `categories` must be `eta_features.PHASE_CATEGORIES` so
    training and scoring never disagree on the mapping.
    """
    lookup = {c: i for i, c in enumerate(categories)}
    unknown_idx = lookup.get("unknown", 0)
    return np.array([lookup.get(p, unknown_idx) for p in phases], dtype=np.int64)
