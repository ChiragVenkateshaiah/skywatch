# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — Model 1 Transformer variant: shared sequence builder
# MAGIC
# MAGIC `%run` this (it `%run`s `eta_features` itself) from `train_eta_transformer.py` and from
# MAGIC `promote_eta.py` when the model under evaluation is `model_flavor=pytorch_transformer` —
# MAGIC see `docs/M1_TRANSFORMER_PLAN.md`.
# MAGIC
# MAGIC The point-wise LightGBM model (`train_eta.py` / `score_eta.py`) calls `model.predict(X)`
# MAGIC with one flat row of `ETA_FEATURES` per prediction — no history. A sequence model needs
# MAGIC the last `SEQ_LEN` reports of the *same arrival*, not one row, but the gate and scoring
# MAGIC call sites only ever pass a flat DataFrame. So the window is built **upstream**, once,
# MAGIC here — one wide row per original point, columns = the flattened causal history — and
# MAGIC handed to `.predict()` exactly the way `ETA_FEATURES` already is for LightGBM. The
# MAGIC windowing math itself (`causal_windows` / `flatten_windows`) lives in `src/lib/sequences.py`
# MAGIC — pure numpy, unit-tested in `tests/test_sequences.py` — so this file is Spark/pandas glue
# MAGIC only: group by `seg_id`, order by `snapshot_ts`, call into `lib.sequences`.

# COMMAND ----------
# MAGIC %run ./eta_features

# COMMAND ----------
import numpy as np
import pandas as pd

from lib.sequences import SEQ_LEN, causal_windows, flatten_windows, phase_to_index

# Fixed per-timestep numeric feature order for the window — every continuous ETA_FEATURES
# column, then the phase category as an integer index (nn.Embedding input, not a one-hot / the
# categorical dtype LightGBM needs — see promote_eta.py's model_flavor branch for why the two
# models can't share a loader).
SEQ_FEATURE_COLUMNS = [c for c in ETA_FEATURES if c != "phase"] + ["phase_idx"]
N_SEQ_FEATURES = len(SEQ_FEATURE_COLUMNS)


def build_sequence_frame(sdf):
    """`sdf`: a Spark DataFrame already carrying `seg_id`, `snapshot_ts`, `ETA_TARGET`, and the
    raw columns `add_eta_features` needs (same shape `train_eta.py` / `promote_eta.py` query).

    Returns a pandas DataFrame, one row per input report, row order preserved, with:
    - `seg_id`, `obs_date`, `ETA_TARGET`, `dist_to_apt_nm`, `gs_kt` (`obs_date` for the same
      time-based train/val/test split `train_eta.py` uses; `dist_to_apt_nm` for the gate's
      by-band bucketing; `dist_to_apt_nm` + `gs_kt` together reproduce the same
      `dist_to_apt_nm / gs_kt * 60` baseline `train_eta.py` reports against)
    - `seq_0 .. seq_{SEQ_LEN*(N_SEQ_FEATURES+1)-1}`: the flattened causal window (mask included)
      — pass these columns straight into the Transformer pyfunc model's `.predict()`.
    """
    pdf = eta_pandas(
        sdf.select("seg_id", "snapshot_ts", ETA_TARGET, *ETA_FEATURES).toPandas()
    ).sort_values(["seg_id", "snapshot_ts"], kind="stable")
    pdf["obs_date"] = pd.to_datetime(pdf["snapshot_ts"]).dt.date

    pdf["phase_idx"] = phase_to_index(pdf["phase"].astype(str), PHASE_CATEGORIES)

    seq_cols = [f"seq_{i}" for i in range(SEQ_LEN * (N_SEQ_FEATURES + 1))]
    out_chunks = []
    for _seg_id, group in pdf.groupby("seg_id", sort=False):
        feats = group[SEQ_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        windows, mask = causal_windows(feats)
        flat = flatten_windows(windows, mask)
        chunk = group[["seg_id", "obs_date", ETA_TARGET, "dist_to_apt_nm", "gs_kt"]].copy()
        chunk[seq_cols] = flat
        out_chunks.append(chunk)

    return pd.concat(out_chunks, axis=0).reset_index(drop=True)
