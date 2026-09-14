"""In-process Model 1 load for the App's click-to-predict panel.

Free Edition has no model-serving endpoint, so click-to-predict needs the model loaded
directly into the app process. That normally means granting the app's service principal
`EXECUTE` on the UC registered model — but that grant **isn't available on this
metastore** (`GRANT EXECUTE ON MODEL` / the Grants API both fail with
`REGISTERED_MODEL is not enabled`, confirmed against the real workspace).

So instead: `score_eta.py` exports the exact `eta_touchdown@champion` object to a UC
**Volume** on every scoring run (a plain, always-grantable securable), and this module
reads it back with `joblib` — no MLflow registry access needed at all, only
`READ VOLUME` on `skywatch.ml.models`.

**Databricks Apps don't FUSE-mount Volumes as local paths** (`/Volumes/...` only works
on cluster / job / notebook compute) — the same reason `data.py` reads Delta tables
through the SQL Statement Execution API instead of a local Spark session. So this reads
the Volume file the same way: the **Files REST API** (`WorkspaceClient().files`), not
`open()`.
"""

from __future__ import annotations

import io
import json
import os

import pandas as pd
from databricks.sdk import WorkspaceClient

from data import ETA_FEATURES, PHASE_CATEGORIES

CATALOG = os.environ.get("SKYWATCH_CATALOG", "skywatch")
VOL_DIR = f"/Volumes/{CATALOG}/ml/models"
MODEL_PATH = f"{VOL_DIR}/eta_touchdown_champion.joblib"
META_PATH = f"{VOL_DIR}/eta_touchdown_champion.meta.json"

_w = WorkspaceClient()


def load_eta_model():
    """The exported champion model + its version string. Raises if the App SP lacks
    `READ VOLUME` on `skywatch.ml.models`, or score_eta.py hasn't run yet."""
    import joblib

    model_bytes = _w.files.download(MODEL_PATH).contents.read()
    model = joblib.load(io.BytesIO(model_bytes))

    version = "?"
    try:
        meta_bytes = _w.files.download(META_PATH).contents.read()
        version = str(json.loads(meta_bytes).get("model_version", "?"))
    except Exception:  # noqa: BLE001 — the version string is cosmetic, never block on it
        pass
    return model, version


def _frame(row: dict) -> pd.DataFrame:
    """One-row feature frame in ETA_FEATURES order, with `phase` as the trained Categorical."""
    x = pd.DataFrame([{f: row.get(f) for f in ETA_FEATURES}])
    for c in ETA_FEATURES:
        if c != "phase":
            x[c] = pd.to_numeric(x[c], errors="coerce").astype("float64")
    x["phase"] = pd.Categorical(x["phase"].fillna("unknown"), categories=PHASE_CATEGORIES)
    return x[ETA_FEATURES]


def predict_eta(model, row: dict) -> float:
    return float(max(0.0, model.predict(_frame(row))[0]))


def contributions(model, row: dict) -> pd.Series:
    """Per-feature contribution to the prediction (LightGBM raw SHAP), minutes.
    Last element LightGBM returns is the base value — drop it."""
    raw = model.predict(_frame(row), pred_contrib=True)[0]
    return pd.Series(raw[:-1], index=ETA_FEATURES).sort_values(key=abs, ascending=False)
