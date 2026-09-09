"""In-process Model 1 load for the App's click-to-predict panel.

Free Edition has no model-serving endpoint, so the App loads `eta_touchdown@champion`
from the UC registry once at startup and calls it directly. Loading the **raw LightGBM
model** (`mlflow.lightgbm.load_model`) rather than the pyfunc wrapper keeps the `phase`
category dtype the model was trained on and gives access to `pred_contrib` (SHAP-style
per-feature contributions).
"""

from __future__ import annotations

import os

import pandas as pd

from data import ETA_FEATURES, PHASE_CATEGORIES

MODEL_NAME = os.environ.get("SKYWATCH_ETA_MODEL", "skywatch.ml.eta_touchdown")
MODEL_ALIAS = os.environ.get("SKYWATCH_ETA_ALIAS", "champion")


def load_eta_model():
    """Raw LightGBM model + its version string. Raises if the App SP lacks EXECUTE."""
    import mlflow

    mlflow.set_registry_uri("databricks-uc")
    uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    version = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    return mlflow.lightgbm.load_model(uri), str(version)


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
