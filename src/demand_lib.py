# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — Model 2 shared forecasting library
# MAGIC
# MAGIC `%run` this from `forecast_demand.py` (backtest + train) and, later, `score_demand.py`
# MAGIC (serving) so the two can never drift.
# MAGIC
# MAGIC **Task:** within-day arrival-demand forecast. The `readsb-hist` archive only has the 1st
# MAGIC of each month, so there is no continuous timeline — each day is an independent 96-bin
# MAGIC (15-minute) series. Given bins `0..T` of a day, forecast bins `T+1..T+H`.
# MAGIC
# MAGIC Near-term demand (0–45 min) comes from aggregating Model 1's live ETAs; Model 2 owns the
# MAGIC 45 min – 3 h horizon.

# COMMAND ----------
import numpy as np
import pandas as pd

BINS_PER_DAY = 96                      # 24 h at 15 min
DEFAULT_H = 12                         # forecast horizon in bins (3 h)
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def to_day_frames(pdf):
    """gold_demand_15m rows -> {date: DataFrame with slot 0..95, arrivals, dow}."""
    pdf = pdf.sort_values("bin_start_ts").copy()
    pdf["slot"] = pdf["hour_utc"].astype(int) * 4 + (pd.to_datetime(pdf["bin_start_ts"]).dt.minute // 15)
    out = {}
    for d, g in pdf.groupby("bin_date"):
        s = (g.set_index("slot")["arrivals"].reindex(range(BINS_PER_DAY)).astype(float))
        out[d] = pd.DataFrame({"slot": range(BINS_PER_DAY), "arrivals": s.values,
                               "dow": int(g["dow"].iloc[0])})
    return out


# --------------------------------------------------------------------- baselines
def climatological_mean(day_frames, ref_days, slots, dow=None):
    """Mean arrivals at each (hour,quarter) slot across `ref_days`. If `dow` is given and any
    ref day matches it, average only those (falls back to all when none match)."""
    use = [d for d in ref_days if dow is not None and day_frames[d]["dow"].iloc[0] == dow]
    use = use or list(ref_days)
    stack = np.vstack([day_frames[d].set_index("slot")["arrivals"].values for d in use])
    return np.nanmean(stack, axis=0)[list(slots)]


def seasonal_naive(day_frames, ref_days, slots):
    """Same (hour,quarter) slot from the most recent reference day."""
    return day_frames[ref_days[-1]].set_index("slot")["arrivals"].reindex(slots).to_numpy(float)


def context_mean(context, slots, tail=8):
    return np.full(len(slots), np.nanmean(context[-tail:]))


def blended(day_frames, ref_days, context, slots, dow=None):
    """Climatology, but the first two horizon bins nudged toward the context tail (the arrivals
    already in the air dominate the very short horizon)."""
    clim = climatological_mean(day_frames, ref_days, slots, dow)
    ctxm = np.nanmean(context[-4:]) if len(context) else clim[0]
    w = np.array([0.6, 0.35, 0.15] + [0.0] * (len(slots) - 3))[: len(slots)]
    return (1 - w) * clim + w * ctxm


# --------------------------------------------------------------------- statsforecast
def statsforecast_forecast(context, h):
    """AutoETS / AutoARIMA on the partial-day context. Returns {name: array(h)} (may be empty)."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS

    sdf = pd.DataFrame({"unique_id": "d",
                        "ds": pd.date_range("2020-01-01", periods=len(context), freq="15min"),
                        "y": np.asarray(context, float)})
    out = {}
    for name, model in [("AutoETS", AutoETS(season_length=1)), ("AutoARIMA", AutoARIMA(season_length=1))]:
        try:
            f = StatsForecast(models=[model], freq="15min", n_jobs=1).forecast(df=sdf, h=h)
            out[name] = f.iloc[:, -1].to_numpy(float).clip(min=0)
        except Exception:  # noqa: BLE001
            pass
    return out


# --------------------------------------------------------------------- Chronos
_CHRONOS = {}


def chronos_pipeline(model_id="amazon/chronos-bolt-small", device="cpu"):
    import torch
    from chronos import BaseChronosPipeline

    key = (model_id, device)
    if key not in _CHRONOS:
        _CHRONOS[key] = BaseChronosPipeline.from_pretrained(
            model_id, device_map=device, torch_dtype=torch.float32)
    return _CHRONOS[key]


def chronos_forecast(context, h, model_id="amazon/chronos-bolt-small", device="cpu"):
    """Returns (median array(h), quantiles array(h, len(QUANTILES)))."""
    import torch

    q, _ = chronos_pipeline(model_id, device).predict_quantiles(
        torch.tensor(np.asarray(context, float), dtype=torch.float32),
        prediction_length=h, quantile_levels=QUANTILES)
    qa = q[0].cpu().numpy().clip(min=0)
    return qa[:, QUANTILES.index(0.5)], qa


# --------------------------------------------------------------------- metrics
def mae(actual, pred):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred))))


def mase(actual, pred, naive_pred):
    d = mae(actual, naive_pred)
    return mae(actual, pred) / d if d > 1e-9 else np.nan


def wql(actual, quantile_pred):
    """Weighted quantile loss (a.k.a. weighted pinball), normalised by total actual."""
    actual = np.asarray(actual, float)
    total = max(actual.sum(), 1.0)
    per_q = []
    for i, ql in enumerate(QUANTILES):
        e = actual - quantile_pred[:, i]
        per_q.append(np.sum(np.maximum(ql * e, (ql - 1) * e)))
    return float(2 * np.mean(per_q) / total)


# --------------------------------------------------------------------- pyfunc wrapper
try:
    import mlflow

    class DemandForecastModel(mlflow.pyfunc.PythonModel):
        """Serves the winning approach. `predict()` input: a DataFrame with one row per
        forecast request — columns `context` (list[float], arrivals in bins 0..T of the day),
        `dow` (int), `horizon` (int, <= trained H). Output: columns q10..q90, plus `mean`.

        `climatology` mode carries the per-(dow, slot) profile; `chronos` mode carries a
        model id and calls the pipeline. The profile is small enough to pickle."""

        def __init__(self, mode, horizon, profile=None, chronos_model_id=None):
            self.mode = mode
            self.horizon = horizon
            self.profile = profile                # dict[(dow, slot)] -> mean, for climatology
            self.chronos_model_id = chronos_model_id

        def _one(self, context, dow, horizon):
            start = len(context)
            slots = [(start + i) % BINS_PER_DAY for i in range(horizon)]
            if self.mode == "chronos":
                _, qa = chronos_forecast(context, horizon, self.chronos_model_id)
                return qa
            # climatology: point profile + a Poisson-ish spread for the quantiles
            mu = np.array([self.profile.get((dow, s), self.profile.get(("*", s), 0.0))
                           for s in slots])
            sd = np.sqrt(np.maximum(mu, 1.0))
            from scipy.stats import norm
            return np.clip(np.stack([mu + norm.ppf(q) * sd for q in QUANTILES], axis=1), 0, None)

        def predict(self, context, model_input):
            rows = []
            for _, r in model_input.iterrows():
                h = int(r.get("horizon", self.horizon))
                qa = self._one(list(r["context"]), int(r["dow"]), h)
                rec = {f"q{int(q*100)}": qa[:, i].tolist() for i, q in enumerate(QUANTILES)}
                rec["mean"] = qa[:, QUANTILES.index(0.5)].tolist()
                rows.append(rec)
            return pd.DataFrame(rows)
except ImportError:
    pass
