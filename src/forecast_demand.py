# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — Model 2: arrival demand forecast
# MAGIC
# MAGIC Within-day forecast: given a day's arrival counts in 15-minute bins `0..T`, predict bins
# MAGIC `T+1..T+H` (H = 12 = next 3 h). Series = `skywatch.stream.gold_demand_15m`.
# MAGIC
# MAGIC **The archive only has the 1st of each month**, so there is no day-to-day continuity —
# MAGIC each day is an independent 96-bin series. With only a handful of days the strong baseline
# MAGIC is the **climatological mean profile**; a fine-tuned foundation model needs ≥ ~14 days to
# MAGIC earn its complexity (roadmap §7). This notebook builds the full harness now so that is a
# MAGIC one-parameter change later.
# MAGIC
# MAGIC - **Backtest:** leave-one-day-out × several cut points → many windows from few days.
# MAGIC - **Baselines:** climatological mean (+ day-of-week), seasonal-naive, blended, context-mean,
# MAGIC   `statsforecast` AutoETS / AutoARIMA.
# MAGIC - **Foundation model:** Chronos-Bolt zero-shot; fine-tune behind `run_finetune`.
# MAGIC - **Metrics:** MAE, MASE (vs seasonal-naive; < 1 = beats it), weighted quantile loss.
# MAGIC - Registers `skywatch.ml.demand_forecast` — the backtest winner, wrapped for serving.

# COMMAND ----------
# MAGIC %pip install -q chronos-forecasting statsforecast scipy
# MAGIC %restart_python

# COMMAND ----------
# MAGIC %run ./demand_lib

# COMMAND ----------
try:
    dbutils.widgets.text("stream_schema", "skywatch.stream")
    dbutils.widgets.text("model_name", "skywatch.ml.demand_forecast")
    dbutils.widgets.text("horizon_bins", "12")
    dbutils.widgets.text("cut_bins", "32,44,56,68,80")
    dbutils.widgets.text("run_finetune", "false")
    STREAM = dbutils.widgets.get("stream_schema")
    MODEL_NAME = dbutils.widgets.get("model_name")
    H = int(dbutils.widgets.get("horizon_bins"))
    CUTS = [int(x) for x in dbutils.widgets.get("cut_bins").split(",")]
    RUN_FINETUNE = dbutils.widgets.get("run_finetune").lower() == "true"
except Exception:
    STREAM, MODEL_NAME, H, CUTS, RUN_FINETUNE = (
        "skywatch.stream", "skywatch.ml.demand_forecast", 12, [32, 44, 56, 68, 80], False,
    )
print(f"series {STREAM}.gold_demand_15m | horizon {H} bins | cuts {CUTS} | finetune {RUN_FINETUNE}")

# COMMAND ----------
# MAGIC %md ## 1. Load the demand series

# COMMAND ----------
import numpy as np
import pandas as pd

demand = spark.table(f"{STREAM}.gold_demand_15m").toPandas()
demand["arrivals"] = demand["arrivals"].astype(float)
frames = to_day_frames(demand)
days = sorted(frames)
print(f"{len(days)} days: {[str(d) for d in days]}")
display(demand.groupby("bin_date").agg(bins=("arrivals", "size"), total=("arrivals", "sum"),
                                       peak=("arrivals", "max")))
if len(days) < 2:
    raise RuntimeError("need at least 2 days to leave one out — run more backfill first")

# COMMAND ----------
# MAGIC %md ## 2. Backtest — leave-one-day-out × cut points

# COMMAND ----------
records, curves = [], []


def eval_window(name, actual, point, naive, qa=None):
    rec = {"model": name, "mae": mae(actual, point), "mase": mase(actual, point, naive)}
    if qa is not None:
        rec["wql"] = wql(actual, qa)
    records.append(rec)


for d in days:
    ref = [x for x in days if x != d]
    y = frames[d].set_index("slot")["arrivals"].to_numpy(float)
    dow = int(frames[d]["dow"].iloc[0])
    for T in CUTS:
        if T + H > BINS_PER_DAY:
            continue
        slots = list(range(T, T + H))
        actual, ctx = y[slots], y[:T]
        naive = seasonal_naive(frames, ref, slots)

        eval_window("climatological_mean", actual, climatological_mean(frames, ref, slots), naive)
        eval_window("climatological_dow", actual,
                    climatological_mean(frames, ref, slots, dow=dow), naive)
        eval_window("seasonal_naive", actual, naive, naive)
        eval_window("blended", actual, blended(frames, ref, ctx, slots, dow=dow), naive)
        eval_window("context_mean", actual, context_mean(ctx, slots), naive)
        for n, p in statsforecast_forecast(ctx, H).items():
            eval_window(n, actual, p, naive)

        med, qa = chronos_forecast(ctx, H)
        eval_window("chronos_bolt_zeroshot", actual, med, naive, qa=qa)
        curves.append({"day": str(d), "cut": T, "slot": slots, "actual": actual.tolist(),
                       "clim": climatological_mean(frames, ref, slots, dow=dow).tolist(),
                       "chronos": med.tolist()})

bt = pd.DataFrame(records)
summary = (bt.groupby("model")
           .agg(mae=("mae", "mean"), mae_sd=("mae", "std"),
                mase=("mase", "mean"), wql=("wql", "mean"), n=("mae", "size"))
           .sort_values("mae").round(3))
print("=== backtest mean over", len(days) * len(CUTS), "windows ===")
display(summary.reset_index())
print("MASE < 1 beats seasonal-naive.")

# COMMAND ----------
# MAGIC %md ## 3. Chronos-Bolt fine-tune (optional — `run_finetune`)
# MAGIC Needs GPU to be quick and ≥ ~14 backfill days to beat the climatology baseline. Uses
# MAGIC AutoGluon-TS, which wraps the Chronos fine-tuning loop.

# COMMAND ----------
finetune_summary = None
if RUN_FINETUNE:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "autogluon.timeseries"], check=True)
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    long = demand.assign(item_id=demand["bin_date"].astype(str),
                         timestamp=pd.to_datetime(demand["bin_start_ts"]),
                         target=demand["arrivals"])[["item_id", "timestamp", "target"]]
    tsdf = TimeSeriesDataFrame.from_data_frame(long)
    predictor = TimeSeriesPredictor(prediction_length=H, target="target", eval_metric="MASE").fit(
        tsdf, hyperparameters={"Chronos": [
            {"model_path": "bolt_small", "fine_tune": False, "ag_args": {"name_suffix": "ZeroShot"}},
            {"model_path": "bolt_small", "fine_tune": True, "fine_tune_steps": 500,
             "ag_args": {"name_suffix": "FineTuned"}}]},
        time_limit=1200, enable_ensemble=False)
    finetune_summary = predictor.leaderboard(tsdf)
    display(finetune_summary)

# COMMAND ----------
# MAGIC %md ## 4. Pick the champion + register

# COMMAND ----------
import mlflow
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")

CANDIDATES = {"climatological_dow", "climatological_mean", "blended", "chronos_bolt_zeroshot"}
ranked = summary[summary.index.isin(CANDIDATES)].sort_values("mae")
champ_name = ranked.index[0]
print(f"champion: {champ_name}  (MAE {ranked.iloc[0]['mae']:.2f}, MASE {ranked.iloc[0]['mase']:.2f})")

# per-(dow, slot) profile for the climatology wrapper
prof = {}
for d in days:
    fr, dw = frames[d], int(frames[d]["dow"].iloc[0])
    for s, a in zip(fr["slot"], fr["arrivals"]):
        prof.setdefault((dw, s), []).append(float(a))
        prof.setdefault(("*", s), []).append(float(a))
prof = {k: float(np.nanmean(v)) for k, v in prof.items()}

mode = "chronos" if champ_name == "chronos_bolt_zeroshot" else "climatology"
wrapper = DemandForecastModel(mode=mode, horizon=H, profile=prof,
                              chronos_model_id="amazon/chronos-bolt-small")

example = pd.DataFrame([{"context": [10.0] * 40, "dow": int(frames[days[-1]]["dow"].iloc[0]),
                         "horizon": H}])
pred_example = wrapper.predict(None, example)

with mlflow.start_run(run_name="demand_forecast") as run:
    mlflow.log_params({"champion": champ_name, "mode": mode, "horizon_bins": H,
                       "n_days": len(days), "n_windows": len(days) * len(CUTS)})
    for m_, row in summary.iterrows():
        mlflow.log_metric(f"bt_mae__{m_}", row["mae"])
        if not np.isnan(row["mase"]):
            mlflow.log_metric(f"bt_mase__{m_}", row["mase"])
    mlflow.log_metric("champion_bt_mae", float(ranked.iloc[0]["mae"]))
    mlflow.log_metric("champion_bt_mase", float(ranked.iloc[0]["mase"]))
    mlflow.log_dict({"summary": summary.reset_index().to_dict("records"),
                     "curves": curves}, "backtest.json")

    sig = infer_signature(example, pred_example)
    mlflow.pyfunc.log_model("model", python_model=wrapper, signature=sig,
                            input_example=example,
                            pip_requirements=["chronos-forecasting", "scipy", "pandas", "numpy"])
    model_uri = f"runs:/{run.info.run_id}/model"

from mlflow import MlflowClient

client = MlflowClient()
mv = mlflow.register_model(model_uri, MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
client.set_model_version_tag(MODEL_NAME, mv.version, "champion_method", champ_name)
client.set_model_version_tag(MODEL_NAME, mv.version, "bt_mase", f"{ranked.iloc[0]['mase']:.3f}")

# promote if it beats seasonal-naive over the backtest
if ranked.iloc[0]["mase"] < 1.0:
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    print(f"registered {MODEL_NAME} v{mv.version} @challenger + @champion "
          f"({champ_name}, MASE {ranked.iloc[0]['mase']:.2f} < 1)")
else:
    print(f"registered {MODEL_NAME} v{mv.version} @challenger only "
          f"(MASE {ranked.iloc[0]['mase']:.2f} — did not beat seasonal-naive)")
