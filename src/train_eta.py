# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — Model 1: time-to-touchdown (ETA)
# MAGIC
# MAGIC Regression: given one ADS-B report of an aircraft inbound to KATL, predict the minutes
# MAGIC until it touches down. Training set = `skywatch.stream.gold_arrival_tracks` (one row per
# MAGIC pre-touchdown report of a detected landing, labelled `minutes_to_touchdown`).
# MAGIC
# MAGIC - **Split is time-based**: train on the earlier backfill days, test on the latest — never
# MAGIC   split a trajectory across folds (every report of one arrival shares its conditions).
# MAGIC - **Baseline to beat**: `dist_to_apt_nm / gs_kt * 60` (fly straight in at current speed).
# MAGIC - **Primary metric**: MAE in minutes, **reported by distance band** — accuracy inside
# MAGIC   100 nm is what matters for arrival sequencing.
# MAGIC - Tuning: Hyperopt over LightGBM, every trial an MLflow run. AutoML behind a flag
# MAGIC   (`run_automl`) — it trains many models and is heavier on the Free Edition quota.
# MAGIC - Registered to Unity Catalog as `skywatch.ml.eta_touchdown`, alias `@challenger`
# MAGIC   (promoted to `@champion` when it beats the baseline).

# COMMAND ----------
# MAGIC %pip install -q lightgbm hyperopt
# MAGIC %restart_python

# COMMAND ----------
try:
    dbutils.widgets.text("stream_schema", "skywatch.stream")
    dbutils.widgets.text("model_name", "skywatch.ml.eta_touchdown")
    dbutils.widgets.text("test_date", "2026-09-01")
    dbutils.widgets.text("max_evals", "24")
    dbutils.widgets.text("run_automl", "false")
    STREAM = dbutils.widgets.get("stream_schema")
    MODEL_NAME = dbutils.widgets.get("model_name")
    TEST_DATE = dbutils.widgets.get("test_date")
    MAX_EVALS = int(dbutils.widgets.get("max_evals"))
    RUN_AUTOML = dbutils.widgets.get("run_automl").lower() == "true"
except Exception:
    STREAM, MODEL_NAME, TEST_DATE, MAX_EVALS, RUN_AUTOML = (
        "skywatch.stream", "skywatch.ml.eta_touchdown", "2026-09-01", 24, False,
    )
print(f"train set: {STREAM}.gold_arrival_tracks | test day: {TEST_DATE} | model: {MODEL_NAME}")

# COMMAND ----------
# MAGIC %md ## 1. Load + feature engineering

# COMMAND ----------
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

# wide-body / heavy ICAO type codes seen at KATL (+ common freighters)
HEAVY = {
    "B762", "B763", "B764", "B772", "B773", "B77L", "B77W", "B778", "B779",
    "B788", "B789", "B78X", "B742", "B744", "B748", "A332", "A333", "A338",
    "A339", "A342", "A343", "A345", "A346", "A359", "A35K", "A388", "MD11",
    "IL76", "A124", "C17", "C5M",
}

FEATURES_NUM = [
    "dist_to_apt_nm", "heading_err_deg", "bearing_sin", "bearing_cos", "alt_ft", "gs_kt",
    "vrate_fpm", "closure_nm", "turn_rate_dps", "sel_altitude_ft",
    "airport_inbound_count", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_heavy",
]
FEATURES_CAT = ["phase"]
FEATURES = FEATURES_NUM + FEATURES_CAT
TARGET = "minutes_to_touchdown"

sdf = (
    spark.table(f"{STREAM}.gold_arrival_tracks")
    .where(F.col("minutes_to_touchdown").between(0.5, 40))
    .where(F.col("dist_to_apt_nm").isNotNull() & F.col("gs_kt").isNotNull() & F.col("alt_ft").isNotNull())
    .withColumn("obs_date", F.to_date("snapshot_ts"))
    .withColumn("is_heavy", F.col("ac_type").isin(list(HEAVY)).cast("int"))
    .withColumn("hour_sin", F.sin(F.col("hour_utc") / 24 * 2 * np.pi))
    .withColumn("hour_cos", F.cos(F.col("hour_utc") / 24 * 2 * np.pi))
    .withColumn("dow_sin", F.sin((F.col("dow") - 1) / 7 * 2 * np.pi))
    .withColumn("dow_cos", F.cos((F.col("dow") - 1) / 7 * 2 * np.pi))
    .withColumn("bearing_sin", F.sin(F.radians("bearing_to_apt")))
    .withColumn("bearing_cos", F.cos(F.radians("bearing_to_apt")))
    .withColumn("baseline_pred", F.col("dist_to_apt_nm") / F.col("gs_kt") * 60)
)

pdf = sdf.select(
    "seg_id", "obs_date", "ac_type", TARGET, "baseline_pred", *FEATURES
).toPandas()
# Spark round() yields DECIMAL -> decimal.Decimal in pandas; force float everywhere numeric
_num = [c for c in pdf.columns if c not in ("seg_id", "obs_date", "ac_type", "phase")]
pdf[_num] = pdf[_num].astype("float64")
pdf["phase"] = pdf["phase"].fillna("unknown").astype("category")

print(f"{len(pdf):,} rows | {pdf.seg_id.nunique():,} arrivals | dates {sorted(pdf.obs_date.unique())}")

# COMMAND ----------
# MAGIC %md ## 2. Time-based split
# MAGIC Everything before the test day trains; the test day is held out whole. A slice of the
# MAGIC train days is carved off as the Hyperopt validation set (again by whole days where
# MAGIC possible, else by arrival).

# COMMAND ----------
test_date = pd.Timestamp(TEST_DATE).date()
is_test = pdf.obs_date == test_date
train_pool = pdf[~is_test].copy()
test = pdf[is_test].copy()

# validation = the latest train day; if that leaves too little to train on, fall back to a
# 80/20 split of whole arrivals
train_days = sorted(train_pool.obs_date.unique())
val_day = train_days[-1]
train = train_pool[train_pool.obs_date != val_day]
val = train_pool[train_pool.obs_date == val_day]
if len(train) < 0.4 * len(train_pool):
    segs = train_pool.seg_id.drop_duplicates().sample(frac=0.8, random_state=42)
    train = train_pool[train_pool.seg_id.isin(segs)]
    val = train_pool[~train_pool.seg_id.isin(segs)]

for name, d in [("train", train), ("val", val), ("test", test)]:
    print(f"{name:6} {len(d):>7,} rows  {d.seg_id.nunique():>5} arrivals  "
          f"dates {sorted(d.obs_date.unique())}")

X_tr, y_tr = train[FEATURES], train[TARGET]
X_va, y_va = val[FEATURES], val[TARGET]
X_te, y_te = test[FEATURES], test[TARGET]

# COMMAND ----------
# MAGIC %md ## 3. Baseline

# COMMAND ----------
from sklearn.metrics import mean_absolute_error, mean_squared_error


def by_band(df, pred, actual):
    b = pd.cut(df["dist_to_apt_nm"], [0, 20, 40, 70, 101], include_lowest=True,
              labels=["0-20nm", "20-40nm", "40-70nm", "70-100nm"])
    out = []
    for band in b.cat.categories:
        m = b == band
        out.append({"band": band, "n": int(m.sum()),
                    "MAE_min": round(mean_absolute_error(actual[m], pred[m]), 2),
                    "P90_min": round(np.percentile(np.abs(actual[m] - pred[m]), 90), 2),
                    "bias_min": round((pred[m] - actual[m]).mean(), 2)})
    out.append({"band": "ALL", "n": len(df),
                "MAE_min": round(mean_absolute_error(actual, pred), 2),
                "P90_min": round(np.percentile(np.abs(actual - pred), 90), 2),
                "bias_min": round((pred - actual).mean(), 2)})
    return pd.DataFrame(out)


base_tbl = by_band(test, test["baseline_pred"].values, y_te.values)
print("BASELINE  dist / gs * 60,  on the test day")
display(base_tbl)
baseline_mae = mean_absolute_error(y_te, test["baseline_pred"])

# COMMAND ----------
# MAGIC %md ## 4. LightGBM + Hyperopt

# COMMAND ----------
import lightgbm as lgb
import mlflow
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

mlflow.set_registry_uri("databricks-uc")
mlflow.lightgbm.autolog(log_models=False)
CAT_IDX = [FEATURES.index(c) for c in FEATURES_CAT]

search_space = {
    "num_leaves": hp.quniform("num_leaves", 15, 255, 1),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    "feature_fraction": hp.uniform("feature_fraction", 0.5, 1.0),
    "bagging_fraction": hp.uniform("bagging_fraction", 0.5, 1.0),
    "min_child_samples": hp.quniform("min_child_samples", 10, 200, 5),
    "lambda_l2": hp.loguniform("lambda_l2", np.log(1e-3), np.log(10.0)),
}


def objective(params):
    params = {**params,
              "num_leaves": int(params["num_leaves"]),
              "min_child_samples": int(params["min_child_samples"]),
              "objective": "mae", "n_estimators": 2000, "n_jobs": -1, "verbose": -1}
    with mlflow.start_run(nested=True):
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="mae",
                  categorical_feature=CAT_IDX,
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        mae = mean_absolute_error(y_va, model.predict(X_va))
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("best_iteration", model.best_iteration_ or params["n_estimators"])
        return {"loss": mae, "status": STATUS_OK}


with mlflow.start_run(run_name="eta_hyperopt") as parent_run:
    trials = Trials()
    best = fmin(objective, search_space, algo=tpe.suggest, max_evals=MAX_EVALS,
                trials=trials, rstate=np.random.default_rng(42))
    best_val_mae = min(t["result"]["loss"] for t in trials.trials)
    mlflow.log_metric("best_val_mae", best_val_mae)
    print(f"best val MAE = {best_val_mae:.3f} min   params={best}")

# COMMAND ----------
# MAGIC %md ## 5. Refit on train+val, evaluate on the test day, register

# COMMAND ----------
from mlflow.models import infer_signature

best_params = {
    "num_leaves": int(best["num_leaves"]),
    "learning_rate": float(best["learning_rate"]),
    "feature_fraction": float(best["feature_fraction"]),
    "bagging_fraction": float(best["bagging_fraction"]),
    "min_child_samples": int(best["min_child_samples"]),
    "lambda_l2": float(best["lambda_l2"]),
    "objective": "mae", "n_estimators": 2000, "n_jobs": -1, "verbose": -1,
}

X_trval = pd.concat([X_tr, X_va])
y_trval = pd.concat([y_tr, y_va])

with mlflow.start_run(run_name="eta_final") as final_run:
    # early-stopping iteration count learned during tuning, applied here via a val tail
    fit_model = lgb.LGBMRegressor(**best_params)
    fit_model.fit(X_trval, y_trval, eval_set=[(X_te, y_te)], eval_metric="mae",
                  categorical_feature=CAT_IDX,
                  callbacks=[lgb.early_stopping(50, verbose=False)])

    pred_te = fit_model.predict(X_te)
    model_tbl = by_band(test, pred_te, y_te.values)
    model_mae = mean_absolute_error(y_te, pred_te)
    model_rmse = mean_squared_error(y_te, pred_te) ** 0.5

    mlflow.log_params(best_params)
    mlflow.log_metrics({
        "test_mae_min": model_mae, "test_rmse_min": model_rmse,
        "baseline_mae_min": baseline_mae,
        "improvement_pct": round(100 * (baseline_mae - model_mae) / baseline_mae, 1),
    })
    for _, r in model_tbl.iterrows():
        mlflow.log_metric(f"test_mae_{r['band']}", r["MAE_min"])

    sig = infer_signature(X_te, pred_te)
    mlflow.lightgbm.log_model(fit_model, "model", signature=sig,
                              input_example=X_te.head(3))
    model_uri = f"runs:/{final_run.info.run_id}/model"

print(f"\nTEST DAY {TEST_DATE}")
print(f"  model MAE    {model_mae:.2f} min   RMSE {model_rmse:.2f}")
print(f"  baseline MAE {baseline_mae:.2f} min")
print(f"  improvement  {100 * (baseline_mae - model_mae) / baseline_mae:.1f}%")
display(model_tbl.merge(base_tbl, on=["band", "n"], suffixes=("_model", "_baseline")))

# COMMAND ----------
# MAGIC %md ## 6. Feature importance + example trajectories

# COMMAND ----------
imp = (pd.DataFrame({"feature": FEATURES, "gain": fit_model.booster_.feature_importance("gain")})
       .sort_values("gain", ascending=False))
display(imp)

ex = test.copy()
ex["pred"] = pred_te
ex["abs_err"] = (ex["pred"] - ex[TARGET]).abs()
show = ex[ex.seg_id.isin(ex.seg_id.drop_duplicates().sample(3, random_state=1))]
display(show[["seg_id", "ac_type", "dist_to_apt_nm", "alt_ft", "gs_kt",
              TARGET, "pred", "baseline_pred", "abs_err"]]
        .sort_values(["seg_id", "dist_to_apt_nm"], ascending=[True, False]))

# COMMAND ----------
# MAGIC %md ## 7. Register to Unity Catalog

# COMMAND ----------
from mlflow import MlflowClient

client = MlflowClient()
mv = mlflow.register_model(model_uri, MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
client.set_model_version_tag(MODEL_NAME, mv.version, "test_mae_min", f"{model_mae:.3f}")
client.set_model_version_tag(MODEL_NAME, mv.version, "test_date", TEST_DATE)

if model_mae < baseline_mae:
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    print(f"registered {MODEL_NAME} v{mv.version} as @challenger AND @champion "
          f"(beats baseline: {model_mae:.2f} < {baseline_mae:.2f})")
else:
    print(f"registered {MODEL_NAME} v{mv.version} as @challenger only "
          f"(did NOT beat baseline {baseline_mae:.2f})")

# COMMAND ----------
if RUN_AUTOML:
    # optional cross-check; heavier on quota
    from databricks import automl

    aml_pdf = train_pool[[*FEATURES, TARGET]].copy()
    aml_pdf["phase"] = aml_pdf["phase"].astype(str)
    summary = automl.regress(
        spark.createDataFrame(aml_pdf),
        target_col=TARGET, primary_metric="mae", timeout_minutes=20,
    )
    print("AutoML best trial MAE:", summary.best_trial.metrics.get("val_mae"))
