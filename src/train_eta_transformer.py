# Databricks notebook source
# MAGIC %md
# MAGIC # SkyWatch — Model 1 Transformer variant (time-to-touchdown, sequence model)
# MAGIC
# MAGIC The same regression `train_eta.py` solves — minutes to touchdown, given an arrival's
# MAGIC reports so far — but over a **causal window of the last `SEQ_LEN` reports** instead of one
# MAGIC point, via a small Transformer encoder. Same target, same features (`ETA_FEATURES`), same
# MAGIC confirmed-touchdown filter, same time-based split, same distance-band MAE report as
# MAGIC `train_eta.py` — the point is a fair, apples-to-apples comparison, not a different problem.
# MAGIC See `docs/M1_TRANSFORMER_PLAN.md` for why this exists and how it plugs into the existing
# MAGIC promotion gate (`promote_eta.py`) without changing the gate's own logic.
# MAGIC
# MAGIC Registers to the **same** `skywatch.ml.eta_touchdown` model as the LightGBM champion,
# MAGIC tagged `model_flavor=pytorch_transformer` so `promote_eta.py` knows to load it via
# MAGIC `mlflow.pyfunc` (generic) instead of `mlflow.lightgbm` (LightGBM-specific — see
# MAGIC `score_eta.py`'s note on why the two can't share a loader). Sets `@challenger`; never
# MAGIC touches `@champion` — same promotion boundary as every other training notebook in this
# MAGIC project.
# MAGIC
# MAGIC CPU only, deliberately — the model is small and the dataset (25k arrivals) doesn't need
# MAGIC Free Edition's quota-gated serverless GPU (same call the project made for Model 2's
# MAGIC Chronos-Bolt fine-tune).

# COMMAND ----------
# MAGIC %pip install -q torch --index-url https://download.pytorch.org/whl/cpu
# MAGIC %restart_python

# COMMAND ----------
try:
    dbutils.widgets.text("stream_schema", "skywatch.stream")
    dbutils.widgets.text("read_stream_schema", "")
    dbutils.widgets.text("model_name", "skywatch.ml.eta_touchdown")
    dbutils.widgets.text("test_date", "2026-09-01")
    dbutils.widgets.text("epochs", "40")
    dbutils.widgets.text("patience", "5")
    dbutils.widgets.text("batch_size", "256")
    dbutils.widgets.text("learning_rate", "0.001")
    dbutils.widgets.text("d_model", "64")
    dbutils.widgets.text("nhead", "4")
    dbutils.widgets.text("num_layers", "2")
    STREAM = dbutils.widgets.get("stream_schema")
    READ_STREAM = dbutils.widgets.get("read_stream_schema").strip() or STREAM
    MODEL_NAME = dbutils.widgets.get("model_name")
    TEST_DATE = dbutils.widgets.get("test_date")
    EPOCHS = int(dbutils.widgets.get("epochs"))
    PATIENCE = int(dbutils.widgets.get("patience"))
    BATCH_SIZE = int(dbutils.widgets.get("batch_size"))
    LEARNING_RATE = float(dbutils.widgets.get("learning_rate"))
    D_MODEL = int(dbutils.widgets.get("d_model"))
    NHEAD = int(dbutils.widgets.get("nhead"))
    NUM_LAYERS = int(dbutils.widgets.get("num_layers"))
except Exception:
    STREAM, MODEL_NAME, TEST_DATE = "skywatch.stream", "skywatch.ml.eta_touchdown", "2026-09-01"
    READ_STREAM = STREAM
    EPOCHS, PATIENCE, BATCH_SIZE, LEARNING_RATE = 40, 5, 256, 0.001
    D_MODEL, NHEAD, NUM_LAYERS = 64, 4, 2

print(f"train set: {READ_STREAM}.gold_arrival_tracks | test day: {TEST_DATE} | model: {MODEL_NAME}")

# COMMAND ----------
# MAGIC %run ./eta_sequences

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Load + build causal sequence windows
# MAGIC Identical filters to `train_eta.py`: confirmed touchdowns only, `minutes_to_touchdown` in
# MAGIC `[0.5, 40]`, non-null distance/altitude, `gs_kt > 60` (guards the baseline's divide-by-zero
# MAGIC and excludes hovering/near-stationary fixes).

# COMMAND ----------
from pyspark.sql import functions as F

confirmed_segs = (
    spark.table(f"{READ_STREAM}.gold_touchdowns")
    .where(F.col("touchdown_confidence") == "confirmed")
    .select("seg_id")
)

sdf = add_eta_features(
    spark.table(f"{READ_STREAM}.gold_arrival_tracks")
    .join(confirmed_segs, "seg_id")
    .where(F.col(ETA_TARGET).between(0.5, 40))
    .where(F.col("dist_to_apt_nm").isNotNull() & F.col("alt_ft").isNotNull())
    .where(F.col("gs_kt") > 60)
)
frame = build_sequence_frame(sdf)
frame["baseline_pred"] = frame["dist_to_apt_nm"] / frame["gs_kt"] * 60
seq_cols = [c for c in frame.columns if c.startswith("seq_")]

print(f"{len(frame):,} rows | {frame.seg_id.nunique():,} arrivals | "
      f"dates {sorted(frame.obs_date.unique())}")

# COMMAND ----------
# MAGIC %md ## 2. Time-based split — same scheme as `train_eta.py`

# COMMAND ----------
import pandas as pd

test_date = pd.Timestamp(TEST_DATE).date()
is_test = frame.obs_date == test_date
train_pool = frame[~is_test].copy()
test = frame[is_test].copy()

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

# COMMAND ----------
# MAGIC %md ## 3. Baseline (same metric `train_eta.py` reports against)

# COMMAND ----------
import numpy as np
from sklearn.metrics import mean_absolute_error


def by_band(dist, pred, actual):
    bands = pd.cut(dist, [0, 20, 40, 70, 101], include_lowest=True,
                   labels=["0-20nm", "20-40nm", "40-70nm", "70-100nm"])
    out = {}
    for band in bands.cat.categories:
        m = (bands == band).to_numpy()
        if m.sum():
            out[band] = mean_absolute_error(actual[m], pred[m])
    out["ALL"] = mean_absolute_error(actual, pred)
    return out


baseline_bands = by_band(test["dist_to_apt_nm"], test["baseline_pred"].to_numpy(), test[ETA_TARGET].to_numpy())
print("BASELINE  dist / gs * 60,  on the test day")
print({k: round(v, 3) for k, v in baseline_bands.items()})

# COMMAND ----------
# MAGIC %md ## 4. Transformer encoder over the causal window

# COMMAND ----------
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lib.sequences import SEQ_LEN, unflatten_windows

torch.manual_seed(42)
N_PHASE = len(PHASE_CATEGORIES)


def to_tensors(df):
    flat = df[seq_cols].to_numpy(dtype=np.float32)
    windows, mask = unflatten_windows(flat, n_features=N_SEQ_FEATURES)
    continuous = torch.from_numpy(windows[:, :, :-1].copy())
    phase_idx = torch.from_numpy(windows[:, :, -1].astype(np.int64).copy())
    mask_t = torch.from_numpy(mask.copy())
    y = torch.from_numpy(df[ETA_TARGET].to_numpy(dtype=np.float32))
    return continuous, phase_idx, mask_t, y


class EtaTransformer(nn.Module):
    def __init__(self, n_continuous, n_phase, d_model, nhead, num_layers, seq_len,
                 phase_embed_dim=8, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.phase_embed = nn.Embedding(n_phase, phase_embed_dim)
        self.input_proj = nn.Linear(n_continuous + phase_embed_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, continuous, phase_idx, mask):
        x = torch.cat([continuous, self.phase_embed(phase_idx)], dim=-1)
        x = self.input_proj(x) + self.pos_embed
        # our windowing (src/lib/sequences.py) always puts the current report at the last
        # position, real (never padding) — the encoder output there is the "as of now" state.
        enc = self.encoder(x, src_key_padding_mask=~mask)
        return self.head(enc[:, -1, :]).squeeze(-1)


MODEL_CONFIG = dict(
    n_continuous=N_SEQ_FEATURES - 1, n_phase=N_PHASE, d_model=D_MODEL, nhead=NHEAD,
    num_layers=NUM_LAYERS, seq_len=SEQ_LEN,
)


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, ns = [], []
    for cont, phase_idx, mask, y in loader:
        with torch.set_grad_enabled(training):
            pred = model(cont, phase_idx, mask)
            loss = nn.functional.l1_loss(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(loss.item() * len(y))
        ns.append(len(y))
    return sum(losses) / sum(ns)


def make_loader(df, shuffle):
    cont, phase_idx, mask, y = to_tensors(df)
    return DataLoader(TensorDataset(cont, phase_idx, mask, y), batch_size=BATCH_SIZE, shuffle=shuffle)


def train_model(train_df, val_df, max_epochs, patience):
    """`val_df=None` trains for exactly `max_epochs` with no early stopping and no per-epoch
    validation (stage 2 — `val_df` is folded into `train_df` by then, so evaluating against it
    again would be a leak, same reason `train_eta.py`'s stage-2 LightGBM refit passes no
    `eval_set`). Otherwise trains with early stopping on `val_df`.
    Returns `(best_state_dict, best_epoch, best_val_mae)` — `best_val_mae` is `nan` when
    `val_df=None`.
    """
    model = EtaTransformer(**MODEL_CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False) if val_df is not None else None

    best_val, best_epoch, best_state, bad_epochs = float("inf"), 0, None, 0
    for epoch in range(1, max_epochs + 1):
        train_mae = run_epoch(model, train_loader, optimizer)
        if val_loader is None:
            print(f"  epoch {epoch:>3}  train MAE {train_mae:.3f}")
            best_epoch, best_state = epoch, model.state_dict()
            continue
        val_mae = run_epoch(model, val_loader)
        print(f"  epoch {epoch:>3}  train MAE {train_mae:.3f}  val MAE {val_mae:.3f}")
        if val_mae < best_val:
            best_val, best_epoch, best_state, bad_epochs = val_mae, epoch, model.state_dict(), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stopping at epoch {epoch} (best was epoch {best_epoch})")
                break
    return best_state, best_epoch, best_val

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Stage 1 — learn the epoch count on a real held-out validation day
# MAGIC Mirrors `train_eta.py`'s two-stage LightGBM fit: calibrate on train/val, then refit on
# MAGIC train+val at a fixed budget (`best_epoch * 1.15`, same ~15%-more-data adjustment factor)
# MAGIC with no further early stopping, and touch the test day exactly once.

# COMMAND ----------
import mlflow

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="eta_transformer") as run:
    mlflow.log_params({
        "d_model": D_MODEL, "nhead": NHEAD, "num_layers": NUM_LAYERS,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "seq_len": SEQ_LEN,
    })

    print("stage 1 — calibrating epoch count on the validation day")
    _, best_epoch, best_val_mae = train_model(train, val, EPOCHS, PATIENCE)
    n_final_epochs = max(1, round(best_epoch * 1.15))
    mlflow.log_metrics({"best_val_mae": best_val_mae, "n_epochs_final": n_final_epochs})
    print(f"best val MAE = {best_val_mae:.3f} min at epoch {best_epoch} -> "
          f"refitting for {n_final_epochs} epochs on train+val")

    print("stage 2 — refit on train+val, no early stopping")
    trainval = pd.concat([train, val])
    final_state, _, _ = train_model(trainval, None, n_final_epochs, patience=n_final_epochs)

    final_model = EtaTransformer(**MODEL_CONFIG)
    final_model.load_state_dict(final_state)
    final_model.eval()

    with torch.no_grad():
        cont, phase_idx, mask, y = to_tensors(test)
        pred_te = final_model(cont, phase_idx, mask).numpy()

    model_bands = by_band(test["dist_to_apt_nm"], pred_te, y.numpy())
    model_mae = model_bands["ALL"]
    baseline_mae = baseline_bands["ALL"]
    mlflow.log_metrics({
        "test_mae_min": model_mae,
        "baseline_mae_min": baseline_mae,
        "improvement_pct": round(100 * (baseline_mae - model_mae) / baseline_mae, 1),
    })
    for band, mae in model_bands.items():
        mlflow.log_metric(f"test_mae_{band}", mae)

    print(f"\nTEST DAY {TEST_DATE}")
    print(f"  model MAE    {model_mae:.3f} min")
    print(f"  baseline MAE {baseline_mae:.3f} min")
    print(f"  improvement  {100 * (baseline_mae - model_mae) / baseline_mae:.1f}%")
    print({k: round(v, 3) for k, v in model_bands.items()})

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Log as a custom pyfunc + register as `@challenger`
# MAGIC State dict saved via `torch.save`/`context.artifacts` (MLflow's documented pattern for
# MAGIC wrapping a PyTorch model), not cloudpickled inline — the standard, portable way to package
# MAGIC a non-native-flavor model as pyfunc.

# COMMAND ----------
import tempfile

from mlflow.models import infer_signature


class EtaTransformerPyfunc(mlflow.pyfunc.PythonModel):
    def __init__(self, model_config, n_seq_features):
        self.model_config = model_config
        self.n_seq_features = n_seq_features

    def load_context(self, context):
        import torch as _torch

        model = EtaTransformer(**self.model_config)
        model.load_state_dict(_torch.load(context.artifacts["state_dict"], map_location="cpu"))
        model.eval()
        self._model = model

    def predict(self, context, model_input, params=None):
        import numpy as _np
        import torch as _torch

        from lib.sequences import unflatten_windows

        flat = model_input.to_numpy(dtype=_np.float32)
        windows, mask = unflatten_windows(flat, n_features=self.n_seq_features)
        with _torch.no_grad():
            cont = _torch.from_numpy(windows[:, :, :-1].copy())
            phase_idx = _torch.from_numpy(windows[:, :, -1].astype(_np.int64).copy())
            mask_t = _torch.from_numpy(mask.copy())
            return self._model(cont, phase_idx, mask_t).numpy()


with mlflow.start_run(run_id=run.info.run_id):
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_path = f"{tmp_dir}/state_dict.pt"
        torch.save(final_state, state_path)

        X_sig = test[seq_cols]
        sig = infer_signature(X_sig, pred_te)
        mlflow.pyfunc.log_model(
            "model",
            python_model=EtaTransformerPyfunc(MODEL_CONFIG, N_SEQ_FEATURES),
            artifacts={"state_dict": state_path},
            signature=sig,
            input_example=X_sig.head(3),
            pip_requirements=["torch", "numpy<2", "pandas"],
        )
    model_uri = f"runs:/{run.info.run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)

client = mlflow.MlflowClient()
client.set_model_version_tag(MODEL_NAME, mv.version, "model_flavor", "pytorch_transformer")
client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
print(f"\nregistered {MODEL_NAME} v{mv.version} as @challenger (model_flavor=pytorch_transformer)")
print("run src/promote_eta.py to evaluate it against @champion through the same gate.")
