
# -*- coding: utf-8 -*-
"""
BTC_USDC_Forecast_Lab_Advanced_Integrated.py
by Igor Azevedo - Data Scientist
Keeps the original advanced pipeline (features, TF+Optuna, classic ML) and
adds a full PyTorch LSTM with Optuna tuning, early stopping, gradient clipping,
and ReduceLROnPlateau. Everything controlled by flags.

Install (if needed):
    %pip install -q pandas numpy scikit-learn matplotlib requests tensorflow torch optuna
"""

# ==============================
# Imports
# ==============================
import math
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os

import tempfile, shutil

MODEL_DIR = "outputs/model"
MODEL_PATH = os.path.join(MODEL_DIR, "bitcoinhomebroker.pth")
os.makedirs(MODEL_DIR, exist_ok=True)

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# === Checkpoint helpers ===
def _extract_state_dict(chk):
    # aceita tanto dict puro de state_dict quanto dict com metadados
    if isinstance(chk, dict) and "state_dict" in chk:
        return chk["state_dict"]
    return chk

def _extract_hparams(chk):
    if isinstance(chk, dict) and "hparams" in chk:
        return chk["hparams"]
    return None

def load_weights_if_compatible(model, path, device, expect_hparams=None, verbose=True):
    if not path or not os.path.exists(path):
        if verbose: print(f"[ckpt] Não existe checkpoint em: {path}")
        return False
    chk = torch.load(path, map_location=device)
    ck_hparams = _extract_hparams(chk)
    if expect_hparams is not None and ck_hparams is not None:
        # Verificação rápida de compatibilidade estrutural
        keys_to_check = ["hidden", "layers", "bidir"]
        diff = {k: (ck_hparams.get(k), expect_hparams.get(k)) for k in keys_to_check
                if ck_hparams.get(k) != expect_hparams.get(k)}
        if diff:
            if verbose: print(f"[ckpt] Hparams incompatíveis, pulando load: {diff}")
            return False

    sd = _extract_state_dict(chk)
    model_sd = model.state_dict()
    # Filtra apenas tensores com mesmas shapes
    filtered = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    if not filtered:
        if verbose: print("[ckpt] Nenhum tensor compatível encontrado; pulando load.")
        return False
    # Carrega de forma não estrita (o que faltar usa inicialização atual)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if verbose:
        print(f"[ckpt] Pesos carregados: {len(filtered)} | faltando: {len(missing)} | inesperados ignorados: {len(unexpected)}")
    return True

def save_checkpoint(model, path, hparams=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if hparams is not None:
        payload["hparams"] = dict(hparams)  # grava os melhores params p/ compat-check futuro
    torch.save(payload, path)
    print(f"[ckpt] Modelo salvo em: {path}")

def save_serving_artifacts(model, model_path, best_params, scaler_obj, lookback, horizon, extra_cfg=None):
    """
    Salva:
      - checkpoint .pth com state_dict + hparams
      - scaler.json com mean/scale do CLOSE
      - config.json com lookback/horizon + hparams necessários p/ reconstruir o modelo no serving
    """
    # 1) hparams essenciais p/ reconstruir a arquitetura
    hparams = {
        "hidden": int(best_params["hidden"]),
        "layers": int(best_params["layers"]),
        "dropout": float(best_params["dropout"]),
        "bidir": bool(best_params["bidir"]),
        # (lr, batch, epochs não são necessários p/ servir, mas podem ser úteis p/ auditoria)
        "lr": float(best_params["lr"]),
        "batch": int(best_params["batch"]),
        "epochs": int(best_params["epochs"]),
    }

    # 2) checkpoint consolidado com hparams
    save_checkpoint(model, model_path, hparams=hparams)

    # 3) scaler do CLOSE (índice 0 do StandardScaler que você usou)
    scaler_payload = {
        "mean": float(scaler_obj.mean_[0]),
        "scale": float(scaler_obj.scale_[0]),
    }
    with open(SCALER_PATH, "w") as f:
        json.dump(scaler_payload, f)

    # 4) config.json para o servidor
    cfg = {
        "lookback": int(lookback),
        "horizon": int(horizon),
        "input_feature": "close",
        # opcional: reescrevemos os hparams aqui também
        **hparams
    }
    if extra_cfg:
        cfg.update(extra_cfg)

    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f)

    print(f"[serve] Artefatos salvos:\n - {model_path}\n - {SCALER_PATH}\n - {CONFIG_PATH}")


# Try Optuna import
try:
    import optuna
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True

# ==============================
# Utils
# ==============================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def ensure_datetime(df, col="time"):
    df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
    return df

def plot_series(df, title):
    fig, ax = plt.subplots()
    ax.plot(df["time"], df["close"], linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("close")
    plt.show()

def plot_history_tf(history, title="Training history (TF)"):
    fig, ax = plt.subplots()
    ax.plot(history.history.get("loss", []), label="loss")
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="val_loss")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    plt.show()

def evaluation_regression(y_true, y_pred, label=""):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} - RMSE: {rmse:.6f} | MAE: {mae:.6f} | R^2: {r2:.6f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

def evaluation_classification(y_true, y_pred, label=""):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"{label} - ACC: {acc:.6f} | F1: {f1:.6f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return {"acc": acc, "f1": f1}
def plot_line(y_true, y_pred, title):
    fig, ax = plt.subplots()
    ax.plot(y_true, label="true")
    ax.plot(y_pred, label="pred")
    ax.set_title(title)
    ax.legend()
    plt.show()

# ==============================
# Parameters & Flags
# ==============================
INTERVAL = "15m"     # "15m" or "1d"
LOOKBACK = 48        # sequence window length for LSTM (used for initial model; Optuna tunes others)
HORIZON = 1          # steps ahead to predict
TEST_PCT = 0.2       # proportion for test split (time-based)

# Classic ML toggles
RUN_CLASSIC_ML = True

# TensorFlow LSTM (defaults; may be overridden by Optuna)
RUN_TF = True
DO_TUNE_TF = True
TF_EPOCHS = 50
TF_HIDDEN = 64
TF_LAYERS = 1
TF_DROPOUT = 0.0
TF_BATCH = 64
TF_LR = 1e-3
N_TRIALS_TF = 20

# PyTorch LSTM (baseline & tuning)
RUN_PT_BASELINE = False  # keep baseline optional
RUN_PT_TUNING   = True   # new: Optuna tuning for PT
TORCH_EPOCHS = 20
TORCH_HIDDEN = 64
TORCH_BATCH = 64
LR = 1e-3
N_TRIALS_PT = 25

# ==============================
# Data Fetching — KuCoin, Poloniex
# ==============================
def fetch_kucoin(symbol="BTC-USDC", interval="15min", limit=1000, start_at=None, end_at=None):
    url = "https://api.kucoin.com/api/v1/market/candles"
    params = dict(type=interval, symbol=symbol)
    if start_at is not None: params["startAt"] = int(start_at)
    if end_at   is not None: params["endAt"]   = int(end_at)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["data"]
    rows = []
    for a in data[:limit][::-1]:
        t_ms = int(a[0]) * 1000
        rows.append([t_ms, float(a[1]), float(a[3]), float(a[4]), float(a[2]), float(a[5])])
    df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
    return df.sort_values("time").reset_index(drop=True)

def fetch_poloniex(symbol="BTC_USDT", interval="MINUTE_15", limit=500, start_ms=None, end_ms=None):
    url = f"https://api.poloniex.com/markets/{symbol}/candles"
    params = {"interval": interval, "limit": min(limit, 500)}
    if start_ms is not None: params["startTime"] = int(start_ms)
    if end_ms   is not None: params["endTime"]   = int(end_ms)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    arr = r.json()
    rows = []
    for a in arr:
        time_ms = int(a[12])
        o = float(a[2]); h = float(a[1]); l = float(a[0]); c = float(a[3])
        vol_base = float(a[5])
        rows.append([time_ms, o, h, l, c, vol_base])
    df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
    return df.sort_values("time").reset_index(drop=True)

if INTERVAL == "15m":
    df_ku  = fetch_kucoin(symbol="BTC-USDC", interval="15min", limit=1000)
    df_po  = fetch_poloniex(symbol="BTC_USDT", interval="MINUTE_15", limit=500)
elif INTERVAL == "1d":
    df_ku  = fetch_kucoin(symbol="BTC-USDC", interval="1day", limit=1000)
    df_po  = fetch_poloniex(symbol="BTC_USDT", interval="DAY_1", limit=500)
else:
    raise ValueError("INTERVAL must be '15m' or '1d'.")

print(len(df_ku), len(df_po))
plot_series(ensure_datetime(df_ku.copy()),  "KuCoin Close")
plot_series(ensure_datetime(df_po.copy()),  "Poloniex Close (USDT proxy)")

# ==============================
# Merge & Base Features
# ==============================
def merge_exchanges(dfs):
    keyed = []
    for name, df in dfs.items():
        d = df[["time","open","high","low","close","volume"]].copy()
        d = ensure_datetime(d, "time")
        d.set_index("time", inplace=True)
        d.columns = pd.MultiIndex.from_product([[name], d.columns])
        keyed.append(d)
    big = pd.concat(keyed, axis=1).sort_index()

    close_cols = [c for c in big.columns if c[1]=="close"]
    vol_cols   = [c for c in big.columns if c[1]=="volume"]

    agg = pd.DataFrame(index=big.index)
    agg["close"]  = big[close_cols].mean(axis=1, skipna=True)
    agg["open"]   = big.xs("open", axis=1, level=1).mean(axis=1, skipna=True)
    agg["high"]   = big.xs("high", axis=1, level=1).mean(axis=1, skipna=True)
    agg["low"]    = big.xs("low", axis=1, level=1).mean(axis=1, skipna=True)
    agg["volume"] = big[vol_cols].sum(axis=1, skipna=True)

    agg = agg.dropna().reset_index().rename(columns={"index":"time"})
    agg["ret_1"]   = agg["close"].pct_change()
    agg["ret_4"]   = agg["close"].pct_change(4)
    agg["ret_12"]  = agg["close"].pct_change(12)
    agg["vol_mean_12"]  = agg["volume"].rolling(12).mean()
    agg["price_ma_12"]  = agg["close"].rolling(12).mean()
    agg["price_std_12"] = agg["close"].rolling(12).std()
    return agg

dfs = { "kucoin": df_ku, "poloniex": df_po }
df = merge_exchanges(dfs)

# ==============================
# Technical Indicators & Time Features
# ==============================
def add_ta_features(df):
    df = df.copy()
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll = 14
    avg_gain = pd.Series(gain, index=df.index).rolling(roll).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(roll).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']   = df['ema_12'] - df['ema_26']
    df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_sig']
    bb_win = 20
    ma = df['close'].rolling(bb_win).mean()
    sd = df['close'].rolling(bb_win).std()
    df['bb_mid_20'] = ma
    df['bb_up_20']  = ma + 2*sd
    df['bb_lo_20']  = ma - 2*sd
    df['bb_bw_20']  = (df['bb_up_20'] - df['bb_lo_20']) / (df['bb_mid_20'] + 1e-12)
    df['bb_pct_20'] = (df['close'] - df['bb_lo_20']) / (df['bb_up_20'] - df['bb_lo_20'] + 1e-12)
    hl = (df['high'] - df['low']).abs()
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low']  - df['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    direction = np.sign(df['close'].diff().fillna(0.0))
    df['obv'] = (direction * df['volume']).fillna(0.0).cumsum()
    for k in [1,2,3,4,6,12]:
        df[f'close_lag{k}'] = df['close'].shift(k)
        df[f'vol_lag{k}']   = df['volume'].shift(k)
    for w in [6,12,24,48]:
        df[f'close_ma_{w}']  = df['close'].rolling(w).mean()
        df[f'close_std_{w}'] = df['close'].rolling(w).std()
        df[f'vol_ma_{w}']    = df['volume'].rolling(w).mean()
    return df

def add_time_features(df):
    df = df.copy()
    df = ensure_datetime(df, "time")
    df['hour'] = df['time'].dt.hour
    df['dow']  = df['time'].dt.dayofweek
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24.0)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24.0)
    df['dow_sin']  = np.sin(2*np.pi*df['dow']/7.0)
    df['dow_cos']  = np.cos(2*np.pi*df['dow']/7.0)
    return df

df = add_ta_features(df)
df = add_time_features(df)
df["target_reg"] = df["close"].shift(-1)
df["target_cls"] = (df["target_reg"] > df["close"]).astype(int)
df = df.dropna().reset_index(drop=True)

# ==============================
# Train/Test Split (Time-based)
# ==============================
split_idx = int(len(df) * (1 - TEST_PCT))
train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)

# ==============================
# Classic ML — Linear & Logistic Regression
# ==============================
feature_cols_base = [
    "open","high","low","close","volume",
    "ret_1","ret_4","ret_12",
    "vol_mean_12","price_ma_12","price_std_12",
    "rsi_14","ema_12","ema_26","macd","macd_sig","macd_hist",
    "bb_mid_20","bb_up_20","bb_lo_20","bb_bw_20","bb_pct_20",
    "atr_14","obv",
    "close_lag1","close_lag2","close_lag3","close_lag4","close_lag6","close_lag12",
    "vol_lag1","vol_lag2","vol_lag3","vol_lag4","vol_lag6","vol_lag12",
    "close_ma_6","close_std_6","vol_ma_6",
    "close_ma_12","close_std_12","vol_ma_12",
    "close_ma_24","close_std_24","vol_ma_24",
    "close_ma_48","close_std_48","vol_ma_48",
    "hour_sin","hour_cos","dow_sin","dow_cos"
]

RUN_CLASSIC_ML = RUN_CLASSIC_ML
if RUN_CLASSIC_ML:
    X = df[feature_cols_base].values
    y_reg = df["target_reg"].values
    y_cls = df["target_cls"].values

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]
    y_cls_train, y_cls_test = y_cls[:split_idx], y_cls[split_idx:]

    lin_pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("lr", LinearRegression())])
    lin_pipe.fit(X_train, y_reg_train)
    pred_lin = lin_pipe.predict(X_test)
    evaluation_regression(y_reg_test, pred_lin, "LinearRegression")

    fig, ax = plt.subplots()
    ax.plot(y_reg_test, label="true")
    ax.plot(pred_lin, label="pred")
    ax.set_title("LinearRegression — Test (next close)")
    ax.legend()
    plt.show()

    log_pipe = Pipeline([("scaler", StandardScaler()), ("log", LogisticRegression(max_iter=1000, class_weight='balanced'))])
    param_grid = {"log__C":[0.1,1.0,2.0,5.0], "log__penalty":["l2"]}
    cv = TimeSeriesSplit(n_splits=5)
    gs = GridSearchCV(log_pipe, param_grid, cv=cv, n_jobs=-1, scoring="f1_macro")
    gs.fit(X_train, y_cls_train)
    print("Best params (LogReg):", gs.best_params_)
    pred_log = gs.predict(X_test)
    evaluation_classification(y_cls_test, pred_log, "LogisticRegression")

# ==============================
# Deep Learning — LSTM Data Windows
# ==============================
def make_windows(arr, window, horizon=1):
    X, y = [], []
    for i in range(len(arr) - window - horizon + 1):
        X.append(arr[i:i+window])
        y.append(arr[i+window:i+window+horizon])
    return np.array(X), np.array(y).squeeze()

seq_cols = ["close","volume"]
arr = df[seq_cols].values
sc_seq = StandardScaler()
arr_scaled = sc_seq.fit_transform(arr)

X_seq, y_seq = make_windows(arr_scaled[:,0], window=LOOKBACK, horizon=HORIZON)
split_seq = int(len(X_seq) * (1 - TEST_PCT))
X_tr_seq, X_te_seq = X_seq[:split_seq], X_seq[split_seq:]
y_tr_seq, y_te_seq = y_seq[:split_seq], y_seq[split_seq:]
X_tr_seq = np.expand_dims(X_tr_seq, -1)
X_te_seq = np.expand_dims(X_te_seq, -1)

# safety checks against length/shape mismatches
assert len(X_tr_seq) == len(y_tr_seq), f'Train len mismatch: {len(X_tr_seq)} vs {len(y_tr_seq)}'
assert len(X_te_seq) == len(y_te_seq), f'Test len mismatch: {len(X_te_seq)} vs {len(y_te_seq)}'

# ==============================
# TensorFlow LSTM (with optional Optuna tuning)
# ==============================
def build_tf_model(n_features, hidden=64, layers_n=1, dropout=0.0, lr=1e-3):
    model = models.Sequential()
    for i in range(layers_n-1):
        model.add(layers.LSTM(hidden, return_sequences=True))
        if dropout>0: model.add(layers.Dropout(dropout))
    model.add(layers.LSTM(hidden))
    if dropout>0: model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

if RUN_TF:
    best_params_tf = dict(hidden=64, layers=1, dropout=0.0, lr=1e-3, batch=64, epochs=50)
    if DO_TUNE_TF and _HAS_OPTUNA:
        def objective(trial):
            hidden = trial.suggest_int("hidden", 32, 256, step=32)
            layers_n = trial.suggest_int("layers", 1, 3)
            dropout = trial.suggest_float("dropout", 0.0, 0.4)
            lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            batch = trial.suggest_categorical("batch", [32, 64, 128])
            epochs = trial.suggest_int("epochs", 10, 40)
            model = build_tf_model(n_features=1, hidden=hidden, layers_n=layers_n, dropout=dropout, lr=lr)
            es = callbacks.EarlyStopping(patience=8, restore_best_weights=True)
            hist = model.fit(X_tr_seq, y_tr_seq, validation_split=0.2, epochs=epochs, batch_size=batch, verbose=0, callbacks=[es])
            return min(hist.history["val_loss"])
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        bp = study.best_params
        best_params_tf.update({"hidden":bp["hidden"],"layers":bp["layers"],"dropout":bp["dropout"],"lr":bp["lr"],"batch":bp["batch"],"epochs":bp["epochs"]})
        print("Best TF params (Optuna):", best_params_tf)
    elif DO_TUNE_TF and not _HAS_OPTUNA:
        print("Optuna não encontrado para TF. Usando defaults.")

    model_tf = build_tf_model(1, best_params_tf["hidden"], best_params_tf["layers"], best_params_tf["dropout"], best_params_tf["lr"])
    es = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    hist = model_tf.fit(X_tr_seq, y_tr_seq, validation_split=0.2, epochs=best_params_tf["epochs"], batch_size=best_params_tf["batch"], verbose=0, callbacks=[es])
    plot_history_tf(hist, "TF LSTM — Loss")
    pred_tf = model_tf.predict(X_te_seq, verbose=0).reshape(-1)
    assert pred_tf.shape[0] == y_te_seq.shape[0], f'Pred/test length mismatch: {pred_tf.shape[0]} vs {y_te_seq.shape[0]}'
    evaluation_regression(y_te_seq, pred_tf, "TF LSTM (scaled)")
    scale_close = sc_seq.scale_[0]
    mean_close  = sc_seq.mean_[0]
    y_te_real   = y_te_seq * scale_close + mean_close
    pred_tf_real= pred_tf   * scale_close + mean_close
    evaluation_regression(y_te_real, pred_tf_real, "TF LSTM (USD)")
    fig, ax = plt.subplots()
    ax.plot(y_te_real, label="true (USD)")
    ax.plot(pred_tf_real, label="pred (USD)")
    ax.set_title("TF LSTM — Test set (USD)")
    ax.legend()
    plt.show()

# ==============================
# PyTorch — baseline (optional)
# ==============================
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTM_PyTorch(nn.Module):
    def __init__(self, n_features, hidden=64, layers_n=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=layers_n, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)

def train_torch_model(X_train, y_train, X_val, y_val, epochs=10, lr=1e-3, batch_size=64, hidden=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SeqDataset(X_train, y_train)
    val_ds = SeqDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = LSTM_PyTorch(n_features=X_train.shape[-1], hidden=hidden).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_losses, val_losses = [], []
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            tl += loss.item()*len(xb)
        tl /= len(train_ds)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                vl += loss.item()*len(xb)
        vl /= len(val_ds)
        train_losses.append(tl)
        val_losses.append(vl)
    return model, train_losses, val_losses

if RUN_PT_BASELINE:
    model_t, tl, vl = train_torch_model(X_tr_seq, y_tr_seq, X_te_seq, y_te_seq, epochs=20, lr=1e-3, batch_size=64, hidden=64)
    fig, ax = plt.subplots()
    ax.plot(tl, label="train_loss")
    ax.plot(vl, label="val_loss")
    ax.set_title("PyTorch LSTM — Loss (baseline)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    plt.show()
    model_t.eval()
    with torch.no_grad():
        pred_t = model_t(torch.tensor(X_te_seq, dtype=torch.float32)).numpy()
    evaluation_regression(y_te_seq, pred_t, "Torch LSTM (scaled, baseline)")
    scale_close = sc_seq.scale_[0]
    mean_close  = sc_seq.mean_[0]
    y_te_real   = y_te_seq * scale_close + mean_close
    pred_t_real = pred_t * scale_close + mean_close
    evaluation_regression(y_te_real, pred_t_real, "Torch LSTM (USD, baseline)")
    fig, ax = plt.subplots()
    ax.plot(y_te_real, label="true (USD)")
    ax.plot(pred_t_real, label="pred (USD)")
    ax.set_title("Torch LSTM — Test set (USD, baseline)")
    ax.legend()
    plt.show()

# ==============================
# PyTorch — Optuna tuning
# ==============================
class LSTM_PT(nn.Module):
    def __init__(self, n_features=1, hidden=128, layers_n=2, dropout=0.2, bidir=False):
        super().__init__()
        self.bidir = bidir
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers_n,
            batch_first=True,
            dropout=(dropout if layers_n>1 else 0.0),
            bidirectional=bidir
        )
        out_dim = hidden * (2 if bidir else 1)
        self.fc = nn.Linear(out_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)

def train_eval_pt(params, X_train, y_train, X_val, y_val, init_state_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch   = params["batch"]
    hidden  = params["hidden"]
    layers  = params["layers"]
    dropout = params["dropout"]
    lr      = params["lr"]
    epochs  = params["epochs"]
    bidir   = params["bidir"]

    train_ds = SeqDataset(X_train, y_train)
    val_ds   = SeqDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False)

    model = LSTM_PT(
        n_features=X_train.shape[-1],
        hidden=hidden, layers_n=layers, dropout=dropout, bidir=bidir
    ).to(device)

    #Carrega pesos iniciais apenas se compatíveis
    if init_state_path:
        print(f"Carregando pesos iniciais de {init_state_path}")
        _ = load_weights_if_compatible(
            model, init_state_path, device, expect_hparams=params, verbose=True
        )

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=4, verbose=False
    )
    loss_fn = nn.MSELoss()
    best_val = float("inf"); best_state=None; patience=10; no_improve=0

    for ep in range(epochs):
        model.train(); tr_loss=0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            tr_loss += loss.item()*len(xb)
        tr_loss /= len(train_ds)

        model.eval(); va_loss=0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss += loss.item()*len(xb)
        va_loss /= len(val_ds)
        sched.step(va_loss)

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


if RUN_PT_TUNING:
    if not _HAS_OPTUNA:
        print("Optuna não encontrado. Instale com `%pip install optuna` para habilitar tuning PT.")
    else:
        val_split = int(len(X_tr_seq) * 0.8)
        X_tr_sub, X_val = X_tr_seq[:val_split], X_tr_seq[val_split:]
        y_tr_sub, y_val = y_tr_seq[:val_split], y_tr_seq[val_split:]
        best_params_pt = dict(hidden=192, layers=2, dropout=0.2, lr=5e-4, batch=32, epochs=50, bidir=False)
        def objective(trial):
            params = dict(
                hidden   = trial.suggest_int("hidden", 64, 384, step=32),
                layers   = trial.suggest_int("layers", 1, 3),
                dropout  = trial.suggest_float("dropout", 0.0, 0.4),
                lr       = trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                batch    = trial.suggest_categorical("batch", [32, 64, 128]),
                epochs   = trial.suggest_int("epochs", 20, 80),
                bidir    = trial.suggest_categorical("bidir", [False, True])
            )
            _, val_loss = train_eval_pt(params, X_tr_sub, y_tr_sub, X_val, y_val)
            return val_loss
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS_PT, show_progress_bar=False)
        bp = study.best_params
        best_params_pt.update({"hidden":bp["hidden"],"layers":bp["layers"],"dropout":bp["dropout"],"lr":bp["lr"],"batch":bp["batch"],"epochs":bp["epochs"],"bidir":bp["bidir"]})
        print("Best PT params (Optuna):", best_params_pt)
        final_model, _ = train_eval_pt(best_params_pt, X_tr_seq, y_tr_seq, X_te_seq, y_te_seq, init_state_path=MODEL_PATH)
        final_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            y_pred_scaled = final_model(torch.tensor(X_te_seq, dtype=torch.float32).to(device)).cpu().numpy()
        evaluation_regression(y_te_seq, y_pred_scaled, "PT LSTM (scaled, tuned)")
        scale_close = sc_seq.scale_[0]
        mean_close  = sc_seq.mean_[0]
        y_te_real   = y_te_seq * scale_close + mean_close
        y_pred_real = y_pred_scaled * scale_close + mean_close
        evaluation_regression(y_te_real, y_pred_real, "PT LSTM (USD, tuned)")
        fig, ax = plt.subplots()
        ax.plot(y_te_real, label="true (USD)")
        ax.plot(y_pred_real, label="pred (USD)")
        ax.set_title("PyTorch LSTM (tuned) — Test set (USD)")
        ax.legend()
        plt.show()
        
        print("Salvando modelo (bundle) e espelhos JSON...")
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # 1) monta o payload completo para o .pth (bundle único)
        bundle = {
            "state_dict": final_model.state_dict(),
            "hparams": {
                "hidden": int(best_params_pt["hidden"]),
                "layers": int(best_params_pt["layers"]),
                "dropout": float(best_params_pt["dropout"]),
                "bidir": bool(best_params_pt["bidir"]),
                # meta úteis (não usados no serving, mas bons p/ auditoria):
                "lr": float(best_params_pt["lr"]),
                "batch": int(best_params_pt["batch"]),
                "epochs": int(best_params_pt["epochs"]),
            },
            "scaler": {
                "mean": float(sc_seq.mean_[0]),
                "scale": float(sc_seq.scale_[0]),
            },
            "config": {
                "lookback": int(LOOKBACK),
                "horizon": int(HORIZON),
                "input_feature": "close"
            },
        }
        
        # 2) escrita atômica do .pth (evita arquivo corrompido se o processo for interrompido)
        with tempfile.NamedTemporaryFile(dir=MODEL_DIR, delete=False) as tmp:
            torch.save(bundle, tmp.name)
        tmp_pth = tmp.name
        shutil.move(tmp_pth, MODEL_PATH)  # rename atômico na maioria dos FS
        
        # 3) JSONs espelho (legíveis para debug/observabilidade)
        SCALER_PATH = os.path.join(MODEL_DIR, "scaler.json")
        CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
        
        def _atomic_write_json(path, obj):
            with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(path), delete=False) as f:
                json.dump(obj, f)
                tmpname = f.name
            shutil.move(tmpname, path)
        
        _atomic_write_json(SCALER_PATH, bundle["scaler"])
        # opcionalmente, incluo os hparams dentro do config para ficar tudo num lugar legível
        cfg_pretty = {**bundle["config"], **bundle["hparams"]}
        _atomic_write_json(CONFIG_PATH, cfg_pretty)
        
        print("Modelo (.pth bundle) e JSONs salvos em outputs/model/.")


