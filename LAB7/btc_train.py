#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import keras_tuner as kt


# =============================
# Helpers
# =============================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)

def detect_compute() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    print("[INFO] TensorFlow:", tf.__version__)
    print("[INFO] GPUs:", gpus if gpus else "None (CPU only)")

def save_plot_learning(history: keras.callbacks.History, out_png: str, title: str) -> None:
    h = history.history
    plt.figure()
    plt.plot(h.get("loss", []), label="train_loss")
    plt.plot(h.get("val_loss", []), label="val_loss")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_plot_pred(y_true: np.ndarray, y_pred: np.ndarray, out_png: str, title: str, n: int = 400) -> None:
    n = min(n, len(y_true))
    plt.figure()
    plt.plot(y_true[:n], label="true")
    plt.plot(y_pred[:n], label="pred")
    plt.title(title)
    plt.xlabel("sample")
    plt.ylabel("log-return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def mse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return {"mse": mse, "mae": mae}


# =============================
# Config
# =============================

@dataclass
class Cfg:
    file_path: str
    out_dir: str

    date_col: str
    value_col: str

    window: int
    horizon: int

    train_ratio: float
    val_ratio: float

    epochs: int
    batch_size: int

    # features
    harm_k: int
    use_rolling: bool
    rolling_window: int

    # keras tuner
    tune: bool
    kt_max_trials: int
    kt_executions_per_trial: int
    kt_epochs: int

    # hp ranges provided by args
    rnn_type: str  # LSTM | GRU | both
    units_min: int
    units_max: int
    units_step: int
    layers_min: int
    layers_max: int
    dropout_min: float
    dropout_max: float
    dropout_step: float
    lr_min: float
    lr_max: float

    # dense baseline (fixed)
    dense_units: int
    dense_depth: int
    dense_dropout: float
    dense_lr: float

    seed: int


# =============================
# Data + features
# =============================

def load_csv(path: str, date_col: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise ValueError(f"Brak kolumny date_col='{date_col}'. Dostępne: {list(df.columns)}")
    if value_col not in df.columns:
        raise ValueError(f"Brak kolumny value_col='{value_col}'. Dostępne: {list(df.columns)}")

    df = df[[date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna().sort_values(date_col).reset_index(drop=True)
    return df

def calendar_year_month_harmonics(dt: pd.Series, k: int) -> np.ndarray:
    """
    Harmoniczne roczne + miesięczne.
    Zwraca (n, 4k): [sin/cos roczne dla i=1..k] + [sin/cos miesięczne dla i=1..k]
    """
    dt = pd.to_datetime(dt, errors="coerce", utc=True)
    if dt.isna().any():
        raise ValueError("Nie udało się sparsować dat do harmonicznych.")

    # --- roczne
    day_of_year = dt.dt.dayofyear.astype(np.float32)
    year = dt.dt.year.astype(np.int32)
    is_leap = ((year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))).astype(np.int32)
    days_in_year = (365 + is_leap).astype(np.float32)
    phase_year = (day_of_year - 1.0) / days_in_year  # 0..1

    # --- miesięczne
    day_of_month = dt.dt.day.astype(np.float32)
    days_in_month = dt.dt.days_in_month.astype(np.float32)
    phase_month = (day_of_month - 1.0) / days_in_month  # 0..1

    feats = []
    for i in range(1, k + 1):
        w = 2.0 * np.pi * i
        # roczne
        feats.append(np.sin(w * phase_year))
        feats.append(np.cos(w * phase_year))
        # miesięczne
        feats.append(np.sin(w * phase_month))
        feats.append(np.cos(w * phase_month))

    return np.vstack(feats).T.astype(np.float32)  # (n, 4k)


def make_features(df: pd.DataFrame, date_col: str, value_col: str,
                  harm_k: int, use_rolling: bool, rolling_window: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    prices = df[value_col].to_numpy(dtype=np.float32)
    eps = 1e-8
    logp = np.log(prices + eps)

    # target = log-return
    r = np.diff(logp).astype(np.float32)
    dt = df[date_col].iloc[1:].reset_index(drop=True)

    harm = calendar_year_month_harmonics(dt, harm_k)
    feats = [r.reshape(-1, 1), harm]

    if use_rolling:
        s = pd.Series(r)
        rm = s.rolling(rolling_window).mean().to_numpy(dtype=np.float32)
        rs = s.rolling(rolling_window).std().to_numpy(dtype=np.float32)
        feats += [np.nan_to_num(rm, nan=0.0).reshape(-1, 1),
                  np.nan_to_num(rs, nan=0.0).reshape(-1, 1)]

    X = np.concatenate(feats, axis=1).astype(np.float32)
    y = r.copy()

    meta = {
        "date_col": date_col,
        "value_col": value_col,
        "n_prices": int(len(prices)),
        "n_returns": int(len(r)),
        "feature_dim": int(X.shape[1]),
        "harm_k": int(harm_k),
        "use_rolling": bool(use_rolling),
        "rolling_window": int(rolling_window),
        "last_date": str(df[date_col].iloc[-1]),
        "last_price": float(prices[-1]),
    }
    return X, y, meta


# =============================
# Windowing + scaling
# =============================

def make_windows(X: np.ndarray, y: np.ndarray, window: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    if horizon < 1:
        raise ValueError("horizon musi być >= 1")
    xs, ys = [], []
    for t in range(window, len(X) - horizon + 1):
        xs.append(X[t - window:t])
        ys.append(y[t + horizon - 1])
    return np.stack(xs), np.array(ys, dtype=np.float32)

def split_tvt(Xw: np.ndarray, yw: np.ndarray, train_ratio: float, val_ratio: float):
    n = len(Xw)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train, y_train = Xw[:n_train], yw[:n_train]
    X_val, y_val = Xw[n_train:n_train + n_val], yw[n_train:n_train + n_val]
    X_test, y_test = Xw[n_train + n_val:], yw[n_train + n_val:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def scale_3d(train_X: np.ndarray, val_X: np.ndarray, test_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    _, _, d = train_X.shape
    scaler.fit(train_X.reshape(-1, d))

    def tr(a: np.ndarray) -> np.ndarray:
        flat = scaler.transform(a.reshape(-1, d))
        return flat.reshape(a.shape)

    return tr(train_X), tr(val_X), tr(test_X), scaler


# =============================
# Models
# =============================

def build_rnn_from_hp(hp: kt.HyperParameters, window: int, feature_dim: int, cfg: Cfg) -> keras.Model:
    if cfg.rnn_type == "both":
        rnn_type = hp.Choice("rnn_type", ["LSTM", "GRU"])
    else:
        rnn_type = cfg.rnn_type

    units = hp.Int("units", cfg.units_min, cfg.units_max, step=cfg.units_step)
    n_layers = hp.Int("layers", cfg.layers_min, cfg.layers_max, step=1)
    dropout = hp.Float("dropout", cfg.dropout_min, cfg.dropout_max, step=cfg.dropout_step)
    lr = hp.Float("lr", cfg.lr_min, cfg.lr_max, sampling="log")

    inp = keras.Input(shape=(window, feature_dim))
    x = inp
    for i in range(n_layers):
        ret_seq = (i < n_layers - 1)
        if rnn_type == "LSTM":
            x = layers.LSTM(units, return_sequences=ret_seq)(x)
        else:
            x = layers.GRU(units, return_sequences=ret_seq)(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)

    model = keras.Model(inp, out, name="rnn_forecaster")
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model

def build_dense_baseline(window: int, feature_dim: int, cfg: Cfg) -> keras.Model:
    inp = keras.Input(shape=(window, feature_dim))
    x = layers.Flatten()(inp)
    for _ in range(cfg.dense_depth):
        x = layers.Dense(cfg.dense_units, activation="relu")(x)
        if cfg.dense_dropout > 0:
            x = layers.Dropout(cfg.dense_dropout)(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out, name="dense_baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.dense_lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


# =============================
# Training
# =============================

def fit_model(model: keras.Model, X_train, y_train, X_val, y_val, cfg: Cfg) -> keras.callbacks.History:
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6),
    ]
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=1,
        callbacks=callbacks
    )
    return hist

def tune_rnn(X_train, y_train, X_val, y_val, cfg: Cfg) -> Tuple[keras.Model, Dict]:
    """
    KONIECZNE: max_trials to twardy limit => tuning się kończy.
    """
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_rnn_from_hp(hp, cfg.window, X_train.shape[-1], cfg),
        objective="val_loss",
        max_trials=cfg.kt_max_trials,
        executions_per_trial=cfg.kt_executions_per_trial,
        directory=os.path.join(cfg.out_dir, "kt"),
        project_name="rnn_search",
        overwrite=True,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.kt_epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = build_rnn_from_hp(best_hp, cfg.window, X_train.shape[-1], cfg)

    # final fit (pełne epochs)
    hist = fit_model(best_model, X_train, y_train, X_val, y_val, cfg)

    # zapis learning curve RNN
    save_plot_learning(
        hist,
        os.path.join(cfg.out_dir, "learning_rnn.png"),
        "RNN – learning curve (after tuning)"
    )

    info = {
        "best_hp": best_hp.values,
        "final_history": hist.history,          # pełna historia do meta.json
        "final_history_keys": list(hist.history.keys())
    }
    return best_model, info


def predict_1(model: keras.Model, X: np.ndarray) -> np.ndarray:
    return model.predict(X, verbose=0).reshape(-1)

def dump_test_preds(model: keras.Model, X_test, y_test, out_csv: str) -> Dict[str, float]:
    yp = predict_1(model, X_test)
    pd.DataFrame({"y_true": y_test, "y_pred": yp}).to_csv(out_csv, index=False)
    return mse_mae(y_test, yp)


# =============================
# CLI
# =============================

def parse_args() -> Cfg:
    ap = argparse.ArgumentParser(description="BTC train – RNN + Dense + optional Keras Tuner")

    ap.add_argument("--file_path", required=True, help="CSV z historią")
    ap.add_argument("--out_dir", default="runs/btc", help="katalog na artefakty")

    ap.add_argument("--date_col", default="Open time")
    ap.add_argument("--value_col", default="Close")

    ap.add_argument("--window", type=int, default=90)
    ap.add_argument("--horizon", type=int, default=1)

    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)

    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--harm_k", type=int, default=2)
    ap.add_argument("--use_rolling", action="store_true")
    ap.add_argument("--rolling_window", type=int, default=20)

    ap.add_argument("--tune", action="store_true", help="włącz Keras Tuner dla RNN")
    ap.add_argument("--kt_max_trials", type=int, default=10)
    ap.add_argument("--kt_executions_per_trial", type=int, default=1)
    ap.add_argument("--kt_epochs", type=int, default=6, help="ile epok na trial (żeby było szybko)")

    # ranges from args
    ap.add_argument("--rnn_type", choices=["LSTM", "GRU", "both"], default="both")
    ap.add_argument("--units_min", type=int, default=32)
    ap.add_argument("--units_max", type=int, default=128)
    ap.add_argument("--units_step", type=int, default=32)
    ap.add_argument("--layers_min", type=int, default=1)
    ap.add_argument("--layers_max", type=int, default=2)
    ap.add_argument("--dropout_min", type=float, default=0.0)
    ap.add_argument("--dropout_max", type=float, default=0.3)
    ap.add_argument("--dropout_step", type=float, default=0.1)
    ap.add_argument("--lr_min", type=float, default=1e-4)
    ap.add_argument("--lr_max", type=float, default=3e-3)

    # dense baseline
    ap.add_argument("--dense_units", type=int, default=128)
    ap.add_argument("--dense_depth", type=int, default=2)
    ap.add_argument("--dense_dropout", type=float, default=0.2)
    ap.add_argument("--dense_lr", type=float, default=1e-3)

    ap.add_argument("--seed", type=int, default=42)

    a = ap.parse_args()

    return Cfg(
        file_path=a.file_path,
        out_dir=a.out_dir,

        date_col=a.date_col,
        value_col=a.value_col,

        window=a.window,
        horizon=a.horizon,

        train_ratio=a.train_ratio,
        val_ratio=a.val_ratio,

        epochs=a.epochs,
        batch_size=a.batch_size,

        harm_k=a.harm_k,
        use_rolling=bool(a.use_rolling),
        rolling_window=a.rolling_window,

        tune=bool(a.tune),
        kt_max_trials=a.kt_max_trials,
        kt_executions_per_trial=a.kt_executions_per_trial,
        kt_epochs=a.kt_epochs,

        rnn_type=a.rnn_type,
        units_min=a.units_min,
        units_max=a.units_max,
        units_step=a.units_step,
        layers_min=a.layers_min,
        layers_max=a.layers_max,
        dropout_min=a.dropout_min,
        dropout_max=a.dropout_max,
        dropout_step=a.dropout_step,
        lr_min=a.lr_min,
        lr_max=a.lr_max,

        dense_units=a.dense_units,
        dense_depth=a.dense_depth,
        dense_dropout=a.dense_dropout,
        dense_lr=a.dense_lr,

        seed=a.seed,
    )


def main():
    cfg = parse_args()
    ensure_dir(cfg.out_dir)
    set_seed(cfg.seed)
    detect_compute()

    df = load_csv(cfg.file_path, cfg.date_col, cfg.value_col)
    X_raw, y_raw, meta = make_features(df, cfg.date_col, cfg.value_col, cfg.harm_k, cfg.use_rolling, cfg.rolling_window)

    Xw, yw = make_windows(X_raw, y_raw, cfg.window, cfg.horizon)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_tvt(Xw, yw, cfg.train_ratio, cfg.val_ratio)
    X_train, X_val, X_test, scaler = scale_3d(X_train, X_val, X_test)

    # RNN
    kt_info: Optional[Dict] = None
    if cfg.tune:
        model_rnn, kt_info = tune_rnn(X_train, y_train, X_val, y_val, cfg)
    else:
        # minimal: bierzemy units_min, layers_min, lr_max (prosto)
        hp = kt.HyperParameters()
        if cfg.rnn_type == "both":
            hp.Fixed("rnn_type", "GRU")
        else:
            hp.Fixed("rnn_type", cfg.rnn_type)
        hp.Fixed("units", cfg.units_min)
        hp.Fixed("layers", cfg.layers_min)
        hp.Fixed("dropout", cfg.dropout_min)
        hp.Fixed("lr", cfg.lr_max)
        model_rnn = build_rnn_from_hp(hp, cfg.window, X_train.shape[-1], cfg)
        hist_rnn = fit_model(model_rnn, X_train, y_train, X_val, y_val, cfg)
        save_plot_learning(hist_rnn, os.path.join(cfg.out_dir, "learning_rnn.png"), "RNN – learning curve")

    # Dense baseline
    model_dense = build_dense_baseline(cfg.window, X_train.shape[-1], cfg)
    hist_dense = fit_model(model_dense, X_train, y_train, X_val, y_val, cfg)
    save_plot_learning(hist_dense, os.path.join(cfg.out_dir, "learning_dense.png"), "Dense – learning curve")

    # Test preds + metrics + plots
    rnn_csv = os.path.join(cfg.out_dir, "test_pred_rnn.csv")
    dense_csv = os.path.join(cfg.out_dir, "test_pred_dense.csv")

    rnn_metrics = dump_test_preds(model_rnn, X_test, y_test, rnn_csv)
    dense_metrics = dump_test_preds(model_dense, X_test, y_test, dense_csv)

    rnn_df = pd.read_csv(rnn_csv)
    dense_df = pd.read_csv(dense_csv)
    save_plot_pred(rnn_df["y_true"].to_numpy(), rnn_df["y_pred"].to_numpy(),
                   os.path.join(cfg.out_dir, "test_pred_rnn.png"), "RNN – test predictions")
    save_plot_pred(dense_df["y_true"].to_numpy(), dense_df["y_pred"].to_numpy(),
                   os.path.join(cfg.out_dir, "test_pred_dense.png"), "Dense – test predictions")

    # Save artifacts
    model_rnn.save(os.path.join(cfg.out_dir, "model_rnn.keras"))
    model_dense.save(os.path.join(cfg.out_dir, "model_dense.keras"))
    joblib.dump(scaler, os.path.join(cfg.out_dir, "scaler.pkl"))

    pack = {
        "cfg": asdict(cfg),
        "meta": meta,
        "metrics": {"rnn": rnn_metrics, "dense": dense_metrics},
        "kt_info": kt_info,
    }
    with open(os.path.join(cfg.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)

    print("\n[DONE]")
    print("RNN test:", rnn_metrics)
    print("Dense test:", dense_metrics)
    if kt_info:
        print("KT best_hp:", kt_info.get("best_hp"))
    print("Saved to:", cfg.out_dir)


if __name__ == "__main__":
    main()
