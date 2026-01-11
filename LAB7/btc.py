#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import List

import numpy as np
import pandas as pd
import joblib

from tensorflow import keras


# =============================
# Data + features (muszą być 1:1 jak w treningu)
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
                  harm_k: int, use_rolling: bool, rolling_window: int):
    prices = df[value_col].to_numpy(dtype=np.float32)
    eps = 1e-8
    logp = np.log(prices + eps)

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
    return X, r, prices

def infer_next_dates(last_date: pd.Timestamp, n: int) -> List[pd.Timestamp]:
    base = pd.Timestamp(last_date).tz_convert("UTC") if last_date.tzinfo else pd.Timestamp(last_date, tz="UTC")
    return [base + pd.Timedelta(days=i) for i in range(1, n + 1)]

def build_feature_row_from_return_and_date(
    r_t: float,
    dt: pd.Timestamp,
    harm_k: int,
    use_rolling: bool,
    rolling_window: int,
    recent_returns: List[float],
) -> np.ndarray:
    harm = calendar_year_month_harmonics(pd.Series([dt]), harm_k)[0]
    feats = [np.array([r_t], dtype=np.float32), harm.astype(np.float32)]
    if use_rolling:
        arr = np.array(recent_returns[-rolling_window:], dtype=np.float32)
        rm = float(arr.mean()) if len(arr) else 0.0
        rs = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        feats += [np.array([rm], dtype=np.float32), np.array([rs], dtype=np.float32)]
    return np.concatenate(feats).astype(np.float32)

def forecast_n_steps(
    model: keras.Model,
    scaler,
    df_hist: pd.DataFrame,
    date_col: str,
    value_col: str,
    window: int,
    harm_k: int,
    use_rolling: bool,
    rolling_window: int,
    n: int
) -> pd.DataFrame:
    df = df_hist.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)

    X_raw, returns, prices = make_features(df, date_col, value_col, harm_k, use_rolling, rolling_window)
    if len(X_raw) < window:
        raise ValueError(f"Za mało danych na window={window}. Masz {len(X_raw)} returnów.")

    last_date = df[date_col].iloc[-1]
    cur_price = float(prices[-1])

    recent_returns = returns.tolist()

    # start window
    X_seq_raw = X_raw[-window:]  # (window, d)
    d = X_seq_raw.shape[1]
    X_seq = scaler.transform(X_seq_raw.reshape(-1, d)).reshape(1, window, d).astype(np.float32)

    future_dates = infer_next_dates(pd.Timestamp(last_date), n)
    preds_r = []
    preds_price = []

    for dt in future_dates:
        r_hat = float(model.predict(X_seq, verbose=0).reshape(-1)[0])
        preds_r.append(r_hat)

        cur_price = float(cur_price * np.exp(r_hat))
        preds_price.append(cur_price)

        recent_returns.append(r_hat)
        x_new_raw = build_feature_row_from_return_and_date(
            r_t=r_hat,
            dt=dt,
            harm_k=harm_k,
            use_rolling=use_rolling,
            rolling_window=rolling_window,
            recent_returns=recent_returns
        )  # (d,)

        x_new = scaler.transform(x_new_raw.reshape(1, -1)).astype(np.float32)
        X_seq = np.concatenate([X_seq[:, 1:, :], x_new.reshape(1, 1, d)], axis=1)

    return pd.DataFrame({
        "date": future_dates,
        "pred_log_return": preds_r,
        "pred_price": preds_price
    })


# =============================
# CLI
# =============================

def parse_args():
    ap = argparse.ArgumentParser(description="BTC predict – N-step forecast using trained model")

    ap.add_argument("--file_path", required=True, help="CSV z historią (ten sam format co trening)")
    ap.add_argument("--out_dir", required=True, help="katalog z artefaktami z treningu")
    ap.add_argument("--result", default="yourluckynumbers.csv", help="plik wynikowy CSV")

    ap.add_argument("--date_col", default="Open time")
    ap.add_argument("--value_col", default="Close")

    ap.add_argument("--window", type=int, default=90)
    ap.add_argument("--n", type=int, default=10)

    ap.add_argument("--harm_k", type=int, default=2)
    ap.add_argument("--use_rolling", action="store_true")
    ap.add_argument("--rolling_window", type=int, default=20)

    ap.add_argument("--model", choices=["rnn", "dense"], default="rnn")

    return ap.parse_args()

def main():
    a = parse_args()

    meta_path = os.path.join(a.out_dir, "meta.json")
    scaler_path = os.path.join(a.out_dir, "scaler.pkl")
    model_path = os.path.join(a.out_dir, f"model_{a.model}.keras")

    for p in [meta_path, scaler_path, model_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Brak pliku: {p}")

    _ = json.load(open(meta_path, "r", encoding="utf-8"))  # tylko żeby sprawdzić, że istnieje
    scaler = joblib.load(scaler_path)
    model = keras.models.load_model(model_path)

    df = load_csv(a.file_path, a.date_col, a.value_col)

    pred = forecast_n_steps(
        model=model,
        scaler=scaler,
        df_hist=df,
        date_col=a.date_col,
        value_col=a.value_col,
        window=a.window,
        harm_k=a.harm_k,
        use_rolling=bool(a.use_rolling),
        rolling_window=a.rolling_window,
        n=a.n
    )

    pred.to_csv(a.result, index=False)
    print("[DONE] Saved:", a.result)
    print(pred.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
