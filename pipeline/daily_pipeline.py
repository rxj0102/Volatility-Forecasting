#!/usr/bin/env python3
"""
Automated daily volatility forecasting pipeline for SPY.

Workflow
--------
1. Download the latest 5 years of SPY OHLCV from yfinance.
2. Compute Garman-Klass, Parkinson, and close-to-close realized vol estimators.
3. Refit GJR-GARCH(1,1,1)-t on the full available history for the next-day σ forecast.
4. Retrain XGBoost (log-vol target) with rolling-window cross-validation (no lookahead).
5. Blend GARCH and XGBoost forecasts with QLIKE-optimal convex weights.
6. Append the forecast, date, and realized vol (when observable) to outputs/forecast_log.csv.
7. Log all steps at INFO level to outputs/pipeline.log.

Usage
-----
    python -m pipeline.daily_pipeline                     # run for the latest available date
    python -m pipeline.daily_pipeline --date 2025-01-15   # run as-of a specific date
    python -m pipeline.daily_pipeline --split IS          # label forecast as in-sample
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from arch import arch_model
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_engineering import build_lookback_windows, walk_forward_cv
from src.models import constrained_weights

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUTS_DIR = ROOT / "outputs"
FORECAST_LOG = OUTPUTS_DIR / "forecast_log.csv"
PIPELINE_LOG = OUTPUTS_DIR / "pipeline.log"

# ── Hyperparameters ────────────────────────────────────────────────────────────
LOOKBACK: int = 20        # days of history per XGBoost sample
N_SPLITS: int = 5         # walk-forward CV folds
EMBARGO: int = 5          # days purged between train and validation (matches RV5 horizon)
REFIT_FREQ: int = 21      # GARCH refit cadence (trading days)
BURNIN: int = 252         # minimum obs before first GARCH fit
YEARS_OF_DATA: int = 5    # history window downloaded from yfinance
SEED: int = 42

FEATURE_COLS: list[str] = ["GK_Vol", "Park_Vol", "CC_Vol", "Log_Ret", "GARCH_sigma"]
TARGET_COL: str = "Target_RV5"

CSV_COLUMNS: list[str] = [
    "forecast_as_of",    # last date with available market data
    "target_date",       # next business day (the date being forecast)
    "garch_forecast",    # GJR-GARCH(1,1,1)-t one-step-ahead annualised vol
    "xgb_forecast",      # XGBoost annualised vol forecast
    "ensemble_forecast", # QLIKE-optimal convex blend
    "realized_rv5",      # actual 5-day rolling vol on target_date (back-filled)
    "split",             # 'IS' (in-sample) or 'OOS' (out-of-sample)
    "run_timestamp",     # UTC timestamp of pipeline execution
]


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("daily_pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(PIPELINE_LOG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_spy_data(years: int = YEARS_OF_DATA) -> pd.DataFrame:
    """Download SPY OHLCV for the past *years* years from yfinance."""
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=years)
    raw = yf.download(
        "SPY",
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    # Flatten potential MultiIndex columns produced by newer yfinance versions
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).capitalize() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index)
    raw = raw.dropna(subset=["Open", "High", "Low", "Close"])
    return raw


# ── Realized Vol Estimators ───────────────────────────────────────────────────

def compute_rv_estimators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log-returns and three realized vol estimators, then build the target.

    Added columns
    -------------
    Log_Ret     : daily log-return ln(C_t / C_{t-1})
    GK_Vol      : Garman-Klass annualised vol (BLUE under GBM; ~7× lower variance)
    Park_Vol    : Parkinson annualised vol (range-only; efficient intraday proxy)
    CC_Vol      : 5-day close-to-close rolling std, annualised
    RV5         : 5-day realised vol (identical formula to CC_Vol, kept separately
                  for easy back-filling of realized_rv5 in the forecast log)
    Target_RV5  : RV5.shift(-1) — next-day 5-day realised vol (the forecast target)
    """
    out = df.copy()

    log_ret = np.log(out["Close"] / out["Close"].shift(1))
    out["Log_Ret"] = log_ret

    hl = np.log(out["High"] / out["Low"])
    co = np.log(out["Close"] / out["Open"])
    out["GK_Vol"] = np.sqrt(np.maximum(0.5 * hl**2 - (2 * np.log(2) - 1) * co**2, 0.0)) * np.sqrt(252)
    out["Park_Vol"] = np.sqrt(hl**2 / (4 * np.log(2))) * np.sqrt(252)
    out["CC_Vol"] = log_ret.rolling(5).std() * np.sqrt(252)
    out["RV5"] = log_ret.rolling(5).std() * np.sqrt(252)
    out["Target_RV5"] = out["RV5"].shift(-1)

    return out


# ── GJR-GARCH Features ────────────────────────────────────────────────────────

def fit_gjr_garch_expanding(
    log_ret: pd.Series,
    refit_freq: int = REFIT_FREQ,
    burnin: int = BURNIN,
) -> pd.Series:
    """
    Compute σ̂_{t|t-1} for every trading day using an expanding-window GJR-GARCH.

    The model is refitted every *refit_freq* days to mimic a production schedule.
    Returns NaN for the first *burnin* days.  σ̂_{t|t-1} at index t is the
    one-step-ahead forecast derived from data through t-1, preventing any
    same-day look-ahead bias when this series is used as an XGBoost feature.
    """
    n = len(log_ret)
    sigma = np.full(n, np.nan)
    last_res = None

    for t in range(burnin, n):
        if (t - burnin) % refit_freq == 0:
            ret_scaled = log_ret.iloc[:t].dropna() * 100.0  # scale for ARCH numerics
            if len(ret_scaled) < burnin:
                continue
            try:
                am = arch_model(ret_scaled, vol="Garch", p=1, o=1, q=1, dist="t", mean="Zero")
                last_res = am.fit(disp="off", show_warning=False)
            except Exception:
                pass

        if last_res is not None:
            try:
                fc = last_res.forecast(horizon=1, reindex=False)
                sigma[t] = float(np.sqrt(fc.variance.values[-1, 0])) / 100.0
            except Exception:
                pass

    return pd.Series(sigma, index=log_ret.index, name="GARCH_sigma")


def fit_gjr_garch_final(log_ret: pd.Series) -> float:
    """
    Fit GJR-GARCH(1,1,1)-t on the full series and return the next one-step-ahead σ.

    This is used to produce the standalone GARCH component of the ensemble.
    The `o=1` parameter adds the GJR asymmetric term (negative-return leverage effect).
    """
    ret_scaled = log_ret.dropna() * 100.0
    am = arch_model(ret_scaled, vol="Garch", p=1, o=1, q=1, dist="t", mean="Zero")
    res = am.fit(disp="off", show_warning=False)
    fc = res.forecast(horizon=1, reindex=False)
    return float(np.sqrt(fc.variance.values[-1, 0])) / 100.0


# ── Feature Matrix ────────────────────────────────────────────────────────────

def build_feature_df(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """
    Return a clean DataFrame with FEATURE_COLS (and optionally TARGET_COL).

    Rows with any NaN in the required columns are dropped.
    """
    required = FEATURE_COLS + ([TARGET_COL] if require_target else [])
    available = [c for c in required if c in df.columns]
    return df[available].dropna(subset=available)


# ── XGBoost Training ──────────────────────────────────────────────────────────

def _make_xgb() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )


def train_xgboost_with_cv(
    feat_df: pd.DataFrame,
    lookback: int = LOOKBACK,
    n_splits: int = N_SPLITS,
    embargo: int = EMBARGO,
) -> tuple[xgb.XGBRegressor, RobustScaler, float]:
    """
    Train XGBoost on log(Target_RV5) with walk-forward cross-validation.

    The log-vol target ensures strictly positive inverse-transformed forecasts.
    RobustScaler is fit on the full training set (no future data in this path
    because *feat_df* already contains only data available as of the run date).

    Parameters
    ----------
    feat_df : DataFrame with FEATURE_COLS + TARGET_COL, no NaNs.
    lookback : int   — rolling window width per sample.
    n_splits : int   — number of walk-forward CV folds.
    embargo  : int   — rows purged between train and val to avoid label overlap.

    Returns
    -------
    (fitted XGBRegressor, fitted RobustScaler, mean OOF QLIKE across folds)
    """
    X, y, _ = build_lookback_windows(feat_df, FEATURE_COLS, TARGET_COL, lookback)
    y_log = np.log(np.maximum(y, 1e-10))

    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X)

    folds = walk_forward_cv(X_sc, y_log, n_splits=n_splits, embargo=embargo)
    fold_qlikes: list[float] = []
    for tr_idx, val_idx in folds:
        m = _make_xgb()
        m.fit(X_sc[tr_idx], y_log[tr_idx])
        pred_vol = np.exp(m.predict(X_sc[val_idx]))
        y_val = y[val_idx]
        ratio = np.maximum(y_val, 1e-10) / np.maximum(pred_vol, 1e-10)
        fold_qlikes.append(float(np.mean(ratio - np.log(ratio) - 1)))

    final = _make_xgb()
    final.fit(X_sc, y_log)
    return final, scaler, float(np.mean(fold_qlikes))


# ── Inference ─────────────────────────────────────────────────────────────────

def get_xgb_forecast(
    feat_df: pd.DataFrame,
    model: xgb.XGBRegressor,
    scaler: RobustScaler,
    lookback: int = LOOKBACK,
) -> float:
    """
    Build a single inference row from the last *lookback* rows of *feat_df*
    and return the forecast in vol-space (annualised).
    """
    if len(feat_df) < lookback:
        raise ValueError(f"Need at least {lookback} rows; got {len(feat_df)}")
    window = feat_df[FEATURE_COLS].tail(lookback).to_numpy(dtype=float)
    x_sc = scaler.transform(window.ravel().reshape(1, -1))
    return float(np.exp(model.predict(x_sc)[0]))


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def _init_forecast_log() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if not FORECAST_LOG.exists():
        with FORECAST_LOG.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def append_forecast(row: dict) -> None:
    """Append a single forecast row to *forecast_log.csv*."""
    _init_forecast_log()
    with FORECAST_LOG.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)


def update_realized_vol(df: pd.DataFrame) -> None:
    """
    Back-fill *realized_rv5* for rows in *forecast_log.csv* whose *target_date*
    is now in the past and whose RV5 value is available in *df*.
    """
    if not FORECAST_LOG.exists() or "RV5" not in df.columns:
        return
    log_df = pd.read_csv(FORECAST_LOG, parse_dates=["target_date"])
    rv5 = df["RV5"]
    changed = False
    for i, row in log_df.iterrows():
        td = row["target_date"]
        if pd.isna(row.get("realized_rv5", np.nan)) and td in rv5.index:
            val = rv5.loc[td]
            if not (isinstance(val, float) and np.isnan(val)):
                log_df.at[i, "realized_rv5"] = round(float(val), 8)
                changed = True
    if changed:
        log_df.to_csv(FORECAST_LOG, index=False)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    as_of_date: Optional[str] = None,
    split: str = "OOS",
    years: int = YEARS_OF_DATA,
) -> dict:
    """
    Run the full daily forecasting pipeline end-to-end.

    Parameters
    ----------
    as_of_date : ISO date string (YYYY-MM-DD) or None.
        When None the pipeline uses the latest available trading day from yfinance.
        Pass a date to replay a historical run (used by backfill.py).
    split : 'IS' or 'OOS'.  Written to *forecast_log.csv*.
    years : years of OHLCV history to download.

    Returns
    -------
    dict with keys matching CSV_COLUMNS (excluding *realized_rv5* which is NaN
    at run time and back-filled on subsequent runs).
    """
    logger = _setup_logging()
    logger.info("=== Daily Pipeline Start  as_of=%s  split=%s ===", as_of_date or "latest", split)

    # 1 ── Download data ────────────────────────────────────────────────────────
    logger.info("Downloading %d years of SPY OHLCV", years)
    df = load_spy_data(years=years)
    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date)
        df = df.loc[df.index <= cutoff]
        if df.empty:
            raise ValueError(f"No SPY data on or before {as_of_date}")
    logger.info("  rows=%d  last_date=%s", len(df), df.index[-1].date())

    # 2 ── Realized vol estimators ──────────────────────────────────────────────
    logger.info("Computing GK / Parkinson / CC realized vol estimators")
    df = compute_rv_estimators(df)

    # 3 ── GJR-GARCH expanding features (no lookahead) ──────────────────────────
    logger.info("Building GJR-GARCH expanding-window features (refit every %d days)", REFIT_FREQ)
    df["GARCH_sigma"] = fit_gjr_garch_expanding(df["Log_Ret"])

    # 4 ── Final GJR-GARCH forecast for next day ────────────────────────────────
    logger.info("Fitting final GJR-GARCH on full history for next-day σ")
    garch_next = fit_gjr_garch_final(df["Log_Ret"])
    logger.info("  GARCH forecast = %.6f", garch_next)

    # 5 ── Build clean feature matrix ───────────────────────────────────────────
    feat_df = build_feature_df(df, require_target=True)
    logger.info("Feature matrix: %d rows × %d features", len(feat_df), len(FEATURE_COLS))
    if len(feat_df) < LOOKBACK + BURNIN:
        raise RuntimeError(
            f"Insufficient data: {len(feat_df)} rows (need ≥ {LOOKBACK + BURNIN})"
        )

    # 6 ── Train XGBoost with rolling-window CV ────────────────────────────────
    logger.info("Training XGBoost: %d-fold walk-forward CV  embargo=%d", N_SPLITS, EMBARGO)
    xgb_model, scaler, cv_qlike = train_xgboost_with_cv(feat_df)
    logger.info("  XGBoost OOF QLIKE = %.6f", cv_qlike)

    # 7 ── XGBoost next-day forecast ────────────────────────────────────────────
    # Use all feature rows including the latest (no target needed for inference)
    feat_df_inf = build_feature_df(df, require_target=False)
    xgb_next = get_xgb_forecast(feat_df_inf, xgb_model, scaler)
    logger.info("  XGBoost forecast  = %.6f", xgb_next)

    # 8 ── QLIKE-optimal ensemble weights ──────────────────────────────────────
    logger.info("Computing QLIKE-optimal ensemble weights on training data")
    X_tr, y_tr, _ = build_lookback_windows(feat_df, FEATURE_COLS, TARGET_COL, LOOKBACK)
    garch_tr = feat_df["GARCH_sigma"].iloc[LOOKBACK:].to_numpy()
    xgb_tr_preds = np.exp(xgb_model.predict(scaler.transform(X_tr)))
    valid = ~np.isnan(garch_tr) & (garch_tr > 0) & (y_tr > 0)
    if valid.sum() >= 20:
        w = constrained_weights(
            np.column_stack([garch_tr[valid], xgb_tr_preds[valid]]),
            y_tr[valid],
        )
    else:
        w = np.array([0.5, 0.5])
    ensemble_next = float(w[0] * garch_next + w[1] * xgb_next)
    logger.info("  Ensemble weights: GARCH=%.3f  XGB=%.3f", w[0], w[1])
    logger.info("  Ensemble forecast = %.6f", ensemble_next)

    # 9 ── Determine forecast dates ────────────────────────────────────────────
    last_date = df.index[-1]
    next_bday = last_date + pd.offsets.BDay(1)
    forecast_as_of = last_date.strftime("%Y-%m-%d")
    target_date_str = next_bday.strftime("%Y-%m-%d")

    # 10 ── Back-fill realized vol for previous rows ────────────────────────────
    logger.info("Back-filling realized_rv5 for observable past forecasts")
    update_realized_vol(df)

    # 11 ── Append new forecast ────────────────────────────────────────────────
    row = {
        "forecast_as_of": forecast_as_of,
        "target_date": target_date_str,
        "garch_forecast": round(garch_next, 8),
        "xgb_forecast": round(xgb_next, 8),
        "ensemble_forecast": round(ensemble_next, 8),
        "realized_rv5": "",
        "split": split,
        "run_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    append_forecast(row)
    logger.info("Forecast appended → %s", FORECAST_LOG)
    logger.info("=== Daily Pipeline Complete ===")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily SPY volatility forecasting pipeline")
    parser.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                        help="As-of date (defaults to latest available trading day)")
    parser.add_argument("--split", default="OOS", choices=["IS", "OOS"],
                        help="IS/OOS label written to forecast_log.csv")
    parser.add_argument("--years", type=int, default=YEARS_OF_DATA,
                        help="Years of history to download (default: 5)")
    args = parser.parse_args()
    result = run_pipeline(as_of_date=args.date, split=args.split, years=args.years)
    print(
        f"Forecast  as_of={result['forecast_as_of']}  "
        f"target={result['target_date']}  "
        f"ensemble={result['ensemble_forecast']:.6f}"
    )


if __name__ == "__main__":
    main()
