#!/usr/bin/env python3
"""
Backfill forecast_log.csv over the last 2 years of trading days.

IS / OOS labelling
------------------
The backfill window is split in half:

  IS (in-sample)     — forecasts in the *first* year of the 2-year window.
                       These dates fall within the approximate training period
                       used by the research notebook, so the model may have
                       been calibrated on some of the same data.

  OOS (out-of-sample) — forecasts in the *second* (more recent) year.
                        These dates are strictly beyond the notebook's training
                        window, giving a clean evaluation of live performance.

Design
------
* GJR-GARCH features (σ̂_{t|t-1}) are precomputed *once* on the full history
  using an expanding window — calling fit_gjr_garch_expanding repeatedly would
  be O(n²) and unnecessary because the series is already bias-free by design.
* XGBoost is retrained every *xgb_refit_freq* trading days (default 21) on
  all data available before each target date, giving a true walk-forward
  evaluation without lookahead.
* For each backfill date the realized_rv5 is populated immediately (since it
  is a historical date and the RV5 is already observable).

Usage
-----
    python -m pipeline.backfill
    python -m pipeline.backfill --backfill-years 2 --history-years 5 --xgb-refit-freq 21
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.daily_pipeline import (
    BURNIN,
    EMBARGO,
    FEATURE_COLS,
    LOOKBACK,
    N_SPLITS,
    REFIT_FREQ,
    TARGET_COL,
    YEARS_OF_DATA,
    _init_forecast_log,
    _setup_logging,
    append_forecast,
    build_feature_df,
    compute_rv_estimators,
    fit_gjr_garch_expanding,
    get_xgb_forecast,
    load_spy_data,
    train_xgboost_with_cv,
)
from src.feature_engineering import build_lookback_windows
from src.models import constrained_weights

BACKFILL_YEARS: int = 2


def run_backfill(
    backfill_years: int = BACKFILL_YEARS,
    history_years: int = YEARS_OF_DATA,
    xgb_refit_freq: int = REFIT_FREQ,
) -> None:
    """
    Simulate the daily pipeline over the last *backfill_years* years.

    For every trading day in the backfill window the function:
      1. Slices all feature data available *strictly before* that day.
      2. Retrains XGBoost (every *xgb_refit_freq* days for speed).
      3. Reads the precomputed GARCH σ̂_{t|t-1} for that day.
      4. Blends with QLIKE-optimal weights.
      5. Appends the row (including *realized_rv5*) to forecast_log.csv.

    Parameters
    ----------
    backfill_years  : number of years to backfill (default 2).
    history_years   : total history to download for feature computation.
    xgb_refit_freq  : how often (in trading days) to retrain XGBoost.
    """
    logger = _setup_logging()
    logger.info("=== Backfill Start  years=%d ===", backfill_years)
    _init_forecast_log()

    # ── 1. Download and compute features on full history ──────────────────────
    logger.info("Downloading %d years of SPY OHLCV", history_years)
    df_raw = load_spy_data(years=history_years)
    df_raw = compute_rv_estimators(df_raw)

    logger.info("Computing GJR-GARCH expanding features on full dataset")
    df_raw["GARCH_sigma"] = fit_gjr_garch_expanding(df_raw["Log_Ret"])

    # Full feature df (rows where every feature AND target is non-NaN)
    feat_full = build_feature_df(df_raw, require_target=True)
    # Full feature df without target requirement (for inference feature vectors)
    feat_inf_full = build_feature_df(df_raw, require_target=False)
    logger.info("Feature matrix: %d rows", len(feat_full))

    # ── 2. Determine backfill window and IS/OOS cutoff ────────────────────────
    last_date: pd.Timestamp = feat_full.index[-1]
    backfill_start: pd.Timestamp = last_date - pd.DateOffset(years=backfill_years)
    # The IS/OOS boundary sits at the midpoint of the backfill window
    is_oos_boundary: pd.Timestamp = last_date - pd.DateOffset(years=backfill_years // 2)

    backfill_dates = feat_full.index[feat_full.index >= backfill_start]
    logger.info(
        "Backfill window: %s → %s  (%d trading days)",
        backfill_dates[0].date(),
        backfill_dates[-1].date(),
        len(backfill_dates),
    )
    logger.info(
        "IS boundary: %s  (dates before → IS, on or after → OOS)",
        is_oos_boundary.date(),
    )

    # ── 3. Walk-forward loop ──────────────────────────────────────────────────
    xgb_model: xgb.XGBRegressor | None = None
    scaler: RobustScaler | None = None
    last_fit_step: int = -(xgb_refit_freq + 1)  # force fit on first iteration

    for step, target_date in enumerate(backfill_dates):
        # All feature/target rows strictly before target_date (train slice)
        slice_train = feat_full.loc[feat_full.index < target_date]
        # All feature rows (no target required) strictly before target_date (inference slice)
        slice_inf = feat_inf_full.loc[feat_inf_full.index < target_date]

        min_rows = LOOKBACK + BURNIN + 10
        if len(slice_train) < min_rows:
            logger.debug("Skip %s — only %d rows available", target_date.date(), len(slice_train))
            continue

        # Retrain XGBoost on schedule
        if step - last_fit_step >= xgb_refit_freq:
            logger.info(
                "Retraining XGBoost on %d rows  (target_date=%s)",
                len(slice_train),
                target_date.date(),
            )
            try:
                xgb_model, scaler, cv_qlike = train_xgboost_with_cv(
                    slice_train, lookback=LOOKBACK, n_splits=N_SPLITS, embargo=EMBARGO
                )
                logger.info("  OOF QLIKE = %.6f", cv_qlike)
                last_fit_step = step
            except Exception as exc:
                logger.warning("XGBoost training failed for %s: %s", target_date.date(), exc)
                continue

        if xgb_model is None or scaler is None:
            continue

        # GARCH forecast for target_date = σ̂_{target_date|target_date-1}
        # This is stored at index target_date in df_raw["GARCH_sigma"]
        if target_date not in df_raw.index:
            logger.debug("Skip %s — not in raw index", target_date.date())
            continue
        garch_next = float(df_raw.loc[target_date, "GARCH_sigma"])
        if np.isnan(garch_next) or garch_next <= 0:
            logger.debug("Skip %s — GARCH sigma unavailable", target_date.date())
            continue

        # XGBoost forecast using features available before target_date
        try:
            xgb_next = get_xgb_forecast(slice_inf, xgb_model, scaler)
        except Exception as exc:
            logger.warning("XGBoost inference failed for %s: %s", target_date.date(), exc)
            continue

        # QLIKE-optimal ensemble weights on the training slice
        try:
            X_tr, y_tr, _ = build_lookback_windows(slice_train, FEATURE_COLS, TARGET_COL, LOOKBACK)
            garch_tr = slice_train["GARCH_sigma"].iloc[LOOKBACK:].to_numpy()
            xgb_tr_pred = np.exp(xgb_model.predict(scaler.transform(X_tr)))
            valid = ~np.isnan(garch_tr) & (garch_tr > 0) & (y_tr > 0)
            w = (
                constrained_weights(
                    np.column_stack([garch_tr[valid], xgb_tr_pred[valid]]),
                    y_tr[valid],
                )
                if valid.sum() >= 20
                else np.array([0.5, 0.5])
            )
        except Exception:
            w = np.array([0.5, 0.5])

        ensemble_next = float(w[0] * garch_next + w[1] * xgb_next)

        # Realized RV5 — available because target_date is historical
        realized = ""
        if target_date in df_raw.index and "RV5" in df_raw.columns:
            rv_val = df_raw.loc[target_date, "RV5"]
            if not np.isnan(rv_val):
                realized = round(float(rv_val), 8)

        # IS/OOS label: IS for the first half of the backfill window
        split = "IS" if target_date < is_oos_boundary else "OOS"

        forecast_as_of = (target_date - pd.offsets.BDay(1)).strftime("%Y-%m-%d")
        row = {
            "forecast_as_of": forecast_as_of,
            "target_date": target_date.strftime("%Y-%m-%d"),
            "garch_forecast": round(garch_next, 8),
            "xgb_forecast": round(xgb_next, 8),
            "ensemble_forecast": round(ensemble_next, 8),
            "realized_rv5": realized,
            "split": split,
            "run_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        append_forecast(row)

        if (step + 1) % 50 == 0:
            logger.info("Progress: %d / %d days processed", step + 1, len(backfill_dates))

    logger.info("=== Backfill Complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill forecast_log.csv with historical IS/OOS predictions"
    )
    parser.add_argument("--backfill-years", type=int, default=BACKFILL_YEARS,
                        help="How many years to backfill (default: 2)")
    parser.add_argument("--history-years", type=int, default=YEARS_OF_DATA,
                        help="Total history to download (default: 5)")
    parser.add_argument("--xgb-refit-freq", type=int, default=REFIT_FREQ,
                        help="Days between XGBoost retrains (default: 21)")
    args = parser.parse_args()
    run_backfill(
        backfill_years=args.backfill_years,
        history_years=args.history_years,
        xgb_refit_freq=args.xgb_refit_freq,
    )


if __name__ == "__main__":
    main()
