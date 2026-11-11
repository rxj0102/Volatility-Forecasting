"""
Model helpers for the volatility forecasting pipeline.

Provides:
  - har_rv               : HAR-RV expanding-window predictor (Corsi 2009)
  - constrained_weights  : QLIKE-optimal convex combination of base forecasts
  - get_monotone_constraints : XGBoost monotone constraint vector for
                               flattened lookback features

References
----------
Corsi (2009)            : A simple approximate long-memory model of realized
                          volatility. J. Financial Econometrics 7, 174–196.
Müller et al. (1997)    : Volatilities of different time resolutions.
                          J. Empirical Finance 4, 213–239.
Patton (2011)           : Volatility forecast comparison using imperfect proxies.
                          J. Econometrics 160, 246–256.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def har_rv(
    rv: pd.Series,
    burnin: int = 252,
    daily_lag: int = 1,
    weekly_lag: int = 5,
    monthly_lag: int = 22,
) -> pd.Series:
    """
    Expanding-window HAR-RV predictor (Corsi 2009).

    The Heterogeneous Autoregressive model for Realized Volatility decomposes
    the volatility signal into three components operated at different time scales:

        RV_{t+1} = c + β_D·RV^(d)_t + β_W·RV^(w)_t + β_M·RV^(m)_t + ε_{t+1}

    where:
      - RV^(d)_t = RV_t                              (daily component)
      - RV^(w)_t = (1/5)·ΣRV_{t-j}, j=0..4          (weekly component)
      - RV^(m)_t = (1/22)·ΣRV_{t-j}, j=0..21        (monthly component)

    **Why HAR captures long memory without being long-memory.**  The Heterogeneous
    Market Hypothesis (Müller et al. 1997) posits that daily traders, weekly
    portfolio managers, and monthly institutions react to volatility at different
    horizons.  Aggregating their linear responses produces an ARMA-like model
    whose autocorrelation function decays so slowly that it *mimics* the
    hyperbolic decay of a true long-memory process — but with only three free
    parameters.  This parsimony explains why HAR consistently outperforms richer
    GARCH variants in realized-measure settings (Andersen et al. 2003).

    The model is estimated by OLS on an expanding window.  Refitting at every
    step is O(T²) but unavoidable for a strict out-of-sample evaluation.

    Parameters
    ----------
    rv : pd.Series
        Daily realised volatility series, indexed by date.
    burnin : int
        Minimum observations before the first OLS fit (default: 1 trading year).
    daily_lag : int
        Lag for the daily RV component (default: 1).
    weekly_lag : int
        Rolling window for the weekly component (default: 5).
    monthly_lag : int
        Rolling window for the monthly component (default: 22).

    Returns
    -------
    pd.Series
        One-step-ahead HAR-RV forecasts, aligned with `rv.index`.
        Values before the burn-in period are NaN.

    Notes
    -----
    This implementation forecasts the *same* target as the ML models
    (next-day RV5), not daily RV.  For a proper literature comparison
    using intraday realized kernels, replace `rv` with tick-level estimates
    (Barndorff-Nielsen et al. 2008).
    """
    rv_arr = rv.to_numpy(dtype=float)
    n = len(rv_arr)
    forecasts = np.full(n, np.nan)

    for t in range(burnin - 1, n - 1):
        # Build multi-scale components up to and including t
        rv_d = rv_arr[t - daily_lag + 1 : t + 1].mean()
        rv_w = rv_arr[max(0, t - weekly_lag + 1) : t + 1].mean()
        rv_m = rv_arr[max(0, t - monthly_lag + 1) : t + 1].mean()

        # Design matrix: [1, RV_d, RV_w, RV_m] for each past row up to t
        rows = []
        for i in range(monthly_lag, t + 1):
            rd = rv_arr[i - daily_lag + 1 : i + 1].mean()
            rw = rv_arr[max(0, i - weekly_lag + 1) : i + 1].mean()
            rm = rv_arr[max(0, i - monthly_lag + 1) : i + 1].mean()
            rows.append([1.0, rd, rw, rm])

        X_fit = np.array(rows[:-1])              # features up to t-1
        y_fit = rv_arr[monthly_lag + 1 : t + 1]  # targets: RV_{t+1}

        if len(y_fit) < 4:  # need at least 4 obs for 4-parameter OLS
            continue

        # OLS via normal equations: β = (X'X)^{-1} X'y
        try:
            beta, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
        except np.linalg.LinAlgError:
            continue

        forecasts[t + 1] = max(beta[0] + beta[1] * rv_d + beta[2] * rv_w + beta[3] * rv_m, 0.0)

    return pd.Series(forecasts, index=rv.index, name="HAR_RV")


def constrained_weights(
    predictions: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Solve for the QLIKE-optimal convex blend weights over base model forecasts.

    Minimises E[σ/h − log(σ/h) − 1] subject to w ≥ 0 and Σw = 1 using
    SLSQP.  This is equivalent to maximum-likelihood estimation of a
    log-normal mixture and is consistent with the loss used in the
    volatility literature (Patton 2011).

    Parameters
    ----------
    predictions : np.ndarray, shape (n_samples, n_models)
        Out-of-fold (or validation) forecasts from each base model.
    y_true : np.ndarray, shape (n_samples,)
        Realised volatility targets (same units as predictions).
    eps : float
        Lower clip applied to blended forecast to prevent division by zero.

    Returns
    -------
    np.ndarray of shape (n_models,)
        Optimal convex weights (non-negative, sum to 1).
    """
    n_models = predictions.shape[1]

    def _qlike(w: np.ndarray) -> float:
        h = np.clip(predictions @ w, eps, None)
        ratio = y_true / h
        return float(np.mean(ratio - np.log(ratio) - 1))

    result = minimize(
        _qlike,
        x0=np.ones(n_models) / n_models,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-9, "maxiter": 500},
    )
    if not result.success:
        warnings.warn(
            f"constrained_weights: optimizer did not converge — {result.message}. "
            "Returning best-found weights; results may be suboptimal.",
            RuntimeWarning,
            stacklevel=2,
        )
    return result.x


def get_monotone_constraints(
    feature_cols: list[str],
    lookback: int,
    constraint_map: dict[str, int],
) -> tuple[int, ...]:
    """
    Build the XGBoost ``monotone_constraints`` tuple for a flattened lookback matrix.

    When features are repeated `lookback` times (one per lagged step), the
    same domain constraint applies at every lag.  Features absent from
    `constraint_map` receive constraint 0 (unconstrained).

    Domain knowledge encoded here (from the notebook):
      - GK_Vol, Park_Vol, Gap  → +1  (higher realised vol ↑ → higher forecast)
      - Log_Ret                → -1  (large negative return ↑ magnitude ↑ vol,
                                       but the sign is already captured by abs)

    Parameters
    ----------
    feature_cols : list of str
        Feature names in column order (must match X columns).
    lookback : int
        Number of lagged steps per sample row.
    constraint_map : dict[str, int]
        Maps feature name → {-1, 0, +1}.

    Returns
    -------
    tuple of int, length = len(feature_cols) × lookback

    Example
    -------
    >>> get_monotone_constraints(['GK_Vol', 'Log_Ret'], 2, {'GK_Vol': 1, 'Log_Ret': -1})
    (1, -1, 1, -1)

    The pattern interleaves by lag step (matching ``build_lookback_windows``
    which calls ``.ravel()`` on an ``(lookback, n_features)`` slice):
    lag-0 features first, then lag-1, etc.
    """
    per_step = [constraint_map.get(col, 0) for col in feature_cols]
    return tuple(per_step * lookback)


def naive_forecasts(
    rv: pd.Series,
    dates: pd.DatetimeIndex,
    info_lag: int = 1,
    mean_window: int = 22,
    har: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Simple benchmarks for the next-day RV target, using the same
    information set as the ML features.

    In this pipeline a sample dated ``d`` sees features through ``d - 1``
    and predicts ``RV5`` at ``d + 1`` (see ``build_lookback_windows``), so
    every benchmark here is built from ``rv`` up to ``d - info_lag`` only:

      - ``Random walk``: the latest available RV5.
      - ``22d mean``:    the mean of the last ``mean_window`` RV5 values.
      - ``HAR-RV``:      the expanding-window HAR one-step forecast made at
                         ``d - info_lag`` (pass ``har = har_rv(rv)``).

    A model that cannot beat the random walk on RV5, whose 5-day windows
    overlap by four days, has not shown forecasting skill.
    """
    out = pd.DataFrame(index=dates)
    out["Random walk"] = rv.shift(info_lag).reindex(dates)
    out["22d mean"] = rv.rolling(mean_window).mean().shift(info_lag).reindex(dates)
    if har is not None:
        # har[t] is the forecast of rv[t] made with data through t - 1.
        out["HAR-RV"] = har.shift(info_lag - 1).reindex(dates)
    return out
