"""
Feature engineering for volatility forecasting.

Key components:

1. **Fractional Differentiation** (Granger & Joyeux 1980; Hosking 1981).
   The classical stationarity–memory trade-off: integer differencing achieves
   I(0) but eliminates all long-range dependence.  For a price series X_t ~ I(1),
   the fractionally differenced series Y_t = (1−L)^d X_t is I(1−d).  Choosing
   d ∈ (0, 1) produces a covariance-stationary series that retains long memory.

2. **Lookback window construction** — flattens a 3-D panel (time × features × lags)
   into a 2-D matrix compatible with scikit-learn and gradient-boosted tree APIs.

3. **Purged walk-forward CV** (López de Prado 2018, Ch. 7) — embargo parameter
   prevents label overlap across train/validation folds when the target is a
   multi-day aggregation.

References
----------
Granger & Joyeux (1980)  : J. Time Series Analysis 1, 15–29.
Hosking (1981)           : Biometrika 68, 165–176.
López de Prado (2018)    : Advances in Financial Machine Learning. Wiley.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# ── Fractional Differentiation ────────────────────────────────────────────────

def frac_diff_vec(series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
    """
    Apply fractional differentiation with fixed-width window truncation.

    The fractional differencing operator (1−L)^d is defined via the binomial series
    (Granger & Joyeux 1980; Hosking 1981):

        (1−L)^d = Σ_{k=0}^∞ C(d, k)·(−L)^k = Σ_{k=0}^∞ w_k·L^k

    where the generalised binomial coefficient gives the recurrence:

        w_0 = 1,   w_k = −w_{k-1} · (d − k + 1) / k

    For d ∈ (0, 1), the weights satisfy |w_k| ~ k^{-(d+1)} / Γ(−d) as k → ∞.
    This *hyperbolic* decay is slower than the exponential decay of ARMA weights,
    which is precisely what makes the differenced series long-memory.  The
    autocorrelation function of (1−L)^d X_t decays as ρ(k) ~ k^{2d−1} — slow
    enough to be non-summable, yet fast enough for the series to be I(0).

    Standard integer differencing (d=1) forces geometric weights to zero in a
    single step, destroying all this memory.  Fractional differencing with the
    *minimum* d that passes an ADF test preserves the maximum memory consistent
    with stationarity — the goal of `find_min_d`.

    The weight series is truncated at `thres` to keep computation tractable.
    The first ``len(weights) - 1`` outputs are NaN, so `thres` sets how much
    history is lost. At 1e-5 and d = 0.3 the window is about 2,300 days,
    which leaves only a few months of a 2014 to 2024 daily sample; the
    default 1e-4 keeps the window under 400 days for d >= 0.3.

    Parameters
    ----------
    series : pd.Series
        Input series (e.g. log-price).  Name is preserved in output.
    d : float
        Fractional order, typically in (0, 1].
    thres : float
        Absolute weight below which older lags are discarded.

    Returns
    -------
    pd.Series
        Fractionally differenced series with the same index as `series`.
        The first ``len(weights) - 1`` entries are NaN (warm-up period).
    """
    # Build weight vector: w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k
    weights: list[float] = [1.0]
    k, w = 1, 1.0
    while abs(w) >= thres:
        w = -weights[-1] * (d - k + 1) / k
        weights.append(w)
        k += 1
    weights = np.array(weights[::-1], dtype=float)
    # After reversal: weights[0] = oldest-lag coefficient, weights[-1] = w_0 = 1.0 (most recent)

    n = len(series)
    w_len = len(weights)
    result = np.full(n, np.nan)
    arr = series.to_numpy(dtype=float)

    for i in range(w_len - 1, n):
        result[i] = float(np.dot(weights, arr[i - w_len + 1 : i + 1]))

    name = f"FracDiff_{series.name}" if series.name else "FracDiff"
    return pd.Series(result, index=series.index, name=name)


def find_min_d(
    series: pd.Series,
    thres: float = 1e-5,
    d_start: float = 0.1,
    d_end: float = 1.0,
    n_steps: int = 10,
    adf_pval: float = 0.05,
) -> float:
    """
    Grid-search for the minimum fractional order d that yields stationarity.

    The ADF test is used as the stationarity criterion.  The smallest d that
    rejects the unit-root null at level `adf_pval` is returned, maximising
    retained memory.

    Parameters
    ----------
    series : pd.Series
        Log-price or similar non-stationary series.
    thres : float
        Truncation threshold passed to `frac_diff_vec`.
    d_start, d_end : float
        Grid search bounds.
    n_steps : int
        Number of candidate values between d_start and d_end.
    adf_pval : float
        ADF significance level (default 5%).

    Returns
    -------
    float
        Minimum d achieving stationarity, or `d_end` if none found.

    Warnings
    --------
    Pass only **training-fold data** to avoid look-ahead bias.  Calling this
    function on the full time series (including validation or test observations)
    causes d* to be selected with knowledge of future data.
    """
    for d in np.linspace(d_start, d_end, n_steps):
        fd = frac_diff_vec(series, float(d), thres).dropna()
        if len(fd) < 30:
            continue
        p_value = adfuller(fd, maxlag=1, autolag=None)[1]
        if p_value <= adf_pval:
            return round(float(d), 3)
    return float(d_end)


# ── Lookback Window Construction ──────────────────────────────────────────────

def build_lookback_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Flatten a rolling lookback window into a 2-D feature matrix.

    Each row in X corresponds to a date t and contains the values of all
    `feature_cols` for the preceding `lookback` periods
    (i.e. t-lookback+1 … t).  The target y[i] is the value of `target_col`
    at the row immediately *after* the window (next-day forecast).

    Shape: X → (n_samples, n_features × lookback),  y → (n_samples,)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by date, containing feature and target columns.
    feature_cols : list of str
    target_col : str
        Column holding the forward-looking target (must already be shifted).
    lookback : int
        Number of past time steps per sample (≥ 1).

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features * lookback)
    y : np.ndarray, shape (n_samples,)
    dates : pd.DatetimeIndex
        Date index aligned with each sample row (the *last* date in the window).
    """
    feat = df[feature_cols].to_numpy(dtype=float)
    tgt = df[target_col].to_numpy(dtype=float)
    idx = df.index

    n = len(df) - lookback
    n_features = len(feature_cols)
    X = np.empty((n, n_features * lookback), dtype=float)
    for i in range(n):
        X[i] = feat[i : i + lookback].ravel()
    y = tgt[lookback:]
    dates = idx[lookback:]
    return X, y, dates


# ── Cross-Validation Utilities ────────────────────────────────────────────────

def walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    embargo: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward (expanding-window) cross-validation with purging and embargo.

    **Why purging matters for overlapping targets.**  The target in this project
    is a 5-day forward RV: y_t = std(r_{t+1}, ..., r_{t+5}).  Adjacent samples
    share four of the five returns, so a model trained on y_t and validated on
    y_{t+1} has seen almost all of the validation label during training.  The
    embargo removes the last `embargo` samples from each training fold so that
    no training label overlaps with any validation label.  This is equivalent
    to Combinatorial Purged Cross-Validation (López de Prado 2018, Ch. 12)
    for a single-path expanding window.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (used only for its length).
    y : np.ndarray
        Target vector (used only for its length).
    n_splits : int
        Number of CV folds.
    embargo : int
        Rows purged from the end of each training fold.

    Returns
    -------
    list of (train_idx, val_idx) pairs.
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for tr_idx, val_idx in tscv.split(X):
        purged = tr_idx[:-embargo] if len(tr_idx) > embargo else tr_idx
        folds.append((purged, val_idx))
    return folds


def make_oof_predictions(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    embargo: int = 5,
) -> np.ndarray:
    """
    Generate out-of-fold (OOF) predictions for stacking / meta-learning.

    OOF predictions ensure that the meta-learner never trains on
    in-sample base-model outputs, preventing overfitting of the stack.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Must implement `fit(X, y)` and `predict(X)`.
    X_train : np.ndarray
    y_train : np.ndarray
    n_splits : int
    embargo : int

    Returns
    -------
    np.ndarray of shape (len(X_train),)
        OOF predictions; rows not covered by any validation fold are 0.
    """
    oof = np.zeros(len(X_train))
    folds = walk_forward_cv(X_train, y_train, n_splits=n_splits, embargo=embargo)
    for tr_idx, val_idx in folds:
        clone = type(estimator)(**estimator.get_params())
        clone.fit(X_train[tr_idx], y_train[tr_idx])
        oof[val_idx] = clone.predict(X_train[val_idx])
    return oof
