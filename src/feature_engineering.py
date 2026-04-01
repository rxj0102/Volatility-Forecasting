"""
Feature engineering for volatility forecasting.

Implements fractional differentiation (Lopez de Prado 2018) and
the lookback-window construction used by the tabular ML models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# ── Fractional Differentiation ────────────────────────────────────────────────

def frac_diff_vec(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
    """
    Apply fractional differentiation with fixed-width window truncation.

    Standard integer differencing (d=1) achieves stationarity but destroys
    all long-range memory in the price series.  Fractional differencing with
    d ∈ (0, 1) achieves stationarity while preserving the maximum amount of
    memory — a key insight from *Advances in Financial Machine Learning*
    (Lopez de Prado, 2018, Chapter 5).

    The weight vector w_k = Π_{j=0}^{k-1} (d - j) / (j + 1) decays to zero,
    so we truncate at `thres` to keep computation tractable.

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
    # weights[0] applies to the most recent observation
    weights = np.array(weights[::-1], dtype=float)  # oldest weight first

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
    Walk-forward (expanding-window) cross-validation with purging.

    The `embargo` parameter removes the last `embargo` rows from each
    training fold.  This prevents leakage when the target overlaps across
    adjacent samples (e.g. 5-day realised volatility windows).

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
