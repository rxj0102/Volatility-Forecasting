"""
Model helpers for the volatility forecasting pipeline.

Provides:
  - constrained_weights : QLIKE-optimal convex combination of base forecasts
  - get_monotone_constraints : XGBoost monotone constraint vector for
                               flattened lookback features
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


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
    (1, 1, -1, -1)
    """
    per_step = [constraint_map.get(col, 0) for col in feature_cols]
    return tuple(per_step * lookback)
