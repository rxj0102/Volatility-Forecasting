"""
Evaluation metrics for volatility forecasting models.

All functions accept 1-D NumPy arrays of equal length and return a scalar.

References
----------
Patton (2011)   : Volatility forecast comparison using imperfect proxies.
                  J. Econometrics 160, 246–256.
Diebold-Mariano : Comparing predictive accuracy.
                  J. Bus. Econ. Stat. 13, 253–263 (1995).
"""
import warnings

import numpy as np
from scipy import stats


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def qlike(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    QLIKE loss — the standard loss function for volatility model evaluation.

    Derived from the log-likelihood of a normal distribution with time-varying
    variance.  Unlike RMSE, QLIKE penalises underprediction asymmetrically,
    making it the preferred criterion in the volatility literature
    (Patton 2011; Hansen & Lunde 2005).

    QLIKE = E[ σ/h  −  log(σ/h)  −  1 ]

    where σ is realised volatility and h is the model forecast.
    The loss is minimised (= 0) when h = σ almost surely.
    """
    h = np.clip(y_pred, eps, None)
    ratio = y_true / h
    return float(np.mean(ratio - np.log(ratio) - 1))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error."""
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def diebold_mariano(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    loss: str = "qlike",
    h: int = 1,
) -> tuple[float, float]:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    Tests H₀: E[L(e₁_t) − L(e₂_t)] = 0 against H₁: model 1 is less accurate.
    The test statistic is:

        DM = d̄ / √(σ̂²_d / T)  →  N(0, 1)  under H₀

    where d_t = L(e₁_t) − L(e₂_t) is the loss differential and σ̂²_d is its
    Newey-West long-run variance with bandwidth h−1.

    **Why QLIKE, not MSE?**  Patton (2011, Theorem 1) proves that QLIKE produces
    *consistent* model rankings under a noisy volatility proxy: if model A is
    truly better than B under the latent variance, QLIKE-DM correctly rejects
    H₀ regardless of proxy quality.  MSE rankings can be reversed by proxy noise
    (Patton 2011, Table I), making DM with MSE unreliable in this setting.

    Parameters
    ----------
    y_true : np.ndarray
        Realised volatility proxy (e.g. RV5).
    y_pred_1 : np.ndarray
        Forecasts from model 1 (baseline — positive DM favours model 2).
    y_pred_2 : np.ndarray
        Forecasts from model 2 (challenger).
    loss : {'qlike', 'mse'}
        Loss function used to form the differential.
    h : int
        Forecast horizon; sets the Newey-West bandwidth to max(0, h−1).

    Returns
    -------
    dm_stat : float
        DM test statistic (positive → model 2 is more accurate).
    p_value : float
        Two-sided p-value under the standard normal approximation.

    Raises
    ------
    ValueError
        If `loss` is not 'qlike' or 'mse', or arrays have different lengths.
    """
    if loss not in ("qlike", "mse"):
        raise ValueError(f"loss must be 'qlike' or 'mse', got '{loss}'")
    n = len(y_true)
    if not (len(y_pred_1) == n == len(y_pred_2)):
        raise ValueError("y_true, y_pred_1, y_pred_2 must have the same length")

    eps = 1e-8
    if loss == "qlike":
        h1 = np.clip(y_pred_1, eps, None)
        h2 = np.clip(y_pred_2, eps, None)
        l1 = y_true / h1 - np.log(y_true / h1) - 1
        l2 = y_true / h2 - np.log(y_true / h2) - 1
    else:  # mse
        l1 = (y_true - y_pred_1) ** 2
        l2 = (y_true - y_pred_2) ** 2

    d = l1 - l2
    d_bar = d.mean()

    # Newey-West long-run variance with bandwidth = max(0, h-1)
    bw = max(0, h - 1)
    gamma_0 = np.var(d, ddof=0)
    nw_var = gamma_0
    for lag in range(1, bw + 1):
        gamma_k = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        nw_var += 2 * (1 - lag / (bw + 1)) * gamma_k

    se = np.sqrt(max(nw_var, eps) / n)
    dm_stat = d_bar / se
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print a one-line metric summary for a named model."""
    n = min(len(y_true), len(y_pred))
    if len(y_true) != len(y_pred):
        warnings.warn(
            f"report: length mismatch (y_true={len(y_true)}, y_pred={len(y_pred)}); "
            f"truncating both to {n}.",
            UserWarning,
            stacklevel=2,
        )
    yt, yp = y_true[:n], y_pred[:n]
    print(
        f"{name:<30}  RMSE={rmse(yt, yp):.6f}  "
        f"MAE={mae(yt, yp):.6f}  "
        f"QLIKE={qlike(yt, yp):.4f}  "
        f"MAPE={mape(yt, yp):.2%}"
    )
