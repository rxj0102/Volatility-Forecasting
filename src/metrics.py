"""
Evaluation metrics for volatility forecasting models.

All functions accept 1-D NumPy arrays of equal length and return a scalar.
"""
import numpy as np


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


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print a one-line metric summary for a named model."""
    n = min(len(y_true), len(y_pred))
    yt, yp = y_true[:n], y_pred[:n]
    print(
        f"{name:<30}  RMSE={rmse(yt, yp):.6f}  "
        f"MAE={mae(yt, yp):.6f}  "
        f"QLIKE={qlike(yt, yp):.4f}  "
        f"MAPE={mape(yt, yp):.2%}"
    )
