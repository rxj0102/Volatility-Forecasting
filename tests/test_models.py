"""
Unit tests for src/models.py

Run with:  pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import constrained_weights, get_monotone_constraints, har_rv


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def rv_series():
    """Realistic synthetic RV series: positive, mean-reverting."""
    rng = np.random.default_rng(7)
    n = 600
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    # AR(1) process with positive mean
    rv = np.empty(n)
    rv[0] = 0.01
    for t in range(1, n):
        rv[t] = max(0.005 + 0.7 * rv[t - 1] + rng.standard_normal() * 0.002, 1e-5)
    return pd.Series(rv, index=idx, name="RV5")


@pytest.fixture
def blend_data():
    """Synthetic base-model predictions and targets for weight estimation."""
    rng = np.random.default_rng(3)
    n = 200
    y_true = np.abs(rng.standard_normal(n)) * 0.01 + 0.015
    preds = np.column_stack([
        y_true + rng.standard_normal(n) * 0.002,   # model 1 — low noise
        y_true + rng.standard_normal(n) * 0.004,   # model 2 — high noise
        y_true + rng.standard_normal(n) * 0.003,   # model 3 — medium noise
    ])
    return preds, y_true


# ── har_rv ────────────────────────────────────────────────────────────────────

def test_har_rv_returns_series(rv_series):
    result = har_rv(rv_series)
    assert isinstance(result, pd.Series)
    assert len(result) == len(rv_series)


def test_har_rv_same_index(rv_series):
    result = har_rv(rv_series)
    assert result.index.equals(rv_series.index)


def test_har_rv_nans_during_burnin(rv_series):
    """All values within burn-in period must be NaN."""
    burnin = 252
    result = har_rv(rv_series, burnin=burnin)
    assert result.iloc[:burnin].isna().all()


def test_har_rv_first_forecast_at_burnin(rv_series):
    """First non-NaN forecast must appear at index burnin (not burnin+1)."""
    burnin = 252
    result = har_rv(rv_series, burnin=burnin)
    assert not np.isnan(result.iloc[burnin]), (
        f"Expected first forecast at iloc[{burnin}], got NaN"
    )


def test_har_rv_forecasts_positive(rv_series):
    """All non-NaN forecasts must be non-negative (clipped at 0)."""
    result = har_rv(rv_series).dropna()
    assert (result >= 0).all()


def test_har_rv_out_of_sample_count(rv_series):
    """Should produce forecasts for at least (n - burnin - 22) steps."""
    burnin = 252
    result = har_rv(rv_series, burnin=burnin)
    n_forecasts = result.dropna().shape[0]
    assert n_forecasts >= len(rv_series) - burnin - 25


def test_har_rv_forecast_reasonable_magnitude(rv_series):
    """Forecasts should be in the same ballpark as the input series."""
    result = har_rv(rv_series).dropna()
    assert result.mean() == pytest.approx(rv_series.mean(), rel=0.5)


def test_har_rv_forecast_matches_ols(rv_series):
    """First forecast must equal the OLS prediction from the manual calculation."""
    burnin = 252
    daily_lag, weekly_lag, monthly_lag = 1, 5, 22
    rv_arr = rv_series.to_numpy(dtype=float)

    result = har_rv(rv_series, burnin=burnin)

    # Replicate the first iteration: t = burnin - 1, forecast → forecasts[burnin]
    t = burnin - 1
    rows = []
    for i in range(monthly_lag, t + 1):
        rd = rv_arr[i - daily_lag + 1 : i + 1].mean()
        rw = rv_arr[max(0, i - weekly_lag + 1) : i + 1].mean()
        rm = rv_arr[max(0, i - monthly_lag + 1) : i + 1].mean()
        rows.append([1.0, rd, rw, rm])

    X_fit = np.array(rows[:-1])
    y_fit = rv_arr[monthly_lag + 1 : t + 1]
    beta, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)

    rv_d = rv_arr[t - daily_lag + 1 : t + 1].mean()
    rv_w = rv_arr[max(0, t - weekly_lag + 1) : t + 1].mean()
    rv_m = rv_arr[max(0, t - monthly_lag + 1) : t + 1].mean()
    expected = max(beta[0] + beta[1] * rv_d + beta[2] * rv_w + beta[3] * rv_m, 0.0)

    assert result.iloc[burnin] == pytest.approx(expected, rel=1e-6)


# ── constrained_weights ───────────────────────────────────────────────────────

def test_constrained_weights_shape(blend_data):
    preds, y = blend_data
    w = constrained_weights(preds, y)
    assert w.shape == (3,)


def test_constrained_weights_sum_to_one(blend_data):
    preds, y = blend_data
    w = constrained_weights(preds, y)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_constrained_weights_nonneg(blend_data):
    preds, y = blend_data
    w = constrained_weights(preds, y)
    assert (w >= -1e-8).all()


def test_constrained_weights_favours_better_model(blend_data):
    """The lower-noise model 1 should receive the highest weight."""
    preds, y = blend_data
    w = constrained_weights(preds, y)
    assert w[0] >= w[1], f"Expected w[0]={w[0]:.4f} >= w[1]={w[1]:.4f}"


# ── get_monotone_constraints ──────────────────────────────────────────────────

def test_monotone_constraints_length():
    features = ["GK_Vol", "Log_Ret", "VIX_Daily"]
    lookback = 5
    result = get_monotone_constraints(features, lookback, {"GK_Vol": 1, "Log_Ret": -1})
    assert len(result) == len(features) * lookback


def test_monotone_constraints_values():
    # build_lookback_windows uses .ravel() on (lookback, n_features) → interleaved by row:
    # [GK_Vol_lag0, Log_Ret_lag0, GK_Vol_lag1, Log_Ret_lag1]
    result = get_monotone_constraints(["GK_Vol", "Log_Ret"], 2, {"GK_Vol": 1, "Log_Ret": -1})
    assert result == (1, -1, 1, -1)


def test_monotone_constraints_unconstrained_zero():
    """Features absent from constraint_map must have constraint 0."""
    result = get_monotone_constraints(["VIX", "RSI"], 3, {"VIX": 1})
    # Interleaved: [VIX_lag0=1, RSI_lag0=0, VIX_lag1=1, RSI_lag1=0, VIX_lag2=1, RSI_lag2=0]
    assert result == (1, 0, 1, 0, 1, 0)


def test_monotone_constraints_returns_tuple():
    result = get_monotone_constraints(["a", "b"], 2, {})
    assert isinstance(result, tuple)


# ── naive_forecasts ───────────────────────────────────────────────────────────

from models import naive_forecasts


def test_naive_forecasts_use_only_information_through_d_minus_1(rv_series):
    dates = rv_series.index[300:400]
    har = har_rv(rv_series)
    base = naive_forecasts(rv_series, dates, har=har)
    d = dates[50]
    shocked = rv_series.copy()
    shocked.loc[d:] *= 10.0  # change day d and everything after it
    alt = naive_forecasts(shocked, dates, har=har_rv(shocked))
    pd.testing.assert_frame_equal(base.loc[:d], alt.loc[:d])


def test_random_walk_is_previous_value(rv_series):
    dates = rv_series.index[100:110]
    out = naive_forecasts(rv_series, dates)
    assert np.allclose(out["Random walk"].values, rv_series.shift(1).loc[dates].values)
