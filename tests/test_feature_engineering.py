"""
Unit tests for src/feature_engineering.py

Run with:  pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from feature_engineering import (
    build_lookback_windows,
    frac_diff_vec,
    find_min_d,
    make_oof_predictions,
    walk_forward_cv,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def random_series():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    return pd.Series(rng.standard_normal(200).cumsum() + 100, index=idx, name="Close")


@pytest.fixture
def simple_df():
    """Small deterministic DataFrame for shape / value checks."""
    n = 50
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"feat_a": rng.random(n), "feat_b": rng.random(n), "target": rng.random(n)},
        index=idx,
    )


# ── frac_diff_vec ─────────────────────────────────────────────────────────────

def test_frac_diff_returns_series(random_series):
    result = frac_diff_vec(random_series, d=0.4)
    assert isinstance(result, pd.Series)
    assert len(result) == len(random_series)


def test_frac_diff_same_index(random_series):
    result = frac_diff_vec(random_series, d=0.4)
    assert result.index.equals(random_series.index)


def test_frac_diff_leading_nans(random_series):
    """With a large threshold the weight vector stays short, leaving leading NaNs."""
    # thres=0.1 keeps the weight vector short (< len(series)) so warm-up NaNs appear
    result = frac_diff_vec(random_series, d=0.4, thres=0.1)
    first_valid = result.first_valid_index()
    assert first_valid is not None, "Expected some non-NaN values in output"
    assert result.loc[:first_valid].iloc[:-1].isna().all()


def test_frac_diff_d1_approx_integer_diff(random_series):
    """With d=1, fractional diff should approximate standard first difference."""
    fd = frac_diff_vec(random_series, d=1.0, thres=1e-5).dropna()
    standard_diff = random_series.diff().dropna()
    # Align on common index
    common = fd.index.intersection(standard_diff.index)
    np.testing.assert_allclose(
        fd.loc[common].values,
        standard_diff.loc[common].values,
        atol=1e-6,
    )


def test_frac_diff_name(random_series):
    result = frac_diff_vec(random_series, d=0.3)
    assert result.name == "FracDiff_Close"


# ── find_min_d ────────────────────────────────────────────────────────────────

def test_find_min_d_returns_float(random_series):
    d = find_min_d(random_series, n_steps=5)
    assert isinstance(d, float)


def test_find_min_d_in_range(random_series):
    d = find_min_d(random_series, d_start=0.1, d_end=1.0, n_steps=5)
    assert 0.1 <= d <= 1.0


def test_find_min_d_achieves_stationarity(random_series):
    """The returned d must produce a series that passes the ADF test at the 5% level."""
    d = find_min_d(random_series, n_steps=10)
    fd = frac_diff_vec(random_series, d).dropna()
    p_value = adfuller(fd, maxlag=1, autolag=None)[1]
    assert p_value <= 0.05, (
        f"find_min_d returned d={d} but ADF p-value={p_value:.4f} > 0.05"
    )


# ── build_lookback_windows ────────────────────────────────────────────────────

def test_lookback_output_shapes(simple_df):
    lookback = 5
    X, y, dates = build_lookback_windows(simple_df, ["feat_a", "feat_b"], "target", lookback)
    n_expected = len(simple_df) - lookback
    assert X.shape == (n_expected, 2 * lookback)
    assert y.shape == (n_expected,)
    assert len(dates) == n_expected


def test_lookback_dates_alignment(simple_df):
    lookback = 3
    _, _, dates = build_lookback_windows(simple_df, ["feat_a"], "target", lookback)
    assert dates[0] == simple_df.index[lookback]
    assert dates[-1] == simple_df.index[-1]


def test_lookback_no_future_leakage(simple_df):
    """X[i] should only contain rows i … i+lookback-1 of the DataFrame."""
    lookback = 4
    X, _, _ = build_lookback_windows(simple_df, ["feat_a"], "target", lookback)
    # Check first sample manually
    expected = simple_df["feat_a"].iloc[:lookback].values
    np.testing.assert_array_equal(X[0], expected)


def test_lookback_target_values(simple_df):
    """y[i] should equal target at row i + lookback."""
    lookback = 3
    _, y, _ = build_lookback_windows(simple_df, ["feat_a"], "target", lookback)
    expected = simple_df["target"].iloc[lookback:].values
    np.testing.assert_array_equal(y, expected)


# ── walk_forward_cv ───────────────────────────────────────────────────────────

def test_walk_forward_cv_returns_list():
    X = np.zeros((100, 5))
    y = np.zeros(100)
    folds = walk_forward_cv(X, y, n_splits=3, embargo=5)
    assert isinstance(folds, list)
    assert len(folds) == 3


def test_walk_forward_cv_no_overlap():
    """Validation folds must not overlap with their corresponding training fold."""
    X = np.zeros((100, 5))
    y = np.zeros(100)
    for tr_idx, val_idx in walk_forward_cv(X, y, n_splits=4, embargo=5):
        assert len(set(tr_idx) & set(val_idx)) == 0


def test_walk_forward_cv_embargo():
    """Training indices must end at least `embargo` steps before validation starts."""
    X = np.zeros((100, 5))
    y = np.zeros(100)
    embargo = 5
    for tr_idx, val_idx in walk_forward_cv(X, y, n_splits=3, embargo=embargo):
        if len(tr_idx) > 0 and len(val_idx) > 0:
            # Gap must be strictly greater than embargo (TimeSeriesSplit already
            # ensures no overlap, so without embargo the gap would be 1; with
            # embargo applied the gap must be > embargo).
            assert val_idx[0] - tr_idx[-1] > embargo


# ── make_oof_predictions ──────────────────────────────────────────────────────

def test_make_oof_shape():
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(1)
    X = rng.random((80, 4))
    y = rng.random(80)
    oof = make_oof_predictions(Ridge(), X, y, n_splits=3, embargo=3)
    assert oof.shape == (80,)


def test_make_oof_non_zero():
    """OOF predictions covering at least one validation fold should be non-zero."""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(2)
    X = rng.random((80, 4))
    y = rng.random(80) + 1.0
    oof = make_oof_predictions(Ridge(), X, y, n_splits=3, embargo=3)
    assert np.any(oof != 0.0)
