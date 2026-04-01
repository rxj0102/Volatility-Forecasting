"""
Unit tests for src/metrics.py

Run with:  pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import diebold_mariano, mae, mape, qlike, report, rmse


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect():
    """Identical forecasts — all metrics should be 0."""
    y = np.array([0.01, 0.02, 0.015, 0.03, 0.025])
    return y, y.copy()


@pytest.fixture
def simple():
    y_true = np.array([0.01, 0.02, 0.03])
    y_pred = np.array([0.011, 0.018, 0.032])
    return y_true, y_pred


# ── RMSE ──────────────────────────────────────────────────────────────────────

def test_rmse_perfect(perfect):
    y, yp = perfect
    assert rmse(y, yp) == pytest.approx(0.0)


def test_rmse_known():
    y = np.array([1.0, 2.0])
    yp = np.array([2.0, 2.0])
    assert rmse(y, yp) == pytest.approx(np.sqrt(0.5))


def test_rmse_symmetric():
    y = np.array([1.0, 2.0, 3.0])
    yp = np.array([2.0, 3.0, 4.0])
    assert rmse(y, yp) == rmse(yp, y)


# ── MAE ───────────────────────────────────────────────────────────────────────

def test_mae_perfect(perfect):
    y, yp = perfect
    assert mae(y, yp) == pytest.approx(0.0)


def test_mae_known():
    y = np.array([1.0, 3.0])
    yp = np.array([2.0, 2.0])
    assert mae(y, yp) == pytest.approx(1.0)


# ── QLIKE ─────────────────────────────────────────────────────────────────────

def test_qlike_perfect(perfect):
    """QLIKE = 0 iff forecast = realised exactly."""
    y, yp = perfect
    assert qlike(y, yp) == pytest.approx(0.0, abs=1e-10)


def test_qlike_nonneg(simple):
    """QLIKE is non-negative for any positive inputs."""
    y, yp = simple
    assert qlike(y, yp) >= 0.0


def test_qlike_asymmetric():
    """QLIKE penalises underprediction more than overprediction."""
    y = np.array([0.02])
    under = qlike(y, np.array([0.01]))   # forecast too low
    over  = qlike(y, np.array([0.04]))   # forecast too high by same ratio
    assert under > over


def test_qlike_known():
    """Manually compute QLIKE for a single observation."""
    y = np.array([2.0])
    h = np.array([1.0])
    # QLIKE = y/h - log(y/h) - 1 = 2 - log(2) - 1 = 1 - log(2)
    expected = 1.0 - np.log(2.0)
    assert qlike(y, h) == pytest.approx(expected, rel=1e-6)


# ── MAPE ──────────────────────────────────────────────────────────────────────

def test_mape_perfect(perfect):
    y, yp = perfect
    assert mape(y, yp) == pytest.approx(0.0)


def test_mape_known():
    y  = np.array([1.0, 2.0])
    yp = np.array([1.1, 1.8])
    # |1.0-1.1|/1.0=0.10, |2.0-1.8|/2.0=0.10 → mean=0.10
    expected = 0.10
    assert mape(y, yp) == pytest.approx(expected, rel=1e-6)


# ── report ────────────────────────────────────────────────────────────────────

def test_report_runs_without_error(simple, capsys):
    y, yp = simple
    report("TestModel", y, yp)
    captured = capsys.readouterr()
    assert "TestModel" in captured.out
    assert "RMSE" in captured.out
    assert "QLIKE" in captured.out


def test_report_handles_length_mismatch(capsys):
    """report() should use min(len(y_true), len(y_pred)) without raising."""
    y  = np.array([0.01, 0.02, 0.03])
    yp = np.array([0.01, 0.02])
    report("LenMismatch", y, yp)
    captured = capsys.readouterr()
    assert "LenMismatch" in captured.out


# ── diebold_mariano ───────────────────────────────────────────────────────────

@pytest.fixture
def dm_arrays():
    """y_true, a near-perfect predictor, and a noisy predictor."""
    rng = np.random.default_rng(42)
    n = 300
    y_true = np.abs(rng.standard_normal(n)) * 0.01 + 0.015
    y_good = y_true + rng.standard_normal(n) * 0.001   # low noise
    y_bad  = y_true + rng.standard_normal(n) * 0.005   # high noise
    return y_true, y_good, y_bad


def test_dm_returns_tuple(dm_arrays):
    y, yg, yb = dm_arrays
    result = diebold_mariano(y, yb, yg, loss="qlike")
    assert isinstance(result, tuple) and len(result) == 2


def test_dm_stat_positive_when_model2_better(dm_arrays):
    """DM > 0 means model 1 has higher loss, i.e. model 2 is better."""
    y, y_good, y_bad = dm_arrays
    dm_stat, _ = diebold_mariano(y, y_bad, y_good, loss="qlike")
    assert dm_stat > 0


def test_dm_p_value_in_unit_interval(dm_arrays):
    y, yg, yb = dm_arrays
    _, p = diebold_mariano(y, yb, yg, loss="mse")
    assert 0.0 <= p <= 1.0


def test_dm_equal_models_pvalue_large():
    """If both models are identical, the test should not reject H0."""
    rng = np.random.default_rng(0)
    y  = np.abs(rng.standard_normal(500)) * 0.01 + 0.02
    yp = y + rng.standard_normal(500) * 0.0005
    # Same predictor — DM stat ≈ 0, p ≈ 1
    _, p = diebold_mariano(y, yp, yp.copy(), loss="qlike")
    assert p > 0.5


def test_dm_invalid_loss_raises():
    y = np.ones(10) * 0.02
    with pytest.raises(ValueError, match="loss must be"):
        diebold_mariano(y, y, y, loss="mae")


def test_dm_length_mismatch_raises():
    y1 = np.ones(10) * 0.02
    y2 = np.ones(11) * 0.02
    with pytest.raises(ValueError, match="same length"):
        diebold_mariano(y1, y1, y2)
