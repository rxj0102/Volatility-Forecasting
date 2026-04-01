"""
Unit tests for src/metrics.py

Run with:  pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import mae, mape, qlike, report, rmse


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
