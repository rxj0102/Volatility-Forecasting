"""cached_download must return what it saved, without touching the network."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data_cache  # noqa: E402


def test_round_trip(tmp_path, monkeypatch):
    idx = pd.date_range("2020-01-01", periods=3, name="Date")
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["SPY"]], names=["Price", "Ticker"])
    df = pd.DataFrame([[1.5, 1.0], [2.25, 2.0], [3.125, 3.0]], index=idx, columns=cols)
    calls = []
    monkeypatch.setattr(data_cache, "SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(data_cache.yf, "download", lambda *a, **k: calls.append(1) or df)
    first = data_cache.cached_download("SPY", start="2020-01-01", progress=False)
    second = data_cache.cached_download("SPY", start="2020-01-01", progress=False)
    assert len(calls) == 1
    assert first.equals(second)
    assert float(second["Close"]["SPY"].iloc[2]) == 3.125
    assert list(second.columns.get_level_values(0)) == ["Close", "Open"]
