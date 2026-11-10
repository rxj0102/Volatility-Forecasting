"""Frozen yfinance downloads.

Yahoo serves slightly different adjusted history at different times, which moved
the reported model metrics between two runs on the same code. The first call
for a given request saves the result in data/snapshot/; later calls read it back,
so the README numbers can be reproduced exactly. Set VF_REFRESH_DATA=1 to
download again and overwrite the snapshot.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import yfinance as yf

SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "data" / "snapshot"


def _key(tickers, kwargs) -> str:
    names = [tickers] if isinstance(tickers, str) else list(tickers)
    spec = "|".join(names) + "|" + "|".join(f"{k}={kwargs[k]}" for k in sorted(kwargs) if k != "progress")
    label = "_".join(n.replace("^", "") for n in names)
    return f"{label}_{hashlib.sha1(spec.encode()).hexdigest()[:8]}.csv"


def cached_download(tickers, **kwargs) -> pd.DataFrame:
    """Drop-in replacement for yf.download that reads from data/snapshot/ when present."""
    path = SNAPSHOT_DIR / _key(tickers, kwargs)
    if path.exists() and os.environ.get("VF_REFRESH_DATA") != "1":
        df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
        df.index.name = "Date"
        return df
    df = yf.download(tickers, **kwargs)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {tickers}")
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_product([df.columns, [tickers if isinstance(tickers, str) else ""]])
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    df = df.rename_axis(index=None, columns=[None, None])  # keep exactly two header rows
    df.to_csv(path, float_format="%.8g")
    return cached_download(tickers, **kwargs)
