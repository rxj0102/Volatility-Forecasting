# Production Pipeline

Automated daily retraining and forecasting pipeline for the SPY GJR-GARCH + XGBoost
volatility ensemble.  Every run downloads fresh market data, retrains both models on
the full available history, blends their forecasts with QLIKE-optimal weights, and
appends the result to a persistent forecast log.

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the daily pipeline (uses today's latest available data)
python -m pipeline.daily_pipeline

# Run for a specific historical date
python -m pipeline.daily_pipeline --date 2025-03-15

# Populate the forecast log with 2 years of historical predictions
python -m pipeline.backfill

# Backfill with custom parameters
python -m pipeline.backfill --backfill-years 2 --history-years 5 --xgb-refit-freq 21
```

Logs are written to `outputs/pipeline.log`.  Forecasts are appended to
`outputs/forecast_log.csv`.

---

## Pipeline steps (`daily_pipeline.py`)

| Step | Description |
|------|-------------|
| 1 | Download the latest 5 years of SPY OHLCV via `yfinance` |
| 2 | Compute **Garman-Klass**, **Parkinson**, and **close-to-close** realized vol |
| 3 | Build a bias-free GJR-GARCH feature series: σ̂_{t\|t-1} on an expanding window (refit every 21 days) |
| 4 | Fit GJR-GARCH on the full history to obtain the next-day σ forecast |
| 5 | Train XGBoost on `log(RV5)` with 5-fold walk-forward CV and a 5-day embargo |
| 6 | Solve for QLIKE-optimal convex blend weights over GARCH and XGBoost |
| 7 | Back-fill `realized_rv5` for past rows whose `target_date` is now observable |
| 8 | Append the new forecast row to `outputs/forecast_log.csv` |

---

## Forecast log columns (`outputs/forecast_log.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `forecast_as_of` | date | Last trading day with available market data when the pipeline ran |
| `target_date` | date | The next business day — the date being forecast |
| `garch_forecast` | float | GJR-GARCH(1,1,1)-t one-step-ahead annualised volatility |
| `xgb_forecast` | float | XGBoost log-vol model forecast (annualised, in vol space) |
| `ensemble_forecast` | float | QLIKE-optimal convex blend: `w_GARCH · σ̂_GARCH + w_XGB · σ̂_XGB` |
| `realized_rv5` | float | Actual 5-day rolling close-to-close vol on `target_date` (back-filled on subsequent runs; empty until observable) |
| `split` | string | `IS` or `OOS` — see below |
| `run_timestamp` | datetime | UTC timestamp of pipeline execution |

All volatility values are **annualised** (multiplied by √252) and expressed as
decimals (e.g. `0.012` = 1.2% daily volatility × √252 ≈ 19% annualised).

---

## IS vs OOS interpretation

### Daily pipeline
Every forecast produced by `daily_pipeline.py` is labelled **`OOS`** by default.
Pass `--split IS` only if you are re-running the pipeline for a date that falls
within the original notebook training window (pre-2023 approximately).

### Backfill (`backfill.py`)
The 2-year backfill window is split at its midpoint:

```
|───── IS (year -2 to -1) ─────|───── OOS (year -1 to today) ─────|
     ↑ backfill_start                    ↑ is_oos_boundary          ↑ last_date
```

* **IS** rows reflect predictions made within the approximate training period of
  the research notebook.  Model parameters may have been calibrated on data that
  overlaps with these dates, so IS metrics are optimistic.

* **OOS** rows reflect predictions made on dates strictly *after* the notebook's
  training window.  These are the most reliable indicators of production
  performance.

### Evaluating IS vs OOS
```python
import pandas as pd
from src.metrics import qlike, rmse

log = pd.read_csv("outputs/forecast_log.csv", parse_dates=["target_date"])
log = log.dropna(subset=["realized_rv5"])  # keep only rows with realised data

for split in ["IS", "OOS"]:
    sub = log[log["split"] == split]
    y   = sub["realized_rv5"].values
    ens = sub["ensemble_forecast"].values
    gch = sub["garch_forecast"].values
    print(f"\n{split} ({len(sub)} rows)")
    print(f"  Ensemble  QLIKE={qlike(y, ens):.4f}  RMSE={rmse(y, ens):.6f}")
    print(f"  GARCH     QLIKE={qlike(y, gch):.4f}  RMSE={rmse(y, gch):.6f}")
```

A well-functioning pipeline should show:
* **IS QLIKE** slightly better than OOS (model saw some of the training data).
* **OOS QLIKE** close to the notebook's reported −8.768 for XGBoost.
* **OOS RMSE** improvement over GARCH (notebook: 16% reduction, RMSE 0.002334 vs 0.002793).

---

## Scheduling (cron example)

```cron
# Run at 18:00 EST every weekday (after US market close + yfinance update)
0 18 * * 1-5 cd /path/to/Volatility-Forecasting && python -m pipeline.daily_pipeline >> outputs/cron.log 2>&1
```

---

## Files

```
pipeline/
├── daily_pipeline.py   # Full daily retrain + forecast
├── backfill.py         # Historical backfill over last 2 years
└── README.md           # This file

outputs/                # Created automatically at first run
├── forecast_log.csv    # Persistent forecast record
└── pipeline.log        # INFO-level run logs
```
