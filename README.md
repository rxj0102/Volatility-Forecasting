# Volatility Forecasting — SPY Multi-Model Engine

> Can gradient-boosted trees reliably outperform GARCH on next-day realized volatility?  
> A rigorous ML pipeline with strict temporal validation, regime detection, and a robust stacking ensemble.

---

## Motivation

Accurate short-horizon volatility forecasts are central to options pricing, risk management, and dynamic portfolio construction. GARCH(1,1) remains the industry workhorse, yet it imposes strong parametric assumptions and cannot incorporate cross-asset signals or non-linear interactions. This project benchmarks a suite of ML models against an expanding-window GARCH(1,1)-t baseline on SPY (2014–2024), with a focus on avoiding the subtle look-ahead biases that invalidate most published ML-vol results.

---

## Methodology

### Data
- **Asset:** SPY ETF — daily OHLCV, 2014–2024 (~2,500 trading days)
- **Target:** 5-day realized volatility, shifted forward by one day (`Target_RV5`)
- **Cross-asset signals:** VIX (^VIX), VIX3M term structure, HYG/IEF credit stress

### Volatility Estimators
| Estimator | Formula | Property |
|---|---|---|
| Garman-Klass | `√(0.5·ln²(H/L) − (2ln2−1)·ln²(C/O))` | ~7× lower variance than close-to-close |
| Parkinson | `√(ln²(H/L) / 4ln2)` | Range-only intraday stress proxy |
| GARCH(1,1)-t | `σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}` | Expanding window, monthly refit, t-distribution innovations |

### Feature Engineering
- **VIX/VIX3M term structure** — forward-looking implied vol and fear slope
- **VRP (Variance Risk Premium)** — GARCH vol minus implied vol; sign predicts vol direction
- **Fractional Differentiation** (López de Prado 2018) — achieves stationarity while preserving long-range memory in log-price
- **GMM Regime Labels** — BIC-selected Gaussian mixture on (RV5, VIX); one-hot encoded to avoid ordinal bias in trees
- **Lookback windows** — 20-day (≈1 trading month) rolling feature matrix flattened per sample

### Models
| Model | Key Design Choice |
|---|---|
| GARCH(1,1)-t | Expanding window; σ̂_{t\|t-1} prevents same-day leakage |
| XGBoost | Log-vol target → forecasts strictly positive after `exp()` |
| LightGBM GBDT | Wider leaves (63) for genuine diversity from XGBoost |
| Entropy XGBoost | Monotone constraints encode leverage effect as hard prior |
| Quantile Regression | Q10/Q50/Q90 with isotonic sort; 80% PI coverage target |
| Stacking (Ridge meta) | Manual OOF stacking; meta-learner sees only base predictions |
| **Sovereign Prime** | Huber (ε=1.35) meta-learner; L1 tail behaviour during vol spikes |

### Validation Protocol
- **80/20 strict chronological split** — no random shuffling, no cross-period contamination
- **5-fold TimeSeriesSplit** with 5-day embargo to match the 5-day RV aggregation horizon
- **RobustScaler** fit on training data only (prevents the March 2020 spike from compressing all other values)
- **Primary metric:** QLIKE (standard in the volatility literature; asymmetrically penalises underprediction)

---

## Key Results

| Model | RMSE | QLIKE | MAPE |
|---|---|---|---|
| GARCH(1,1)-t | baseline | baseline | baseline |
| XGBoost | ↓ ~8% | ↓ ~5% | ↓ ~7% |
| LightGBM GBDT | ↓ ~9% | ↓ ~6% | ↓ ~8% |
| Entropy (monotone) | ↓ ~9% | ↓ ~6% | ↓ ~8% |
| Sovereign Prime | ↓ ~10% | ↓ ~8% | ↓ ~10% |
| Q50 median | ↓ ~7% | ↓ ~4% | ↓ ~6% |
| 80% PI coverage | — | — | ≈80% ✓ |

> **Finding:** GARCH is a formidable benchmark. ML gains are real but modest (~5–10% in QLIKE). Monotone constraints (Entropy model) offer the best interpretability–performance tradeoff. Huber meta-learning is the most stable across vol regimes.

---

## Tech Stack

| Category | Library |
|---|---|
| Data | `yfinance`, `pandas` |
| Econometrics | `arch`, `statsmodels` |
| ML | `scikit-learn`, `xgboost`, `lightgbm` |
| Numerics | `numpy`, `scipy` |
| Visualisation | `matplotlib`, `seaborn` |
| Serialisation | `joblib` |

---

## Project Structure

```
Volatility-Forecasting/
├── notebooks/
│   └── 01_volatility_forecasting.ipynb   # End-to-end research notebook
├── src/
│   ├── metrics.py                         # rmse, mae, qlike, mape, report
│   ├── feature_engineering.py             # frac_diff, lookback windows, CV utils
│   └── models.py                          # constrained_weights, monotone constraints
├── tests/
│   ├── test_metrics.py
│   └── test_feature_engineering.py
├── models/                                # Serialised model artefacts (gitignored)
├── data/                                  # Downloaded at runtime via yfinance
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/rxj0102/volatility-forecasting.git
cd volatility-forecasting

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the tests
pytest tests/ -v

# 4. Launch the notebook
cd notebooks
jupyter lab 01_volatility_forecasting.ipynb
```

Data is downloaded automatically from Yahoo Finance on the first run. No API keys required.

---

## Future Improvements

- **HAR-RV benchmark** — HAR (Heterogeneous Autoregressive Realized Volatility) often outperforms GARCH in pure vol forecasting; a fairer comparison
- **Options market features** — VIX term structure, skew, and put/call ratio for richer forward-looking signals
- **Online learning** — incremental model updates as new regime data arrives (Mondrian conformal intervals)
- **Transaction-cost-aware evaluation** — backtest a vol-targeting strategy (e.g. 10% annualised target) using model forecasts as position sizing signals
- **Conformal prediction intervals** — distribution-free coverage guarantees without parametric assumptions

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
- Hansen, P. R., & Lunde, A. (2005). A forecast comparison of volatility models. *Journal of Applied Econometrics*, 20(7), 873–889.
- Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*, 53(1), 67–78.
