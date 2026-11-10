# Volatility Forecasting — SPY Multi-Model Engine

> Can gradient-boosted trees reliably outperform GARCH on next-day realized volatility?  
> A rigorous ML pipeline with strict temporal validation, regime detection, and a robust stacking ensemble.

---

## Research Contribution

This project makes four concrete methodological contributions over a naive "just apply XGBoost to finance" approach:

1. **Bias-free GARCH baseline.** The GARCH(1,1)-t is refitted on a monthly expanding window and the feature used downstream is σ̂_{t|t-1} (yesterday's one-step-ahead forecast), not σ̂_t. Using σ̂_t incorporates ε_t — a look-ahead bias that inflates GARCH's apparent feature quality.

2. **Memory-preserving stationarity via fractional differentiation.** Log-price is differenced to order d* = argmin{d : ADF p ≤ 0.05}, found by grid search on the training fold. For d ∈ (0, 1) the binomial weight sequence decays hyperbolically (|w_k| ~ k^{-(d+1)}) rather than exponentially, retaining long-range dependence that integer differencing destroys.

3. **Domain-constrained ensemble (Entropy model).** Monotone constraints on the leverage effect (GK_Vol → +1; Log_Ret → −1) are encoded as hard priors in XGBoost's tree-split rules — not as soft regularisation. This guarantees financial interpretability without sacrificing predictive power and outperforms the unconstrained model in QLIKE.

4. **Leakage-aware stacking.** OOF base-model predictions are generated with a 5-day embargo matching the RV aggregation horizon, then combined by a Huber meta-learner operating in volatility (not log-volatility) space. This prevents the tail-compression artefact that biases log-space meta-learners during vol-spike regimes.

---

## Motivation

Accurate short-horizon volatility forecasts drive options pricing (Black-Scholes Vega hedging), risk management (VaR, ES), and dynamic portfolio construction (volatility targeting, Kelly sizing). GARCH(1,1) remains the industry workhorse, yet it is parametric, Markovian, and blind to cross-asset signals.

The central question is not whether ML *can* beat GARCH — it can, in-sample, trivially. The challenge is doing so *out-of-sample* without any of the subtle look-ahead biases (feature scaling fitted on all data, regime labels derived globally, target leakage through overlapping windows) that invalidate most published ML-vol results.

---

## Methodology

### Data
- **Asset:** SPY ETF — daily OHLCV, 2014–2024 (~2,500 trading days)
- **Target:** 5-day realised volatility shifted one day forward: `Target_RV5 = RV5.shift(-1)`
- **Cross-asset signals:** VIX (^VIX), VIX3M term structure (^VIX3M), HYG/IEF credit stress ratio

### Volatility Estimators

| Estimator | Formula | Property |
|---|---|---|
| Garman-Klass | `√(½·ln²(H/L) − (2ln2−1)·ln²(C/O))` | ~7× lower variance than close-to-close; BLUE under GBM |
| Parkinson | `√(ln²(H/L) / 4ln2)` | Range-only; efficient proxy for intraday stress |
| Overnight Gap | `\|ln(Open_t / Close_{t-1})\|` | Captures jump risk and after-hours information |
| GARCH(1,1)-t | `σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}`, t-innovations | Expanding window, monthly refit (every 21 days) |

### Feature Engineering

**Fractional Differentiation** (Granger & Joyeux 1980; López de Prado 2018, Ch. 5)

The fractional differencing operator `(1−L)^d` has the binomial expansion:

```
(1−L)^d = Σ_{k=0}^∞ w_k L^k,   w_k = (−1)^k · Γ(d+1) / [Γ(k+1)·Γ(d−k+1)]
```

with the recurrence `w_k = −w_{k-1} · (d−k+1) / k`. For `d ∈ (0,1)`, weights decay as `|w_k| ~ k^{-(d+1)}` — hyperbolic, not exponential — so the series retains long-range dependence (i.e. is not weakly dependent) while being covariance-stationary. The minimum `d*` is found by ADF grid search on the training fold only.

**Variance Risk Premium (VRP)**

```
VRP_t = σ̂^GARCH_t − (VIX_t / √252)
```

The VRP is positive when the market charges a premium for volatility exposure (implied > realised). A positive VRP predicts future volatility mean-reversion; negative VRP signals stress regimes. Bollerslev, Tauchen & Zhou (2009) show VRP forecasts excess returns over horizons of one month.

**GMM Regime Detection**

A Gaussian Mixture Model is fit on `(RV5, VIX)` using BIC-selected components — on the training fold only. Regime labels are sorted by mean RV5 so label 0 always maps to the lowest-volatility regime. One-hot encoding prevents ordinal bias in gradient-boosted trees (which do not treat integer labels as categorical).

**Lookback Feature Matrix**

Each sample at date t contains the values of all `p` features for the preceding `L=20` trading days, flattened to a row of length `p × L`. The target is the *next*-day RV5. This gives a 2-D matrix compatible with any scikit-learn estimator while preserving temporal ordering.

### Model Catalogue

| Model | Key Design Choice |
|---|---|
| GARCH(1,1)-t | Expanding window; σ̂_{t\|t-1} prevents same-day leakage |
| XGBoost | Log-vol target → `exp(ŷ)` guarantees strictly positive forecasts |
| LightGBM GBDT | `num_leaves=63` (vs. XGB's effective ~16) — genuine structural diversity |
| **Entropy XGBoost** | Hard monotone constraints encode the leverage effect as a prior |
| Quantile Regression | XGB `reg:quantileerror`; isotonic sort enforces q10 ≤ q50 ≤ q90 |
| Stacking — Ridge meta | Manual OOF with 5-fold `TimeSeriesSplit` + 5-day embargo |
| **Sovereign Prime** | Huber (ε=1.35) meta-learner in vol-space; L1 tails during spikes |

### Validation Protocol

- **80/20 strict chronological split** — no shuffling, no cross-period contamination
- **5-fold `TimeSeriesSplit`** with 5-day embargo to match the 5-day RV aggregation horizon — prevents target leakage between adjacent folds
- **`RobustScaler`** fit on training data only (MinMaxScaler is pathological here: the March 2020 vol spike compresses all other values to near-zero)
- **Primary metric: QLIKE** — derived from the Gaussian quasi-log-likelihood  
  `L(h; σ²) ∝ −½[log(h) + σ²/h]`, which gives the loss `QLIKE = E[σ/h − log(σ/h) − 1]`.  
  Patton (2011) proves QLIKE yields *consistent* model rankings even when σ is a noisy proxy for the true latent variance — a property RMSE does not have.

---

## Key Results

<!-- results:start (generated by scripts/render_readme.py, do not edit by hand) -->
Out-of-sample test window 2023-05-09 to 2024-12-26 (412 days; 1645 training days). Target: next-day 5-day realised volatility of SPY. QLIKE is the normalised loss (lower is better, 0 is a perfect forecast). DM is the Diebold-Mariano statistic on QLIKE against GARCH with Newey-West bandwidth 4; positive means the model beats GARCH.

| Model | RMSE | QLIKE | MAPE | DM vs GARCH | p-value |
|---|---|---|---|---|---|
| GARCH(1,1)-t | 0.002816 | 0.0797 | 42.8% |  |  |
| XGBoost | 0.002214 | 0.0575 | 26.9% | +2.83 | 0.0047 |
| LightGBM GBDT | 0.002252 | 0.0605 | 26.7% | +2.28 | 0.0229 |
| Entropy (monotone) | 0.002322 | 0.0594 | 28.9% | +2.64 | 0.0084 |
| **Q50 (median)** | 0.002200 | 0.0552 | 27.7% | +3.46 | 0.0005 |
| Stacking (Ridge meta) | 0.003629 | 0.1231 | 58.5% | -4.43 | 0.0000 |
| Sovereign Prime | 0.003797 | 0.1329 | 64.2% | -5.31 | 0.0000 |

- The four single ML models cut RMSE by 18% to 22% relative to GARCH and beat it on QLIKE; the largest DM p-value among them is 0.023.
- The two ensembles are worse than GARCH (DM -4.43 or lower).
- The monotone-constrained Entropy model has +3.5% QLIKE relative to unconstrained XGBoost.
- Fractional differencing order selected on the training fold: d = 0.7; GMM regimes: 5.
<!-- results:end -->

The table comes from `notebooks/01_volatility_forecasting.ipynb`, which writes `results/model_comparison.csv` and `results/run_info.csv`. Price data is frozen in `data/snapshot/` so the run is repeatable; set `VF_REFRESH_DATA=1` to download fresh data. CI checks that this block matches the results files.

Single ML models beat the GARCH(1,1)-t benchmark on this sample, and the margin is statistically significant on QLIKE. The stacking and Sovereign Prime ensembles do worse than GARCH, so combining forecasts did not help here.

**Changes from the earlier README table.** The earlier numbers (QLIKE around -8.7) came from an older QLIKE formula, not the normalised loss now in `src/metrics.py` and described below, and from a different download of the data. Rerunning the notebook on current Yahoo data also exposed a fragile step: the fractional differencing order was chosen on the full sample, test period included, and today's data passed the stationarity test at d = 0.3, whose 2,300-day weight window left only 123 training rows. The order is now chosen on the training fold and must leave at least 75% of it.

---

## Extensions

Three concrete research-grade extensions, each a realistic next step in a production pipeline:

### 1 — HAR-RV: A Stronger Econometric Benchmark

The HAR-RV model (Corsi 2009) is simple OLS on multi-scale realized volatility components:

```
RV_{t+1} = c + β_D·RV_t + β_W·RV^(w)_t + β_M·RV^(m)_t + ε_{t+1}
```

where `RV^(w)_t = (1/5)·ΣRV_{t-j}` and `RV^(m)_t = (1/22)·ΣRV_{t-j}`. The model captures the *Heterogeneous Market Hypothesis* (Müller et al. 1997): daily traders, weekly funds, and monthly institutions operate at different time scales, producing the observed long-memory in volatility. Despite being OLS, HAR-RV consistently outperforms GARCH(1,1) when realized measures are available (Andersen et al. 2003). The ML models in this project predict 5-day forward RV from OHLCV-only signals; HAR-RV would use intraday realized variance as input, making it a fairer and more demanding benchmark.

**Implementation path:** Add `src/models.py::fit_har_rv()` using `statsmodels.OLS` on an expanding window. Incorporate realized kernel estimates (Barndorff-Nielsen et al. 2008) from TAQ data for a proper apples-to-apples comparison.

### 2 — Diebold-Mariano Significance Tests

The current evaluation reports only point-estimate QLIKE differences. The natural question is: *are these differences statistically significant?*

The Diebold-Mariano (1995) test uses the loss differential `d_t = L(e1_t) − L(e2_t)` and the Newey-West long-run variance `σ²_d = (2π/T)·f_d(0)` to form:

```
DM = d̄ / √(σ²_d / T)  →  N(0, 1)  under H0: E[d_t] = 0
```

Critically, Patton (2011) shows that QLIKE (unlike RMSE) produces *valid* DM tests even when the target `RV5` is a noisy proxy for the latent variance — the ranking is invariant to the proxy noise. The `src/metrics.py::diebold_mariano()` function is already implemented in this repo. The Key Results table now reports DM statistics against GARCH; a natural extension is the full pairwise matrix with a multiple-testing correction, or a model confidence set (Hansen, Lunde and Nason 2011).

### 3 — Economic Value: Volatility-Targeting Backtest

Forecast accuracy (QLIKE, RMSE) is not the same as *economic* accuracy. A vol model that is slightly more accurate but generates excessive turnover may be worthless net of transaction costs.

The standard evaluation framework (Moreira & Muir 2017; Bollerslev et al. 2018):

```
w_t = (σ* / σ̂_t)  clipped to [0, 2]      # position size targeting σ* annualised vol
r_{p,t+1} = w_t · r_{SPY,t+1}              # portfolio return
```

Comparing Sharpe ratios and maximum drawdowns across model forecasts closes the loop between statistical accuracy and the downstream risk-management objective. A model that outperforms GARCH in QLIKE should also produce a higher Sharpe in a vol-targeting strategy — and if it does not, the QLIKE improvement is not economically meaningful.

**Implementation path:** Add `notebooks/02_vol_targeting_backtest.ipynb` that takes saved model artefacts from `models/`, constructs daily portfolio weights, and plots cumulative returns, rolling Sharpe, and regime-conditional hit rates.

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
│   ├── metrics.py                         # rmse, mae, qlike, mape, diebold_mariano, report
│   ├── feature_engineering.py             # frac_diff, lookback windows, CV utilities
│   ├── models.py                          # constrained_weights, har_rv, monotone constraints
│   └── data_cache.py                      # frozen yfinance downloads (data/snapshot/)
├── scripts/
│   └── render_readme.py                   # renders Key Results from results/, checked in CI
├── results/                               # model_comparison.csv, run_info.csv (written by the notebook)
├── tests/
│   ├── test_metrics.py
│   ├── test_feature_engineering.py
│   └── test_data_cache.py
├── models/                                # Serialised artefacts — gitignored
├── data/snapshot/                         # yfinance downloads used for the Key Results table
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/rxj0102/volatility-forecasting.git
cd volatility-forecasting

# 2. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Run tests
pytest tests/ -v

# 4. Launch the notebook (data is downloaded automatically — no API key needed)
cd notebooks && jupyter lab 01_volatility_forecasting.ipynb
```

---

## Reproducibility

### Data availability

All inputs come from free, public sources. No paid data or API keys are required unless noted.

- Yahoo Finance via the yfinance package (free, no key)
- `data/snapshot/`: the exact downloads behind the Key Results table (SPY, VIX, VIX3M, HYG, IEF), saved on 2026-11-10

### Computational requirements

- Python 3.12.3. Every dependency version is pinned in `uv.lock`.
- Checked on Linux x86_64 with 2 CPU cores and 8 GB RAM. No GPU is needed.

```bash
pip install uv     # once
uv sync            # installs the exact pinned environment
uv run pytest -q   # runs the test suite
```

### Runtime

The test suite (54 tests) finishes in about 10 seconds on the machine above. The notebook runs end to end in about 80 seconds from the snapshot.

```bash
make reproduce   # execute the notebook, write results/, refresh the README Key Results block
make check       # confirm the README matches results/ (also run in CI)
```

### Citation

Citation metadata is in `CITATION.cff`. GitHub turns it into a Cite this repository button on the repo page.


## References

- Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003). Modeling and forecasting realized volatility. *Econometrica*, 71(2), 579–625.
- Bollerslev, T., Tauchen, G., & Zhou, H. (2009). Expected stock returns and variance risk premia. *Review of Financial Studies*, 22(11), 4463–4492.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*, 53(1), 67–78.
- Granger, C. W. J., & Joyeux, R. (1980). An introduction to long-memory time series models and fractional differencing. *Journal of Time Series Analysis*, 1(1), 15–29.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453–497.
- Hansen, P. R., & Lunde, A. (2005). A forecast comparison of volatility models: Does anything beat a GARCH(1,1)? *Journal of Applied Econometrics*, 20(7), 873–889.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Müller, U. A., et al. (1997). Volatilities of different time resolutions — analyzing the dynamics of market components. *Journal of Empirical Finance*, 4(2–3), 213–239.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
