# Outline: Machine Learning vs. Econometric Approaches to Equity Volatility Forecasting

---

## I. Research Question

**Central question:** Can machine learning models — particularly gradient-boosted trees with domain-constrained priors and fractional feature engineering — produce economically and statistically meaningful improvements over the GARCH(1,1)-t benchmark for next-day 5-day realized volatility (RV5) forecasting on US equities?

**Sub-questions:**
1. Does fractional differentiation (d ∈ (0,1)) preserve financially relevant long-memory properties while achieving stationarity, compared to integer differencing?
2. Do hard monotone constraints (leverage effect, volatility clustering) improve out-of-sample QLIKE without sacrificing full interpretability?
3. Does a leakage-aware stacking ensemble with purged/embargoed walk-forward cross-validation outperform its base models?
4. Is the median quantile forecast (Q50) a better point forecast than a mean-targeting model under QLIKE loss?
5. Do stacking ensembles underperform base models in volatility forecasting, consistent with Rapach & Zhou (2013)?

---

## II. Data

### A. Primary Asset
- **Instrument:** SPY ETF (S&P 500 proxy)
- **Frequency:** Daily OHLCV
- **Raw date range:** 2014–2024 (~2,510 trading days before NA removal)
- **Usable sample after feature engineering:** 2,421 trading days
- **Source:** `yfinance` (Yahoo Finance API)

### B. Cross-Asset / Macro Signals
- **VIX (^VIX):** CBOE 30-day implied volatility index; converted to daily decimal scale as `VIX / (100 × √252)`
- **VIX3M (^VIX3M):** 3-month implied volatility; used to construct term-structure signal
- **HYG:** iShares iBoxx High-Yield Corporate Bond ETF
- **IEF:** iShares 7–10 Year Treasury Bond ETF
- All downloaded via `yfinance`

### C. Train / Test Split
- **Method:** Chronological 80/20 split (no shuffling)
- **Training set:** 2015-05-15 → 2023-01-23 (1,936 samples, ~8 years)
- **Test set:** 2023-01-24 → 2024-12-26 (485 samples, ~2 years)

---

## III. Target Variable

- **Realized Volatility (RV5):** 5-day rolling standard deviation of log returns:
  `RV5_t = std(r_{t-4}, ..., r_t)`
- **Forecast target:** `Target_RV5_t = RV5_{t+1}` (forward-shifted by 1 day — strictly next-day forecast)
- **Log-vol transformation:** Models trained on `log(Target_RV5)`, predictions exponentiated back; guarantees positive forecasts

---

## IV. Volatility Estimators (Features)

### A. Close-to-Close Log Return
- `r_t = ln(Close_t / Close_{t-1})`

### B. Garman-Klass Estimator (GK_Vol)
- Formula: `σ_GK = √(½·ln²(H/L) − (2ln2−1)·ln²(C/O))`
- Uses full OHLC bar; approximately 7× lower variance than close-to-close
- Reference: Garman & Klass (1980)

### C. Parkinson Estimator (Park_Vol)
- Formula: `σ_Park = √(ln²(H/L) / (4 ln 2))`
- Range-only estimator; efficient under Brownian motion with no drift

### D. Overnight Gap (Gap)
- Formula: `Gap_t = |ln(Open_t / Close_{t-1})|`
- Captures discrete jumps and after-hours information

### E. GARCH(1,1)-t Conditional Volatility (GARCH_Vol)
- Expanding-window GARCH(1,1) with Student-t innovations, refitted every 21 trading days
- Used as a feature representing the econometric vol forecast

---

## V. Feature Engineering

### A. Technical / Price-Based Features
| Feature | Construction |
|---------|-------------|
| RSI | 14-period Relative Strength Index |
| BB_Width | 4 × rolling 20-day std of log returns (Bollinger Band width) |
| Return_Mom | Rolling 5-day mean of log returns |
| Vol_of_Vol | Rolling 20-day std of GK_Vol |
| Skewness | Rolling 60-day skewness of log returns |

### B. Cross-Asset / Macro Features
| Feature | Construction |
|---------|-------------|
| VIX_Daily | `VIX / (100 × √252)` — annualized % → daily decimal |
| VRP_Signal | `GARCH_Vol − VIX_Daily` (Variance Risk Premium) |
| Credit_Stress | `HYG / IEF` ratio |
| Fear_Slope | `VIX / VIX3M` (implied vol term structure) |

### C. Fractional Differentiation
- **Method:** Granger-Joyeux (1980) fractional differencing via binomial expansion `(1−L)^d`
- **Weight recurrence:** `w_k = −w_{k-1} · (d − k + 1) / k`, truncated at `thres = 1e-5`
- **Key property for d ∈ (0,1):** Weights decay hyperbolically (`|w_k| ~ k^{-(d+1)}`), preserving long-range dependence; integer differencing (d=1) decays exponentially (destroys memory)
- **Optimal d search:** Grid search over d ∈ [0.1, 1.0] (10 steps) on training data only; minimum d such that ADF p-value ≤ 0.05
- **Result for SPY log-price:** `d = 0.6`, ADF p-value = 0.0007
- **Feature produced:** `FracDiff_Close`
- References: Granger & Joyeux (1980); López de Prado (2018)

### D. Regime Features (GMM)
- **Model:** Gaussian Mixture Model on `(RV5, VIX_Daily)` fitted on training fold only
- **Component selection:** BIC criterion, tested range 2–7 components
- **Result:** Optimal = 2 regimes; regimes sorted by mean RV5 (regime 0 = low-vol)
- **Features produced:** `Regime_0`, `Regime_1` (one-hot dummies)

### E. Lookback Window Flattening
- **Lookback window:** 20 trading days (~1 calendar month)
- **Method:** For each day t, concatenate feature values from t−19 through t into a single row (3D panel → 2D matrix)
- **Final feature matrix shape:** (2,421 − 20) × (18 features × 20 lags) = 2,401 × 360

### F. Scaling
- **Scaler:** `RobustScaler` (IQR-based) — robust to outlier vol spikes (March 2020)
- Fitted on training set only; applied to train and test

---

## VI. Models

### A. Econometric Baseline: GARCH(1,1)-t
- **Specification:** GARCH(1,1) with Student-t innovations
- **Training regime:** Expanding window with monthly refit (every 21 trading days)
- **Leakage fix:** Forecast is `σ̂_{t|t-1}` (yesterday's conditional variance forecast), not same-day; prevents incorporating same-day shock ε_t
- **Units:** Returns estimated in percentage units; output converted to daily decimal vol
- **OOS result:** RMSE=0.002793, MAE=0.002167, QLIKE=−8.7463, MAPE=40.89%

### B. Machine Learning Models (trained on log-vol target)

#### B1. XGBoost (Base)
- **Objective:** `reg:squarederror` on log(RV5)
- **Hyperparameters:** n_estimators=300, learning_rate=0.04, max_depth=4, subsample=0.8, colsample_bytree=0.8, reg_lambda=2, tree_method='hist', random_state=42
- **OOS result:** RMSE=0.002334, MAE=0.001770, QLIKE=−8.7678, MAPE=28.26%
- **Improvement over GARCH:** 16.4% lower RMSE, 12.5pp lower MAPE

#### B2. LightGBM GBDT
- **Boosting type:** GBDT (standard gradient boosting)
- **Design note:** DART mode discarded — causes systematic underfitting on regression (observed MAPE ≈77% in ablation); DART uses all trees at inference despite dropout training
- **Hyperparameters:** n_estimators=400, learning_rate=0.04, num_leaves=63, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=2, reg_alpha=0.1
- **Diversity vs XGB:** num_leaves=63 vs XGB's effective ~16 (via max_depth=4) — genuine architectural diversity for ensemble
- **OOS result:** RMSE=0.002441, MAE=0.001829, QLIKE=−8.7429, MAPE=28.86%

#### B3. Entropy XGBoost (Monotone-Constrained)
- **Key innovation:** Hard monotone constraints embedded in tree-split rules (domain priors as architecture, not regularization)
- **Constraints:**
  - `GK_Vol, Park_Vol, Gap → +1` (higher realized vol implies higher forecast — volatility clustering)
  - `Log_Ret → −1` (leverage effect — negative returns raise vol)
  - All other features → 0 (unconstrained)
- **Constraint application:** Same constraint applied at every lag position in the lookback window (interleaved tuple of length 360)
- **Hyperparameters:** n_estimators=350, learning_rate=0.03, max_depth=4, reg_lambda=3
- **OOS result:** RMSE=0.002511, MAE=0.001885, QLIKE=−8.7358, MAPE=30.53%
- **Tradeoff:** ~0.032% QLIKE drop vs XGBoost; gains full financial interpretability guarantee

#### B4. Quantile Regression (Q10, Q50, Q90)
- **Method:** Three separate XGBoost models with `objective='reg:quantileerror'`
- **Quantiles:** τ = 0.10, 0.50, 0.90
- **Ordering fix:** Isotonic sort per row enforces `q10 ≤ q50 ≤ q90` (prevents crossing)
- **Q50 OOS result:** RMSE=0.002156, MAE=0.001607, QLIKE=−8.8054, MAPE=25.62%
- **Coverage check:** Empirical 80% interval (Q10–Q90) coverage ≈80% (matches theoretical)
- **Note:** Q50 achieves best QLIKE and best RMSE among all point forecasts; median is robust to heavy tails

### C. Stacking Ensembles

#### C1. Stacking — Ridge Meta-Learner
- **Base models:** XGBoost, Entropy XGBoost, LightGBM GBDT (3 models)
- **OOF procedure:** Manual generation (not `sklearn.StackingRegressor`) — sklearn's `cross_val_predict` with `TimeSeriesSplit` violates the requirement that each sample appears in exactly one fold (expanding windows overlap); manual OOF avoids this
- **CV scheme:** 5-fold `TimeSeriesSplit`, embargo=5 (last 5 train-fold samples dropped before fitting each base model to prevent label overlap with 5-day RV target)
- **OOF matrix:** Shape (1,936 × 3)
- **Meta-learner:** `RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])` on OOF signals only (no raw feature pass-through)
- **Selected alpha:** 0.1
- **OOS result:** RMSE=0.003313, MAE=0.002547, QLIKE=−8.5366, MAPE=39.61%
- **Finding:** Underperforms all three base models on all four metrics

#### C2. Sovereign Prime — Huber Meta-Learner
- **Design:** Meta-learner operates in volatility space (not log-vol) to allow Huber robustness to act on actual vol magnitudes
- **Meta-learner:** `HuberRegressor(epsilon=1.35, max_iter=500)`
- **ε=1.35 rationale:** Transition from L2 to L1 at 1.35σ dampens influence of vol-spike observations
- **Same OOF base as Ridge meta**
- **OOS result:** RMSE=0.003535, MAE=0.002791, QLIKE=−8.4336, MAPE=48.09%
- **Finding:** Worst-performing model; Huber robustness does not overcome ensemble aggregation losses

### D. HAR-RV (Implemented but Not Integrated in Notebook)
- **Specification (Corsi 2009):**
  `RV_{t+1} = c + β_D·RV^(d)_t + β_W·(1/5)·Σ_{j=0}^{4}RV_{t-j} + β_M·(1/22)·Σ_{j=0}^{21}RV_{t-j} + ε_{t+1}`
- **Economic motivation:** Heterogeneous Market Hypothesis — daily traders, weekly funds, monthly institutions each produce distinct vol components
- **Estimation:** OLS on expanding window (burnin=252 days)
- **Implementation:** `src/models.py::har_rv()` (213 lines), fully tested (6 test cases), but not called in the main notebook
- **Status:** Proposed extension; benchmark comparison not reported in notebook results
- References: Andersen et al. (2003); Hansen & Lunde (2005) — HAR-RV consistently outperforms GARCH(1,1) when realized measures are available

---

## VII. Statistical Tests

### A. ADF Stationarity Test (used in feature engineering)
- **Purpose:** Verify that fractionally differenced series achieves stationarity
- **Threshold:** p-value ≤ 0.05
- **Result for FracDiff_Close (d=0.6):** ADF p-value = 0.0007 (highly significant)
- Implemented via `statsmodels.tsa.stattools.adfuller`

### B. Diebold-Mariano Test (implemented, not called in notebook)
- **Purpose:** Test equal predictive accuracy between two models
- **H₀:** `E[L(e₁_t) − L(e₂_t)] = 0`
- **Test statistic:** `DM = d̄ / √(V̂(d̄))` where d̄ is mean loss differential
- **Long-run variance:** Newey-West with bandwidth `max(0, h−1)` (h=forecast horizon; critical for overlapping forecasts)
- **Loss functions supported:** QLIKE and MSE
- **Critical design note (Patton 2011, Theorem 1):** QLIKE-DM produces valid rankings under noisy volatility proxy; MSE-DM rankings can be reversed by proxy noise — unreliable for vol forecasting
- **Implementation:** `src/metrics.py::diebold_mariano()` (152 lines), 8 test cases
- **Status:** Implemented and tested; not reported in notebook results (proposed extension)

### C. Heteroscedasticity Diagnostic
- **Method:** Spearman rank correlation between |residuals| and predicted volatility
- **Purpose:** Check whether residual variance is still predictable (model misspecification)
- **Reported:** Notebook Section 7b

### D. Ljung-Box Autocorrelation Test
- **Purpose:** Test for residual autocorrelation (forecast inefficiency)
- **Lags tested:** 5 and 10
- **Reported:** Notebook Section 7b

### E. BIC Model Selection (GMM Regime Detection)
- **Purpose:** Select optimal number of GMM components
- **Range tested:** 2–7 components
- **Result:** Optimal = 2 regimes

---

## VIII. Evaluation Metrics

### A. RMSE (Root Mean Squared Error)
- `RMSE = √(E[(y − ŷ)²])`
- Symmetric; penalizes large errors quadratically

### B. MAE (Mean Absolute Error)
- `MAE = E[|y − ŷ|]`
- Robust to outliers; equal weighting of all errors

### C. QLIKE Loss (Primary Metric)
- Formula: `QLIKE = E[σ/h − log(σ/h) − 1]` where σ = realized, h = forecast
- Derived from Gaussian quasi-log-likelihood; equivalent to negative log-likelihood of log-normal
- Asymmetrically penalizes underprediction (h < σ) more than overprediction
- **Key property (Patton 2011):** Produces consistent model rankings even when σ (realized vol) is a noisy proxy for true latent vol; MSE does not have this property
- More negative = better (all reported values negative because volatilities are small)

### D. MAPE (Mean Absolute Percentage Error)
- `MAPE = E[|y − ŷ| / y]`
- Scale-free; interpretable as average relative error

---

## IX. Key Results

### A. Full Out-of-Sample Leaderboard (Test Set: Jan 2023 – Dec 2024, n=485)

| Model | RMSE | MAE | QLIKE | MAPE |
|-------|------|-----|-------|------|
| GARCH(1,1)-t | 0.002793 | 0.002167 | −8.7463 | 40.89% |
| XGBoost | 0.002334 | 0.001770 | −8.7678 | 28.26% |
| LightGBM GBDT | 0.002441 | 0.001829 | −8.7429 | 28.86% |
| Entropy (monotone XGB) | 0.002511 | 0.001885 | −8.7358 | 30.53% |
| **Q50 (median quantile)** | **0.002156** | **0.001607** | **−8.8054** | **25.62%** |
| Stacking — Ridge meta | 0.003313 | 0.002547 | −8.5366 | 39.61% |
| Sovereign Prime (Huber) | 0.003535 | 0.002791 | −8.4336 | 48.09% |

### B. Performance Improvements over GARCH Baseline

| Model | RMSE ∆ | QLIKE ∆ | MAPE ∆ |
|-------|---------|---------|--------|
| XGBoost | −16.4% | −0.025 (better) | −12.6pp |
| LightGBM GBDT | −12.6% | +0.003 (worse) | −12.0pp |
| Entropy XGB | −10.1% | +0.010 (worse) | −10.4pp |
| Q50 | −22.8% | −0.059 (better) | −15.3pp |
| Stacking — Ridge | +18.6% | +0.210 (worse) | −1.3pp |
| Sovereign Prime | +26.6% | +0.313 (worse) | +7.2pp |

### C. Critical Findings

1. **GARCH is a formidable baseline:** Low RMSE (0.002793) despite simple specification; any ML model must earn its complexity
2. **XGBoost achieves meaningful RMSE gains (+16%)** with a small QLIKE improvement; best overall ML model for mean forecasting
3. **Q50 (median quantile XGB) dominates on all metrics** — best RMSE (0.002156), best QLIKE (−8.8054), best MAPE (25.62%); median estimation is more robust under asymmetric loss than mean estimation
4. **Monotone constraints cost ~0.032 QLIKE points** but guarantee full financial interpretability (leverage effect direction, vol clustering direction — never violable by splits)
5. **Stacking ensembles fail:** Both meta-learners (Ridge and Huber) underperform every single base model; consistent with Rapach & Zhou (2013) finding on equity return predictability
6. **Fractional differentiation d=0.6** achieves stationarity (ADF p=0.0007) while retaining long-memory; integer differencing (d=1) would over-difference and destroy mean-reversion signal
7. **GMM detects 2 volatility regimes** (BIC-optimal): low-vol and high-vol regimes — regime dummies included as features

### D. Quantile Forecast Coverage
- Empirical 80% interval coverage (Q10–Q90): ≈80% (matches theoretical target)
- Isotonic sorting prevents quantile crossing

### E. Stacking Failure Analysis
- Ridge meta-learner: RMSE 3.313 vs best base 2.334 (+42% worse)
- Huber meta-learner: RMSE 3.535 vs best base 2.334 (+52% worse)
- Root cause (hypothesis): Volatility forecasts are highly correlated across base models; stacking gains require diverse base model errors; in vol-forecasting, all models share the same dominant signal (lagged vol)

---

## X. Implementation Details & Methodological Safeguards

### A. Leakage Prevention
1. **GARCH leakage fix:** Uses `σ̂_{t|t-1}` (shift by 1); never uses same-day ε_t
2. **Fractional differentiation leakage fix:** `find_min_d()` called only on training fold data
3. **Scaler leakage fix:** `RobustScaler` fitted only on training set, then applied to test
4. **GMM leakage fix:** GMM fitted only on training fold features
5. **OOF leakage fix:** Manual OOF with embargo=5 (5 trading days purged from train-fold boundary to prevent 5-day RV target overlap)

### B. Cross-Validation
- **Walk-forward (expanding window) CV:** TimeSeriesSplit, n_splits=5
- **Embargo:** 5 samples removed from end of each training fold (critical: RV5_t and RV5_{t+1} share 4 of 5 returns; leakage without embargo)
- **OOF matrix shape:** (1,936 × 3) for stacking

### C. Software Stack
- Python: numpy, pandas, scipy, statsmodels
- Econometrics: arch (GARCH estimation)
- ML: scikit-learn, xgboost (≥2.0), lightgbm (≥4.1)
- Data: yfinance (≥0.2.36)
- Serialization: joblib

### D. Test Suite
- **Total test cases:** 51 across 3 test files
- `test_metrics.py`: 21 tests (RMSE, MAE, QLIKE, MAPE, DM test, report function)
- `test_feature_engineering.py`: 16 tests (frac_diff, find_min_d, lookback windows, walk-forward CV, OOF)
- `test_models.py`: 14 tests (HAR-RV correctness, QLIKE-optimal weights, monotone constraints)
- Execution: `pytest tests/ -v`

---

## XI. Literature Connections

| Claim | Reference |
|-------|-----------|
| QLIKE consistent under noisy vol proxy | Patton (2011), Theorem 1 |
| MSE-DM rankings unreliable under noisy proxy | Patton (2011) |
| Diebold-Mariano equal predictive accuracy test | Diebold & Mariano (1995) |
| HAR-RV multi-scale volatility model | Corsi (2009) |
| HAR-RV vs GARCH benchmarking | Andersen et al. (2003); Hansen & Lunde (2005) |
| Fractional differencing long-memory | Granger & Joyeux (1980) |
| Fractional diff. for ML features | López de Prado (2018) |
| Stacking underperforms in return prediction | Rapach & Zhou (2013) |
| Garman-Klass volatility estimator | Garman & Klass (1980) |

---

## XII. Proposed Extensions (Not Yet Implemented)

1. **HAR-RV benchmark:** `har_rv()` is implemented in `src/models.py` and tested (6 cases), but not integrated into the main notebook pipeline — no OOS performance numbers reported; adding it would complete the econometric baseline comparison
2. **Diebold-Mariano significance tests:** `diebold_mariano()` is implemented in `src/metrics.py` (8 test cases), but not called in the notebook — all leaderboard comparisons are point estimates without statistical significance
3. **Volatility-targeting backtest:** No economic value evaluation; a volatility-targeting portfolio strategy (risk parity weight = k / σ̂_t) would translate QLIKE improvements into Sharpe ratio or VaR backtests
4. **Higher-frequency data / realized measures:** Intraday 5-minute returns would allow true realized variance computation (not close-to-close proxy), enabling cleaner HAR-RV and sharper DM tests
