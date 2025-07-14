# Improving S&P 500 Volatility Forecasting through Regime-Switching Methods

**Ava Blake · Nivika Gandhi · Anurag Jakkula**  
*Columbia Summer Undergraduate Research Experiences in Mathematical Modeling*  
*Advised by George Dragomir, Vihan Pandey, Dobrin Marchev*

---

## Motivations

Market volatility forecasts are essential for portfolio management, risk control, derivatives pricing, and regulatory decisions. Traditional models like the Heterogeneous Autoregressive (HAR) model use past volatility over different time horizons (daily, weekly, monthly) to predict future volatility, assuming stable market dynamics with fixed parameters.

However, financial markets often move through distinct regimes, such as calm, crisis, and recovery phases, that cause volatility behavior to change over time. In these settings, the statistical relationship between past and future volatility is not constant, and fixed-parameter models may fail to capture such evolving patterns, reducing forecast accuracy.

Our research addresses these challenges by developing regime-aware forecasting frameworks that detect structural breaks and adapt to changing market conditions. By combining historical realized volatility with forward-looking indicators like the Implied Volatility Index (VIX), our models adjust dynamically to different volatility regimes, improving predictive performance.

Ultimately, this work enhances the robustness and reliability of volatility forecasts, supporting better-informed real-time decisions in risk management and hedging.

---

## Background

The original HAR model, introduced by Corsi (2009), was designed to capture realized volatility (RV) behavior across multiple time scales (daily, weekly, and monthly) reflecting the activity of different types of investors:

```
RV_t = β₀ + β_d · RV_{t-1} + β_w · RV̄_{t-1}^{(w)} + β_m · RV̄_{t-1}^{(m)} + ε_t
```

where β₀, β_d, β_w, β_m are the HAR parameters to be estimated, and ε_t is the error term.

Subsequent research has shown that while the HAR model captures important long-memory features, volatility dynamics often shift across distinct market regimes, exhibiting structural breaks and regime-dependent behavior. To better capture these time-varying patterns, regime-switching extensions of HAR have been proposed. For instance, Zhang et al. demonstrate that incorporating regime switching enhances forecasts of Chinese stock market volatility by accounting for structural shifts driven by international markets.

---

## Data Collection

Our dataset consists of high-frequency intraday price data for the S&P 500 index (SPX), spanning eleven years from June 2, 2014 to April 29, 2025, sourced via Bloomberg Terminal. Using 5-minute closing prices, we calculate intraday log-returns as:

```
r_{t,i} = ln(P_{t,i} / P_{t,i-1})
```

These returns are then aggregated to compute daily realized volatility (RV), adjusted to account for shorter trading days:

```
RV_t = sqrt(N/n * Σ r_{t,i}²)
```

Here, n is the number of intraday returns observed on day t (which may vary due to holidays), and N is the standard number of returns in a full trading day—78 for intraday data only, or 79 including overnight returns. This scaling ensures comparability of RV across days with varying lengths.

---

## Feature Engineering

We extend the HAR model by introducing a dual-memory structure that captures both historical volatility patterns and forward-looking market sentiment. Our model applies HAR-style lags to implied volatility features (VIX), allowing the model to respond to shifts in investor expectations.

To build this structure, we engineered:

- Lagged VIX values (daily, 5-day, 22-day): market fear over varying time horizons  
- Short-term reversal factor (STR): captures recent return reversals and mean reversion  
- Realized kurtosis: measures tail risk and extreme return behavior  
- Jump variation: isolates large discontinuous price moves from continuous volatility  

This feature design allows the model to adapt across regimes by integrating both behavioral and structural signals, improving forecast accuracy under changing market conditions.

---

## Methodology Overview

1. **Soft Markov Regime Switching**  
   Gaussian HMM on smoothed volatility → soft regime probabilities → weighted WLS/Ridge in EM loop → forward-propagated regime weights for smooth forecasts.

2. **Distributional Clustering with Spectral-XGBoost**  
   Mood test + Wasserstein clustering → HAR models per cluster → XGBoost for test-time regime prediction.

3. **Coefficient-Based Soft Clustering**  
   HAR coefficients → PCA + BGMM → soft regime weights → WLS per regime → XGBoost for smooth forecasts.

---

## Model Comparisons

Model performance is evaluated across Pre-COVID, COVID, and Post-COVID periods using MAPE and MSE metrics. Regime-aware models consistently outperform the baseline HAR model.

---

## Significance

Regime-aware HAR extensions consistently improve predictive accuracy over standard HAR. Coefficient-based clustering performs best pre/post-COVID, while distributional clustering excels during COVID. Inclusion of VIX enhances sensitivity to sentiment.

These results reinforce volatility’s regime-dependence and motivate future exploration of LSTMs or ensemble hybrid methods.

---

## References

```
@article{corsi,
  title={A Simple Approximate Long-Memory Model of Realized Volatility},
  author={Corsi, Fulvio},
  journal={Journal of Financial Econometrics},
  year={2009}
}

@article{zhang_Chinese_International_regime_switching,
  author = "Y. Zhang and L. Lei and Y. Wei",
  title = "Forecasting the Chinese stock market volatility with international market volatilities: The role of regime switching",
  journal = "The North American Journal of Economics and Finance"
}
```

---
