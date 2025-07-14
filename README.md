# Improving S&P 500 Volatility Forecasting through Regime-Switching Methods

**Ava Blake · Nivika Gandhi · Anurag Jakkula**  
*Columbia Summer Undergraduate Research Experiences in Mathematical Modeling*  
*Advised by George Dragomir, Vihan Pandey, Dobrin Marchev*

---

## About the Project

This repository contains the results, code, poster, and supporting materials for this research project on forecasting S&P 500 volatility using regime-switching methods. Here, we include more detailed results for comparison, and cover methods such as recursive forecasting which are not detailed in our poster. The project integrates techniques from financial econometrics and machine learning to improve volatility prediction accuracy by accounting for structural changes in market dynamics.

We explore a hybrid modeling framework that combines:

- Regime identification using methods like Hidden Markov Models (HMM), spectral clustering, and nonparametric change point detection (Mood test)
- Volatility modeling via HAR (Heterogeneous Autoregressive) models
- Forecasting enhancement with LSTM and Transformer neural networks
- Supervised learning using XGBoost to predict regime labels
- Clustering techniques to group similar volatility patterns

By incorporating these dynamic regime-based approaches, our models aim to better capture the nonlinear, heteroskedastic structure of financial time series data.

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

! [HAR Image](HAR_Features_Engineered.png)

---

## Methodology Overview

Traditional regime switching models use techniques such as Markov regime-switching, implementing a Hidden Markov Model (HMM) to capture volatility behavior as the market evolves through differing regimes. Our models build off of existing frameworks and introduce new regime-switching techniques.

1. **Soft Markov Regime Switching**  
   Gaussian HMM on smoothed volatility → soft regime probabilities → weighted WLS/Ridge in EM loop → forward-propagated regime weights for smooth forecasts.

We extend HAR by fitting a Gaussian HMM to smoothed volatility and estimating soft regime probabilities. These probabilities are used to weight regime-specific WLS or Ridge regressions within an Expectation-Maximization (EM) loop. Forecasts are blended using forward-propagated regime weights for smooth, robust transitions.

2. **Distributional Clustering with Spectral-XGBoost**  
   Mood test + Wasserstein clustering → HAR models per cluster → XGBoost for test-time regime prediction.

We detect regime shifts using the Mood test and cluster segments via Wasserstein distances and spectral clustering. Each cluster has an HAR model, with XGBoost assigning regimes at test time. This enables structural break adaptation based on feature distributions.

4. **Coefficient-Based Soft Clustering**  
   HAR coefficients → PCA + BGMM → soft regime weights → WLS per regime → XGBoost for smooth forecasts.

We extract HAR coefficients from mood-based segments and cluster them using PCA and BGMM to obtain soft regime weights. These weights inform WLS regressions per regime. XGBoost predicts regime probabilities, allowing for smooth, probabilistic forecasts.

---

## Model Comparisons

Model performance is evaluated across Pre-COVID, COVID, and Post-COVID periods using MAPE and MSE metrics. Regime-aware models consistently outperform the baseline HAR model.

---

## Significance

Our empirical results demonstrate that regime-aware HAR extensions consistently yield lower forecasting errors than the standard HAR model, confirming the value of incorporating time-varying dynamics. Specifically, coefficient-based soft clustering effectively captures gradual shifts and identifies structural breaks in volatility distributions, achieving superior performance before and after the COVID time period. Meanwhile, distributional clustering better captures volatility behavior during the highly volatile COVID-19 time period. Inclusion of the VIX as a forward-looking feature enhances model responsiveness to shifts in market sentiment, thereby refining predictive accuracy.

These outcomes illustrate that volatility dynamics exhibit regime-dependent statistical properties, and that flexible models adapting regime-specific parameters provide a meaningful advantage. Future research could explore hybrid frameworks that integrate clustering and Markov switching, as well as advanced sequential models like LSTMs to capture more complex temporal dependencies.


---

## References

```
@article{zhang_Chinese_International_regime_switching,
    author = "Y. Zhang and L. Lei and Y. Wei",
    title = "Forecasting the Chinese stock market volatility with international market volatilities: The role of regime switching",
    journal = "The North American Journal of Economics and Finance",
    year = 2020,
    volume = 52,
    number = 101145
}

@article{luo_infinite_HAR_HMM,
    author = "J. Luo and T. Klein and Q. Ji and C. Hou",
    title = "Forecasting realized volatility of agricultural commodity futures with infinite Hidden Markov HAR models",
    journal = "International Journal of Forecasting",
    year = 2022,
    volume = 38,
    number = 1,
    pages = "51-73"
}

@article{gallo_Changing_average,
    author = "G. M. Gallo and E. Otranto",
    title = "Forecasting realized volatility with changing average levels",
    journal = "International Journal of Forecasting",
    year = 2020,
    volume = 52,
    number = 3,
    pages = "620-634"
}

@article{ding_regine_switching,
    author = "Y. Ding and D. Kambouroudis and D. G. McMillan",
    title = "Forecasting realised volatility using regime-switching models",
    journal = "International Review of Economics \& Finance",
    year = 2025,
    volume = 101,
    number = 104171,
}

@article{sullivan_vol_lstm,
    author = "J. C. Sullivan",
    title = "Stock Price Volatility Prediction with Long Short-Term Memory Neural Networks",
    journal = "Stanford University",
    year = 2018,
}

@article{bucci_vol_NN,
    author = "A. Bucci",
    title = "Realized Volatility Forecasting with Neural Networks",
    journal = "Munich Personal RePEc Archive",
    year = 2019,
    number = 95443,
}

@article{li_HAR_LSTM,
    author = "X. Li and D. Li and Y. Cheng and W. Li",
    title = "Forecasting the volatility of educational firms based on HAR model and LSTM models considering sentiment and educational policy",
    journal = "Heliyon",
    year = 2024,
    volume = 10,
    number = 19,
}

@online{stavrianos_HARRV,
    author = "S. Stavrianos",
    title = "Forecasting S&P 500 Volatility with the HAR-RV Model",
    year = 2024,
    url = "https://www.stavrianoseconblog.eu/2024/09/forecasting-volatility-with-har-rv.html",
}

@article{clements_harnessing_HAR_vol,
    author = "A. Clements and D. P. A. Preve",
    title = "A Practical Guide to harnessing the HAR volatility model",
    journal = "Journal of Banking & Finance",
    year = 2021,
    volume = 133,
    number = 106285,
}

@article{hu_HAR_graphNN,
    author = "N. Hu and X. Yin and Y. Yao",
    title = "A novel HAR-type realized volatility forecasting model using graph neural network",
    journal = "International Review of Financial Analysis",
    year = 2025,
    volume = 98,
    number = 103881,
}

@article{corsi,
    author = "F. Corsi",
    title = "A Simple Approximate Long-Memory Model of Realized Volatility",
    journal = "Journal of Financial Econometrics",
    year = 2009,
    volume = 7,
    number = 2,
    pages = "174-196"
}

@online{WagnerInvestopedia,
    author = "H. Wagner",
    title = "Why Volatility Is Important for Investors",
    year = 2022,
    url = "https://www.investopedia.com/articles/financial-theory/08/volatility.asp"
}

@article{gunnarsson_implications,
    author = "E. S. Gunnarsson and H. R. Isern and A Kaloudis and M. Risstad and B. Vigdel and S. Westgaard",
    title = "Prediction of realized volatility and implied volatility indices using AI and machine learning: A review",
    journal = "International Review of Financial Analysis",
    year = 2024,
    volume = 93,
    number = 103221,
}

@article{hamilton_regime_switching,
    author = "J. D. Hamilton",
    title = "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle",
    journal = "Econometrica",
    year = 1989,
    volume = 52,
    number = 2,
    pages = "357-384"
}

@article{Prakash_2021,
   title={Structural Clustering of Volatility Regimes for Dynamic Trading Strategies},
   volume={28},
   ISSN={1466-4313},
   url={http://dx.doi.org/10.1080/1350486X.2021.2007146},
   DOI={10.1080/1350486x.2021.2007146},
   number={3},
   journal={Applied Mathematical Finance},
   publisher={Informa UK Limited},
   author={A. Prakash and N. James and M. Menzies and G. Francis},
   year={2021},
   month=may, pages={236–274} 
}

@article{Zheng_ClusteringHiddenMarkov,
    author = "K. Zheng and Y. Li and W. Xu",
    title = "Regime switching model estimation: spectral clustering hidden Markov model",
    journal = "Annals of Operations Research",
    year = 2019
}

@phdthesis{Wang_hybridmodel,
    author = {X. Wang},
    title = {Hybrid Volatility Forecasting Models Based on Machine Learning of High-Frequency Data},
    school = {Florida State University College of Arts and Sciences},
    year = 2022,
}

@article{Hu_hybridmodel,
    title = {Forecasting volatility of China’s crude oil futures based on hybrid ML-HAR-RV models},
    journal = {The North American Journal of Economics and Finance},
    volume = {78},
    pages = {102428},
    year = {2025},
    issn = {1062-9408},
    doi = {https://doi.org/10.1016/j.najef.2025.102428},
    url = {https://www.sciencedirect.com/science/article/pii/S1062940825000683},
    author = {G. Hu and X. Ma and T. Zhu},
}

@article{Otranto_bayesian,
    author = {E. Otranto and G. M. Gallo},
    title = {A Nonparametric Bayesian Approach to Detect the Number of Regimes in Markov Switching Models},
    journal = {Econometric Reviews},
    year = {2007}
}

@inproceedings{jebara_spectral,
  title={Spectral Clustering and Embedding with Hidden Markov Models},
  author={Jebara, Tony and Song, Yingbo and Thadani, Kapil},
  booktitle={Proceedings of the European Conference on Machine Learning (ECML)},
  pages={164--175},
  year={2007},
  publisher={Springer}
}
```
---
