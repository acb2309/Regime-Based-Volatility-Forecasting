# Volatility-Forecasting

Market volatility forecasts are essential for portfolio management, risk control, derivatives pricing, and regulatory decisions. Traditional models like the Heterogeneous Autoregressive (HAR) model use past volatility over different time horizons (daily, weekly, monthly) to predict future volatility, assuming stable market dynamics with fixed parameters.

However, financial markets often move through distinct regimes, such as calm, crisis, and recovery phases, that cause volatility behavior to change over time. In these settings, the statistical relationship between past and future volatility is not constant, and fixed-parameter models may fail to capture such evolving patterns, reducing forecast accuracy.

Our research addresses these challenges by developing regime-aware forecasting frameworks that detect structural breaks and adapt to changing market conditions. By combining historical realized volatility with forward-looking indicators like the Implied Volatility Index (VIX), our models adjust dynamically to different volatility regimes, improving predictive performance.

Ultimately, this work enhances the robustness and reliability of volatility forecasts, supporting better-informed real-time decisions in risk management and hedging.

# Data Collection 

Our dataset consists of high-frequency intraday price data for the S\&P 500 index (SPX), spanning eleven years from June 2, 2014 to April 29, 2025, sourced via Bloomberg Terminal. Using \hbox{5-minute} closing prices, we calculate intraday log-returns as:

{\setlength{\abovedisplayskip}{8pt}
 \setlength{\belowdisplayskip}{8pt}
\begin{equation}
    r_{t,i} = \ln\left(\frac{P_{t,i}}{P_{t,i-1}}\right)
    \label{eq:returns}
\end{equation}
}

where $P_{t,i}$ is the price at the $i^\text{th}$ 5-minute interval on day $t$. These returns are then aggregated to compute daily realized volatility (RV), adjusted to account for shorter trading days:

{\setlength{\abovedisplayskip}{8pt}
 \setlength{\belowdisplayskip}{8pt}
\begin{equation}
    RV_t = \sqrt{\frac{N}{n} \sum_{i=1}^{n} r_{t,i}^2}
    \label{eq:RVt}
\end{equation}
}

Here, $n$ is the number of intraday returns observed on day $t$ (which may vary due to holidays), and $N$ is the standard number of returns in a full trading day---78 for intraday data only, or 79 including overnight returns. This scaling ensures comparability of RV across days with varying lengths.

# Methodology 

The original HAR model, introduced by Corsi \cite{corsi} in 2009, was designed to capture realized volatility (RV) behavior across multiple time scales (daily, weekly, and monthly) reflecting the activity of different types of investors:

{\setlength{\abovedisplayskip}{8pt}
 \setlength{\belowdisplayskip}{8pt}
\begin{equation}
    RV_t = \beta_0 
    + \beta_d \cdot RV_{t-1} 
    + \beta_w \cdot \overline{RV}_{t-1}^{(w)} 
    + \beta_m \cdot \overline{RV}_{t-1}^{(m)} 
    + \varepsilon_t
\end{equation}
}

where $\beta_0, \beta_d, \beta_w, \beta_m$ are the HAR parameters to be estimated, and $\varepsilon_t$ is the error term. Subsequent research has shown that while the HAR model captures important long-memory features, volatility dynamics often shift across distinct market regimes, exhibiting structural breaks and regime-dependent behavior. To better capture these time-varying patterns, regime-switching extensions of HAR have been proposed. For instance, Zhang \cite{zhang_Chinese_International_regime_switching} demonstrates that incorporating regime switching enhances forecasts of Chinese stock market volatility by accounting for structural shifts driven by international market conditions.


# Results

# Discussion 

# Conclusion
