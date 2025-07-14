# Volatility-Forecasting

Market volatility forecasts are essential for portfolio management, risk control, derivatives pricing, and regulatory decisions. Traditional models like the Heterogeneous Autoregressive (HAR) model use past volatility over different time horizons (daily, weekly, monthly) to predict future volatility, assuming stable market dynamics with fixed parameters.

However, financial markets often move through distinct regimes, such as calm, crisis, and recovery phases, that cause volatility behavior to change over time. In these settings, the statistical relationship between past and future volatility is not constant, and fixed-parameter models may fail to capture such evolving patterns, reducing forecast accuracy.

Our research addresses these challenges by developing regime-aware forecasting frameworks that detect structural breaks and adapt to changing market conditions. By combining historical realized volatility with forward-looking indicators like the Implied Volatility Index (VIX), our models adjust dynamically to different volatility regimes, improving predictive performance.

Ultimately, this work enhances the robustness and reliability of volatility forecasts, supporting better-informed real-time decisions in risk management and hedging.

# Data Collection 

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
