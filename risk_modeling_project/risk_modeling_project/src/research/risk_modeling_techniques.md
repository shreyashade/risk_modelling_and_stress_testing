# Research on Risk Modeling Techniques

## Monte Carlo Simulation

Monte Carlo simulation is a computational technique that uses random sampling to obtain numerical results. In risk modeling, it's used to simulate the behavior of financial assets and portfolios under various market conditions.

### Key Components:

1. **Random Number Generation**: The foundation of Monte Carlo methods, generating sequences of random numbers following specific probability distributions.

2. **Stochastic Process Modeling**: Simulating the evolution of asset prices over time using models like:
   - Geometric Brownian Motion (GBM)
   - Jump-Diffusion processes
   - Stochastic volatility models (e.g., Heston model)
   - GARCH processes

3. **Correlation Modeling**: Capturing dependencies between assets using:
   - Constant correlation matrices
   - Dynamic conditional correlation (DCC)
   - Copula functions (Gaussian, t-copula, Archimedean)

4. **Risk Factor Decomposition**: Breaking down complex instruments into underlying risk factors.

5. **Scenario Generation**: Creating thousands or millions of potential future states of the portfolio.

### Advantages:

- Can model complex, non-linear relationships and instruments
- Handles path-dependent securities effectively
- Provides full distribution of outcomes, not just summary statistics
- Flexible for incorporating various market assumptions

### Limitations:

- Computationally intensive
- Sensitive to model assumptions and parameters
- May require large number of simulations for convergence
- Model risk can be significant

## Historical Simulation

Historical simulation is a non-parametric approach that uses actual historical returns to estimate the distribution of future returns.

### Key Components:

1. **Historical Data Collection**: Gathering relevant historical price data for all assets in the portfolio.

2. **Return Calculation**: Computing historical returns over the relevant time horizon.

3. **Portfolio Revaluation**: Applying historical returns to current portfolio positions.

4. **Bootstrapping**: Resampling historical returns to generate more scenarios.

5. **Filtered Historical Simulation**: Combining historical returns with volatility models to account for changing market conditions.

### Advantages:

- No distributional assumptions required
- Preserves empirical correlations and dependencies
- Captures fat tails and other non-normal features
- Intuitive and easy to explain

### Limitations:

- Limited by available historical data
- May not capture extreme events if they're not in the historical sample
- Assumes the future will resemble the past
- Equal weighting of all historical periods may be unrealistic

## Hybrid Approaches

Many advanced risk modeling frameworks combine elements of both Monte Carlo and historical simulation:

1. **Filtered Historical Simulation**: Using GARCH models to scale historical returns based on current volatility.

2. **Historical Simulation with Monte Carlo Bootstrapping**: Resampling historical returns with replacement and applying Monte Carlo techniques.

3. **Scenario-Enhanced Historical Simulation**: Augmenting historical data with hypothetical stress scenarios.

## Advanced Risk Metrics

### Value at Risk (VaR)

- Measures the potential loss in value of a portfolio over a defined period for a given confidence interval.
- Parametric VaR, Historical VaR, and Monte Carlo VaR approaches.
- Regulatory standard but has limitations in capturing tail risk.

### Expected Shortfall (ES) / Conditional VaR (CVaR)

- Measures the expected loss given that the loss exceeds the VaR threshold.
- More coherent risk measure than VaR (satisfies subadditivity).
- Better captures tail risk and is less prone to manipulation.

### Drawdown Measures

- Maximum Drawdown: The largest peak-to-trough decline in portfolio value.
- Average Drawdown: The average of all drawdowns over a period.
- Conditional Expected Drawdown: Expected value of drawdowns exceeding a threshold.

### Entropy-Based Risk Measures

- Measures uncertainty in the return distribution.
- Shannon Entropy, Rényi Entropy, and Tsallis Entropy.
- Useful for capturing non-normal return distributions.

### Spectral Risk Measures

- Weighted averages of quantiles of the loss distribution.
- Weights reflect risk aversion of the decision-maker.
- Generalizes VaR and ES.

### Distortion Risk Measures

- Transform the cumulative distribution function of losses.
- Examples include Wang Transform and proportional hazard transform.
- Provides flexibility in emphasizing different parts of the loss distribution.

## Regulatory Frameworks

### Basel III

- International regulatory framework for banks.
- Capital requirements based on risk-weighted assets.
- Introduced Liquidity Coverage Ratio (LCR) and Net Stable Funding Ratio (NSFR).
- Fundamental Review of the Trading Book (FRTB) revised market risk framework.

### Solvency II

- Regulatory framework for insurance companies in the EU.
- Risk-based capital requirements.
- Three pillar approach: quantitative requirements, governance and supervision, disclosure and transparency.

### Dodd-Frank Act

- U.S. legislation in response to the 2008 financial crisis.
- Stress testing requirements for large financial institutions.
- Comprehensive Capital Analysis and Review (CCAR) and Dodd-Frank Act Stress Testing (DFAST).

### IFRS 9 / CECL

- Accounting standards for financial instruments.
- Expected credit loss models for impairment.
- Forward-looking approach to credit risk.
