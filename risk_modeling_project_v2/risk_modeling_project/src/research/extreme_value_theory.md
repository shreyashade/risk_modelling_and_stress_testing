# Research on Extreme Value Theory (EVT)

## Introduction to Extreme Value Theory

Extreme Value Theory (EVT) is a branch of statistics that deals with the extreme deviations from the median of probability distributions. In the context of financial risk management, EVT provides a framework for modeling and estimating the probability of rare but severe events, which are typically found in the tails of return distributions.

Traditional risk models often assume normal distributions, which significantly underestimate the probability of extreme events in financial markets. EVT addresses this limitation by focusing specifically on modeling the tail behavior of distributions.

## Theoretical Foundation

### Fisher-Tippett-Gnedenko Theorem

The central theorem in EVT states that properly normalized maxima of a sequence of independent and identically distributed random variables converge in distribution to one of three possible distributions:

1. **Gumbel distribution** (Type I): For distributions with exponentially decaying tails (e.g., normal, exponential)
2. **Fréchet distribution** (Type II): For distributions with heavy tails (e.g., Pareto, Cauchy)
3. **Weibull distribution** (Type III): For distributions with bounded tails

These three distributions can be unified into a single parametric family known as the Generalized Extreme Value (GEV) distribution.

### Generalized Extreme Value (GEV) Distribution

The cumulative distribution function of the GEV is given by:

F(x; μ, σ, ξ) = exp(-(1 + ξ(x-μ)/σ)^(-1/ξ))

Where:
- μ is the location parameter
- σ is the scale parameter
- ξ is the shape parameter (determines the tail behavior)

The shape parameter ξ determines which of the three extreme value distributions is represented:
- ξ > 0: Fréchet distribution (heavy-tailed)
- ξ = 0: Gumbel distribution (light-tailed)
- ξ < 0: Weibull distribution (bounded tail)

### Pickands-Balkema-de Haan Theorem

This theorem provides the theoretical basis for the Peaks-Over-Threshold (POT) approach. It states that for a large class of distributions, the distribution of excesses over a high threshold converges to a Generalized Pareto Distribution (GPD) as the threshold increases.

## Main Approaches in EVT

### Block Maxima Method

1. **Methodology**:
   - Divide the data into non-overlapping blocks of equal size
   - Extract the maximum (or minimum) value from each block
   - Fit a GEV distribution to these block maxima

2. **Advantages**:
   - Conceptually simple
   - Well-established statistical properties
   - Directly models extreme events

3. **Limitations**:
   - Inefficient use of data (only one extreme value per block)
   - Sensitive to block size selection
   - May not capture all extreme events

### Peaks-Over-Threshold (POT) Method

1. **Methodology**:
   - Select a high threshold
   - Consider all observations exceeding this threshold
   - Fit a Generalized Pareto Distribution (GPD) to the exceedances

2. **Advantages**:
   - More efficient use of data
   - Focuses directly on tail behavior
   - Often provides better estimates with limited data

3. **Limitations**:
   - Threshold selection is critical and can be subjective
   - Assumes exceedances are independent
   - Sensitive to temporal clustering of extreme events

### Generalized Pareto Distribution (GPD)

The GPD is defined by its cumulative distribution function:

F(x; σ, ξ) = 1 - (1 + ξx/σ)^(-1/ξ)

Where:
- σ is the scale parameter
- ξ is the shape parameter

Similar to the GEV distribution, the shape parameter ξ determines the tail behavior:
- ξ > 0: Heavy-tailed distribution
- ξ = 0: Exponential distribution
- ξ < 0: Distribution with a finite upper bound

## Application in Financial Risk Management

### Tail Risk Estimation

1. **Value at Risk (VaR) Estimation**:
   Using EVT to estimate extreme quantiles of the loss distribution:
   
   VaR_α = u + (σ/ξ) * ((n/N_u * (1-α))^(-ξ) - 1)
   
   Where:
   - u is the threshold
   - n is the total number of observations
   - N_u is the number of exceedances
   - α is the confidence level

2. **Expected Shortfall (ES) Estimation**:
   
   ES_α = VaR_α + (σ + ξ(VaR_α - u))/(1-ξ)
   
   This provides a more coherent risk measure than VaR by considering the average loss beyond VaR.

### Return Level Estimation

EVT can be used to estimate the return level x_p, which is the level expected to be exceeded on average once every 1/p periods:

x_p = u + (σ/ξ) * ((n*p/N_u)^ξ - 1)

This is particularly useful for stress testing and long-term risk assessment.

## Advanced EVT Techniques

### Multivariate Extreme Value Theory

Extends EVT to model the joint distribution of extreme events across multiple assets or risk factors.

1. **Dependence Measures**:
   - Extremal dependence coefficient
   - Tail dependence coefficient
   - Extremogram

2. **Multivariate Models**:
   - Logistic model
   - Asymmetric logistic model
   - Negative logistic model

### Conditional Extreme Value Theory

Incorporates time-varying parameters to account for changing market conditions:

1. **GARCH-EVT Models**:
   - Use GARCH models to capture volatility clustering
   - Apply EVT to standardized residuals
   - Combine to obtain conditional risk measures

2. **Markov-Switching EVT Models**:
   - Allow parameters to change according to market regimes
   - Capture structural breaks in tail behavior

### Bayesian EVT

Incorporates prior information and provides full posterior distributions for parameters:

1. **Advantages**:
   - Uncertainty quantification for parameter estimates
   - Incorporation of expert knowledge
   - Better performance with small samples

2. **Implementation**:
   - Markov Chain Monte Carlo (MCMC) methods
   - Hamiltonian Monte Carlo
   - Approximate Bayesian Computation (ABC)

## Challenges and Considerations

1. **Data Requirements**:
   - EVT is data-intensive, especially for reliable estimation of tail parameters
   - Limited historical data for truly extreme events

2. **Model Risk**:
   - Sensitivity to threshold/block size selection
   - Parameter estimation uncertainty
   - Extrapolation beyond observed data range

3. **Time-Varying Nature of Extremes**:
   - Financial markets exhibit changing tail behavior over time
   - Need for adaptive approaches

4. **Regulatory Considerations**:
   - Basel III/IV requirements for expected shortfall
   - Stress testing frameworks
   - Model validation requirements

## Implementation Considerations for Risk Modeling Project

1. **Data Selection**:
   - Use high-frequency data when possible
   - Consider different time horizons (daily, weekly, monthly)
   - Include periods of market stress

2. **Threshold Selection Methods**:
   - Mean excess function plots
   - Hill plots
   - Threshold stability plots
   - Goodness-of-fit tests

3. **Parameter Estimation**:
   - Maximum likelihood estimation
   - Method of moments
   - Probability-weighted moments
   - L-moments

4. **Validation Techniques**:
   - Backtesting procedures
   - Out-of-sample testing
   - Sensitivity analysis
   - Stress testing of the EVT model itself
