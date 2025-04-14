"""
This module provides a Jupyter notebook example demonstrating the usage of the
Risk Modeling and Stress Testing Framework.
"""

# %% [markdown]
# # Risk Modeling and Stress Testing Framework - Demo
# 
# This notebook demonstrates the capabilities of the advanced risk modeling framework, including:
# 
# - Portfolio construction and analysis
# - Monte Carlo simulation
# - Historical simulation
# - Extreme Value Theory (EVT) analysis
# - Stress testing
# - Regime analysis
# - Interactive visualizations

# %%
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Load and Prepare Data

# %%
# Import project modules
import sys
sys.path.append('..')  # Add parent directory to path

from src.models.portfolio import Portfolio
from src.models.data_loader import DataLoader
from src.models.monte_carlo import MonteCarloSimulation
from src.models.historical_simulation import HistoricalSimulation
from src.models.extreme_value_theory import EVTAnalysis
from src.models.stress_testing import StressTesting
from src.models.regime_analysis import RegimeAnalysis
from src.visualization.dashboard import (
    RiskDashboard, StressTestDashboard, RegimeAnalysisDashboard,
    EVTTailVisualization, ComprehensiveRiskDashboard
)

# %%
# Load market data
data_loader = DataLoader()

# Define portfolio assets
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ']
start_date = '2018-01-01'
end_date = '2023-12-31'

# Load returns data
try:
    returns = data_loader.load_returns(tickers, start_date, end_date)
    prices = data_loader.load_prices(tickers, start_date, end_date)
    print(f"Successfully loaded data for {len(tickers)} assets from {start_date} to {end_date}")
except Exception as e:
    print(f"Error loading data: {e}")
    # Use synthetic data for demonstration
    print("Generating synthetic data for demonstration")
    returns = data_loader.generate_synthetic_returns(
        n_assets=len(tickers),
        n_periods=1000,
        asset_names=tickers
    )
    prices = (1 + returns).cumprod()
    prices.index = pd.date_range(start=start_date, periods=len(returns), freq='B')

# %%
# Display the first few rows of returns data
returns.head()

# %%
# Plot asset prices
plt.figure(figsize=(12, 8))
for ticker in tickers:
    plt.plot(prices.index, prices[ticker] / prices[ticker].iloc[0], label=ticker)
plt.title('Normalized Asset Prices')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 2. Portfolio Construction

# %%
# Create portfolio with equal weights
weights = np.ones(len(tickers)) / len(tickers)
portfolio = Portfolio(returns, prices=prices)
portfolio.set_weights(weights)
portfolio.calculate_portfolio_returns()

# %%
# Display portfolio weights
plt.figure(figsize=(10, 6))
plt.bar(tickers, weights)
plt.title('Portfolio Weights')
plt.xlabel('Asset')
plt.ylabel('Weight')
plt.grid(True)
plt.show()

# %%
# Plot portfolio value over time
portfolio_value = portfolio.calculate_portfolio_value()
plt.figure(figsize=(12, 6))
plt.plot(portfolio_value.index, portfolio_value)
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# %%
# Calculate and display basic portfolio statistics
stats = portfolio.calculate_statistics()
pd.DataFrame(stats, index=['Value']).T

# %% [markdown]
# ## 3. Monte Carlo Simulation

# %%
# Run Monte Carlo simulation
mc_sim = MonteCarloSimulation(portfolio)
mc_results = mc_sim.run_simulation(
    n_simulations=1000,
    horizon=20,
    return_type='log',
    random_seed=42
)

# %%
# Plot Monte Carlo simulation paths
plt.figure(figsize=(12, 6))
for i in range(min(100, mc_results.shape[1])):  # Plot first 100 paths
    plt.plot(mc_results.index, mc_results.iloc[:, i], 'b-', alpha=0.1)
    
# Plot mean path
mean_path = mc_results.mean(axis=1)
plt.plot(mc_results.index, mean_path, 'r-', linewidth=2, label='Mean Path')

# Plot 5th and 95th percentiles
percentile_5 = mc_results.quantile(0.05, axis=1)
percentile_95 = mc_results.quantile(0.95, axis=1)
plt.plot(mc_results.index, percentile_5, 'g--', linewidth=2, label='5th Percentile')
plt.plot(mc_results.index, percentile_95, 'g--', linewidth=2, label='95th Percentile')

plt.title('Monte Carlo Simulation Paths')
plt.xlabel('Time Horizon')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Calculate VaR and ES from Monte Carlo results
var_95 = mc_sim.calculate_var(confidence_level=0.95)
es_95 = mc_sim.calculate_expected_shortfall(confidence_level=0.95)
print(f"Monte Carlo VaR (95%): {var_95:.2%}")
print(f"Monte Carlo ES (95%): {es_95:.2%}")

# %%
# Plot terminal distribution
plt.figure(figsize=(12, 6))
terminal_values = mc_results.iloc[-1]
sns.histplot(terminal_values, kde=True, bins=50)
plt.axvline(terminal_values.quantile(0.05), color='r', linestyle='--', 
           label=f'VaR (95%): {var_95:.2%}')
plt.axvline(terminal_values[terminal_values <= terminal_values.quantile(0.05)].mean(), 
           color='darkred', linestyle='--', 
           label=f'ES (95%): {es_95:.2%}')
plt.title('Terminal Value Distribution')
plt.xlabel('Portfolio Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 4. Historical Simulation

# %%
# Run Historical simulation
hist_sim = HistoricalSimulation(portfolio)
hist_results = hist_sim.run_simulation(
    n_samples=1000,
    horizon=20,
    method='bootstrap'
)

# %%
# Plot Historical simulation paths
plt.figure(figsize=(12, 6))
for i in range(min(100, hist_results.shape[1])):  # Plot first 100 paths
    plt.plot(hist_results.index, hist_results.iloc[:, i], 'b-', alpha=0.1)
    
# Plot mean path
mean_path = hist_results.mean(axis=1)
plt.plot(hist_results.index, mean_path, 'r-', linewidth=2, label='Mean Path')

# Plot 5th and 95th percentiles
percentile_5 = hist_results.quantile(0.05, axis=1)
percentile_95 = hist_results.quantile(0.95, axis=1)
plt.plot(hist_results.index, percentile_5, 'g--', linewidth=2, label='5th Percentile')
plt.plot(hist_results.index, percentile_95, 'g--', linewidth=2, label='95th Percentile')

plt.title('Historical Simulation Paths')
plt.xlabel('Time Horizon')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Calculate VaR and ES from Historical simulation results
hist_var_95 = hist_sim.calculate_var(confidence_level=0.95)
hist_es_95 = hist_sim.calculate_expected_shortfall(confidence_level=0.95)
print(f"Historical VaR (95%): {hist_var_95:.2%}")
print(f"Historical ES (95%): {hist_es_95:.2%}")

# %%
# Compare Monte Carlo and Historical simulation results
comparison_data = {
    'Method': ['Monte Carlo', 'Historical'],
    'VaR (95%)': [var_95, hist_var_95],
    'ES (95%)': [es_95, hist_es_95]
}
comparison_df = pd.DataFrame(comparison_data)
comparison_df

# %%
# Plot comparison
plt.figure(figsize=(10, 6))
x = np.arange(len(comparison_df['Method']))
width = 0.35

plt.bar(x - width/2, comparison_df['VaR (95%)'], width, label='VaR (95%)')
plt.bar(x + width/2, comparison_df['ES (95%)'], width, label='ES (95%)')

plt.xlabel('Method')
plt.ylabel('Value')
plt.title('Risk Metrics Comparison')
plt.xticks(x, comparison_df['Method'])
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 5. Extreme Value Theory (EVT) Analysis

# %%
# Perform EVT analysis
evt_analyzer = EVTAnalysis(portfolio.returns)
evt_results = evt_analyzer.fit_gpd(threshold_method='quantile', threshold_value=0.05)

# %%
# Print EVT parameters
print(f"EVT Parameters:")
print(f"Threshold: {evt_results['threshold']:.4f}")
print(f"Shape parameter (ξ): {evt_results['shape']:.4f}")
print(f"Scale parameter (β): {evt_results['scale']:.4f}")

# %%
# Calculate EVT-based VaR and ES
evt_var_95 = evt_analyzer.calculate_var(confidence_level=0.95)
evt_es_95 = evt_analyzer.calculate_expected_shortfall(confidence_level=0.95)
print(f"EVT VaR (95%): {evt_var_95:.2%}")
print(f"EVT ES (95%): {evt_es_95:.2%}")

# %%
# Plot EVT tail distribution
plt.figure(figsize=(12, 8))

# Create a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Full return distribution
sns.histplot(portfolio.returns, bins=50, kde=True, ax=axes[0, 0])
axes[0, 0].axvline(evt_results['threshold'], color='red', linestyle='--', 
                 label=f'Threshold: {evt_results["threshold"]:.2%}')
axes[0, 0].set_title('Full Return Distribution')
axes[0, 0].set_xlabel('Return')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# Plot 2: Tail distribution
tail_returns = portfolio.returns[portfolio.returns <= evt_results['threshold']]
sns.histplot(tail_returns, bins=30, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Tail Distribution')
axes[0, 1].set_xlabel('Return')
axes[0, 1].set_ylabel('Frequency')

# Plot 3: QQ plot
from scipy import stats
stats.probplot(tail_returns, dist='expon', plot=axes[1, 0])
axes[1, 0].set_title('Exponential QQ Plot of Tail')

# Plot 4: Mean excess plot
thresholds = np.linspace(portfolio.returns.min(), portfolio.returns.quantile(0.2), 50)
mean_excess = []

for u in thresholds:
    exceedances = portfolio.returns[portfolio.returns <= u] - u
    if len(exceedances) > 0:
        mean_excess.append(np.abs(exceedances.mean()))
    else:
        mean_excess.append(np.nan)

axes[1, 1].plot(thresholds, mean_excess, 'o-')
axes[1, 1].set_title('Mean Excess Plot')
axes[1, 1].set_xlabel('Threshold')
axes[1, 1].set_ylabel('Mean Excess')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# %%
# Compare all risk metrics
comparison_data = {
    'Method': ['Monte Carlo', 'Historical', 'EVT'],
    'VaR (95%)': [var_95, hist_var_95, evt_var_95],
    'ES (95%)': [es_95, hist_es_95, evt_es_95]
}
comparison_df = pd.DataFrame(comparison_data)
comparison_df

# %%
# Plot comparison of all methods
plt.figure(figsize=(10, 6))
x = np.arange(len(comparison_df['Method']))
width = 0.35

plt.bar(x - width/2, comparison_df['VaR (95%)'], width, label='VaR (95%)')
plt.bar(x + width/2, comparison_df['ES (95%)'], width, label='ES (95%)')

plt.xlabel('Method')
plt.ylabel('Value')
plt.title('Risk Metrics Comparison')
plt.xticks(x, comparison_df['Method'])
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 6. Stress Testing

# %%
# Perform stress testing
stress_test = StressTesting(portfolio)

# Define stress scenarios
scenarios = [
    'market_crash',
    'interest_rate_shock',
    'volatility_spike',
    'liquidity_crisis',
    'correlation_breakdown'
]

# Add custom tariff scenario
custom_tariff_scenario = {
    'name': 'tariff_war',
    'shocks': {ticker: np.random.uniform(-0.25, -0.05) for ticker in tickers},
    'volatility_multiplier': 1.5,
    'correlation_adjustment': 0.2  # Increase correlations
}

# Run stress tests
stressed_data = stress_test.run_scenarios(scenarios)
tariff_results = stress_test.apply_custom_scenario(custom_tariff_scenario)
stressed_data['tariff_war'] = tariff_results

# %%
# Calculate impact of stress scenarios
impact_data = {}
for scenario in stressed_data:
    impact = stress_test.calculate_scenario_impact(stressed_data[scenario])
    impact_data[scenario] = impact
    print(f"Impact of {scenario}: {impact:.2%}")

# %%
# Plot stress test impacts
plt.figure(figsize=(12, 6))
plt.bar(impact_data.keys(), impact_data.values())
plt.title('Stress Test Scenario Impacts')
plt.xlabel('Scenario')
plt.ylabel('Impact (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Plot portfolio values under different stress scenarios
plt.figure(figsize=(12, 6))

# Calculate portfolio values
original_portfolio = portfolio.calculate_portfolio_value()
original_portfolio = original_portfolio / original_portfolio.iloc[0]  # Normalize

# Plot original portfolio value
plt.plot(original_portfolio.index, original_portfolio, 'b-', label='Original')

# Plot stressed portfolio values
for scenario, data in stressed_data.items():
    # Calculate portfolio returns
    stressed_returns = data @ weights
    
    # Calculate portfolio value
    stressed_value = (1 + stressed_returns).cumprod()
    stressed_value = stressed_value / stressed_value.iloc[0]  # Normalize
    
    plt.plot(stressed_value.index, stressed_value, '--', label=scenario)

plt.title('Portfolio Value Under Stress Scenarios')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 7. Regime Analysis

# %%
# Perform regime analysis
regime_analyzer = RegimeAnalysis(returns)

# Detect volatility regimes
vol_regimes = regime_analyzer.detect_volatility_regimes(
    n_regimes=3,
    window=60
)

# Detect correlation regimes
corr_regimes = regime_analyzer.detect_correlation_regimes(
    n_regimes=2,
    window=60
)

# Use Markov switching model for combined regime detection
ms_regimes = regime_analyzer.detect_regimes(
    method='markov_switching',
    n_regimes=3
)

# %%
# Plot regimes
plt.figure(figsize=(12, 8))

# Create a 3x1 grid
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Plot 1: Portfolio returns with volatility regimes
portfolio_returns = portfolio.returns
axes[0].plot(portfolio_returns.index, portfolio_returns, 'b-', alpha=0.5)
for regime in range(vol_regimes.max() + 1):
    regime_data = portfolio_returns[vol_regimes == regime]
    axes[0].scatter(regime_data.index, regime_data, 
                  label=f'Regime {regime}', alpha=0.7)
axes[0].set_title('Portfolio Returns with Volatility Regimes')
axes[0].set_ylabel('Return')
axes[0].legend()
axes[0].grid(True)

# Plot 2: Portfolio returns with correlation regimes
axes[1].plot(portfolio_returns.index, portfolio_returns, 'b-', alpha=0.5)
for regime in range(corr_regimes.max() + 1):
    regime_data = portfolio_returns[corr_regimes == regime]
    axes[1].scatter(regime_data.index, regime_data, 
                  label=f'Regime {regime}', alpha=0.7)
axes[1].set_title('Portfolio Returns with Correlation Regimes')
axes[1].set_ylabel('Return')
axes[1].legend()
axes[1].grid(True)

# Plot 3: Portfolio returns with Markov switching regimes
axes[2].plot(portfolio_returns.index, portfolio_returns, 'b-', alpha=0.5)
for regime in range(ms_regimes.max() + 1):
    regime_data = portfolio_returns[ms_regimes == regime]
    axes[2].scatter(regime_data.index, regime_data, 
                  label=f'Regime {regime}', alpha=0.7)
axes[2].set_title('Portfolio Returns with Markov Switching Regimes')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Return')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()

# %%
# Calculate risk metrics by regime
regime_metrics = regime_analyzer.calculate_regime_risk_metrics(ms_regimes)
regime_metrics_df = pd.DataFrame({
    regime: {
        'Mean': metrics['mean'],
        'Volatility': metrics['volatility'],
        'VaR (95%)': metrics['var'],
        'ES (95%)': metrics['es']
    } for regime, metrics in regime_metrics.items()
}).T

# Format as percentages
regime_metrics_df = regime_metrics_df.applymap(lambda x: f"{x:.2%}")
regime_metrics_df

# %%
# Plot risk metrics by regime
metrics_to_plot = ['Mean', 'Volatility', 'VaR (95%)', 'ES (95%)']
regime_metrics_plot = pd.DataFrame({
    regime: {
        'Mean': metrics['mean'],
        'Volatility': metrics['volatility'],
        'VaR (95%)': metrics['var'],
        'ES (95%)': metrics['es']
    } for regime, metrics in regime_metrics.items()
}).T

plt.figure(figsize=(12, 8))
regime_metrics_plot.plot(kind='bar', figsize=(12, 8))
plt.title('Risk Metrics by Regime')
plt.xlabel('Regime')
plt.ylabel('Value')
plt.grid(True)
plt.legend(title='Metric')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Interactive Dashboards

# %%
# Create basic risk dashboard
risk_dashboard = RiskDashboard("Portfolio Risk Analysis")
risk_dashboard.create_dashboard(returns, weights)

# %%
# Create stress test dashboard
stress_dashboard = StressTestDashboard("Stress Test Analysis")
stress_dashboard.create_dashboard(returns, stressed_data, weights)

# %%
# Create regime analysis dashboard
regime_dashboard = RegimeAnalysisDashboard("Regime Analysis")
regime_dashboard.create_dashboard(returns, ms_regimes, weights)

# %%
# Create comprehensive dashboard
comprehensive_dashboard = ComprehensiveRiskDashboard("Comprehensive Risk Analysis")
comprehensive_dashboard.create_dashboard(
    returns=returns,
    stressed_data=stressed_data,
    regimes=ms_regimes,
    evt_results=evt_results,
    weights=weights
)

# %% [markdown]
# ## 9. Conclusion
# 
# This notebook has demonstrated the capabilities of our advanced risk modeling framework, including:
# 
# - Portfolio construction and analysis
# - Monte Carlo and historical simulation methods
# - Extreme Value Theory for tail risk modeling
# - Stress testing with various scenarios including a custom tariff war scenario
# - Regime analysis to understand performance under different market conditions
# - Interactive visualizations and dashboards
# 
# The framework provides a comprehensive toolkit for risk management and stress testing, allowing for sophisticated analysis of portfolio risk under various scenarios and market conditions.
