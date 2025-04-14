"""
Main application for the Risk Modeling and Stress Testing Framework.

This script demonstrates the capabilities of the risk modeling framework
by running a complete analysis on a sample portfolio.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
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

def main():
    """
    Main function to run the risk modeling framework demonstration.
    """
    logger.info("Starting Risk Modeling and Stress Testing Framework demonstration")
    
    # Create output directories
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/figures', exist_ok=True)
    os.makedirs('output/reports', exist_ok=True)
    
    # Step 1: Load data
    logger.info("Loading market data")
    data_loader = DataLoader()
    
    # Load data for a portfolio of stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ']
    start_date = '2018-01-01'
    end_date = '2023-12-31'
    
    try:
        returns = data_loader.load_returns(tickers, start_date, end_date)
        prices = data_loader.load_prices(tickers, start_date, end_date)
        logger.info(f"Successfully loaded data for {len(tickers)} assets from {start_date} to {end_date}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Use synthetic data for demonstration if real data loading fails
        logger.info("Generating synthetic data for demonstration")
        returns = data_loader.generate_synthetic_returns(
            n_assets=len(tickers),
            n_periods=1000,
            asset_names=tickers
        )
        prices = (1 + returns).cumprod()
        prices.index = pd.date_range(start=start_date, periods=len(returns), freq='B')
    
    # Step 2: Create portfolio
    logger.info("Creating portfolio")
    # Use equal weights for demonstration
    weights = np.ones(len(tickers)) / len(tickers)
    portfolio = Portfolio(returns, prices=prices)
    portfolio.set_weights(weights)
    portfolio.calculate_portfolio_returns()
    
    # Step 3: Run Monte Carlo simulation
    logger.info("Running Monte Carlo simulation")
    mc_sim = MonteCarloSimulation(portfolio)
    mc_results = mc_sim.run_simulation(
        n_simulations=10000,
        horizon=20,
        return_type='log',
        random_seed=42
    )
    
    # Calculate VaR and ES from Monte Carlo results
    var_95 = mc_sim.calculate_var(confidence_level=0.95)
    es_95 = mc_sim.calculate_expected_shortfall(confidence_level=0.95)
    logger.info(f"Monte Carlo VaR (95%): {var_95:.2%}")
    logger.info(f"Monte Carlo ES (95%): {es_95:.2%}")
    
    # Step 4: Run Historical simulation
    logger.info("Running Historical simulation")
    hist_sim = HistoricalSimulation(portfolio)
    hist_results = hist_sim.run_simulation(
        n_samples=10000,
        horizon=20,
        method='bootstrap'
    )
    
    # Calculate VaR and ES from Historical simulation results
    hist_var_95 = hist_sim.calculate_var(confidence_level=0.95)
    hist_es_95 = hist_sim.calculate_expected_shortfall(confidence_level=0.95)
    logger.info(f"Historical VaR (95%): {hist_var_95:.2%}")
    logger.info(f"Historical ES (95%): {hist_es_95:.2%}")
    
    # Step 5: Perform EVT analysis
    logger.info("Performing Extreme Value Theory analysis")
    evt_analyzer = EVTAnalysis(portfolio.returns)
    evt_results = evt_analyzer.fit_gpd(threshold_method='quantile', threshold_value=0.05)
    
    # Calculate EVT-based VaR and ES
    evt_var_95 = evt_analyzer.calculate_var(confidence_level=0.95)
    evt_es_95 = evt_analyzer.calculate_expected_shortfall(confidence_level=0.95)
    logger.info(f"EVT VaR (95%): {evt_var_95:.2%}")
    logger.info(f"EVT ES (95%): {evt_es_95:.2%}")
    
    # Step 6: Perform stress testing
    logger.info("Performing stress testing")
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
    
    # Calculate impact of stress scenarios
    for scenario in stressed_data:
        impact = stress_test.calculate_scenario_impact(stressed_data[scenario])
        logger.info(f"Impact of {scenario}: {impact:.2%}")
    
    # Step 7: Perform regime analysis
    logger.info("Performing regime analysis")
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
    
    # Calculate risk metrics by regime
    regime_metrics = regime_analyzer.calculate_regime_risk_metrics(ms_regimes)
    for regime, metrics in regime_metrics.items():
        logger.info(f"Regime {regime} metrics: Mean={metrics['mean']:.2%}, Volatility={metrics['volatility']:.2%}, VaR={metrics['var']:.2%}")
    
    # Step 8: Create visualizations and dashboards
    logger.info("Creating visualizations and dashboards")
    
    # Create basic risk dashboard
    risk_dashboard = RiskDashboard("Portfolio Risk Analysis")
    risk_dashboard.create_dashboard(returns, weights)
    risk_dashboard.save_html('output/reports/risk_dashboard.html')
    
    # Create stress test dashboard
    stress_dashboard = StressTestDashboard("Stress Test Analysis")
    stress_dashboard.create_dashboard(returns, stressed_data, weights)
    stress_dashboard.save_html('output/reports/stress_test_dashboard.html')
    
    # Create regime analysis dashboard
    regime_dashboard = RegimeAnalysisDashboard("Regime Analysis")
    regime_dashboard.create_dashboard(returns, ms_regimes, weights)
    regime_dashboard.save_html('output/reports/regime_analysis_dashboard.html')
    
    # Create EVT visualization
    evt_viz = EVTTailVisualization("EVT Tail Analysis")
    evt_viz.plot(portfolio.returns, evt_results['threshold'])
    plt.savefig('output/figures/evt_analysis.png')
    
    # Create comprehensive dashboard
    comprehensive_dashboard = ComprehensiveRiskDashboard("Comprehensive Risk Analysis")
    comprehensive_dashboard.create_dashboard(
        returns=returns,
        stressed_data=stressed_data,
        regimes=ms_regimes,
        evt_results=evt_results,
        weights=weights
    )
    comprehensive_dashboard.save_html('output/reports/comprehensive_risk_report.html')
    
    logger.info("Risk modeling demonstration completed successfully")
    logger.info("Results and reports saved to the 'output' directory")

if __name__ == "__main__":
    main()
