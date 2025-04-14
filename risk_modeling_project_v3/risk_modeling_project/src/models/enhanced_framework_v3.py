"""
Enhanced Risk Modeling Framework Version 3 Module.

This module provides the enhanced risk modeling framework version 3,
which integrates Bayesian methods, reinforcement learning, and specialized
modules for alternative assets.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gymnasium as gym
import networkx as nx
import pymc as pm

# Import base models
from src.models.portfolio import Portfolio
from src.models.monte_carlo import MonteCarloSimulation
from src.models.historical_simulation import HistoricalSimulation
from src.models.extreme_value_theory import ExtremeValueTheory
from src.models.stress_testing import StressTesting
from src.models.regime_analysis import RegimeAnalysis

# Import Version 2 enhancements
from src.models.regime_switching import RegimeSwitchingModel
from src.models.dynamic_calibration import DynamicCalibration
from src.models.ensemble_risk_model import EnsembleRiskModel
from src.models.machine_learning import MachineLearningRiskModel

# Import Version 3 enhancements
from src.models.bayesian_regime_detection import BayesianRegimeDetection
from src.models.hierarchical_bayesian_model import HierarchicalBayesianModel
from src.models.reinforcement_learning import RLRiskManager
from src.models.adversarial_training import AdversarialTraining
from src.models.multi_agent_system import MultiAgentSystem
from src.models.sector_risk_analysis import SectorRiskAnalysis
from src.models.alternative_assets import CryptoRiskModel, PrivateEquityRiskModel


class EnhancedRiskModelingFrameworkV3:
    """
    Enhanced Risk Modeling Framework Version 3 class.
    
    This class integrates all components of the risk modeling framework,
    including Bayesian methods, reinforcement learning, and specialized
    modules for alternative assets.
    """
    
    def __init__(self):
        """
        Initialize the enhanced risk modeling framework version 3.
        """
        # Initialize base components
        self.portfolio = None
        self.monte_carlo = MonteCarloSimulation()
        self.historical_simulation = HistoricalSimulation()
        self.extreme_value_theory = ExtremeValueTheory()
        self.stress_testing = StressTesting()
        self.regime_analysis = RegimeAnalysis()
        
        # Initialize Version 2 enhancements
        self.regime_switching = RegimeSwitchingModel()
        self.dynamic_calibration = DynamicCalibration()
        self.ensemble_risk_model = EnsembleRiskModel()
        self.machine_learning = MachineLearningRiskModel()
        
        # Initialize Version 3 enhancements
        self.bayesian_regime_detection = BayesianRegimeDetection()
        self.hierarchical_bayesian_model = HierarchicalBayesianModel()
        self.rl_risk_manager = RLRiskManager()
        self.adversarial_training = AdversarialTraining()
        self.multi_agent_system = MultiAgentSystem()
        self.sector_risk_analysis = SectorRiskAnalysis()
        self.crypto_risk_model = CryptoRiskModel()
        self.private_equity_risk_model = PrivateEquityRiskModel()
        
        # Initialize framework state
        self.calibrated = False
        self.current_regime = None
        self.uncertainty_estimates = None
        self.optimal_parameters = None
    
    def calibrate(self, historical_data, portfolio_weights=None, 
                  alternative_data=None, calibration_window=252,
                  use_bayesian=True, use_reinforcement_learning=True,
                  use_alternative_assets=True):
        """
        Calibrate the enhanced risk modeling framework with historical data.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        portfolio_weights : pandas.Series or dict, optional
            Portfolio weights
        alternative_data : dict, optional
            Dictionary containing alternative data sources
        calibration_window : int, default=252
            Calibration window length
        use_bayesian : bool, default=True
            Whether to use Bayesian methods
        use_reinforcement_learning : bool, default=True
            Whether to use reinforcement learning
        use_alternative_assets : bool, default=True
            Whether to use specialized modules for alternative assets
        """
        # Create portfolio
        if portfolio_weights is not None:
            self.portfolio = Portfolio(historical_data, portfolio_weights)
        else:
            # Equal weight portfolio
            weights = pd.Series(
                index=historical_data.columns,
                data=1.0 / len(historical_data.columns)
            )
            self.portfolio = Portfolio(historical_data, weights)
        
        # Determine optimal calibration window using dynamic calibration
        optimal_window = self.dynamic_calibration.optimize_window(
            historical_data,
            min_window=63,
            max_window=504,
            step=21
        )
        
        # Use optimal window if available
        if optimal_window is not None:
            calibration_window = optimal_window
        
        # Get calibration data
        calibration_data = historical_data.iloc[-calibration_window:]
        
        # Calibrate base components
        self.monte_carlo.calibrate(calibration_data)
        self.historical_simulation.calibrate(calibration_data)
        self.extreme_value_theory.calibrate(calibration_data)
        self.stress_testing.calibrate(calibration_data)
        self.regime_analysis.calibrate(calibration_data)
        
        # Calibrate Version 2 enhancements
        self.regime_switching.calibrate(calibration_data)
        self.ensemble_risk_model.calibrate(
            calibration_data,
            [self.monte_carlo, self.historical_simulation, self.extreme_value_theory]
        )
        self.machine_learning.calibrate(calibration_data)
        
        # Calibrate Version 3 enhancements
        if use_bayesian:
            self.bayesian_regime_detection.calibrate(calibration_data)
            self.hierarchical_bayesian_model.calibrate(calibration_data)
            
            # Set current regime based on Bayesian regime detection
            self.current_regime = self.bayesian_regime_detection.detect_regime(
                calibration_data.iloc[-21:]
            )
            
            # Calculate uncertainty estimates
            self.uncertainty_estimates = self.hierarchical_bayesian_model.estimate_uncertainty(
                calibration_data.iloc[-21:]
            )
        
        if use_reinforcement_learning:
            self.rl_risk_manager.calibrate(calibration_data)
            self.adversarial_training.calibrate(calibration_data)
            self.multi_agent_system.calibrate(calibration_data)
            
            # Get optimal parameters from RL risk manager
            self.optimal_parameters = self.rl_risk_manager.get_optimal_parameters(
                calibration_data.iloc[-21:]
            )
        
        if use_alternative_assets:
            self.sector_risk_analysis.calibrate(calibration_data)
            
            # Identify crypto assets
            crypto_assets = []
            for asset in historical_data.columns:
                if 'crypto' in asset.lower() or 'coin' in asset.lower() or 'token' in asset.lower():
                    crypto_assets.append(asset)
            
            # Identify private equity assets
            pe_assets = []
            for asset in historical_data.columns:
                if 'pe' in asset.lower() or 'private' in asset.lower() or 'equity' in asset.lower():
                    pe_assets.append(asset)
            
            # Calibrate specialized models if assets are available
            if crypto_assets:
                crypto_data = historical_data[crypto_assets]
                self.crypto_risk_model.calibrate(crypto_data)
            
            if pe_assets:
                pe_data = historical_data[pe_assets]
                self.private_equity_risk_model.calibrate(pe_data)
        
        self.calibrated = True
    
    def calculate_var(self, confidence_level=0.95, horizon=1, method='ensemble'):
        """
        Calculate Value at Risk (VaR) for the portfolio.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for VaR
        horizon : int, default=1
            Forecast horizon
        method : str, default='ensemble'
            Method for VaR calculation
            
        Returns:
        --------
        float
            VaR for the portfolio
        """
        if not self.calibrated:
            raise ValueError("Framework not calibrated")
        
        if method == 'monte_carlo':
            var = self.monte_carlo.calculate_var(
                self.portfolio,
                confidence_level,
                horizon
            )
        elif method == 'historical':
            var = self.historical_simulation.calculate_var(
                self.portfolio,
                confidence_level,
                horizon
            )
        elif method == 'evt':
            var = self.extreme_value_theory.calculate_var(
                self.portfolio,
                confidence_level,
                horizon
            )
        elif method == 'machine_learning':
            var = self.machine_learning.calculate_var(
                self.portfolio,
                confidence_level,
                horizon
            )
        elif method == 'bayesian':
            var = self.hierarchical_bayesian_model.calculate_var(
                self.portfolio,
                confidence_level,
                horizon
            )
        elif method == 'ensemble':
            # Use ensemble model with optimal weights
            var = self.ensemble_risk_model.calculate_var(
                self.portfolio,
                confidence_level,
                horizon
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Adjust VaR based on current regime if available
        if self.current_regime is not None:
            if self.current_regime == 'high_volatility':
                var *= 1.2  # Increase VaR in high volatility regime
            elif self.current_regime == 'low_volatility':
                var *= 0.9  # Decrease VaR in low volatility regime
        
        # Adjust VaR based on RL optimal parameters if available
        if self.optimal_parameters is not None and 'var_adjustment' in self.optimal_parameters:
            var *= self.optimal_parameters['var_adjustment']
        
        return var
    
    def calculate_expected_shortfall(self, confidence_level=0.95, horizon=1, method='ensemble'):
        """
        Calculate Expected Shortfall (ES) for the portfolio.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for ES
        horizon : int, default=1
            Forecast horizon
        method : str, default='ensemble'
            Method for ES calculation
            
        Returns:
        --------
        float
            ES for the portfolio
        """
        if not self.calibrated:
            raise ValueError("Framework not calibrated")
        
        if method == 'monte_carlo':
            es = self.monte_carlo.calculate_expected_shortfall(
                self.portfolio,
                confidence_level,
                horizon
            )
        elif method == 'historical':
            es = self.historical_simulation.calculate_expected_shortfall(
                self.portfolio,
                confidence_level,
                horizon
            )
        elif method == 'evt':
            es = self.extreme_value_theory.calculate_expected_shortfall(
                self.portfolio,
                confidence_level,
                horizon
            )
        elif method == 'machine_learning':
            es = self.machine_learning.calculate_expected_shortfall(
                self.portfolio,
                confidence_level,
                horizon
            )
        elif method == 'bayesian':
            es = self.hierarchical_bayesian_model.calculate_expected_shortfall(
                self.portfolio,
                confidence_level,
                horizon
            )
        elif method == 'ensemble':
            # Use ensemble model with optimal weights
            es = self.ensemble_risk_model.calculate_expected_shortfall(
                self.portfolio,
                confidence_level,
                horizon
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Adjust ES based on current regime if available
        if self.current_regime is not None:
            if self.current_regime == 'high_volatility':
                es *= 1.2  # Increase ES in high volatility regime
            elif self.current_regime == 'low_volatility':
                es *= 0.9  # Decrease ES in low volatility regime
        
        # Adjust ES based on RL optimal parameters if available
        if self.optimal_parameters is not None and 'es_adjustment' in self.optimal_parameters:
            es *= self.optimal_parameters['es_adjustment']
        
        return es
    
    def stress_test(self, scenarios, method='ensemble'):
        """
        Perform stress testing for the portfolio.
        
        Parameters:
        -----------
        scenarios : dict
            Dictionary mapping scenario names to scenario parameters
        method : str, default='ensemble'
            Method for stress testing
            
        Returns:
        --------
        pandas.DataFrame
            Stress test results
        """
        if not self.calibrated:
            raise ValueError("Framework not calibrated")
        
        # Use adversarial training to generate additional scenarios
        adversarial_scenarios = self.adversarial_training.generate_scenarios(
            self.portfolio.returns,
            num_scenarios=3
        )
        
        # Combine user-defined and adversarial scenarios
        all_scenarios = {**scenarios, **adversarial_scenarios}
        
        # Perform stress testing
        results = self.stress_testing.stress_test(
            self.portfolio,
            all_scenarios
        )
        
        return results
    
    def optimize_portfolio(self, constraints=None, objective='sharpe'):
        """
        Optimize portfolio weights.
        
        Parameters:
        -----------
        constraints : dict, optional
            Dictionary containing optimization constraints
        objective : str, default='sharpe'
            Optimization objective
            
        Returns:
        --------
        pandas.Series
            Optimized portfolio weights
        """
        if not self.calibrated:
            raise ValueError("Framework not calibrated")
        
        # Use multi-agent system for portfolio optimization
        optimal_weights = self.multi_agent_system.get_optimal_portfolio_allocation(
            [self.portfolio.returns.iloc[-20:]]
        )
        
        return pd.Series(
            index=self.portfolio.weights.index,
            data=optimal_weights
        )
    
    def get_risk_decomposition(self, method='ensemble'):
        """
        Get risk decomposition for the portfolio.
        
        Parameters:
        -----------
        method : str, default='ensemble'
            Method for risk decomposition
            
        Returns:
        --------
        pandas.Series
            Risk contribution of each asset
        """
        if not self.calibrated:
            raise ValueError("Framework not calibrated")
        
        # Calculate risk contributions
        if method == 'monte_carlo':
            risk_contrib = self.monte_carlo.calculate_risk_contribution(self.portfolio)
        elif method == 'historical':
            risk_contrib = self.historical_simulation.calculate_risk_contribution(self.portfolio)
        elif method == 'evt':
            risk_contrib = self.extreme_value_theory.calculate_risk_contribution(self.portfolio)
        elif method == 'machine_learning':
            risk_contrib = self.machine_learning.calculate_risk_contribution(self.portfolio)
        elif method == 'bayesian':
            risk_contrib = self.hierarchical_bayesian_model.calculate_risk_contribution(self.portfolio)
        elif method == 'ensemble':
            # Use ensemble model with optimal weights
            risk_contrib = self.ensemble_risk_model.calculate_risk_contribution(self.portfolio)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return risk_contrib
    
    def get_uncertainty_estimates(self, confidence_level=0.95):
        """
        Get uncertainty estimates for risk measures.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for uncertainty estimates
            
        Returns:
        --------
        dict
            Dictionary containing uncertainty estimates
        """
        if not self.calibrated:
            raise ValueError("Framework not calibrated")
        
        if self.uncertainty_estimates is None:
            raise ValueError("Uncertainty estimates not available")
        
        return self.uncertainty_estimates
    
    def get_regime_probabilities(self):
        """
        Get regime probabilities.
        
        Returns:
        --------
        pandas.Series
            Regime probabilities
        """
        if not self.calibrated:
            raise ValueError("Framework not calibrated")
        
        # Get regime probabilities from Bayesian regime detection
        regime_probs = self.bayesian_regime_detection.get_regime_probabilities()
        
        return regime_probs
    
    def plot_var_comparison(self, confidence_levels=[0.9, 0.95, 0.99], horizon=1, figsize=(12, 8)):
        """
        Plot VaR comparison across different methods.
        
        Parameters:
        -----------
        confidence_levels : list, default=[0.9, 0.95, 0.99]
            List of confidence levels
        horizon : int, default=1
            Forecast horizon
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.calibrated:
            raise ValueError("Framework not calibrated")
        
        # Initialize results
        methods = ['monte_carlo', 'historical', 'evt', 'machine_learning', 'bayesian', 'ensemble']
        results = pd.DataFrame(
            index=methods,
            columns=[f'VaR_{int(cl*100)}' for cl in confidence_levels]
        )
        
        # Calculate VaR for each method and confidence level
        for method in methods:
            for cl in confidence_levels:
                try:
                    results.loc[method, f'VaR_{int(cl*100)}'] = self.calculate_var(
                        confidence_level=cl,
                        horizon=horizon,
                        method=method
                    )
                except:
                    results.loc[method, f'VaR_{int(cl*100)}'] = np.nan
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot results
        results.plot(kind='bar', ax=ax)
        
        # Set title and labels
        ax.set_title(f'VaR Comparison (Horizon: {horizon} day)')
        ax.set_xlabel('Method')
        ax.set_ylabel('VaR')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_regime_detection(self, figsize=(12, 6)):
        """
        Plot regime detection results.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.calibrated:
            raise ValueError("Framework not calibrated")
        
        # Get regime probabilities
        regime_probs = self.get_regime_probabilities()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot regime probabilities
        regime_probs.plot(ax=ax)
        
        # Set title and labels
        ax.set_title('Regime Probabilities')
        ax.set_xlabel('Date')
        ax.set_ylabel('Probability')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_uncertainty_estimates(self, figsize=(12, 6)):
        """
        Plot uncertainty estimates for risk measures.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.calibrated:
            raise ValueError("Framework not calibrated")
        
        if self.uncertainty_estimates is None:
            raise ValueError("Uncertainty estimates not available")
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot VaR uncertainty
        var_mean = self.uncertainty_estimates['var_mean']
        var_lower = self.uncertainty_estimates['var_lower']
        var_upper = self.uncertainty_estimates['var_upper']
        
        axes[0].bar(0, var_mean, yerr=[[var_mean - var_lower], [var_upper - var_mean]])
        axes[0].set_title('VaR with Uncertainty')
        axes[0].set_ylabel('VaR')
        axes[0].set_xticks([0])
        axes[0].set_xticklabels(['VaR (95%)'])
        axes[0].grid(True, alpha=0.3)
        
        # Plot ES uncertainty
        es_mean = self.uncertainty_estimates['es_mean']
        es_lower = self.uncertainty_estimates['es_lower']
        es_upper = self.uncertainty_estimates['es_upper']
        
        axes[1].bar(0, es_mean, yerr=[[es_mean - es_lower], [es_upper - es_mean]])
        axes[1].set_title('Expected Shortfall with Uncertainty')
        axes[1].set_ylabel('ES')
        axes[1].set_xticks([0])
        axes[1].set_xticklabels(['ES (95%)'])
        axes[1].grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_stress_test_results(self, scenarios, figsize=(12, 8)):
        """
        Plot stress test results.
        
        Parameters:
        -----------
        scenarios : dict
            Dictionary mapping scenario names to scenario parameters
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.calibrated:
            raise ValueError("Framework not calibrated")
        
        # Perform stress testing
        results = self.stress_test(scenarios)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot results
        results.plot(kind='bar', ax=ax)
        
        # Set title and labels
        ax.set_title('Stress Test Results')
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Portfolio Return')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_risk_decomposition(self, method='ensemble', figsize=(12, 6)):
        """
        Plot risk decomposition for the portfolio.
        
        Parameters:
        -----------
        method : str, default='ensemble'
            Method for risk decomposition
        figsize : tuple, default=(12, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.calibrated:
            raise ValueError("Framework not calibrated")
        
        # Get risk decomposition
        risk_contrib = self.get_risk_decomposition(method)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot risk decomposition
        risk_contrib.plot(kind='pie', ax=ax, autopct='%1.1f%%')
        
        # Set title
        ax.set_title(f'Risk Decomposition ({method.capitalize()} Method)')
        
        # Equal aspect ratio
        ax.set_aspect('equal')
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_optimal_portfolio(self, figsize=(12, 6)):
        """
        Plot optimal portfolio weights.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.calibrated:
            raise ValueError("Framework not calibrated")
        
        # Get optimal portfolio weights
        optimal_weights = self.optimize_portfolio()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot optimal weights
        optimal_weights.plot(kind='bar', ax=ax)
        
        # Set title and labels
        ax.set_title('Optimal Portfolio Weights')
        ax.set_xlabel('Asset')
        ax.set_ylabel('Weight')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def generate_report(self, output_file=None):
        """
        Generate comprehensive risk report.
        
        Parameters:
        -----------
        output_file : str, optional
            Output file path
            
        Returns:
        --------
        str
            Report content
        """
        if not self.calibrated:
            raise ValueError("Framework not calibrated")
        
        # Initialize report
        report = []
        
        # Add header
        report.append("# Risk Modeling Framework Version 3 Report")
        report.append("")
        
        # Add portfolio summary
        report.append("## Portfolio Summary")
        report.append("")
        report.append(f"Number of assets: {len(self.portfolio.weights)}")
        report.append(f"Portfolio expected return: {self.portfolio.expected_return:.2%}")
        report.append(f"Portfolio volatility: {self.portfolio.volatility:.2%}")
        report.append(f"Portfolio Sharpe ratio: {self.portfolio.sharpe_ratio:.2f}")
        report.append("")
        
        # Add risk measures
        report.append("## Risk Measures")
        report.append("")
        report.append("### Value at Risk (VaR)")
        report.append("")
        report.append("| Method | VaR (95%) | VaR (99%) |")
        report.append("|--------|-----------|-----------|")
        
        for method in ['monte_carlo', 'historical', 'evt', 'machine_learning', 'bayesian', 'ensemble']:
            try:
                var_95 = self.calculate_var(confidence_level=0.95, method=method)
                var_99 = self.calculate_var(confidence_level=0.99, method=method)
                report.append(f"| {method.capitalize()} | {var_95:.2%} | {var_99:.2%} |")
            except:
                report.append(f"| {method.capitalize()} | N/A | N/A |")
        
        report.append("")
        report.append("### Expected Shortfall (ES)")
        report.append("")
        report.append("| Method | ES (95%) | ES (99%) |")
        report.append("|--------|----------|----------|")
        
        for method in ['monte_carlo', 'historical', 'evt', 'machine_learning', 'bayesian', 'ensemble']:
            try:
                es_95 = self.calculate_expected_shortfall(confidence_level=0.95, method=method)
                es_99 = self.calculate_expected_shortfall(confidence_level=0.99, method=method)
                report.append(f"| {method.capitalize()} | {es_95:.2%} | {es_99:.2%} |")
            except:
                report.append(f"| {method.capitalize()} | N/A | N/A |")
        
        report.append("")
        
        # Add regime detection
        report.append("## Regime Detection")
        report.append("")
        report.append(f"Current regime: {self.current_regime}")
        report.append("")
        report.append("### Regime Probabilities")
        report.append("")
        
        regime_probs = self.get_regime_probabilities()
        for regime, prob in regime_probs.items():
            report.append(f"- {regime}: {prob:.2%}")
        
        report.append("")
        
        # Add uncertainty estimates
        if self.uncertainty_estimates is not None:
            report.append("## Uncertainty Estimates")
            report.append("")
            report.append("### VaR Uncertainty")
            report.append("")
            report.append(f"- Mean: {self.uncertainty_estimates['var_mean']:.2%}")
            report.append(f"- Lower bound (95%): {self.uncertainty_estimates['var_lower']:.2%}")
            report.append(f"- Upper bound (95%): {self.uncertainty_estimates['var_upper']:.2%}")
            report.append("")
            report.append("### ES Uncertainty")
            report.append("")
            report.append(f"- Mean: {self.uncertainty_estimates['es_mean']:.2%}")
            report.append(f"- Lower bound (95%): {self.uncertainty_estimates['es_lower']:.2%}")
            report.append(f"- Upper bound (95%): {self.uncertainty_estimates['es_upper']:.2%}")
            report.append("")
        
        # Add stress testing
        report.append("## Stress Testing")
        report.append("")
        report.append("### Standard Scenarios")
        report.append("")
        
        standard_scenarios = {
            'Market Crash': {'market_shock': -0.1},
            'Interest Rate Hike': {'market_shock': -0.05},
            'Volatility Spike': {'volatility_multiplier': 2.0},
            'Correlation Breakdown': {'correlation_multiplier': 0.5}
        }
        
        results = self.stress_test(standard_scenarios)
        
        report.append("| Scenario | Portfolio Return | VaR (95%) | ES (95%) |")
        report.append("|----------|------------------|-----------|----------|")
        
        for scenario in results.index:
            portfolio_return = results.loc[scenario, 'portfolio_return']
            var_95 = results.loc[scenario, 'var_95']
            es_95 = results.loc[scenario, 'expected_shortfall_95']
            
            report.append(f"| {scenario} | {portfolio_return:.2%} | {var_95:.2%} | {es_95:.2%} |")
        
        report.append("")
        
        # Add adversarial scenarios
        report.append("### Adversarial Scenarios")
        report.append("")
        
        adversarial_scenarios = self.adversarial_training.generate_scenarios(
            self.portfolio.returns,
            num_scenarios=3
        )
        
        results = self.stress_test(adversarial_scenarios)
        
        report.append("| Scenario | Portfolio Return | VaR (95%) | ES (95%) |")
        report.append("|----------|------------------|-----------|----------|")
        
        for scenario in results.index:
            portfolio_return = results.loc[scenario, 'portfolio_return']
            var_95 = results.loc[scenario, 'var_95']
            es_95 = results.loc[scenario, 'expected_shortfall_95']
            
            report.append(f"| {scenario} | {portfolio_return:.2%} | {var_95:.2%} | {es_95:.2%} |")
        
        report.append("")
        
        # Add optimal portfolio
        report.append("## Optimal Portfolio")
        report.append("")
        
        optimal_weights = self.optimize_portfolio()
        
        report.append("| Asset | Weight |")
        report.append("|-------|--------|")
        
        for asset, weight in optimal_weights.items():
            report.append(f"| {asset} | {weight:.2%} |")
        
        report.append("")
        
        # Add risk decomposition
        report.append("## Risk Decomposition")
        report.append("")
        
        risk_contrib = self.get_risk_decomposition()
        
        report.append("| Asset | Risk Contribution |")
        report.append("|-------|------------------|")
        
        for asset, contrib in risk_contrib.items():
            report.append(f"| {asset} | {contrib:.2%} |")
        
        report.append("")
        
        # Add conclusion
        report.append("## Conclusion")
        report.append("")
        report.append("This report was generated using the Enhanced Risk Modeling Framework Version 3, which integrates:")
        report.append("")
        report.append("1. Bayesian methods for uncertainty quantification")
        report.append("2. Reinforcement learning for adaptive risk management")
        report.append("3. Specialized modules for alternative assets")
        report.append("")
        report.append("The framework provides a comprehensive view of portfolio risk, including:")
        report.append("")
        report.append("- Multiple risk measures (VaR, ES) with uncertainty estimates")
        report.append("- Regime detection and regime-specific risk adjustments")
        report.append("- Stress testing with standard and adversarial scenarios")
        report.append("- Portfolio optimization and risk decomposition")
        report.append("")
        
        # Join report
        report_content = "\n".join(report)
        
        # Save report if output file is provided
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write(report_content)
        
        return report_content
