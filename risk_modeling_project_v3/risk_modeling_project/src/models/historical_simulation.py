"""
Historical Simulation Module for Risk Modeling Framework.

This module provides functionality for historical simulation-based risk modeling.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class HistoricalSimulation:
    """
    Historical Simulation class for risk modeling.
    
    This class provides methods for estimating risk metrics using historical simulation.
    """
    
    def __init__(self):
        """
        Initialize the historical simulation model.
        """
        self.returns = None
        self.weights = None
        self.portfolio_returns = None
        self.bootstrap_samples = None
        self.num_samples = 10000
        self.sample_size = None
        self.calibrated = False
    
    def calibrate(self, historical_data, weights=None, regime=None):
        """
        Calibrate the historical simulation model with historical data.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        weights : array-like, optional
            Portfolio weights (if None, use equal weights)
        regime : str, optional
            Current market regime (not used in basic implementation)
        """
        self.returns = historical_data
        
        # Set portfolio weights
        if weights is None:
            self.weights = np.ones(len(historical_data.columns)) / len(historical_data.columns)
        else:
            self.weights = weights
        
        # Calculate portfolio returns
        self.portfolio_returns = historical_data.dot(self.weights)
        
        # Set sample size
        self.sample_size = len(self.portfolio_returns)
        
        self.calibrated = True
    
    def run_simulation(self, num_samples=None, bootstrap=False):
        """
        Run historical simulation.
        
        Parameters:
        -----------
        num_samples : int, optional
            Number of bootstrap samples (if using bootstrap)
        bootstrap : bool, default=False
            Whether to use bootstrap sampling
            
        Returns:
        --------
        pandas.Series or pandas.DataFrame
            Series or DataFrame containing simulation results
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Set number of samples
        if num_samples is not None:
            self.num_samples = num_samples
        
        if bootstrap:
            # Generate bootstrap samples
            bootstrap_indices = np.random.choice(
                len(self.portfolio_returns),
                size=(self.num_samples, len(self.portfolio_returns)),
                replace=True
            )
            
            # Create bootstrap samples
            bootstrap_samples = np.array([self.portfolio_returns.iloc[indices].values for indices in bootstrap_indices])
            
            # Store bootstrap samples
            self.bootstrap_samples = pd.DataFrame(bootstrap_samples.reshape(self.num_samples * len(self.portfolio_returns)))
            
            return self.bootstrap_samples
        else:
            # Use original portfolio returns
            return self.portfolio_returns
    
    def estimate_risk(self, confidence_level=0.95, bootstrap=False, num_samples=None):
        """
        Estimate risk metrics using historical simulation.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        bootstrap : bool, default=False
            Whether to use bootstrap sampling
        num_samples : int, optional
            Number of bootstrap samples (if using bootstrap)
            
        Returns:
        --------
        dict
            Dictionary containing risk metrics
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Run simulation if needed
        if bootstrap and (self.bootstrap_samples is None or (num_samples is not None and num_samples != self.num_samples)):
            self.run_simulation(num_samples=num_samples, bootstrap=True)
            simulation_results = self.bootstrap_samples
        elif bootstrap and self.bootstrap_samples is not None:
            simulation_results = self.bootstrap_samples
        else:
            simulation_results = self.portfolio_returns
        
        # Calculate VaR
        var = -np.percentile(simulation_results, 100 * (1 - confidence_level))
        
        # Calculate ES
        es = -simulation_results[simulation_results <= -var].mean()
        
        # Calculate other risk metrics
        mean = simulation_results.mean()
        std = simulation_results.std()
        skew = stats.skew(simulation_results)
        kurt = stats.kurtosis(simulation_results)
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'VaR': var,
            'ES': es,
            'mean': mean,
            'std': std,
            'skew': skew,
            'kurt': kurt,
            'max_drawdown': max_drawdown,
            'confidence_level': confidence_level
        }
    
    def plot_histogram(self, bootstrap=False, num_samples=None, figsize=(12, 6)):
        """
        Plot histogram of simulation results.
        
        Parameters:
        -----------
        bootstrap : bool, default=False
            Whether to use bootstrap sampling
        num_samples : int, optional
            Number of bootstrap samples (if using bootstrap)
        figsize : tuple, default=(12, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Run simulation if needed
        if bootstrap and (self.bootstrap_samples is None or (num_samples is not None and num_samples != self.num_samples)):
            self.run_simulation(num_samples=num_samples, bootstrap=True)
            simulation_results = self.bootstrap_samples
        elif bootstrap and self.bootstrap_samples is not None:
            simulation_results = self.bootstrap_samples
        else:
            simulation_results = self.portfolio_returns
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        sns.histplot(simulation_results, kde=True, ax=ax)
        
        # Calculate VaR and ES
        var_95 = -np.percentile(simulation_results, 5)
        var_99 = -np.percentile(simulation_results, 1)
        es_95 = -simulation_results[simulation_results <= -var_95].mean()
        es_99 = -simulation_results[simulation_results <= -var_99].mean()
        
        # Add VaR and ES lines
        ax.axvline(x=-var_95, color='red', linestyle='--', label=f'95% VaR: {var_95:.4f}')
        ax.axvline(x=-var_99, color='darkred', linestyle='--', label=f'99% VaR: {var_99:.4f}')
        ax.axvline(x=-es_95, color='blue', linestyle='--', label=f'95% ES: {es_95:.4f}')
        ax.axvline(x=-es_99, color='darkblue', linestyle='--', label=f'99% ES: {es_99:.4f}')
        
        # Set labels and title
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Returns (Historical Simulation)')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_returns(self, figsize=(12, 6)):
        """
        Plot historical portfolio returns.
        
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
            raise ValueError("Model not calibrated")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot returns
        ax.plot(self.portfolio_returns.index, self.portfolio_returns, color='blue', alpha=0.7)
        
        # Add mean line
        ax.axhline(y=self.portfolio_returns.mean(), color='red', linestyle='--', label=f'Mean: {self.portfolio_returns.mean():.4f}')
        
        # Add VaR lines
        var_95 = -np.percentile(self.portfolio_returns, 5)
        var_99 = -np.percentile(self.portfolio_returns, 1)
        
        ax.axhline(y=-var_95, color='orange', linestyle='--', label=f'95% VaR: {var_95:.4f}')
        ax.axhline(y=-var_99, color='purple', linestyle='--', label=f'99% VaR: {var_99:.4f}')
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Return')
        ax.set_title('Historical Portfolio Returns')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_cumulative_returns(self, figsize=(12, 6)):
        """
        Plot cumulative portfolio returns.
        
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
            raise ValueError("Model not calibrated")
        
        # Calculate cumulative returns
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot cumulative returns
        ax.plot(cumulative_returns.index, cumulative_returns, color='blue', alpha=0.7)
        
        # Calculate drawdown
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Create twin axis for drawdown
        ax2 = ax.twinx()
        
        # Plot drawdown
        ax2.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3, label='Drawdown')
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return', color='blue')
        ax2.set_ylabel('Drawdown', color='red')
        ax.set_title('Cumulative Portfolio Returns and Drawdown')
        
        # Set y-axis colors
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
