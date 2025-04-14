"""
Monte Carlo Simulation Module for Risk Modeling Framework.

This module provides functionality for Monte Carlo simulation-based risk modeling.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class MonteCarloSimulation:
    """
    Monte Carlo Simulation class for risk modeling.
    
    This class provides methods for simulating portfolio returns and estimating
    risk metrics using Monte Carlo simulation.
    """
    
    def __init__(self):
        """
        Initialize the Monte Carlo simulation model.
        """
        self.returns = None
        self.cov_matrix = None
        self.mean_returns = None
        self.weights = None
        self.simulation_results = None
        self.num_simulations = 10000
        self.time_horizon = 252  # Default to 1 year of trading days
        self.distribution = 'normal'  # Default to normal distribution
        self.calibrated = False
    
    def calibrate(self, historical_data, weights=None, regime=None):
        """
        Calibrate the Monte Carlo simulation model with historical data.
        
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
        
        # Calculate mean returns and covariance matrix
        self.mean_returns = historical_data.mean()
        self.cov_matrix = historical_data.cov()
        
        # Set portfolio weights
        if weights is None:
            self.weights = np.ones(len(historical_data.columns)) / len(historical_data.columns)
        else:
            self.weights = weights
        
        self.calibrated = True
    
    def run_simulation(self, num_simulations=None, time_horizon=None, distribution=None):
        """
        Run Monte Carlo simulation.
        
        Parameters:
        -----------
        num_simulations : int, optional
            Number of simulation paths
        time_horizon : int, optional
            Time horizon in days
        distribution : str, optional
            Distribution to use ('normal', 't', 'skewed_t')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing simulation results
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Set simulation parameters
        if num_simulations is not None:
            self.num_simulations = num_simulations
        
        if time_horizon is not None:
            self.time_horizon = time_horizon
        
        if distribution is not None:
            self.distribution = distribution
        
        # Initialize simulation results
        simulation_results = np.zeros((self.time_horizon, self.num_simulations))
        
        # Generate random returns based on distribution
        if self.distribution == 'normal':
            # Generate random returns from multivariate normal distribution
            random_returns = np.random.multivariate_normal(
                self.mean_returns,
                self.cov_matrix,
                size=(self.time_horizon, self.num_simulations)
            )
        elif self.distribution == 't':
            # Generate random returns from multivariate t distribution
            df = 5  # Degrees of freedom
            random_returns = self._generate_multivariate_t(
                self.mean_returns,
                self.cov_matrix,
                df,
                size=(self.time_horizon, self.num_simulations)
            )
        elif self.distribution == 'skewed_t':
            # Generate random returns from multivariate skewed t distribution
            df = 5  # Degrees of freedom
            skew = 0.5  # Skewness parameter
            random_returns = self._generate_multivariate_skewed_t(
                self.mean_returns,
                self.cov_matrix,
                df,
                skew,
                size=(self.time_horizon, self.num_simulations)
            )
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
        
        # Calculate portfolio returns
        for i in range(self.time_horizon):
            for j in range(self.num_simulations):
                simulation_results[i, j] = np.dot(random_returns[i, j], self.weights)
        
        # Convert to DataFrame
        self.simulation_results = pd.DataFrame(simulation_results)
        
        return self.simulation_results
    
    def _generate_multivariate_t(self, mean, cov, df, size):
        """
        Generate random samples from a multivariate t distribution.
        
        Parameters:
        -----------
        mean : array-like
            Mean vector
        cov : array-like
            Covariance matrix
        df : int
            Degrees of freedom
        size : tuple
            Size of the output
            
        Returns:
        --------
        numpy.ndarray
            Array of random samples
        """
        # Generate chi-squared random variables
        chi2 = np.random.chisquare(df, size=size[:-1] + (1,))
        
        # Generate multivariate normal random variables
        mvn = np.random.multivariate_normal(np.zeros_like(mean), cov, size=size)
        
        # Combine to create multivariate t distribution
        t = np.sqrt(df / chi2) * mvn
        
        # Add mean
        t += mean
        
        return t
    
    def _generate_multivariate_skewed_t(self, mean, cov, df, skew, size):
        """
        Generate random samples from a multivariate skewed t distribution.
        
        Parameters:
        -----------
        mean : array-like
            Mean vector
        cov : array-like
            Covariance matrix
        df : int
            Degrees of freedom
        skew : float
            Skewness parameter
        size : tuple
            Size of the output
            
        Returns:
        --------
        numpy.ndarray
            Array of random samples
        """
        # Generate multivariate t random variables
        t = self._generate_multivariate_t(np.zeros_like(mean), cov, df, size)
        
        # Generate random variables for skewness
        u = np.random.uniform(0, 1, size=size[:-1] + (1,))
        
        # Apply skewness transformation
        skewed_t = np.where(u < 0.5, t * (1 - skew), t * (1 + skew))
        
        # Add mean
        skewed_t += mean
        
        return skewed_t
    
    def estimate_risk(self, confidence_level=0.95, time_horizon=None):
        """
        Estimate risk metrics using Monte Carlo simulation results.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        time_horizon : int, optional
            Time horizon for risk metrics (if None, use all simulation results)
            
        Returns:
        --------
        dict
            Dictionary containing risk metrics
        """
        if self.simulation_results is None:
            self.run_simulation()
        
        # Set time horizon
        if time_horizon is None:
            time_horizon = self.time_horizon
        elif time_horizon > self.time_horizon:
            raise ValueError(f"Time horizon ({time_horizon}) exceeds simulation time horizon ({self.time_horizon})")
        
        # Get simulation results for the specified time horizon
        results = self.simulation_results.iloc[time_horizon - 1]
        
        # Calculate VaR
        var = -np.percentile(results, 100 * (1 - confidence_level))
        
        # Calculate ES
        es = -results[results <= -var].mean()
        
        # Calculate other risk metrics
        mean = results.mean()
        std = results.std()
        skew = stats.skew(results)
        kurt = stats.kurtosis(results)
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + self.simulation_results).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min().min()
        
        return {
            'VaR': var,
            'ES': es,
            'mean': mean,
            'std': std,
            'skew': skew,
            'kurt': kurt,
            'max_drawdown': max_drawdown,
            'confidence_level': confidence_level,
            'time_horizon': time_horizon
        }
    
    def plot_simulation_paths(self, num_paths=100, figsize=(12, 6)):
        """
        Plot simulation paths.
        
        Parameters:
        -----------
        num_paths : int, default=100
            Number of paths to plot
        figsize : tuple, default=(12, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.simulation_results is None:
            self.run_simulation()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert returns to cumulative returns
        cumulative_returns = (1 + self.simulation_results).cumprod()
        
        # Plot simulation paths
        for i in range(min(num_paths, self.num_simulations)):
            ax.plot(cumulative_returns.index, cumulative_returns.iloc[:, i], alpha=0.3)
        
        # Plot mean path
        mean_path = cumulative_returns.mean(axis=1)
        ax.plot(cumulative_returns.index, mean_path, color='red', linewidth=2, label='Mean Path')
        
        # Add confidence intervals
        lower_5 = cumulative_returns.quantile(0.05, axis=1)
        upper_95 = cumulative_returns.quantile(0.95, axis=1)
        
        ax.fill_between(cumulative_returns.index, lower_5, upper_95, color='red', alpha=0.2, label='90% Confidence Interval')
        
        # Set labels and title
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Monte Carlo Simulation Paths')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_histogram(self, time_horizon=None, figsize=(12, 6)):
        """
        Plot histogram of simulation results.
        
        Parameters:
        -----------
        time_horizon : int, optional
            Time horizon for histogram (if None, use all simulation results)
        figsize : tuple, default=(12, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.simulation_results is None:
            self.run_simulation()
        
        # Set time horizon
        if time_horizon is None:
            time_horizon = self.time_horizon
        elif time_horizon > self.time_horizon:
            raise ValueError(f"Time horizon ({time_horizon}) exceeds simulation time horizon ({self.time_horizon})")
        
        # Get simulation results for the specified time horizon
        results = self.simulation_results.iloc[time_horizon - 1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        sns.histplot(results, kde=True, ax=ax)
        
        # Calculate VaR and ES
        var_95 = -np.percentile(results, 5)
        var_99 = -np.percentile(results, 1)
        es_95 = -results[results <= -var_95].mean()
        es_99 = -results[results <= -var_99].mean()
        
        # Add VaR and ES lines
        ax.axvline(x=-var_95, color='red', linestyle='--', label=f'95% VaR: {var_95:.4f}')
        ax.axvline(x=-var_99, color='darkred', linestyle='--', label=f'99% VaR: {var_99:.4f}')
        ax.axvline(x=-es_95, color='blue', linestyle='--', label=f'95% ES: {es_95:.4f}')
        ax.axvline(x=-es_99, color='darkblue', linestyle='--', label=f'99% ES: {es_99:.4f}')
        
        # Set labels and title
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Returns (Time Horizon: {time_horizon} days)')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
