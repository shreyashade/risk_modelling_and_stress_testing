"""
Extreme Value Theory Module for Risk Modeling Framework.

This module provides functionality for tail risk modeling using Extreme Value Theory (EVT).
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
from arch.bootstrap import IIDBootstrap

class ExtremeValueTheory:
    """
    Extreme Value Theory class for risk modeling.
    
    This class provides methods for modeling tail risk using Extreme Value Theory (EVT),
    including Peak-Over-Threshold (POT) method with Generalized Pareto Distribution (GPD)
    and Block Maxima method with Generalized Extreme Value (GEV) distribution.
    """
    
    def __init__(self):
        """
        Initialize the Extreme Value Theory model.
        """
        self.returns = None
        self.threshold = None
        self.gpd_params = None
        self.gev_params = None
        self.tail_index = None
        self.exceedances = None
        self.calibrated = False
        self.confidence_intervals = None
    
    def calibrate(self, historical_data, threshold_method='quantile', 
                  threshold_value=0.05, block_size=21):
        """
        Calibrate the EVT model with historical data.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        threshold_method : str, default='quantile'
            Method for threshold selection ('quantile', 'hill', or 'fixed')
        threshold_value : float, default=0.05
            Value for threshold selection (quantile or fixed value)
        block_size : int, default=21
            Block size for Block Maxima method
        """
        self.returns = historical_data
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        portfolio_returns = self.returns.dot(portfolio_weights)
        
        # Select threshold
        if threshold_method == 'quantile':
            self.threshold = portfolio_returns.quantile(threshold_value)
        elif threshold_method == 'hill':
            self.threshold = self._hill_estimator(portfolio_returns)
        elif threshold_method == 'fixed':
            self.threshold = threshold_value
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")
        
        # Extract exceedances
        exceedances = portfolio_returns[portfolio_returns < -self.threshold] * -1
        self.exceedances = exceedances
        
        # Fit GPD to exceedances
        if len(exceedances) > 0:
            try:
                # Fit GPD using MLE
                gpd_shape, _, gpd_scale = stats.genpareto.fit(exceedances)
                self.gpd_params = {'shape': gpd_shape, 'scale': gpd_scale}
                
                # Calculate tail index
                self.tail_index = 1 / gpd_shape if gpd_shape > 0 else np.inf
                
                # Calculate confidence intervals using bootstrap
                self._calculate_confidence_intervals(exceedances)
            except:
                warnings.warn("Failed to fit GPD, using empirical distribution instead")
                self.gpd_params = None
        else:
            warnings.warn("No exceedances found, using empirical distribution instead")
            self.gpd_params = None
        
        # Block Maxima method
        try:
            # Reshape data into blocks
            n_blocks = len(portfolio_returns) // block_size
            if n_blocks > 0:
                blocks = portfolio_returns.values[:n_blocks * block_size].reshape(n_blocks, block_size)
                block_minima = -np.min(blocks, axis=1)  # Convert to maxima of negative returns
                
                # Fit GEV to block minima
                gev_shape, gev_loc, gev_scale = stats.genextreme.fit(block_minima)
                self.gev_params = {'shape': gev_shape, 'loc': gev_loc, 'scale': gev_scale}
            else:
                self.gev_params = None
        except:
            warnings.warn("Failed to fit GEV, using empirical distribution instead")
            self.gev_params = None
        
        self.calibrated = True
    
    def _hill_estimator(self, returns, k=None):
        """
        Hill estimator for threshold selection.
        
        Parameters:
        -----------
        returns : pandas.Series
            Returns data
        k : int, optional
            Number of order statistics to use
            
        Returns:
        --------
        float
            Threshold value
        """
        # Sort returns in descending order (for negative returns)
        sorted_returns = -np.sort(-returns.values)
        
        # Set k if not provided
        if k is None:
            k = int(np.sqrt(len(returns)))
        
        # Calculate Hill estimator
        log_returns = np.log(sorted_returns[:k])
        log_k_return = np.log(sorted_returns[k-1])
        hill_estimator = np.mean(log_returns) - log_k_return
        
        # Return threshold
        return sorted_returns[k-1]
    
    def _calculate_confidence_intervals(self, exceedances, confidence_level=0.95, n_bootstrap=1000):
        """
        Calculate confidence intervals for GPD parameters using bootstrap.
        
        Parameters:
        -----------
        exceedances : pandas.Series
            Exceedances data
        confidence_level : float, default=0.95
            Confidence level
        n_bootstrap : int, default=1000
            Number of bootstrap samples
        """
        # Initialize bootstrap
        bootstrap = IIDBootstrap(exceedances)
        
        # Define function to estimate GPD parameters
        def gpd_params(x):
            try:
                shape, _, scale = stats.genpareto.fit(x)
                return np.array([shape, scale])
            except:
                return np.array([np.nan, np.nan])
        
        # Perform bootstrap
        bootstrap_results = bootstrap.apply(gpd_params, n_bootstrap)
        
        # Calculate confidence intervals
        lower_percentile = (1 - confidence_level) / 2
        upper_percentile = 1 - lower_percentile
        
        # Extract results
        shape_samples = bootstrap_results[:, 0]
        scale_samples = bootstrap_results[:, 1]
        
        # Remove NaN values
        valid_shape = ~np.isnan(shape_samples)
        valid_scale = ~np.isnan(scale_samples)
        
        if np.sum(valid_shape) > 0 and np.sum(valid_scale) > 0:
            shape_ci = np.percentile(shape_samples[valid_shape], [lower_percentile * 100, upper_percentile * 100])
            scale_ci = np.percentile(scale_samples[valid_scale], [lower_percentile * 100, upper_percentile * 100])
            
            self.confidence_intervals = {
                'shape': {'lower': shape_ci[0], 'upper': shape_ci[1]},
                'scale': {'lower': scale_ci[0], 'upper': scale_ci[1]}
            }
        else:
            self.confidence_intervals = None
    
    def calculate_var(self, portfolio, confidence_level=0.95, horizon=1, method='gpd'):
        """
        Calculate Value at Risk (VaR) using EVT.
        
        Parameters:
        -----------
        portfolio : Portfolio
            Portfolio object
        confidence_level : float, default=0.95
            Confidence level for VaR
        horizon : int, default=1
            Forecast horizon
        method : str, default='gpd'
            Method for VaR calculation ('gpd', 'gev', or 'empirical')
            
        Returns:
        --------
        float
            VaR for the portfolio
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Calculate portfolio returns (equal weight if weights not provided)
        if portfolio.weights is None:
            portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        else:
            portfolio_weights = portfolio.weights.values
        
        portfolio_returns = self.returns.dot(portfolio_weights)
        
        # Calculate VaR
        if method == 'gpd' and self.gpd_params is not None:
            # GPD parameters
            shape = self.gpd_params['shape']
            scale = self.gpd_params['scale']
            
            # Number of exceedances
            n_exceed = len(self.exceedances)
            n_total = len(portfolio_returns)
            
            # Probability of exceeding threshold
            p_exceed = n_exceed / n_total
            
            # Calculate VaR
            q = 1 - confidence_level
            if shape != 0:
                var = self.threshold + (scale / shape) * ((q / p_exceed) ** (-shape) - 1)
            else:
                var = self.threshold - scale * np.log(q / p_exceed)
        elif method == 'gev' and self.gev_params is not None:
            # GEV parameters
            shape = self.gev_params['shape']
            loc = self.gev_params['loc']
            scale = self.gev_params['scale']
            
            # Calculate VaR
            q = 1 - confidence_level
            var = -stats.genextreme.ppf(q, shape, loc=loc, scale=scale)
        else:
            # Empirical VaR
            var = -portfolio_returns.quantile(1 - confidence_level)
        
        # Scale VaR for horizon
        var = var * np.sqrt(horizon)
        
        return var
    
    def calculate_expected_shortfall(self, portfolio, confidence_level=0.95, horizon=1, method='gpd'):
        """
        Calculate Expected Shortfall (ES) using EVT.
        
        Parameters:
        -----------
        portfolio : Portfolio
            Portfolio object
        confidence_level : float, default=0.95
            Confidence level for ES
        horizon : int, default=1
            Forecast horizon
        method : str, default='gpd'
            Method for ES calculation ('gpd', 'gev', or 'empirical')
            
        Returns:
        --------
        float
            ES for the portfolio
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Calculate portfolio returns (equal weight if weights not provided)
        if portfolio.weights is None:
            portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        else:
            portfolio_weights = portfolio.weights.values
        
        portfolio_returns = self.returns.dot(portfolio_weights)
        
        # Calculate VaR
        var = self.calculate_var(portfolio, confidence_level, horizon, method)
        
        # Calculate ES
        if method == 'gpd' and self.gpd_params is not None:
            # GPD parameters
            shape = self.gpd_params['shape']
            scale = self.gpd_params['scale']
            
            # Number of exceedances
            n_exceed = len(self.exceedances)
            n_total = len(portfolio_returns)
            
            # Probability of exceeding threshold
            p_exceed = n_exceed / n_total
            
            # Calculate ES
            q = 1 - confidence_level
            if shape < 1:
                es = var / (1 - shape) + (scale - shape * self.threshold) / (1 - shape)
            else:
                # Shape >= 1 means infinite mean, use empirical ES
                es = self._calculate_empirical_es(portfolio_returns, confidence_level)
        elif method == 'gev' and self.gev_params is not None:
            # GEV parameters
            shape = self.gev_params['shape']
            loc = self.gev_params['loc']
            scale = self.gev_params['scale']
            
            # Calculate ES
            if shape < 1:
                # Calculate ES using numerical integration
                q = 1 - confidence_level
                var_q = -stats.genextreme.ppf(q, shape, loc=loc, scale=scale)
                
                def integrand(p):
                    return -stats.genextreme.ppf(p, shape, loc=loc, scale=scale)
                
                # Numerical integration
                n_points = 1000
                p_values = np.linspace(0, q, n_points)
                integrand_values = integrand(p_values)
                es = np.trapz(integrand_values, p_values) / q
            else:
                # Shape >= 1 means infinite mean, use empirical ES
                es = self._calculate_empirical_es(portfolio_returns, confidence_level)
        else:
            # Empirical ES
            es = self._calculate_empirical_es(portfolio_returns, confidence_level)
        
        # Scale ES for horizon
        es = es * np.sqrt(horizon)
        
        return es
    
    def _calculate_empirical_es(self, returns, confidence_level):
        """
        Calculate empirical Expected Shortfall.
        
        Parameters:
        -----------
        returns : pandas.Series
            Returns data
        confidence_level : float
            Confidence level
            
        Returns:
        --------
        float
            Empirical ES
        """
        # Calculate VaR
        var = -returns.quantile(1 - confidence_level)
        
        # Calculate ES
        es = -returns[returns < -var].mean()
        
        return es
    
    def calculate_risk_contribution(self, portfolio):
        """
        Calculate risk contribution of each asset.
        
        Parameters:
        -----------
        portfolio : Portfolio
            Portfolio object
            
        Returns:
        --------
        pandas.Series
            Risk contribution of each asset
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Calculate portfolio returns
        portfolio_returns = self.returns.dot(portfolio.weights)
        
        # Calculate VaR
        var = self.calculate_var(portfolio)
        
        # Calculate risk contribution
        risk_contrib = np.zeros(len(portfolio.weights))
        
        for i in range(len(portfolio.weights)):
            # Calculate marginal VaR
            h = 0.0001
            weights_up = portfolio.weights.copy()
            weights_up[i] += h
            
            # Normalize weights
            weights_up = weights_up / weights_up.sum()
            
            # Create temporary portfolio
            temp_portfolio = type(portfolio)(self.returns, weights_up)
            
            # Calculate VaR with increased weight
            var_up = self.calculate_var(temp_portfolio)
            
            # Calculate marginal VaR
            marginal_var = (var_up - var) / h
            
            # Calculate risk contribution
            risk_contrib[i] = portfolio.weights[i] * marginal_var
        
        # Normalize risk contribution
        risk_contrib = risk_contrib / risk_contrib.sum()
        
        return pd.Series(
            index=portfolio.weights.index,
            data=risk_contrib
        )
    
    def simulate_returns(self, num_scenarios=1000, horizon=1):
        """
        Simulate returns using EVT model.
        
        Parameters:
        -----------
        num_scenarios : int, default=1000
            Number of scenarios to simulate
        horizon : int, default=1
            Simulation horizon
            
        Returns:
        --------
        pandas.DataFrame
            Simulated returns
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        portfolio_returns = self.returns.dot(portfolio_weights)
        
        # Initialize simulated returns
        simulated_returns = np.zeros((num_scenarios, horizon))
        
        # Simulate returns for each scenario
        for i in range(num_scenarios):
            for t in range(horizon):
                # Generate random uniform
                u = np.random.uniform()
                
                # Generate return
                if u < len(self.exceedances) / len(portfolio_returns):
                    # Generate from GPD
                    if self.gpd_params is not None:
                        shape = self.gpd_params['shape']
                        scale = self.gpd_params['scale']
                        
                        # Generate from GPD
                        if shape != 0:
                            exceedance = stats.genpareto.rvs(shape, scale=scale)
                        else:
                            exceedance = stats.expon.rvs(scale=scale)
                        
                        simulated_returns[i, t] = -(self.threshold + exceedance)
                    else:
                        # Sample from empirical exceedances
                        simulated_returns[i, t] = -np.random.choice(self.exceedances)
                else:
                    # Sample from empirical non-exceedances
                    non_exceedances = portfolio_returns[portfolio_returns >= -self.threshold]
                    simulated_returns[i, t] = np.random.choice(non_exceedances)
        
        # Convert to DataFrame
        simulated_returns_df = pd.DataFrame(
            index=range(num_scenarios),
            columns=[f'Scenario_{i}' for i in range(horizon)],
            data=simulated_returns
        )
        
        return simulated_returns_df
    
    def plot_tail_distribution(self, figsize=(12, 6)):
        """
        Plot tail distribution.
        
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
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        portfolio_returns = self.returns.dot(portfolio_weights)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot empirical tail distribution
        sorted_returns = np.sort(-portfolio_returns.values)
        p = np.arange(1, len(sorted_returns) + 1) / (len(sorted_returns) + 1)
        
        ax.loglog(sorted_returns, 1 - p, 'o', markersize=3, label='Empirical')
        
        # Plot GPD tail distribution
        if self.gpd_params is not None:
            shape = self.gpd_params['shape']
            scale = self.gpd_params['scale']
            
            # Generate points for GPD
            x = np.linspace(self.threshold, sorted_returns.max(), 1000)
            
            # Calculate GPD CDF
            if shape != 0:
                y = 1 - (1 + shape * (x - self.threshold) / scale) ** (-1 / shape)
            else:
                y = 1 - np.exp(-(x - self.threshold) / scale)
            
            # Adjust for threshold
            n_exceed = len(self.exceedances)
            n_total = len(portfolio_returns)
            p_exceed = n_exceed / n_total
            
            y = 1 - p_exceed * (1 - y)
            
            ax.loglog(x, 1 - y, 'r-', linewidth=2, label='GPD')
            
            # Plot confidence intervals if available
            if self.confidence_intervals is not None:
                shape_lower = self.confidence_intervals['shape']['lower']
                shape_upper = self.confidence_intervals['shape']['upper']
                scale_lower = self.confidence_intervals['scale']['lower']
                scale_upper = self.confidence_intervals['scale']['upper']
                
                # Lower bound
                if shape_lower != 0:
                    y_lower = 1 - (1 + shape_lower * (x - self.threshold) / scale_lower) ** (-1 / shape_lower)
                else:
                    y_lower = 1 - np.exp(-(x - self.threshold) / scale_lower)
                
                y_lower = 1 - p_exceed * (1 - y_lower)
                
                # Upper bound
                if shape_upper != 0:
                    y_upper = 1 - (1 + shape_upper * (x - self.threshold) / scale_upper) ** (-1 / shape_upper)
                else:
                    y_upper = 1 - np.exp(-(x - self.threshold) / scale_upper)
                
                y_upper = 1 - p_exceed * (1 - y_upper)
                
                ax.loglog(x, 1 - y_lower, 'r--', linewidth=1, alpha=0.5)
                ax.loglog(x, 1 - y_upper, 'r--', linewidth=1, alpha=0.5)
                ax.fill_between(x, 1 - y_lower, 1 - y_upper, color='r', alpha=0.1)
        
        # Set title and labels
        ax.set_title('Tail Distribution')
        ax.set_xlabel('Loss')
        ax.set_ylabel('Exceedance Probability')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_mean_excess(self, figsize=(12, 6)):
        """
        Plot mean excess function.
        
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
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        portfolio_returns = self.returns.dot(portfolio_weights)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate mean excess function
        sorted_returns = np.sort(-portfolio_returns.values)
        thresholds = sorted_returns[:-1]  # Exclude maximum
        mean_excess = np.zeros(len(thresholds))
        
        for i, threshold in enumerate(thresholds):
            exceedances = sorted_returns[sorted_returns > threshold] - threshold
            mean_excess[i] = exceedances.mean()
        
        # Plot mean excess function
        ax.plot(thresholds, mean_excess, 'o-', markersize=3)
        
        # Add vertical line at selected threshold
        ax.axvline(self.threshold, color='r', linestyle='--', label=f'Threshold: {self.threshold:.4f}')
        
        # Set title and labels
        ax.set_title('Mean Excess Function')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Mean Excess')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_hill_estimator(self, figsize=(12, 6)):
        """
        Plot Hill estimator.
        
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
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        portfolio_returns = self.returns.dot(portfolio_weights)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate Hill estimator
        sorted_returns = -np.sort(-portfolio_returns.values)
        k_values = np.arange(10, min(100, len(sorted_returns) // 2))
        hill_estimates = np.zeros(len(k_values))
        
        for i, k in enumerate(k_values):
            log_returns = np.log(sorted_returns[:k])
            log_k_return = np.log(sorted_returns[k-1])
            hill_estimates[i] = 1 / (np.mean(log_returns) - log_k_return)
        
        # Plot Hill estimator
        ax.plot(k_values, hill_estimates, 'o-', markersize=3)
        
        # Set title and labels
        ax.set_title('Hill Estimator')
        ax.set_xlabel('k')
        ax.set_ylabel('Hill Estimator')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_var_es(self, confidence_levels=np.linspace(0.9, 0.99, 10), figsize=(12, 6)):
        """
        Plot VaR and ES for different confidence levels.
        
        Parameters:
        -----------
        confidence_levels : array-like, default=np.linspace(0.9, 0.99, 10)
            Confidence levels
        figsize : tuple, default=(12, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Create dummy portfolio
        portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        portfolio = type('Portfolio', (), {'weights': pd.Series(portfolio_weights)})
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate VaR and ES for different confidence levels
        var_gpd = np.zeros(len(confidence_levels))
        es_gpd = np.zeros(len(confidence_levels))
        var_empirical = np.zeros(len(confidence_levels))
        es_empirical = np.zeros(len(confidence_levels))
        
        for i, cl in enumerate(confidence_levels):
            var_gpd[i] = self.calculate_var(portfolio, cl, method='gpd')
            es_gpd[i] = self.calculate_expected_shortfall(portfolio, cl, method='gpd')
            var_empirical[i] = self.calculate_var(portfolio, cl, method='empirical')
            es_empirical[i] = self.calculate_expected_shortfall(portfolio, cl, method='empirical')
        
        # Plot VaR and ES
        ax.plot(confidence_levels, var_gpd, 'b-', linewidth=2, label='VaR (GPD)')
        ax.plot(confidence_levels, es_gpd, 'r-', linewidth=2, label='ES (GPD)')
        ax.plot(confidence_levels, var_empirical, 'b--', linewidth=1, label='VaR (Empirical)')
        ax.plot(confidence_levels, es_empirical, 'r--', linewidth=1, label='ES (Empirical)')
        
        # Set title and labels
        ax.set_title('VaR and ES for Different Confidence Levels')
        ax.set_xlabel('Confidence Level')
        ax.set_ylabel('Value')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_return_distribution(self, figsize=(12, 6)):
        """
        Plot return distribution with GPD fit.
        
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
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        portfolio_returns = self.returns.dot(portfolio_weights)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot return distribution
        sns.histplot(portfolio_returns, bins=50, kde=True, ax=ax)
        
        # Add vertical line at threshold
        ax.axvline(-self.threshold, color='r', linestyle='--', label=f'Threshold: {self.threshold:.4f}')
        
        # Set title and labels
        ax.set_title('Return Distribution')
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        
        # Add legend
        ax.legend()
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_qq_plot(self, figsize=(12, 6)):
        """
        Plot QQ plot for GPD fit.
        
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
        
        if self.gpd_params is None or len(self.exceedances) == 0:
            raise ValueError("GPD not fitted or no exceedances found")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get GPD parameters
        shape = self.gpd_params['shape']
        scale = self.gpd_params['scale']
        
        # Calculate theoretical quantiles
        p = np.arange(1, len(self.exceedances) + 1) / (len(self.exceedances) + 1)
        theoretical_quantiles = stats.genpareto.ppf(p, shape, scale=scale)
        
        # Sort exceedances
        empirical_quantiles = np.sort(self.exceedances)
        
        # Plot QQ plot
        ax.scatter(theoretical_quantiles, empirical_quantiles)
        
        # Add diagonal line
        max_val = max(theoretical_quantiles.max(), empirical_quantiles.max())
        ax.plot([0, max_val], [0, max_val], 'r--')
        
        # Set title and labels
        ax.set_title('QQ Plot for GPD Fit')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Empirical Quantiles')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_pp_plot(self, figsize=(12, 6)):
        """
        Plot PP plot for GPD fit.
        
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
        
        if self.gpd_params is None or len(self.exceedances) == 0:
            raise ValueError("GPD not fitted or no exceedances found")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get GPD parameters
        shape = self.gpd_params['shape']
        scale = self.gpd_params['scale']
        
        # Calculate empirical CDF
        sorted_exceedances = np.sort(self.exceedances)
        p_empirical = np.arange(1, len(sorted_exceedances) + 1) / (len(sorted_exceedances) + 1)
        
        # Calculate theoretical CDF
        p_theoretical = stats.genpareto.cdf(sorted_exceedances, shape, scale=scale)
        
        # Plot PP plot
        ax.scatter(p_theoretical, p_empirical)
        
        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'r--')
        
        # Set title and labels
        ax.set_title('PP Plot for GPD Fit')
        ax.set_xlabel('Theoretical CDF')
        ax.set_ylabel('Empirical CDF')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_simulated_returns(self, num_scenarios=100, horizon=1, figsize=(12, 6)):
        """
        Plot simulated returns.
        
        Parameters:
        -----------
        num_scenarios : int, default=100
            Number of scenarios to simulate
        horizon : int, default=1
            Simulation horizon
        figsize : tuple, default=(12, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Simulate returns
        simulated_returns = self.simulate_returns(num_scenarios, horizon)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot simulated returns
        for i in range(num_scenarios):
            ax.plot(
                range(horizon),
                simulated_returns.iloc[i],
                alpha=0.1,
                color='blue'
            )
        
        # Plot mean and percentiles
        mean_returns = simulated_returns.mean()
        percentile_5 = simulated_returns.quantile(0.05)
        percentile_95 = simulated_returns.quantile(0.95)
        
        ax.plot(
            range(horizon),
            mean_returns,
            color='red',
            linewidth=2,
            label='Mean'
        )
        
        ax.plot(
            range(horizon),
            percentile_5,
            color='green',
            linewidth=2,
            label='5th Percentile'
        )
        
        ax.plot(
            range(horizon),
            percentile_95,
            color='green',
            linewidth=2,
            label='95th Percentile'
        )
        
        # Fill between percentiles
        ax.fill_between(
            range(horizon),
            percentile_5,
            percentile_95,
            color='green',
            alpha=0.1
        )
        
        # Set title and labels
        ax.set_title('Simulated Returns')
        ax.set_xlabel('Time')
        ax.set_ylabel('Return')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig

    # Add estimate_risk method for compatibility with benchmark framework
    def estimate_risk(self, portfolio=None, confidence_level=0.95, horizon=1):
        """
        Estimate risk metrics for a portfolio.
        
        Parameters:
        -----------
        portfolio : Portfolio, optional
            Portfolio object. If None, uses equal weights.
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        horizon : int, default=1
            Forecast horizon
            
        Returns:
        --------
        dict
            Dictionary containing risk metrics
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Create dummy portfolio if not provided
        if portfolio is None:
            portfolio_weights = pd.Series(
                index=self.returns.columns,
                data=np.ones(self.returns.shape[1]) / self.returns.shape[1]
            )
            portfolio = type('Portfolio', (), {'weights': portfolio_weights})
        
        # Calculate risk metrics
        var = self.calculate_var(portfolio, confidence_level, horizon)
        es = self.calculate_expected_shortfall(portfolio, confidence_level, horizon)
        
        # Return risk metrics
        return {
            'var': var,
            'expected_shortfall': es,
            'confidence_level': confidence_level,
            'horizon': horizon
        }

# Alias for backward compatibility
EVTAnalysis = ExtremeValueTheory
