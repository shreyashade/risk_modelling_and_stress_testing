"""
Enhanced Extreme Value Theory (EVT) Implementation for Risk Modeling

This module implements advanced EVT methods for improved tail risk estimation,
including dynamic threshold selection, conditional EVT, and Bayesian approaches.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize
from typing import Dict, List, Union, Optional, Tuple, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.kde import KDEUnivariate
import logging
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EVTAnalysis:
    """Base class for Extreme Value Theory analysis"""
    
    def __init__(self):
        """Initialize the EVT analysis."""
        self.fitted = False
        self.params = {}
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the EVT model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Value-at-Risk using EVT.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for VaR
            
        Returns:
        --------
        float
            VaR estimate
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def calculate_expected_shortfall(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Expected Shortfall using EVT.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for ES
            
        Returns:
        --------
        float
            ES estimate
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def plot_tail_distribution(self) -> plt.Figure:
        """
        Plot the fitted tail distribution.
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure with tail distribution visualization
        """
        raise NotImplementedError("Subclasses must implement this method")


class GPDAnalysis(EVTAnalysis):
    """
    Generalized Pareto Distribution (GPD) analysis for EVT.
    
    This class implements the Peaks-Over-Threshold (POT) method using GPD
    for modeling the tail distribution of returns.
    """
    
    def __init__(self, 
                 threshold_method: str = 'quantile',
                 threshold_value: float = 0.05,
                 min_exceedances: int = 50):
        """
        Initialize the GPD analysis.
        
        Parameters:
        -----------
        threshold_method : str, default='quantile'
            Method for threshold selection ('quantile', 'hill', 'mean_excess')
        threshold_value : float, default=0.05
            Value for threshold selection (interpretation depends on method)
        min_exceedances : int, default=50
            Minimum number of exceedances required for fitting
        """
        super().__init__()
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.min_exceedances = min_exceedances
        self.threshold = None
        self.exceedances = None
        self.n_exceedances = 0
        self.n_total = 0
        self.shape = None  # ξ (xi) parameter
        self.scale = None  # β (beta) parameter
        self.location = None  # μ (mu) parameter
        self.tail_index = None  # α = 1/ξ
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the GPD model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info(f"Fitting GPD with {self.threshold_method} threshold method")
        
        # Prepare data - use portfolio returns if multiple assets
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
            data = -portfolio_returns  # Negative returns for loss distribution
        else:
            data = -returns.values.flatten()  # Negative returns for loss distribution
        
        # Determine threshold
        self.threshold = self._select_threshold(data)
        logger.info(f"Selected threshold: {self.threshold:.4f}")
        
        # Extract exceedances
        self.exceedances = data[data > self.threshold] - self.threshold
        self.n_exceedances = len(self.exceedances)
        self.n_total = len(data)
        
        if self.n_exceedances < self.min_exceedances:
            logger.warning(f"Insufficient exceedances: {self.n_exceedances} < {self.min_exceedances}")
            logger.warning("Using lower threshold to ensure sufficient exceedances")
            
            # Use quantile method with lower threshold
            self.threshold = np.percentile(data, 90)  # 90th percentile
            self.exceedances = data[data > self.threshold] - self.threshold
            self.n_exceedances = len(self.exceedances)
            
            if self.n_exceedances < self.min_exceedances:
                logger.error(f"Still insufficient exceedances: {self.n_exceedances} < {self.min_exceedances}")
                logger.error("EVT analysis may be unreliable")
        
        # Fit GPD parameters using MLE
        self._fit_gpd_mle()
        
        # Calculate tail index
        if self.shape != 0:
            self.tail_index = 1 / self.shape
        else:
            self.tail_index = float('inf')  # Exponential tail
        
        # Store parameters
        self.params = {
            'threshold': self.threshold,
            'shape': self.shape,
            'scale': self.scale,
            'location': self.location,
            'tail_index': self.tail_index,
            'n_exceedances': self.n_exceedances,
            'n_total': self.n_total,
            'exceedance_rate': self.n_exceedances / self.n_total
        }
        
        self.fitted = True
        logger.info(f"GPD fitted with shape={self.shape:.4f}, scale={self.scale:.4f}")
    
    def _select_threshold(self, data: np.ndarray) -> float:
        """
        Select threshold for GPD fitting.
        
        Parameters:
        -----------
        data : np.ndarray
            Loss data (negative returns)
            
        Returns:
        --------
        float
            Selected threshold
        """
        if self.threshold_method == 'quantile':
            # Use quantile method
            threshold = np.percentile(data, 100 * (1 - self.threshold_value))
            
        elif self.threshold_method == 'hill':
            # Use Hill plot to determine threshold
            sorted_data = np.sort(data)[::-1]  # Sort in descending order
            k_values = np.arange(10, min(500, len(sorted_data) // 2))
            hill_estimators = np.zeros(len(k_values))
            
            for i, k in enumerate(k_values):
                hill_estimators[i] = np.mean(np.log(sorted_data[:k])) - np.log(sorted_data[k])
            
            # Find stable region in Hill plot
            stability_index = self._find_stability(hill_estimators)
            k_optimal = k_values[stability_index]
            threshold = sorted_data[k_optimal]
            
        elif self.threshold_method == 'mean_excess':
            # Use mean excess plot to determine threshold
            sorted_data = np.sort(data)
            thresholds = sorted_data[sorted_data > 0]
            thresholds = thresholds[:-self.min_exceedances]  # Ensure sufficient exceedances
            
            mean_excess = np.zeros(len(thresholds))
            for i, u in enumerate(thresholds):
                exceedances = data[data > u] - u
                mean_excess[i] = np.mean(exceedances)
            
            # Find threshold where mean excess becomes linear
            linearity_index = self._find_linearity(thresholds, mean_excess)
            threshold = thresholds[linearity_index]
            
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")
        
        return threshold
    
    def _find_stability(self, estimators: np.ndarray) -> int:
        """
        Find region of stability in Hill plot.
        
        Parameters:
        -----------
        estimators : np.ndarray
            Hill estimators
            
        Returns:
        --------
        int
            Index of optimal k value
        """
        # Calculate rolling standard deviation
        window = min(20, len(estimators) // 4)
        rolling_std = np.array([np.std(estimators[i:i+window]) for i in range(len(estimators) - window)])
        
        # Find region with low standard deviation
        stability_index = np.argmin(rolling_std) + window // 2
        
        return min(stability_index, len(estimators) - 1)
    
    def _find_linearity(self, x: np.ndarray, y: np.ndarray) -> int:
        """
        Find region of linearity in mean excess plot.
        
        Parameters:
        -----------
        x : np.ndarray
            Threshold values
        y : np.ndarray
            Mean excess values
            
        Returns:
        --------
        int
            Index of optimal threshold
        """
        # Calculate local linear fit quality
        window = min(20, len(x) // 4)
        r_squared = np.zeros(len(x) - window)
        
        for i in range(len(x) - window):
            x_window = x[i:i+window]
            y_window = y[i:i+window]
            
            # Linear regression
            slope, intercept, r_value, _, _ = stats.linregress(x_window, y_window)
            r_squared[i] = r_value ** 2
        
        # Find region with good linear fit
        linearity_index = np.argmax(r_squared)
        
        return min(linearity_index, len(x) - 1)
    
    def _gpd_neg_log_likelihood(self, params: np.ndarray) -> float:
        """
        Negative log-likelihood function for GPD.
        
        Parameters:
        -----------
        params : np.ndarray
            GPD parameters [shape, scale]
            
        Returns:
        --------
        float
            Negative log-likelihood
        """
        shape, scale = params
        
        # Ensure scale is positive
        if scale <= 0:
            return 1e10
        
        # Calculate log-likelihood
        n = len(self.exceedances)
        
        if abs(shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
            log_likelihood = -n * np.log(scale) - np.sum(self.exceedances) / scale
        else:
            # Check if all points are valid for this shape parameter
            if shape < 0 and np.any(self.exceedances > -scale / shape):
                return 1e10
            
            log_likelihood = -n * np.log(scale) - (1 + 1/shape) * np.sum(
                np.log(1 + shape * self.exceedances / scale)
            )
        
        return -log_likelihood
    
    def _fit_gpd_mle(self) -> None:
        """Fit GPD parameters using Maximum Likelihood Estimation."""
        # Initial parameter guesses
        initial_shape = 0.1
        initial_scale = np.mean(self.exceedances)
        
        # Optimize negative log-likelihood
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                result = optimize.minimize(
                    self._gpd_neg_log_likelihood,
                    [initial_shape, initial_scale],
                    method='Nelder-Mead',
                    options={'maxiter': 1000}
                )
                
                if result.success:
                    self.shape, self.scale = result.x
                else:
                    logger.warning(f"GPD fitting did not converge: {result.message}")
                    # Use method of moments as fallback
                    self._fit_gpd_mom()
            except Exception as e:
                logger.error(f"Error in GPD MLE fitting: {e}")
                # Use method of moments as fallback
                self._fit_gpd_mom()
        
        # Set location parameter (threshold)
        self.location = self.threshold
    
    def _fit_gpd_mom(self) -> None:
        """Fit GPD parameters using Method of Moments."""
        # Calculate first and second moments
        m1 = np.mean(self.exceedances)
        m2 = np.mean(self.exceedances ** 2)
        
        # Estimate shape and scale
        self.shape = 0.5 * (1 - (m1 ** 2) / m2)
        self.scale = m1 * (1 - self.shape)
        
        # Ensure scale is positive
        if self.scale <= 0:
            logger.warning("Negative scale parameter, using exponential distribution")
            self.shape = 0
            self.scale = m1
    
    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Value-at-Risk using GPD.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for VaR
            
        Returns:
        --------
        float
            VaR estimate
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating VaR")
        
        # Calculate exceedance probability
        p = 1 - confidence_level
        
        # Calculate probability of exceeding threshold
        p_threshold = self.n_exceedances / self.n_total
        
        # Calculate VaR
        if abs(self.shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
            var = self.threshold + self.scale * np.log(p_threshold / p)
        else:
            var = self.threshold + (self.scale / self.shape) * (
                (p / p_threshold) ** (-self.shape) - 1
            )
        
        return var
    
    def calculate_expected_shortfall(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Expected Shortfall using GPD.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for ES
            
        Returns:
        --------
        float
            ES estimate
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating ES")
        
        # Calculate VaR
        var = self.calculate_var(confidence_level)
        
        # Calculate ES
        if abs(self.shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
            es = var + self.scale
        elif self.shape < 1:
            es = (var + self.scale - self.shape * (self.threshold - var)) / (1 - self.shape)
        else:
            # Shape >= 1 means infinite mean
            logger.warning("Shape parameter >= 1, ES is infinite")
            es = float('inf')
        
        return es
    
    def plot_tail_distribution(self) -> plt.Figure:
        """
        Plot the fitted tail distribution.
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure with tail distribution visualization
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Empirical vs. Fitted Tail Distribution
        sorted_exceedances = np.sort(self.exceedances)
        empirical_cdf = np.arange(1, len(sorted_exceedances) + 1) / (len(sorted_exceedances) + 1)
        
        # Generate fitted CDF
        x = np.linspace(0, max(sorted_exceedances) * 1.2, 1000)
        if abs(self.shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
            fitted_cdf = 1 - np.exp(-x / self.scale)
        else:
            fitted_cdf = 1 - (1 + self.shape * x / self.scale) ** (-1 / self.shape)
            # Ensure valid range for negative shape
            if self.shape < 0:
                fitted_cdf = fitted_cdf[x < -self.scale / self.shape]
                x = x[x < -self.scale / self.shape]
        
        axes[0, 0].plot(sorted_exceedances, empirical_cdf, 'o', markersize=3, label='Empirical')
        axes[0, 0].plot(x, fitted_cdf, 'r-', label=f'GPD (ξ={self.shape:.3f}, β={self.scale:.3f})')
        axes[0, 0].set_title('Tail Distribution (CDF)')
        axes[0, 0].set_xlabel('Exceedance')
        axes[0, 0].set_ylabel('Cumulative Probability')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: QQ Plot
        if abs(self.shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
            theoretical_quantiles = -self.scale * np.log(1 - empirical_cdf)
        else:
            theoretical_quantiles = (self.scale / self.shape) * (
                (1 - empirical_cdf) ** (-self.shape) - 1
            )
        
        axes[0, 1].scatter(theoretical_quantiles, sorted_exceedances)
        axes[0, 1].plot([0, max(theoretical_quantiles)], [0, max(theoretical_quantiles)], 'r--')
        axes[0, 1].set_title('QQ Plot')
        axes[0, 1].set_xlabel('Theoretical Quantiles')
        axes[0, 1].set_ylabel('Empirical Quantiles')
        axes[0, 1].grid(True)
        
        # Plot 3: Return Level Plot
        return_periods = np.logspace(0, 3, 100)
        p = 1 / return_periods
        
        if abs(self.shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
            return_levels = self.threshold + self.scale * np.log(self.n_total / (self.n_exceedances * p))
        else:
            return_levels = self.threshold + (self.scale / self.shape) * (
                (self.n_total / (self.n_exceedances * p)) ** self.shape - 1
            )
        
        axes[1, 0].semilogx(return_periods, return_levels, 'b-')
        axes[1, 0].set_title('Return Level Plot')
        axes[1, 0].set_xlabel('Return Period')
        axes[1, 0].set_ylabel('Return Level')
        axes[1, 0].grid(True)
        
        # Plot 4: Tail Density
        if abs(self.shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
            fitted_pdf = (1 / self.scale) * np.exp(-x / self.scale)
        else:
            fitted_pdf = (1 / self.scale) * (1 + self.shape * x / self.scale) ** (-1 / self.shape - 1)
            # Ensure valid range for negative shape
            if self.shape < 0:
                fitted_pdf = fitted_pdf[x < -self.scale / self.shape]
                x = x[x < -self.scale / self.shape]
        
        # Kernel density estimation for empirical density
        kde = KDEUnivariate(self.exceedances)
        kde.fit()
        
        axes[1, 1].plot(x, fitted_pdf, 'r-', label='GPD Density')
        axes[1, 1].plot(kde.support, kde.density, 'b-', label='Empirical Density')
        axes[1, 1].set_title('Tail Density')
        axes[1, 1].set_xlabel('Exceedance')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig


class DynamicEVT(EVTAnalysis):
    """
    Dynamic Extreme Value Theory analysis.
    
    This class implements a dynamic approach to EVT that adapts to changing
    market conditions by using a rolling window and dynamic threshold selection.
    """
    
    def __init__(self, 
                 window_size: int = 252,
                 threshold_quantile: float = 0.05,
                 min_exceedances: int = 30,
                 overlap: float = 0.5):
        """
        Initialize the dynamic EVT analysis.
        
        Parameters:
        -----------
        window_size : int, default=252
            Rolling window size (trading days)
        threshold_quantile : float, default=0.05
            Quantile for threshold selection
        min_exceedances : int, default=30
            Minimum number of exceedances required for fitting
        overlap : float, default=0.5
            Overlap between consecutive windows (0 to 1)
        """
        super().__init__()
        self.window_size = window_size
        self.threshold_quantile = threshold_quantile
        self.min_exceedances = min_exceedances
        self.overlap = overlap
        self.windows = []
        self.window_params = []
        self.current_params = None
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the dynamic EVT model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info(f"Fitting dynamic EVT with window size {self.window_size}")
        
        # Prepare data - use portfolio returns if multiple assets
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
            data = pd.Series(-portfolio_returns, index=returns.index)  # Negative returns for loss distribution
        else:
            data = -returns.iloc[:, 0]  # Negative returns for loss distribution
        
        # Calculate step size for rolling windows
        step_size = int(self.window_size * (1 - self.overlap))
        step_size = max(step_size, 1)  # Ensure at least 1 day step
        
        # Create rolling windows
        self.windows = []
        self.window_params = []
        
        for start_idx in range(0, len(data) - self.window_size + 1, step_size):
            window_data = data.iloc[start_idx:start_idx + self.window_size]
            window_date = window_data.index[-1]
            
            # Fit GPD to window
            gpd = GPDAnalysis(
                threshold_method='quantile',
                threshold_value=self.threshold_quantile,
                min_exceedances=self.min_exceedances
            )
            
            try:
                gpd.fit(pd.DataFrame(window_data))
                
                # Store window parameters
                self.windows.append(window_date)
                self.window_params.append(gpd.params)
                
                logger.debug(f"Window {len(self.windows)}: {window_date}, shape={gpd.shape:.4f}, scale={gpd.scale:.4f}")
            except Exception as e:
                logger.warning(f"Failed to fit window ending {window_date}: {e}")
        
        if not self.window_params:
            raise ValueError("Failed to fit any windows")
        
        # Set current parameters to most recent window
        self.current_params = self.window_params[-1]
        
        # Store parameters
        self.params = {
            'windows': self.windows,
            'window_params': self.window_params,
            'current_params': self.current_params
        }
        
        self.fitted = True
        logger.info(f"Dynamic EVT fitted with {len(self.windows)} windows")
    
    def calculate_var(self, confidence_level: float = 0.95, window_idx: int = -1) -> float:
        """
        Calculate Value-at-Risk using dynamic EVT.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for VaR
        window_idx : int, default=-1
            Window index to use (-1 for most recent)
            
        Returns:
        --------
        float
            VaR estimate
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating VaR")
        
        # Get parameters for specified window
        if window_idx < 0 or window_idx >= len(self.window_params):
            params = self.current_params
        else:
            params = self.window_params[window_idx]
        
        # Extract GPD parameters
        threshold = params['threshold']
        shape = params['shape']
        scale = params['scale']
        p_threshold = params['exceedance_rate']
        
        # Calculate exceedance probability
        p = 1 - confidence_level
        
        # Calculate VaR
        if abs(shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
            var = threshold + scale * np.log(p_threshold / p)
        else:
            var = threshold + (scale / shape) * (
                (p / p_threshold) ** (-shape) - 1
            )
        
        return var
    
    def calculate_expected_shortfall(self, confidence_level: float = 0.95, window_idx: int = -1) -> float:
        """
        Calculate Expected Shortfall using dynamic EVT.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for ES
        window_idx : int, default=-1
            Window index to use (-1 for most recent)
            
        Returns:
        --------
        float
            ES estimate
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating ES")
        
        # Get parameters for specified window
        if window_idx < 0 or window_idx >= len(self.window_params):
            params = self.current_params
        else:
            params = self.window_params[window_idx]
        
        # Extract GPD parameters
        threshold = params['threshold']
        shape = params['shape']
        scale = params['scale']
        
        # Calculate VaR
        var = self.calculate_var(confidence_level, window_idx)
        
        # Calculate ES
        if abs(shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
            es = var + scale
        elif shape < 1:
            es = (var + scale - shape * (threshold - var)) / (1 - shape)
        else:
            # Shape >= 1 means infinite mean
            logger.warning("Shape parameter >= 1, ES is infinite")
            es = float('inf')
        
        return es
    
    def plot_parameter_evolution(self) -> plt.Figure:
        """
        Plot the evolution of GPD parameters over time.
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure with parameter evolution visualization
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Extract parameters
        dates = self.windows
        shapes = [params['shape'] for params in self.window_params]
        scales = [params['scale'] for params in self.window_params]
        thresholds = [params['threshold'] for params in self.window_params]
        tail_indices = [params['tail_index'] for params in self.window_params]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        
        # Plot shape parameter
        axes[0, 0].plot(dates, shapes, 'b-')
        axes[0, 0].set_title('Shape Parameter (ξ)')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True)
        
        # Plot scale parameter
        axes[0, 1].plot(dates, scales, 'r-')
        axes[0, 1].set_title('Scale Parameter (β)')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True)
        
        # Plot threshold
        axes[1, 0].plot(dates, thresholds, 'g-')
        axes[1, 0].set_title('Threshold (u)')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].grid(True)
        
        # Plot tail index
        axes[1, 1].plot(dates, tail_indices, 'm-')
        axes[1, 1].set_title('Tail Index (α = 1/ξ)')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_risk_metrics(self, confidence_levels: List[float] = [0.95, 0.99]) -> plt.Figure:
        """
        Plot the evolution of risk metrics over time.
        
        Parameters:
        -----------
        confidence_levels : List[float], default=[0.95, 0.99]
            Confidence levels for risk metrics
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with risk metrics visualization
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Create figure
        fig, axes = plt.subplots(len(confidence_levels), 2, figsize=(14, 5 * len(confidence_levels)), sharex=True)
        
        # Ensure axes is 2D even with single confidence level
        if len(confidence_levels) == 1:
            axes = np.array([axes])
        
        # Calculate risk metrics for each window and confidence level
        dates = self.windows
        
        for i, cl in enumerate(confidence_levels):
            var_values = [self.calculate_var(cl, j) for j in range(len(self.windows))]
            es_values = [self.calculate_expected_shortfall(cl, j) for j in range(len(self.windows))]
            
            # Plot VaR
            axes[i, 0].plot(dates, var_values, 'b-')
            axes[i, 0].set_title(f'Value-at-Risk ({cl*100:.0f}%)')
            axes[i, 0].set_ylabel('Value')
            axes[i, 0].grid(True)
            
            # Plot ES
            axes[i, 1].plot(dates, es_values, 'r-')
            axes[i, 1].set_title(f'Expected Shortfall ({cl*100:.0f}%)')
            axes[i, 1].set_ylabel('Value')
            axes[i, 1].grid(True)
        
        # Add date labels to bottom row
        for ax in axes[-1, :]:
            ax.set_xlabel('Date')
        
        plt.tight_layout()
        return fig


class ConditionalEVT(EVTAnalysis):
    """
    Conditional Extreme Value Theory analysis.
    
    This class implements a conditional approach to EVT that incorporates
    market factors to adjust tail risk estimates based on current conditions.
    """
    
    def __init__(self, 
                 threshold_quantile: float = 0.05,
                 min_exceedances: int = 50,
                 condition_factors: Optional[List[str]] = None):
        """
        Initialize the conditional EVT analysis.
        
        Parameters:
        -----------
        threshold_quantile : float, default=0.05
            Quantile for threshold selection
        min_exceedances : int, default=50
            Minimum number of exceedances required for fitting
        condition_factors : List[str], optional
            List of factor names to condition on (e.g., ['volatility', 'trend'])
        """
        super().__init__()
        self.threshold_quantile = threshold_quantile
        self.min_exceedances = min_exceedances
        self.condition_factors = condition_factors or ['volatility']
        self.threshold = None
        self.exceedances = None
        self.factor_values = None
        self.factor_betas = None
        self.base_shape = None
        self.base_scale = None
    
    def _calculate_factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate conditioning factors.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with factor values
        """
        factors = pd.DataFrame(index=returns.index)
        
        # Calculate portfolio returns if multiple assets
        if returns.shape[1] > 1:
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = pd.Series(returns.values @ weights, index=returns.index)
        else:
            portfolio_returns = returns.iloc[:, 0]
        
        # Calculate factors
        for factor in self.condition_factors:
            if factor == 'volatility':
                # Rolling volatility (21-day)
                factors['volatility'] = portfolio_returns.rolling(window=21).std() * np.sqrt(252)
                # Fill NaN values with unconditional volatility
                factors['volatility'] = factors['volatility'].fillna(portfolio_returns.std() * np.sqrt(252))
                
            elif factor == 'trend':
                # Rolling mean (63-day)
                factors['trend'] = portfolio_returns.rolling(window=63).mean() * 252
                # Fill NaN values with unconditional mean
                factors['trend'] = factors['trend'].fillna(portfolio_returns.mean() * 252)
                
            elif factor == 'skewness':
                # Rolling skewness (63-day)
                factors['skewness'] = portfolio_returns.rolling(window=63).skew()
                # Fill NaN values with unconditional skewness
                factors['skewness'] = factors['skewness'].fillna(stats.skew(portfolio_returns))
                
            elif factor == 'kurtosis':
                # Rolling excess kurtosis (63-day)
                factors['kurtosis'] = portfolio_returns.rolling(window=63).kurt()
                # Fill NaN values with unconditional kurtosis
                factors['kurtosis'] = factors['kurtosis'].fillna(stats.kurtosis(portfolio_returns))
                
            else:
                raise ValueError(f"Unknown factor: {factor}")
        
        # Standardize factors
        for col in factors.columns:
            factors[col] = (factors[col] - factors[col].mean()) / factors[col].std()
        
        return factors
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the conditional EVT model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info(f"Fitting conditional EVT with factors: {self.condition_factors}")
        
        # Prepare data - use portfolio returns if multiple assets
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
            data = -portfolio_returns  # Negative returns for loss distribution
        else:
            data = -returns.values.flatten()  # Negative returns for loss distribution
        
        # Calculate conditioning factors
        self.factor_values = self._calculate_factors(returns)
        
        # Determine threshold
        self.threshold = np.percentile(data, 100 * (1 - self.threshold_quantile))
        logger.info(f"Selected threshold: {self.threshold:.4f}")
        
        # Extract exceedances
        exceedance_indices = data > self.threshold
        self.exceedances = data[exceedance_indices] - self.threshold
        factor_values_exceedances = self.factor_values.iloc[exceedance_indices]
        
        n_exceedances = len(self.exceedances)
        n_total = len(data)
        
        if n_exceedances < self.min_exceedances:
            logger.warning(f"Insufficient exceedances: {n_exceedances} < {self.min_exceedances}")
            logger.warning("Using lower threshold to ensure sufficient exceedances")
            
            # Use lower threshold
            self.threshold = np.percentile(data, 90)  # 90th percentile
            exceedance_indices = data > self.threshold
            self.exceedances = data[exceedance_indices] - self.threshold
            factor_values_exceedances = self.factor_values.iloc[exceedance_indices]
            
            n_exceedances = len(self.exceedances)
            
            if n_exceedances < self.min_exceedances:
                logger.error(f"Still insufficient exceedances: {n_exceedances} < {self.min_exceedances}")
                logger.error("Conditional EVT analysis may be unreliable")
        
        # Fit base GPD parameters using MLE
        gpd = GPDAnalysis(threshold_method='quantile', threshold_value=0)
        gpd.threshold = 0  # Exceedances are already threshold-shifted
        gpd.exceedances = self.exceedances
        gpd.n_exceedances = n_exceedances
        gpd.n_total = n_total
        gpd._fit_gpd_mle()
        
        self.base_shape = gpd.shape
        self.base_scale = gpd.scale
        
        # Fit factor betas using regression
        self._fit_factor_betas(factor_values_exceedances)
        
        # Store parameters
        self.params = {
            'threshold': self.threshold,
            'base_shape': self.base_shape,
            'base_scale': self.base_scale,
            'factor_betas': self.factor_betas,
            'n_exceedances': n_exceedances,
            'n_total': n_total,
            'exceedance_rate': n_exceedances / n_total
        }
        
        self.fitted = True
        logger.info(f"Conditional EVT fitted with base shape={self.base_shape:.4f}, scale={self.base_scale:.4f}")
    
    def _fit_factor_betas(self, factor_values: pd.DataFrame) -> None:
        """
        Fit factor betas using regression.
        
        Parameters:
        -----------
        factor_values : pd.DataFrame
            Factor values for exceedances
        """
        # Initialize factor betas
        self.factor_betas = {
            'shape': {factor: 0.0 for factor in self.condition_factors},
            'scale': {factor: 0.0 for factor in self.condition_factors}
        }
        
        # Fit scale betas using regression
        # log(exceedance) = log(scale) + factor_effects + noise
        log_exceedances = np.log(self.exceedances + 1e-10)  # Add small constant to avoid log(0)
        
        try:
            # Prepare design matrix
            X = factor_values.values
            y = log_exceedances
            
            # Add constant term
            X = np.column_stack([np.ones(len(X)), X])
            
            # Fit regression
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            
            # Extract betas
            for i, factor in enumerate(factor_values.columns):
                self.factor_betas['scale'][factor] = beta[i+1]
            
            logger.info(f"Scale factor betas: {self.factor_betas['scale']}")
        except Exception as e:
            logger.warning(f"Failed to fit scale factor betas: {e}")
        
        # Fit shape betas using more complex approach
        # This is a simplified approach; a full implementation would use MLE
        # with factor-dependent shape parameter
        try:
            # Divide exceedances into factor quantiles
            for factor in self.condition_factors:
                factor_quantiles = pd.qcut(factor_values[factor], 3, labels=False)
                
                shapes = []
                for q in range(3):
                    quantile_exceedances = self.exceedances[factor_quantiles == q]
                    if len(quantile_exceedances) >= 20:
                        # Fit GPD to quantile
                        gpd = GPDAnalysis(threshold_method='quantile', threshold_value=0)
                        gpd.threshold = 0
                        gpd.exceedances = quantile_exceedances
                        gpd.n_exceedances = len(quantile_exceedances)
                        gpd.n_total = len(self.exceedances)
                        gpd._fit_gpd_mle()
                        shapes.append(gpd.shape)
                
                if len(shapes) >= 2:
                    # Estimate shape beta from quantile differences
                    high_shape = shapes[-1]
                    low_shape = shapes[0]
                    self.factor_betas['shape'][factor] = (high_shape - low_shape) / 2
            
            logger.info(f"Shape factor betas: {self.factor_betas['shape']}")
        except Exception as e:
            logger.warning(f"Failed to fit shape factor betas: {e}")
    
    def _get_conditional_parameters(self, factor_values: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
        """
        Get conditional GPD parameters based on factor values.
        
        Parameters:
        -----------
        factor_values : Dict[str, float], optional
            Dictionary of factor values
            
        Returns:
        --------
        Tuple[float, float]
            Conditional shape and scale parameters
        """
        if factor_values is None:
            # Use most recent factor values
            factor_values = {
                factor: self.factor_values[factor].iloc[-1]
                for factor in self.condition_factors
            }
        
        # Calculate conditional shape
        cond_shape = self.base_shape
        for factor, value in factor_values.items():
            if factor in self.factor_betas['shape']:
                cond_shape += self.factor_betas['shape'][factor] * value
        
        # Calculate conditional scale
        cond_scale = self.base_scale
        for factor, value in factor_values.items():
            if factor in self.factor_betas['scale']:
                # Exponential effect on scale
                cond_scale *= np.exp(self.factor_betas['scale'][factor] * value)
        
        return cond_shape, cond_scale
    
    def calculate_var(self, 
                      confidence_level: float = 0.95, 
                      factor_values: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate conditional Value-at-Risk.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for VaR
        factor_values : Dict[str, float], optional
            Dictionary of factor values
            
        Returns:
        --------
        float
            Conditional VaR estimate
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating VaR")
        
        # Get conditional parameters
        cond_shape, cond_scale = self._get_conditional_parameters(factor_values)
        
        # Calculate exceedance probability
        p = 1 - confidence_level
        
        # Calculate probability of exceeding threshold
        p_threshold = self.params['exceedance_rate']
        
        # Calculate VaR
        if abs(cond_shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
            var = self.threshold + cond_scale * np.log(p_threshold / p)
        else:
            var = self.threshold + (cond_scale / cond_shape) * (
                (p / p_threshold) ** (-cond_shape) - 1
            )
        
        return var
    
    def calculate_expected_shortfall(self, 
                                     confidence_level: float = 0.95, 
                                     factor_values: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate conditional Expected Shortfall.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for ES
        factor_values : Dict[str, float], optional
            Dictionary of factor values
            
        Returns:
        --------
        float
            Conditional ES estimate
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating ES")
        
        # Get conditional parameters
        cond_shape, cond_scale = self._get_conditional_parameters(factor_values)
        
        # Calculate VaR
        var = self.calculate_var(confidence_level, factor_values)
        
        # Calculate ES
        if abs(cond_shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
            es = var + cond_scale
        elif cond_shape < 1:
            es = (var + cond_scale - cond_shape * (self.threshold - var)) / (1 - cond_shape)
        else:
            # Shape >= 1 means infinite mean
            logger.warning("Conditional shape parameter >= 1, ES is infinite")
            es = float('inf')
        
        return es
    
    def plot_conditional_effects(self) -> plt.Figure:
        """
        Plot the effects of conditioning factors on risk metrics.
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure with conditional effects visualization
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Create figure
        n_factors = len(self.condition_factors)
        fig, axes = plt.subplots(n_factors, 2, figsize=(14, 5 * n_factors))
        
        # Ensure axes is 2D even with single factor
        if n_factors == 1:
            axes = np.array([axes])
        
        # Calculate risk metrics across factor values
        for i, factor in enumerate(self.condition_factors):
            # Generate range of factor values
            factor_range = np.linspace(-2, 2, 100)  # -2 to 2 standard deviations
            
            var_values = []
            es_values = []
            
            for val in factor_range:
                # Set factor values (only vary one factor at a time)
                factor_values = {f: 0.0 for f in self.condition_factors}
                factor_values[factor] = val
                
                # Calculate risk metrics
                var = self.calculate_var(0.95, factor_values)
                es = self.calculate_expected_shortfall(0.95, factor_values)
                
                var_values.append(var)
                es_values.append(es)
            
            # Plot VaR
            axes[i, 0].plot(factor_range, var_values, 'b-')
            axes[i, 0].set_title(f'VaR (95%) vs {factor}')
            axes[i, 0].set_xlabel(f'{factor} (standardized)')
            axes[i, 0].set_ylabel('VaR')
            axes[i, 0].grid(True)
            
            # Plot ES
            axes[i, 1].plot(factor_range, es_values, 'r-')
            axes[i, 1].set_title(f'ES (95%) vs {factor}')
            axes[i, 1].set_xlabel(f'{factor} (standardized)')
            axes[i, 1].set_ylabel('ES')
            axes[i, 1].grid(True)
        
        plt.tight_layout()
        return fig


class BayesianEVT(EVTAnalysis):
    """
    Bayesian Extreme Value Theory analysis.
    
    This class implements a Bayesian approach to EVT that quantifies
    parameter uncertainty and provides credible intervals for risk metrics.
    """
    
    def __init__(self, 
                 threshold_quantile: float = 0.05,
                 min_exceedances: int = 50,
                 n_samples: int = 1000,
                 prior_shape_mean: float = 0.0,
                 prior_shape_std: float = 0.5,
                 prior_scale_alpha: float = 2.0,
                 prior_scale_beta: float = 1.0):
        """
        Initialize the Bayesian EVT analysis.
        
        Parameters:
        -----------
        threshold_quantile : float, default=0.05
            Quantile for threshold selection
        min_exceedances : int, default=50
            Minimum number of exceedances required for fitting
        n_samples : int, default=1000
            Number of posterior samples
        prior_shape_mean : float, default=0.0
            Mean of prior distribution for shape parameter
        prior_shape_std : float, default=0.5
            Standard deviation of prior distribution for shape parameter
        prior_scale_alpha : float, default=2.0
            Alpha parameter of prior distribution for scale parameter
        prior_scale_beta : float, default=1.0
            Beta parameter of prior distribution for scale parameter
        """
        super().__init__()
        self.threshold_quantile = threshold_quantile
        self.min_exceedances = min_exceedances
        self.n_samples = n_samples
        self.prior_shape_mean = prior_shape_mean
        self.prior_shape_std = prior_shape_std
        self.prior_scale_alpha = prior_scale_alpha
        self.prior_scale_beta = prior_scale_beta
        self.threshold = None
        self.exceedances = None
        self.shape_samples = None
        self.scale_samples = None
    
    def _log_posterior(self, params: np.ndarray) -> float:
        """
        Calculate log posterior density.
        
        Parameters:
        -----------
        params : np.ndarray
            GPD parameters [shape, scale]
            
        Returns:
        --------
        float
            Log posterior density
        """
        shape, scale = params
        
        # Prior for shape (normal)
        log_prior_shape = -0.5 * ((shape - self.prior_shape_mean) / self.prior_shape_std) ** 2
        
        # Prior for scale (gamma)
        log_prior_scale = (self.prior_scale_alpha - 1) * np.log(scale) - scale / self.prior_scale_beta
        
        # Likelihood
        n = len(self.exceedances)
        
        if scale <= 0:
            return -np.inf
        
        if abs(shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
            log_likelihood = -n * np.log(scale) - np.sum(self.exceedances) / scale
        else:
            # Check if all points are valid for this shape parameter
            if shape < 0 and np.any(self.exceedances > -scale / shape):
                return -np.inf
            
            log_likelihood = -n * np.log(scale) - (1 + 1/shape) * np.sum(
                np.log(1 + shape * self.exceedances / scale)
            )
        
        return log_prior_shape + log_prior_scale + log_likelihood
    
    def _metropolis_hastings(self, 
                             initial_params: np.ndarray, 
                             proposal_std: np.ndarray,
                             n_samples: int,
                             burn_in: int = 100,
                             thinning: int = 5) -> np.ndarray:
        """
        Run Metropolis-Hastings algorithm for posterior sampling.
        
        Parameters:
        -----------
        initial_params : np.ndarray
            Initial parameter values [shape, scale]
        proposal_std : np.ndarray
            Standard deviations for proposal distribution [shape_std, scale_std]
        n_samples : int
            Number of posterior samples
        burn_in : int, default=100
            Number of burn-in iterations
        thinning : int, default=5
            Thinning factor
            
        Returns:
        --------
        np.ndarray
            Posterior samples with shape (n_samples, 2)
        """
        # Initialize
        current_params = initial_params.copy()
        current_log_posterior = self._log_posterior(current_params)
        
        samples = []
        n_total = burn_in + n_samples * thinning
        
        # Run MCMC
        for i in range(n_total):
            # Propose new parameters
            proposal = current_params + np.random.normal(0, proposal_std, size=2)
            
            # Calculate acceptance probability
            proposal_log_posterior = self._log_posterior(proposal)
            log_acceptance = proposal_log_posterior - current_log_posterior
            
            # Accept or reject
            if np.log(np.random.uniform()) < log_acceptance:
                current_params = proposal
                current_log_posterior = proposal_log_posterior
            
            # Store sample after burn-in and thinning
            if i >= burn_in and (i - burn_in) % thinning == 0:
                samples.append(current_params.copy())
        
        return np.array(samples)
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the Bayesian EVT model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info("Fitting Bayesian EVT")
        
        # Prepare data - use portfolio returns if multiple assets
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
            data = -portfolio_returns  # Negative returns for loss distribution
        else:
            data = -returns.values.flatten()  # Negative returns for loss distribution
        
        # Determine threshold
        self.threshold = np.percentile(data, 100 * (1 - self.threshold_quantile))
        logger.info(f"Selected threshold: {self.threshold:.4f}")
        
        # Extract exceedances
        self.exceedances = data[data > self.threshold] - self.threshold
        n_exceedances = len(self.exceedances)
        n_total = len(data)
        
        if n_exceedances < self.min_exceedances:
            logger.warning(f"Insufficient exceedances: {n_exceedances} < {self.min_exceedances}")
            logger.warning("Using lower threshold to ensure sufficient exceedances")
            
            # Use lower threshold
            self.threshold = np.percentile(data, 90)  # 90th percentile
            self.exceedances = data[data > self.threshold] - self.threshold
            n_exceedances = len(self.exceedances)
            
            if n_exceedances < self.min_exceedances:
                logger.error(f"Still insufficient exceedances: {n_exceedances} < {self.min_exceedances}")
                logger.error("Bayesian EVT analysis may be unreliable")
        
        # Get initial parameter estimates using MLE
        gpd = GPDAnalysis(threshold_method='quantile', threshold_value=0)
        gpd.threshold = 0  # Exceedances are already threshold-shifted
        gpd.exceedances = self.exceedances
        gpd.n_exceedances = n_exceedances
        gpd.n_total = n_total
        gpd._fit_gpd_mle()
        
        initial_params = np.array([gpd.shape, gpd.scale])
        
        # Set proposal standard deviations
        proposal_std = np.array([0.05, 0.1 * gpd.scale])
        
        # Run MCMC
        try:
            samples = self._metropolis_hastings(
                initial_params=initial_params,
                proposal_std=proposal_std,
                n_samples=self.n_samples
            )
            
            self.shape_samples = samples[:, 0]
            self.scale_samples = samples[:, 1]
            
            # Calculate posterior means
            shape_mean = np.mean(self.shape_samples)
            scale_mean = np.mean(self.scale_samples)
            
            logger.info(f"Bayesian EVT fitted with mean shape={shape_mean:.4f}, mean scale={scale_mean:.4f}")
        except Exception as e:
            logger.error(f"MCMC sampling failed: {e}")
            logger.info("Using MLE estimates as fallback")
            
            # Use MLE estimates as fallback
            self.shape_samples = np.ones(self.n_samples) * gpd.shape
            self.scale_samples = np.ones(self.n_samples) * gpd.scale
        
        # Store parameters
        self.params = {
            'threshold': self.threshold,
            'shape_mean': np.mean(self.shape_samples),
            'shape_std': np.std(self.shape_samples),
            'scale_mean': np.mean(self.scale_samples),
            'scale_std': np.std(self.scale_samples),
            'n_exceedances': n_exceedances,
            'n_total': n_total,
            'exceedance_rate': n_exceedances / n_total
        }
        
        self.fitted = True
    
    def calculate_var(self, confidence_level: float = 0.95, credible_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate Value-at-Risk with credible interval.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for VaR
        credible_level : float, default=0.95
            Credible level for interval
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with VaR estimate and credible interval
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating VaR")
        
        # Calculate exceedance probability
        p = 1 - confidence_level
        
        # Calculate probability of exceeding threshold
        p_threshold = self.params['exceedance_rate']
        
        # Calculate VaR for each posterior sample
        var_samples = np.zeros(self.n_samples)
        
        for i in range(self.n_samples):
            shape = self.shape_samples[i]
            scale = self.scale_samples[i]
            
            if abs(shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
                var_samples[i] = self.threshold + scale * np.log(p_threshold / p)
            else:
                var_samples[i] = self.threshold + (scale / shape) * (
                    (p / p_threshold) ** (-shape) - 1
                )
        
        # Calculate credible interval
        alpha = 1 - credible_level
        lower_bound = np.percentile(var_samples, 100 * alpha / 2)
        upper_bound = np.percentile(var_samples, 100 * (1 - alpha / 2))
        
        return {
            'var': np.mean(var_samples),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    def calculate_expected_shortfall(self, confidence_level: float = 0.95, credible_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate Expected Shortfall with credible interval.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for ES
        credible_level : float, default=0.95
            Credible level for interval
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with ES estimate and credible interval
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating ES")
        
        # Calculate VaR for each posterior sample
        var_samples = np.zeros(self.n_samples)
        es_samples = np.zeros(self.n_samples)
        
        # Calculate exceedance probability
        p = 1 - confidence_level
        
        # Calculate probability of exceeding threshold
        p_threshold = self.params['exceedance_rate']
        
        for i in range(self.n_samples):
            shape = self.shape_samples[i]
            scale = self.scale_samples[i]
            
            # Calculate VaR
            if abs(shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
                var_samples[i] = self.threshold + scale * np.log(p_threshold / p)
            else:
                var_samples[i] = self.threshold + (scale / shape) * (
                    (p / p_threshold) ** (-shape) - 1
                )
            
            # Calculate ES
            if abs(shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
                es_samples[i] = var_samples[i] + scale
            elif shape < 1:
                es_samples[i] = (var_samples[i] + scale - shape * (self.threshold - var_samples[i])) / (1 - shape)
            else:
                # Shape >= 1 means infinite mean
                es_samples[i] = float('inf')
        
        # Remove infinite values
        es_samples = es_samples[np.isfinite(es_samples)]
        
        if len(es_samples) == 0:
            logger.warning("All ES samples are infinite")
            return {
                'es': float('inf'),
                'lower_bound': float('inf'),
                'upper_bound': float('inf')
            }
        
        # Calculate credible interval
        alpha = 1 - credible_level
        lower_bound = np.percentile(es_samples, 100 * alpha / 2)
        upper_bound = np.percentile(es_samples, 100 * (1 - alpha / 2))
        
        return {
            'es': np.mean(es_samples),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    def plot_posterior_distributions(self) -> plt.Figure:
        """
        Plot posterior distributions of GPD parameters.
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure with posterior distributions
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot shape posterior
        sns.histplot(self.shape_samples, kde=True, ax=axes[0, 0])
        axes[0, 0].axvline(np.mean(self.shape_samples), color='r', linestyle='--',
                          label=f'Mean: {np.mean(self.shape_samples):.4f}')
        axes[0, 0].set_title('Posterior Distribution of Shape Parameter (ξ)')
        axes[0, 0].set_xlabel('Shape')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        
        # Plot scale posterior
        sns.histplot(self.scale_samples, kde=True, ax=axes[0, 1])
        axes[0, 1].axvline(np.mean(self.scale_samples), color='r', linestyle='--',
                          label=f'Mean: {np.mean(self.scale_samples):.4f}')
        axes[0, 1].set_title('Posterior Distribution of Scale Parameter (β)')
        axes[0, 1].set_xlabel('Scale')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        
        # Plot joint posterior
        axes[1, 0].scatter(self.shape_samples, self.scale_samples, alpha=0.5)
        axes[1, 0].set_title('Joint Posterior Distribution')
        axes[1, 0].set_xlabel('Shape')
        axes[1, 0].set_ylabel('Scale')
        
        # Plot tail index posterior
        tail_index_samples = 1 / self.shape_samples[self.shape_samples > 0]
        if len(tail_index_samples) > 0:
            sns.histplot(tail_index_samples, kde=True, ax=axes[1, 1])
            axes[1, 1].axvline(np.mean(tail_index_samples), color='r', linestyle='--',
                              label=f'Mean: {np.mean(tail_index_samples):.4f}')
            axes[1, 1].set_title('Posterior Distribution of Tail Index (α = 1/ξ)')
            axes[1, 1].set_xlabel('Tail Index')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].legend()
        else:
            axes[1, 1].set_title('No positive shape parameters for tail index')
        
        plt.tight_layout()
        return fig
    
    def plot_risk_metrics_distributions(self, confidence_level: float = 0.95) -> plt.Figure:
        """
        Plot posterior distributions of risk metrics.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for risk metrics
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with risk metrics distributions
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Calculate exceedance probability
        p = 1 - confidence_level
        
        # Calculate probability of exceeding threshold
        p_threshold = self.params['exceedance_rate']
        
        # Calculate VaR and ES for each posterior sample
        var_samples = np.zeros(self.n_samples)
        es_samples = np.zeros(self.n_samples)
        
        for i in range(self.n_samples):
            shape = self.shape_samples[i]
            scale = self.scale_samples[i]
            
            # Calculate VaR
            if abs(shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
                var_samples[i] = self.threshold + scale * np.log(p_threshold / p)
            else:
                var_samples[i] = self.threshold + (scale / shape) * (
                    (p / p_threshold) ** (-shape) - 1
                )
            
            # Calculate ES
            if abs(shape) < 1e-6:  # Shape ≈ 0 (exponential distribution)
                es_samples[i] = var_samples[i] + scale
            elif shape < 1:
                es_samples[i] = (var_samples[i] + scale - shape * (self.threshold - var_samples[i])) / (1 - shape)
            else:
                # Shape >= 1 means infinite mean
                es_samples[i] = float('inf')
        
        # Remove infinite values
        es_samples = es_samples[np.isfinite(es_samples)]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot VaR posterior
        sns.histplot(var_samples, kde=True, ax=axes[0])
        axes[0].axvline(np.mean(var_samples), color='r', linestyle='--',
                       label=f'Mean: {np.mean(var_samples):.4f}')
        axes[0].set_title(f'Posterior Distribution of VaR ({confidence_level*100:.0f}%)')
        axes[0].set_xlabel('VaR')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        
        # Plot ES posterior
        if len(es_samples) > 0:
            sns.histplot(es_samples, kde=True, ax=axes[1])
            axes[1].axvline(np.mean(es_samples), color='r', linestyle='--',
                           label=f'Mean: {np.mean(es_samples):.4f}')
            axes[1].set_title(f'Posterior Distribution of ES ({confidence_level*100:.0f}%)')
            axes[1].set_xlabel('ES')
            axes[1].set_ylabel('Density')
            axes[1].legend()
        else:
            axes[1].set_title('All ES samples are infinite')
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Sample data
    import yfinance as yf
    
    # Download data
    tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'AGG']
    data = yf.download(tickers, start='2018-01-01', end='2023-12-31')
    
    # Calculate returns
    prices = data['Adj Close']
    returns = prices.pct_change().dropna()
    
    # Initialize and fit GPD
    gpd = GPDAnalysis()
    gpd.fit(returns)
    
    # Calculate risk metrics
    var_95 = gpd.calculate_var(0.95)
    es_95 = gpd.calculate_expected_shortfall(0.95)
    
    print(f"GPD VaR (95%): {var_95:.4f}")
    print(f"GPD ES (95%): {es_95:.4f}")
    
    # Plot tail distribution
    gpd_fig = gpd.plot_tail_distribution()
    gpd_fig.savefig('gpd_tail.png')
    
    # Initialize and fit dynamic EVT
    dynamic_evt = DynamicEVT()
    dynamic_evt.fit(returns)
    
    # Plot parameter evolution
    param_fig = dynamic_evt.plot_parameter_evolution()
    param_fig.savefig('dynamic_evt_params.png')
    
    # Plot risk metrics
    risk_fig = dynamic_evt.plot_risk_metrics()
    risk_fig.savefig('dynamic_evt_risk.png')
    
    # Initialize and fit conditional EVT
    cond_evt = ConditionalEVT()
    cond_evt.fit(returns)
    
    # Plot conditional effects
    cond_fig = cond_evt.plot_conditional_effects()
    cond_fig.savefig('conditional_evt_effects.png')
    
    # Initialize and fit Bayesian EVT
    bayes_evt = BayesianEVT()
    bayes_evt.fit(returns)
    
    # Plot posterior distributions
    post_fig = bayes_evt.plot_posterior_distributions()
    post_fig.savefig('bayesian_evt_posterior.png')
    
    # Plot risk metrics distributions
    risk_dist_fig = bayes_evt.plot_risk_metrics_distributions()
    risk_dist_fig.savefig('bayesian_evt_risk.png')
