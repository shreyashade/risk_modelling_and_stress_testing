"""
Extreme Value Theory (EVT) Models for Risk Modeling and Stress Testing

This module implements various Extreme Value Theory methods for tail risk estimation,
including Block Maxima and Peaks-Over-Threshold approaches.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Union, Optional, Tuple, Callable
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EVTModel:
    """Base class for Extreme Value Theory models"""
    
    def __init__(self, name: str):
        """
        Initialize an EVT model
        
        Parameters:
        -----------
        name : str
            Name of the EVT model
        """
        self.name = name
        self.fitted = False
        self.params = {}
    
    def fit(self, data: np.ndarray, **kwargs) -> Dict:
        """
        Fit the EVT model to data
        
        Parameters:
        -----------
        data : np.ndarray
            Data to fit the model to
        **kwargs : dict
            Additional parameters for fitting
            
        Returns:
        --------
        dict
            Dictionary with fitted parameters
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def estimate_var(self, alpha: float, **kwargs) -> float:
        """
        Estimate Value at Risk (VaR) using the fitted model
        
        Parameters:
        -----------
        alpha : float
            Confidence level (e.g., 0.05 for 95% VaR)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated VaR
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        raise NotImplementedError("Subclasses must implement this method")
    
    def estimate_es(self, alpha: float, **kwargs) -> float:
        """
        Estimate Expected Shortfall (ES) using the fitted model
        
        Parameters:
        -----------
        alpha : float
            Confidence level (e.g., 0.05 for 95% ES)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated ES
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        raise NotImplementedError("Subclasses must implement this method")
    
    def estimate_return_level(self, return_period: float, **kwargs) -> float:
        """
        Estimate return level for a given return period
        
        Parameters:
        -----------
        return_period : float
            Return period (e.g., 100 for 1-in-100 event)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated return level
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        raise NotImplementedError("Subclasses must implement this method")


class BlockMaximaModel(EVTModel):
    """Block Maxima approach using Generalized Extreme Value (GEV) distribution"""
    
    def __init__(self):
        """Initialize a Block Maxima model"""
        super().__init__("Block Maxima Model")
        self.block_size = None
        self.n_blocks = None
        self.maxima = None
    
    def fit(self, data: np.ndarray, block_size: Optional[int] = None, 
           n_blocks: Optional[int] = None, method: str = 'mle', **kwargs) -> Dict:
        """
        Fit the Block Maxima model to data
        
        Parameters:
        -----------
        data : np.ndarray
            Data to fit the model to
        block_size : int, optional
            Size of blocks (if None, calculated from n_blocks)
        n_blocks : int, optional
            Number of blocks (if None, calculated from block_size)
        method : str, default='mle'
            Method for parameter estimation ('mle' or 'pwm')
        **kwargs : dict
            Additional parameters for fitting
            
        Returns:
        --------
        dict
            Dictionary with fitted parameters
        """
        # Determine block size or number of blocks
        if block_size is None and n_blocks is None:
            # Default to approximately 20-30 blocks
            n_blocks = min(30, len(data) // 10)
            block_size = len(data) // n_blocks
        elif block_size is None:
            block_size = len(data) // n_blocks
        elif n_blocks is None:
            n_blocks = len(data) // block_size
        
        self.block_size = block_size
        self.n_blocks = n_blocks
        
        # Extract block maxima
        n_full_blocks = len(data) // block_size
        self.maxima = np.array([
            np.max(data[i * block_size:(i + 1) * block_size])
            for i in range(n_full_blocks)
        ])
        
        # Fit GEV distribution
        if method == 'mle':
            self._fit_mle(**kwargs)
        elif method == 'pwm':
            self._fit_pwm(**kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        self.fitted = True
        
        return self.params
    
    def _fit_mle(self, initial_params: Optional[List[float]] = None, **kwargs) -> None:
        """
        Fit GEV distribution using Maximum Likelihood Estimation
        
        Parameters:
        -----------
        initial_params : list of float, optional
            Initial parameters [location, scale, shape]
        **kwargs : dict
            Additional parameters for optimization
        """
        if initial_params is None:
            # Initialize with method of moments
            mean = np.mean(self.maxima)
            var = np.var(self.maxima)
            skew = stats.skew(self.maxima)
            
            # Approximations based on method of moments
            shape = -0.1  # Start with a small negative shape
            scale = np.sqrt(var) * 0.78  # Approximation
            location = mean - 0.57722 * scale / (1 - shape)  # Approximation
            
            initial_params = [location, scale, shape]
        
        # Define negative log-likelihood function
        def neg_log_likelihood(params):
            loc, scale, shape = params
            
            # Check for valid parameters
            if scale <= 0:
                return np.inf
            
            # Calculate negative log-likelihood
            if abs(shape) < 1e-6:  # Shape close to zero (Gumbel)
                z = (self.maxima - loc) / scale
                ll = -np.sum(-z - np.exp(-z) - np.log(scale))
            else:
                z = 1 + shape * (self.maxima - loc) / scale
                if np.any(z <= 0):
                    return np.inf
                ll = -np.sum(-(1 + 1/shape) * np.log(z) - z**(-1/shape) - np.log(scale))
            
            return -ll
        
        # Optimize parameters
        result = minimize(neg_log_likelihood, initial_params, method='Nelder-Mead')
        
        if not result.success:
            logger.warning("MLE optimization did not converge")
        
        # Store parameters
        self.params = {
            'location': result.x[0],
            'scale': result.x[1],
            'shape': result.x[2]
        }
    
    def _fit_pwm(self, **kwargs) -> None:
        """
        Fit GEV distribution using Probability Weighted Moments
        
        Parameters:
        -----------
        **kwargs : dict
            Additional parameters for fitting
        """
        # Sort data
        sorted_data = np.sort(self.maxima)
        n = len(sorted_data)
        
        # Calculate probability weighted moments
        b0 = np.mean(sorted_data)
        b1 = np.sum(sorted_data * np.arange(n)) / (n * (n - 1))
        b2 = np.sum(sorted_data * np.arange(n) * (np.arange(n) - 1)) / (n * (n - 1) * (n - 2))
        
        # Calculate L-moments
        l1 = b0
        l2 = 2 * b1 - b0
        l3 = 6 * b2 - 6 * b1 + b0
        
        # Calculate L-moment ratios
        t2 = l2 / l1
        t3 = l3 / l2
        
        # Estimate shape parameter using approximation
        c = 2 / (3 + t3) - np.log(2) / np.log(3)
        shape = 7.8590 * c + 2.9554 * c**2
        
        # Estimate scale and location
        gamma = np.exp(np.math.lgamma(1 - shape))
        scale = l2 * shape / (gamma * (1 - 2**(-shape)))
        location = l1 - scale * (1 - gamma) / shape
        
        # Store parameters
        self.params = {
            'location': location,
            'scale': scale,
            'shape': shape
        }
    
    def estimate_var(self, alpha: float, **kwargs) -> float:
        """
        Estimate Value at Risk (VaR) using the fitted GEV model
        
        Parameters:
        -----------
        alpha : float
            Confidence level (e.g., 0.05 for 95% VaR)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated VaR
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        location = self.params['location']
        scale = self.params['scale']
        shape = self.params['shape']
        
        # Calculate VaR using GEV quantile function
        p = 1 - alpha
        
        if abs(shape) < 1e-6:  # Shape close to zero (Gumbel)
            var = location - scale * np.log(-np.log(p))
        else:
            var = location + scale * (((-np.log(p))**(-shape) - 1) / shape)
        
        return var
    
    def estimate_es(self, alpha: float, **kwargs) -> float:
        """
        Estimate Expected Shortfall (ES) using the fitted GEV model
        
        Parameters:
        -----------
        alpha : float
            Confidence level (e.g., 0.05 for 95% ES)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated ES
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        location = self.params['location']
        scale = self.params['scale']
        shape = self.params['shape']
        
        # Calculate VaR
        var = self.estimate_var(alpha)
        
        # Calculate ES
        if shape >= 1:
            raise ValueError("ES is infinite for shape >= 1")
        
        p = 1 - alpha
        
        if abs(shape) < 1e-6:  # Shape close to zero (Gumbel)
            es = var + scale * (np.euler_gamma + np.log(-np.log(p)))
        else:
            z = -np.log(p)
            es = var / (1 - shape) + scale * (z**(-shape) / (1 - shape) - 1) / shape
        
        return es
    
    def estimate_return_level(self, return_period: float, **kwargs) -> float:
        """
        Estimate return level for a given return period
        
        Parameters:
        -----------
        return_period : float
            Return period (e.g., 100 for 1-in-100 event)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated return level
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert return period to probability
        p = 1 - 1 / return_period
        
        # Calculate return level using GEV quantile function
        return self.estimate_var(1 - p)
    
    def plot_diagnostic(self, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot diagnostic plots for the fitted GEV model
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 10)
            Figure size
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        location = self.params['location']
        scale = self.params['scale']
        shape = self.params['shape']
        
        plt.figure(figsize=figsize)
        
        # Probability plot
        plt.subplot(2, 2, 1)
        n = len(self.maxima)
        sorted_data = np.sort(self.maxima)
        p = np.arange(1, n + 1) / (n + 1)
        
        # Calculate theoretical quantiles
        if abs(shape) < 1e-6:  # Shape close to zero (Gumbel)
            theoretical_quantiles = location - scale * np.log(-np.log(p))
        else:
            theoretical_quantiles = location + scale * (((-np.log(p))**(-shape) - 1) / shape)
        
        plt.scatter(theoretical_quantiles, sorted_data)
        plt.plot([min(theoretical_quantiles), max(theoretical_quantiles)],
                [min(theoretical_quantiles), max(theoretical_quantiles)], 'r--')
        plt.xlabel('Model')
        plt.ylabel('Empirical')
        plt.title('Probability Plot')
        plt.grid(True)
        
        # Quantile plot
        plt.subplot(2, 2, 2)
        plt.scatter(sorted_data, theoretical_quantiles)
        plt.plot([min(sorted_data), max(sorted_data)],
                [min(sorted_data), max(sorted_data)], 'r--')
        plt.xlabel('Empirical')
        plt.ylabel('Model')
        plt.title('Quantile Plot')
        plt.grid(True)
        
        # Return level plot
        plt.subplot(2, 2, 3)
        return_periods = np.logspace(0, 2, 100)
        return_levels = [self.estimate_return_level(rp) for rp in return_periods]
        
        plt.semilogx(return_periods, return_levels, 'b-')
        
        # Add empirical return levels
        empirical_return_periods = n / np.arange(n, 0, -1)
        plt.scatter(empirical_return_periods, sorted_data, color='red', alpha=0.5)
        
        plt.xlabel('Return Period')
        plt.ylabel('Return Level')
        plt.title('Return Level Plot')
        plt.grid(True)
        
        # Density plot
        plt.subplot(2, 2, 4)
        x = np.linspace(min(self.maxima), max(self.maxima), 1000)
        
        # Calculate GEV density
        if abs(shape) < 1e-6:  # Shape close to zero (Gumbel)
            z = (x - location) / scale
            density = np.exp(-z - np.exp(-z)) / scale
        else:
            z = 1 + shape * (x - location) / scale
            valid_idx = z > 0
            density = np.zeros_like(x)
            density[valid_idx] = z[valid_idx]**(-1/shape - 1) * np.exp(-z[valid_idx]**(-1/shape)) / scale
        
        plt.plot(x, density, 'b-', label='GEV Density')
        
        # Add histogram of data
        plt.hist(self.maxima, bins=20, density=True, alpha=0.5, label='Data')
        
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Density Plot')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


class PeaksOverThresholdModel(EVTModel):
    """Peaks-Over-Threshold approach using Generalized Pareto Distribution (GPD)"""
    
    def __init__(self):
        """Initialize a Peaks-Over-Threshold model"""
        super().__init__("Peaks-Over-Threshold Model")
        self.threshold = None
        self.exceedances = None
        self.n_exceedances = None
        self.n_total = None
    
    def fit(self, data: np.ndarray, threshold: Optional[float] = None, 
           threshold_method: str = 'quantile', quantile: float = 0.9,
           method: str = 'mle', **kwargs) -> Dict:
        """
        Fit the Peaks-Over-Threshold model to data
        
        Parameters:
        -----------
        data : np.ndarray
            Data to fit the model to
        threshold : float, optional
            Threshold for exceedances (if None, determined by threshold_method)
        threshold_method : str, default='quantile'
            Method for threshold selection ('quantile', 'mean_excess', 'fixed')
        quantile : float, default=0.9
            Quantile for threshold selection if threshold_method='quantile'
        method : str, default='mle'
            Method for parameter estimation ('mle' or 'pwm')
        **kwargs : dict
            Additional parameters for fitting
            
        Returns:
        --------
        dict
            Dictionary with fitted parameters
        """
        self.n_total = len(data)
        
        # Determine threshold
        if threshold is None:
            if threshold_method == 'quantile':
                threshold = np.quantile(data, quantile)
            elif threshold_method == 'mean_excess':
                threshold = self._select_threshold_mean_excess(data, **kwargs)
            elif threshold_method == 'fixed':
                raise ValueError("Threshold must be provided for threshold_method='fixed'")
            else:
                raise ValueError(f"Unsupported threshold_method: {threshold_method}")
        
        self.threshold = threshold
        
        # Extract exceedances
        self.exceedances = data[data > threshold] - threshold
        self.n_exceedances = len(self.exceedances)
        
        if self.n_exceedances < 10:
            logger.warning(f"Only {self.n_exceedances} exceedances. Consider lowering the threshold.")
        
        # Fit GPD distribution
        if method == 'mle':
            self._fit_mle(**kwargs)
        elif method == 'pwm':
            self._fit_pwm(**kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        self.fitted = True
        
        return self.params
    
    def _select_threshold_mean_excess(self, data: np.ndarray, 
                                    start_quantile: float = 0.5, 
                                    end_quantile: float = 0.95,
                                    n_points: int = 20) -> float:
        """
        Select threshold using mean excess plot
        
        Parameters:
        -----------
        data : np.ndarray
            Data for threshold selection
        start_quantile : float, default=0.5
            Starting quantile for threshold range
        end_quantile : float, default=0.95
            Ending quantile for threshold range
        n_points : int, default=20
            Number of points to evaluate
            
        Returns:
        --------
        float
            Selected threshold
        """
        # Generate threshold range
        thresholds = np.linspace(
            np.quantile(data, start_quantile),
            np.quantile(data, end_quantile),
            n_points
        )
        
        # Calculate mean excess for each threshold
        mean_excess = []
        for u in thresholds:
            exceedances = data[data > u] - u
            if len(exceedances) > 0:
                mean_excess.append(np.mean(exceedances))
            else:
                mean_excess.append(np.nan)
        
        # Find the threshold where mean excess becomes approximately linear
        # This is a simplified approach - in practice, visual inspection is often used
        # Here we use the threshold where the second derivative of the mean excess function is minimized
        mean_excess = np.array(mean_excess)
        valid_idx = ~np.isnan(mean_excess)
        
        if np.sum(valid_idx) < 5:
            logger.warning("Not enough valid points for threshold selection")
            return np.quantile(data, 0.9)  # Default to 90% quantile
        
        # Calculate first and second differences
        first_diff = np.diff(mean_excess[valid_idx])
        second_diff = np.abs(np.diff(first_diff))
        
        # Find the point where second difference is minimized
        min_idx = np.argmin(second_diff) + 1  # +1 due to double differencing
        
        # Return the corresponding threshold
        valid_thresholds = thresholds[valid_idx]
        if min_idx < len(valid_thresholds):
            return valid_thresholds[min_idx]
        else:
            return valid_thresholds[-1]
    
    def _fit_mle(self, initial_params: Optional[List[float]] = None, **kwargs) -> None:
        """
        Fit GPD distribution using Maximum Likelihood Estimation
        
        Parameters:
        -----------
        initial_params : list of float, optional
            Initial parameters [scale, shape]
        **kwargs : dict
            Additional parameters for optimization
        """
        if initial_params is None:
            # Initialize with method of moments
            mean = np.mean(self.exceedances)
            var = np.var(self.exceedances)
            
            # Approximations based on method of moments
            shape = 0.5 * (((mean**2 / var) + 1) - 1)
            scale = mean * (1 + shape)
            
            initial_params = [scale, shape]
        
        # Define negative log-likelihood function
        def neg_log_likelihood(params):
            scale, shape = params
            
            # Check for valid parameters
            if scale <= 0:
                return np.inf
            
            # Calculate negative log-likelihood
            if abs(shape) < 1e-6:  # Shape close to zero (Exponential)
                ll = -np.sum(-self.exceedances / scale - np.log(scale))
            else:
                z = 1 + shape * self.exceedances / scale
                if np.any(z <= 0):
                    return np.inf
                ll = -np.sum(-(1 + 1/shape) * np.log(z) - np.log(scale))
            
            return -ll
        
        # Optimize parameters
        result = minimize(neg_log_likelihood, initial_params, method='Nelder-Mead')
        
        if not result.success:
            logger.warning("MLE optimization did not converge")
        
        # Store parameters
        self.params = {
            'scale': result.x[0],
            'shape': result.x[1],
            'threshold': self.threshold,
            'exceedance_rate': self.n_exceedances / self.n_total
        }
    
    def _fit_pwm(self, **kwargs) -> None:
        """
        Fit GPD distribution using Probability Weighted Moments
        
        Parameters:
        -----------
        **kwargs : dict
            Additional parameters for fitting
        """
        # Sort exceedances
        sorted_data = np.sort(self.exceedances)
        n = len(sorted_data)
        
        # Calculate probability weighted moments
        b0 = np.mean(sorted_data)
        b1 = np.sum(sorted_data * np.arange(n)) / (n * (n - 1))
        
        # Estimate parameters
        shape = 2 * (b0 / (b0 - 2 * b1) - 1)
        scale = 2 * b0 * b1 / (b0 - 2 * b1)
        
        # Store parameters
        self.params = {
            'scale': scale,
            'shape': shape,
            'threshold': self.threshold,
            'exceedance_rate': self.n_exceedances / self.n_total
        }
    
    def estimate_var(self, alpha: float, **kwargs) -> float:
        """
        Estimate Value at Risk (VaR) using the fitted GPD model
        
        Parameters:
        -----------
        alpha : float
            Confidence level (e.g., 0.05 for 95% VaR)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated VaR
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        scale = self.params['scale']
        shape = self.params['shape']
        threshold = self.params['threshold']
        exceedance_rate = self.params['exceedance_rate']
        
        # Calculate VaR using GPD quantile function
        p = 1 - alpha
        q = (1 - p) / exceedance_rate
        
        if abs(shape) < 1e-6:  # Shape close to zero (Exponential)
            var = threshold - scale * np.log(q)
        else:
            var = threshold + scale * ((q**(-shape) - 1) / shape)
        
        return var
    
    def estimate_es(self, alpha: float, **kwargs) -> float:
        """
        Estimate Expected Shortfall (ES) using the fitted GPD model
        
        Parameters:
        -----------
        alpha : float
            Confidence level (e.g., 0.05 for 95% ES)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated ES
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        scale = self.params['scale']
        shape = self.params['shape']
        threshold = self.params['threshold']
        
        # Calculate VaR
        var = self.estimate_var(alpha)
        
        # Calculate ES
        if shape >= 1:
            raise ValueError("ES is infinite for shape >= 1")
        
        if abs(shape) < 1e-6:  # Shape close to zero (Exponential)
            es = var + scale
        else:
            es = var / (1 - shape) + scale - shape * (threshold - var) / (1 - shape)
        
        return es
    
    def estimate_return_level(self, return_period: float, **kwargs) -> float:
        """
        Estimate return level for a given return period
        
        Parameters:
        -----------
        return_period : float
            Return period (e.g., 100 for 1-in-100 event)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated return level
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert return period to probability
        p = 1 - 1 / return_period
        
        # Calculate return level using GPD quantile function
        return self.estimate_var(1 - p)
    
    def plot_diagnostic(self, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot diagnostic plots for the fitted GPD model
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 10)
            Figure size
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        scale = self.params['scale']
        shape = self.params['shape']
        threshold = self.params['threshold']
        
        plt.figure(figsize=figsize)
        
        # Probability plot
        plt.subplot(2, 2, 1)
        n = len(self.exceedances)
        sorted_data = np.sort(self.exceedances)
        p = np.arange(1, n + 1) / (n + 1)
        
        # Calculate theoretical quantiles
        if abs(shape) < 1e-6:  # Shape close to zero (Exponential)
            theoretical_quantiles = -scale * np.log(1 - p)
        else:
            theoretical_quantiles = scale * ((1 - p)**(-shape) - 1) / shape
        
        plt.scatter(theoretical_quantiles, sorted_data)
        plt.plot([0, max(theoretical_quantiles)], [0, max(theoretical_quantiles)], 'r--')
        plt.xlabel('Model')
        plt.ylabel('Empirical')
        plt.title('Probability Plot')
        plt.grid(True)
        
        # Quantile plot
        plt.subplot(2, 2, 2)
        plt.scatter(sorted_data, theoretical_quantiles)
        plt.plot([0, max(sorted_data)], [0, max(sorted_data)], 'r--')
        plt.xlabel('Empirical')
        plt.ylabel('Model')
        plt.title('Quantile Plot')
        plt.grid(True)
        
        # Return level plot
        plt.subplot(2, 2, 3)
        return_periods = np.logspace(0, 2, 100)
        return_levels = [self.estimate_return_level(rp) for rp in return_periods]
        
        plt.semilogx(return_periods, return_levels, 'b-')
        
        # Add empirical return levels
        exceedance_rate = self.params['exceedance_rate']
        empirical_return_periods = 1 / (exceedance_rate * (1 - np.arange(1, n + 1) / (n + 1)))
        plt.scatter(empirical_return_periods, threshold + sorted_data, color='red', alpha=0.5)
        
        plt.xlabel('Return Period')
        plt.ylabel('Return Level')
        plt.title('Return Level Plot')
        plt.grid(True)
        
        # Density plot
        plt.subplot(2, 2, 4)
        x = np.linspace(0, max(self.exceedances), 1000)
        
        # Calculate GPD density
        if abs(shape) < 1e-6:  # Shape close to zero (Exponential)
            density = np.exp(-x / scale) / scale
        else:
            z = 1 + shape * x / scale
            valid_idx = z > 0
            density = np.zeros_like(x)
            density[valid_idx] = z[valid_idx]**(-1/shape - 1) / scale
        
        plt.plot(x, density, 'b-', label='GPD Density')
        
        # Add histogram of exceedances
        plt.hist(self.exceedances, bins=20, density=True, alpha=0.5, label='Exceedances')
        
        plt.xlabel('Exceedance')
        plt.ylabel('Density')
        plt.title('Density Plot')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_mean_excess(self, data: np.ndarray, 
                       start_quantile: float = 0.5, 
                       end_quantile: float = 0.95,
                       n_points: int = 20,
                       figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot mean excess function for threshold selection
        
        Parameters:
        -----------
        data : np.ndarray
            Data for threshold selection
        start_quantile : float, default=0.5
            Starting quantile for threshold range
        end_quantile : float, default=0.95
            Ending quantile for threshold range
        n_points : int, default=20
            Number of points to evaluate
        figsize : tuple, default=(10, 6)
            Figure size
        """
        # Generate threshold range
        thresholds = np.linspace(
            np.quantile(data, start_quantile),
            np.quantile(data, end_quantile),
            n_points
        )
        
        # Calculate mean excess for each threshold
        mean_excess = []
        confidence_intervals = []
        
        for u in thresholds:
            exceedances = data[data > u] - u
            if len(exceedances) > 1:
                mean = np.mean(exceedances)
                std = np.std(exceedances) / np.sqrt(len(exceedances))
                mean_excess.append(mean)
                confidence_intervals.append(1.96 * std)  # 95% confidence interval
            else:
                mean_excess.append(np.nan)
                confidence_intervals.append(np.nan)
        
        # Plot mean excess function
        plt.figure(figsize=figsize)
        
        plt.errorbar(thresholds, mean_excess, yerr=confidence_intervals, fmt='o-')
        
        if self.fitted:
            # Add the selected threshold
            plt.axvline(self.threshold, color='r', linestyle='--', 
                      label=f'Selected threshold: {self.threshold:.2f}')
            
            # Add the theoretical mean excess function for the fitted GPD
            shape = self.params['shape']
            scale = self.params['scale']
            
            if shape < 1:
                x = np.linspace(self.threshold, max(thresholds), 100)
                if abs(shape) < 1e-6:  # Shape close to zero (Exponential)
                    y = np.full_like(x, scale)
                else:
                    y = scale / (1 - shape) + shape * (x - self.threshold) / (1 - shape)
                
                plt.plot(x, y, 'r-', label='Theoretical')
        
        plt.xlabel('Threshold')
        plt.ylabel('Mean Excess')
        plt.title('Mean Excess Plot')
        plt.grid(True)
        plt.legend()
        plt.show()


class HillEstimator(EVTModel):
    """Hill estimator for heavy-tailed distributions"""
    
    def __init__(self):
        """Initialize a Hill estimator"""
        super().__init__("Hill Estimator")
        self.order_statistics = None
        self.hill_estimates = None
    
    def fit(self, data: np.ndarray, k: Optional[int] = None, **kwargs) -> Dict:
        """
        Fit the Hill estimator to data
        
        Parameters:
        -----------
        data : np.ndarray
            Data to fit the estimator to
        k : int, optional
            Number of upper order statistics to use (if None, determined automatically)
        **kwargs : dict
            Additional parameters for fitting
            
        Returns:
        --------
        dict
            Dictionary with fitted parameters
        """
        # Sort data in descending order
        sorted_data = -np.sort(-data)
        n = len(sorted_data)
        
        # Calculate Hill estimates for different values of k
        log_data = np.log(sorted_data)
        self.hill_estimates = np.zeros(n - 1)
        
        for i in range(1, n):
            self.hill_estimates[i - 1] = np.mean(log_data[:i]) - log_data[i - 1]
        
        self.hill_estimates = 1 / self.hill_estimates
        
        # Determine optimal k if not provided
        if k is None:
            k = self._select_optimal_k(**kwargs)
        
        # Store parameters
        self.params = {
            'tail_index': self.hill_estimates[k - 1],
            'k': k,
            'threshold': sorted_data[k - 1]
        }
        
        self.fitted = True
        
        return self.params
    
    def _select_optimal_k(self, method: str = 'stability', **kwargs) -> int:
        """
        Select optimal number of upper order statistics
        
        Parameters:
        -----------
        method : str, default='stability'
            Method for selecting k ('stability', 'bootstrap', 'fixed')
        **kwargs : dict
            Additional parameters for selection
            
        Returns:
        --------
        int
            Optimal k
        """
        n = len(self.hill_estimates) + 1
        
        if method == 'stability':
            # Find the region where Hill estimates are most stable
            # Calculate rolling standard deviation
            window = kwargs.get('window', min(100, n // 10))
            rolling_std = np.zeros(n - window)
            
            for i in range(n - window):
                rolling_std[i] = np.std(self.hill_estimates[i:i+window])
            
            # Find the minimum
            k = np.argmin(rolling_std) + window // 2
            
        elif method == 'bootstrap':
            # Use bootstrap to estimate the variance of the Hill estimator
            # This is a simplified implementation
            n_bootstrap = kwargs.get('n_bootstrap', 100)
            bootstrap_var = np.zeros(n - 1)
            
            for i in range(1, n):
                bootstrap_estimates = []
                for _ in range(n_bootstrap):
                    # Generate bootstrap sample
                    indices = np.random.choice(i, i, replace=True)
                    bootstrap_sample = np.log(np.sort(-indices)[:i])
                    hill = 1 / (np.mean(bootstrap_sample) - bootstrap_sample[i - 1])
                    bootstrap_estimates.append(hill)
                
                bootstrap_var[i - 1] = np.var(bootstrap_estimates)
            
            # Find the minimum
            k = np.argmin(bootstrap_var) + 1
            
        elif method == 'fixed':
            # Use a fixed fraction of the data
            fraction = kwargs.get('fraction', 0.1)
            k = int(n * fraction)
            
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return min(max(1, k), n - 1)
    
    def estimate_var(self, alpha: float, **kwargs) -> float:
        """
        Estimate Value at Risk (VaR) using the Hill estimator
        
        Parameters:
        -----------
        alpha : float
            Confidence level (e.g., 0.05 for 95% VaR)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated VaR
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        tail_index = self.params['tail_index']
        k = self.params['k']
        threshold = self.params['threshold']
        n = len(self.hill_estimates) + 1
        
        # Calculate VaR using Weissman estimator
        p = 1 - alpha
        var = threshold * (k / (n * p))**(1 / tail_index)
        
        return var
    
    def estimate_es(self, alpha: float, **kwargs) -> float:
        """
        Estimate Expected Shortfall (ES) using the Hill estimator
        
        Parameters:
        -----------
        alpha : float
            Confidence level (e.g., 0.05 for 95% ES)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated ES
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        tail_index = self.params['tail_index']
        
        if tail_index <= 1:
            raise ValueError("ES is infinite for tail_index <= 1")
        
        # Calculate VaR
        var = self.estimate_var(alpha)
        
        # Calculate ES
        es = var * tail_index / (tail_index - 1)
        
        return es
    
    def estimate_return_level(self, return_period: float, **kwargs) -> float:
        """
        Estimate return level for a given return period
        
        Parameters:
        -----------
        return_period : float
            Return period (e.g., 100 for 1-in-100 event)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated return level
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert return period to probability
        p = 1 - 1 / return_period
        
        # Calculate return level using Hill estimator
        return self.estimate_var(1 - p)
    
    def plot_hill(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot Hill estimates for different values of k
        
        Parameters:
        -----------
        figsize : tuple, default=(10, 6)
            Figure size
        """
        plt.figure(figsize=figsize)
        
        k_values = np.arange(1, len(self.hill_estimates) + 1)
        plt.plot(k_values, self.hill_estimates, 'b-')
        
        if self.fitted:
            k = self.params['k']
            tail_index = self.params['tail_index']
            plt.axvline(k, color='r', linestyle='--', 
                      label=f'Selected k: {k}')
            plt.axhline(tail_index, color='g', linestyle='--', 
                      label=f'Tail index: {tail_index:.2f}')
        
        plt.xlabel('k (Number of Upper Order Statistics)')
        plt.ylabel('Hill Estimate')
        plt.title('Hill Plot')
        plt.grid(True)
        plt.legend()
        plt.show()


class PickandsEstimator(EVTModel):
    """Pickands estimator for extreme value index"""
    
    def __init__(self):
        """Initialize a Pickands estimator"""
        super().__init__("Pickands Estimator")
        self.pickands_estimates = None
    
    def fit(self, data: np.ndarray, k: Optional[int] = None, **kwargs) -> Dict:
        """
        Fit the Pickands estimator to data
        
        Parameters:
        -----------
        data : np.ndarray
            Data to fit the estimator to
        k : int, optional
            Number of upper order statistics to use (if None, determined automatically)
        **kwargs : dict
            Additional parameters for fitting
            
        Returns:
        --------
        dict
            Dictionary with fitted parameters
        """
        # Sort data in descending order
        sorted_data = -np.sort(-data)
        n = len(sorted_data)
        
        # Calculate Pickands estimates for different values of k
        self.pickands_estimates = np.zeros(n // 4)
        
        for i in range(1, n // 4 + 1):
            m_i = sorted_data[i - 1]
            m_2i = sorted_data[2*i - 1]
            m_4i = sorted_data[4*i - 1]
            
            if m_i == m_2i or m_2i == m_4i:
                self.pickands_estimates[i - 1] = 0  # Avoid division by zero
            else:
                self.pickands_estimates[i - 1] = np.log((m_i - m_2i) / (m_2i - m_4i)) / np.log(2)
        
        # Determine optimal k if not provided
        if k is None:
            k = self._select_optimal_k(**kwargs)
        
        # Store parameters
        self.params = {
            'shape': self.pickands_estimates[k - 1],
            'k': k,
            'threshold': sorted_data[k - 1]
        }
        
        self.fitted = True
        
        return self.params
    
    def _select_optimal_k(self, method: str = 'stability', **kwargs) -> int:
        """
        Select optimal number of upper order statistics
        
        Parameters:
        -----------
        method : str, default='stability'
            Method for selecting k ('stability', 'fixed')
        **kwargs : dict
            Additional parameters for selection
            
        Returns:
        --------
        int
            Optimal k
        """
        n = len(self.pickands_estimates) + 1
        
        if method == 'stability':
            # Find the region where Pickands estimates are most stable
            # Calculate rolling standard deviation
            window = kwargs.get('window', min(50, n // 10))
            rolling_std = np.zeros(n - window)
            
            for i in range(n - window):
                rolling_std[i] = np.std(self.pickands_estimates[i:i+window])
            
            # Find the minimum
            k = np.argmin(rolling_std) + window // 2
            
        elif method == 'fixed':
            # Use a fixed fraction of the data
            fraction = kwargs.get('fraction', 0.05)
            k = int(n * fraction)
            
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return min(max(1, k), n - 1)
    
    def estimate_var(self, alpha: float, **kwargs) -> float:
        """
        Estimate Value at Risk (VaR) using the Pickands estimator
        
        Parameters:
        -----------
        alpha : float
            Confidence level (e.g., 0.05 for 95% VaR)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated VaR
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        shape = self.params['shape']
        k = self.params['k']
        threshold = self.params['threshold']
        n = len(self.pickands_estimates) * 4
        
        # Calculate VaR using Weissman-type estimator
        p = 1 - alpha
        var = threshold * (k / (n * p))**shape
        
        return var
    
    def estimate_es(self, alpha: float, **kwargs) -> float:
        """
        Estimate Expected Shortfall (ES) using the Pickands estimator
        
        Parameters:
        -----------
        alpha : float
            Confidence level (e.g., 0.05 for 95% ES)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated ES
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        shape = self.params['shape']
        
        if shape >= 1:
            raise ValueError("ES is infinite for shape >= 1")
        
        # Calculate VaR
        var = self.estimate_var(alpha)
        
        # Calculate ES
        es = var / (1 - shape)
        
        return es
    
    def estimate_return_level(self, return_period: float, **kwargs) -> float:
        """
        Estimate return level for a given return period
        
        Parameters:
        -----------
        return_period : float
            Return period (e.g., 100 for 1-in-100 event)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated return level
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert return period to probability
        p = 1 - 1 / return_period
        
        # Calculate return level using Pickands estimator
        return self.estimate_var(1 - p)
    
    def plot_pickands(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot Pickands estimates for different values of k
        
        Parameters:
        -----------
        figsize : tuple, default=(10, 6)
            Figure size
        """
        plt.figure(figsize=figsize)
        
        k_values = np.arange(1, len(self.pickands_estimates) + 1)
        plt.plot(k_values, self.pickands_estimates, 'b-')
        
        if self.fitted:
            k = self.params['k']
            shape = self.params['shape']
            plt.axvline(k, color='r', linestyle='--', 
                      label=f'Selected k: {k}')
            plt.axhline(shape, color='g', linestyle='--', 
                      label=f'Shape parameter: {shape:.2f}')
        
        plt.xlabel('k (Number of Upper Order Statistics)')
        plt.ylabel('Pickands Estimate')
        plt.title('Pickands Plot')
        plt.grid(True)
        plt.legend()
        plt.show()


class EVTRiskModel:
    """Main class for EVT-based risk modeling"""
    
    def __init__(self):
        """Initialize an EVT Risk Model"""
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, model: EVTModel) -> None:
        """
        Add an EVT model
        
        Parameters:
        -----------
        name : str
            Name of the model
        model : EVTModel
            EVT model object
        """
        self.models[name] = model
    
    def fit_model(self, model_name: str, data: np.ndarray, **kwargs) -> str:
        """
        Fit an EVT model to data
        
        Parameters:
        -----------
        model_name : str
            Name of the model to fit
        data : np.ndarray
            Data to fit the model to
        **kwargs : dict
            Additional parameters for fitting
            
        Returns:
        --------
        str
            Result ID for retrieving model results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        # Fit model
        params = model.fit(data, **kwargs)
        
        # Store results
        result_id = f"{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.results[result_id] = {
            'model': model_name,
            'params': params,
            'data': data
        }
        
        return result_id
    
    def estimate_var(self, result_id: str, alpha: float, **kwargs) -> float:
        """
        Estimate Value at Risk (VaR) using a fitted model
        
        Parameters:
        -----------
        result_id : str
            Result ID from a previous fit
        alpha : float
            Confidence level (e.g., 0.05 for 95% VaR)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated VaR
        """
        if result_id not in self.results:
            raise ValueError(f"Result '{result_id}' not found")
        
        model_name = self.results[result_id]['model']
        model = self.models[model_name]
        
        return model.estimate_var(alpha, **kwargs)
    
    def estimate_es(self, result_id: str, alpha: float, **kwargs) -> float:
        """
        Estimate Expected Shortfall (ES) using a fitted model
        
        Parameters:
        -----------
        result_id : str
            Result ID from a previous fit
        alpha : float
            Confidence level (e.g., 0.05 for 95% ES)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated ES
        """
        if result_id not in self.results:
            raise ValueError(f"Result '{result_id}' not found")
        
        model_name = self.results[result_id]['model']
        model = self.models[model_name]
        
        return model.estimate_es(alpha, **kwargs)
    
    def estimate_return_level(self, result_id: str, return_period: float, **kwargs) -> float:
        """
        Estimate return level for a given return period
        
        Parameters:
        -----------
        result_id : str
            Result ID from a previous fit
        return_period : float
            Return period (e.g., 100 for 1-in-100 event)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated return level
        """
        if result_id not in self.results:
            raise ValueError(f"Result '{result_id}' not found")
        
        model_name = self.results[result_id]['model']
        model = self.models[model_name]
        
        return model.estimate_return_level(return_period, **kwargs)
    
    def compare_models(self, result_ids: List[str], metric: str = 'var_95',
                     figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Compare results from different models
        
        Parameters:
        -----------
        result_ids : list of str
            Result IDs to compare
        metric : str, default='var_95'
            Risk metric to compare ('var_95', 'var_99', 'es_95', 'es_99')
        figsize : tuple, default=(10, 6)
            Figure size
        """
        for result_id in result_ids:
            if result_id not in self.results:
                raise ValueError(f"Result '{result_id}' not found")
        
        plt.figure(figsize=figsize)
        
        labels = []
        values = []
        
        for result_id in result_ids:
            model_name = self.results[result_id]['model']
            
            if metric.startswith('var'):
                alpha = float(metric.split('_')[1]) / 100
                value = self.estimate_var(result_id, alpha)
            elif metric.startswith('es'):
                alpha = float(metric.split('_')[1]) / 100
                value = self.estimate_es(result_id, alpha)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            labels.append(model_name)
            values.append(value)
        
        plt.bar(labels, values)
        plt.title(f"Comparison of {metric.upper()} across Models")
        plt.ylabel(metric.upper())
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()
    
    def plot_return_levels(self, result_ids: List[str], 
                         return_periods: Optional[np.ndarray] = None,
                         figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot return levels for different models
        
        Parameters:
        -----------
        result_ids : list of str
            Result IDs to compare
        return_periods : np.ndarray, optional
            Return periods to evaluate (if None, uses default range)
        figsize : tuple, default=(10, 6)
            Figure size
        """
        for result_id in result_ids:
            if result_id not in self.results:
                raise ValueError(f"Result '{result_id}' not found")
        
        if return_periods is None:
            return_periods = np.logspace(0, 3, 100)
        
        plt.figure(figsize=figsize)
        
        for result_id in result_ids:
            model_name = self.results[result_id]['model']
            
            return_levels = [self.estimate_return_level(result_id, rp) for rp in return_periods]
            
            plt.semilogx(return_periods, return_levels, label=model_name)
        
        plt.xlabel('Return Period')
        plt.ylabel('Return Level')
        plt.title('Return Level Plot')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def save_results(self, result_id: str, filepath: str) -> None:
        """
        Save model results to file
        
        Parameters:
        -----------
        result_id : str
            Result ID to save
        filepath : str
            Path to save the results
        """
        if result_id not in self.results:
            raise ValueError(f"Result '{result_id}' not found")
        
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.results[result_id], f)
    
    @classmethod
    def load_results(cls, filepath: str) -> Dict:
        """
        Load model results from file
        
        Parameters:
        -----------
        filepath : str
            Path to load the results from
            
        Returns:
        --------
        dict
            Loaded results
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class ConditionalEVT:
    """Conditional EVT model with time-varying parameters"""
    
    def __init__(self, evt_model: EVTModel, volatility_model: Optional[Callable] = None):
        """
        Initialize a Conditional EVT model
        
        Parameters:
        -----------
        evt_model : EVTModel
            EVT model to use
        volatility_model : callable, optional
            Function to estimate volatility
        """
        self.evt_model = evt_model
        self.volatility_model = volatility_model
        self.fitted = False
        self.params = {}
    
    def fit(self, returns: np.ndarray, volatility: Optional[np.ndarray] = None, **kwargs) -> Dict:
        """
        Fit the Conditional EVT model to data
        
        Parameters:
        -----------
        returns : np.ndarray
            Return data
        volatility : np.ndarray, optional
            Volatility estimates (if None, estimated using volatility_model)
        **kwargs : dict
            Additional parameters for fitting
            
        Returns:
        --------
        dict
            Dictionary with fitted parameters
        """
        # Estimate volatility if not provided
        if volatility is None:
            if self.volatility_model is None:
                raise ValueError("Volatility model must be provided if volatility is not provided")
            
            volatility = self.volatility_model(returns)
        
        # Standardize returns
        std_returns = returns / volatility
        
        # Fit EVT model to standardized returns
        self.evt_model.fit(std_returns, **kwargs)
        
        # Store parameters
        self.params = {
            'evt_params': self.evt_model.params,
            'last_volatility': volatility[-1]
        }
        
        self.fitted = True
        
        return self.params
    
    def estimate_var(self, alpha: float, forecast_volatility: Optional[float] = None, **kwargs) -> float:
        """
        Estimate Value at Risk (VaR) using the fitted model
        
        Parameters:
        -----------
        alpha : float
            Confidence level (e.g., 0.05 for 95% VaR)
        forecast_volatility : float, optional
            Forecasted volatility (if None, uses last observed volatility)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated VaR
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Use forecasted volatility or last observed volatility
        if forecast_volatility is None:
            forecast_volatility = self.params['last_volatility']
        
        # Estimate VaR for standardized returns
        std_var = self.evt_model.estimate_var(alpha, **kwargs)
        
        # Scale by volatility
        var = std_var * forecast_volatility
        
        return var
    
    def estimate_es(self, alpha: float, forecast_volatility: Optional[float] = None, **kwargs) -> float:
        """
        Estimate Expected Shortfall (ES) using the fitted model
        
        Parameters:
        -----------
        alpha : float
            Confidence level (e.g., 0.05 for 95% ES)
        forecast_volatility : float, optional
            Forecasted volatility (if None, uses last observed volatility)
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated ES
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Use forecasted volatility or last observed volatility
        if forecast_volatility is None:
            forecast_volatility = self.params['last_volatility']
        
        # Estimate ES for standardized returns
        std_es = self.evt_model.estimate_es(alpha, **kwargs)
        
        # Scale by volatility
        es = std_es * forecast_volatility
        
        return es


class MultivariateEVT:
    """Multivariate EVT model using copulas"""
    
    def __init__(self, marginal_models: List[EVTModel], copula_type: str = 'gaussian'):
        """
        Initialize a Multivariate EVT model
        
        Parameters:
        -----------
        marginal_models : list of EVTModel
            EVT models for marginal distributions
        copula_type : str, default='gaussian'
            Type of copula to use ('gaussian', 't', 'clayton', 'gumbel', 'frank')
        """
        self.marginal_models = marginal_models
        self.copula_type = copula_type
        self.n_variables = len(marginal_models)
        self.fitted = False
        self.params = {}
    
    def fit(self, data: np.ndarray, **kwargs) -> Dict:
        """
        Fit the Multivariate EVT model to data
        
        Parameters:
        -----------
        data : np.ndarray
            Data matrix of shape (n_samples, n_variables)
        **kwargs : dict
            Additional parameters for fitting
            
        Returns:
        --------
        dict
            Dictionary with fitted parameters
        """
        if data.shape[1] != self.n_variables:
            raise ValueError(f"Data has {data.shape[1]} variables, but {self.n_variables} models were provided")
        
        # Fit marginal models
        marginal_params = []
        for i in range(self.n_variables):
            params = self.marginal_models[i].fit(data[:, i], **kwargs)
            marginal_params.append(params)
        
        # Transform data to uniform using fitted marginal models
        uniform_data = np.zeros_like(data)
        for i in range(self.n_variables):
            # Calculate empirical CDF
            sorted_idx = np.argsort(data[:, i])
            ranks = np.zeros(len(data))
            ranks[sorted_idx] = np.arange(1, len(data) + 1) / (len(data) + 1)  # Use (n+1) to avoid 0 and 1
            uniform_data[:, i] = ranks
        
        # Fit copula
        if self.copula_type == 'gaussian':
            # Calculate correlation matrix of uniform data transformed to normal
            normal_data = stats.norm.ppf(uniform_data)
            correlation_matrix = np.corrcoef(normal_data, rowvar=False)
            copula_params = {'correlation_matrix': correlation_matrix}
            
        elif self.copula_type == 't':
            # Estimate correlation matrix and degrees of freedom
            normal_data = stats.norm.ppf(uniform_data)
            correlation_matrix = np.corrcoef(normal_data, rowvar=False)
            
            # Estimate degrees of freedom (simplified approach)
            df = kwargs.get('df', 4)  # Default to 4 degrees of freedom
            copula_params = {'correlation_matrix': correlation_matrix, 'df': df}
            
        elif self.copula_type in ['clayton', 'gumbel', 'frank']:
            # Estimate copula parameter (simplified approach for bivariate case)
            if self.n_variables == 2:
                # Calculate Kendall's tau
                tau = stats.kendalltau(data[:, 0], data[:, 1])[0]
                
                # Convert tau to copula parameter
                if self.copula_type == 'clayton':
                    theta = 2 * tau / (1 - tau)
                elif self.copula_type == 'gumbel':
                    theta = 1 / (1 - tau)
                elif self.copula_type == 'frank':
                    theta = -1  # Placeholder, actual conversion is more complex
                
                copula_params = {'theta': theta}
            else:
                raise ValueError(f"{self.copula_type} copula not implemented for dimensions > 2")
        else:
            raise ValueError(f"Unsupported copula type: {self.copula_type}")
        
        # Store parameters
        self.params = {
            'marginal_params': marginal_params,
            'copula_type': self.copula_type,
            'copula_params': copula_params
        }
        
        self.fitted = True
        
        return self.params
    
    def simulate(self, n_samples: int, **kwargs) -> np.ndarray:
        """
        Simulate from the fitted multivariate EVT model
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        **kwargs : dict
            Additional parameters for simulation
            
        Returns:
        --------
        np.ndarray
            Simulated samples of shape (n_samples, n_variables)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Generate samples from copula
        if self.copula_type == 'gaussian':
            correlation_matrix = self.params['copula_params']['correlation_matrix']
            
            # Generate multivariate normal samples
            normal_samples = np.random.multivariate_normal(
                np.zeros(self.n_variables),
                correlation_matrix,
                n_samples
            )
            
            # Transform to uniform using normal CDF
            uniform_samples = stats.norm.cdf(normal_samples)
            
        elif self.copula_type == 't':
            correlation_matrix = self.params['copula_params']['correlation_matrix']
            df = self.params['copula_params']['df']
            
            # Generate multivariate t samples
            normal_samples = np.random.multivariate_normal(
                np.zeros(self.n_variables),
                correlation_matrix,
                n_samples
            )
            
            chi_square_samples = np.random.chisquare(df, n_samples) / df
            t_samples = normal_samples / np.sqrt(chi_square_samples)[:, np.newaxis]
            
            # Transform to uniform using t CDF
            uniform_samples = stats.t.cdf(t_samples, df)
            
        elif self.copula_type in ['clayton', 'gumbel', 'frank']:
            # Simplified implementation for bivariate case
            if self.n_variables == 2:
                theta = self.params['copula_params']['theta']
                
                # Generate uniform samples
                uniform_samples = np.zeros((n_samples, 2))
                uniform_samples[:, 0] = np.random.uniform(0, 1, n_samples)
                
                # Generate conditional samples
                if self.copula_type == 'clayton':
                    u1 = uniform_samples[:, 0]
                    u2 = np.random.uniform(0, 1, n_samples)
                    
                    uniform_samples[:, 1] = ((u1**(-theta) * (u2**(-theta/(1+theta)) - 1) + 1)**(-1/theta))
                    
                else:
                    raise ValueError(f"{self.copula_type} copula simulation not implemented")
            else:
                raise ValueError(f"{self.copula_type} copula not implemented for dimensions > 2")
        else:
            raise ValueError(f"Unsupported copula type: {self.copula_type}")
        
        # Transform uniform samples to original scale using marginal models
        samples = np.zeros_like(uniform_samples)
        
        for i in range(self.n_variables):
            # Use inverse CDF of marginal distribution
            # This is a simplified approach - actual implementation depends on the specific EVT model
            p = uniform_samples[:, i]
            
            # Calculate quantiles
            for j in range(n_samples):
                samples[j, i] = self.marginal_models[i].estimate_var(1 - p[j])
        
        return samples
    
    def estimate_var(self, alpha: float, n_simulations: int = 10000, **kwargs) -> np.ndarray:
        """
        Estimate Value at Risk (VaR) for each variable
        
        Parameters:
        -----------
        alpha : float
            Confidence level (e.g., 0.05 for 95% VaR)
        n_simulations : int, default=10000
            Number of simulations for VaR estimation
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        np.ndarray
            Estimated VaR for each variable
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Simulate from the model
        samples = self.simulate(n_simulations, **kwargs)
        
        # Calculate VaR for each variable
        var = np.zeros(self.n_variables)
        for i in range(self.n_variables):
            var[i] = -np.percentile(samples[:, i], alpha * 100)
        
        return var
    
    def estimate_portfolio_var(self, alpha: float, weights: np.ndarray, 
                             n_simulations: int = 10000, **kwargs) -> float:
        """
        Estimate portfolio Value at Risk (VaR)
        
        Parameters:
        -----------
        alpha : float
            Confidence level (e.g., 0.05 for 95% VaR)
        weights : np.ndarray
            Portfolio weights
        n_simulations : int, default=10000
            Number of simulations for VaR estimation
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        float
            Estimated portfolio VaR
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if len(weights) != self.n_variables:
            raise ValueError(f"Weights has length {len(weights)}, but model has {self.n_variables} variables")
        
        # Simulate from the model
        samples = self.simulate(n_simulations, **kwargs)
        
        # Calculate portfolio returns
        portfolio_returns = samples @ weights
        
        # Calculate VaR
        var = -np.percentile(portfolio_returns, alpha * 100)
        
        return var
