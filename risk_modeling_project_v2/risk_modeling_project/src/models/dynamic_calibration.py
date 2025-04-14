"""
Dynamic Calibration Windows for Risk Modeling Framework

This module implements various dynamic calibration window approaches to adapt
risk models to changing market conditions, improving prediction accuracy.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Union, Optional, Tuple, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DynamicCalibrationWindow:
    """Base class for dynamic calibration window approaches"""
    
    def __init__(self, min_window: int = 63, max_window: int = 504):
        """
        Initialize the dynamic calibration window.
        
        Parameters:
        -----------
        min_window : int, default=63
            Minimum window size (trading days, ~3 months)
        max_window : int, default=504
            Maximum window size (trading days, ~2 years)
        """
        self.min_window = min_window
        self.max_window = max_window
        self.optimal_window = None
        self.fitted = False
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Determine the optimal calibration window.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_calibration_window(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Get the optimal calibration window for the current data.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
            
        Returns:
        --------
        pd.DataFrame
            Returns data in the optimal calibration window
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting calibration window")
        
        window_size = self.get_window_size(returns)
        
        if len(returns) <= window_size:
            return returns
        else:
            return returns.iloc[-window_size:]
    
    def get_window_size(self, returns: pd.DataFrame) -> int:
        """
        Get the optimal window size for the current data.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
            
        Returns:
        --------
        int
            Optimal window size
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def plot_window_size(self, returns: pd.DataFrame) -> plt.Figure:
        """
        Plot the dynamic window size over time.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with window size visualization
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting window size")
        
        # Calculate window sizes for each point in time
        window_sizes = []
        dates = []
        
        for i in range(self.min_window, len(returns)):
            current_returns = returns.iloc[:i]
            window_size = self.get_window_size(current_returns)
            window_sizes.append(window_size)
            dates.append(returns.index[i-1])
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot window sizes
        axes[0].plot(dates, window_sizes, 'b-')
        axes[0].set_title('Dynamic Calibration Window Size')
        axes[0].set_ylabel('Window Size (days)')
        axes[0].grid(True)
        
        # Plot returns volatility
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
            rolling_vol = pd.Series(
                portfolio_returns, 
                index=returns.index
            ).rolling(window=21).std() * np.sqrt(252)  # Annualized
        else:
            rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)  # Annualized
        
        axes[1].plot(rolling_vol.index, rolling_vol, 'r-')
        axes[1].set_title('Market Volatility')
        axes[1].set_ylabel('Annualized Volatility')
        axes[1].set_xlabel('Date')
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig


class VolatilityAdjustedWindow(DynamicCalibrationWindow):
    """
    Volatility-adjusted calibration window.
    
    This approach adjusts the window size based on market volatility,
    using shorter windows during high volatility periods and longer
    windows during low volatility periods.
    """
    
    def __init__(self, 
                 min_window: int = 63, 
                 max_window: int = 504,
                 vol_lookback: int = 21,
                 vol_percentile_low: float = 0.25,
                 vol_percentile_high: float = 0.75):
        """
        Initialize the volatility-adjusted window.
        
        Parameters:
        -----------
        min_window : int, default=63
            Minimum window size (trading days, ~3 months)
        max_window : int, default=504
            Maximum window size (trading days, ~2 years)
        vol_lookback : int, default=21
            Lookback period for volatility calculation (trading days)
        vol_percentile_low : float, default=0.25
            Percentile for low volatility threshold
        vol_percentile_high : float, default=0.75
            Percentile for high volatility threshold
        """
        super().__init__(min_window, max_window)
        self.vol_lookback = vol_lookback
        self.vol_percentile_low = vol_percentile_low
        self.vol_percentile_high = vol_percentile_high
        self.vol_threshold_low = None
        self.vol_threshold_high = None
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Determine volatility thresholds for window adjustment.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info("Fitting volatility-adjusted calibration window")
        
        # Calculate rolling volatility
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
            rolling_vol = pd.Series(
                portfolio_returns, 
                index=returns.index
            ).rolling(window=self.vol_lookback).std() * np.sqrt(252)  # Annualized
        else:
            rolling_vol = returns.rolling(window=self.vol_lookback).std() * np.sqrt(252)  # Annualized
        
        # Determine volatility thresholds
        self.vol_threshold_low = np.nanpercentile(rolling_vol, self.vol_percentile_low * 100)
        self.vol_threshold_high = np.nanpercentile(rolling_vol, self.vol_percentile_high * 100)
        
        logger.info(f"Volatility thresholds: Low={self.vol_threshold_low:.2%}, High={self.vol_threshold_high:.2%}")
        
        self.fitted = True
    
    def get_window_size(self, returns: pd.DataFrame) -> int:
        """
        Get the optimal window size based on current volatility.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
            
        Returns:
        --------
        int
            Optimal window size
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting window size")
        
        # Calculate current volatility
        if len(returns) < self.vol_lookback:
            # Not enough data for volatility calculation
            return self.min_window
        
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.iloc[-self.vol_lookback:].values @ weights
            current_vol = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        else:
            current_vol = returns.iloc[-self.vol_lookback:].std().iloc[0] * np.sqrt(252)  # Annualized
        
        # Adjust window size based on volatility
        if current_vol <= self.vol_threshold_low:
            # Low volatility: use longer window
            window_size = self.max_window
        elif current_vol >= self.vol_threshold_high:
            # High volatility: use shorter window
            window_size = self.min_window
        else:
            # Medium volatility: interpolate window size
            vol_range = self.vol_threshold_high - self.vol_threshold_low
            vol_ratio = (current_vol - self.vol_threshold_low) / vol_range
            window_range = self.max_window - self.min_window
            window_size = int(self.max_window - vol_ratio * window_range)
        
        # Ensure window size is within bounds
        window_size = min(max(window_size, self.min_window), self.max_window)
        
        # Ensure window size doesn't exceed available data
        window_size = min(window_size, len(returns))
        
        return window_size


class GARCHAdjustedWindow(DynamicCalibrationWindow):
    """
    GARCH-adjusted calibration window.
    
    This approach uses GARCH volatility forecasts to adjust the window size,
    providing more responsive adaptation to changing market conditions.
    """
    
    def __init__(self, 
                 min_window: int = 63, 
                 max_window: int = 504,
                 garch_p: int = 1,
                 garch_q: int = 1,
                 forecast_horizon: int = 5,
                 vol_percentile_low: float = 0.25,
                 vol_percentile_high: float = 0.75):
        """
        Initialize the GARCH-adjusted window.
        
        Parameters:
        -----------
        min_window : int, default=63
            Minimum window size (trading days, ~3 months)
        max_window : int, default=504
            Maximum window size (trading days, ~2 years)
        garch_p : int, default=1
            GARCH model p parameter (lag order of GARCH terms)
        garch_q : int, default=1
            GARCH model q parameter (lag order of ARCH terms)
        forecast_horizon : int, default=5
            Forecast horizon for volatility prediction (trading days)
        vol_percentile_low : float, default=0.25
            Percentile for low volatility threshold
        vol_percentile_high : float, default=0.75
            Percentile for high volatility threshold
        """
        super().__init__(min_window, max_window)
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.forecast_horizon = forecast_horizon
        self.vol_percentile_low = vol_percentile_low
        self.vol_percentile_high = vol_percentile_high
        self.vol_threshold_low = None
        self.vol_threshold_high = None
        self.garch_model = None
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit GARCH model and determine volatility thresholds.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info("Fitting GARCH-adjusted calibration window")
        
        # Prepare portfolio returns if multiple assets
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
        else:
            portfolio_returns = returns.values.flatten()
        
        # Fit GARCH model
        self.garch_model = arch_model(
            portfolio_returns * 100,  # Scale returns for numerical stability
            vol='GARCH',
            p=self.garch_p,
            q=self.garch_q,
            rescale=False
        )
        
        try:
            self.garch_fit = self.garch_model.fit(disp='off')
            logger.info("GARCH model fitted successfully")
        except Exception as e:
            logger.warning(f"GARCH model fitting failed: {e}")
            logger.info("Using simplified approach")
            self.garch_model = None
        
        # Calculate historical volatility forecasts
        vol_forecasts = []
        
        if self.garch_model is not None:
            # Use rolling window to generate historical forecasts
            min_fit_window = 252  # Minimum data for GARCH fitting
            
            if len(portfolio_returns) < min_fit_window + 10:
                logger.warning("Insufficient data for historical GARCH forecasts")
                # Use simplified approach
                self.garch_model = None
            else:
                for i in range(min_fit_window, len(portfolio_returns), 10):
                    try:
                        # Fit model on data up to point i
                        garch_fit = self.garch_model.fit(
                            last_obs=i,
                            disp='off'
                        )
                        
                        # Forecast volatility
                        forecast = garch_fit.forecast(horizon=self.forecast_horizon)
                        vol_forecast = np.sqrt(forecast.variance.iloc[-1, self.forecast_horizon-1]) / 100
                        vol_forecasts.append(vol_forecast)
                    except Exception as e:
                        logger.debug(f"GARCH forecast failed at index {i}: {e}")
                        # Use last successful forecast or default value
                        if vol_forecasts:
                            vol_forecasts.append(vol_forecasts[-1])
                        else:
                            vol_forecasts.append(np.std(portfolio_returns[:i]) * np.sqrt(252))
        
        if not vol_forecasts or self.garch_model is None:
            # Use rolling volatility as fallback
            rolling_vol = pd.Series(portfolio_returns).rolling(window=21).std() * np.sqrt(252)
            vol_forecasts = rolling_vol.dropna().values
        
        # Determine volatility thresholds
        self.vol_threshold_low = np.percentile(vol_forecasts, self.vol_percentile_low * 100)
        self.vol_threshold_high = np.percentile(vol_forecasts, self.vol_percentile_high * 100)
        
        logger.info(f"Volatility thresholds: Low={self.vol_threshold_low:.2%}, High={self.vol_threshold_high:.2%}")
        
        self.fitted = True
    
    def get_window_size(self, returns: pd.DataFrame) -> int:
        """
        Get the optimal window size based on GARCH volatility forecast.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
            
        Returns:
        --------
        int
            Optimal window size
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting window size")
        
        # Prepare portfolio returns if multiple assets
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
        else:
            portfolio_returns = returns.values.flatten()
        
        # Forecast volatility
        if self.garch_model is not None and len(portfolio_returns) >= 252:
            try:
                # Fit model on current data
                garch_fit = self.garch_model.fit(
                    data=portfolio_returns * 100,
                    disp='off'
                )
                
                # Forecast volatility
                forecast = garch_fit.forecast(horizon=self.forecast_horizon)
                forecast_vol = np.sqrt(forecast.variance.iloc[-1, self.forecast_horizon-1]) / 100
            except Exception as e:
                logger.warning(f"GARCH forecast failed: {e}")
                # Use rolling volatility as fallback
                forecast_vol = np.std(portfolio_returns[-21:]) * np.sqrt(252)
        else:
            # Use rolling volatility as fallback
            forecast_vol = np.std(portfolio_returns[-21:]) * np.sqrt(252)
        
        # Adjust window size based on volatility forecast
        if forecast_vol <= self.vol_threshold_low:
            # Low volatility: use longer window
            window_size = self.max_window
        elif forecast_vol >= self.vol_threshold_high:
            # High volatility: use shorter window
            window_size = self.min_window
        else:
            # Medium volatility: interpolate window size
            vol_range = self.vol_threshold_high - self.vol_threshold_low
            vol_ratio = (forecast_vol - self.vol_threshold_low) / vol_range
            window_range = self.max_window - self.min_window
            window_size = int(self.max_window - vol_ratio * window_range)
        
        # Ensure window size is within bounds
        window_size = min(max(window_size, self.min_window), self.max_window)
        
        # Ensure window size doesn't exceed available data
        window_size = min(window_size, len(returns))
        
        return window_size


class TimeWeightedCalibration:
    """
    Time-weighted calibration for risk models.
    
    This approach applies time-based weights to observations, giving more
    importance to recent data while still utilizing the full history.
    """
    
    def __init__(self, 
                 half_life: int = 63,
                 min_weight: float = 0.1):
        """
        Initialize the time-weighted calibration.
        
        Parameters:
        -----------
        half_life : int, default=63
            Half-life parameter for exponential weighting (trading days)
        min_weight : float, default=0.1
            Minimum weight for oldest observations
        """
        self.half_life = half_life
        self.min_weight = min_weight
        self.decay_factor = np.exp(np.log(0.5) / half_life)
    
    def get_weights(self, n_obs: int) -> np.ndarray:
        """
        Calculate time-based weights for observations.
        
        Parameters:
        -----------
        n_obs : int
            Number of observations
            
        Returns:
        --------
        np.ndarray
            Array of weights for each observation
        """
        # Calculate raw exponential weights
        raw_weights = np.power(self.decay_factor, np.arange(n_obs-1, -1, -1))
        
        # Apply minimum weight
        if self.min_weight > 0:
            min_raw_weight = np.min(raw_weights)
            if min_raw_weight < self.min_weight:
                # Scale weights to ensure minimum weight
                scale_factor = (1 - self.min_weight) / (1 - min_raw_weight)
                weights = self.min_weight + scale_factor * (raw_weights - min_raw_weight)
            else:
                weights = raw_weights
        else:
            weights = raw_weights
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        return weights
    
    def weighted_mean(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate time-weighted mean.
        
        Parameters:
        -----------
        data : np.ndarray
            Data array with shape (n_obs, n_features)
            
        Returns:
        --------
        np.ndarray
            Weighted mean with shape (n_features,)
        """
        weights = self.get_weights(len(data))
        
        # Reshape weights for broadcasting
        weights_reshaped = weights.reshape(-1, 1)
        
        # Calculate weighted mean
        weighted_mean = np.sum(data * weights_reshaped, axis=0)
        
        return weighted_mean
    
    def weighted_cov(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate time-weighted covariance matrix.
        
        Parameters:
        -----------
        data : np.ndarray
            Data array with shape (n_obs, n_features)
            
        Returns:
        --------
        np.ndarray
            Weighted covariance matrix with shape (n_features, n_features)
        """
        weights = self.get_weights(len(data))
        
        # Calculate weighted mean
        weighted_mean = self.weighted_mean(data)
        
        # Center the data
        centered_data = data - weighted_mean
        
        # Calculate weighted covariance
        weights_reshaped = weights.reshape(-1, 1)
        weighted_cov = np.dot(centered_data.T * weights_reshaped.T, centered_data)
        
        return weighted_cov
    
    def weighted_var(self, data: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate time-weighted Value-at-Risk.
        
        Parameters:
        -----------
        data : np.ndarray
            Data array with shape (n_obs,)
        confidence_level : float, default=0.95
            Confidence level for VaR
            
        Returns:
        --------
        float
            Weighted VaR
        """
        weights = self.get_weights(len(data))
        
        # Sort data and weights together
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # Calculate cumulative weights
        cumulative_weights = np.cumsum(sorted_weights)
        
        # Find VaR threshold
        var_index = np.searchsorted(cumulative_weights, 1 - confidence_level)
        
        if var_index >= len(sorted_data):
            var_index = len(sorted_data) - 1
        
        var = sorted_data[var_index]
        
        return var
    
    def weighted_es(self, data: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate time-weighted Expected Shortfall.
        
        Parameters:
        -----------
        data : np.ndarray
            Data array with shape (n_obs,)
        confidence_level : float, default=0.95
            Confidence level for ES
            
        Returns:
        --------
        float
            Weighted ES
        """
        weights = self.get_weights(len(data))
        
        # Sort data and weights together
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # Calculate cumulative weights
        cumulative_weights = np.cumsum(sorted_weights)
        
        # Find VaR threshold
        var_index = np.searchsorted(cumulative_weights, 1 - confidence_level)
        
        if var_index >= len(sorted_data):
            var_index = len(sorted_data) - 1
        
        # Calculate ES using weights
        tail_data = sorted_data[:var_index+1]
        tail_weights = sorted_weights[:var_index+1]
        
        # Normalize tail weights
        normalized_tail_weights = tail_weights / np.sum(tail_weights)
        
        # Calculate weighted ES
        es = np.sum(tail_data * normalized_tail_weights)
        
        return es
    
    def plot_weights(self, n_obs: int = 252) -> plt.Figure:
        """
        Plot time-based weights.
        
        Parameters:
        -----------
        n_obs : int, default=252
            Number of observations
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with weights visualization
        """
        weights = self.get_weights(n_obs)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot weights
        ax.plot(np.arange(n_obs), weights, 'b-')
        ax.set_title(f'Time-Weighted Calibration (Half-Life: {self.half_life} days)')
        ax.set_xlabel('Days Ago')
        ax.set_ylabel('Weight')
        ax.grid(True)
        
        # Add half-life marker
        ax.axvline(x=self.half_life, color='r', linestyle='--', 
                   label=f'Half-Life: {self.half_life} days')
        
        # Add cumulative weight line
        cumulative_weights = np.cumsum(weights)
        ax_cum = ax.twinx()
        ax_cum.plot(np.arange(n_obs), cumulative_weights, 'g-', alpha=0.5,
                   label='Cumulative Weight')
        ax_cum.set_ylabel('Cumulative Weight')
        ax_cum.set_ylim([0, 1.05])
        
        # Add legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_cum.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='best')
        
        plt.tight_layout()
        return fig


class AdaptiveCalibration:
    """
    Adaptive calibration for risk models.
    
    This class combines dynamic window sizing and time-weighted calibration
    to provide optimal adaptation to changing market conditions.
    """
    
    def __init__(self, 
                 window_method: str = 'volatility',
                 min_window: int = 63,
                 max_window: int = 504,
                 half_life: int = 63,
                 min_weight: float = 0.1,
                 garch_p: int = 1,
                 garch_q: int = 1):
        """
        Initialize the adaptive calibration.
        
        Parameters:
        -----------
        window_method : str, default='volatility'
            Method for dynamic window sizing ('volatility' or 'garch')
        min_window : int, default=63
            Minimum window size (trading days, ~3 months)
        max_window : int, default=504
            Maximum window size (trading days, ~2 years)
        half_life : int, default=63
            Half-life parameter for exponential weighting (trading days)
        min_weight : float, default=0.1
            Minimum weight for oldest observations
        garch_p : int, default=1
            GARCH model p parameter (lag order of GARCH terms)
        garch_q : int, default=1
            GARCH model q parameter (lag order of ARCH terms)
        """
        self.window_method = window_method
        
        # Initialize dynamic window
        if window_method == 'volatility':
            self.window_model = VolatilityAdjustedWindow(
                min_window=min_window,
                max_window=max_window
            )
        elif window_method == 'garch':
            self.window_model = GARCHAdjustedWindow(
                min_window=min_window,
                max_window=max_window,
                garch_p=garch_p,
                garch_q=garch_q
            )
        else:
            raise ValueError(f"Unknown window method: {window_method}")
        
        # Initialize time weighting
        self.time_weighting = TimeWeightedCalibration(
            half_life=half_life,
            min_weight=min_weight
        )
        
        self.fitted = False
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the adaptive calibration model.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info(f"Fitting adaptive calibration with {self.window_method} window method")
        
        # Fit dynamic window model
        self.window_model.fit(returns)
        
        self.fitted = True
        logger.info("Adaptive calibration fitted successfully")
    
    def get_calibration_window(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Get the optimal calibration window for the current data.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
            
        Returns:
        --------
        pd.DataFrame
            Returns data in the optimal calibration window
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting calibration window")
        
        return self.window_model.get_calibration_window(returns)
    
    def get_weighted_returns(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Get time-weighted returns for calibration.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
            
        Returns:
        --------
        Tuple[pd.DataFrame, np.ndarray]
            Calibration window returns and corresponding weights
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting weighted returns")
        
        # Get optimal calibration window
        window_returns = self.get_calibration_window(returns)
        
        # Calculate time-based weights
        weights = self.time_weighting.get_weights(len(window_returns))
        
        return window_returns, weights
    
    def estimate_var(self, 
                     returns: pd.DataFrame, 
                     weights: np.ndarray, 
                     confidence_level: float = 0.95,
                     method: str = 'parametric') -> float:
        """
        Estimate Value-at-Risk using adaptive calibration.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for VaR
        method : str, default='parametric'
            VaR estimation method ('parametric', 'historical', or 'weighted')
            
        Returns:
        --------
        float
            VaR estimate
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before estimating VaR")
        
        # Get calibration window and time weights
        window_returns, time_weights = self.get_weighted_returns(returns)
        
        if method == 'parametric':
            # Calculate weighted mean and covariance
            returns_array = window_returns.values
            weighted_mean = self.time_weighting.weighted_mean(returns_array)
            weighted_cov = self.time_weighting.weighted_cov(returns_array)
            
            # Calculate portfolio parameters
            portfolio_mean = np.dot(weights, weighted_mean)
            portfolio_var = np.dot(weights, np.dot(weighted_cov, weights))
            portfolio_std = np.sqrt(portfolio_var)
            
            # Calculate VaR
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(portfolio_mean + z_score * portfolio_std)
            
        elif method == 'historical':
            # Calculate portfolio returns
            portfolio_returns = window_returns.values @ weights
            
            # Calculate historical VaR
            var = -np.percentile(portfolio_returns, 100 * (1 - confidence_level))
            
        elif method == 'weighted':
            # Calculate portfolio returns
            portfolio_returns = window_returns.values @ weights
            
            # Calculate weighted VaR
            var = -self.time_weighting.weighted_var(portfolio_returns, confidence_level)
            
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        return var
    
    def estimate_es(self, 
                    returns: pd.DataFrame, 
                    weights: np.ndarray, 
                    confidence_level: float = 0.95,
                    method: str = 'parametric') -> float:
        """
        Estimate Expected Shortfall using adaptive calibration.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for ES
        method : str, default='parametric'
            ES estimation method ('parametric', 'historical', or 'weighted')
            
        Returns:
        --------
        float
            ES estimate
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before estimating ES")
        
        # Get calibration window and time weights
        window_returns, time_weights = self.get_weighted_returns(returns)
        
        if method == 'parametric':
            # Calculate weighted mean and covariance
            returns_array = window_returns.values
            weighted_mean = self.time_weighting.weighted_mean(returns_array)
            weighted_cov = self.time_weighting.weighted_cov(returns_array)
            
            # Calculate portfolio parameters
            portfolio_mean = np.dot(weights, weighted_mean)
            portfolio_var = np.dot(weights, np.dot(weighted_cov, weights))
            portfolio_std = np.sqrt(portfolio_var)
            
            # Calculate VaR
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(portfolio_mean + z_score * portfolio_std)
            
            # Calculate ES adjustment for normal distribution
            es_adjustment = portfolio_std * stats.norm.pdf(z_score) / (1 - confidence_level)
            es = var + es_adjustment
            
        elif method == 'historical':
            # Calculate portfolio returns
            portfolio_returns = window_returns.values @ weights
            
            # Calculate historical ES
            var_threshold = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
            es = -np.mean(portfolio_returns[portfolio_returns <= var_threshold])
            
        elif method == 'weighted':
            # Calculate portfolio returns
            portfolio_returns = window_returns.values @ weights
            
            # Calculate weighted ES
            es = -self.time_weighting.weighted_es(portfolio_returns, confidence_level)
            
        else:
            raise ValueError(f"Unknown ES method: {method}")
        
        return es
    
    def plot_calibration(self, returns: pd.DataFrame, asset_prices: Optional[pd.DataFrame] = None) -> plt.Figure:
        """
        Plot the adaptive calibration characteristics.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        asset_prices : pd.DataFrame, optional
            Historical asset prices for visualization
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with calibration visualization
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting calibration")
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot asset prices or cumulative returns
        if asset_prices is not None:
            for col in asset_prices.columns:
                axes[0].plot(asset_prices.index, asset_prices[col], label=col)
            axes[0].set_ylabel('Price')
        else:
            cum_returns = (1 + returns).cumprod()
            for col in cum_returns.columns:
                axes[0].plot(cum_returns.index, cum_returns[col], label=col)
            axes[0].set_ylabel('Cumulative Return')
        
        axes[0].set_title('Asset Performance and Calibration Windows')
        axes[0].legend(loc='upper left')
        axes[0].grid(True)
        
        # Calculate window sizes for each point in time
        window_sizes = []
        dates = []
        
        min_data = self.window_model.min_window
        
        for i in range(min_data, len(returns)):
            current_returns = returns.iloc[:i]
            window_size = self.window_model.get_window_size(current_returns)
            window_sizes.append(window_size)
            dates.append(returns.index[i-1])
        
        # Plot window sizes
        axes[1].plot(dates, window_sizes, 'b-')
        axes[1].set_ylabel('Window Size (days)')
        axes[1].set_title('Dynamic Calibration Window Size')
        axes[1].grid(True)
        
        # Plot volatility
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
            rolling_vol = pd.Series(
                portfolio_returns, 
                index=returns.index
            ).rolling(window=21).std() * np.sqrt(252)  # Annualized
        else:
            rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)  # Annualized
        
        axes[2].plot(rolling_vol.index, rolling_vol, 'r-')
        axes[2].set_ylabel('Annualized Volatility')
        axes[2].set_xlabel('Date')
        axes[2].set_title('Market Volatility')
        axes[2].grid(True)
        
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
    
    # Initialize and fit volatility-adjusted window
    vol_window = VolatilityAdjustedWindow()
    vol_window.fit(returns)
    
    # Plot window size
    vol_fig = vol_window.plot_window_size(returns)
    vol_fig.savefig('volatility_window.png')
    
    # Initialize and fit GARCH-adjusted window
    garch_window = GARCHAdjustedWindow()
    garch_window.fit(returns)
    
    # Plot window size
    garch_fig = garch_window.plot_window_size(returns)
    garch_fig.savefig('garch_window.png')
    
    # Initialize and plot time-weighted calibration
    time_weighting = TimeWeightedCalibration()
    weight_fig = time_weighting.plot_weights()
    weight_fig.savefig('time_weights.png')
    
    # Initialize and fit adaptive calibration
    adaptive_cal = AdaptiveCalibration()
    adaptive_cal.fit(returns)
    
    # Plot calibration
    cal_fig = adaptive_cal.plot_calibration(returns, prices)
    cal_fig.savefig('adaptive_calibration.png')
    
    # Estimate VaR and ES
    portfolio_weights = np.ones(len(tickers)) / len(tickers)
    
    var_parametric = adaptive_cal.estimate_var(returns, portfolio_weights, method='parametric')
    var_historical = adaptive_cal.estimate_var(returns, portfolio_weights, method='historical')
    var_weighted = adaptive_cal.estimate_var(returns, portfolio_weights, method='weighted')
    
    es_parametric = adaptive_cal.estimate_es(returns, portfolio_weights, method='parametric')
    es_historical = adaptive_cal.estimate_es(returns, portfolio_weights, method='historical')
    es_weighted = adaptive_cal.estimate_es(returns, portfolio_weights, method='weighted')
    
    print("VaR Estimates (95%):")
    print(f"Parametric: {var_parametric:.2%}")
    print(f"Historical: {var_historical:.2%}")
    print(f"Weighted: {var_weighted:.2%}")
    
    print("\nES Estimates (95%):")
    print(f"Parametric: {es_parametric:.2%}")
    print(f"Historical: {es_historical:.2%}")
    print(f"Weighted: {es_weighted:.2%}")
