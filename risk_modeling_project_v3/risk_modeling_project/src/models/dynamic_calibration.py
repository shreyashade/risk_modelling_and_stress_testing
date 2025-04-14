"""
Dynamic Calibration Module for Risk Modeling Framework.

This module provides functionality for dynamically calibrating risk models
based on market conditions and regimes.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model


class DynamicCalibration:
    """
    Dynamic Calibration class for risk modeling.
    
    This class provides methods for dynamically calibrating risk models
    based on market conditions and regimes.
    """
    
    def __init__(self):
        """
        Initialize the dynamic calibration model.
        """
        self.returns = None
        self.optimal_window = None
        self.window_metrics = None
        self.min_window = 63  # ~3 months of trading days
        self.max_window = 504  # ~2 years of trading days
        self.window_step = 21  # ~1 month of trading days
        self.calibrated = False
    
    def calibrate(self, historical_data, regime=None):
        """
        Calibrate the dynamic calibration model with historical data.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        regime : dict, optional
            Current market regime information
            
        Returns:
        --------
        int
            Optimal calibration window
        """
        self.returns = historical_data
        
        # Optimize calibration window
        self.optimal_window = self.optimize_window(historical_data, regime)
        
        self.calibrated = True
        
        return self.optimal_window
    
    def optimize_window(self, historical_data=None, regime=None):
        """
        Optimize calibration window based on historical data and regime.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame, optional
            Historical returns data (if None, use self.returns)
        regime : dict, optional
            Current market regime information
            
        Returns:
        --------
        int
            Optimal calibration window
        """
        if historical_data is None:
            if self.returns is None:
                raise ValueError("No historical data provided")
            historical_data = self.returns
        
        # Initialize window metrics
        window_metrics = {}
        
        # Define candidate windows
        windows = range(self.min_window, self.max_window + 1, self.window_step)
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(historical_data.shape[1]) / historical_data.shape[1]
        portfolio_returns = historical_data.dot(portfolio_weights)
        
        # Evaluate each window
        for window in windows:
            # Skip if window is larger than available data
            if window >= len(historical_data):
                continue
            
            # Calculate metrics for this window
            metrics = self._evaluate_window(portfolio_returns, window)
            
            # Store metrics
            window_metrics[window] = metrics
        
        # Store window metrics
        self.window_metrics = window_metrics
        
        # Determine optimal window based on regime
        if regime is not None and 'current_regime' in regime:
            current_regime = regime['current_regime']
            
            if current_regime == 0:  # Low volatility regime
                # Prefer longer windows in low volatility regimes
                optimal_window = self._select_window_by_criterion(window_metrics, 'stability', ascending=False)
            elif current_regime == 2:  # High volatility regime
                # Prefer shorter windows in high volatility regimes
                optimal_window = self._select_window_by_criterion(window_metrics, 'responsiveness', ascending=True)
            else:  # Normal regime
                # Balance stability and responsiveness
                optimal_window = self._select_window_by_criterion(window_metrics, 'combined_score', ascending=False)
        else:
            # Default to balanced approach
            optimal_window = self._select_window_by_criterion(window_metrics, 'combined_score', ascending=False)
        
        # Store optimal window
        self.optimal_window = optimal_window
        
        return optimal_window
    
    def _evaluate_window(self, returns, window):
        """
        Evaluate a calibration window.
        
        Parameters:
        -----------
        returns : pandas.Series
            Portfolio returns
        window : int
            Window size to evaluate
            
        Returns:
        --------
        dict
            Dictionary containing window evaluation metrics
        """
        # Initialize metrics
        metrics = {}
        
        # Calculate number of test periods
        n_test = len(returns) - window
        
        if n_test < 10:
            # Not enough data for reliable evaluation
            metrics['stability'] = 0
            metrics['responsiveness'] = 0
            metrics['var_violation_error'] = float('inf')
            metrics['combined_score'] = 0
            return metrics
        
        # Initialize arrays for storing results
        var_estimates = np.zeros(n_test)
        var_violations = np.zeros(n_test)
        
        # Calculate VaR for each test period
        for i in range(n_test):
            # Get training data
            train_data = returns.iloc[i:i+window]
            
            # Get test data
            test_data = returns.iloc[i+window]
            
            # Calculate VaR (95%)
            var = -np.percentile(train_data, 5)
            
            # Store VaR estimate
            var_estimates[i] = var
            
            # Check for VaR violation
            var_violations[i] = 1 if test_data < -var else 0
        
        # Calculate stability (inverse of VaR estimate volatility)
        var_volatility = np.std(var_estimates)
        metrics['stability'] = 1 / (1 + var_volatility)
        
        # Calculate responsiveness (inverse of window size)
        metrics['responsiveness'] = 1 / window
        
        # Calculate VaR violation rate
        var_violation_rate = np.mean(var_violations)
        
        # Calculate VaR violation error (difference from expected 5%)
        var_violation_error = abs(var_violation_rate - 0.05) / 0.05
        metrics['var_violation_error'] = var_violation_error
        
        # Calculate combined score (higher is better)
        metrics['combined_score'] = (metrics['stability'] + metrics['responsiveness']) / (1 + var_violation_error)
        
        return metrics
    
    def _select_window_by_criterion(self, window_metrics, criterion, ascending=False):
        """
        Select window based on a criterion.
        
        Parameters:
        -----------
        window_metrics : dict
            Dictionary containing window metrics
        criterion : str
            Criterion to use for selection
        ascending : bool, default=False
            Whether to sort in ascending order
            
        Returns:
        --------
        int
            Selected window
        """
        # Extract windows and criterion values
        windows = list(window_metrics.keys())
        values = [window_metrics[w][criterion] for w in windows]
        
        # Sort windows by criterion
        sorted_indices = np.argsort(values)
        
        if not ascending:
            sorted_indices = sorted_indices[::-1]
        
        # Return best window
        return windows[sorted_indices[0]]
    
    def estimate_volatility(self, returns=None, window=None, model_type='garch'):
        """
        Estimate volatility using GARCH or EWMA models.
        
        Parameters:
        -----------
        returns : pandas.Series, optional
            Returns data (if None, use portfolio returns from self.returns)
        window : int, optional
            Calibration window (if None, use optimal window)
        model_type : str, default='garch'
            Volatility model type ('garch', 'ewma')
            
        Returns:
        --------
        pandas.Series
            Estimated volatility
        """
        if returns is None:
            if self.returns is None:
                raise ValueError("No returns data provided")
            
            # Calculate portfolio returns (equal weight)
            portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
            returns = self.returns.dot(portfolio_weights)
        
        if window is None:
            if self.optimal_window is None:
                raise ValueError("No window provided and no optimal window available")
            window = self.optimal_window
        
        # Use only the most recent window of data
        if len(returns) > window:
            returns = returns.iloc[-window:]
        
        if model_type == 'garch':
            # Fit GARCH(1,1) model
            model = arch_model(returns, vol='Garch', p=1, q=1)
            result = model.fit(disp='off')
            
            # Get conditional volatility
            volatility = result.conditional_volatility
            
            # Forecast future volatility
            forecast = result.forecast(horizon=10)
            future_volatility = np.sqrt(forecast.variance.iloc[-1].values)
            
            return volatility, future_volatility
        
        elif model_type == 'ewma':
            # Calculate EWMA volatility
            lambda_param = 0.94  # RiskMetrics default
            
            # Initialize volatility series
            volatility = pd.Series(index=returns.index)
            
            # Set initial volatility
            volatility.iloc[0] = returns.iloc[0] ** 2
            
            # Calculate EWMA volatility
            for i in range(1, len(returns)):
                volatility.iloc[i] = lambda_param * volatility.iloc[i-1] + (1 - lambda_param) * returns.iloc[i-1] ** 2
            
            # Convert to standard deviation
            volatility = np.sqrt(volatility)
            
            # Forecast future volatility (last value)
            future_volatility = volatility.iloc[-1]
            
            return volatility, future_volatility
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def estimate_risk(self, confidence_level=0.95, window=None):
        """
        Estimate risk metrics using dynamic calibration.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        window : int, optional
            Calibration window (if None, use optimal window)
            
        Returns:
        --------
        dict
            Dictionary containing risk metrics
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        if window is None:
            window = self.optimal_window
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        portfolio_returns = self.returns.dot(portfolio_weights)
        
        # Use only the most recent window of data
        if len(portfolio_returns) > window:
            calibration_returns = portfolio_returns.iloc[-window:]
        else:
            calibration_returns = portfolio_returns
        
        # Calculate VaR
        var = -np.percentile(calibration_returns, 100 * (1 - confidence_level))
        
        # Calculate ES
        es = -calibration_returns[calibration_returns <= -var].mean()
        
        # Estimate volatility
        volatility, future_volatility = self.estimate_volatility(portfolio_returns, window)
        
        # Scale risk metrics by volatility ratio
        current_vol = volatility.iloc[-1]
        vol_ratio = future_volatility / current_vol
        
        var_adjusted = var * vol_ratio
        es_adjusted = es * vol_ratio
        
        return {
            'VaR': var_adjusted,
            'ES': es_adjusted,
            'confidence_level': confidence_level,
            'window': window,
            'current_volatility': current_vol,
            'forecast_volatility': future_volatility,
            'volatility_ratio': vol_ratio
        }
    
    def plot_window_metrics(self, figsize=(12, 8)):
        """
        Plot window evaluation metrics.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.window_metrics is None:
            raise ValueError("No window metrics available")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Extract windows and metrics
        windows = list(self.window_metrics.keys())
        stability = [self.window_metrics[w]['stability'] for w in windows]
        responsiveness = [self.window_metrics[w]['responsiveness'] for w in windows]
        var_violation_error = [self.window_metrics[w]['var_violation_error'] for w in windows]
        combined_score = [self.window_metrics[w]['combined_score'] for w in windows]
        
        # Plot stability
        axes[0, 0].plot(windows, stability, 'o-')
        axes[0, 0].set_xlabel('Window Size')
        axes[0, 0].set_ylabel('Stability')
        axes[0, 0].set_title('Stability vs. Window Size')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot responsiveness
        axes[0, 1].plot(windows, responsiveness, 'o-')
        axes[0, 1].set_xlabel('Window Size')
        axes[0, 1].set_ylabel('Responsiveness')
        axes[0, 1].set_title('Responsiveness vs. Window Size')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot VaR violation error
        axes[1, 0].plot(windows, var_violation_error, 'o-')
        axes[1, 0].set_xlabel('Window Size')
        axes[1, 0].set_ylabel('VaR Violation Error')
        axes[1, 0].set_title('VaR Violation Error vs. Window Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot combined score
        axes[1, 1].plot(windows, combined_score, 'o-')
        axes[1, 1].set_xlabel('Window Size')
        axes[1, 1].set_ylabel('Combined Score')
        axes[1, 1].set_title('Combined Score vs. Window Size')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add optimal window line
        for ax in axes.flat:
            ax.axvline(x=self.optimal_window, color='red', linestyle='--', label=f'Optimal Window: {self.optimal_window}')
            ax.legend()
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_volatility(self, returns=None, window=None, model_type='garch', figsize=(12, 6)):
        """
        Plot estimated volatility.
        
        Parameters:
        -----------
        returns : pandas.Series, optional
            Returns data (if None, use portfolio returns from self.returns)
        window : int, optional
            Calibration window (if None, use optimal window)
        model_type : str, default='garch'
            Volatility model type ('garch', 'ewma')
        figsize : tuple, default=(12, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if returns is None:
            if self.returns is None:
                raise ValueError("No returns data provided")
            
            # Calculate portfolio returns (equal weight)
            portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
            returns = self.returns.dot(portfolio_weights)
        
        if window is None:
            if self.optimal_window is None:
                raise ValueError("No window provided and no optimal window available")
            window = self.optimal_window
        
        # Estimate volatility
        volatility, future_volatility = self.estimate_volatility(returns, window, model_type)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot returns
        ax.plot(returns.index, returns, color='blue', alpha=0.5, label='Returns')
        
        # Create twin axis for volatility
        ax2 = ax.twinx()
        
        # Plot volatility
        ax2.plot(volatility.index, volatility, color='red', label='Volatility')
        
        # Add forecast volatility
        ax2.axhline(y=future_volatility, color='darkred', linestyle='--', label=f'Forecast Volatility: {future_volatility:.4f}')
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Returns', color='blue')
        ax2.set_ylabel('Volatility', color='red')
        ax.set_title(f'Returns and Volatility ({model_type.upper()} Model)')
        
        # Set y-axis colors
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
