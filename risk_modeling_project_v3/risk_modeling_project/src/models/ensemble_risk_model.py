"""
Ensemble Risk Model for Risk Modeling Framework.

This module provides functionality for combining multiple risk models into an ensemble.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class EnsembleRiskModel:
    """
    Ensemble Risk Model class for combining multiple risk models.
    
    This class provides methods for combining predictions from multiple risk models
    using various ensemble techniques such as simple averaging, weighted averaging,
    and dynamic weighting based on recent performance.
    """
    
    def __init__(self, models=None):
        """
        Initialize the Ensemble Risk Model.
        
        Parameters:
        -----------
        models : list, optional
            List of risk models to include in the ensemble
        """
        self.models = models if models is not None else []
        self.weights = None
        self.dynamic_weights = False
        self.performance_window = 63  # Default to 3 months of daily data
        self.performance_history = None
        self.calibrated = False
    
    def add_model(self, model):
        """
        Add a model to the ensemble.
        
        Parameters:
        -----------
        model : object
            Risk model to add to the ensemble
        """
        self.models.append(model)
    
    def set_weights(self, weights):
        """
        Set weights for the ensemble models.
        
        Parameters:
        -----------
        weights : array-like
            Weights for each model in the ensemble
        """
        if len(weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights
        weights = np.array(weights)
        self.weights = weights / weights.sum()
    
    def calibrate(self, historical_data, weights_method='equal', 
                  performance_metric='mse', dynamic_weights=False,
                  performance_window=63):
        """
        Calibrate the ensemble model with historical data.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        weights_method : str, default='equal'
            Method for setting weights ('equal', 'performance', or 'optimal')
        performance_metric : str, default='mse'
            Metric for evaluating model performance ('mse', 'mae', or 'directional')
        dynamic_weights : bool, default=False
            Whether to use dynamic weights based on recent performance
        performance_window : int, default=63
            Window size for calculating performance-based weights
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
        
        # Calibrate each model
        for model in self.models:
            if hasattr(model, 'calibrate'):
                model.calibrate(historical_data)
        
        # Set weights
        if weights_method == 'equal':
            self.set_weights(np.ones(len(self.models)))
        elif weights_method == 'performance':
            self._set_performance_weights(historical_data, performance_metric)
        elif weights_method == 'optimal':
            self._set_optimal_weights(historical_data)
        else:
            raise ValueError(f"Unknown weights method: {weights_method}")
        
        # Set dynamic weights parameters
        self.dynamic_weights = dynamic_weights
        self.performance_window = performance_window
        
        # Initialize performance history
        if self.dynamic_weights:
            self.performance_history = pd.DataFrame(
                index=historical_data.index[-performance_window:],
                columns=[f'model_{i}' for i in range(len(self.models))],
                data=np.zeros((performance_window, len(self.models)))
            )
        
        self.calibrated = True
    
    def _set_performance_weights(self, historical_data, performance_metric):
        """
        Set weights based on model performance.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        performance_metric : str
            Metric for evaluating model performance
        """
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(historical_data.shape[1]) / historical_data.shape[1]
        portfolio_returns = historical_data.dot(portfolio_weights)
        
        # Calculate performance for each model
        performance = np.zeros(len(self.models))
        
        for i, model in enumerate(self.models):
            # Create dummy portfolio
            portfolio = type('Portfolio', (), {'weights': pd.Series(portfolio_weights)})
            
            # Get risk metrics
            if hasattr(model, 'estimate_risk'):
                risk_metrics = model.estimate_risk(portfolio)
                
                # Calculate performance based on metric
                if performance_metric == 'mse':
                    # Mean squared error of VaR
                    var = risk_metrics['var']
                    actual_losses = -portfolio_returns[portfolio_returns < 0]
                    if len(actual_losses) > 0:
                        performance[i] = 1 / (mean_squared_error([var] * len(actual_losses), actual_losses) + 1e-10)
                    else:
                        performance[i] = 0
                elif performance_metric == 'mae':
                    # Mean absolute error of VaR
                    var = risk_metrics['var']
                    actual_losses = -portfolio_returns[portfolio_returns < 0]
                    if len(actual_losses) > 0:
                        performance[i] = 1 / (np.mean(np.abs([var] * len(actual_losses) - actual_losses)) + 1e-10)
                    else:
                        performance[i] = 0
                elif performance_metric == 'directional':
                    # Directional accuracy
                    var = risk_metrics['var']
                    var_violations = (portfolio_returns < -var).mean()
                    target_violation_rate = 0.05  # For 95% VaR
                    performance[i] = 1 / (np.abs(var_violations - target_violation_rate) + 1e-10)
                else:
                    raise ValueError(f"Unknown performance metric: {performance_metric}")
        
        # Set weights based on performance
        if np.sum(performance) > 0:
            self.set_weights(performance)
        else:
            # If all models perform poorly, use equal weights
            self.set_weights(np.ones(len(self.models)))
    
    def _set_optimal_weights(self, historical_data):
        """
        Set optimal weights to minimize portfolio risk.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        """
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(historical_data.shape[1]) / historical_data.shape[1]
        portfolio_returns = historical_data.dot(portfolio_weights)
        
        # Create dummy portfolio
        portfolio = type('Portfolio', (), {'weights': pd.Series(portfolio_weights)})
        
        # Get risk metrics for each model
        var_predictions = np.zeros((len(historical_data), len(self.models)))
        
        for i, model in enumerate(self.models):
            if hasattr(model, 'estimate_risk'):
                # Get initial risk metrics
                risk_metrics = model.estimate_risk(portfolio)
                var_predictions[0, i] = risk_metrics['var']
                
                # Get risk metrics for each time step
                for t in range(1, len(historical_data)):
                    # Update model with new data
                    if hasattr(model, 'update'):
                        model.update(historical_data.iloc[t:t+1])
                    
                    # Get risk metrics
                    risk_metrics = model.estimate_risk(portfolio)
                    var_predictions[t, i] = risk_metrics['var']
        
        # Calculate optimal weights using linear regression
        actual_losses = -portfolio_returns[portfolio_returns < 0].values
        
        if len(actual_losses) > 0:
            # Extract corresponding predictions
            negative_return_indices = np.where(portfolio_returns < 0)[0]
            predictions = var_predictions[negative_return_indices]
            
            # Fit linear regression without intercept
            reg = LinearRegression(fit_intercept=False)
            reg.fit(predictions, actual_losses)
            
            # Get weights from regression coefficients
            weights = np.maximum(reg.coef_, 0)  # Ensure non-negative weights
            
            # Set weights
            if np.sum(weights) > 0:
                self.set_weights(weights)
            else:
                # If all weights are zero, use equal weights
                self.set_weights(np.ones(len(self.models)))
        else:
            # If no negative returns, use equal weights
            self.set_weights(np.ones(len(self.models)))
    
    def update_weights(self, new_data):
        """
        Update weights based on recent performance.
        
        Parameters:
        -----------
        new_data : pandas.DataFrame
            New returns data
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        if not self.dynamic_weights:
            return
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(new_data.shape[1]) / new_data.shape[1]
        portfolio_returns = new_data.dot(portfolio_weights)
        
        # Create dummy portfolio
        portfolio = type('Portfolio', (), {'weights': pd.Series(portfolio_weights)})
        
        # Calculate performance for each model
        performance = np.zeros(len(self.models))
        
        for i, model in enumerate(self.models):
            # Update model with new data
            if hasattr(model, 'update'):
                model.update(new_data)
            
            # Get risk metrics
            if hasattr(model, 'estimate_risk'):
                risk_metrics = model.estimate_risk(portfolio)
                
                # Calculate performance (MSE of VaR)
                var = risk_metrics['var']
                actual_losses = -portfolio_returns[portfolio_returns < 0]
                if len(actual_losses) > 0:
                    performance[i] = 1 / (mean_squared_error([var] * len(actual_losses), actual_losses) + 1e-10)
                else:
                    performance[i] = 0
        
        # Update performance history
        self.performance_history = self.performance_history.iloc[1:].copy()
        self.performance_history.loc[new_data.index[-1]] = performance
        
        # Calculate weights based on recent performance
        recent_performance = self.performance_history.mean().values
        
        # Set weights based on recent performance
        if np.sum(recent_performance) > 0:
            self.set_weights(recent_performance)
        else:
            # If all models perform poorly, use equal weights
            self.set_weights(np.ones(len(self.models)))
    
    def estimate_risk(self, portfolio=None, confidence_level=0.95, horizon=1):
        """
        Estimate risk metrics for a portfolio.
        
        Parameters:
        -----------
        portfolio : Portfolio, optional
            Portfolio object
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
        
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
        
        # Create dummy portfolio if not provided
        if portfolio is None:
            # Assume all models have the same returns data
            if hasattr(self.models[0], 'returns'):
                returns = self.models[0].returns
                portfolio_weights = pd.Series(
                    index=returns.columns,
                    data=np.ones(returns.shape[1]) / returns.shape[1]
                )
                portfolio = type('Portfolio', (), {'weights': portfolio_weights})
        
        # Get risk metrics for each model
        var_predictions = np.zeros(len(self.models))
        es_predictions = np.zeros(len(self.models))
        
        for i, model in enumerate(self.models):
            if hasattr(model, 'estimate_risk'):
                risk_metrics = model.estimate_risk(portfolio, confidence_level, horizon)
                var_predictions[i] = risk_metrics['var']
                es_predictions[i] = risk_metrics.get('expected_shortfall', risk_metrics['var'] * 1.2)  # Default ES if not provided
        
        # Combine predictions using weights
        var = np.sum(var_predictions * self.weights)
        es = np.sum(es_predictions * self.weights)
        
        # Return risk metrics
        return {
            'var': var,
            'expected_shortfall': es,
            'confidence_level': confidence_level,
            'horizon': horizon,
            'model_vars': var_predictions,
            'model_es': es_predictions,
            'weights': self.weights
        }
    
    def plot_weights(self, figsize=(12, 6)):
        """
        Plot model weights.
        
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
        
        # Plot weights
        model_names = [f'Model {i+1}' for i in range(len(self.models))]
        ax.bar(model_names, self.weights)
        
        # Set title and labels
        ax.set_title('Ensemble Model Weights')
        ax.set_xlabel('Model')
        ax.set_ylabel('Weight')
        
        # Add values on top of bars
        for i, v in enumerate(self.weights):
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_performance_history(self, figsize=(12, 6)):
        """
        Plot performance history for dynamic weights.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.calibrated or not self.dynamic_weights:
            raise ValueError("Model not calibrated or dynamic weights not enabled")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot performance history
        for i in range(len(self.models)):
            ax.plot(
                self.performance_history.index,
                self.performance_history[f'model_{i}'],
                label=f'Model {i+1}'
            )
        
        # Set title and labels
        ax.set_title('Model Performance History')
        ax.set_xlabel('Date')
        ax.set_ylabel('Performance')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_var_comparison(self, portfolio=None, confidence_level=0.95, horizon=1, figsize=(12, 6)):
        """
        Plot VaR comparison between models.
        
        Parameters:
        -----------
        portfolio : Portfolio, optional
            Portfolio object
        confidence_level : float, default=0.95
            Confidence level for VaR
        horizon : int, default=1
            Forecast horizon
        figsize : tuple, default=(12, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Get risk metrics
        risk_metrics = self.estimate_risk(portfolio, confidence_level, horizon)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot VaR for each model
        model_names = [f'Model {i+1}' for i in range(len(self.models))]
        ax.bar(model_names, risk_metrics['model_vars'])
        
        # Add ensemble VaR
        ax.axhline(
            risk_metrics['var'],
            color='r',
            linestyle='--',
            label=f'Ensemble VaR: {risk_metrics["var"]:.4f}'
        )
        
        # Set title and labels
        ax.set_title('VaR Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('VaR')
        
        # Add values on top of bars
        for i, v in enumerate(risk_metrics['model_vars']):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_es_comparison(self, portfolio=None, confidence_level=0.95, horizon=1, figsize=(12, 6)):
        """
        Plot ES comparison between models.
        
        Parameters:
        -----------
        portfolio : Portfolio, optional
            Portfolio object
        confidence_level : float, default=0.95
            Confidence level for ES
        horizon : int, default=1
            Forecast horizon
        figsize : tuple, default=(12, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Get risk metrics
        risk_metrics = self.estimate_risk(portfolio, confidence_level, horizon)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ES for each model
        model_names = [f'Model {i+1}' for i in range(len(self.models))]
        ax.bar(model_names, risk_metrics['model_es'])
        
        # Add ensemble ES
        ax.axhline(
            risk_metrics['expected_shortfall'],
            color='r',
            linestyle='--',
            label=f'Ensemble ES: {risk_metrics["expected_shortfall"]:.4f}'
        )
        
        # Set title and labels
        ax.set_title('Expected Shortfall Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('ES')
        
        # Add values on top of bars
        for i, v in enumerate(risk_metrics['model_es']):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
