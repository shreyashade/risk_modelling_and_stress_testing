"""
Bayesian Regime Detection Module for Risk Modeling Framework.

This module provides functionality for detecting market regimes using Bayesian methods.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
# Replace theano with pytensor which is the modern replacement
import pytensor.tensor as tt


class BayesianRegimeDetection:
    """
    Bayesian Regime Detection class for risk modeling.
    
    This class provides methods for detecting market regimes using Bayesian methods.
    """
    
    def __init__(self):
        """
        Initialize the Bayesian regime detection model.
        """
        self.returns = None
        self.regime_model = None
        self.trace = None
        self.regime_probabilities = None
        self.current_regime = None
        self.num_regimes = 3  # Default: low, normal, high volatility
        self.lookback_period = 252  # Default: 1 year of trading days
        self.calibrated = False
    
    def calibrate(self, historical_data, num_regimes=None, lookback_period=None):
        """
        Calibrate the Bayesian regime detection model with historical data.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        num_regimes : int, optional
            Number of regimes to detect
        lookback_period : int, optional
            Lookback period for regime detection
            
        Returns:
        --------
        dict
            Dictionary containing regime detection results
        """
        self.returns = historical_data
        
        # Set parameters
        if num_regimes is not None:
            self.num_regimes = num_regimes
        
        if lookback_period is not None:
            self.lookback_period = lookback_period
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(historical_data.shape[1]) / historical_data.shape[1]
        portfolio_returns = historical_data.dot(portfolio_weights)
        
        # Use only the most recent data
        if len(portfolio_returns) > self.lookback_period:
            portfolio_returns = portfolio_returns.iloc[-self.lookback_period:]
        
        # Detect regimes
        self._detect_regimes(portfolio_returns)
        
        self.calibrated = True
        
        return {
            'regime_probabilities': self.regime_probabilities,
            'current_regime': self.current_regime,
            'num_regimes': self.num_regimes
        }
    
    def _detect_regimes(self, returns):
        """
        Detect regimes using Bayesian methods.
        
        Parameters:
        -----------
        returns : pandas.Series
            Portfolio returns
        """
        # Create Bayesian model
        with pm.Model() as self.regime_model:
            # Regime switching probabilities
            alpha = pm.Dirichlet('alpha', a=np.ones(self.num_regimes))
            
            # Regime volatilities
            sigma = pm.HalfNormal('sigma', sigma=0.1, shape=self.num_regimes)
            
            # Regime means
            mu = pm.Normal('mu', mu=0, sigma=0.1, shape=self.num_regimes)
            
            # Latent regime state
            z = pm.Categorical('z', p=alpha, shape=len(returns))
            
            # Returns likelihood
            pm.Normal('returns', mu=mu[z], sigma=sigma[z], observed=returns.values)
            
            # Sample from posterior
            self.trace = pm.sample(1000, tune=1000, chains=2, cores=1, return_inferencedata=True)
        
        # Calculate regime probabilities
        self._calculate_regime_probabilities()
        
        # Determine current regime
        self._determine_current_regime()
    
    def _calculate_regime_probabilities(self):
        """
        Calculate regime probabilities from MCMC trace.
        """
        # Get regime assignments
        z_samples = self.trace.posterior['z'].values
        
        # Calculate regime probabilities for each time point
        regime_probs = np.zeros((len(z_samples[0, 0]), self.num_regimes))
        
        for i in range(self.num_regimes):
            regime_probs[:, i] = np.mean(z_samples == i, axis=(0, 1))
        
        # Convert to DataFrame
        self.regime_probabilities = pd.DataFrame(
            regime_probs,
            index=self.returns.index[-len(regime_probs):],
            columns=[f'Regime {i}' for i in range(self.num_regimes)]
        )
    
    def _determine_current_regime(self):
        """
        Determine current regime based on regime probabilities.
        """
        # Get regime probabilities for the last time point
        last_probs = self.regime_probabilities.iloc[-1].values
        
        # Determine current regime
        self.current_regime = np.argmax(last_probs)
    
    def predict_regime(self, new_data=None, horizon=1):
        """
        Predict future regime.
        
        Parameters:
        -----------
        new_data : pandas.DataFrame, optional
            New data for prediction (if None, use last available data)
        horizon : int, default=1
            Prediction horizon
            
        Returns:
        --------
        dict
            Dictionary containing regime prediction results
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Get transition probabilities
        transition_probs = self._estimate_transition_probabilities()
        
        # Get current regime probabilities
        current_probs = self.regime_probabilities.iloc[-1].values
        
        # Initialize future probabilities
        future_probs = current_probs.copy()
        
        # Predict future regimes
        for _ in range(horizon):
            future_probs = np.dot(future_probs, transition_probs)
        
        # Determine predicted regime
        predicted_regime = np.argmax(future_probs)
        
        return {
            'predicted_regime': predicted_regime,
            'regime_probabilities': future_probs,
            'horizon': horizon
        }
    
    def _estimate_transition_probabilities(self):
        """
        Estimate regime transition probabilities.
        
        Returns:
        --------
        numpy.ndarray
            Transition probability matrix
        """
        # Get regime assignments
        z_samples = self.trace.posterior['z'].values
        
        # Calculate mean regime assignments
        z_mean = np.mean(z_samples, axis=(0, 1))
        
        # Determine most likely regime for each time point
        regimes = np.argmax(
            np.array([z_mean == i for i in range(self.num_regimes)]),
            axis=0
        )
        
        # Initialize transition count matrix
        transition_counts = np.zeros((self.num_regimes, self.num_regimes))
        
        # Count transitions
        for t in range(len(regimes) - 1):
            transition_counts[regimes[t], regimes[t+1]] += 1
        
        # Calculate transition probabilities
        transition_probs = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)
        
        # Handle zero counts
        transition_probs[np.isnan(transition_probs)] = 1.0 / self.num_regimes
        
        return transition_probs
    
    def get_regime_parameters(self):
        """
        Get parameters for each regime.
        
        Returns:
        --------
        dict
            Dictionary containing regime parameters
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Get posterior samples
        mu_samples = self.trace.posterior['mu'].values
        sigma_samples = self.trace.posterior['sigma'].values
        
        # Calculate mean parameters for each regime
        mu_mean = np.mean(mu_samples, axis=(0, 1))
        sigma_mean = np.mean(sigma_samples, axis=(0, 1))
        
        # Calculate credible intervals
        mu_hdi = az.hdi(self.trace.posterior['mu'])
        sigma_hdi = az.hdi(self.trace.posterior['sigma'])
        
        # Create parameter dictionary
        parameters = {}
        
        for i in range(self.num_regimes):
            parameters[f'Regime {i}'] = {
                'mean': mu_mean[i],
                'mean_hdi': (mu_hdi[i, 0], mu_hdi[i, 1]),
                'volatility': sigma_mean[i],
                'volatility_hdi': (sigma_hdi[i, 0], sigma_hdi[i, 1])
            }
        
        return parameters
    
    def plot_regime_probabilities(self, figsize=(12, 6)):
        """
        Plot regime probabilities.
        
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
        
        # Plot regime probabilities
        self.regime_probabilities.plot(ax=ax, linewidth=2)
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Probability')
        ax.set_title('Regime Probabilities')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_regime_returns(self, figsize=(12, 8)):
        """
        Plot returns by regime.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
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
        
        # Get regime assignments
        z_samples = self.trace.posterior['z'].values
        
        # Calculate mean regime assignments
        z_mean = np.mean(z_samples, axis=(0, 1))
        
        # Determine most likely regime for each time point
        regimes = np.argmax(
            np.array([z_mean == i for i in range(self.num_regimes)]),
            axis=0
        )
        
        # Create DataFrame with returns and regimes
        returns_with_regimes = pd.DataFrame({
            'Returns': portfolio_returns.iloc[-len(regimes):].values,
            'Regime': regimes
        }, index=portfolio_returns.index[-len(regimes):])
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot returns colored by regime
        for i in range(self.num_regimes):
            regime_returns = returns_with_regimes[returns_with_regimes['Regime'] == i]
            axes[0].scatter(
                regime_returns.index,
                regime_returns['Returns'],
                label=f'Regime {i}',
                alpha=0.7
            )
        
        # Set labels and title for returns plot
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Returns')
        axes[0].set_title('Returns by Regime')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot return distributions by regime
        for i in range(self.num_regimes):
            regime_returns = returns_with_regimes[returns_with_regimes['Regime'] == i]['Returns']
            sns.kdeplot(regime_returns, ax=axes[1], label=f'Regime {i}')
        
        # Set labels and title for distribution plot
        axes[1].set_xlabel('Returns')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Return Distributions by Regime')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_regime_parameters(self, figsize=(12, 6)):
        """
        Plot regime parameters.
        
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
        
        # Get regime parameters
        parameters = self.get_regime_parameters()
        
        # Extract means and volatilities
        means = [parameters[f'Regime {i}']['mean'] for i in range(self.num_regimes)]
        volatilities = [parameters[f'Regime {i}']['volatility'] for i in range(self.num_regimes)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot regime parameters
        ax.scatter(means, volatilities, s=100)
        
        # Add regime labels
        for i in range(self.num_regimes):
            ax.annotate(
                f'Regime {i}',
                (means[i], volatilities[i]),
                xytext=(10, 10),
                textcoords='offset points'
            )
        
        # Set labels and title
        ax.set_xlabel('Mean')
        ax.set_ylabel('Volatility')
        ax.set_title('Regime Parameters')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
