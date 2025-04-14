"""
Regime Switching Model Module for Risk Modeling Framework.

This module provides functionality for regime switching models to detect
and adapt to changing market conditions.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm


class RegimeSwitchingModel:
    """
    Regime Switching Model class for risk modeling.
    
    This class provides methods for detecting and modeling different market regimes
    using various statistical techniques including Hidden Markov Models (HMM),
    Gaussian Mixture Models (GMM), and K-means clustering.
    """
    
    def __init__(self):
        """
        Initialize the regime switching model.
        """
        self.returns = None
        self.hmm_model = None
        self.gmm_model = None
        self.kmeans_model = None
        self.regime_probabilities = None
        self.regime_states = None
        self.regime_volatilities = None
        self.regime_correlations = None
        self.calibrated = False
    
    def calibrate(self, historical_data, num_regimes=3, method='hmm'):
        """
        Calibrate the regime switching model with historical data.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        num_regimes : int, default=3
            Number of regimes to detect
        method : str, default='hmm'
            Method for regime detection ('hmm', 'gmm', or 'kmeans')
        """
        self.returns = historical_data
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        portfolio_returns = self.returns.dot(portfolio_weights)
        
        # Detect regimes
        if method == 'hmm':
            self._calibrate_hmm(portfolio_returns, num_regimes)
        elif method == 'gmm':
            self._calibrate_gmm(portfolio_returns, num_regimes)
        elif method == 'kmeans':
            self._calibrate_kmeans(portfolio_returns, num_regimes)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate regime-specific parameters
        self._calculate_regime_parameters()
        
        self.calibrated = True
    
    def _calibrate_hmm(self, portfolio_returns, num_regimes):
        """
        Calibrate Hidden Markov Model for regime detection.
        
        Parameters:
        -----------
        portfolio_returns : pandas.Series
            Portfolio returns
        num_regimes : int
            Number of regimes to detect
        """
        # Prepare data
        X = np.array(portfolio_returns).reshape(-1, 1)
        
        # Initialize HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=num_regimes,
            covariance_type='full',
            n_iter=1000,
            random_state=42
        )
        
        # Fit HMM
        self.hmm_model.fit(X)
        
        # Predict hidden states
        hidden_states = self.hmm_model.predict(X)
        
        # Calculate state probabilities
        state_probs = np.zeros((len(portfolio_returns), num_regimes))
        for i in range(len(portfolio_returns)):
            state_probs[i, hidden_states[i]] = 1.0
        
        # Create regime probabilities DataFrame
        self.regime_probabilities = pd.DataFrame(
            index=portfolio_returns.index,
            columns=[f'Regime {i}' for i in range(num_regimes)],
            data=state_probs
        )
        
        # Create regime states Series
        self.regime_states = pd.Series(
            index=portfolio_returns.index,
            data=hidden_states
        )
    
    def _calibrate_gmm(self, portfolio_returns, num_regimes):
        """
        Calibrate Gaussian Mixture Model for regime detection.
        
        Parameters:
        -----------
        portfolio_returns : pandas.Series
            Portfolio returns
        num_regimes : int
            Number of regimes to detect
        """
        # Prepare data
        X = np.array(portfolio_returns).reshape(-1, 1)
        
        # Initialize GMM
        self.gmm_model = GaussianMixture(
            n_components=num_regimes,
            covariance_type='full',
            random_state=42
        )
        
        # Fit GMM
        self.gmm_model.fit(X)
        
        # Predict probabilities
        state_probs = self.gmm_model.predict_proba(X)
        
        # Create regime probabilities DataFrame
        self.regime_probabilities = pd.DataFrame(
            index=portfolio_returns.index,
            columns=[f'Regime {i}' for i in range(num_regimes)],
            data=state_probs
        )
        
        # Predict most likely regime
        hidden_states = self.gmm_model.predict(X)
        
        # Create regime states Series
        self.regime_states = pd.Series(
            index=portfolio_returns.index,
            data=hidden_states
        )
    
    def _calibrate_kmeans(self, portfolio_returns, num_regimes):
        """
        Calibrate K-means clustering for regime detection.
        
        Parameters:
        -----------
        portfolio_returns : pandas.Series
            Portfolio returns
        num_regimes : int
            Number of regimes to detect
        """
        # Calculate features
        returns = portfolio_returns.values
        vol = pd.Series(returns).rolling(window=21).std().values
        
        # Prepare data
        X = np.column_stack([returns, vol])
        X = X[~np.isnan(X).any(axis=1)]  # Remove NaN values
        
        # Initialize K-means
        self.kmeans_model = KMeans(
            n_clusters=num_regimes,
            random_state=42
        )
        
        # Fit K-means
        self.kmeans_model.fit(X)
        
        # Predict clusters
        # Need to handle NaN values at the beginning
        nan_mask = np.isnan(vol)
        hidden_states = np.zeros(len(returns))
        hidden_states[~nan_mask] = self.kmeans_model.predict(X)
        hidden_states[nan_mask] = np.nan
        
        # Create one-hot encoded probabilities
        state_probs = np.zeros((len(returns), num_regimes))
        for i in range(len(returns)):
            if not np.isnan(hidden_states[i]):
                state_probs[i, int(hidden_states[i])] = 1.0
        
        # Create regime probabilities DataFrame
        self.regime_probabilities = pd.DataFrame(
            index=portfolio_returns.index,
            columns=[f'Regime {i}' for i in range(num_regimes)],
            data=state_probs
        )
        
        # Create regime states Series
        self.regime_states = pd.Series(
            index=portfolio_returns.index,
            data=hidden_states
        )
    
    def _calculate_regime_parameters(self):
        """
        Calculate regime-specific parameters.
        """
        # Initialize regime volatilities
        self.regime_volatilities = {}
        
        # Initialize regime correlations
        self.regime_correlations = {}
        
        # Calculate parameters for each regime
        for regime in range(len(self.regime_probabilities.columns)):
            # Get data for this regime
            regime_mask = self.regime_states == regime
            regime_data = self.returns[regime_mask]
            
            if len(regime_data) > 0:
                # Calculate volatility
                self.regime_volatilities[regime] = regime_data.std()
                
                # Calculate correlation
                self.regime_correlations[regime] = regime_data.corr()
    
    def detect_regime(self, data, method='hmm'):
        """
        Detect regime for new data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            New data
        method : str, default='hmm'
            Method for regime detection ('hmm', 'gmm', or 'kmeans')
            
        Returns:
        --------
        int
            Detected regime
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(data.shape[1]) / data.shape[1]
        portfolio_returns = data.dot(portfolio_weights)
        
        # Prepare data
        X = np.array(portfolio_returns).reshape(-1, 1)
        
        # Detect regime
        if method == 'hmm' and self.hmm_model is not None:
            regime = self.hmm_model.predict(X)[-1]
        elif method == 'gmm' and self.gmm_model is not None:
            regime = self.gmm_model.predict(X)[-1]
        elif method == 'kmeans' and self.kmeans_model is not None:
            # Calculate features
            returns = portfolio_returns.values
            vol = pd.Series(returns).rolling(window=min(21, len(returns))).std().values
            
            # Prepare data
            X = np.column_stack([returns[-1], vol[-1]]).reshape(1, -1)
            
            # Predict regime
            regime = self.kmeans_model.predict(X)[0]
        else:
            raise ValueError(f"Unknown method or model not calibrated: {method}")
        
        return regime
    
    def get_regime_probabilities(self, data=None, method='hmm'):
        """
        Get regime probabilities.
        
        Parameters:
        -----------
        data : pandas.DataFrame, optional
            New data
        method : str, default='hmm'
            Method for regime detection ('hmm', 'gmm', or 'kmeans')
            
        Returns:
        --------
        pandas.Series or pandas.DataFrame
            Regime probabilities
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        if data is None:
            # Return latest regime probabilities from calibration
            return self.regime_probabilities.iloc[-1]
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(data.shape[1]) / data.shape[1]
        portfolio_returns = data.dot(portfolio_weights)
        
        # Prepare data
        X = np.array(portfolio_returns).reshape(-1, 1)
        
        # Get regime probabilities
        if method == 'hmm' and self.hmm_model is not None:
            # For HMM, we don't have direct probability output
            # So we use one-hot encoding based on predicted state
            regime = self.hmm_model.predict(X)[-1]
            probs = np.zeros(len(self.regime_probabilities.columns))
            probs[regime] = 1.0
        elif method == 'gmm' and self.gmm_model is not None:
            probs = self.gmm_model.predict_proba(X)[-1]
        elif method == 'kmeans' and self.kmeans_model is not None:
            # For K-means, we don't have direct probability output
            # So we use one-hot encoding based on predicted cluster
            # Calculate features
            returns = portfolio_returns.values
            vol = pd.Series(returns).rolling(window=min(21, len(returns))).std().values
            
            # Prepare data
            X = np.column_stack([returns[-1], vol[-1]]).reshape(1, -1)
            
            # Predict regime
            regime = self.kmeans_model.predict(X)[0]
            probs = np.zeros(len(self.regime_probabilities.columns))
            probs[regime] = 1.0
        else:
            raise ValueError(f"Unknown method or model not calibrated: {method}")
        
        return pd.Series(
            index=self.regime_probabilities.columns,
            data=probs
        )
    
    def get_regime_volatility(self, regime):
        """
        Get volatility for a specific regime.
        
        Parameters:
        -----------
        regime : int
            Regime index
            
        Returns:
        --------
        pandas.Series
            Volatility for the specified regime
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        if regime not in self.regime_volatilities:
            raise ValueError(f"Unknown regime: {regime}")
        
        return self.regime_volatilities[regime]
    
    def get_regime_correlation(self, regime):
        """
        Get correlation matrix for a specific regime.
        
        Parameters:
        -----------
        regime : int
            Regime index
            
        Returns:
        --------
        pandas.DataFrame
            Correlation matrix for the specified regime
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        if regime not in self.regime_correlations:
            raise ValueError(f"Unknown regime: {regime}")
        
        return self.regime_correlations[regime]
    
    def simulate_returns(self, num_scenarios=1000, horizon=21, initial_regime=None):
        """
        Simulate returns using regime switching model.
        
        Parameters:
        -----------
        num_scenarios : int, default=1000
            Number of scenarios to simulate
        horizon : int, default=21
            Simulation horizon
        initial_regime : int, optional
            Initial regime
            
        Returns:
        --------
        pandas.DataFrame
            Simulated returns
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Set initial regime
        if initial_regime is None:
            initial_regime = self.regime_states.iloc[-1]
        
        # Initialize simulated returns
        simulated_returns = np.zeros((num_scenarios, horizon, self.returns.shape[1]))
        
        # Simulate returns for each scenario
        for i in range(num_scenarios):
            # Initialize current regime
            current_regime = initial_regime
            
            # Simulate returns for each time step
            for t in range(horizon):
                # Get regime parameters
                regime_vol = self.regime_volatilities[current_regime]
                regime_corr = self.regime_correlations[current_regime]
                
                # Calculate covariance matrix
                cov_matrix = np.diag(regime_vol) @ regime_corr @ np.diag(regime_vol)
                
                # Simulate returns
                simulated_returns[i, t, :] = np.random.multivariate_normal(
                    mean=np.zeros(self.returns.shape[1]),
                    cov=cov_matrix
                )
                
                # Transition to next regime
                if self.hmm_model is not None:
                    # Use HMM transition matrix
                    transition_probs = self.hmm_model.transmat_[current_regime]
                    current_regime = np.random.choice(
                        len(transition_probs),
                        p=transition_probs
                    )
                else:
                    # Use empirical transition probabilities
                    regime_counts = self.regime_states.value_counts()
                    regime_probs = regime_counts / regime_counts.sum()
                    current_regime = np.random.choice(
                        len(regime_probs),
                        p=regime_probs
                    )
        
        # Convert to DataFrame
        simulated_returns_df = pd.DataFrame(
            index=range(num_scenarios),
            columns=[f'Scenario_{i}' for i in range(horizon)]
        )
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        
        for i in range(num_scenarios):
            for t in range(horizon):
                simulated_returns_df.loc[i, f'Scenario_{t}'] = simulated_returns[i, t, :].dot(portfolio_weights)
        
        return simulated_returns_df
    
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
        self.regime_probabilities.plot(ax=ax)
        
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
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot returns by regime
        for regime in range(len(self.regime_probabilities.columns)):
            regime_mask = self.regime_states == regime
            ax.scatter(
                portfolio_returns.index[regime_mask],
                portfolio_returns[regime_mask],
                label=f'Regime {regime}',
                alpha=0.5
            )
        
        # Set title and labels
        ax.set_title('Returns by Regime')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_regime_volatility(self, figsize=(12, 6)):
        """
        Plot volatility by regime.
        
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
        
        # Plot volatility by regime
        for regime, vol in self.regime_volatilities.items():
            vol.plot(kind='bar', ax=ax, alpha=0.5, label=f'Regime {regime}')
        
        # Set title and labels
        ax.set_title('Volatility by Regime')
        ax.set_xlabel('Asset')
        ax.set_ylabel('Volatility')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_regime_correlation(self, regime=0, figsize=(12, 10)):
        """
        Plot correlation matrix for a specific regime.
        
        Parameters:
        -----------
        regime : int, default=0
            Regime index
        figsize : tuple, default=(12, 10)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        if regime not in self.regime_correlations:
            raise ValueError(f"Unknown regime: {regime}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot correlation matrix
        sns.heatmap(
            self.regime_correlations[regime],
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            annot=True,
            fmt='.2f',
            ax=ax
        )
        
        # Set title
        ax.set_title(f'Correlation Matrix for Regime {regime}')
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_simulated_returns(self, num_scenarios=100, horizon=21, figsize=(12, 6)):
        """
        Plot simulated returns.
        
        Parameters:
        -----------
        num_scenarios : int, default=100
            Number of scenarios to simulate
        horizon : int, default=21
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
