"""
Regime Analysis Module for Risk Modeling Framework.

This module provides functionality for detecting and analyzing market regimes
to improve risk modeling accuracy.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm


class RegimeAnalysis:
    """
    Regime Analysis class for risk modeling.
    
    This class provides methods for detecting and analyzing market regimes
    to improve risk modeling accuracy.
    """
    
    def __init__(self):
        """
        Initialize the regime analysis model.
        """
        self.regimes = {}
        self.model = None
        self.model_type = None
        self.historical_data = None
        self.regime_labels = None
        self.regime_probabilities = None
        self.current_regime = None
    
    def calibrate(self, historical_data, method='hmm', n_regimes=3):
        """
        Calibrate the regime analysis model with historical data.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        method : str, default='hmm'
            Method to use for regime detection ('hmm', 'kmeans', 'gmm')
        n_regimes : int, default=3
            Number of regimes to detect
            
        Returns:
        --------
        dict
            Dictionary containing regime information
        """
        self.historical_data = historical_data
        
        # Detect regimes
        regimes = self.detect_regimes(historical_data, method=method, n_regimes=n_regimes)
        
        return regimes
    
    def detect_regimes(self, data=None, method='hmm', n_regimes=3, window=None):
        """
        Detect market regimes in the data.
        
        Parameters:
        -----------
        data : pandas.DataFrame, optional
            Returns data (if None, use historical_data)
        method : str, default='hmm'
            Method to use for regime detection ('hmm', 'kmeans', 'gmm')
        n_regimes : int, default=3
            Number of regimes to detect
        window : int, optional
            Rolling window size for regime detection
            
        Returns:
        --------
        dict
            Dictionary containing regime information
        """
        if data is None:
            if self.historical_data is None:
                raise ValueError("No data provided and no historical data available")
            data = self.historical_data
        
        # Store model type
        self.model_type = method
        
        # Calculate features for regime detection
        features = self._calculate_regime_features(data, window=window)
        
        # Detect regimes using the specified method
        if method == 'hmm':
            regimes = self._detect_regimes_hmm(features, n_regimes)
        elif method == 'kmeans':
            regimes = self._detect_regimes_kmeans(features, n_regimes)
        elif method == 'gmm':
            regimes = self._detect_regimes_gmm(features, n_regimes)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Store regimes
        self.regimes = regimes
        
        return regimes
    
    def _calculate_regime_features(self, data, window=None):
        """
        Calculate features for regime detection.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Returns data
        window : int, optional
            Rolling window size for feature calculation
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing features for regime detection
        """
        # If window is provided, calculate rolling features
        if window is not None:
            # Calculate rolling volatility
            rolling_vol = data.rolling(window=window).std()
            
            # Calculate rolling correlation
            rolling_corr = pd.DataFrame(index=data.index)
            for i in range(len(data.columns)):
                for j in range(i+1, len(data.columns)):
                    col_i = data.columns[i]
                    col_j = data.columns[j]
                    corr_name = f"corr_{col_i}_{col_j}"
                    rolling_corr[corr_name] = data[col_i].rolling(window=window).corr(data[col_j])
            
            # Calculate rolling skewness
            rolling_skew = data.rolling(window=window).skew()
            
            # Calculate rolling kurtosis
            rolling_kurt = data.rolling(window=window).kurt()
            
            # Combine features
            features = pd.concat([rolling_vol, rolling_corr, rolling_skew, rolling_kurt], axis=1)
            
            # Drop NaN values
            features = features.dropna()
            
            return features
        
        # Otherwise, calculate global features
        else:
            # Calculate volatility
            volatility = data.std()
            
            # Calculate correlation matrix
            correlation = data.corr()
            
            # Calculate skewness
            skewness = data.skew()
            
            # Calculate kurtosis
            kurtosis = data.kurt()
            
            # Combine features
            features = pd.DataFrame()
            
            # Add volatility features
            for col in volatility.index:
                features[f"vol_{col}"] = [volatility[col]]
            
            # Add correlation features
            for i in range(len(correlation.columns)):
                for j in range(i+1, len(correlation.columns)):
                    col_i = correlation.columns[i]
                    col_j = correlation.columns[j]
                    features[f"corr_{col_i}_{col_j}"] = [correlation.loc[col_i, col_j]]
            
            # Add skewness features
            for col in skewness.index:
                features[f"skew_{col}"] = [skewness[col]]
            
            # Add kurtosis features
            for col in kurtosis.index:
                features[f"kurt_{col}"] = [kurtosis[col]]
            
            return features
    
    def _detect_regimes_hmm(self, features, n_regimes):
        """
        Detect regimes using Hidden Markov Model.
        
        Parameters:
        -----------
        features : pandas.DataFrame
            Features for regime detection
        n_regimes : int
            Number of regimes to detect
            
        Returns:
        --------
        dict
            Dictionary containing regime information
        """
        # Initialize HMM model
        model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=1000, random_state=42)
        
        # Fit model
        model.fit(features.values)
        
        # Predict regime states
        regime_states = model.predict(features.values)
        
        # Calculate regime probabilities
        regime_probs = model.predict_proba(features.values)
        
        # Create regime labels DataFrame
        regime_labels = pd.Series(regime_states, index=features.index, name='regime')
        
        # Create regime probabilities DataFrame
        regime_probabilities = pd.DataFrame(regime_probs, index=features.index, columns=[f'regime_{i}' for i in range(n_regimes)])
        
        # Determine current regime (last regime in the data)
        current_regime = int(regime_labels.iloc[-1])
        
        # Calculate regime characteristics
        regime_characteristics = self._calculate_regime_characteristics(regime_labels)
        
        # Store model and results
        self.model = model
        self.regime_labels = regime_labels
        self.regime_probabilities = regime_probabilities
        self.current_regime = current_regime
        
        # Return regime information
        return {
            'model': 'hmm',
            'n_regimes': n_regimes,
            'regime_labels': regime_labels,
            'regime_probabilities': regime_probabilities,
            'current_regime': current_regime,
            'regime_characteristics': regime_characteristics,
            'transition_matrix': model.transmat_,
            'log_likelihood': model.score(features.values)
        }
    
    def _detect_regimes_kmeans(self, features, n_regimes):
        """
        Detect regimes using K-means clustering.
        
        Parameters:
        -----------
        features : pandas.DataFrame
            Features for regime detection
        n_regimes : int
            Number of regimes to detect
            
        Returns:
        --------
        dict
            Dictionary containing regime information
        """
        # Initialize K-means model
        model = KMeans(n_clusters=n_regimes, random_state=42)
        
        # Fit model
        model.fit(features.values)
        
        # Predict regime states
        regime_states = model.predict(features.values)
        
        # Calculate distances to cluster centers
        distances = model.transform(features.values)
        
        # Convert distances to probabilities using softmax
        def softmax(x):
            e_x = np.exp(-x)
            return e_x / e_x.sum(axis=1, keepdims=True)
        
        regime_probs = softmax(distances)
        
        # Create regime labels DataFrame
        regime_labels = pd.Series(regime_states, index=features.index, name='regime')
        
        # Create regime probabilities DataFrame
        regime_probabilities = pd.DataFrame(regime_probs, index=features.index, columns=[f'regime_{i}' for i in range(n_regimes)])
        
        # Determine current regime (last regime in the data)
        current_regime = int(regime_labels.iloc[-1])
        
        # Calculate regime characteristics
        regime_characteristics = self._calculate_regime_characteristics(regime_labels)
        
        # Store model and results
        self.model = model
        self.regime_labels = regime_labels
        self.regime_probabilities = regime_probabilities
        self.current_regime = current_regime
        
        # Return regime information
        return {
            'model': 'kmeans',
            'n_regimes': n_regimes,
            'regime_labels': regime_labels,
            'regime_probabilities': regime_probabilities,
            'current_regime': current_regime,
            'regime_characteristics': regime_characteristics,
            'cluster_centers': model.cluster_centers_,
            'inertia': model.inertia_
        }
    
    def _detect_regimes_gmm(self, features, n_regimes):
        """
        Detect regimes using Gaussian Mixture Model.
        
        Parameters:
        -----------
        features : pandas.DataFrame
            Features for regime detection
        n_regimes : int
            Number of regimes to detect
            
        Returns:
        --------
        dict
            Dictionary containing regime information
        """
        # Initialize GMM model
        model = GaussianMixture(n_components=n_regimes, covariance_type='full', random_state=42)
        
        # Fit model
        model.fit(features.values)
        
        # Predict regime states
        regime_states = model.predict(features.values)
        
        # Calculate regime probabilities
        regime_probs = model.predict_proba(features.values)
        
        # Create regime labels DataFrame
        regime_labels = pd.Series(regime_states, index=features.index, name='regime')
        
        # Create regime probabilities DataFrame
        regime_probabilities = pd.DataFrame(regime_probs, index=features.index, columns=[f'regime_{i}' for i in range(n_regimes)])
        
        # Determine current regime (last regime in the data)
        current_regime = int(regime_labels.iloc[-1])
        
        # Calculate regime characteristics
        regime_characteristics = self._calculate_regime_characteristics(regime_labels)
        
        # Store model and results
        self.model = model
        self.regime_labels = regime_labels
        self.regime_probabilities = regime_probabilities
        self.current_regime = current_regime
        
        # Return regime information
        return {
            'model': 'gmm',
            'n_regimes': n_regimes,
            'regime_labels': regime_labels,
            'regime_probabilities': regime_probabilities,
            'current_regime': current_regime,
            'regime_characteristics': regime_characteristics,
            'means': model.means_,
            'covariances': model.covariances_,
            'weights': model.weights_,
            'log_likelihood': model.score(features.values)
        }
    
    def _calculate_regime_characteristics(self, regime_labels):
        """
        Calculate characteristics of each regime.
        
        Parameters:
        -----------
        regime_labels : pandas.Series
            Series containing regime labels
            
        Returns:
        --------
        dict
            Dictionary containing regime characteristics
        """
        if self.historical_data is None:
            raise ValueError("No historical data available")
        
        # Align historical data with regime labels
        aligned_data = self.historical_data.loc[regime_labels.index]
        
        # Initialize regime characteristics
        regime_characteristics = {}
        
        # Calculate characteristics for each regime
        for regime in range(regime_labels.max() + 1):
            # Get data for this regime
            regime_data = aligned_data[regime_labels == regime]
            
            # Calculate mean returns
            mean_returns = regime_data.mean()
            
            # Calculate volatility
            volatility = regime_data.std()
            
            # Calculate correlation matrix
            correlation = regime_data.corr()
            
            # Calculate skewness
            skewness = regime_data.skew()
            
            # Calculate kurtosis
            kurtosis = regime_data.kurt()
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            sharpe_ratio = mean_returns / volatility
            
            # Calculate drawdown
            cum_returns = (1 + regime_data).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Store characteristics
            regime_characteristics[regime] = {
                'mean_returns': mean_returns,
                'volatility': volatility,
                'correlation': correlation,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'duration': len(regime_data),
                'frequency': len(regime_data) / len(aligned_data)
            }
        
        return regime_characteristics
    
    def predict_regime(self, data):
        """
        Predict regime for new data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            New returns data
            
        Returns:
        --------
        dict
            Dictionary containing regime prediction
        """
        if self.model is None:
            raise ValueError("Model not calibrated")
        
        # Calculate features for regime detection
        features = self._calculate_regime_features(data)
        
        # Predict regime based on model type
        if self.model_type == 'hmm':
            regime_state = self.model.predict(features.values)
            regime_probs = self.model.predict_proba(features.values)
        elif self.model_type == 'kmeans':
            regime_state = self.model.predict(features.values)
            distances = self.model.transform(features.values)
            
            # Convert distances to probabilities using softmax
            def softmax(x):
                e_x = np.exp(-x)
                return e_x / e_x.sum(axis=1, keepdims=True)
            
            regime_probs = softmax(distances)
        elif self.model_type == 'gmm':
            regime_state = self.model.predict(features.values)
            regime_probs = self.model.predict_proba(features.values)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Get predicted regime (last regime in the data)
        predicted_regime = int(regime_state[-1])
        
        # Get regime probabilities
        n_regimes = regime_probs.shape[1]
        regime_probabilities = {f'regime_{i}': regime_probs[-1, i] for i in range(n_regimes)}
        
        return {
            'predicted_regime': predicted_regime,
            'regime_probabilities': regime_probabilities
        }
    
    def get_regime_characteristics(self, regime=None):
        """
        Get characteristics of a specific regime.
        
        Parameters:
        -----------
        regime : int, optional
            Regime to get characteristics for (if None, use current regime)
            
        Returns:
        --------
        dict
            Dictionary containing regime characteristics
        """
        if self.regimes is None or 'regime_characteristics' not in self.regimes:
            raise ValueError("Regime characteristics not available")
        
        if regime is None:
            regime = self.current_regime
        
        if regime not in self.regimes['regime_characteristics']:
            raise ValueError(f"Regime {regime} not found in regime characteristics")
        
        return self.regimes['regime_characteristics'][regime]
    
    def plot_regimes(self, data=None, column=None, figsize=(12, 6)):
        """
        Plot regimes over time.
        
        Parameters:
        -----------
        data : pandas.DataFrame, optional
            Returns data to plot (if None, use historical_data)
        column : str, optional
            Column to plot (if None, use first column)
        figsize : tuple, default=(12, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.regime_labels is None:
            raise ValueError("Regime labels not available")
        
        if data is None:
            if self.historical_data is None:
                raise ValueError("No data provided and no historical data available")
            data = self.historical_data
        
        if column is None:
            column = data.columns[0]
        
        # Align data with regime labels
        aligned_data = data.loc[self.regime_labels.index]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot data
        ax.plot(aligned_data.index, aligned_data[column], color='black', alpha=0.5)
        
        # Plot regime background
        for regime in range(self.regime_labels.max() + 1):
            # Get regime periods
            regime_periods = self.regime_labels == regime
            
            # Skip if no periods for this regime
            if not regime_periods.any():
                continue
            
            # Get start and end dates for each continuous period
            starts = []
            ends = []
            
            in_period = False
            for date, value in regime_periods.items():
                if value and not in_period:
                    starts.append(date)
                    in_period = True
                elif not value and in_period:
                    ends.append(date)
                    in_period = False
            
            # Add last end date if still in period
            if in_period:
                ends.append(regime_periods.index[-1])
            
            # Plot background for each period
            for start, end in zip(starts, ends):
                ax.axvspan(start, end, alpha=0.3, color=f'C{regime}')
        
        # Add legend
        legend_elements = []
        for regime in range(self.regime_labels.max() + 1):
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, alpha=0.3, color=f'C{regime}', label=f'Regime {regime}'))
        
        ax.legend(handles=legend_elements, loc='upper left')
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel(column)
        ax.set_title(f'Regimes over time - {column}')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def estimate_risk(self, confidence_level=0.95, regime=None):
        """
        Estimate risk metrics for a specific regime.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        regime : int, optional
            Regime to estimate risk for (if None, use current regime)
            
        Returns:
        --------
        dict
            Dictionary containing risk metrics
        """
        if self.historical_data is None:
            raise ValueError("No historical data available")
        
        if self.regime_labels is None:
            raise ValueError("Regime labels not available")
        
        if regime is None:
            regime = self.current_regime
        
        # Get data for this regime
        regime_data = self.historical_data.loc[self.regime_labels.index][self.regime_labels == regime]
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(regime_data.shape[1]) / regime_data.shape[1]
        portfolio_returns = regime_data.dot(portfolio_weights)
        
        # Calculate VaR
        var = -np.percentile(portfolio_returns, 100 * (1 - confidence_level))
        
        # Calculate ES
        es = -portfolio_returns[portfolio_returns <= -var].mean()
        
        return {
            'VaR': var,
            'ES': es,
            'confidence_level': confidence_level,
            'regime': regime
        }
