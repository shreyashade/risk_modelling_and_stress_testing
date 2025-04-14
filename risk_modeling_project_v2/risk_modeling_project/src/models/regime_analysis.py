"""
Correlation and Volatility Regime Analysis for Risk Modeling

This module implements various methods for analyzing and modeling different
correlation and volatility regimes in financial markets.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Union, Optional, Tuple, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorrelationRegimeAnalysis:
    """Analysis of correlation regimes in financial markets"""
    
    def __init__(self, window_size: int = 60, n_regimes: int = 3, method: str = 'kmeans'):
        """
        Initialize a Correlation Regime Analysis
        
        Parameters:
        -----------
        window_size : int, default=60
            Size of rolling window for correlation calculation
        n_regimes : int, default=3
            Number of correlation regimes to identify
        method : str, default='kmeans'
            Method for regime identification ('kmeans', 'gmm', 'threshold')
        """
        self.window_size = window_size
        self.n_regimes = n_regimes
        self.method = method
        self.regime_model = None
        self.regime_labels = None
        self.rolling_correlations = None
        self.regime_stats = None
    
    def fit(self, returns: pd.DataFrame, **kwargs) -> Dict:
        """
        Fit correlation regime model to returns data
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns data
        **kwargs : dict
            Additional parameters for fitting
            
        Returns:
        --------
        dict
            Dictionary with regime statistics
        """
        # Calculate rolling correlations
        self._calculate_rolling_correlations(returns)
        
        # Identify regimes
        self._identify_regimes(**kwargs)
        
        # Calculate regime statistics
        self._calculate_regime_statistics(returns)
        
        return self.regime_stats
    
    def _calculate_rolling_correlations(self, returns: pd.DataFrame) -> None:
        """
        Calculate rolling correlations
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns data
        """
        # Initialize rolling correlation matrix
        n_assets = len(returns.columns)
        n_periods = len(returns) - self.window_size + 1
        
        # Create a DataFrame to store average correlations
        self.rolling_correlations = pd.DataFrame(
            index=returns.index[self.window_size-1:],
            columns=['avg_correlation']
        )
        
        # Calculate rolling correlations
        for i in range(n_periods):
            window_data = returns.iloc[i:i+self.window_size]
            corr_matrix = window_data.corr()
            
            # Calculate average correlation (excluding diagonal)
            avg_corr = (corr_matrix.sum().sum() - n_assets) / (n_assets * (n_assets - 1))
            
            self.rolling_correlations.iloc[i, 0] = avg_corr
    
    def _identify_regimes(self, **kwargs) -> None:
        """
        Identify correlation regimes
        
        Parameters:
        -----------
        **kwargs : dict
            Additional parameters for regime identification
        """
        # Prepare data for clustering
        X = self.rolling_correlations.values.reshape(-1, 1)
        
        if self.method == 'kmeans':
            # Use KMeans clustering
            self.regime_model = KMeans(n_clusters=self.n_regimes, random_state=42)
            self.regime_labels = self.regime_model.fit_predict(X)
            
        elif self.method == 'gmm':
            # Use Gaussian Mixture Model
            self.regime_model = GaussianMixture(n_components=self.n_regimes, random_state=42)
            self.regime_labels = self.regime_model.fit_predict(X)
            
        elif self.method == 'threshold':
            # Use threshold-based approach
            thresholds = kwargs.get('thresholds', None)
            
            if thresholds is None:
                # Calculate thresholds based on quantiles
                q_values = np.linspace(0, 1, self.n_regimes + 1)[1:-1]
                thresholds = [np.quantile(X, q) for q in q_values]
            
            # Assign regimes based on thresholds
            self.regime_labels = np.zeros(len(X), dtype=int)
            
            for i, threshold in enumerate(thresholds):
                self.regime_labels[X.flatten() > threshold] = i + 1
            
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # Add regime labels to rolling correlations DataFrame
        self.rolling_correlations['regime'] = self.regime_labels
    
    def _calculate_regime_statistics(self, returns: pd.DataFrame) -> None:
        """
        Calculate statistics for each regime
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns data
        """
        # Initialize regime statistics
        self.regime_stats = {
            'regime_periods': {},
            'regime_correlations': {},
            'regime_returns': {},
            'regime_volatilities': {}
        }
        
        # Calculate statistics for each regime
        for regime in range(self.n_regimes):
            # Get periods for this regime
            regime_periods = self.rolling_correlations[self.rolling_correlations['regime'] == regime].index
            self.regime_stats['regime_periods'][regime] = regime_periods
            
            # Calculate average correlation for this regime
            avg_corr = self.rolling_correlations.loc[regime_periods, 'avg_correlation'].mean()
            self.regime_stats['regime_correlations'][regime] = avg_corr
            
            # Calculate returns and volatilities for this regime
            if len(regime_periods) > 0:
                regime_returns = returns.loc[regime_periods]
                
                # Average returns
                avg_returns = regime_returns.mean()
                self.regime_stats['regime_returns'][regime] = avg_returns
                
                # Volatilities
                volatilities = regime_returns.std()
                self.regime_stats['regime_volatilities'][regime] = volatilities
    
    def predict_regime(self, correlation_value: float) -> int:
        """
        Predict regime for a given correlation value
        
        Parameters:
        -----------
        correlation_value : float
            Correlation value to predict regime for
            
        Returns:
        --------
        int
            Predicted regime
        """
        if self.regime_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Reshape input for prediction
        X = np.array([[correlation_value]])
        
        if self.method == 'kmeans' or self.method == 'gmm':
            # Use fitted model for prediction
            regime = self.regime_model.predict(X)[0]
            
        elif self.method == 'threshold':
            # Use thresholds for prediction
            regime = 0
            thresholds = sorted([c.mean() for c in self.regime_model.means_])
            
            for i, threshold in enumerate(thresholds):
                if correlation_value > threshold:
                    regime = i + 1
            
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        return regime
    
    def plot_regimes(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot correlation regimes
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if self.rolling_correlations is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        plt.figure(figsize=figsize)
        
        # Create a colormap for regimes
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_regimes))
        
        # Plot rolling correlations with regime colors
        for regime in range(self.n_regimes):
            regime_data = self.rolling_correlations[self.rolling_correlations['regime'] == regime]
            plt.scatter(regime_data.index, regime_data['avg_correlation'], 
                      color=colors[regime], label=f'Regime {regime}')
        
        plt.title('Correlation Regimes')
        plt.xlabel('Date')
        plt.ylabel('Average Correlation')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_regime_statistics(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot statistics for each regime
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size
        """
        if self.regime_stats is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        plt.figure(figsize=figsize)
        
        # Plot correlation statistics
        plt.subplot(2, 2, 1)
        regimes = list(self.regime_stats['regime_correlations'].keys())
        correlations = list(self.regime_stats['regime_correlations'].values())
        
        plt.bar(regimes, correlations)
        plt.title('Average Correlation by Regime')
        plt.xlabel('Regime')
        plt.ylabel('Average Correlation')
        plt.grid(True, axis='y')
        
        # Plot return statistics
        plt.subplot(2, 2, 2)
        returns_by_regime = pd.DataFrame(self.regime_stats['regime_returns'])
        
        returns_by_regime.plot(kind='bar', ax=plt.gca())
        plt.title('Average Returns by Regime')
        plt.xlabel('Asset')
        plt.ylabel('Average Return')
        plt.grid(True, axis='y')
        
        # Plot volatility statistics
        plt.subplot(2, 2, 3)
        volatilities_by_regime = pd.DataFrame(self.regime_stats['regime_volatilities'])
        
        volatilities_by_regime.plot(kind='bar', ax=plt.gca())
        plt.title('Volatilities by Regime')
        plt.xlabel('Asset')
        plt.ylabel('Volatility')
        plt.grid(True, axis='y')
        
        # Plot regime distribution
        plt.subplot(2, 2, 4)
        regime_counts = self.rolling_correlations['regime'].value_counts().sort_index()
        
        plt.pie(regime_counts, labels=[f'Regime {r}' for r in regime_counts.index],
              autopct='%1.1f%%', startangle=90)
        plt.title('Regime Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def get_regime_periods(self) -> Dict[int, pd.DatetimeIndex]:
        """
        Get periods for each regime
        
        Returns:
        --------
        dict
            Dictionary mapping regime to periods
        """
        if self.regime_stats is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.regime_stats['regime_periods']
    
    def get_current_regime(self) -> int:
        """
        Get the most recent regime
        
        Returns:
        --------
        int
            Current regime
        """
        if self.rolling_correlations is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.rolling_correlations['regime'].iloc[-1]


class VolatilityRegimeAnalysis:
    """Analysis of volatility regimes in financial markets"""
    
    def __init__(self, window_size: int = 20, n_regimes: int = 2, method: str = 'kmeans'):
        """
        Initialize a Volatility Regime Analysis
        
        Parameters:
        -----------
        window_size : int, default=20
            Size of rolling window for volatility calculation
        n_regimes : int, default=2
            Number of volatility regimes to identify
        method : str, default='kmeans'
            Method for regime identification ('kmeans', 'gmm', 'threshold')
        """
        self.window_size = window_size
        self.n_regimes = n_regimes
        self.method = method
        self.regime_model = None
        self.regime_labels = None
        self.rolling_volatilities = None
        self.regime_stats = None
    
    def fit(self, returns: pd.DataFrame, use_log_vol: bool = True, **kwargs) -> Dict:
        """
        Fit volatility regime model to returns data
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns data
        use_log_vol : bool, default=True
            Whether to use log volatility for regime identification
        **kwargs : dict
            Additional parameters for fitting
            
        Returns:
        --------
        dict
            Dictionary with regime statistics
        """
        # Calculate rolling volatilities
        self._calculate_rolling_volatilities(returns, use_log_vol)
        
        # Identify regimes
        self._identify_regimes(**kwargs)
        
        # Calculate regime statistics
        self._calculate_regime_statistics(returns)
        
        return self.regime_stats
    
    def _calculate_rolling_volatilities(self, returns: pd.DataFrame, use_log_vol: bool) -> None:
        """
        Calculate rolling volatilities
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns data
        use_log_vol : bool
            Whether to use log volatility
        """
        # Calculate rolling volatility for each asset
        vol_dfs = []
        
        for col in returns.columns:
            rolling_vol = returns[col].rolling(window=self.window_size).std()
            vol_dfs.append(rolling_vol.rename(col))
        
        # Combine into a single DataFrame
        all_vols = pd.concat(vol_dfs, axis=1)
        
        # Calculate average volatility across assets
        avg_vol = all_vols.mean(axis=1)
        
        # Create a DataFrame to store volatilities
        self.rolling_volatilities = pd.DataFrame(
            index=returns.index[self.window_size-1:],
            columns=['avg_volatility']
        )
        
        # Apply log transformation if requested
        if use_log_vol:
            self.rolling_volatilities['avg_volatility'] = np.log(avg_vol)
        else:
            self.rolling_volatilities['avg_volatility'] = avg_vol
    
    def _identify_regimes(self, **kwargs) -> None:
        """
        Identify volatility regimes
        
        Parameters:
        -----------
        **kwargs : dict
            Additional parameters for regime identification
        """
        # Prepare data for clustering
        X = self.rolling_volatilities['avg_volatility'].values.reshape(-1, 1)
        
        if self.method == 'kmeans':
            # Use KMeans clustering
            self.regime_model = KMeans(n_clusters=self.n_regimes, random_state=42)
            self.regime_labels = self.regime_model.fit_predict(X)
            
        elif self.method == 'gmm':
            # Use Gaussian Mixture Model
            self.regime_model = GaussianMixture(n_components=self.n_regimes, random_state=42)
            self.regime_labels = self.regime_model.fit_predict(X)
            
        elif self.method == 'threshold':
            # Use threshold-based approach
            thresholds = kwargs.get('thresholds', None)
            
            if thresholds is None:
                # Calculate thresholds based on quantiles
                q_values = np.linspace(0, 1, self.n_regimes + 1)[1:-1]
                thresholds = [np.quantile(X, q) for q in q_values]
            
            # Assign regimes based on thresholds
            self.regime_labels = np.zeros(len(X), dtype=int)
            
            for i, threshold in enumerate(thresholds):
                self.regime_labels[X.flatten() > threshold] = i + 1
            
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # Add regime labels to rolling volatilities DataFrame
        self.rolling_volatilities['regime'] = self.regime_labels
    
    def _calculate_regime_statistics(self, returns: pd.DataFrame) -> None:
        """
        Calculate statistics for each regime
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns data
        """
        # Initialize regime statistics
        self.regime_stats = {
            'regime_periods': {},
            'regime_volatilities': {},
            'regime_returns': {},
            'regime_correlations': {},
            'regime_var': {},
            'regime_es': {}
        }
        
        # Calculate statistics for each regime
        for regime in range(self.n_regimes):
            # Get periods for this regime
            regime_periods = self.rolling_volatilities[self.rolling_volatilities['regime'] == regime].index
            self.regime_stats['regime_periods'][regime] = regime_periods
            
            # Calculate average volatility for this regime
            avg_vol = self.rolling_volatilities.loc[regime_periods, 'avg_volatility'].mean()
            self.regime_stats['regime_volatilities'][regime] = avg_vol
            
            # Calculate returns, correlations, VaR, and ES for this regime
            if len(regime_periods) > 0:
                regime_returns = returns.loc[regime_periods]
                
                # Average returns
                avg_returns = regime_returns.mean()
                self.regime_stats['regime_returns'][regime] = avg_returns
                
                # Correlations
                corr_matrix = regime_returns.corr()
                avg_corr = (corr_matrix.sum().sum() - len(returns.columns)) / (len(returns.columns) * (len(returns.columns) - 1))
                self.regime_stats['regime_correlations'][regime] = avg_corr
                
                # VaR (95%)
                var_95 = regime_returns.quantile(0.05)
                self.regime_stats['regime_var'][regime] = var_95
                
                # ES (95%)
                es_95 = {}
                for col in regime_returns.columns:
                    col_returns = regime_returns[col]
                    es_95[col] = col_returns[col_returns <= var_95[col]].mean()
                
                self.regime_stats['regime_es'][regime] = es_95
    
    def predict_regime(self, volatility_value: float) -> int:
        """
        Predict regime for a given volatility value
        
        Parameters:
        -----------
        volatility_value : float
            Volatility value to predict regime for
            
        Returns:
        --------
        int
            Predicted regime
        """
        if self.regime_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Reshape input for prediction
        X = np.array([[volatility_value]])
        
        if self.method == 'kmeans' or self.method == 'gmm':
            # Use fitted model for prediction
            regime = self.regime_model.predict(X)[0]
            
        elif self.method == 'threshold':
            # Use thresholds for prediction
            regime = 0
            thresholds = sorted([c.mean() for c in self.regime_model.means_])
            
            for i, threshold in enumerate(thresholds):
                if volatility_value > threshold:
                    regime = i + 1
            
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        return regime
    
    def plot_regimes(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot volatility regimes
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if self.rolling_volatilities is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        plt.figure(figsize=figsize)
        
        # Create a colormap for regimes
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_regimes))
        
        # Plot rolling volatilities with regime colors
        for regime in range(self.n_regimes):
            regime_data = self.rolling_volatilities[self.rolling_volatilities['regime'] == regime]
            plt.scatter(regime_data.index, regime_data['avg_volatility'], 
                      color=colors[regime], label=f'Regime {regime}')
        
        plt.title('Volatility Regimes')
        plt.xlabel('Date')
        plt.ylabel('Average Volatility')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_regime_statistics(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot statistics for each regime
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size
        """
        if self.regime_stats is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        plt.figure(figsize=figsize)
        
        # Plot volatility statistics
        plt.subplot(2, 2, 1)
        regimes = list(self.regime_stats['regime_volatilities'].keys())
        volatilities = list(self.regime_stats['regime_volatilities'].values())
        
        plt.bar(regimes, volatilities)
        plt.title('Average Volatility by Regime')
        plt.xlabel('Regime')
        plt.ylabel('Average Volatility')
        plt.grid(True, axis='y')
        
        # Plot return statistics
        plt.subplot(2, 2, 2)
        returns_by_regime = pd.DataFrame(self.regime_stats['regime_returns'])
        
        returns_by_regime.plot(kind='bar', ax=plt.gca())
        plt.title('Average Returns by Regime')
        plt.xlabel('Asset')
        plt.ylabel('Average Return')
        plt.grid(True, axis='y')
        
        # Plot correlation statistics
        plt.subplot(2, 2, 3)
        correlations = list(self.regime_stats['regime_correlations'].values())
        
        plt.bar(regimes, correlations)
        plt.title('Average Correlation by Regime')
        plt.xlabel('Regime')
        plt.ylabel('Average Correlation')
        plt.grid(True, axis='y')
        
        # Plot regime distribution
        plt.subplot(2, 2, 4)
        regime_counts = self.rolling_volatilities['regime'].value_counts().sort_index()
        
        plt.pie(regime_counts, labels=[f'Regime {r}' for r in regime_counts.index],
              autopct='%1.1f%%', startangle=90)
        plt.title('Regime Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def plot_risk_metrics(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot risk metrics by regime
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size
        """
        if self.regime_stats is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        plt.figure(figsize=figsize)
        
        # Plot VaR by regime
        plt.subplot(2, 1, 1)
        var_by_regime = pd.DataFrame(self.regime_stats['regime_var'])
        
        var_by_regime.plot(kind='bar', ax=plt.gca())
        plt.title('Value at Risk (95%) by Regime')
        plt.xlabel('Asset')
        plt.ylabel('VaR')
        plt.grid(True, axis='y')
        
        # Plot ES by regime
        plt.subplot(2, 1, 2)
        es_by_regime = pd.DataFrame(self.regime_stats['regime_es'])
        
        es_by_regime.plot(kind='bar', ax=plt.gca())
        plt.title('Expected Shortfall (95%) by Regime')
        plt.xlabel('Asset')
        plt.ylabel('ES')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def get_regime_periods(self) -> Dict[int, pd.DatetimeIndex]:
        """
        Get periods for each regime
        
        Returns:
        --------
        dict
            Dictionary mapping regime to periods
        """
        if self.regime_stats is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.regime_stats['regime_periods']
    
    def get_current_regime(self) -> int:
        """
        Get the most recent regime
        
        Returns:
        --------
        int
            Current regime
        """
        if self.rolling_volatilities is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.rolling_volatilities['regime'].iloc[-1]


class MarkovRegimeSwitchingModel:
    """Markov regime-switching model for financial time series"""
    
    def __init__(self, n_regimes: int = 2, n_iter: int = 1000):
        """
        Initialize a Markov Regime-Switching Model
        
        Parameters:
        -----------
        n_regimes : int, default=2
            Number of regimes
        n_iter : int, default=1000
            Number of iterations for EM algorithm
        """
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.params = None
        self.regime_probs = None
        self.smoothed_probs = None
        self.filtered_probs = None
        self.predicted_probs = None
        self.log_likelihood = None
    
    def fit(self, returns: pd.Series, **kwargs) -> Dict:
        """
        Fit Markov regime-switching model to returns data
        
        Parameters:
        -----------
        returns : pd.Series
            Asset returns data
        **kwargs : dict
            Additional parameters for fitting
            
        Returns:
        --------
        dict
            Dictionary with fitted parameters
        """
        # Initialize parameters
        self._initialize_parameters(returns)
        
        # Run EM algorithm
        self._run_em_algorithm(returns)
        
        # Calculate smoothed probabilities
        self._calculate_smoothed_probabilities()
        
        return self.params
    
    def _initialize_parameters(self, returns: pd.Series) -> None:
        """
        Initialize model parameters
        
        Parameters:
        -----------
        returns : pd.Series
            Asset returns data
        """
        # Initialize means, variances, and transition probabilities
        means = np.linspace(returns.mean() - returns.std(), returns.mean() + returns.std(), self.n_regimes)
        variances = np.full(self.n_regimes, returns.var())
        
        # Initialize transition matrix with high persistence
        transition_matrix = np.full((self.n_regimes, self.n_regimes), 0.1 / (self.n_regimes - 1))
        np.fill_diagonal(transition_matrix, 0.9)
        
        # Initialize parameters
        self.params = {
            'means': means,
            'variances': variances,
            'transition_matrix': transition_matrix
        }
        
        # Initialize regime probabilities
        self.regime_probs = np.full(self.n_regimes, 1.0 / self.n_regimes)
    
    def _run_em_algorithm(self, returns: pd.Series) -> None:
        """
        Run Expectation-Maximization algorithm
        
        Parameters:
        -----------
        returns : pd.Series
            Asset returns data
        """
        n_samples = len(returns)
        
        # Initialize arrays for filtered and predicted probabilities
        self.filtered_probs = np.zeros((n_samples, self.n_regimes))
        self.predicted_probs = np.zeros((n_samples, self.n_regimes))
        
        # Run EM iterations
        for iteration in range(self.n_iter):
            # E-step: Calculate filtered and predicted probabilities
            log_likelihood = self._e_step(returns)
            
            # M-step: Update parameters
            self._m_step(returns)
            
            # Check for convergence
            if iteration > 0 and abs(log_likelihood - self.log_likelihood) < 1e-6:
                logger.info(f"Converged after {iteration+1} iterations")
                break
            
            self.log_likelihood = log_likelihood
    
    def _e_step(self, returns: pd.Series) -> float:
        """
        E-step of EM algorithm
        
        Parameters:
        -----------
        returns : pd.Series
            Asset returns data
            
        Returns:
        --------
        float
            Log-likelihood
        """
        n_samples = len(returns)
        means = self.params['means']
        variances = self.params['variances']
        transition_matrix = self.params['transition_matrix']
        
        # Initialize log-likelihood
        log_likelihood = 0.0
        
        # Forward algorithm
        for t in range(n_samples):
            # Calculate conditional densities
            densities = np.zeros(self.n_regimes)
            for j in range(self.n_regimes):
                densities[j] = stats.norm.pdf(returns.iloc[t], means[j], np.sqrt(variances[j]))
            
            if t == 0:
                # Initial step
                self.predicted_probs[t] = self.regime_probs
            else:
                # Prediction step
                self.predicted_probs[t] = self.filtered_probs[t-1] @ transition_matrix
            
            # Update step
            joint_probs = self.predicted_probs[t] * densities
            sum_joint_probs = np.sum(joint_probs)
            
            if sum_joint_probs > 0:
                self.filtered_probs[t] = joint_probs / sum_joint_probs
                log_likelihood += np.log(sum_joint_probs)
            else:
                # Handle numerical issues
                self.filtered_probs[t] = self.predicted_probs[t]
        
        return log_likelihood
    
    def _m_step(self, returns: pd.Series) -> None:
        """
        M-step of EM algorithm
        
        Parameters:
        -----------
        returns : pd.Series
            Asset returns data
        """
        n_samples = len(returns)
        
        # Update means and variances
        means = np.zeros(self.n_regimes)
        variances = np.zeros(self.n_regimes)
        
        for j in range(self.n_regimes):
            # Calculate weighted sum for mean
            weighted_sum = np.sum(self.filtered_probs[:, j] * returns.values)
            sum_weights = np.sum(self.filtered_probs[:, j])
            
            if sum_weights > 0:
                means[j] = weighted_sum / sum_weights
                
                # Calculate weighted sum for variance
                weighted_var_sum = np.sum(self.filtered_probs[:, j] * (returns.values - means[j])**2)
                variances[j] = weighted_var_sum / sum_weights
            else:
                # Handle numerical issues
                means[j] = self.params['means'][j]
                variances[j] = self.params['variances'][j]
        
        # Update transition matrix
        transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                # Calculate expected transitions from i to j
                expected_transitions = 0.0
                
                for t in range(1, n_samples):
                    joint_prob = self.filtered_probs[t-1, i] * self.params['transition_matrix'][i, j] * \
                                stats.norm.pdf(returns.iloc[t], means[j], np.sqrt(variances[j])) / \
                                self.predicted_probs[t, j]
                    
                    expected_transitions += joint_prob
                
                # Normalize by total transitions from i
                total_transitions = np.sum(self.filtered_probs[:-1, i])
                
                if total_transitions > 0:
                    transition_matrix[i, j] = expected_transitions / total_transitions
                else:
                    # Handle numerical issues
                    transition_matrix[i, j] = self.params['transition_matrix'][i, j]
        
        # Normalize transition matrix rows
        for i in range(self.n_regimes):
            row_sum = np.sum(transition_matrix[i])
            if row_sum > 0:
                transition_matrix[i] /= row_sum
        
        # Update parameters
        self.params['means'] = means
        self.params['variances'] = variances
        self.params['transition_matrix'] = transition_matrix
    
    def _calculate_smoothed_probabilities(self) -> None:
        """Calculate smoothed regime probabilities using backward algorithm"""
        n_samples = len(self.filtered_probs)
        
        # Initialize smoothed probabilities
        self.smoothed_probs = np.zeros_like(self.filtered_probs)
        self.smoothed_probs[-1] = self.filtered_probs[-1]
        
        # Backward algorithm
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_regimes):
                self.smoothed_probs[t, i] = self.filtered_probs[t, i] * np.sum(
                    self.params['transition_matrix'][i] * self.smoothed_probs[t+1] / self.predicted_probs[t+1]
                )
    
    def predict_regime(self, returns: pd.Series, window_size: int = 20) -> np.ndarray:
        """
        Predict regimes for new data
        
        Parameters:
        -----------
        returns : pd.Series
            New returns data
        window_size : int, default=20
            Size of rolling window for prediction
            
        Returns:
        --------
        np.ndarray
            Predicted regimes
        """
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_samples = len(returns)
        predicted_regimes = np.zeros(n_samples)
        
        # Use filtered probabilities for prediction
        filtered_probs = np.zeros((n_samples, self.n_regimes))
        predicted_probs = np.zeros((n_samples, self.n_regimes))
        
        # Initialize with steady-state probabilities
        if n_samples > 0:
            predicted_probs[0] = self._calculate_steady_state()
        
        # Forward algorithm
        for t in range(n_samples):
            # Calculate conditional densities
            densities = np.zeros(self.n_regimes)
            for j in range(self.n_regimes):
                densities[j] = stats.norm.pdf(returns.iloc[t], self.params['means'][j], np.sqrt(self.params['variances'][j]))
            
            if t == 0:
                # Initial step
                pass
            else:
                # Prediction step
                predicted_probs[t] = filtered_probs[t-1] @ self.params['transition_matrix']
            
            # Update step
            joint_probs = predicted_probs[t] * densities
            sum_joint_probs = np.sum(joint_probs)
            
            if sum_joint_probs > 0:
                filtered_probs[t] = joint_probs / sum_joint_probs
            else:
                # Handle numerical issues
                filtered_probs[t] = predicted_probs[t]
            
            # Predict regime
            predicted_regimes[t] = np.argmax(filtered_probs[t])
        
        return predicted_regimes
    
    def _calculate_steady_state(self) -> np.ndarray:
        """
        Calculate steady-state probabilities of the Markov chain
        
        Returns:
        --------
        np.ndarray
            Steady-state probabilities
        """
        # Create transition matrix with an additional constraint for steady state
        A = np.vstack([
            self.params['transition_matrix'].T - np.eye(self.n_regimes),
            np.ones(self.n_regimes)
        ])
        
        b = np.zeros(self.n_regimes + 1)
        b[-1] = 1.0
        
        # Solve the linear system
        try:
            steady_state = np.linalg.lstsq(A, b, rcond=None)[0]
            return steady_state
        except:
            # Fallback to uniform distribution
            return np.full(self.n_regimes, 1.0 / self.n_regimes)
    
    def plot_regime_probabilities(self, returns: pd.Series, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot regime probabilities
        
        Parameters:
        -----------
        returns : pd.Series
            Returns data
        figsize : tuple, default=(12, 8)
            Figure size
        """
        if self.smoothed_probs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        plt.figure(figsize=figsize)
        
        # Plot returns
        plt.subplot(2, 1, 1)
        plt.plot(returns.index, returns.values)
        plt.title('Returns')
        plt.grid(True)
        
        # Plot regime probabilities
        plt.subplot(2, 1, 2)
        
        for j in range(self.n_regimes):
            plt.plot(returns.index, self.smoothed_probs[:, j], label=f'Regime {j}')
        
        plt.title('Regime Probabilities')
        plt.xlabel('Date')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_regime_statistics(self, returns: pd.Series, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot statistics for each regime
        
        Parameters:
        -----------
        returns : pd.Series
            Returns data
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if self.smoothed_probs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Assign each point to the most likely regime
        regimes = np.argmax(self.smoothed_probs, axis=1)
        
        # Calculate statistics for each regime
        regime_stats = {}
        
        for j in range(self.n_regimes):
            regime_returns = returns.iloc[regimes == j]
            
            if len(regime_returns) > 0:
                regime_stats[f'Regime {j}'] = {
                    'Mean': regime_returns.mean(),
                    'Std Dev': regime_returns.std(),
                    'Skewness': stats.skew(regime_returns),
                    'Kurtosis': stats.kurtosis(regime_returns),
                    'VaR (95%)': np.percentile(regime_returns, 5),
                    'ES (95%)': regime_returns[regime_returns <= np.percentile(regime_returns, 5)].mean(),
                    'Count': len(regime_returns),
                    'Frequency': len(regime_returns) / len(returns)
                }
        
        # Create DataFrame for plotting
        stats_df = pd.DataFrame(regime_stats)
        
        # Plot statistics
        plt.figure(figsize=figsize)
        
        # Plot mean and standard deviation
        plt.subplot(1, 2, 1)
        stats_df.loc[['Mean', 'Std Dev']].plot(kind='bar', ax=plt.gca())
        plt.title('Mean and Standard Deviation by Regime')
        plt.grid(True, axis='y')
        
        # Plot VaR and ES
        plt.subplot(1, 2, 2)
        stats_df.loc[['VaR (95%)', 'ES (95%)']].plot(kind='bar', ax=plt.gca())
        plt.title('VaR and ES by Regime')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        # Print additional statistics
        print("Regime Statistics:")
        print(stats_df.T)
        
        # Print transition matrix
        print("\nTransition Matrix:")
        transition_df = pd.DataFrame(
            self.params['transition_matrix'],
            index=[f'From Regime {i}' for i in range(self.n_regimes)],
            columns=[f'To Regime {j}' for j in range(self.n_regimes)]
        )
        print(transition_df)
    
    def get_regime_periods(self, returns: pd.Series) -> Dict[int, pd.DatetimeIndex]:
        """
        Get periods for each regime
        
        Parameters:
        -----------
        returns : pd.Series
            Returns data
            
        Returns:
        --------
        dict
            Dictionary mapping regime to periods
        """
        if self.smoothed_probs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Assign each point to the most likely regime
        regimes = np.argmax(self.smoothed_probs, axis=1)
        
        # Get periods for each regime
        regime_periods = {}
        
        for j in range(self.n_regimes):
            regime_periods[j] = returns.index[regimes == j]
        
        return regime_periods
    
    def get_current_regime(self) -> int:
        """
        Get the most recent regime
        
        Returns:
        --------
        int
            Current regime
        """
        if self.smoothed_probs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return np.argmax(self.smoothed_probs[-1])


class DynamicConditionalCorrelation:
    """Dynamic Conditional Correlation (DCC) model for time-varying correlations"""
    
    def __init__(self, alpha: float = 0.05, beta: float = 0.94):
        """
        Initialize a DCC model
        
        Parameters:
        -----------
        alpha : float, default=0.05
            GARCH parameter for news impact
        beta : float, default=0.94
            GARCH parameter for persistence
        """
        self.alpha = alpha
        self.beta = beta
        self.fitted = False
        self.params = {}
        self.conditional_correlations = None
        self.conditional_covariances = None
        self.conditional_volatilities = None
        self.standardized_residuals = None
    
    def fit(self, returns: pd.DataFrame, garch_params: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Fit DCC model to returns data
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns data
        garch_params : dict, optional
            Parameters for univariate GARCH models
        **kwargs : dict
            Additional parameters for fitting
            
        Returns:
        --------
        dict
            Dictionary with fitted parameters
        """
        # Fit univariate GARCH models
        self._fit_univariate_garch(returns, garch_params)
        
        # Fit DCC model
        self._fit_dcc_model(returns)
        
        self.fitted = True
        
        return self.params
    
    def _fit_univariate_garch(self, returns: pd.DataFrame, garch_params: Optional[Dict]) -> None:
        """
        Fit univariate GARCH models to each asset
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns data
        garch_params : dict, optional
            Parameters for univariate GARCH models
        """
        n_samples, n_assets = returns.shape
        
        # Initialize conditional volatilities and standardized residuals
        self.conditional_volatilities = np.zeros((n_samples, n_assets))
        self.standardized_residuals = np.zeros((n_samples, n_assets))
        
        # Default GARCH parameters if not provided
        if garch_params is None:
            garch_params = {
                'omega': 0.05,
                'alpha': 0.1,
                'beta': 0.85
            }
        
        # Fit GARCH model for each asset
        for i in range(n_assets):
            asset_returns = returns.iloc[:, i].values
            
            # Initialize conditional variance
            conditional_variance = np.zeros(n_samples)
            conditional_variance[0] = np.var(asset_returns)
            
            # GARCH recursion
            for t in range(1, n_samples):
                conditional_variance[t] = garch_params['omega'] + \
                                        garch_params['alpha'] * asset_returns[t-1]**2 + \
                                        garch_params['beta'] * conditional_variance[t-1]
            
            # Calculate conditional volatility and standardized residuals
            self.conditional_volatilities[:, i] = np.sqrt(conditional_variance)
            self.standardized_residuals[:, i] = asset_returns / self.conditional_volatilities[:, i]
        
        # Store GARCH parameters
        self.params['garch'] = garch_params
    
    def _fit_dcc_model(self, returns: pd.DataFrame) -> None:
        """
        Fit DCC model to standardized residuals
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns data
        """
        n_samples, n_assets = returns.shape
        
        # Calculate unconditional correlation matrix
        unconditional_corr = np.corrcoef(self.standardized_residuals, rowvar=False)
        
        # Initialize conditional correlation matrix
        self.conditional_correlations = np.zeros((n_samples, n_assets, n_assets))
        self.conditional_correlations[0] = unconditional_corr
        
        # Initialize Q matrix
        Q_bar = np.cov(self.standardized_residuals, rowvar=False)
        Q = np.zeros((n_samples, n_assets, n_assets))
        Q[0] = Q_bar
        
        # DCC recursion
        for t in range(1, n_samples):
            # Calculate Q matrix
            epsilon = self.standardized_residuals[t-1].reshape(-1, 1)
            Q[t] = (1 - self.alpha - self.beta) * Q_bar + \
                  self.alpha * (epsilon @ epsilon.T) + \
                  self.beta * Q[t-1]
            
            # Calculate conditional correlation matrix
            Q_diag = np.diag(1 / np.sqrt(np.diag(Q[t])))
            self.conditional_correlations[t] = Q_diag @ Q[t] @ Q_diag
        
        # Calculate conditional covariances
        self.conditional_covariances = np.zeros((n_samples, n_assets, n_assets))
        
        for t in range(n_samples):
            vol_diag = np.diag(self.conditional_volatilities[t])
            self.conditional_covariances[t] = vol_diag @ self.conditional_correlations[t] @ vol_diag
        
        # Store DCC parameters
        self.params['dcc'] = {
            'alpha': self.alpha,
            'beta': self.beta,
            'unconditional_corr': unconditional_corr
        }
    
    def forecast(self, horizon: int = 1, last_returns: Optional[np.ndarray] = None) -> Dict:
        """
        Forecast conditional correlations and covariances
        
        Parameters:
        -----------
        horizon : int, default=1
            Forecast horizon
        last_returns : np.ndarray, optional
            Last observed returns (if None, uses last returns from fitted data)
            
        Returns:
        --------
        dict
            Dictionary with forecasted values
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_assets = self.conditional_correlations.shape[1]
        
        # Get last values
        last_vol = self.conditional_volatilities[-1]
        last_corr = self.conditional_correlations[-1]
        last_cov = self.conditional_covariances[-1]
        
        if last_returns is None:
            last_std_resid = self.standardized_residuals[-1]
        else:
            last_std_resid = last_returns / last_vol
        
        # Initialize forecasts
        forecasted_vol = np.zeros((horizon, n_assets))
        forecasted_corr = np.zeros((horizon, n_assets, n_assets))
        forecasted_cov = np.zeros((horizon, n_assets, n_assets))
        
        # Get parameters
        garch_params = self.params['garch']
        dcc_params = self.params['dcc']
        
        # Forecast for each horizon
        for h in range(horizon):
            if h == 0:
                # Use last values for first step
                last_var = last_vol**2
                last_Q = last_corr.copy()
                
                # Forecast volatility
                forecasted_var = garch_params['omega'] + \
                               garch_params['alpha'] * (last_std_resid * last_vol)**2 + \
                               garch_params['beta'] * last_var
                
                forecasted_vol[h] = np.sqrt(forecasted_var)
                
                # Forecast correlation
                Q_bar = dcc_params['unconditional_corr']
                forecasted_Q = (1 - dcc_params['alpha'] - dcc_params['beta']) * Q_bar + \
                             dcc_params['alpha'] * (last_std_resid.reshape(-1, 1) @ last_std_resid.reshape(1, -1)) + \
                             dcc_params['beta'] * last_Q
                
                # Normalize to get correlation matrix
                Q_diag = np.diag(1 / np.sqrt(np.diag(forecasted_Q)))
                forecasted_corr[h] = Q_diag @ forecasted_Q @ Q_diag
                
            else:
                # Use previous forecasts
                # For volatility, use GARCH formula
                forecasted_var = garch_params['omega'] + \
                               (garch_params['alpha'] + garch_params['beta']) * forecasted_vol[h-1]**2
                
                forecasted_vol[h] = np.sqrt(forecasted_var)
                
                # For correlation, use DCC formula (mean reversion to unconditional correlation)
                Q_bar = dcc_params['unconditional_corr']
                alpha_beta_sum = dcc_params['alpha'] + dcc_params['beta']
                
                forecasted_Q = (1 - alpha_beta_sum) * Q_bar + alpha_beta_sum * forecasted_Q
                
                # Normalize to get correlation matrix
                Q_diag = np.diag(1 / np.sqrt(np.diag(forecasted_Q)))
                forecasted_corr[h] = Q_diag @ forecasted_Q @ Q_diag
            
            # Calculate covariance
            vol_diag = np.diag(forecasted_vol[h])
            forecasted_cov[h] = vol_diag @ forecasted_corr[h] @ vol_diag
        
        return {
            'volatility': forecasted_vol,
            'correlation': forecasted_corr,
            'covariance': forecasted_cov
        }
    
    def plot_conditional_correlations(self, returns: pd.DataFrame, 
                                   asset_pairs: Optional[List[Tuple[int, int]]] = None,
                                   figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot conditional correlations
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns data
        asset_pairs : list of tuple, optional
            List of asset pairs to plot (if None, plots all pairs)
        figsize : tuple, default=(12, 8)
            Figure size
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_assets = returns.shape[1]
        
        # Default to all pairs if not specified
        if asset_pairs is None:
            asset_pairs = [(i, j) for i in range(n_assets) for j in range(i+1, n_assets)]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot conditional correlations for each pair
        for i, (asset1, asset2) in enumerate(asset_pairs):
            if i >= 9:  # Limit to 9 subplots
                break
            
            plt.subplot(3, 3, i+1)
            
            # Extract correlation series
            corr_series = [self.conditional_correlations[t, asset1, asset2] for t in range(len(returns))]
            
            plt.plot(returns.index, corr_series)
            plt.title(f'{returns.columns[asset1]} - {returns.columns[asset2]}')
            plt.ylim(-1, 1)
            plt.grid(True)
            
            if i >= 6:  # Add x-label for bottom row
                plt.xlabel('Date')
            
            if i % 3 == 0:  # Add y-label for left column
                plt.ylabel('Correlation')
        
        plt.tight_layout()
        plt.show()
    
    def plot_conditional_volatilities(self, returns: pd.DataFrame, 
                                    figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot conditional volatilities
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns data
        figsize : tuple, default=(12, 8)
            Figure size
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_assets = returns.shape[1]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot conditional volatilities for each asset
        for i in range(min(n_assets, 9)):  # Limit to 9 subplots
            plt.subplot(3, 3, i+1)
            
            plt.plot(returns.index, self.conditional_volatilities[:, i])
            plt.title(returns.columns[i])
            plt.grid(True)
            
            if i >= 6:  # Add x-label for bottom row
                plt.xlabel('Date')
            
            if i % 3 == 0:  # Add y-label for left column
                plt.ylabel('Volatility')
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_distribution(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot distribution of conditional correlations
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_samples, n_assets, _ = self.conditional_correlations.shape
        
        # Extract all pairwise correlations
        all_correlations = []
        
        for t in range(n_samples):
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    all_correlations.append(self.conditional_correlations[t, i, j])
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot histogram
        plt.subplot(1, 2, 1)
        plt.hist(all_correlations, bins=50, alpha=0.7)
        plt.title('Distribution of Conditional Correlations')
        plt.xlabel('Correlation')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Plot time evolution of average correlation
        plt.subplot(1, 2, 2)
        
        avg_correlations = []
        for t in range(n_samples):
            corr_sum = 0
            count = 0
            
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    corr_sum += self.conditional_correlations[t, i, j]
                    count += 1
            
            avg_correlations.append(corr_sum / count)
        
        plt.plot(range(n_samples), avg_correlations)
        plt.title('Average Conditional Correlation Over Time')
        plt.xlabel('Time')
        plt.ylabel('Average Correlation')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


class RegimeBasedRiskModel:
    """Risk model that incorporates regime information"""
    
    def __init__(self, correlation_model=None, volatility_model=None):
        """
        Initialize a Regime-Based Risk Model
        
        Parameters:
        -----------
        correlation_model : object, optional
            Model for correlation regimes
        volatility_model : object, optional
            Model for volatility regimes
        """
        self.correlation_model = correlation_model
        self.volatility_model = volatility_model
        self.regime_stats = {}
        self.fitted = False
    
    def fit(self, returns: pd.DataFrame, **kwargs) -> Dict:
        """
        Fit regime models to returns data
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns data
        **kwargs : dict
            Additional parameters for fitting
            
        Returns:
        --------
        dict
            Dictionary with regime statistics
        """
        # Fit correlation model if provided
        if self.correlation_model is not None:
            corr_stats = self.correlation_model.fit(returns, **kwargs)
            self.regime_stats['correlation'] = corr_stats
        
        # Fit volatility model if provided
        if self.volatility_model is not None:
            vol_stats = self.volatility_model.fit(returns, **kwargs)
            self.regime_stats['volatility'] = vol_stats
        
        # Calculate combined regime statistics
        self._calculate_combined_regime_statistics(returns)
        
        self.fitted = True
        
        return self.regime_stats
    
    def _calculate_combined_regime_statistics(self, returns: pd.DataFrame) -> None:
        """
        Calculate statistics for combined regimes
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns data
        """
        # Check if both models are fitted
        if self.correlation_model is None or self.volatility_model is None:
            return
        
        # Get regime labels
        if hasattr(self.correlation_model, 'rolling_correlations') and hasattr(self.volatility_model, 'rolling_volatilities'):
            corr_regimes = self.correlation_model.rolling_correlations['regime']
            vol_regimes = self.volatility_model.rolling_volatilities['regime']
            
            # Ensure same length
            min_length = min(len(corr_regimes), len(vol_regimes))
            corr_regimes = corr_regimes.iloc[:min_length]
            vol_regimes = vol_regimes.iloc[:min_length]
            
            # Create combined regime labels
            n_corr_regimes = self.correlation_model.n_regimes
            n_vol_regimes = self.volatility_model.n_regimes
            
            combined_regimes = corr_regimes * n_vol_regimes + vol_regimes
            unique_regimes = combined_regimes.unique()
            
            # Initialize combined regime statistics
            self.regime_stats['combined'] = {
                'regime_periods': {},
                'regime_returns': {},
                'regime_volatilities': {},
                'regime_correlations': {},
                'regime_var': {},
                'regime_es': {}
            }
            
            # Calculate statistics for each combined regime
            for regime in unique_regimes:
                # Get periods for this regime
                regime_periods = combined_regimes[combined_regimes == regime].index
                self.regime_stats['combined']['regime_periods'][regime] = regime_periods
                
                # Calculate returns, volatilities, correlations, VaR, and ES for this regime
                if len(regime_periods) > 0:
                    regime_returns = returns.loc[regime_periods]
                    
                    # Average returns
                    avg_returns = regime_returns.mean()
                    self.regime_stats['combined']['regime_returns'][regime] = avg_returns
                    
                    # Volatilities
                    volatilities = regime_returns.std()
                    self.regime_stats['combined']['regime_volatilities'][regime] = volatilities
                    
                    # Correlations
                    corr_matrix = regime_returns.corr()
                    avg_corr = (corr_matrix.sum().sum() - len(returns.columns)) / (len(returns.columns) * (len(returns.columns) - 1))
                    self.regime_stats['combined']['regime_correlations'][regime] = avg_corr
                    
                    # VaR (95%)
                    var_95 = regime_returns.quantile(0.05)
                    self.regime_stats['combined']['regime_var'][regime] = var_95
                    
                    # ES (95%)
                    es_95 = {}
                    for col in regime_returns.columns:
                        col_returns = regime_returns[col]
                        es_95[col] = col_returns[col_returns <= var_95[col]].mean()
                    
                    self.regime_stats['combined']['regime_es'][regime] = es_95
    
    def predict_regime(self, correlation_value: Optional[float] = None, 
                     volatility_value: Optional[float] = None) -> Dict[str, int]:
        """
        Predict regime for given correlation and volatility values
        
        Parameters:
        -----------
        correlation_value : float, optional
            Correlation value to predict regime for
        volatility_value : float, optional
            Volatility value to predict regime for
            
        Returns:
        --------
        dict
            Dictionary with predicted regimes
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predicted_regimes = {}
        
        # Predict correlation regime if model is available
        if self.correlation_model is not None and correlation_value is not None:
            predicted_regimes['correlation'] = self.correlation_model.predict_regime(correlation_value)
        
        # Predict volatility regime if model is available
        if self.volatility_model is not None and volatility_value is not None:
            predicted_regimes['volatility'] = self.volatility_model.predict_regime(volatility_value)
        
        # Predict combined regime if both models are available
        if 'correlation' in predicted_regimes and 'volatility' in predicted_regimes:
            n_vol_regimes = self.volatility_model.n_regimes
            predicted_regimes['combined'] = predicted_regimes['correlation'] * n_vol_regimes + predicted_regimes['volatility']
        
        return predicted_regimes
    
    def estimate_var(self, returns: pd.DataFrame, alpha: float = 0.05, 
                   method: str = 'historical', **kwargs) -> Dict:
        """
        Estimate Value at Risk (VaR) based on current regime
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns data
        alpha : float, default=0.05
            Confidence level (e.g., 0.05 for 95% VaR)
        method : str, default='historical'
            Method for VaR estimation ('historical', 'parametric', 'monte_carlo')
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        dict
            Dictionary with VaR estimates
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get current regimes
        current_regimes = self._get_current_regimes()
        
        # Initialize VaR estimates
        var_estimates = {}
        
        # Estimate VaR for each regime type
        for regime_type, regime in current_regimes.items():
            if regime_type == 'combined' and regime in self.regime_stats['combined']['regime_var']:
                # Use pre-calculated VaR for combined regime
                var_estimates[regime_type] = self.regime_stats['combined']['regime_var'][regime]
            else:
                # Calculate VaR based on regime-specific data
                regime_periods = self._get_regime_periods(regime_type, regime)
                
                if len(regime_periods) > 0:
                    regime_returns = returns.loc[regime_periods]
                    
                    if method == 'historical':
                        # Historical VaR
                        var_estimates[regime_type] = regime_returns.quantile(alpha)
                    
                    elif method == 'parametric':
                        # Parametric VaR (assuming normal distribution)
                        mean = regime_returns.mean()
                        std = regime_returns.std()
                        z_score = stats.norm.ppf(alpha)
                        var_estimates[regime_type] = mean + z_score * std
                    
                    elif method == 'monte_carlo':
                        # Monte Carlo VaR
                        n_simulations = kwargs.get('n_simulations', 10000)
                        horizon = kwargs.get('horizon', 1)
                        
                        # Estimate parameters
                        mean = regime_returns.mean()
                        cov = regime_returns.cov()
                        
                        # Generate simulations
                        simulations = np.random.multivariate_normal(
                            mean.values,
                            cov.values,
                            n_simulations
                        )
                        
                        # Calculate portfolio returns
                        weights = kwargs.get('weights', np.ones(len(mean)) / len(mean))
                        portfolio_returns = simulations @ weights
                        
                        # Calculate VaR
                        var_estimates[regime_type] = np.percentile(portfolio_returns, alpha * 100)
                    
                    else:
                        raise ValueError(f"Unsupported method: {method}")
        
        return var_estimates
    
    def estimate_es(self, returns: pd.DataFrame, alpha: float = 0.05, 
                  method: str = 'historical', **kwargs) -> Dict:
        """
        Estimate Expected Shortfall (ES) based on current regime
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns data
        alpha : float, default=0.05
            Confidence level (e.g., 0.05 for 95% ES)
        method : str, default='historical'
            Method for ES estimation ('historical', 'parametric', 'monte_carlo')
        **kwargs : dict
            Additional parameters for estimation
            
        Returns:
        --------
        dict
            Dictionary with ES estimates
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get current regimes
        current_regimes = self._get_current_regimes()
        
        # Initialize ES estimates
        es_estimates = {}
        
        # Estimate ES for each regime type
        for regime_type, regime in current_regimes.items():
            if regime_type == 'combined' and regime in self.regime_stats['combined']['regime_es']:
                # Use pre-calculated ES for combined regime
                es_estimates[regime_type] = self.regime_stats['combined']['regime_es'][regime]
            else:
                # Calculate ES based on regime-specific data
                regime_periods = self._get_regime_periods(regime_type, regime)
                
                if len(regime_periods) > 0:
                    regime_returns = returns.loc[regime_periods]
                    
                    if method == 'historical':
                        # Historical ES
                        var = regime_returns.quantile(alpha)
                        es = {}
                        
                        for col in regime_returns.columns:
                            col_returns = regime_returns[col]
                            es[col] = col_returns[col_returns <= var[col]].mean()
                        
                        es_estimates[regime_type] = es
                    
                    elif method == 'parametric':
                        # Parametric ES (assuming normal distribution)
                        mean = regime_returns.mean()
                        std = regime_returns.std()
                        z_score = stats.norm.ppf(alpha)
                        pdf_z = stats.norm.pdf(z_score)
                        es_estimates[regime_type] = mean - std * pdf_z / alpha
                    
                    elif method == 'monte_carlo':
                        # Monte Carlo ES
                        n_simulations = kwargs.get('n_simulations', 10000)
                        horizon = kwargs.get('horizon', 1)
                        
                        # Estimate parameters
                        mean = regime_returns.mean()
                        cov = regime_returns.cov()
                        
                        # Generate simulations
                        simulations = np.random.multivariate_normal(
                            mean.values,
                            cov.values,
                            n_simulations
                        )
                        
                        # Calculate portfolio returns
                        weights = kwargs.get('weights', np.ones(len(mean)) / len(mean))
                        portfolio_returns = simulations @ weights
                        
                        # Calculate VaR
                        var = np.percentile(portfolio_returns, alpha * 100)
                        
                        # Calculate ES
                        es_estimates[regime_type] = portfolio_returns[portfolio_returns <= var].mean()
                    
                    else:
                        raise ValueError(f"Unsupported method: {method}")
        
        return es_estimates
    
    def _get_current_regimes(self) -> Dict[str, int]:
        """
        Get current regimes
        
        Returns:
        --------
        dict
            Dictionary with current regimes
        """
        current_regimes = {}
        
        # Get current correlation regime if model is available
        if self.correlation_model is not None:
            current_regimes['correlation'] = self.correlation_model.get_current_regime()
        
        # Get current volatility regime if model is available
        if self.volatility_model is not None:
            current_regimes['volatility'] = self.volatility_model.get_current_regime()
        
        # Calculate combined regime if both models are available
        if 'correlation' in current_regimes and 'volatility' in current_regimes:
            n_vol_regimes = self.volatility_model.n_regimes
            current_regimes['combined'] = current_regimes['correlation'] * n_vol_regimes + current_regimes['volatility']
        
        return current_regimes
    
    def _get_regime_periods(self, regime_type: str, regime: int) -> pd.DatetimeIndex:
        """
        Get periods for a specific regime
        
        Parameters:
        -----------
        regime_type : str
            Type of regime ('correlation', 'volatility', 'combined')
        regime : int
            Regime index
            
        Returns:
        --------
        pd.DatetimeIndex
            Periods for the specified regime
        """
        if regime_type not in self.regime_stats:
            return pd.DatetimeIndex([])
        
        if 'regime_periods' not in self.regime_stats[regime_type]:
            return pd.DatetimeIndex([])
        
        if regime not in self.regime_stats[regime_type]['regime_periods']:
            return pd.DatetimeIndex([])
        
        return self.regime_stats[regime_type]['regime_periods'][regime]
    
    def plot_regime_risk_metrics(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot risk metrics by regime
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Check if combined regimes are available
        if 'combined' not in self.regime_stats:
            raise ValueError("Combined regimes not available. Ensure both correlation and volatility models are fitted.")
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot volatility by regime
        plt.subplot(2, 2, 1)
        volatilities_by_regime = pd.DataFrame(self.regime_stats['combined']['regime_volatilities'])
        
        volatilities_by_regime.plot(kind='bar', ax=plt.gca())
        plt.title('Volatility by Regime')
        plt.xlabel('Asset')
        plt.ylabel('Volatility')
        plt.grid(True, axis='y')
        
        # Plot correlation by regime
        plt.subplot(2, 2, 2)
        correlations = list(self.regime_stats['combined']['regime_correlations'].values())
        regimes = list(self.regime_stats['combined']['regime_correlations'].keys())
        
        plt.bar(regimes, correlations)
        plt.title('Average Correlation by Regime')
        plt.xlabel('Regime')
        plt.ylabel('Average Correlation')
        plt.grid(True, axis='y')
        
        # Plot VaR by regime
        plt.subplot(2, 2, 3)
        var_by_regime = pd.DataFrame(self.regime_stats['combined']['regime_var'])
        
        var_by_regime.plot(kind='bar', ax=plt.gca())
        plt.title('Value at Risk (95%) by Regime')
        plt.xlabel('Asset')
        plt.ylabel('VaR')
        plt.grid(True, axis='y')
        
        # Plot ES by regime
        plt.subplot(2, 2, 4)
        es_by_regime = pd.DataFrame({k: v for k, v in self.regime_stats['combined']['regime_es'].items()})
        
        es_by_regime.plot(kind='bar', ax=plt.gca())
        plt.title('Expected Shortfall (95%) by Regime')
        plt.xlabel('Asset')
        plt.ylabel('ES')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def plot_regime_distribution(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot distribution of regimes
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot correlation regime distribution
        if 'correlation' in self.regime_stats and self.correlation_model is not None:
            plt.subplot(1, 3, 1)
            
            regime_counts = self.correlation_model.rolling_correlations['regime'].value_counts().sort_index()
            
            plt.pie(regime_counts, labels=[f'Regime {r}' for r in regime_counts.index],
                  autopct='%1.1f%%', startangle=90)
            plt.title('Correlation Regime Distribution')
        
        # Plot volatility regime distribution
        if 'volatility' in self.regime_stats and self.volatility_model is not None:
            plt.subplot(1, 3, 2)
            
            regime_counts = self.volatility_model.rolling_volatilities['regime'].value_counts().sort_index()
            
            plt.pie(regime_counts, labels=[f'Regime {r}' for r in regime_counts.index],
                  autopct='%1.1f%%', startangle=90)
            plt.title('Volatility Regime Distribution')
        
        # Plot combined regime distribution
        if 'combined' in self.regime_stats:
            plt.subplot(1, 3, 3)
            
            # Count occurrences of each combined regime
            regime_counts = {}
            
            for regime, periods in self.regime_stats['combined']['regime_periods'].items():
                regime_counts[regime] = len(periods)
            
            # Convert to Series for plotting
            regime_counts = pd.Series(regime_counts)
            
            plt.pie(regime_counts, labels=[f'Regime {r}' for r in regime_counts.index],
                  autopct='%1.1f%%', startangle=90)
            plt.title('Combined Regime Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def plot_regime_transitions(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot regime transitions over time
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot correlation regimes
        if 'correlation' in self.regime_stats and self.correlation_model is not None:
            plt.subplot(3, 1, 1)
            
            plt.plot(self.correlation_model.rolling_correlations.index,
                   self.correlation_model.rolling_correlations['regime'])
            plt.title('Correlation Regimes')
            plt.ylabel('Regime')
            plt.grid(True)
        
        # Plot volatility regimes
        if 'volatility' in self.regime_stats and self.volatility_model is not None:
            plt.subplot(3, 1, 2)
            
            plt.plot(self.volatility_model.rolling_volatilities.index,
                   self.volatility_model.rolling_volatilities['regime'])
            plt.title('Volatility Regimes')
            plt.ylabel('Regime')
            plt.grid(True)
        
        # Plot combined regimes
        if 'combined' in self.regime_stats and self.correlation_model is not None and self.volatility_model is not None:
            plt.subplot(3, 1, 3)
            
            # Create combined regime series
            n_vol_regimes = self.volatility_model.n_regimes
            
            # Ensure same length
            min_length = min(len(self.correlation_model.rolling_correlations), 
                           len(self.volatility_model.rolling_volatilities))
            
            corr_regimes = self.correlation_model.rolling_correlations['regime'].iloc[:min_length]
            vol_regimes = self.volatility_model.rolling_volatilities['regime'].iloc[:min_length]
            
            combined_regimes = corr_regimes * n_vol_regimes + vol_regimes
            
            plt.plot(corr_regimes.index, combined_regimes)
            plt.title('Combined Regimes')
            plt.xlabel('Date')
            plt.ylabel('Regime')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()


# Helper functions for regime analysis

def calculate_rolling_correlation(returns: pd.DataFrame, window_size: int = 60) -> pd.Series:
    """
    Calculate rolling average correlation
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns data
    window_size : int, default=60
        Size of rolling window
        
    Returns:
    --------
    pd.Series
        Rolling average correlation
    """
    n_assets = len(returns.columns)
    n_periods = len(returns) - window_size + 1
    
    # Create a Series to store average correlations
    rolling_corr = pd.Series(
        index=returns.index[window_size-1:],
        dtype=float
    )
    
    # Calculate rolling correlations
    for i in range(n_periods):
        window_data = returns.iloc[i:i+window_size]
        corr_matrix = window_data.corr()
        
        # Calculate average correlation (excluding diagonal)
        avg_corr = (corr_matrix.sum().sum() - n_assets) / (n_assets * (n_assets - 1))
        
        rolling_corr.iloc[i] = avg_corr
    
    return rolling_corr

def calculate_rolling_volatility(returns: pd.DataFrame, window_size: int = 20) -> pd.Series:
    """
    Calculate rolling average volatility
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns data
    window_size : int, default=20
        Size of rolling window
        
    Returns:
    --------
    pd.Series
        Rolling average volatility
    """
    # Calculate rolling volatility for each asset
    vol_series = []
    
    for col in returns.columns:
        rolling_vol = returns[col].rolling(window=window_size).std()
        vol_series.append(rolling_vol)
    
    # Calculate average volatility across assets
    avg_vol = pd.concat(vol_series, axis=1).mean(axis=1)
    
    return avg_vol

def identify_market_regimes(returns: pd.DataFrame, 
                          corr_window: int = 60, 
                          vol_window: int = 20,
                          n_regimes: int = 4) -> pd.DataFrame:
    """
    Identify market regimes based on correlation and volatility
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns data
    corr_window : int, default=60
        Window size for correlation calculation
    vol_window : int, default=20
        Window size for volatility calculation
    n_regimes : int, default=4
        Number of regimes to identify
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with regime classifications
    """
    # Calculate rolling correlation and volatility
    rolling_corr = calculate_rolling_correlation(returns, corr_window)
    rolling_vol = calculate_rolling_volatility(returns, vol_window)
    
    # Align indices
    common_index = rolling_corr.index.intersection(rolling_vol.index)
    rolling_corr = rolling_corr.loc[common_index]
    rolling_vol = rolling_vol.loc[common_index]
    
    # Create DataFrame for regime classification
    regimes = pd.DataFrame({
        'correlation': rolling_corr,
        'volatility': rolling_vol
    })
    
    # Classify correlation regimes
    corr_median = rolling_corr.median()
    regimes['corr_regime'] = np.where(rolling_corr > corr_median, 'High', 'Low')
    
    # Classify volatility regimes
    vol_median = rolling_vol.median()
    regimes['vol_regime'] = np.where(rolling_vol > vol_median, 'High', 'Low')
    
    # Combine regimes
    regime_map = {
        ('Low', 'Low'): 'Low Vol, Low Corr',
        ('Low', 'High'): 'Low Vol, High Corr',
        ('High', 'Low'): 'High Vol, Low Corr',
        ('High', 'High'): 'High Vol, High Corr'
    }
    
    regimes['regime'] = regimes.apply(
        lambda x: regime_map[(x['vol_regime'], x['corr_regime'])],
        axis=1
    )
    
    return regimes

def plot_regime_map(regimes: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot regime map
    
    Parameters:
    -----------
    regimes : pd.DataFrame
        DataFrame with regime classifications
    figsize : tuple, default=(10, 8)
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create scatter plot
    colors = {
        'Low Vol, Low Corr': 'green',
        'Low Vol, High Corr': 'blue',
        'High Vol, Low Corr': 'orange',
        'High Vol, High Corr': 'red'
    }
    
    for regime in colors:
        regime_data = regimes[regimes['regime'] == regime]
        plt.scatter(regime_data['correlation'], regime_data['volatility'], 
                  color=colors[regime], label=regime, alpha=0.7)
    
    plt.title('Market Regimes')
    plt.xlabel('Correlation')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_regime_statistics(returns: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for each regime
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns data
    regimes : pd.DataFrame
        DataFrame with regime classifications
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with regime statistics
    """
    # Align indices
    common_index = returns.index.intersection(regimes.index)
    returns = returns.loc[common_index]
    regimes = regimes.loc[common_index]
    
    # Initialize statistics DataFrame
    stats = pd.DataFrame(index=regimes['regime'].unique())
    
    # Calculate statistics for each regime
    for regime in stats.index:
        regime_returns = returns[regimes['regime'] == regime]
        
        # Basic statistics
        stats.loc[regime, 'Count'] = len(regime_returns)
        stats.loc[regime, 'Frequency'] = len(regime_returns) / len(returns)
        stats.loc[regime, 'Mean Return'] = regime_returns.mean().mean()
        stats.loc[regime, 'Volatility'] = regime_returns.std().mean()
        
        # Risk metrics
        portfolio_returns = regime_returns.mean(axis=1)
        stats.loc[regime, 'VaR (95%)'] = portfolio_returns.quantile(0.05)
        stats.loc[regime, 'ES (95%)'] = portfolio_returns[portfolio_returns <= stats.loc[regime, 'VaR (95%)']].mean()
        
        # Correlation
        avg_corr = regimes.loc[regimes['regime'] == regime, 'correlation'].mean()
        stats.loc[regime, 'Avg Correlation'] = avg_corr
    
    return stats
