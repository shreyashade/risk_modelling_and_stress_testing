"""
Adaptive Regime-Switching Models for Risk Modeling Framework

This module implements various regime-switching models to detect and adapt to
different market states, improving risk prediction accuracy during regime changes.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple, Callable
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RegimeDetector:
    """Base class for regime detection models"""
    
    def __init__(self, n_regimes: int = 2):
        """
        Initialize the regime detector.
        
        Parameters:
        -----------
        n_regimes : int, default=2
            Number of regimes to detect (e.g., bull/bear or low/medium/high volatility)
        """
        self.n_regimes = n_regimes
        self.model = None
        self.regime_params = {}
        self.fitted = False
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the regime detection model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict_regime(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Predict the regime for each time point.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
            
        Returns:
        --------
        np.ndarray
            Array of regime labels
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_regime_params(self) -> Dict:
        """
        Get the parameters for each regime.
        
        Returns:
        --------
        Dict
            Dictionary of regime parameters
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting regime parameters")
        return self.regime_params
    
    def plot_regimes(self, returns: pd.DataFrame, asset_prices: Optional[pd.DataFrame] = None) -> plt.Figure:
        """
        Plot the detected regimes along with returns or prices.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        asset_prices : pd.DataFrame, optional
            Asset prices for visualization
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with regime visualization
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting regimes")
        
        regimes = self.predict_regime(returns)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
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
        
        axes[0].set_title('Asset Performance and Market Regimes')
        axes[0].legend(loc='upper left')
        axes[0].grid(True)
        
        # Plot regimes
        cmap = plt.cm.get_cmap('viridis', self.n_regimes)
        regime_series = pd.Series(regimes, index=returns.index)
        
        # Create colored background for regimes
        for regime in range(self.n_regimes):
            regime_periods = regime_series == regime
            if not any(regime_periods):
                continue
                
            regime_starts = regime_series.index[regime_periods & ~regime_periods.shift(1, fill_value=False)]
            regime_ends = regime_series.index[regime_periods & ~regime_periods.shift(-1, fill_value=False)]
            
            for start, end in zip(regime_starts, regime_ends):
                axes[0].axvspan(start, end, alpha=0.2, color=cmap(regime))
                axes[1].axvspan(start, end, alpha=0.2, color=cmap(regime))
        
        # Plot volatility
        rolling_vol = returns.std(axis=1) * np.sqrt(252)  # Annualized
        axes[1].plot(rolling_vol.index, rolling_vol, 'k-', label='Volatility')
        axes[1].set_ylabel('Volatility')
        axes[1].set_xlabel('Date')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig


class HMMRegimeDetector(RegimeDetector):
    """
    Hidden Markov Model for regime detection.
    
    This model uses a Hidden Markov Model to identify latent market regimes
    based on the distribution of returns.
    """
    
    def __init__(self, n_regimes: int = 2, n_iter: int = 100, random_state: int = 42):
        """
        Initialize the HMM regime detector.
        
        Parameters:
        -----------
        n_regimes : int, default=2
            Number of regimes to detect
        n_iter : int, default=100
            Number of iterations for HMM fitting
        random_state : int, default=42
            Random seed for reproducibility
        """
        super().__init__(n_regimes)
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the HMM regime detection model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info(f"Fitting HMM with {self.n_regimes} regimes")
        
        # Prepare data - use portfolio returns if multiple assets
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
            X = portfolio_returns.reshape(-1, 1)
        else:
            X = returns.values.reshape(-1, 1)
        
        # Initialize and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        self.model.fit(X)
        
        # Get regime sequence
        hidden_states = self.model.predict(X)
        
        # Extract regime parameters
        self.regime_params = {
            'means': self.model.means_,
            'covars': self.model.covars_,
            'transmat': self.model.transmat_,
            'startprob': self.model.startprob_
        }
        
        # Ensure regimes are ordered by volatility (lowest to highest)
        volatilities = np.sqrt(np.diag(self.regime_params['covars'].reshape(self.n_regimes, -1)))
        regime_order = np.argsort(volatilities.flatten())
        
        # Reorder regimes if necessary
        if not np.array_equal(regime_order, np.arange(self.n_regimes)):
            logger.info("Reordering regimes by volatility")
            reorder_map = {old: new for new, old in enumerate(regime_order)}
            hidden_states = np.array([reorder_map[state] for state in hidden_states])
            
            # Reorder parameters
            self.regime_params['means'] = self.regime_params['means'][regime_order]
            self.regime_params['covars'] = self.regime_params['covars'][regime_order]
            self.regime_params['transmat'] = self.regime_params['transmat'][regime_order][:, regime_order]
            self.regime_params['startprob'] = self.regime_params['startprob'][regime_order]
        
        # Calculate regime statistics
        for i in range(self.n_regimes):
            regime_returns = X[hidden_states == i].flatten()
            self.regime_params[f'regime_{i}'] = {
                'mean': np.mean(regime_returns),
                'std': np.std(regime_returns),
                'skew': stats.skew(regime_returns),
                'kurtosis': stats.kurtosis(regime_returns),
                'var_95': np.percentile(regime_returns, 5),
                'es_95': np.mean(regime_returns[regime_returns <= np.percentile(regime_returns, 5)]),
                'count': len(regime_returns),
                'frequency': len(regime_returns) / len(X)
            }
        
        self.fitted = True
        logger.info("HMM regime detection model fitted successfully")
    
    def predict_regime(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Predict the regime for each time point.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
            
        Returns:
        --------
        np.ndarray
            Array of regime labels
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting")
        
        # Prepare data - use portfolio returns if multiple assets
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
            X = portfolio_returns.reshape(-1, 1)
        else:
            X = returns.values.reshape(-1, 1)
        
        # Predict hidden states
        hidden_states = self.model.predict(X)
        return hidden_states
    
    def predict_regime_probabilities(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Predict the probability of each regime for each time point.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
            
        Returns:
        --------
        np.ndarray
            Array of regime probabilities with shape (n_samples, n_regimes)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting")
        
        # Prepare data - use portfolio returns if multiple assets
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
            X = portfolio_returns.reshape(-1, 1)
        else:
            X = returns.values.reshape(-1, 1)
        
        # Predict regime probabilities
        regime_probs = self.model.predict_proba(X)
        return regime_probs


class MarkovSwitchingModel(RegimeDetector):
    """
    Markov Switching Model for regime detection.
    
    This model implements a Markov switching regression model to identify
    different market regimes with distinct return and volatility characteristics.
    """
    
    def __init__(self, n_regimes: int = 2, max_iter: int = 1000, tol: float = 1e-6, random_state: int = 42):
        """
        Initialize the Markov switching model.
        
        Parameters:
        -----------
        n_regimes : int, default=2
            Number of regimes to detect
        max_iter : int, default=1000
            Maximum number of iterations for EM algorithm
        tol : float, default=1e-6
            Convergence tolerance for EM algorithm
        random_state : int, default=42
            Random seed for reproducibility
        """
        super().__init__(n_regimes)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        np.random.seed(random_state)
    
    def _initialize_parameters(self, X: np.ndarray) -> Dict:
        """
        Initialize model parameters using K-means clustering.
        
        Parameters:
        -----------
        X : np.ndarray
            Returns data
            
        Returns:
        --------
        Dict
            Initial parameter estimates
        """
        # Use K-means for initial clustering
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=self.random_state)
        clusters = kmeans.fit_predict(X)
        
        # Initialize parameters
        params = {}
        
        # Means and variances for each regime
        params['means'] = np.zeros(self.n_regimes)
        params['variances'] = np.zeros(self.n_regimes)
        
        for k in range(self.n_regimes):
            regime_data = X[clusters == k]
            if len(regime_data) > 0:
                params['means'][k] = np.mean(regime_data)
                params['variances'][k] = np.var(regime_data)
            else:
                # Fallback if a cluster is empty
                params['means'][k] = np.random.randn()
                params['variances'][k] = 1.0
        
        # Transition probabilities (slightly biased toward staying in the same regime)
        params['transmat'] = np.ones((self.n_regimes, self.n_regimes)) * 0.1
        np.fill_diagonal(params['transmat'], 0.9)
        params['transmat'] = params['transmat'] / params['transmat'].sum(axis=1, keepdims=True)
        
        # Initial state distribution (uniform)
        params['startprob'] = np.ones(self.n_regimes) / self.n_regimes
        
        return params
    
    def _log_likelihood(self, X: np.ndarray, params: Dict) -> float:
        """
        Calculate log-likelihood of the data given parameters.
        
        Parameters:
        -----------
        X : np.ndarray
            Returns data
        params : Dict
            Model parameters
            
        Returns:
        --------
        float
            Log-likelihood value
        """
        n_samples = len(X)
        log_likelihood = 0.0
        
        # Forward algorithm to compute log-likelihood
        alpha = np.zeros((n_samples, self.n_regimes))
        
        # Initialize with starting probabilities
        for j in range(self.n_regimes):
            alpha[0, j] = params['startprob'][j] * stats.norm.pdf(
                X[0], params['means'][j], np.sqrt(params['variances'][j])
            )
        
        # Scale to prevent underflow
        scale = np.sum(alpha[0, :])
        alpha[0, :] /= scale
        log_likelihood += np.log(scale)
        
        # Forward pass
        for t in range(1, n_samples):
            for j in range(self.n_regimes):
                alpha[t, j] = 0.0
                for i in range(self.n_regimes):
                    alpha[t, j] += alpha[t-1, i] * params['transmat'][i, j]
                
                alpha[t, j] *= stats.norm.pdf(
                    X[t], params['means'][j], np.sqrt(params['variances'][j])
                )
            
            # Scale to prevent underflow
            scale = np.sum(alpha[t, :])
            if scale > 0:
                alpha[t, :] /= scale
                log_likelihood += np.log(scale)
        
        return log_likelihood
    
    def _forward_backward(self, X: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform forward-backward algorithm to compute posterior probabilities.
        
        Parameters:
        -----------
        X : np.ndarray
            Returns data
        params : Dict
            Model parameters
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            gamma: Posterior state probabilities
            xi: Posterior transition probabilities
            log_likelihood: Log-likelihood value
        """
        n_samples = len(X)
        
        # Forward pass (alpha)
        alpha = np.zeros((n_samples, self.n_regimes))
        scales = np.zeros(n_samples)
        
        # Initialize with starting probabilities
        for j in range(self.n_regimes):
            alpha[0, j] = params['startprob'][j] * stats.norm.pdf(
                X[0], params['means'][j], np.sqrt(params['variances'][j])
            )
        
        # Scale to prevent underflow
        scales[0] = np.sum(alpha[0, :])
        alpha[0, :] /= scales[0]
        
        # Forward pass
        for t in range(1, n_samples):
            for j in range(self.n_regimes):
                alpha[t, j] = 0.0
                for i in range(self.n_regimes):
                    alpha[t, j] += alpha[t-1, i] * params['transmat'][i, j]
                
                alpha[t, j] *= stats.norm.pdf(
                    X[t], params['means'][j], np.sqrt(params['variances'][j])
                )
            
            # Scale to prevent underflow
            scales[t] = np.sum(alpha[t, :])
            if scales[t] > 0:
                alpha[t, :] /= scales[t]
        
        # Backward pass (beta)
        beta = np.zeros((n_samples, self.n_regimes))
        
        # Initialize
        beta[n_samples-1, :] = 1.0
        
        # Backward pass
        for t in range(n_samples-2, -1, -1):
            for i in range(self.n_regimes):
                beta[t, i] = 0.0
                for j in range(self.n_regimes):
                    beta[t, i] += params['transmat'][i, j] * stats.norm.pdf(
                        X[t+1], params['means'][j], np.sqrt(params['variances'][j])
                    ) * beta[t+1, j]
            
            # Scale with same factor as alpha
            beta[t, :] /= scales[t+1]
        
        # Compute posterior probabilities (gamma)
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        
        # Compute posterior transition probabilities (xi)
        xi = np.zeros((n_samples-1, self.n_regimes, self.n_regimes))
        
        for t in range(n_samples-1):
            for i in range(self.n_regimes):
                for j in range(self.n_regimes):
                    xi[t, i, j] = alpha[t, i] * params['transmat'][i, j] * stats.norm.pdf(
                        X[t+1], params['means'][j], np.sqrt(params['variances'][j])
                    ) * beta[t+1, j]
            
            # Normalize
            xi[t, :, :] /= np.sum(xi[t, :, :])
        
        # Compute log-likelihood
        log_likelihood = np.sum(np.log(scales))
        
        return gamma, xi, log_likelihood
    
    def _update_parameters(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> Dict:
        """
        Update model parameters using EM algorithm.
        
        Parameters:
        -----------
        X : np.ndarray
            Returns data
        gamma : np.ndarray
            Posterior state probabilities
        xi : np.ndarray
            Posterior transition probabilities
            
        Returns:
        --------
        Dict
            Updated parameters
        """
        n_samples = len(X)
        params = {}
        
        # Update means
        params['means'] = np.zeros(self.n_regimes)
        for j in range(self.n_regimes):
            weighted_sum = np.sum(gamma[:, j] * X)
            weight_sum = np.sum(gamma[:, j])
            if weight_sum > 0:
                params['means'][j] = weighted_sum / weight_sum
        
        # Update variances
        params['variances'] = np.zeros(self.n_regimes)
        for j in range(self.n_regimes):
            weighted_sum = np.sum(gamma[:, j] * (X - params['means'][j])**2)
            weight_sum = np.sum(gamma[:, j])
            if weight_sum > 0:
                params['variances'][j] = weighted_sum / weight_sum
                # Add small constant to prevent zero variance
                params['variances'][j] = max(params['variances'][j], 1e-6)
        
        # Update transition probabilities
        params['transmat'] = np.zeros((self.n_regimes, self.n_regimes))
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                params['transmat'][i, j] = np.sum(xi[:, i, j])
            
            # Normalize
            row_sum = np.sum(params['transmat'][i, :])
            if row_sum > 0:
                params['transmat'][i, :] /= row_sum
        
        # Update starting probabilities
        params['startprob'] = gamma[0, :]
        
        return params
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the Markov switching model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info(f"Fitting Markov Switching Model with {self.n_regimes} regimes")
        
        # Prepare data - use portfolio returns if multiple assets
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
            X = portfolio_returns
        else:
            X = returns.values.flatten()
        
        # Initialize parameters
        params = self._initialize_parameters(X.reshape(-1, 1))
        
        # EM algorithm
        prev_log_likelihood = -np.inf
        converged = False
        
        for iteration in range(self.max_iter):
            # E-step: compute posterior probabilities
            gamma, xi, log_likelihood = self._forward_backward(X, params)
            
            # M-step: update parameters
            params = self._update_parameters(X, gamma, xi)
            
            # Check convergence
            improvement = log_likelihood - prev_log_likelihood
            if iteration > 0 and abs(improvement) < self.tol:
                converged = True
                break
            
            prev_log_likelihood = log_likelihood
            
            if iteration % 10 == 0:
                logger.debug(f"Iteration {iteration}: Log-likelihood = {log_likelihood:.6f}")
        
        if converged:
            logger.info(f"Markov Switching Model converged after {iteration+1} iterations")
        else:
            logger.warning(f"Markov Switching Model did not converge after {self.max_iter} iterations")
        
        # Store model parameters
        self.model = params
        
        # Get regime sequence
        gamma, _, _ = self._forward_backward(X, params)
        hidden_states = np.argmax(gamma, axis=1)
        
        # Ensure regimes are ordered by volatility (lowest to highest)
        volatilities = np.sqrt(params['variances'])
        regime_order = np.argsort(volatilities)
        
        # Reorder regimes if necessary
        if not np.array_equal(regime_order, np.arange(self.n_regimes)):
            logger.info("Reordering regimes by volatility")
            reorder_map = {old: new for new, old in enumerate(regime_order)}
            hidden_states = np.array([reorder_map[state] for state in hidden_states])
            
            # Reorder parameters
            params['means'] = params['means'][regime_order]
            params['variances'] = params['variances'][regime_order]
            params['transmat'] = params['transmat'][regime_order][:, regime_order]
            params['startprob'] = params['startprob'][regime_order]
        
        # Calculate regime statistics
        self.regime_params = params.copy()
        
        for i in range(self.n_regimes):
            regime_returns = X[hidden_states == i]
            self.regime_params[f'regime_{i}'] = {
                'mean': np.mean(regime_returns),
                'std': np.sqrt(params['variances'][i]),
                'annualized_return': np.mean(regime_returns) * 252,
                'annualized_volatility': np.sqrt(params['variances'][i]) * np.sqrt(252),
                'sharpe': (np.mean(regime_returns) * 252) / (np.sqrt(params['variances'][i]) * np.sqrt(252)),
                'var_95': np.percentile(regime_returns, 5),
                'es_95': np.mean(regime_returns[regime_returns <= np.percentile(regime_returns, 5)]),
                'count': len(regime_returns),
                'frequency': len(regime_returns) / len(X),
                'avg_duration': 1 / (1 - params['transmat'][i, i]) if i < params['transmat'].shape[0] else 0
            }
        
        self.fitted = True
        logger.info("Markov Switching Model fitted successfully")
    
    def predict_regime(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Predict the regime for each time point.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
            
        Returns:
        --------
        np.ndarray
            Array of regime labels
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting")
        
        # Prepare data - use portfolio returns if multiple assets
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
            X = portfolio_returns
        else:
            X = returns.values.flatten()
        
        # Compute posterior probabilities
        gamma, _, _ = self._forward_backward(X, self.model)
        
        # Get most likely regime for each time point
        hidden_states = np.argmax(gamma, axis=1)
        
        return hidden_states
    
    def predict_regime_probabilities(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Predict the probability of each regime for each time point.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
            
        Returns:
        --------
        np.ndarray
            Array of regime probabilities with shape (n_samples, n_regimes)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting")
        
        # Prepare data - use portfolio returns if multiple assets
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
            X = portfolio_returns
        else:
            X = returns.values.flatten()
        
        # Compute posterior probabilities
        gamma, _, _ = self._forward_backward(X, self.model)
        
        return gamma


class VolatilityRegimeDetector(RegimeDetector):
    """
    Volatility-based regime detection.
    
    This model identifies market regimes based on rolling volatility levels,
    using threshold-based classification.
    """
    
    def __init__(self, n_regimes: int = 3, window: int = 21, percentiles: Optional[List[float]] = None):
        """
        Initialize the volatility regime detector.
        
        Parameters:
        -----------
        n_regimes : int, default=3
            Number of volatility regimes to detect (low, medium, high)
        window : int, default=21
            Rolling window size for volatility calculation
        percentiles : List[float], optional
            Percentiles for regime thresholds. If None, equally spaced percentiles are used.
        """
        super().__init__(n_regimes)
        self.window = window
        
        if percentiles is None:
            # Equally spaced percentiles
            self.percentiles = [100 * i / n_regimes for i in range(1, n_regimes)]
        else:
            if len(percentiles) != n_regimes - 1:
                raise ValueError(f"Expected {n_regimes-1} percentiles, got {len(percentiles)}")
            self.percentiles = percentiles
        
        self.thresholds = None
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the volatility regime detection model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info(f"Fitting Volatility Regime Detector with {self.n_regimes} regimes")
        
        # Calculate rolling volatility
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
            rolling_vol = pd.Series(portfolio_returns).rolling(
                window=self.window, min_periods=self.window//2
            ).std() * np.sqrt(252)  # Annualized
        else:
            rolling_vol = returns.rolling(
                window=self.window, min_periods=self.window//2
            ).std() * np.sqrt(252)  # Annualized
        
        # Determine volatility thresholds based on percentiles
        self.thresholds = np.percentile(
            rolling_vol.dropna(), self.percentiles
        )
        
        logger.info(f"Volatility thresholds: {self.thresholds}")
        
        # Classify regimes
        regimes = self._classify_regimes(rolling_vol)
        
        # Calculate regime statistics
        self.regime_params = {}
        
        for i in range(self.n_regimes):
            regime_returns = returns.iloc[regimes == i]
            if len(regime_returns) > 0:
                if returns.shape[1] > 1:
                    # Calculate portfolio returns for this regime
                    regime_portfolio_returns = regime_returns.values @ weights
                    
                    self.regime_params[f'regime_{i}'] = {
                        'mean': np.mean(regime_portfolio_returns),
                        'std': np.std(regime_portfolio_returns),
                        'annualized_return': np.mean(regime_portfolio_returns) * 252,
                        'annualized_volatility': np.std(regime_portfolio_returns) * np.sqrt(252),
                        'sharpe': (np.mean(regime_portfolio_returns) * 252) / (np.std(regime_portfolio_returns) * np.sqrt(252)),
                        'var_95': np.percentile(regime_portfolio_returns, 5),
                        'es_95': np.mean(regime_portfolio_returns[regime_portfolio_returns <= np.percentile(regime_portfolio_returns, 5)]),
                        'count': len(regime_returns),
                        'frequency': len(regime_returns) / len(returns)
                    }
                else:
                    flat_returns = regime_returns.values.flatten()
                    
                    self.regime_params[f'regime_{i}'] = {
                        'mean': np.mean(flat_returns),
                        'std': np.std(flat_returns),
                        'annualized_return': np.mean(flat_returns) * 252,
                        'annualized_volatility': np.std(flat_returns) * np.sqrt(252),
                        'sharpe': (np.mean(flat_returns) * 252) / (np.std(flat_returns) * np.sqrt(252)),
                        'var_95': np.percentile(flat_returns, 5),
                        'es_95': np.mean(flat_returns[flat_returns <= np.percentile(flat_returns, 5)]),
                        'count': len(regime_returns),
                        'frequency': len(regime_returns) / len(returns)
                    }
        
        # Store volatility thresholds in regime parameters
        self.regime_params['thresholds'] = self.thresholds
        
        self.fitted = True
        logger.info("Volatility Regime Detector fitted successfully")
    
    def _classify_regimes(self, volatility: pd.Series) -> np.ndarray:
        """
        Classify regimes based on volatility thresholds.
        
        Parameters:
        -----------
        volatility : pd.Series
            Rolling volatility series
            
        Returns:
        --------
        np.ndarray
            Array of regime labels
        """
        regimes = np.zeros(len(volatility), dtype=int)
        
        for i, threshold in enumerate(self.thresholds):
            regimes[volatility > threshold] = i + 1
        
        return regimes
    
    def predict_regime(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Predict the regime for each time point.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
            
        Returns:
        --------
        np.ndarray
            Array of regime labels
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting")
        
        # Calculate rolling volatility
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = returns.values @ weights
            rolling_vol = pd.Series(portfolio_returns).rolling(
                window=self.window, min_periods=self.window//2
            ).std() * np.sqrt(252)  # Annualized
        else:
            rolling_vol = returns.rolling(
                window=self.window, min_periods=self.window//2
            ).std() * np.sqrt(252)  # Annualized
        
        # Classify regimes
        regimes = self._classify_regimes(rolling_vol)
        
        return regimes


class AdaptiveRegimeModel:
    """
    Adaptive regime-switching model for risk estimation.
    
    This model combines multiple regime detection methods and adapts risk
    estimation based on the detected market regime.
    """
    
    def __init__(self, 
                 n_regimes: int = 3, 
                 detection_methods: List[str] = ['hmm', 'markov', 'volatility'],
                 window_size: int = 252,
                 min_regime_obs: int = 63):
        """
        Initialize the adaptive regime model.
        
        Parameters:
        -----------
        n_regimes : int, default=3
            Number of regimes to detect
        detection_methods : List[str], default=['hmm', 'markov', 'volatility']
            List of regime detection methods to use
        window_size : int, default=252
            Rolling window size for model calibration (trading days)
        min_regime_obs : int, default=63
            Minimum number of observations required for a regime (trading days)
        """
        self.n_regimes = n_regimes
        self.detection_methods = detection_methods
        self.window_size = window_size
        self.min_regime_obs = min_regime_obs
        
        self.detectors = {}
        self.regime_models = {}
        self.current_regime = None
        self.regime_history = None
        self.fitted = False
    
    def fit(self, returns: pd.DataFrame, asset_prices: Optional[pd.DataFrame] = None) -> None:
        """
        Fit the adaptive regime model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        asset_prices : pd.DataFrame, optional
            Historical asset prices for visualization
        """
        logger.info(f"Fitting Adaptive Regime Model with {self.n_regimes} regimes")
        
        # Initialize regime detectors
        for method in self.detection_methods:
            if method == 'hmm':
                self.detectors[method] = HMMRegimeDetector(n_regimes=self.n_regimes)
            elif method == 'markov':
                self.detectors[method] = MarkovSwitchingModel(n_regimes=self.n_regimes)
            elif method == 'volatility':
                self.detectors[method] = VolatilityRegimeDetector(n_regimes=self.n_regimes)
            else:
                raise ValueError(f"Unknown detection method: {method}")
            
            # Fit detector
            self.detectors[method].fit(returns)
        
        # Combine regime predictions using ensemble approach
        ensemble_regimes = self._combine_regime_predictions(returns)
        
        # Store regime history
        self.regime_history = pd.Series(ensemble_regimes, index=returns.index)
        
        # Fit regime-specific risk models
        for regime in range(self.n_regimes):
            regime_mask = ensemble_regimes == regime
            regime_returns = returns.loc[regime_mask]
            
            if len(regime_returns) >= self.min_regime_obs:
                logger.info(f"Fitting risk model for regime {regime} with {len(regime_returns)} observations")
                
                # Here we would fit regime-specific risk models
                # For now, just store regime statistics
                self.regime_models[regime] = {
                    'mean': regime_returns.mean(),
                    'cov': regime_returns.cov(),
                    'var_95': regime_returns.quantile(0.05),
                    'es_95': regime_returns[regime_returns <= regime_returns.quantile(0.05)].mean(),
                    'observations': len(regime_returns)
                }
            else:
                logger.warning(f"Insufficient observations for regime {regime}: {len(regime_returns)} < {self.min_regime_obs}")
                
                # Use all data as fallback
                self.regime_models[regime] = {
                    'mean': returns.mean(),
                    'cov': returns.cov(),
                    'var_95': returns.quantile(0.05),
                    'es_95': returns[returns <= returns.quantile(0.05)].mean(),
                    'observations': len(returns)
                }
        
        # Determine current regime
        if len(ensemble_regimes) > 0:
            self.current_regime = ensemble_regimes[-1]
        else:
            self.current_regime = 0
        
        self.fitted = True
        logger.info("Adaptive Regime Model fitted successfully")
        
        # Plot regimes if asset prices are provided
        if asset_prices is not None:
            self.plot_regimes(returns, asset_prices)
    
    def _combine_regime_predictions(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Combine regime predictions from multiple detectors.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
            
        Returns:
        --------
        np.ndarray
            Combined regime predictions
        """
        # Get predictions from each detector
        predictions = {}
        for method, detector in self.detectors.items():
            predictions[method] = detector.predict_regime(returns)
        
        # Simple majority voting for ensemble
        ensemble_regimes = np.zeros(len(returns), dtype=int)
        
        for i in range(len(returns)):
            votes = [predictions[method][i] for method in self.detection_methods]
            # Use most common regime
            unique_regimes, counts = np.unique(votes, return_counts=True)
            ensemble_regimes[i] = unique_regimes[np.argmax(counts)]
        
        return ensemble_regimes
    
    def predict_regime(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Predict the regime for each time point.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
            
        Returns:
        --------
        np.ndarray
            Array of regime labels
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting")
        
        return self._combine_regime_predictions(returns)
    
    def estimate_var(self, 
                     returns: pd.DataFrame, 
                     weights: np.ndarray, 
                     confidence_level: float = 0.95,
                     horizon: int = 1) -> Dict:
        """
        Estimate Value-at-Risk using regime-specific models.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for VaR
        horizon : int, default=1
            Forecast horizon in days
            
        Returns:
        --------
        Dict
            Dictionary with VaR estimates for each regime and ensemble
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before estimating VaR")
        
        # Predict current regime
        current_returns = returns.iloc[-self.window_size:]
        current_regime = self.predict_regime(current_returns)[-1]
        
        # Estimate VaR for each regime
        var_estimates = {}
        
        for regime, model in self.regime_models.items():
            # Calculate portfolio mean and variance
            portfolio_mean = np.dot(weights, model['mean']) * horizon
            portfolio_var = np.dot(weights, np.dot(model['cov'], weights)) * horizon
            portfolio_std = np.sqrt(portfolio_var)
            
            # Calculate VaR
            z_score = stats.norm.ppf(1 - confidence_level)
            var_estimates[f'regime_{regime}'] = -(portfolio_mean + z_score * portfolio_std)
        
        # Ensemble VaR (weighted by regime probability)
        regime_probs = self._estimate_regime_probabilities(current_returns)
        ensemble_var = 0.0
        
        for regime in range(self.n_regimes):
            ensemble_var += regime_probs[regime] * var_estimates[f'regime_{regime}']
        
        var_estimates['ensemble'] = ensemble_var
        var_estimates['current_regime'] = current_regime
        
        return var_estimates
    
    def estimate_es(self, 
                    returns: pd.DataFrame, 
                    weights: np.ndarray, 
                    confidence_level: float = 0.95,
                    horizon: int = 1) -> Dict:
        """
        Estimate Expected Shortfall using regime-specific models.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for ES
        horizon : int, default=1
            Forecast horizon in days
            
        Returns:
        --------
        Dict
            Dictionary with ES estimates for each regime and ensemble
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before estimating ES")
        
        # Predict current regime
        current_returns = returns.iloc[-self.window_size:]
        current_regime = self.predict_regime(current_returns)[-1]
        
        # Estimate ES for each regime
        es_estimates = {}
        
        for regime, model in self.regime_models.items():
            # For normal distribution, ES = VaR + adjustment
            # Calculate portfolio mean and variance
            portfolio_mean = np.dot(weights, model['mean']) * horizon
            portfolio_var = np.dot(weights, np.dot(model['cov'], weights)) * horizon
            portfolio_std = np.sqrt(portfolio_var)
            
            # Calculate VaR
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(portfolio_mean + z_score * portfolio_std)
            
            # Calculate ES adjustment
            es_adjustment = portfolio_std * stats.norm.pdf(z_score) / (1 - confidence_level)
            es_estimates[f'regime_{regime}'] = var + es_adjustment
        
        # Ensemble ES (weighted by regime probability)
        regime_probs = self._estimate_regime_probabilities(current_returns)
        ensemble_es = 0.0
        
        for regime in range(self.n_regimes):
            ensemble_es += regime_probs[regime] * es_estimates[f'regime_{regime}']
        
        es_estimates['ensemble'] = ensemble_es
        es_estimates['current_regime'] = current_regime
        
        return es_estimates
    
    def _estimate_regime_probabilities(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Estimate the probability of each regime.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
            
        Returns:
        --------
        np.ndarray
            Array of regime probabilities
        """
        # Simple approach: use frequency of regimes in recent window
        regimes = self.predict_regime(returns)
        regime_counts = np.zeros(self.n_regimes)
        
        for regime in range(self.n_regimes):
            regime_counts[regime] = np.sum(regimes == regime)
        
        # Add small constant to avoid zero probabilities
        regime_counts += 0.1
        
        # Normalize to get probabilities
        regime_probs = regime_counts / np.sum(regime_counts)
        
        return regime_probs
    
    def plot_regimes(self, returns: pd.DataFrame, asset_prices: pd.DataFrame) -> plt.Figure:
        """
        Plot the detected regimes along with asset prices.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        asset_prices : pd.DataFrame
            Asset prices
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with regime visualization
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting regimes")
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot asset prices
        for col in asset_prices.columns:
            axes[0].plot(asset_prices.index, asset_prices[col], label=col)
        
        axes[0].set_title('Asset Prices and Market Regimes')
        axes[0].legend(loc='upper left')
        axes[0].grid(True)
        
        # Plot regimes as background colors
        cmap = plt.cm.get_cmap('viridis', self.n_regimes)
        
        for regime in range(self.n_regimes):
            regime_periods = self.regime_history == regime
            if not any(regime_periods):
                continue
                
            regime_starts = self.regime_history.index[regime_periods & ~regime_periods.shift(1, fill_value=False)]
            regime_ends = self.regime_history.index[regime_periods & ~regime_periods.shift(-1, fill_value=False)]
            
            for start, end in zip(regime_starts, regime_ends):
                axes[0].axvspan(start, end, alpha=0.2, color=cmap(regime))
                axes[1].axvspan(start, end, alpha=0.2, color=cmap(regime))
                axes[2].axvspan(start, end, alpha=0.2, color=cmap(regime))
        
        # Plot rolling volatility
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
        
        axes[1].plot(rolling_vol.index, rolling_vol, 'k-', label='Volatility (21-day)')
        axes[1].set_ylabel('Annualized Volatility')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot cumulative returns
        if returns.shape[1] > 1:
            # Plot portfolio returns
            portfolio_returns = pd.Series(returns.values @ weights, index=returns.index)
            cum_returns = (1 + portfolio_returns).cumprod()
            axes[2].plot(cum_returns.index, cum_returns, 'b-', label='Portfolio')
        else:
            # Plot single asset returns
            cum_returns = (1 + returns).cumprod()
            axes[2].plot(cum_returns.index, cum_returns, 'b-', label=returns.columns[0])
        
        axes[2].set_ylabel('Cumulative Return')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True)
        
        # Add regime legend
        for regime in range(self.n_regimes):
            if regime in self.regime_models:
                model = self.regime_models[regime]
                label = f"Regime {regime}: Vol={model['cov'].mean().mean():.2%}, VaR={model['var_95'].mean():.2%}"
                axes[0].axhline(y=0, color=cmap(regime), label=label, alpha=0)
        
        axes[0].legend(loc='upper left')
        
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
    
    # Initialize and fit HMM regime detector
    hmm_detector = HMMRegimeDetector(n_regimes=3)
    hmm_detector.fit(returns)
    
    # Plot regimes
    hmm_fig = hmm_detector.plot_regimes(returns, prices)
    hmm_fig.savefig('hmm_regimes.png')
    
    # Initialize and fit Markov switching model
    ms_detector = MarkovSwitchingModel(n_regimes=3)
    ms_detector.fit(returns)
    
    # Plot regimes
    ms_fig = ms_detector.plot_regimes(returns, prices)
    ms_fig.savefig('ms_regimes.png')
    
    # Initialize and fit volatility regime detector
    vol_detector = VolatilityRegimeDetector(n_regimes=3)
    vol_detector.fit(returns)
    
    # Plot regimes
    vol_fig = vol_detector.plot_regimes(returns, prices)
    vol_fig.savefig('vol_regimes.png')
    
    # Initialize and fit adaptive regime model
    adaptive_model = AdaptiveRegimeModel(n_regimes=3)
    adaptive_model.fit(returns, prices)
    
    # Estimate VaR and ES
    weights = np.ones(len(tickers)) / len(tickers)
    var_estimates = adaptive_model.estimate_var(returns, weights)
    es_estimates = adaptive_model.estimate_es(returns, weights)
    
    print("VaR Estimates:")
    for key, value in var_estimates.items():
        print(f"{key}: {value:.2%}")
    
    print("\nES Estimates:")
    for key, value in es_estimates.items():
        print(f"{key}: {value:.2%}")
