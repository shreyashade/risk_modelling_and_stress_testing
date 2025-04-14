"""
Ensemble Risk Model Framework

This module implements an ensemble approach to risk modeling that combines
multiple risk estimation methods to improve overall accuracy and robustness.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Union, Optional, Tuple, Callable, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseRiskModel:
    """Base class for risk models in the ensemble framework"""
    
    def __init__(self, name: str):
        """
        Initialize the base risk model.
        
        Parameters:
        -----------
        name : str
            Name of the risk model
        """
        self.name = name
        self.fitted = False
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the risk model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict_var(self, 
                    returns: pd.DataFrame, 
                    weights: np.ndarray,
                    confidence_level: float = 0.95) -> float:
        """
        Predict Value-at-Risk.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for VaR
            
        Returns:
        --------
        float
            VaR prediction
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict_es(self, 
                   returns: pd.DataFrame, 
                   weights: np.ndarray,
                   confidence_level: float = 0.95) -> float:
        """
        Predict Expected Shortfall.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for ES
            
        Returns:
        --------
        float
            ES prediction
        """
        raise NotImplementedError("Subclasses must implement this method")


class ParametricRiskModel(BaseRiskModel):
    """
    Parametric risk model using normal distribution assumption.
    
    This model assumes returns follow a normal distribution and calculates
    risk metrics based on mean and covariance estimates.
    """
    
    def __init__(self, 
                 name: str = "Parametric",
                 time_weighted: bool = False,
                 half_life: int = 63):
        """
        Initialize the parametric risk model.
        
        Parameters:
        -----------
        name : str, default="Parametric"
            Name of the risk model
        time_weighted : bool, default=False
            Whether to use time-weighted estimates
        half_life : int, default=63
            Half-life parameter for exponential weighting (trading days)
        """
        super().__init__(name)
        self.time_weighted = time_weighted
        self.half_life = half_life
        self.decay_factor = np.exp(np.log(0.5) / half_life) if time_weighted else None
        self.mean = None
        self.cov = None
    
    def _get_weights(self, n_obs: int) -> np.ndarray:
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
        if not self.time_weighted:
            return np.ones(n_obs) / n_obs
        
        # Calculate exponential weights
        weights = np.power(self.decay_factor, np.arange(n_obs-1, -1, -1))
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        return weights
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the parametric risk model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info(f"Fitting {self.name} risk model")
        
        # Calculate weights
        weights = self._get_weights(len(returns))
        
        if self.time_weighted:
            # Calculate weighted mean and covariance
            self.mean = np.average(returns.values, axis=0, weights=weights)
            
            # Center the data
            centered_data = returns.values - self.mean
            
            # Calculate weighted covariance
            weights_reshaped = weights.reshape(-1, 1)
            self.cov = np.dot(centered_data.T * weights_reshaped.T, centered_data)
        else:
            # Use standard mean and covariance
            self.mean = returns.mean().values
            self.cov = returns.cov().values
        
        self.fitted = True
        logger.info(f"{self.name} risk model fitted successfully")
    
    def predict_var(self, 
                    returns: pd.DataFrame, 
                    weights: np.ndarray,
                    confidence_level: float = 0.95) -> float:
        """
        Predict Value-at-Risk using parametric approach.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns (not used in prediction, only for consistency)
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for VaR
            
        Returns:
        --------
        float
            VaR prediction
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Calculate portfolio parameters
        portfolio_mean = np.dot(weights, self.mean)
        portfolio_var = np.dot(weights, np.dot(self.cov, weights))
        portfolio_std = np.sqrt(portfolio_var)
        
        # Calculate VaR
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -(portfolio_mean + z_score * portfolio_std)
        
        return var
    
    def predict_es(self, 
                   returns: pd.DataFrame, 
                   weights: np.ndarray,
                   confidence_level: float = 0.95) -> float:
        """
        Predict Expected Shortfall using parametric approach.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns (not used in prediction, only for consistency)
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for ES
            
        Returns:
        --------
        float
            ES prediction
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Calculate portfolio parameters
        portfolio_mean = np.dot(weights, self.mean)
        portfolio_var = np.dot(weights, np.dot(self.cov, weights))
        portfolio_std = np.sqrt(portfolio_var)
        
        # Calculate VaR
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -(portfolio_mean + z_score * portfolio_std)
        
        # Calculate ES adjustment for normal distribution
        es_adjustment = portfolio_std * stats.norm.pdf(z_score) / (1 - confidence_level)
        es = var + es_adjustment
        
        return es


class HistoricalRiskModel(BaseRiskModel):
    """
    Historical simulation risk model.
    
    This model uses historical returns directly to estimate risk metrics
    without making distributional assumptions.
    """
    
    def __init__(self, 
                 name: str = "Historical",
                 time_weighted: bool = False,
                 half_life: int = 63,
                 bootstrap: bool = False,
                 n_bootstrap: int = 1000):
        """
        Initialize the historical risk model.
        
        Parameters:
        -----------
        name : str, default="Historical"
            Name of the risk model
        time_weighted : bool, default=False
            Whether to use time-weighted estimates
        half_life : int, default=63
            Half-life parameter for exponential weighting (trading days)
        bootstrap : bool, default=False
            Whether to use bootstrap resampling
        n_bootstrap : int, default=1000
            Number of bootstrap samples
        """
        super().__init__(name)
        self.time_weighted = time_weighted
        self.half_life = half_life
        self.decay_factor = np.exp(np.log(0.5) / half_life) if time_weighted else None
        self.bootstrap = bootstrap
        self.n_bootstrap = n_bootstrap
        self.historical_returns = None
    
    def _get_weights(self, n_obs: int) -> np.ndarray:
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
        if not self.time_weighted:
            return np.ones(n_obs) / n_obs
        
        # Calculate exponential weights
        weights = np.power(self.decay_factor, np.arange(n_obs-1, -1, -1))
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        return weights
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the historical risk model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info(f"Fitting {self.name} risk model")
        
        # Store historical returns
        self.historical_returns = returns.copy()
        
        self.fitted = True
        logger.info(f"{self.name} risk model fitted successfully")
    
    def predict_var(self, 
                    returns: pd.DataFrame, 
                    weights: np.ndarray,
                    confidence_level: float = 0.95) -> float:
        """
        Predict Value-at-Risk using historical simulation.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns (not used in prediction, only for consistency)
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for VaR
            
        Returns:
        --------
        float
            VaR prediction
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Calculate portfolio returns
        portfolio_returns = self.historical_returns.values @ weights
        
        if self.bootstrap:
            # Bootstrap resampling
            bootstrap_vars = []
            sample_weights = self._get_weights(len(portfolio_returns))
            
            for _ in range(self.n_bootstrap):
                # Weighted bootstrap sampling
                bootstrap_indices = np.random.choice(
                    len(portfolio_returns),
                    size=len(portfolio_returns),
                    p=sample_weights
                )
                bootstrap_returns = portfolio_returns[bootstrap_indices]
                
                # Calculate VaR for bootstrap sample
                bootstrap_var = -np.percentile(bootstrap_returns, 100 * (1 - confidence_level))
                bootstrap_vars.append(bootstrap_var)
            
            # Use mean of bootstrap VaRs
            var = np.mean(bootstrap_vars)
        elif self.time_weighted:
            # Calculate weighted VaR
            sorted_indices = np.argsort(portfolio_returns)
            sorted_returns = portfolio_returns[sorted_indices]
            sorted_weights = self._get_weights(len(portfolio_returns))[sorted_indices]
            
            # Calculate cumulative weights
            cumulative_weights = np.cumsum(sorted_weights)
            
            # Find VaR threshold
            var_index = np.searchsorted(cumulative_weights, 1 - confidence_level)
            
            if var_index >= len(sorted_returns):
                var_index = len(sorted_returns) - 1
            
            var = -sorted_returns[var_index]
        else:
            # Standard historical VaR
            var = -np.percentile(portfolio_returns, 100 * (1 - confidence_level))
        
        return var
    
    def predict_es(self, 
                   returns: pd.DataFrame, 
                   weights: np.ndarray,
                   confidence_level: float = 0.95) -> float:
        """
        Predict Expected Shortfall using historical simulation.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns (not used in prediction, only for consistency)
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for ES
            
        Returns:
        --------
        float
            ES prediction
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Calculate portfolio returns
        portfolio_returns = self.historical_returns.values @ weights
        
        if self.bootstrap:
            # Bootstrap resampling
            bootstrap_es = []
            sample_weights = self._get_weights(len(portfolio_returns))
            
            for _ in range(self.n_bootstrap):
                # Weighted bootstrap sampling
                bootstrap_indices = np.random.choice(
                    len(portfolio_returns),
                    size=len(portfolio_returns),
                    p=sample_weights
                )
                bootstrap_returns = portfolio_returns[bootstrap_indices]
                
                # Calculate VaR for bootstrap sample
                bootstrap_var_threshold = np.percentile(bootstrap_returns, 100 * (1 - confidence_level))
                bootstrap_tail = bootstrap_returns[bootstrap_returns <= bootstrap_var_threshold]
                
                if len(bootstrap_tail) > 0:
                    bootstrap_es.append(-np.mean(bootstrap_tail))
            
            # Use mean of bootstrap ES
            if bootstrap_es:
                es = np.mean(bootstrap_es)
            else:
                es = -np.mean(portfolio_returns)  # Fallback
        elif self.time_weighted:
            # Calculate weighted ES
            sorted_indices = np.argsort(portfolio_returns)
            sorted_returns = portfolio_returns[sorted_indices]
            sorted_weights = self._get_weights(len(portfolio_returns))[sorted_indices]
            
            # Calculate cumulative weights
            cumulative_weights = np.cumsum(sorted_weights)
            
            # Find VaR threshold
            var_index = np.searchsorted(cumulative_weights, 1 - confidence_level)
            
            if var_index >= len(sorted_returns):
                var_index = len(sorted_returns) - 1
            
            # Calculate ES using weights
            tail_returns = sorted_returns[:var_index+1]
            tail_weights = sorted_weights[:var_index+1]
            
            # Normalize tail weights
            normalized_tail_weights = tail_weights / np.sum(tail_weights)
            
            # Calculate weighted ES
            es = -np.sum(tail_returns * normalized_tail_weights)
        else:
            # Standard historical ES
            var_threshold = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
            tail = portfolio_returns[portfolio_returns <= var_threshold]
            
            if len(tail) > 0:
                es = -np.mean(tail)
            else:
                es = -np.mean(portfolio_returns)  # Fallback
        
        return es


class EVTRiskModel(BaseRiskModel):
    """
    Extreme Value Theory risk model.
    
    This model uses EVT to model the tail distribution of returns for
    more accurate estimation of extreme risks.
    """
    
    def __init__(self, 
                 name: str = "EVT",
                 evt_method: str = "gpd",
                 threshold_quantile: float = 0.05,
                 min_exceedances: int = 50,
                 dynamic: bool = False,
                 window_size: int = 252,
                 conditional: bool = False,
                 condition_factors: Optional[List[str]] = None,
                 bayesian: bool = False,
                 n_samples: int = 1000):
        """
        Initialize the EVT risk model.
        
        Parameters:
        -----------
        name : str, default="EVT"
            Name of the risk model
        evt_method : str, default="gpd"
            EVT method to use ('gpd', 'dynamic', 'conditional', 'bayesian')
        threshold_quantile : float, default=0.05
            Quantile for threshold selection
        min_exceedances : int, default=50
            Minimum number of exceedances required for fitting
        dynamic : bool, default=False
            Whether to use dynamic EVT
        window_size : int, default=252
            Window size for dynamic EVT (trading days)
        conditional : bool, default=False
            Whether to use conditional EVT
        condition_factors : List[str], optional
            List of factor names to condition on (e.g., ['volatility', 'trend'])
        bayesian : bool, default=False
            Whether to use Bayesian EVT
        n_samples : int, default=1000
            Number of posterior samples for Bayesian EVT
        """
        super().__init__(name)
        self.evt_method = evt_method
        self.threshold_quantile = threshold_quantile
        self.min_exceedances = min_exceedances
        self.dynamic = dynamic
        self.window_size = window_size
        self.conditional = conditional
        self.condition_factors = condition_factors or ['volatility']
        self.bayesian = bayesian
        self.n_samples = n_samples
        self.evt_model = None
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the EVT risk model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info(f"Fitting {self.name} risk model with {self.evt_method} method")
        
        # Import EVT models
        try:
            from extreme_value_theory import (
                GPDAnalysis, DynamicEVT, ConditionalEVT, BayesianEVT
            )
        except ImportError:
            logger.error("Failed to import EVT models")
            raise ImportError("extreme_value_theory module not found")
        
        # Initialize EVT model based on method
        if self.evt_method == 'gpd' or (not self.dynamic and not self.conditional and not self.bayesian):
            self.evt_model = GPDAnalysis(
                threshold_method='quantile',
                threshold_value=self.threshold_quantile,
                min_exceedances=self.min_exceedances
            )
        elif self.evt_method == 'dynamic' or self.dynamic:
            self.evt_model = DynamicEVT(
                window_size=self.window_size,
                threshold_quantile=self.threshold_quantile,
                min_exceedances=self.min_exceedances
            )
        elif self.evt_method == 'conditional' or self.conditional:
            self.evt_model = ConditionalEVT(
                threshold_quantile=self.threshold_quantile,
                min_exceedances=self.min_exceedances,
                condition_factors=self.condition_factors
            )
        elif self.evt_method == 'bayesian' or self.bayesian:
            self.evt_model = BayesianEVT(
                threshold_quantile=self.threshold_quantile,
                min_exceedances=self.min_exceedances,
                n_samples=self.n_samples
            )
        else:
            raise ValueError(f"Unknown EVT method: {self.evt_method}")
        
        # Fit EVT model
        self.evt_model.fit(returns)
        
        self.fitted = True
        logger.info(f"{self.name} risk model fitted successfully")
    
    def predict_var(self, 
                    returns: pd.DataFrame, 
                    weights: np.ndarray,
                    confidence_level: float = 0.95) -> float:
        """
        Predict Value-at-Risk using EVT.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns (used for conditional EVT)
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for VaR
            
        Returns:
        --------
        float
            VaR prediction
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # For conditional EVT, calculate factor values
        factor_values = None
        if self.evt_method == 'conditional' or self.conditional:
            # Calculate portfolio returns
            if returns.shape[1] > 1:
                portfolio_returns = returns.values @ weights
            else:
                portfolio_returns = returns.values.flatten()
            
            # Calculate factors
            factor_values = {}
            for factor in self.condition_factors:
                if factor == 'volatility':
                    # Use recent volatility
                    recent_returns = portfolio_returns[-21:]
                    factor_values['volatility'] = np.std(recent_returns) * np.sqrt(252)
                elif factor == 'trend':
                    # Use recent trend
                    recent_returns = portfolio_returns[-63:]
                    factor_values['trend'] = np.mean(recent_returns) * 252
                elif factor == 'skewness':
                    # Use recent skewness
                    recent_returns = portfolio_returns[-63:]
                    factor_values['skewness'] = stats.skew(recent_returns)
                elif factor == 'kurtosis':
                    # Use recent excess kurtosis
                    recent_returns = portfolio_returns[-63:]
                    factor_values['kurtosis'] = stats.kurtosis(recent_returns)
            
            # Standardize factors
            for factor in factor_values:
                factor_values[factor] = (factor_values[factor] - np.mean(factor_values[factor])) / np.std(factor_values[factor])
        
        # Calculate VaR using EVT model
        if self.evt_method == 'gpd' or (not self.dynamic and not self.conditional and not self.bayesian):
            var = self.evt_model.calculate_var(confidence_level)
        elif self.evt_method == 'dynamic' or self.dynamic:
            var = self.evt_model.calculate_var(confidence_level)
        elif self.evt_method == 'conditional' or self.conditional:
            var = self.evt_model.calculate_var(confidence_level, factor_values)
        elif self.evt_method == 'bayesian' or self.bayesian:
            var_result = self.evt_model.calculate_var(confidence_level)
            var = var_result['var']  # Use mean estimate
        else:
            raise ValueError(f"Unknown EVT method: {self.evt_method}")
        
        return var
    
    def predict_es(self, 
                   returns: pd.DataFrame, 
                   weights: np.ndarray,
                   confidence_level: float = 0.95) -> float:
        """
        Predict Expected Shortfall using EVT.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns (used for conditional EVT)
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for ES
            
        Returns:
        --------
        float
            ES prediction
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # For conditional EVT, calculate factor values
        factor_values = None
        if self.evt_method == 'conditional' or self.conditional:
            # Calculate portfolio returns
            if returns.shape[1] > 1:
                portfolio_returns = returns.values @ weights
            else:
                portfolio_returns = returns.values.flatten()
            
            # Calculate factors
            factor_values = {}
            for factor in self.condition_factors:
                if factor == 'volatility':
                    # Use recent volatility
                    recent_returns = portfolio_returns[-21:]
                    factor_values['volatility'] = np.std(recent_returns) * np.sqrt(252)
                elif factor == 'trend':
                    # Use recent trend
                    recent_returns = portfolio_returns[-63:]
                    factor_values['trend'] = np.mean(recent_returns) * 252
                elif factor == 'skewness':
                    # Use recent skewness
                    recent_returns = portfolio_returns[-63:]
                    factor_values['skewness'] = stats.skew(recent_returns)
                elif factor == 'kurtosis':
                    # Use recent excess kurtosis
                    recent_returns = portfolio_returns[-63:]
                    factor_values['kurtosis'] = stats.kurtosis(recent_returns)
            
            # Standardize factors
            for factor in factor_values:
                factor_values[factor] = (factor_values[factor] - np.mean(factor_values[factor])) / np.std(factor_values[factor])
        
        # Calculate ES using EVT model
        if self.evt_method == 'gpd' or (not self.dynamic and not self.conditional and not self.bayesian):
            es = self.evt_model.calculate_expected_shortfall(confidence_level)
        elif self.evt_method == 'dynamic' or self.dynamic:
            es = self.evt_model.calculate_expected_shortfall(confidence_level)
        elif self.evt_method == 'conditional' or self.conditional:
            es = self.evt_model.calculate_expected_shortfall(confidence_level, factor_values)
        elif self.evt_method == 'bayesian' or self.bayesian:
            es_result = self.evt_model.calculate_expected_shortfall(confidence_level)
            es = es_result['es']  # Use mean estimate
        else:
            raise ValueError(f"Unknown EVT method: {self.evt_method}")
        
        return es


class MonteCarloRiskModel(BaseRiskModel):
    """
    Monte Carlo simulation risk model.
    
    This model uses Monte Carlo simulation to generate scenarios and
    estimate risk metrics based on simulated returns.
    """
    
    def __init__(self, 
                 name: str = "MonteCarlo",
                 n_scenarios: int = 10000,
                 simulation_method: str = "normal",
                 time_weighted: bool = False,
                 half_life: int = 63,
                 use_t: bool = False,
                 t_df: int = 5,
                 use_copula: bool = False,
                 copula_type: str = "gaussian"):
        """
        Initialize the Monte Carlo risk model.
        
        Parameters:
        -----------
        name : str, default="MonteCarlo"
            Name of the risk model
        n_scenarios : int, default=10000
            Number of Monte Carlo scenarios
        simulation_method : str, default="normal"
            Simulation method ('normal', 'historical', 't', 'copula')
        time_weighted : bool, default=False
            Whether to use time-weighted estimates
        half_life : int, default=63
            Half-life parameter for exponential weighting (trading days)
        use_t : bool, default=False
            Whether to use t-distribution
        t_df : int, default=5
            Degrees of freedom for t-distribution
        use_copula : bool, default=False
            Whether to use copula
        copula_type : str, default="gaussian"
            Copula type ('gaussian', 't')
        """
        super().__init__(name)
        self.n_scenarios = n_scenarios
        self.simulation_method = simulation_method
        self.time_weighted = time_weighted
        self.half_life = half_life
        self.decay_factor = np.exp(np.log(0.5) / half_life) if time_weighted else None
        self.use_t = use_t
        self.t_df = t_df
        self.use_copula = use_copula
        self.copula_type = copula_type
        self.mean = None
        self.cov = None
        self.historical_returns = None
        self.marginal_params = None
    
    def _get_weights(self, n_obs: int) -> np.ndarray:
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
        if not self.time_weighted:
            return np.ones(n_obs) / n_obs
        
        # Calculate exponential weights
        weights = np.power(self.decay_factor, np.arange(n_obs-1, -1, -1))
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        return weights
    
    def _fit_marginals(self, returns: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Fit marginal distributions for copula.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
            
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Dictionary of marginal distribution parameters
        """
        marginal_params = {}
        
        for col in returns.columns:
            data = returns[col].values
            
            # Fit normal distribution
            loc, scale = stats.norm.fit(data)
            
            # Fit t distribution
            t_params = stats.t.fit(data)
            
            # Store parameters
            marginal_params[col] = {
                'normal': {'loc': loc, 'scale': scale},
                't': {'df': t_params[0], 'loc': t_params[1], 'scale': t_params[2]}
            }
        
        return marginal_params
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the Monte Carlo risk model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info(f"Fitting {self.name} risk model with {self.simulation_method} method")
        
        # Store historical returns
        self.historical_returns = returns.copy()
        
        # Calculate weights
        weights = self._get_weights(len(returns))
        
        if self.time_weighted:
            # Calculate weighted mean and covariance
            self.mean = np.average(returns.values, axis=0, weights=weights)
            
            # Center the data
            centered_data = returns.values - self.mean
            
            # Calculate weighted covariance
            weights_reshaped = weights.reshape(-1, 1)
            self.cov = np.dot(centered_data.T * weights_reshaped.T, centered_data)
        else:
            # Use standard mean and covariance
            self.mean = returns.mean().values
            self.cov = returns.cov().values
        
        # Fit marginal distributions for copula
        if self.use_copula or self.simulation_method == 'copula':
            self.marginal_params = self._fit_marginals(returns)
        
        self.fitted = True
        logger.info(f"{self.name} risk model fitted successfully")
    
    def _simulate_normal(self, n_scenarios: int, n_assets: int) -> np.ndarray:
        """
        Simulate returns using multivariate normal distribution.
        
        Parameters:
        -----------
        n_scenarios : int
            Number of scenarios
        n_assets : int
            Number of assets
            
        Returns:
        --------
        np.ndarray
            Simulated returns with shape (n_scenarios, n_assets)
        """
        return np.random.multivariate_normal(self.mean, self.cov, size=n_scenarios)
    
    def _simulate_t(self, n_scenarios: int, n_assets: int) -> np.ndarray:
        """
        Simulate returns using multivariate t distribution.
        
        Parameters:
        -----------
        n_scenarios : int
            Number of scenarios
        n_assets : int
            Number of assets
            
        Returns:
        --------
        np.ndarray
            Simulated returns with shape (n_scenarios, n_assets)
        """
        # Generate multivariate normal
        normal_samples = np.random.multivariate_normal(np.zeros(n_assets), self.cov, size=n_scenarios)
        
        # Generate chi-squared samples
        chi2_samples = np.random.chisquare(self.t_df, size=n_scenarios) / self.t_df
        
        # Scale normal samples by sqrt(chi2) and add mean
        t_samples = self.mean + normal_samples / np.sqrt(chi2_samples)[:, np.newaxis]
        
        return t_samples
    
    def _simulate_historical(self, n_scenarios: int, n_assets: int) -> np.ndarray:
        """
        Simulate returns using historical bootstrap.
        
        Parameters:
        -----------
        n_scenarios : int
            Number of scenarios
        n_assets : int
            Number of assets
            
        Returns:
        --------
        np.ndarray
            Simulated returns with shape (n_scenarios, n_assets)
        """
        # Calculate weights for sampling
        weights = self._get_weights(len(self.historical_returns))
        
        # Sample indices with replacement
        indices = np.random.choice(
            len(self.historical_returns),
            size=n_scenarios,
            p=weights
        )
        
        # Get historical returns for sampled indices
        return self.historical_returns.values[indices]
    
    def _simulate_copula(self, n_scenarios: int, n_assets: int) -> np.ndarray:
        """
        Simulate returns using copula.
        
        Parameters:
        -----------
        n_scenarios : int
            Number of scenarios
        n_assets : int
            Number of assets
            
        Returns:
        --------
        np.ndarray
            Simulated returns with shape (n_scenarios, n_assets)
        """
        # Generate uniform marginals using copula
        if self.copula_type == 'gaussian':
            # Convert covariance to correlation
            std = np.sqrt(np.diag(self.cov))
            corr = self.cov / np.outer(std, std)
            
            # Generate multivariate normal
            normal_samples = np.random.multivariate_normal(np.zeros(n_assets), corr, size=n_scenarios)
            
            # Convert to uniform using normal CDF
            uniform_samples = stats.norm.cdf(normal_samples)
        elif self.copula_type == 't':
            # Convert covariance to correlation
            std = np.sqrt(np.diag(self.cov))
            corr = self.cov / np.outer(std, std)
            
            # Generate multivariate t
            normal_samples = np.random.multivariate_normal(np.zeros(n_assets), corr, size=n_scenarios)
            chi2_samples = np.random.chisquare(self.t_df, size=n_scenarios) / self.t_df
            t_samples = normal_samples / np.sqrt(chi2_samples)[:, np.newaxis]
            
            # Convert to uniform using t CDF
            uniform_samples = stats.t.cdf(t_samples, self.t_df)
        else:
            raise ValueError(f"Unknown copula type: {self.copula_type}")
        
        # Convert uniform marginals to returns using inverse CDF
        simulated_returns = np.zeros((n_scenarios, n_assets))
        
        for i, col in enumerate(self.historical_returns.columns):
            if self.use_t:
                # Use t distribution
                params = self.marginal_params[col]['t']
                simulated_returns[:, i] = stats.t.ppf(
                    uniform_samples[:, i],
                    df=params['df'],
                    loc=params['loc'],
                    scale=params['scale']
                )
            else:
                # Use normal distribution
                params = self.marginal_params[col]['normal']
                simulated_returns[:, i] = stats.norm.ppf(
                    uniform_samples[:, i],
                    loc=params['loc'],
                    scale=params['scale']
                )
        
        return simulated_returns
    
    def simulate_returns(self, n_scenarios: int = None) -> np.ndarray:
        """
        Simulate returns using specified method.
        
        Parameters:
        -----------
        n_scenarios : int, optional
            Number of scenarios (defaults to self.n_scenarios)
            
        Returns:
        --------
        np.ndarray
            Simulated returns with shape (n_scenarios, n_assets)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before simulation")
        
        if n_scenarios is None:
            n_scenarios = self.n_scenarios
        
        n_assets = len(self.historical_returns.columns)
        
        # Simulate returns based on method
        if self.simulation_method == 'normal' and not self.use_t and not self.use_copula:
            return self._simulate_normal(n_scenarios, n_assets)
        elif self.simulation_method == 't' or self.use_t:
            return self._simulate_t(n_scenarios, n_assets)
        elif self.simulation_method == 'historical':
            return self._simulate_historical(n_scenarios, n_assets)
        elif self.simulation_method == 'copula' or self.use_copula:
            return self._simulate_copula(n_scenarios, n_assets)
        else:
            raise ValueError(f"Unknown simulation method: {self.simulation_method}")
    
    def predict_var(self, 
                    returns: pd.DataFrame, 
                    weights: np.ndarray,
                    confidence_level: float = 0.95) -> float:
        """
        Predict Value-at-Risk using Monte Carlo simulation.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns (not used in prediction, only for consistency)
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for VaR
            
        Returns:
        --------
        float
            VaR prediction
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Simulate returns
        simulated_returns = self.simulate_returns()
        
        # Calculate portfolio returns
        portfolio_returns = simulated_returns @ weights
        
        # Calculate VaR
        var = -np.percentile(portfolio_returns, 100 * (1 - confidence_level))
        
        return var
    
    def predict_es(self, 
                   returns: pd.DataFrame, 
                   weights: np.ndarray,
                   confidence_level: float = 0.95) -> float:
        """
        Predict Expected Shortfall using Monte Carlo simulation.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns (not used in prediction, only for consistency)
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for ES
            
        Returns:
        --------
        float
            ES prediction
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Simulate returns
        simulated_returns = self.simulate_returns()
        
        # Calculate portfolio returns
        portfolio_returns = simulated_returns @ weights
        
        # Calculate VaR threshold
        var_threshold = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
        
        # Calculate ES
        tail = portfolio_returns[portfolio_returns <= var_threshold]
        
        if len(tail) > 0:
            es = -np.mean(tail)
        else:
            es = -np.mean(portfolio_returns)  # Fallback
        
        return es


class EnsembleRiskModel:
    """
    Ensemble risk model that combines multiple risk models.
    
    This class implements various ensemble methods to combine predictions
    from multiple risk models for improved accuracy and robustness.
    """
    
    def __init__(self, 
                 models: List[BaseRiskModel],
                 ensemble_method: str = "average",
                 weights: Optional[Dict[str, float]] = None,
                 train_weights: bool = False,
                 meta_model_type: str = "linear",
                 meta_model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the ensemble risk model.
        
        Parameters:
        -----------
        models : List[BaseRiskModel]
            List of risk models to ensemble
        ensemble_method : str, default="average"
            Ensemble method ('average', 'weighted', 'meta')
        weights : Dict[str, float], optional
            Dictionary of model weights (model name -> weight)
        train_weights : bool, default=False
            Whether to train weights using historical performance
        meta_model_type : str, default="linear"
            Meta-model type for 'meta' ensemble method
            ('linear', 'ridge', 'lasso', 'elasticnet', 'rf', 'gbm')
        meta_model_params : Dict[str, Any], optional
            Parameters for meta-model
        """
        self.models = models
        self.ensemble_method = ensemble_method
        self.weights = weights or {model.name: 1.0 for model in models}
        self.train_weights = train_weights
        self.meta_model_type = meta_model_type
        self.meta_model_params = meta_model_params or {}
        self.meta_model_var = None
        self.meta_model_es = None
        self.fitted = False
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the ensemble risk model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info(f"Fitting ensemble risk model with {self.ensemble_method} method")
        
        # Fit individual models
        for model in self.models:
            model.fit(returns)
        
        # Train weights if requested
        if self.train_weights:
            self._train_weights(returns)
        
        # Train meta-model if using meta ensemble
        if self.ensemble_method == 'meta':
            self._train_meta_model(returns)
        
        self.fitted = True
        logger.info("Ensemble risk model fitted successfully")
    
    def _train_weights(self, returns: pd.DataFrame) -> None:
        """
        Train model weights based on historical performance.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info("Training ensemble weights")
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize performance metrics
        var_errors = {model.name: [] for model in self.models}
        es_errors = {model.name: [] for model in self.models}
        
        # Equal weights for portfolio
        weights = np.ones(returns.shape[1]) / returns.shape[1]
        
        for train_idx, test_idx in tscv.split(returns):
            train_returns = returns.iloc[train_idx]
            test_returns = returns.iloc[test_idx]
            
            # Calculate actual portfolio returns
            portfolio_returns = test_returns.values @ weights
            
            # Calculate actual VaR and ES
            actual_var = -np.percentile(portfolio_returns, 5)  # 95% VaR
            var_threshold = np.percentile(portfolio_returns, 5)
            actual_es = -np.mean(portfolio_returns[portfolio_returns <= var_threshold])
            
            # Get predictions from each model
            for model in self.models:
                # Fit model on training data
                model.fit(train_returns)
                
                # Predict VaR and ES
                pred_var = model.predict_var(test_returns, weights, 0.95)
                pred_es = model.predict_es(test_returns, weights, 0.95)
                
                # Calculate errors
                var_error = np.abs(pred_var - actual_var) / actual_var
                es_error = np.abs(pred_es - actual_es) / actual_es
                
                var_errors[model.name].append(var_error)
                es_errors[model.name].append(es_error)
        
        # Calculate average errors
        avg_var_errors = {name: np.mean(errors) for name, errors in var_errors.items()}
        avg_es_errors = {name: np.mean(errors) for name, errors in es_errors.items()}
        
        # Calculate weights (inverse of errors)
        var_weights = {name: 1 / error if error > 0 else 1.0 for name, error in avg_var_errors.items()}
        es_weights = {name: 1 / error if error > 0 else 1.0 for name, error in avg_es_errors.items()}
        
        # Normalize weights
        var_sum = sum(var_weights.values())
        es_sum = sum(es_weights.values())
        
        var_weights = {name: weight / var_sum for name, weight in var_weights.items()}
        es_weights = {name: weight / es_sum for name, weight in es_weights.items()}
        
        # Use average of VaR and ES weights
        self.weights = {
            name: (var_weights[name] + es_weights[name]) / 2
            for name in var_weights
        }
        
        logger.info(f"Trained weights: {self.weights}")
    
    def _train_meta_model(self, returns: pd.DataFrame) -> None:
        """
        Train meta-model for ensemble.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info(f"Training meta-model with {self.meta_model_type} method")
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Equal weights for portfolio
        weights = np.ones(returns.shape[1]) / returns.shape[1]
        
        # Prepare data for meta-model
        X_var = []
        y_var = []
        X_es = []
        y_es = []
        
        for train_idx, test_idx in tscv.split(returns):
            train_returns = returns.iloc[train_idx]
            test_returns = returns.iloc[test_idx]
            
            # Calculate actual portfolio returns
            portfolio_returns = test_returns.values @ weights
            
            # Calculate actual VaR and ES
            actual_var = -np.percentile(portfolio_returns, 5)  # 95% VaR
            var_threshold = np.percentile(portfolio_returns, 5)
            actual_es = -np.mean(portfolio_returns[portfolio_returns <= var_threshold])
            
            # Get predictions from each model
            var_preds = []
            es_preds = []
            
            for model in self.models:
                # Fit model on training data
                model.fit(train_returns)
                
                # Predict VaR and ES
                pred_var = model.predict_var(test_returns, weights, 0.95)
                pred_es = model.predict_es(test_returns, weights, 0.95)
                
                var_preds.append(pred_var)
                es_preds.append(pred_es)
            
            X_var.append(var_preds)
            y_var.append(actual_var)
            X_es.append(es_preds)
            y_es.append(actual_es)
        
        # Convert to numpy arrays
        X_var = np.array(X_var)
        y_var = np.array(y_var)
        X_es = np.array(X_es)
        y_es = np.array(y_es)
        
        # Initialize meta-models
        if self.meta_model_type == 'linear':
            self.meta_model_var = LinearRegression(**self.meta_model_params)
            self.meta_model_es = LinearRegression(**self.meta_model_params)
        elif self.meta_model_type == 'ridge':
            self.meta_model_var = Ridge(**self.meta_model_params)
            self.meta_model_es = Ridge(**self.meta_model_params)
        elif self.meta_model_type == 'lasso':
            self.meta_model_var = Lasso(**self.meta_model_params)
            self.meta_model_es = Lasso(**self.meta_model_params)
        elif self.meta_model_type == 'elasticnet':
            self.meta_model_var = ElasticNet(**self.meta_model_params)
            self.meta_model_es = ElasticNet(**self.meta_model_params)
        elif self.meta_model_type == 'rf':
            self.meta_model_var = RandomForestRegressor(**self.meta_model_params)
            self.meta_model_es = RandomForestRegressor(**self.meta_model_params)
        elif self.meta_model_type == 'gbm':
            self.meta_model_var = GradientBoostingRegressor(**self.meta_model_params)
            self.meta_model_es = GradientBoostingRegressor(**self.meta_model_params)
        else:
            raise ValueError(f"Unknown meta-model type: {self.meta_model_type}")
        
        # Fit meta-models
        self.meta_model_var.fit(X_var, y_var)
        self.meta_model_es.fit(X_es, y_es)
        
        # Evaluate meta-models
        var_r2 = self.meta_model_var.score(X_var, y_var)
        es_r2 = self.meta_model_es.score(X_es, y_es)
        
        logger.info(f"Meta-model R² scores: VaR={var_r2:.4f}, ES={es_r2:.4f}")
        
        # Log feature importances if available
        if hasattr(self.meta_model_var, 'feature_importances_'):
            var_importances = self.meta_model_var.feature_importances_
            es_importances = self.meta_model_es.feature_importances_
            
            for i, model in enumerate(self.models):
                logger.info(f"Model {model.name} importance: VaR={var_importances[i]:.4f}, ES={es_importances[i]:.4f}")
        elif hasattr(self.meta_model_var, 'coef_'):
            var_coefs = self.meta_model_var.coef_
            es_coefs = self.meta_model_es.coef_
            
            for i, model in enumerate(self.models):
                logger.info(f"Model {model.name} coefficient: VaR={var_coefs[i]:.4f}, ES={es_coefs[i]:.4f}")
    
    def predict_var(self, 
                    returns: pd.DataFrame, 
                    weights: np.ndarray,
                    confidence_level: float = 0.95) -> float:
        """
        Predict Value-at-Risk using ensemble.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for VaR
            
        Returns:
        --------
        float
            VaR prediction
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions from each model
        predictions = []
        for model in self.models:
            pred = model.predict_var(returns, weights, confidence_level)
            predictions.append(pred)
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'average':
            # Simple average
            ensemble_pred = np.mean(predictions)
        elif self.ensemble_method == 'weighted':
            # Weighted average
            model_weights = [self.weights[model.name] for model in self.models]
            ensemble_pred = np.average(predictions, weights=model_weights)
        elif self.ensemble_method == 'meta':
            # Meta-model prediction
            ensemble_pred = self.meta_model_var.predict(np.array([predictions]))[0]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble_pred
    
    def predict_es(self, 
                   returns: pd.DataFrame, 
                   weights: np.ndarray,
                   confidence_level: float = 0.95) -> float:
        """
        Predict Expected Shortfall using ensemble.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for ES
            
        Returns:
        --------
        float
            ES prediction
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions from each model
        predictions = []
        for model in self.models:
            pred = model.predict_es(returns, weights, confidence_level)
            predictions.append(pred)
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'average':
            # Simple average
            ensemble_pred = np.mean(predictions)
        elif self.ensemble_method == 'weighted':
            # Weighted average
            model_weights = [self.weights[model.name] for model in self.models]
            ensemble_pred = np.average(predictions, weights=model_weights)
        elif self.ensemble_method == 'meta':
            # Meta-model prediction
            ensemble_pred = self.meta_model_es.predict(np.array([predictions]))[0]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble_pred
    
    def get_model_predictions(self, 
                              returns: pd.DataFrame, 
                              weights: np.ndarray,
                              confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
        """
        Get predictions from all models.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for risk metrics
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Dictionary of model predictions (model name -> {var, es})
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions from each model
        predictions = {}
        for model in self.models:
            var_pred = model.predict_var(returns, weights, confidence_level)
            es_pred = model.predict_es(returns, weights, confidence_level)
            
            predictions[model.name] = {
                'var': var_pred,
                'es': es_pred
            }
        
        # Add ensemble prediction
        ensemble_var = self.predict_var(returns, weights, confidence_level)
        ensemble_es = self.predict_es(returns, weights, confidence_level)
        
        predictions['Ensemble'] = {
            'var': ensemble_var,
            'es': ensemble_es
        }
        
        return predictions
    
    def plot_model_predictions(self, 
                               returns: pd.DataFrame, 
                               weights: np.ndarray,
                               confidence_level: float = 0.95) -> plt.Figure:
        """
        Plot predictions from all models.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for risk metrics
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with model predictions
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Get predictions from all models
        predictions = self.get_model_predictions(returns, weights, confidence_level)
        
        # Extract VaR and ES predictions
        model_names = list(predictions.keys())
        var_preds = [predictions[name]['var'] for name in model_names]
        es_preds = [predictions[name]['es'] for name in model_names]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot VaR predictions
        axes[0].barh(model_names, var_preds)
        axes[0].set_title(f'Value-at-Risk ({confidence_level*100:.0f}%)')
        axes[0].set_xlabel('VaR')
        axes[0].grid(True, axis='x')
        
        # Plot ES predictions
        axes[1].barh(model_names, es_preds)
        axes[1].set_title(f'Expected Shortfall ({confidence_level*100:.0f}%)')
        axes[1].set_xlabel('ES')
        axes[1].grid(True, axis='x')
        
        # Highlight ensemble prediction
        axes[0].get_children()[model_names.index('Ensemble')].set_color('red')
        axes[1].get_children()[model_names.index('Ensemble')].set_color('red')
        
        plt.tight_layout()
        return fig


class AdaptiveEnsembleRiskModel(EnsembleRiskModel):
    """
    Adaptive ensemble risk model that dynamically adjusts weights.
    
    This class extends the ensemble risk model with dynamic weight adjustment
    based on recent performance and market conditions.
    """
    
    def __init__(self, 
                 models: List[BaseRiskModel],
                 ensemble_method: str = "weighted",
                 weights: Optional[Dict[str, float]] = None,
                 train_weights: bool = True,
                 meta_model_type: str = "linear",
                 meta_model_params: Optional[Dict[str, Any]] = None,
                 adaptive_window: int = 63,
                 regime_dependent: bool = True,
                 learning_rate: float = 0.1):
        """
        Initialize the adaptive ensemble risk model.
        
        Parameters:
        -----------
        models : List[BaseRiskModel]
            List of risk models to ensemble
        ensemble_method : str, default="weighted"
            Ensemble method ('average', 'weighted', 'meta')
        weights : Dict[str, float], optional
            Dictionary of initial model weights (model name -> weight)
        train_weights : bool, default=True
            Whether to train weights using historical performance
        meta_model_type : str, default="linear"
            Meta-model type for 'meta' ensemble method
        meta_model_params : Dict[str, Any], optional
            Parameters for meta-model
        adaptive_window : int, default=63
            Window size for adaptive weight adjustment (trading days)
        regime_dependent : bool, default=True
            Whether to use regime-dependent weights
        learning_rate : float, default=0.1
            Learning rate for weight updates
        """
        super().__init__(
            models=models,
            ensemble_method=ensemble_method,
            weights=weights,
            train_weights=train_weights,
            meta_model_type=meta_model_type,
            meta_model_params=meta_model_params
        )
        self.adaptive_window = adaptive_window
        self.regime_dependent = regime_dependent
        self.learning_rate = learning_rate
        self.regime_weights = {}
        self.current_regime = None
        self.performance_history = {model.name: [] for model in models}
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit the adaptive ensemble risk model to historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info(f"Fitting adaptive ensemble risk model")
        
        # Fit individual models
        for model in self.models:
            model.fit(returns)
        
        # Train initial weights
        if self.train_weights:
            self._train_weights(returns)
        
        # Train meta-model if using meta ensemble
        if self.ensemble_method == 'meta':
            self._train_meta_model(returns)
        
        # Train regime-dependent weights if requested
        if self.regime_dependent:
            self._train_regime_weights(returns)
        
        self.fitted = True
        logger.info("Adaptive ensemble risk model fitted successfully")
    
    def _train_regime_weights(self, returns: pd.DataFrame) -> None:
        """
        Train regime-dependent weights.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info("Training regime-dependent weights")
        
        try:
            # Import regime switching model
            from regime_switching import AdaptiveRegimeModel
            
            # Initialize regime model
            regime_model = AdaptiveRegimeModel(
                n_regimes=3,
                methods=['hmm', 'volatility'],
                lookback_period=63
            )
            
            # Fit regime model
            regime_model.fit(returns)
            
            # Get regime classifications
            regimes = regime_model.classify_regimes(returns)
            
            # Train weights for each regime
            for regime in np.unique(regimes):
                # Get returns for this regime
                regime_returns = returns.iloc[regimes == regime]
                
                if len(regime_returns) > 100:  # Ensure sufficient data
                    # Initialize weights for this regime
                    regime_weights = {model.name: 1.0 for model in self.models}
                    
                    # Use time series cross-validation
                    tscv = TimeSeriesSplit(n_splits=3)
                    
                    # Initialize performance metrics
                    var_errors = {model.name: [] for model in self.models}
                    es_errors = {model.name: [] for model in self.models}
                    
                    # Equal weights for portfolio
                    portfolio_weights = np.ones(returns.shape[1]) / returns.shape[1]
                    
                    for train_idx, test_idx in tscv.split(regime_returns):
                        train_returns = regime_returns.iloc[train_idx]
                        test_returns = regime_returns.iloc[test_idx]
                        
                        # Calculate actual portfolio returns
                        portfolio_returns = test_returns.values @ portfolio_weights
                        
                        # Calculate actual VaR and ES
                        actual_var = -np.percentile(portfolio_returns, 5)  # 95% VaR
                        var_threshold = np.percentile(portfolio_returns, 5)
                        actual_es = -np.mean(portfolio_returns[portfolio_returns <= var_threshold])
                        
                        # Get predictions from each model
                        for model in self.models:
                            # Fit model on training data
                            model.fit(train_returns)
                            
                            # Predict VaR and ES
                            pred_var = model.predict_var(test_returns, portfolio_weights, 0.95)
                            pred_es = model.predict_es(test_returns, portfolio_weights, 0.95)
                            
                            # Calculate errors
                            var_error = np.abs(pred_var - actual_var) / actual_var
                            es_error = np.abs(pred_es - actual_es) / actual_es
                            
                            var_errors[model.name].append(var_error)
                            es_errors[model.name].append(es_error)
                    
                    # Calculate average errors
                    avg_var_errors = {name: np.mean(errors) for name, errors in var_errors.items()}
                    avg_es_errors = {name: np.mean(errors) for name, errors in es_errors.items()}
                    
                    # Calculate weights (inverse of errors)
                    var_weights = {name: 1 / error if error > 0 else 1.0 for name, error in avg_var_errors.items()}
                    es_weights = {name: 1 / error if error > 0 else 1.0 for name, error in avg_es_errors.items()}
                    
                    # Normalize weights
                    var_sum = sum(var_weights.values())
                    es_sum = sum(es_weights.values())
                    
                    var_weights = {name: weight / var_sum for name, weight in var_weights.items()}
                    es_weights = {name: weight / es_sum for name, weight in es_weights.items()}
                    
                    # Use average of VaR and ES weights
                    regime_weights = {
                        name: (var_weights[name] + es_weights[name]) / 2
                        for name in var_weights
                    }
                    
                    # Store weights for this regime
                    self.regime_weights[regime] = regime_weights
                    
                    logger.info(f"Regime {regime} weights: {regime_weights}")
                else:
                    # Use default weights for this regime
                    self.regime_weights[regime] = self.weights.copy()
                    
                    logger.info(f"Insufficient data for regime {regime}, using default weights")
            
            # Set current regime
            self.current_regime = regimes[-1]
            
            # Use regime-specific weights if available
            if self.current_regime in self.regime_weights:
                self.weights = self.regime_weights[self.current_regime].copy()
                
                logger.info(f"Using weights for regime {self.current_regime}")
        except ImportError:
            logger.warning("Failed to import regime switching model, using default weights")
            self.regime_dependent = False
        except Exception as e:
            logger.error(f"Error training regime-dependent weights: {e}")
            self.regime_dependent = False
    
    def update_weights(self, 
                       returns: pd.DataFrame, 
                       weights: np.ndarray,
                       actual_var: float,
                       actual_es: float,
                       confidence_level: float = 0.95) -> None:
        """
        Update model weights based on recent performance.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        actual_var : float
            Actual VaR
        actual_es : float
            Actual ES
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before updating weights")
        
        logger.info("Updating ensemble weights")
        
        # Update current regime if regime-dependent
        if self.regime_dependent:
            try:
                # Import regime switching model
                from regime_switching import AdaptiveRegimeModel
                
                # Initialize regime model
                regime_model = AdaptiveRegimeModel(
                    n_regimes=3,
                    methods=['hmm', 'volatility'],
                    lookback_period=63
                )
                
                # Fit regime model
                regime_model.fit(returns)
                
                # Get current regime
                new_regime = regime_model.classify_regimes(returns)[-1]
                
                # Update weights if regime changed
                if new_regime != self.current_regime:
                    logger.info(f"Regime changed from {self.current_regime} to {new_regime}")
                    
                    self.current_regime = new_regime
                    
                    # Use regime-specific weights if available
                    if self.current_regime in self.regime_weights:
                        self.weights = self.regime_weights[self.current_regime].copy()
                        
                        logger.info(f"Using weights for regime {self.current_regime}")
            except Exception as e:
                logger.error(f"Error updating regime: {e}")
        
        # Get predictions from each model
        for model in self.models:
            # Predict VaR and ES
            pred_var = model.predict_var(returns, weights, confidence_level)
            pred_es = model.predict_es(returns, weights, confidence_level)
            
            # Calculate errors
            var_error = np.abs(pred_var - actual_var) / actual_var
            es_error = np.abs(pred_es - actual_es) / actual_es
            
            # Average error
            avg_error = (var_error + es_error) / 2
            
            # Store performance
            self.performance_history[model.name].append(avg_error)
            
            # Keep only recent performance
            if len(self.performance_history[model.name]) > self.adaptive_window:
                self.performance_history[model.name] = self.performance_history[model.name][-self.adaptive_window:]
        
        # Calculate recent performance
        recent_performance = {
            name: np.mean(errors) if errors else 1.0
            for name, errors in self.performance_history.items()
        }
        
        # Calculate new weights (inverse of errors)
        new_weights = {
            name: 1 / error if error > 0 else 1.0
            for name, error in recent_performance.items()
        }
        
        # Normalize weights
        weight_sum = sum(new_weights.values())
        new_weights = {name: weight / weight_sum for name, weight in new_weights.items()}
        
        # Update weights with learning rate
        for name in self.weights:
            self.weights[name] = (1 - self.learning_rate) * self.weights[name] + self.learning_rate * new_weights[name]
        
        # Normalize weights again
        weight_sum = sum(self.weights.values())
        self.weights = {name: weight / weight_sum for name, weight in self.weights.items()}
        
        logger.info(f"Updated weights: {self.weights}")
        
        # Update regime-specific weights if regime-dependent
        if self.regime_dependent and self.current_regime is not None:
            self.regime_weights[self.current_regime] = self.weights.copy()
    
    def predict_var(self, 
                    returns: pd.DataFrame, 
                    weights: np.ndarray,
                    confidence_level: float = 0.95) -> float:
        """
        Predict Value-at-Risk using adaptive ensemble.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for VaR
            
        Returns:
        --------
        float
            VaR prediction
        """
        # Use parent class implementation
        return super().predict_var(returns, weights, confidence_level)
    
    def predict_es(self, 
                   returns: pd.DataFrame, 
                   weights: np.ndarray,
                   confidence_level: float = 0.95) -> float:
        """
        Predict Expected Shortfall using adaptive ensemble.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for ES
            
        Returns:
        --------
        float
            ES prediction
        """
        # Use parent class implementation
        return super().predict_es(returns, weights, confidence_level)
    
    def plot_weight_evolution(self) -> plt.Figure:
        """
        Plot the evolution of model weights.
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure with weight evolution
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot weights for each model
        for model in self.models:
            if model.name in self.performance_history and len(self.performance_history[model.name]) > 0:
                # Calculate weights over time
                errors = self.performance_history[model.name]
                weights = [1 / error if error > 0 else 1.0 for error in errors]
                
                # Normalize weights
                weight_sums = []
                for i in range(len(weights)):
                    weight_sum = sum(1 / self.performance_history[m.name][i] if self.performance_history[m.name][i] > 0 else 1.0
                                    for m in self.models
                                    if m.name in self.performance_history and i < len(self.performance_history[m.name]))
                    weight_sums.append(weight_sum)
                
                normalized_weights = [w / s for w, s in zip(weights, weight_sums)]
                
                # Plot weights
                ax.plot(normalized_weights, label=model.name)
        
        ax.set_title('Model Weight Evolution')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Weight')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_regime_weights(self) -> plt.Figure:
        """
        Plot weights for different regimes.
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure with regime weights
        """
        if not self.fitted or not self.regime_dependent or not self.regime_weights:
            raise ValueError("Model must be fitted with regime-dependent weights before plotting")
        
        # Create figure
        n_regimes = len(self.regime_weights)
        fig, axes = plt.subplots(1, n_regimes, figsize=(5 * n_regimes, 6))
        
        # Ensure axes is array-like
        if n_regimes == 1:
            axes = [axes]
        
        # Plot weights for each regime
        for i, (regime, weights) in enumerate(self.regime_weights.items()):
            model_names = list(weights.keys())
            model_weights = list(weights.values())
            
            axes[i].bar(model_names, model_weights)
            axes[i].set_title(f'Regime {regime} Weights')
            axes[i].set_ylabel('Weight')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, axis='y')
            
            # Highlight current regime
            if regime == self.current_regime:
                axes[i].set_facecolor('lightyellow')
        
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
    
    # Create individual risk models
    parametric_model = ParametricRiskModel(time_weighted=True)
    historical_model = HistoricalRiskModel(time_weighted=True)
    evt_model = EVTRiskModel(evt_method='gpd')
    monte_carlo_model = MonteCarloRiskModel(simulation_method='normal')
    
    # Create ensemble model
    ensemble_model = EnsembleRiskModel(
        models=[parametric_model, historical_model, evt_model, monte_carlo_model],
        ensemble_method='weighted',
        train_weights=True
    )
    
    # Fit models
    ensemble_model.fit(returns)
    
    # Equal weights for portfolio
    weights = np.ones(len(tickers)) / len(tickers)
    
    # Get predictions
    predictions = ensemble_model.get_model_predictions(returns, weights)
    
    print("Model Predictions:")
    for model_name, preds in predictions.items():
        print(f"{model_name}: VaR={preds['var']:.4f}, ES={preds['es']:.4f}")
    
    # Plot predictions
    pred_fig = ensemble_model.plot_model_predictions(returns, weights)
    pred_fig.savefig('ensemble_predictions.png')
    
    # Create adaptive ensemble model
    adaptive_model = AdaptiveEnsembleRiskModel(
        models=[parametric_model, historical_model, evt_model, monte_carlo_model],
        ensemble_method='weighted',
        train_weights=True,
        regime_dependent=True
    )
    
    # Fit adaptive model
    adaptive_model.fit(returns)
    
    # Simulate performance updates
    for i in range(10):
        # Use recent returns
        recent_returns = returns.iloc[-252:]
        
        # Calculate actual portfolio returns
        portfolio_returns = recent_returns.values @ weights
        
        # Calculate actual VaR and ES
        actual_var = -np.percentile(portfolio_returns, 5)  # 95% VaR
        var_threshold = np.percentile(portfolio_returns, 5)
        actual_es = -np.mean(portfolio_returns[portfolio_returns <= var_threshold])
        
        # Update weights
        adaptive_model.update_weights(recent_returns, weights, actual_var, actual_es)
    
    # Plot weight evolution
    weight_fig = adaptive_model.plot_weight_evolution()
    weight_fig.savefig('weight_evolution.png')
    
    # Plot regime weights if available
    try:
        regime_fig = adaptive_model.plot_regime_weights()
        regime_fig.savefig('regime_weights.png')
    except:
        pass
