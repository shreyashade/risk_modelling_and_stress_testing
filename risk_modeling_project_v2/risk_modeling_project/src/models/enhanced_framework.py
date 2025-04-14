"""
Enhanced Risk Modeling Framework

This module integrates all the enhanced components of the risk modeling framework,
including regime switching, dynamic calibration, improved EVT, ensemble models,
and machine learning approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Callable, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import os
import datetime
import json

# Import enhanced components
from src.models.regime_switching import RegimeSwitchingModel, MarkovRegimeSwitchingModel
from src.models.dynamic_calibration import DynamicCalibrationWindow
from src.models.extreme_value_theory import EnhancedEVT
from src.models.ensemble_risk_model import EnsembleRiskModel, AdaptiveEnsembleRiskModel
from src.models.monte_carlo import MonteCarloSimulation
from src.models.historical_simulation import HistoricalSimulation
from src.models.stress_testing import StressTesting
from src.models.machine_learning import (
    FeatureEngineering, RandomForestRiskModel, GradientBoostingRiskModel,
    NeuralNetworkRiskModel, LSTMRiskModel, EnsembleMLRiskModel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedRiskModelingFramework:
    """
    Enhanced Risk Modeling Framework that integrates all improved components.
    
    This framework provides a comprehensive approach to risk modeling with
    adaptive regime switching, dynamic calibration, improved EVT, ensemble models,
    and machine learning integration.
    """
    
    def __init__(self, 
                 name: str = "EnhancedRiskModel",
                 use_regime_switching: bool = True,
                 use_dynamic_calibration: bool = True,
                 use_enhanced_evt: bool = True,
                 use_ensemble_models: bool = True,
                 use_machine_learning: bool = True,
                 confidence_levels: List[float] = [0.95, 0.99],
                 time_horizons: List[int] = [1, 5, 10, 21],
                 base_currency: str = "USD",
                 random_state: Optional[int] = 42):
        """
        Initialize the enhanced risk modeling framework.
        
        Parameters:
        -----------
        name : str, default="EnhancedRiskModel"
            Name of the risk model
        use_regime_switching : bool, default=True
            Whether to use regime switching models
        use_dynamic_calibration : bool, default=True
            Whether to use dynamic calibration windows
        use_enhanced_evt : bool, default=True
            Whether to use enhanced EVT models
        use_ensemble_models : bool, default=True
            Whether to use ensemble risk models
        use_machine_learning : bool, default=True
            Whether to use machine learning models
        confidence_levels : List[float], default=[0.95, 0.99]
            Confidence levels for risk metrics
        time_horizons : List[int], default=[1, 5, 10, 21]
            Time horizons for risk metrics (in trading days)
        base_currency : str, default="USD"
            Base currency for risk metrics
        random_state : int, optional
            Random state for reproducibility
        """
        self.name = name
        self.use_regime_switching = use_regime_switching
        self.use_dynamic_calibration = use_dynamic_calibration
        self.use_enhanced_evt = use_enhanced_evt
        self.use_ensemble_models = use_ensemble_models
        self.use_machine_learning = use_machine_learning
        self.confidence_levels = confidence_levels
        self.time_horizons = time_horizons
        self.base_currency = base_currency
        self.random_state = random_state
        
        # Initialize components
        self.regime_model = None
        self.calibration_model = None
        self.evt_model = None
        self.ensemble_model = None
        self.ml_models = {}
        
        # Initialize results storage
        self.results = {}
        self.current_regime = None
        self.optimal_window = None
        self.model_weights = None
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
    
    def initialize_components(self, returns: pd.DataFrame) -> None:
        """
        Initialize all components of the framework.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        """
        logger.info("Initializing framework components")
        
        # Initialize regime switching model
        if self.use_regime_switching:
            logger.info("Initializing regime switching model")
            self.regime_model = MarkovRegimeSwitchingModel(
                n_regimes=3,
                regime_names=["Low Volatility", "Normal", "High Volatility"],
                estimation_method="mle",
                random_state=self.random_state
            )
        
        # Initialize dynamic calibration model
        if self.use_dynamic_calibration:
            logger.info("Initializing dynamic calibration model")
            self.calibration_model = DynamicCalibrationWindow(
                min_window=63,  # ~3 months
                max_window=756,  # ~3 years
                step_size=21,    # ~1 month
                metric="var_stability",
                confidence_level=0.95
            )
        
        # Initialize enhanced EVT model
        if self.use_enhanced_evt:
            logger.info("Initializing enhanced EVT model")
            self.evt_model = EnhancedEVT(
                threshold_method="dynamic",
                tail_fraction=0.1,
                min_tail_points=50,
                bootstrap_samples=1000,
                confidence_level=0.95
            )
        
        # Initialize ensemble model
        if self.use_ensemble_models:
            logger.info("Initializing ensemble risk model")
            self.ensemble_model = AdaptiveEnsembleRiskModel(
                models={
                    "historical": HistoricalSimulation(
                        confidence_level=0.95,
                        time_horizon=1
                    ),
                    "monte_carlo": MonteCarloSimulation(
                        confidence_level=0.95,
                        time_horizon=1,
                        n_simulations=10000,
                        random_state=self.random_state
                    ),
                    "evt": EnhancedEVT(
                        threshold_method="dynamic",
                        tail_fraction=0.1,
                        min_tail_points=50,
                        bootstrap_samples=1000,
                        confidence_level=0.95
                    )
                },
                initial_weights={
                    "historical": 0.3,
                    "monte_carlo": 0.3,
                    "evt": 0.4
                },
                adaptation_method="performance",
                performance_metric="var_violation",
                adaptation_lookback=63,  # ~3 months
                regime_dependent=True
            )
        
        # Initialize machine learning models
        if self.use_machine_learning:
            logger.info("Initializing machine learning models")
            
            # Create feature engineering
            feature_eng = FeatureEngineering(
                include_technical=True,
                include_statistical=True,
                include_volatility=True,
                include_correlation=True,
                include_tail=True
            )
            
            # Initialize ML models for different time horizons
            for horizon in self.time_horizons:
                self.ml_models[horizon] = {}
                
                # Initialize ML models for different confidence levels
                for cl in self.confidence_levels:
                    self.ml_models[horizon][cl] = EnsembleMLRiskModel(
                        name=f"ML_Ensemble_h{horizon}_cl{cl}",
                        target="var",
                        confidence_level=cl,
                        forecast_horizon=horizon,
                        feature_engineering=feature_eng,
                        models=["rf", "gbm", "nn"],
                        ensemble_method="weighted",
                        random_state=self.random_state
                    )
    
    def fit(self, returns: pd.DataFrame, prices: Optional[pd.DataFrame] = None) -> None:
        """
        Fit the enhanced risk modeling framework to historical data.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        prices : pd.DataFrame, optional
            Historical asset prices
        """
        logger.info(f"Fitting {self.name} framework")
        
        # Initialize components if not already initialized
        if (self.regime_model is None and self.use_regime_switching) or \
           (self.calibration_model is None and self.use_dynamic_calibration) or \
           (self.evt_model is None and self.use_enhanced_evt) or \
           (self.ensemble_model is None and self.use_ensemble_models) or \
           (len(self.ml_models) == 0 and self.use_machine_learning):
            self.initialize_components(returns)
        
        # Fit regime switching model
        if self.use_regime_switching and self.regime_model is not None:
            logger.info("Fitting regime switching model")
            self.regime_model.fit(returns)
            self.current_regime = self.regime_model.predict_regime(returns)
            logger.info(f"Current regime: {self.current_regime}")
        
        # Fit dynamic calibration model
        if self.use_dynamic_calibration and self.calibration_model is not None:
            logger.info("Fitting dynamic calibration model")
            self.calibration_model.fit(returns)
            self.optimal_window = self.calibration_model.get_optimal_window()
            logger.info(f"Optimal calibration window: {self.optimal_window} days")
        
        # Fit enhanced EVT model
        if self.use_enhanced_evt and self.evt_model is not None:
            logger.info("Fitting enhanced EVT model")
            self.evt_model.fit(returns)
        
        # Fit ensemble model
        if self.use_ensemble_models and self.ensemble_model is not None:
            logger.info("Fitting ensemble risk model")
            self.ensemble_model.fit(returns)
            self.model_weights = self.ensemble_model.get_weights(self.current_regime)
            logger.info(f"Ensemble model weights: {self.model_weights}")
        
        # Fit machine learning models
        if self.use_machine_learning and len(self.ml_models) > 0:
            logger.info("Fitting machine learning models")
            
            for horizon in self.time_horizons:
                for cl in self.confidence_levels:
                    logger.info(f"Fitting ML model for horizon={horizon}, confidence_level={cl}")
                    self.ml_models[horizon][cl].fit(returns, prices)
    
    def predict_var(self, 
                    returns: pd.DataFrame, 
                    weights: np.ndarray,
                    confidence_level: float = 0.95,
                    time_horizon: int = 1,
                    method: str = "ensemble",
                    prices: Optional[pd.DataFrame] = None) -> float:
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
        time_horizon : int, default=1
            Time horizon for VaR (in trading days)
        method : str, default="ensemble"
            Method to use for VaR prediction
            ("ensemble", "historical", "monte_carlo", "evt", "ml")
        prices : pd.DataFrame, optional
            Historical asset prices
            
        Returns:
        --------
        float
            VaR prediction
        """
        # Check if model is fitted
        if (self.use_regime_switching and self.regime_model is None) or \
           (self.use_dynamic_calibration and self.calibration_model is None) or \
           (self.use_enhanced_evt and self.evt_model is None) or \
           (self.use_ensemble_models and self.ensemble_model is None) or \
           (self.use_machine_learning and len(self.ml_models) == 0):
            raise ValueError("Model must be fitted before prediction")
        
        # Update current regime
        if self.use_regime_switching and self.regime_model is not None:
            self.current_regime = self.regime_model.predict_regime(returns)
            logger.info(f"Current regime: {self.current_regime}")
        
        # Update optimal window
        if self.use_dynamic_calibration and self.calibration_model is not None:
            self.optimal_window = self.calibration_model.update_window(returns)
            logger.info(f"Updated optimal window: {self.optimal_window} days")
            
            # Use optimal window for prediction
            if self.optimal_window is not None:
                returns_window = returns.iloc[-self.optimal_window:]
            else:
                returns_window = returns
        else:
            returns_window = returns
        
        # Make prediction based on method
        if method == "ensemble" and self.use_ensemble_models and self.ensemble_model is not None:
            # Update ensemble weights based on current regime
            self.model_weights = self.ensemble_model.get_weights(self.current_regime)
            logger.info(f"Updated ensemble weights: {self.model_weights}")
            
            # Predict VaR using ensemble model
            var = self.ensemble_model.predict_var(
                returns_window, 
                weights, 
                confidence_level, 
                time_horizon
            )
        elif method == "historical":
            # Predict VaR using historical simulation
            historical_model = HistoricalSimulation(
                confidence_level=confidence_level,
                time_horizon=time_horizon
            )
            historical_model.fit(returns_window)
            var = historical_model.predict_var(returns_window, weights)
        elif method == "monte_carlo":
            # Predict VaR using Monte Carlo simulation
            monte_carlo_model = MonteCarloSimulation(
                confidence_level=confidence_level,
                time_horizon=time_horizon,
                n_simulations=10000,
                random_state=self.random_state
            )
            monte_carlo_model.fit(returns_window)
            var = monte_carlo_model.predict_var(returns_window, weights)
        elif method == "evt" and self.use_enhanced_evt and self.evt_model is not None:
            # Predict VaR using enhanced EVT
            self.evt_model.confidence_level = confidence_level
            self.evt_model.time_horizon = time_horizon
            var = self.evt_model.predict_var(returns_window, weights)
        elif method == "ml" and self.use_machine_learning:
            # Check if ML model exists for the specified horizon and confidence level
            if time_horizon in self.ml_models and confidence_level in self.ml_models[time_horizon]:
                # Predict VaR using ML model
                var = self.ml_models[time_horizon][confidence_level].predict_var(
                    returns, 
                    weights, 
                    confidence_level, 
                    prices
                )
            else:
                # Use closest available model
                available_horizons = list(self.ml_models.keys())
                closest_horizon = min(available_horizons, key=lambda x: abs(x - time_horizon))
                
                available_cls = list(self.ml_models[closest_horizon].keys())
                closest_cl = min(available_cls, key=lambda x: abs(x - confidence_level))
                
                logger.warning(f"ML model for horizon={time_horizon}, confidence_level={confidence_level} not found")
                logger.warning(f"Using closest available model: horizon={closest_horizon}, confidence_level={closest_cl}")
                
                var = self.ml_models[closest_horizon][closest_cl].predict_var(
                    returns, 
                    weights, 
                    confidence_level, 
                    prices
                )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Scale VaR to time horizon if not already scaled
        if method != "ml":
            var = var * np.sqrt(time_horizon)
        
        return var
    
    def predict_es(self, 
                   returns: pd.DataFrame, 
                   weights: np.ndarray,
                   confidence_level: float = 0.95,
                   time_horizon: int = 1,
                   method: str = "ensemble",
                   prices: Optional[pd.DataFrame] = None) -> float:
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
        time_horizon : int, default=1
            Time horizon for ES (in trading days)
        method : str, default="ensemble"
            Method to use for ES prediction
            ("ensemble", "historical", "monte_carlo", "evt", "ml")
        prices : pd.DataFrame, optional
            Historical asset prices
            
        Returns:
        --------
        float
            ES prediction
        """
        # Check if model is fitted
        if (self.use_regime_switching and self.regime_model is None) or \
           (self.use_dynamic_calibration and self.calibration_model is None) or \
           (self.use_enhanced_evt and self.evt_model is None) or \
           (self.use_ensemble_models and self.ensemble_model is None) or \
           (self.use_machine_learning and len(self.ml_models) == 0):
            raise ValueError("Model must be fitted before prediction")
        
        # Update current regime
        if self.use_regime_switching and self.regime_model is not None:
            self.current_regime = self.regime_model.predict_regime(returns)
            logger.info(f"Current regime: {self.current_regime}")
        
        # Update optimal window
        if self.use_dynamic_calibration and self.calibration_model is not None:
            self.optimal_window = self.calibration_model.update_window(returns)
            logger.info(f"Updated optimal window: {self.optimal_window} days")
            
            # Use optimal window for prediction
            if self.optimal_window is not None:
                returns_window = returns.iloc[-self.optimal_window:]
            else:
                returns_window = returns
        else:
            returns_window = returns
        
        # Make prediction based on method
        if method == "ensemble" and self.use_ensemble_models and self.ensemble_model is not None:
            # Update ensemble weights based on current regime
            self.model_weights = self.ensemble_model.get_weights(self.current_regime)
            logger.info(f"Updated ensemble weights: {self.model_weights}")
            
            # Predict ES using ensemble model
            es = self.ensemble_model.predict_es(
                returns_window, 
                weights, 
                confidence_level, 
                time_horizon
            )
        elif method == "historical":
            # Predict ES using historical simulation
            historical_model = HistoricalSimulation(
                confidence_level=confidence_level,
                time_horizon=time_horizon
            )
            historical_model.fit(returns_window)
            es = historical_model.predict_es(returns_window, weights)
        elif method == "monte_carlo":
            # Predict ES using Monte Carlo simulation
            monte_carlo_model = MonteCarloSimulation(
                confidence_level=confidence_level,
                time_horizon=time_horizon,
                n_simulations=10000,
                random_state=self.random_state
            )
            monte_carlo_model.fit(returns_window)
            es = monte_carlo_model.predict_es(returns_window, weights)
        elif method == "evt" and self.use_enhanced_evt and self.evt_model is not None:
            # Predict ES using enhanced EVT
            self.evt_model.confidence_level = confidence_level
            self.evt_model.time_horizon = time_horizon
            es = self.evt_model.predict_es(returns_window, weights)
        elif method == "ml" and self.use_machine_learning:
            # Check if ML model exists for the specified horizon and confidence level
            if time_horizon in self.ml_models and confidence_level in self.ml_models[time_horizon]:
                # Predict ES using ML model
                es = self.ml_models[time_horizon][confidence_level].predict_es(
                    returns, 
                    weights, 
                    confidence_level, 
                    prices
                )
            else:
                # Use closest available model
                available_horizons = list(self.ml_models.keys())
                closest_horizon = min(available_horizons, key=lambda x: abs(x - time_horizon))
                
                available_cls = list(self.ml_models[closest_horizon].keys())
                closest_cl = min(available_cls, key=lambda x: abs(x - confidence_level))
                
                logger.warning(f"ML model for horizon={time_horizon}, confidence_level={confidence_level} not found")
                logger.warning(f"Using closest available model: horizon={closest_horizon}, confidence_level={closest_cl}")
                
                es = self.ml_models[closest_horizon][closest_cl].predict_es(
                    returns, 
                    weights, 
                    confidence_level, 
                    prices
                )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Scale ES to time horizon if not already scaled
        if method != "ml":
            es = es * np.sqrt(time_horizon)
        
        return es
    
    def run_stress_test(self, 
                        returns: pd.DataFrame, 
                        weights: np.ndarray,
                        scenario: str,
                        confidence_level: float = 0.95,
                        time_horizon: int = 21) -> Dict[str, float]:
        """
        Run stress test on portfolio.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        scenario : str
            Stress test scenario
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        time_horizon : int, default=21
            Time horizon for risk metrics (in trading days)
            
        Returns:
        --------
        Dict[str, float]
            Stress test results
        """
        # Create stress testing model
        stress_model = StressTesting(
            confidence_level=confidence_level,
            time_horizon=time_horizon
        )
        
        # Run stress test
        results = stress_model.run_stress_test(
            returns, 
            weights, 
            scenario
        )
        
        return results
    
    def analyze_portfolio(self, 
                          returns: pd.DataFrame, 
                          weights: np.ndarray,
                          confidence_levels: Optional[List[float]] = None,
                          time_horizons: Optional[List[int]] = None,
                          methods: Optional[List[str]] = None,
                          run_stress_tests: bool = True,
                          stress_scenarios: Optional[List[str]] = None,
                          prices: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Comprehensive portfolio risk analysis.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        confidence_levels : List[float], optional
            Confidence levels for risk metrics
        time_horizons : List[int], optional
            Time horizons for risk metrics (in trading days)
        methods : List[str], optional
            Methods to use for risk prediction
        run_stress_tests : bool, default=True
            Whether to run stress tests
        stress_scenarios : List[str], optional
            Stress test scenarios
        prices : pd.DataFrame, optional
            Historical asset prices
            
        Returns:
        --------
        Dict[str, Any]
            Portfolio analysis results
        """
        # Use default parameters if not specified
        if confidence_levels is None:
            confidence_levels = self.confidence_levels
        
        if time_horizons is None:
            time_horizons = self.time_horizons
        
        if methods is None:
            methods = ["ensemble"]
            
            if self.use_machine_learning:
                methods.append("ml")
        
        if stress_scenarios is None:
            stress_scenarios = [
                "financial_crisis_2008",
                "covid_crash_2020",
                "interest_rate_shock",
                "inflation_shock",
                "tariff_war"
            ]
        
        # Initialize results
        results = {
            "portfolio": {
                "assets": returns.columns.tolist(),
                "weights": weights.tolist(),
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "regime": {
                "current_regime": self.current_regime,
                "regime_probabilities": self.regime_model.get_regime_probabilities(returns) if self.regime_model is not None else None
            },
            "calibration": {
                "optimal_window": self.optimal_window
            },
            "risk_metrics": {
                "var": {},
                "es": {}
            },
            "stress_tests": {}
        }
        
        # Calculate portfolio statistics
        portfolio_returns = returns @ weights
        
        results["portfolio"]["statistics"] = {
            "mean": portfolio_returns.mean() * 252,  # Annualized mean
            "volatility": portfolio_returns.std() * np.sqrt(252),  # Annualized volatility
            "sharpe_ratio": (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252),  # Annualized Sharpe ratio
            "skewness": portfolio_returns.skew(),
            "kurtosis": portfolio_returns.kurt(),
            "max_drawdown": self._calculate_max_drawdown(portfolio_returns)
        }
        
        # Calculate VaR and ES for different confidence levels, time horizons, and methods
        for cl in confidence_levels:
            results["risk_metrics"]["var"][f"{cl}"] = {}
            results["risk_metrics"]["es"][f"{cl}"] = {}
            
            for horizon in time_horizons:
                results["risk_metrics"]["var"][f"{cl}"][f"{horizon}d"] = {}
                results["risk_metrics"]["es"][f"{cl}"][f"{horizon}d"] = {}
                
                for method in methods:
                    # Calculate VaR
                    var = self.predict_var(
                        returns, 
                        weights, 
                        cl, 
                        horizon, 
                        method, 
                        prices
                    )
                    
                    results["risk_metrics"]["var"][f"{cl}"][f"{horizon}d"][method] = var
                    
                    # Calculate ES
                    es = self.predict_es(
                        returns, 
                        weights, 
                        cl, 
                        horizon, 
                        method, 
                        prices
                    )
                    
                    results["risk_metrics"]["es"][f"{cl}"][f"{horizon}d"][method] = es
        
        # Run stress tests
        if run_stress_tests:
            for scenario in stress_scenarios:
                results["stress_tests"][scenario] = self.run_stress_test(
                    returns, 
                    weights, 
                    scenario, 
                    0.95, 
                    21
                )
        
        return results
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
            
        Returns:
        --------
        float
            Maximum drawdown
        """
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.cummax()
        
        # Calculate drawdown
        drawdown = (cum_returns / running_max) - 1
        
        # Calculate maximum drawdown
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """
        Save analysis results to file.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Analysis results
        filepath : str
            File path to save results
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert numpy arrays to lists
        results_json = self._convert_to_json_serializable(results)
        
        # Save results to file
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=4)
        
        logger.info(f"Results saved to {filepath}")
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON serializable format.
        
        Parameters:
        -----------
        obj : Any
            Object to convert
            
        Returns:
        --------
        Any
            JSON serializable object
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        else:
            return obj
    
    def plot_regime_probabilities(self, returns: pd.DataFrame) -> plt.Figure:
        """
        Plot regime probabilities.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with regime probabilities
        """
        if self.regime_model is None:
            raise ValueError("Regime model must be fitted before plotting")
        
        # Get regime probabilities
        regime_probs = self.regime_model.get_regime_probabilities(returns)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot regime probabilities
        for regime in regime_probs.columns:
            ax.plot(regime_probs.index, regime_probs[regime], label=regime)
        
        ax.set_title('Regime Probabilities')
        ax.set_xlabel('Date')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_var_comparison(self, 
                            returns: pd.DataFrame, 
                            weights: np.ndarray,
                            confidence_level: float = 0.95,
                            time_horizon: int = 21,
                            methods: Optional[List[str]] = None,
                            prices: Optional[pd.DataFrame] = None) -> plt.Figure:
        """
        Plot VaR comparison across different methods.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for VaR
        time_horizon : int, default=21
            Time horizon for VaR (in trading days)
        methods : List[str], optional
            Methods to use for VaR prediction
        prices : pd.DataFrame, optional
            Historical asset prices
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with VaR comparison
        """
        if methods is None:
            methods = ["ensemble", "historical", "monte_carlo", "evt"]
            
            if self.use_machine_learning:
                methods.append("ml")
        
        # Calculate VaR for different methods
        var_results = {}
        
        for method in methods:
            var = self.predict_var(
                returns, 
                weights, 
                confidence_level, 
                time_horizon, 
                method, 
                prices
            )
            
            var_results[method] = var
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot VaR comparison
        methods_list = list(var_results.keys())
        var_values = [var_results[method] for method in methods_list]
        
        ax.bar(methods_list, var_values)
        
        ax.set_title(f'VaR Comparison ({confidence_level*100}%, {time_horizon}-day)')
        ax.set_xlabel('Method')
        ax.set_ylabel('VaR')
        ax.grid(True, axis='y')
        
        # Add values on top of bars
        for i, v in enumerate(var_values):
            ax.text(i, v + 0.001, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        return fig
    
    def plot_stress_test_results(self, 
                                 returns: pd.DataFrame, 
                                 weights: np.ndarray,
                                 scenarios: Optional[List[str]] = None,
                                 confidence_level: float = 0.95,
                                 time_horizon: int = 21) -> plt.Figure:
        """
        Plot stress test results.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        weights : np.ndarray
            Portfolio weights
        scenarios : List[str], optional
            Stress test scenarios
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        time_horizon : int, default=21
            Time horizon for risk metrics (in trading days)
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with stress test results
        """
        if scenarios is None:
            scenarios = [
                "financial_crisis_2008",
                "covid_crash_2020",
                "interest_rate_shock",
                "inflation_shock",
                "tariff_war"
            ]
        
        # Run stress tests
        stress_results = {}
        
        for scenario in scenarios:
            stress_results[scenario] = self.run_stress_test(
                returns, 
                weights, 
                scenario, 
                confidence_level, 
                time_horizon
            )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot stress test results
        scenarios_list = list(stress_results.keys())
        metrics = ["portfolio_return", "var", "es", "max_drawdown"]
        
        x = np.arange(len(scenarios_list))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [stress_results[scenario][metric] for scenario in scenarios_list]
            ax.bar(x + i*width - width*1.5, values, width, label=metric)
        
        ax.set_title('Stress Test Results')
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios_list, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y')
        
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
    
    # Create enhanced risk modeling framework
    framework = EnhancedRiskModelingFramework(
        name="EnhancedRiskModel",
        use_regime_switching=True,
        use_dynamic_calibration=True,
        use_enhanced_evt=True,
        use_ensemble_models=True,
        use_machine_learning=True,
        confidence_levels=[0.95, 0.99],
        time_horizons=[1, 5, 10, 21],
        base_currency="USD",
        random_state=42
    )
    
    # Fit framework
    framework.fit(returns, prices)
    
    # Equal weights for portfolio
    weights = np.ones(len(tickers)) / len(tickers)
    
    # Analyze portfolio
    results = framework.analyze_portfolio(
        returns, 
        weights,
        confidence_levels=[0.95, 0.99],
        time_horizons=[1, 5, 10, 21],
        methods=["ensemble", "historical", "monte_carlo", "evt", "ml"],
        run_stress_tests=True,
        stress_scenarios=[
            "financial_crisis_2008",
            "covid_crash_2020",
            "interest_rate_shock",
            "inflation_shock",
            "tariff_war"
        ],
        prices=prices
    )
    
    # Save results
    framework.save_results(results, "results/portfolio_analysis.json")
    
    # Plot regime probabilities
    fig_regime = framework.plot_regime_probabilities(returns)
    fig_regime.savefig("figures/regime_probabilities.png")
    
    # Plot VaR comparison
    fig_var = framework.plot_var_comparison(
        returns, 
        weights,
        confidence_level=0.95,
        time_horizon=21,
        methods=["ensemble", "historical", "monte_carlo", "evt", "ml"],
        prices=prices
    )
    fig_var.savefig("figures/var_comparison.png")
    
    # Plot stress test results
    fig_stress = framework.plot_stress_test_results(
        returns, 
        weights,
        scenarios=[
            "financial_crisis_2008",
            "covid_crash_2020",
            "interest_rate_shock",
            "inflation_shock",
            "tariff_war"
        ],
        confidence_level=0.95,
        time_horizon=21
    )
    fig_stress.savefig("figures/stress_test_results.png")
