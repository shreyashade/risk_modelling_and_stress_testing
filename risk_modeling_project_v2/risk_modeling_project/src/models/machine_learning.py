"""
Machine Learning Risk Models

This module implements machine learning approaches for risk modeling,
including neural networks, gradient boosting, and other ML techniques
to enhance prediction accuracy.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Union, Optional, Tuple, Callable, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Feature engineering for risk modeling.
    
    This class implements various feature engineering techniques to extract
    relevant information from financial time series data.
    """
    
    def __init__(self, 
                 include_technical: bool = True,
                 include_statistical: bool = True,
                 include_volatility: bool = True,
                 include_correlation: bool = True,
                 include_tail: bool = True,
                 lookback_periods: List[int] = [5, 21, 63, 126, 252]):
        """
        Initialize the feature engineering.
        
        Parameters:
        -----------
        include_technical : bool, default=True
            Whether to include technical indicators
        include_statistical : bool, default=True
            Whether to include statistical features
        include_volatility : bool, default=True
            Whether to include volatility features
        include_correlation : bool, default=True
            Whether to include correlation features
        include_tail : bool, default=True
            Whether to include tail risk features
        lookback_periods : List[int], default=[5, 21, 63, 126, 252]
            Lookback periods for feature calculation (trading days)
        """
        self.include_technical = include_technical
        self.include_statistical = include_statistical
        self.include_volatility = include_volatility
        self.include_correlation = include_correlation
        self.include_tail = include_tail
        self.lookback_periods = lookback_periods
    
    def create_features(self, returns: pd.DataFrame, prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create features from returns and prices.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        prices : pd.DataFrame, optional
            Historical asset prices
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with engineered features
        """
        logger.info("Creating features")
        
        # Initialize features DataFrame
        features = pd.DataFrame(index=returns.index)
        
        # Calculate portfolio returns if multiple assets
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = pd.Series(returns.values @ weights, index=returns.index)
            features['portfolio_return'] = portfolio_returns
        else:
            portfolio_returns = returns.iloc[:, 0]
            features['portfolio_return'] = portfolio_returns
        
        # Technical indicators
        if self.include_technical and prices is not None:
            features = self._add_technical_indicators(features, prices)
        
        # Statistical features
        if self.include_statistical:
            features = self._add_statistical_features(features, returns, portfolio_returns)
        
        # Volatility features
        if self.include_volatility:
            features = self._add_volatility_features(features, returns, portfolio_returns)
        
        # Correlation features
        if self.include_correlation and returns.shape[1] > 1:
            features = self._add_correlation_features(features, returns)
        
        # Tail risk features
        if self.include_tail:
            features = self._add_tail_features(features, returns, portfolio_returns)
        
        # Drop NaN values
        features = features.dropna()
        
        logger.info(f"Created {features.shape[1]} features")
        
        return features
    
    def _add_technical_indicators(self, features: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to features.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Features DataFrame
        prices : pd.DataFrame
            Historical asset prices
            
        Returns:
        --------
        pd.DataFrame
            Updated features DataFrame
        """
        logger.info("Adding technical indicators")
        
        # Use portfolio prices if multiple assets
        if prices.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(prices.shape[1]) / prices.shape[1]
            portfolio_prices = pd.Series(prices.values @ weights, index=prices.index)
        else:
            portfolio_prices = prices.iloc[:, 0]
        
        # Moving averages
        for period in self.lookback_periods:
            features[f'ma_{period}'] = portfolio_prices.rolling(window=period).mean()
            
            # Moving average ratio
            features[f'ma_ratio_{period}'] = portfolio_prices / features[f'ma_{period}']
        
        # Exponential moving averages
        for period in self.lookback_periods:
            features[f'ema_{period}'] = portfolio_prices.ewm(span=period).mean()
            
            # EMA ratio
            features[f'ema_ratio_{period}'] = portfolio_prices / features[f'ema_{period}']
        
        # MACD
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # RSI
        delta = portfolio_prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for period in [20]:
            ma = portfolio_prices.rolling(window=period).mean()
            std = portfolio_prices.rolling(window=period).std()
            
            features[f'bb_upper_{period}'] = ma + 2 * std
            features[f'bb_lower_{period}'] = ma - 2 * std
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / ma
            features[f'bb_pct_{period}'] = (portfolio_prices - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
        
        return features
    
    def _add_statistical_features(self, features: pd.DataFrame, returns: pd.DataFrame, portfolio_returns: pd.Series) -> pd.DataFrame:
        """
        Add statistical features to features.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Features DataFrame
        returns : pd.DataFrame
            Historical asset returns
        portfolio_returns : pd.Series
            Portfolio returns
            
        Returns:
        --------
        pd.DataFrame
            Updated features DataFrame
        """
        logger.info("Adding statistical features")
        
        # Rolling mean
        for period in self.lookback_periods:
            features[f'mean_{period}'] = portfolio_returns.rolling(window=period).mean()
        
        # Rolling median
        for period in self.lookback_periods:
            features[f'median_{period}'] = portfolio_returns.rolling(window=period).median()
        
        # Rolling skewness
        for period in self.lookback_periods:
            features[f'skew_{period}'] = portfolio_returns.rolling(window=period).skew()
        
        # Rolling kurtosis
        for period in self.lookback_periods:
            features[f'kurt_{period}'] = portfolio_returns.rolling(window=period).kurt()
        
        # Rolling quantiles
        for period in self.lookback_periods:
            features[f'q05_{period}'] = portfolio_returns.rolling(window=period).quantile(0.05)
            features[f'q95_{period}'] = portfolio_returns.rolling(window=period).quantile(0.95)
        
        # Rolling autocorrelation
        for period in self.lookback_periods:
            features[f'autocorr_{period}'] = portfolio_returns.rolling(window=period).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan
            )
        
        # Cumulative returns
        for period in self.lookback_periods:
            features[f'cum_return_{period}'] = (1 + portfolio_returns).rolling(window=period).apply(
                lambda x: np.prod(x) - 1
            )
        
        return features
    
    def _add_volatility_features(self, features: pd.DataFrame, returns: pd.DataFrame, portfolio_returns: pd.Series) -> pd.DataFrame:
        """
        Add volatility features to features.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Features DataFrame
        returns : pd.DataFrame
            Historical asset returns
        portfolio_returns : pd.Series
            Portfolio returns
            
        Returns:
        --------
        pd.DataFrame
            Updated features DataFrame
        """
        logger.info("Adding volatility features")
        
        # Rolling volatility
        for period in self.lookback_periods:
            features[f'vol_{period}'] = portfolio_returns.rolling(window=period).std() * np.sqrt(252)
        
        # Exponentially weighted volatility
        for period in self.lookback_periods:
            features[f'ewm_vol_{period}'] = portfolio_returns.ewm(span=period).std() * np.sqrt(252)
        
        # Volatility of volatility
        for period in self.lookback_periods:
            vol_series = portfolio_returns.rolling(window=period).std() * np.sqrt(252)
            features[f'vol_of_vol_{period}'] = vol_series.rolling(window=period).std()
        
        # Parkinson volatility (high-low range based)
        if 'high' in returns.columns and 'low' in returns.columns:
            for period in self.lookback_periods:
                high_low_ratio = np.log(returns['high'] / returns['low'])
                features[f'parkinson_vol_{period}'] = np.sqrt(
                    (1 / (4 * np.log(2))) * high_low_ratio.rolling(window=period).apply(
                        lambda x: np.sum(x**2) / len(x)
                    )
                ) * np.sqrt(252)
        
        # GARCH volatility
        try:
            from arch import arch_model
            
            # Fit GARCH model on full sample
            garch_model = arch_model(
                portfolio_returns.dropna() * 100,  # Scale for numerical stability
                vol='GARCH',
                p=1,
                q=1,
                rescale=False
            )
            
            garch_fit = garch_model.fit(disp='off')
            
            # Get conditional volatility
            features['garch_vol'] = garch_fit.conditional_volatility / 100 * np.sqrt(252)
        except:
            logger.warning("Failed to compute GARCH volatility")
        
        # Volatility regime
        for period in self.lookback_periods:
            vol = features[f'vol_{period}']
            vol_mean = vol.rolling(window=252).mean()
            vol_std = vol.rolling(window=252).std()
            
            features[f'vol_regime_{period}'] = (vol - vol_mean) / vol_std
        
        return features
    
    def _add_correlation_features(self, features: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Add correlation features to features.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Features DataFrame
        returns : pd.DataFrame
            Historical asset returns
            
        Returns:
        --------
        pd.DataFrame
            Updated features DataFrame
        """
        logger.info("Adding correlation features")
        
        # Rolling correlation
        for period in self.lookback_periods:
            # Average pairwise correlation
            corr_sum = 0
            count = 0
            
            for i in range(returns.shape[1]):
                for j in range(i+1, returns.shape[1]):
                    corr = returns.iloc[:, i].rolling(window=period).corr(returns.iloc[:, j])
                    corr_sum += corr
                    count += 1
            
            features[f'avg_corr_{period}'] = corr_sum / count
        
        # Correlation regime
        for period in self.lookback_periods:
            corr = features[f'avg_corr_{period}']
            corr_mean = corr.rolling(window=252).mean()
            corr_std = corr.rolling(window=252).std()
            
            features[f'corr_regime_{period}'] = (corr - corr_mean) / corr_std
        
        # Principal component analysis
        for period in self.lookback_periods:
            try:
                # Rolling PCA
                def rolling_pca_var(x):
                    if len(x) < min(returns.shape[1], period):
                        return np.nan
                    
                    pca = PCA()
                    pca.fit(x)
                    
                    # Variance explained by first component
                    return pca.explained_variance_ratio_[0]
                
                features[f'pca_var_{period}'] = returns.rolling(window=period).apply(
                    rolling_pca_var,
                    raw=False
                )
            except:
                logger.warning(f"Failed to compute PCA for period {period}")
        
        return features
    
    def _add_tail_features(self, features: pd.DataFrame, returns: pd.DataFrame, portfolio_returns: pd.Series) -> pd.DataFrame:
        """
        Add tail risk features to features.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Features DataFrame
        returns : pd.DataFrame
            Historical asset returns
        portfolio_returns : pd.Series
            Portfolio returns
            
        Returns:
        --------
        pd.DataFrame
            Updated features DataFrame
        """
        logger.info("Adding tail risk features")
        
        # Value at Risk
        for period in self.lookback_periods:
            features[f'var95_{period}'] = portfolio_returns.rolling(window=period).quantile(0.05)
            features[f'var99_{period}'] = portfolio_returns.rolling(window=period).quantile(0.01)
        
        # Expected Shortfall
        for period in self.lookback_periods:
            def rolling_es(x, alpha=0.05):
                if len(x) < 10:  # Minimum sample size
                    return np.nan
                
                threshold = np.percentile(x, alpha * 100)
                return np.mean(x[x <= threshold])
            
            features[f'es95_{period}'] = portfolio_returns.rolling(window=period).apply(
                lambda x: rolling_es(x, alpha=0.05)
            )
            
            features[f'es99_{period}'] = portfolio_returns.rolling(window=period).apply(
                lambda x: rolling_es(x, alpha=0.01)
            )
        
        # Hill estimator for tail index
        for period in self.lookback_periods:
            def rolling_hill(x, k_ratio=0.1):
                if len(x) < 20:  # Minimum sample size
                    return np.nan
                
                # Use absolute returns for tail index
                abs_returns = np.abs(x)
                sorted_returns = np.sort(abs_returns)[::-1]  # Sort in descending order
                
                # Use top k_ratio percent of observations
                k = int(len(sorted_returns) * k_ratio)
                k = max(5, min(k, len(sorted_returns) // 2))  # Ensure reasonable k
                
                # Calculate Hill estimator
                log_returns = np.log(sorted_returns[:k])
                log_k_return = np.log(sorted_returns[k-1])
                
                hill = (1 / k) * np.sum(log_returns - log_k_return)
                
                return 1 / hill if hill > 0 else np.nan
            
            features[f'tail_index_{period}'] = portfolio_returns.rolling(window=period).apply(
                rolling_hill
            )
        
        # Maximum drawdown
        for period in self.lookback_periods:
            def rolling_max_drawdown(x):
                if len(x) < 2:
                    return np.nan
                
                # Calculate cumulative returns
                cum_returns = (1 + x).cumprod()
                
                # Calculate running maximum
                running_max = np.maximum.accumulate(cum_returns)
                
                # Calculate drawdown
                drawdown = (cum_returns / running_max) - 1
                
                return np.min(drawdown)
            
            features[f'max_drawdown_{period}'] = portfolio_returns.rolling(window=period).apply(
                rolling_max_drawdown
            )
        
        return features


class MLRiskModel:
    """
    Machine Learning risk model base class.
    
    This class implements the base functionality for ML-based risk models.
    """
    
    def __init__(self, 
                 name: str,
                 target: str = 'var',
                 confidence_level: float = 0.95,
                 forecast_horizon: int = 21,
                 feature_engineering: Optional[FeatureEngineering] = None,
                 scaler_type: str = 'standard',
                 feature_selection: bool = False,
                 n_features: int = 20,
                 use_pca: bool = False,
                 n_components: int = 10):
        """
        Initialize the ML risk model.
        
        Parameters:
        -----------
        name : str
            Name of the risk model
        target : str, default='var'
            Target variable ('var' or 'es')
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        forecast_horizon : int, default=21
            Forecast horizon for risk metrics (trading days)
        feature_engineering : FeatureEngineering, optional
            Feature engineering instance
        scaler_type : str, default='standard'
            Scaler type ('standard' or 'minmax')
        feature_selection : bool, default=False
            Whether to use feature selection
        n_features : int, default=20
            Number of features to select
        use_pca : bool, default=False
            Whether to use PCA
        n_components : int, default=10
            Number of PCA components
        """
        self.name = name
        self.target = target
        self.confidence_level = confidence_level
        self.forecast_horizon = forecast_horizon
        self.feature_engineering = feature_engineering or FeatureEngineering()
        self.scaler_type = scaler_type
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.use_pca = use_pca
        self.n_components = n_components
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.feature_names = None
        self.fitted = False
    
    def _prepare_features(self, returns: pd.DataFrame, prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare features for the model.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        prices : pd.DataFrame, optional
            Historical asset prices
            
        Returns:
        --------
        pd.DataFrame
            Prepared features
        """
        # Create features
        features = self.feature_engineering.create_features(returns, prices)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        return features
    
    def _prepare_target(self, returns: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Prepare target variable for the model.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        features : pd.DataFrame
            Features DataFrame
            
        Returns:
        --------
        pd.Series
            Target variable
        """
        # Calculate portfolio returns if multiple assets
        if returns.shape[1] > 1:
            # Equal-weighted portfolio for simplicity
            weights = np.ones(returns.shape[1]) / returns.shape[1]
            portfolio_returns = pd.Series(returns.values @ weights, index=returns.index)
        else:
            portfolio_returns = returns.iloc[:, 0]
        
        # Calculate forward returns
        forward_returns = pd.Series(index=portfolio_returns.index)
        
        for i in range(len(portfolio_returns) - self.forecast_horizon):
            # Calculate cumulative return over forecast horizon
            forward_return = (1 + portfolio_returns.iloc[i+1:i+1+self.forecast_horizon]).prod() - 1
            forward_returns.iloc[i] = forward_return
        
        # Calculate target variable
        if self.target == 'var':
            # Calculate rolling VaR on forward returns
            target = pd.Series(index=forward_returns.index)
            
            for i in range(len(forward_returns) - 252):
                # Use 1-year rolling window
                var = np.percentile(forward_returns.iloc[i:i+252], (1 - self.confidence_level) * 100)
                target.iloc[i] = var
        elif self.target == 'es':
            # Calculate rolling ES on forward returns
            target = pd.Series(index=forward_returns.index)
            
            for i in range(len(forward_returns) - 252):
                # Use 1-year rolling window
                var_threshold = np.percentile(forward_returns.iloc[i:i+252], (1 - self.confidence_level) * 100)
                tail = forward_returns.iloc[i:i+252][forward_returns.iloc[i:i+252] <= var_threshold]
                es = np.mean(tail)
                target.iloc[i] = es
        else:
            raise ValueError(f"Unknown target: {self.target}")
        
        # Align target with features
        target = target.loc[features.index]
        
        return target
    
    def _preprocess_data(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess data for the model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features DataFrame
            
        Returns:
        --------
        np.ndarray
            Preprocessed features
        """
        # Convert to numpy array
        X_np = X.values
        
        # Scale features
        if self.scaler is not None:
            X_np = self.scaler.transform(X_np)
        
        # Feature selection
        if self.feature_selector is not None:
            X_np = self.feature_selector.transform(X_np)
        
        # PCA
        if self.pca is not None:
            X_np = self.pca.transform(X_np)
        
        return X_np
    
    def fit(self, returns: pd.DataFrame, prices: Optional[pd.DataFrame] = None) -> None:
        """
        Fit the ML risk model to historical data.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        prices : pd.DataFrame, optional
            Historical asset prices
        """
        logger.info(f"Fitting {self.name} risk model")
        
        # Prepare features
        features = self._prepare_features(returns, prices)
        
        # Prepare target
        target = self._prepare_target(returns, features)
        
        # Align data
        data = pd.concat([features, target], axis=1)
        data = data.dropna()
        
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        # Initialize scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        # Fit scaler
        self.scaler.fit(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Feature selection
        if self.feature_selection:
            self.feature_selector = SelectKBest(f_regression, k=min(self.n_features, X.shape[1]))
            self.feature_selector.fit(X_scaled, y)
            
            # Get selected feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            selected_features = [X.columns[i] for i in selected_indices]
            
            logger.info(f"Selected features: {selected_features}")
            
            # Transform features
            X_scaled = self.feature_selector.transform(X_scaled)
        
        # PCA
        if self.use_pca:
            self.pca = PCA(n_components=min(self.n_components, X_scaled.shape[1]))
            self.pca.fit(X_scaled)
            
            # Log explained variance
            explained_variance = self.pca.explained_variance_ratio_
            logger.info(f"PCA explained variance: {explained_variance}")
            
            # Transform features
            X_scaled = self.pca.transform(X_scaled)
        
        # Fit model
        self._fit_model(X_scaled, y)
        
        self.fitted = True
        logger.info(f"{self.name} risk model fitted successfully")
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        X : np.ndarray
            Features array
        y : np.ndarray
            Target array
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict(self, returns: pd.DataFrame, prices: Optional[pd.DataFrame] = None) -> float:
        """
        Predict risk metric.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        prices : pd.DataFrame, optional
            Historical asset prices
            
        Returns:
        --------
        float
            Risk metric prediction
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        features = self._prepare_features(returns, prices)
        
        # Ensure all feature columns are present
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = np.nan
        
        # Reorder columns to match training data
        features = features[self.feature_names]
        
        # Use only the most recent data point
        recent_features = features.iloc[-1:].copy()
        
        # Fill missing values with column means
        recent_features = recent_features.fillna(features.mean())
        
        # Preprocess data
        X = self._preprocess_data(recent_features)
        
        # Make prediction
        prediction = self._predict(X)
        
        return prediction
    
    def _predict(self, X: np.ndarray) -> float:
        """
        Make prediction with the model.
        
        Parameters:
        -----------
        X : np.ndarray
            Features array
            
        Returns:
        --------
        float
            Prediction
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict_var(self, 
                    returns: pd.DataFrame, 
                    weights: np.ndarray,
                    confidence_level: float = 0.95,
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
        prices : pd.DataFrame, optional
            Historical asset prices
            
        Returns:
        --------
        float
            VaR prediction
        """
        if self.target != 'var':
            logger.warning(f"Model was trained to predict {self.target}, not VaR")
        
        # Adjust confidence level if different from training
        if confidence_level != self.confidence_level:
            logger.warning(f"Using different confidence level ({confidence_level}) than training ({self.confidence_level})")
        
        # Make prediction
        prediction = self.predict(returns, prices)
        
        # Ensure prediction is negative (VaR is typically reported as a positive number)
        if prediction > 0:
            prediction = -prediction
        
        return prediction
    
    def predict_es(self, 
                   returns: pd.DataFrame, 
                   weights: np.ndarray,
                   confidence_level: float = 0.95,
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
        prices : pd.DataFrame, optional
            Historical asset prices
            
        Returns:
        --------
        float
            ES prediction
        """
        if self.target != 'es':
            logger.warning(f"Model was trained to predict {self.target}, not ES")
        
        # Adjust confidence level if different from training
        if confidence_level != self.confidence_level:
            logger.warning(f"Using different confidence level ({confidence_level}) than training ({self.confidence_level})")
        
        # Make prediction
        prediction = self.predict(returns, prices)
        
        # Ensure prediction is negative (ES is typically reported as a positive number)
        if prediction > 0:
            prediction = -prediction
        
        return prediction
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance.
        
        Returns:
        --------
        pd.Series, optional
            Feature importance
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Check if model has feature importance
        if hasattr(self.model, 'feature_importances_'):
            # Get feature names
            if self.feature_selector is not None:
                # Get selected feature indices
                selected_indices = self.feature_selector.get_support(indices=True)
                feature_names = [self.feature_names[i] for i in selected_indices]
            else:
                feature_names = self.feature_names
            
            # Return feature importance
            return pd.Series(self.model.feature_importances_, index=feature_names)
        elif hasattr(self.model, 'coef_'):
            # Get feature names
            if self.feature_selector is not None:
                # Get selected feature indices
                selected_indices = self.feature_selector.get_support(indices=True)
                feature_names = [self.feature_names[i] for i in selected_indices]
            else:
                feature_names = self.feature_names
            
            # Return feature importance
            return pd.Series(np.abs(self.model.coef_), index=feature_names)
        else:
            return None
    
    def plot_feature_importance(self, top_n: int = 20) -> Optional[plt.Figure]:
        """
        Plot feature importance.
        
        Parameters:
        -----------
        top_n : int, default=20
            Number of top features to plot
            
        Returns:
        --------
        plt.Figure, optional
            Matplotlib figure with feature importance
        """
        # Get feature importance
        importance = self.get_feature_importance()
        
        if importance is None:
            logger.warning("Model does not have feature importance")
            return None
        
        # Sort importance
        importance = importance.sort_values(ascending=False)
        
        # Select top features
        importance = importance.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot importance
        importance.plot(kind='barh', ax=ax)
        
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.set_xlabel('Importance')
        ax.grid(True, axis='x')
        
        plt.tight_layout()
        return fig


class RandomForestRiskModel(MLRiskModel):
    """
    Random Forest risk model.
    
    This class implements a Random Forest model for risk prediction.
    """
    
    def __init__(self, 
                 name: str = "RandomForest",
                 target: str = 'var',
                 confidence_level: float = 0.95,
                 forecast_horizon: int = 21,
                 feature_engineering: Optional[FeatureEngineering] = None,
                 scaler_type: str = 'standard',
                 feature_selection: bool = False,
                 n_features: int = 20,
                 use_pca: bool = False,
                 n_components: int = 10,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[str] = 'sqrt',
                 random_state: Optional[int] = 42):
        """
        Initialize the Random Forest risk model.
        
        Parameters:
        -----------
        name : str, default="RandomForest"
            Name of the risk model
        target : str, default='var'
            Target variable ('var' or 'es')
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        forecast_horizon : int, default=21
            Forecast horizon for risk metrics (trading days)
        feature_engineering : FeatureEngineering, optional
            Feature engineering instance
        scaler_type : str, default='standard'
            Scaler type ('standard' or 'minmax')
        feature_selection : bool, default=False
            Whether to use feature selection
        n_features : int, default=20
            Number of features to select
        use_pca : bool, default=False
            Whether to use PCA
        n_components : int, default=10
            Number of PCA components
        n_estimators : int, default=100
            Number of trees in the forest
        max_depth : int, optional
            Maximum depth of the trees
        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node
        min_samples_leaf : int, default=1
            Minimum number of samples required to be at a leaf node
        max_features : str, optional
            Number of features to consider for the best split
        random_state : int, optional
            Random state for reproducibility
        """
        super().__init__(
            name=name,
            target=target,
            confidence_level=confidence_level,
            forecast_horizon=forecast_horizon,
            feature_engineering=feature_engineering,
            scaler_type=scaler_type,
            feature_selection=feature_selection,
            n_features=n_features,
            use_pca=use_pca,
            n_components=n_components
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Random Forest model to the data.
        
        Parameters:
        -----------
        X : np.ndarray
            Features array
        y : np.ndarray
            Target array
        """
        # Initialize model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state
        )
        
        # Fit model
        self.model.fit(X, y)
        
        # Log feature importance
        if hasattr(self.model, 'feature_importances_'):
            logger.info("Top 5 feature importance:")
            
            # Get feature names
            if self.feature_selector is not None:
                # Get selected feature indices
                selected_indices = self.feature_selector.get_support(indices=True)
                feature_names = [self.feature_names[i] for i in selected_indices]
            else:
                feature_names = self.feature_names
            
            # Sort feature importance
            importance = pd.Series(self.model.feature_importances_, index=feature_names)
            importance = importance.sort_values(ascending=False)
            
            # Log top 5 features
            for name, imp in importance.head(5).items():
                logger.info(f"{name}: {imp:.4f}")
    
    def _predict(self, X: np.ndarray) -> float:
        """
        Make prediction with the Random Forest model.
        
        Parameters:
        -----------
        X : np.ndarray
            Features array
            
        Returns:
        --------
        float
            Prediction
        """
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        return prediction


class GradientBoostingRiskModel(MLRiskModel):
    """
    Gradient Boosting risk model.
    
    This class implements a Gradient Boosting model for risk prediction.
    """
    
    def __init__(self, 
                 name: str = "GradientBoosting",
                 target: str = 'var',
                 confidence_level: float = 0.95,
                 forecast_horizon: int = 21,
                 feature_engineering: Optional[FeatureEngineering] = None,
                 scaler_type: str = 'standard',
                 feature_selection: bool = False,
                 n_features: int = 20,
                 use_pca: bool = False,
                 n_components: int = 10,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 subsample: float = 1.0,
                 random_state: Optional[int] = 42):
        """
        Initialize the Gradient Boosting risk model.
        
        Parameters:
        -----------
        name : str, default="GradientBoosting"
            Name of the risk model
        target : str, default='var'
            Target variable ('var' or 'es')
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        forecast_horizon : int, default=21
            Forecast horizon for risk metrics (trading days)
        feature_engineering : FeatureEngineering, optional
            Feature engineering instance
        scaler_type : str, default='standard'
            Scaler type ('standard' or 'minmax')
        feature_selection : bool, default=False
            Whether to use feature selection
        n_features : int, default=20
            Number of features to select
        use_pca : bool, default=False
            Whether to use PCA
        n_components : int, default=10
            Number of PCA components
        n_estimators : int, default=100
            Number of boosting stages
        learning_rate : float, default=0.1
            Learning rate
        max_depth : int, default=3
            Maximum depth of the trees
        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node
        min_samples_leaf : int, default=1
            Minimum number of samples required to be at a leaf node
        subsample : float, default=1.0
            Fraction of samples to be used for fitting the individual trees
        random_state : int, optional
            Random state for reproducibility
        """
        super().__init__(
            name=name,
            target=target,
            confidence_level=confidence_level,
            forecast_horizon=forecast_horizon,
            feature_engineering=feature_engineering,
            scaler_type=scaler_type,
            feature_selection=feature_selection,
            n_features=n_features,
            use_pca=use_pca,
            n_components=n_components
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Gradient Boosting model to the data.
        
        Parameters:
        -----------
        X : np.ndarray
            Features array
        y : np.ndarray
            Target array
        """
        # Initialize model
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state
        )
        
        # Fit model
        self.model.fit(X, y)
        
        # Log feature importance
        if hasattr(self.model, 'feature_importances_'):
            logger.info("Top 5 feature importance:")
            
            # Get feature names
            if self.feature_selector is not None:
                # Get selected feature indices
                selected_indices = self.feature_selector.get_support(indices=True)
                feature_names = [self.feature_names[i] for i in selected_indices]
            else:
                feature_names = self.feature_names
            
            # Sort feature importance
            importance = pd.Series(self.model.feature_importances_, index=feature_names)
            importance = importance.sort_values(ascending=False)
            
            # Log top 5 features
            for name, imp in importance.head(5).items():
                logger.info(f"{name}: {imp:.4f}")
    
    def _predict(self, X: np.ndarray) -> float:
        """
        Make prediction with the Gradient Boosting model.
        
        Parameters:
        -----------
        X : np.ndarray
            Features array
            
        Returns:
        --------
        float
            Prediction
        """
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        return prediction


class SVRRiskModel(MLRiskModel):
    """
    Support Vector Regression risk model.
    
    This class implements a Support Vector Regression model for risk prediction.
    """
    
    def __init__(self, 
                 name: str = "SVR",
                 target: str = 'var',
                 confidence_level: float = 0.95,
                 forecast_horizon: int = 21,
                 feature_engineering: Optional[FeatureEngineering] = None,
                 scaler_type: str = 'standard',
                 feature_selection: bool = True,
                 n_features: int = 20,
                 use_pca: bool = False,
                 n_components: int = 10,
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 gamma: Optional[str] = 'scale',
                 random_state: Optional[int] = 42):
        """
        Initialize the SVR risk model.
        
        Parameters:
        -----------
        name : str, default="SVR"
            Name of the risk model
        target : str, default='var'
            Target variable ('var' or 'es')
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        forecast_horizon : int, default=21
            Forecast horizon for risk metrics (trading days)
        feature_engineering : FeatureEngineering, optional
            Feature engineering instance
        scaler_type : str, default='standard'
            Scaler type ('standard' or 'minmax')
        feature_selection : bool, default=True
            Whether to use feature selection
        n_features : int, default=20
            Number of features to select
        use_pca : bool, default=False
            Whether to use PCA
        n_components : int, default=10
            Number of PCA components
        kernel : str, default='rbf'
            Kernel type
        C : float, default=1.0
            Regularization parameter
        epsilon : float, default=0.1
            Epsilon in the epsilon-SVR model
        gamma : str, optional
            Kernel coefficient
        random_state : int, optional
            Random state for reproducibility
        """
        super().__init__(
            name=name,
            target=target,
            confidence_level=confidence_level,
            forecast_horizon=forecast_horizon,
            feature_engineering=feature_engineering,
            scaler_type=scaler_type,
            feature_selection=feature_selection,
            n_features=n_features,
            use_pca=use_pca,
            n_components=n_components
        )
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.random_state = random_state
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the SVR model to the data.
        
        Parameters:
        -----------
        X : np.ndarray
            Features array
        y : np.ndarray
            Target array
        """
        # Initialize model
        self.model = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma
        )
        
        # Fit model
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray) -> float:
        """
        Make prediction with the SVR model.
        
        Parameters:
        -----------
        X : np.ndarray
            Features array
            
        Returns:
        --------
        float
            Prediction
        """
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        return prediction


class NeuralNetworkRiskModel(MLRiskModel):
    """
    Neural Network risk model.
    
    This class implements a Neural Network model for risk prediction.
    """
    
    def __init__(self, 
                 name: str = "NeuralNetwork",
                 target: str = 'var',
                 confidence_level: float = 0.95,
                 forecast_horizon: int = 21,
                 feature_engineering: Optional[FeatureEngineering] = None,
                 scaler_type: str = 'standard',
                 feature_selection: bool = True,
                 n_features: int = 20,
                 use_pca: bool = False,
                 n_components: int = 10,
                 hidden_layers: List[int] = [64, 32],
                 activation: str = 'relu',
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 early_stopping: bool = True,
                 patience: int = 10,
                 dropout_rate: float = 0.2,
                 random_state: Optional[int] = 42):
        """
        Initialize the Neural Network risk model.
        
        Parameters:
        -----------
        name : str, default="NeuralNetwork"
            Name of the risk model
        target : str, default='var'
            Target variable ('var' or 'es')
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        forecast_horizon : int, default=21
            Forecast horizon for risk metrics (trading days)
        feature_engineering : FeatureEngineering, optional
            Feature engineering instance
        scaler_type : str, default='standard'
            Scaler type ('standard' or 'minmax')
        feature_selection : bool, default=True
            Whether to use feature selection
        n_features : int, default=20
            Number of features to select
        use_pca : bool, default=False
            Whether to use PCA
        n_components : int, default=10
            Number of PCA components
        hidden_layers : List[int], default=[64, 32]
            Number of neurons in hidden layers
        activation : str, default='relu'
            Activation function
        learning_rate : float, default=0.001
            Learning rate
        batch_size : int, default=32
            Batch size
        epochs : int, default=100
            Number of epochs
        early_stopping : bool, default=True
            Whether to use early stopping
        patience : int, default=10
            Patience for early stopping
        dropout_rate : float, default=0.2
            Dropout rate
        random_state : int, optional
            Random state for reproducibility
        """
        super().__init__(
            name=name,
            target=target,
            confidence_level=confidence_level,
            forecast_horizon=forecast_horizon,
            feature_engineering=feature_engineering,
            scaler_type=scaler_type,
            feature_selection=feature_selection,
            n_features=n_features,
            use_pca=use_pca,
            n_components=n_components
        )
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
            tf.random.set_seed(self.random_state)
    
    def _build_model(self, input_dim: int) -> tf.keras.Model:
        """
        Build the Neural Network model.
        
        Parameters:
        -----------
        input_dim : int
            Input dimension
            
        Returns:
        --------
        tf.keras.Model
            Neural Network model
        """
        # Initialize model
        model = Sequential()
        
        # Add input layer
        model.add(Dense(self.hidden_layers[0], input_dim=input_dim, activation=self.activation))
        model.add(Dropout(self.dropout_rate))
        
        # Add hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation=self.activation))
            model.add(Dropout(self.dropout_rate))
        
        # Add output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Neural Network model to the data.
        
        Parameters:
        -----------
        X : np.ndarray
            Features array
        y : np.ndarray
            Target array
        """
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Build model
        self.model = self._build_model(X.shape[1])
        
        # Define callbacks
        callbacks = []
        
        if self.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True
                )
            )
        
        # Fit model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        # Log training history
        logger.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
        logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    def _predict(self, X: np.ndarray) -> float:
        """
        Make prediction with the Neural Network model.
        
        Parameters:
        -----------
        X : np.ndarray
            Features array
            
        Returns:
        --------
        float
            Prediction
        """
        # Make prediction
        prediction = self.model.predict(X, verbose=0)[0][0]
        
        return prediction


class LSTMRiskModel(MLRiskModel):
    """
    LSTM risk model.
    
    This class implements an LSTM model for risk prediction.
    """
    
    def __init__(self, 
                 name: str = "LSTM",
                 target: str = 'var',
                 confidence_level: float = 0.95,
                 forecast_horizon: int = 21,
                 feature_engineering: Optional[FeatureEngineering] = None,
                 scaler_type: str = 'standard',
                 feature_selection: bool = True,
                 n_features: int = 20,
                 use_pca: bool = False,
                 n_components: int = 10,
                 sequence_length: int = 20,
                 lstm_units: List[int] = [64, 32],
                 dense_units: List[int] = [16],
                 activation: str = 'relu',
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 early_stopping: bool = True,
                 patience: int = 10,
                 dropout_rate: float = 0.2,
                 random_state: Optional[int] = 42):
        """
        Initialize the LSTM risk model.
        
        Parameters:
        -----------
        name : str, default="LSTM"
            Name of the risk model
        target : str, default='var'
            Target variable ('var' or 'es')
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        forecast_horizon : int, default=21
            Forecast horizon for risk metrics (trading days)
        feature_engineering : FeatureEngineering, optional
            Feature engineering instance
        scaler_type : str, default='standard'
            Scaler type ('standard' or 'minmax')
        feature_selection : bool, default=True
            Whether to use feature selection
        n_features : int, default=20
            Number of features to select
        use_pca : bool, default=False
            Whether to use PCA
        n_components : int, default=10
            Number of PCA components
        sequence_length : int, default=20
            Sequence length for LSTM
        lstm_units : List[int], default=[64, 32]
            Number of units in LSTM layers
        dense_units : List[int], default=[16]
            Number of units in dense layers
        activation : str, default='relu'
            Activation function
        learning_rate : float, default=0.001
            Learning rate
        batch_size : int, default=32
            Batch size
        epochs : int, default=100
            Number of epochs
        early_stopping : bool, default=True
            Whether to use early stopping
        patience : int, default=10
            Patience for early stopping
        dropout_rate : float, default=0.2
            Dropout rate
        random_state : int, optional
            Random state for reproducibility
        """
        super().__init__(
            name=name,
            target=target,
            confidence_level=confidence_level,
            forecast_horizon=forecast_horizon,
            feature_engineering=feature_engineering,
            scaler_type=scaler_type,
            feature_selection=feature_selection,
            n_features=n_features,
            use_pca=use_pca,
            n_components=n_components
        )
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
            tf.random.set_seed(self.random_state)
    
    def _prepare_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare sequences for LSTM.
        
        Parameters:
        -----------
        X : np.ndarray
            Features array
            
        Returns:
        --------
        np.ndarray
            Sequences array
        """
        # Create sequences
        sequences = []
        
        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:i+self.sequence_length])
        
        return np.array(sequences)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build the LSTM model.
        
        Parameters:
        -----------
        input_shape : Tuple[int, int]
            Input shape (sequence_length, n_features)
            
        Returns:
        --------
        tf.keras.Model
            LSTM model
        """
        # Initialize model
        model = Sequential()
        
        # Add LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            
            if i == 0:
                model.add(LSTM(units, return_sequences=return_sequences, input_shape=input_shape))
            else:
                model.add(LSTM(units, return_sequences=return_sequences))
            
            model.add(Dropout(self.dropout_rate))
        
        # Add dense layers
        for units in self.dense_units:
            model.add(Dense(units, activation=self.activation))
            model.add(Dropout(self.dropout_rate))
        
        # Add output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the LSTM model to the data.
        
        Parameters:
        -----------
        X : np.ndarray
            Features array
        y : np.ndarray
            Target array
        """
        # Prepare sequences
        X_seq = self._prepare_sequences(X)
        y_seq = y[self.sequence_length:]
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=self.random_state
        )
        
        # Build model
        self.model = self._build_model((self.sequence_length, X.shape[1]))
        
        # Define callbacks
        callbacks = []
        
        if self.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True
                )
            )
        
        # Fit model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        # Log training history
        logger.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
        logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    def _predict(self, X: np.ndarray) -> float:
        """
        Make prediction with the LSTM model.
        
        Parameters:
        -----------
        X : np.ndarray
            Features array
            
        Returns:
        --------
        float
            Prediction
        """
        # Check if we have enough data for a sequence
        if len(X) < self.sequence_length:
            # Repeat the single data point to create a sequence
            X_seq = np.repeat(X, self.sequence_length, axis=0)
        else:
            # Use the last sequence_length data points
            X_seq = X[-self.sequence_length:]
        
        # Reshape for LSTM
        X_seq = X_seq.reshape(1, self.sequence_length, X.shape[1])
        
        # Make prediction
        prediction = self.model.predict(X_seq, verbose=0)[0][0]
        
        return prediction


class EnsembleMLRiskModel(MLRiskModel):
    """
    Ensemble ML risk model.
    
    This class implements an ensemble of ML models for risk prediction.
    """
    
    def __init__(self, 
                 name: str = "EnsembleML",
                 target: str = 'var',
                 confidence_level: float = 0.95,
                 forecast_horizon: int = 21,
                 feature_engineering: Optional[FeatureEngineering] = None,
                 scaler_type: str = 'standard',
                 feature_selection: bool = True,
                 n_features: int = 20,
                 use_pca: bool = False,
                 n_components: int = 10,
                 models: List[str] = ['rf', 'gbm', 'nn'],
                 ensemble_method: str = 'average',
                 meta_model_type: str = 'linear',
                 random_state: Optional[int] = 42):
        """
        Initialize the Ensemble ML risk model.
        
        Parameters:
        -----------
        name : str, default="EnsembleML"
            Name of the risk model
        target : str, default='var'
            Target variable ('var' or 'es')
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        forecast_horizon : int, default=21
            Forecast horizon for risk metrics (trading days)
        feature_engineering : FeatureEngineering, optional
            Feature engineering instance
        scaler_type : str, default='standard'
            Scaler type ('standard' or 'minmax')
        feature_selection : bool, default=True
            Whether to use feature selection
        n_features : int, default=20
            Number of features to select
        use_pca : bool, default=False
            Whether to use PCA
        n_components : int, default=10
            Number of PCA components
        models : List[str], default=['rf', 'gbm', 'nn']
            List of models to ensemble ('rf', 'gbm', 'svr', 'nn', 'lstm')
        ensemble_method : str, default='average'
            Ensemble method ('average', 'weighted', 'meta')
        meta_model_type : str, default='linear'
            Meta-model type for 'meta' ensemble method
        random_state : int, optional
            Random state for reproducibility
        """
        super().__init__(
            name=name,
            target=target,
            confidence_level=confidence_level,
            forecast_horizon=forecast_horizon,
            feature_engineering=feature_engineering,
            scaler_type=scaler_type,
            feature_selection=feature_selection,
            n_features=n_features,
            use_pca=use_pca,
            n_components=n_components
        )
        self.models_list = models
        self.ensemble_method = ensemble_method
        self.meta_model_type = meta_model_type
        self.random_state = random_state
        self.base_models = []
        self.model_weights = None
        self.meta_model = None
    
    def _initialize_models(self) -> List[MLRiskModel]:
        """
        Initialize base models.
        
        Returns:
        --------
        List[MLRiskModel]
            List of base models
        """
        models = []
        
        for model_type in self.models_list:
            if model_type == 'rf':
                models.append(
                    RandomForestRiskModel(
                        name="RF",
                        target=self.target,
                        confidence_level=self.confidence_level,
                        forecast_horizon=self.forecast_horizon,
                        feature_engineering=None,  # We'll use our own features
                        scaler_type=self.scaler_type,
                        feature_selection=False,  # We'll use our own feature selection
                        use_pca=False,  # We'll use our own PCA
                        random_state=self.random_state
                    )
                )
            elif model_type == 'gbm':
                models.append(
                    GradientBoostingRiskModel(
                        name="GBM",
                        target=self.target,
                        confidence_level=self.confidence_level,
                        forecast_horizon=self.forecast_horizon,
                        feature_engineering=None,
                        scaler_type=self.scaler_type,
                        feature_selection=False,
                        use_pca=False,
                        random_state=self.random_state
                    )
                )
            elif model_type == 'svr':
                models.append(
                    SVRRiskModel(
                        name="SVR",
                        target=self.target,
                        confidence_level=self.confidence_level,
                        forecast_horizon=self.forecast_horizon,
                        feature_engineering=None,
                        scaler_type=self.scaler_type,
                        feature_selection=False,
                        use_pca=False,
                        random_state=self.random_state
                    )
                )
            elif model_type == 'nn':
                models.append(
                    NeuralNetworkRiskModel(
                        name="NN",
                        target=self.target,
                        confidence_level=self.confidence_level,
                        forecast_horizon=self.forecast_horizon,
                        feature_engineering=None,
                        scaler_type=self.scaler_type,
                        feature_selection=False,
                        use_pca=False,
                        random_state=self.random_state
                    )
                )
            elif model_type == 'lstm':
                models.append(
                    LSTMRiskModel(
                        name="LSTM",
                        target=self.target,
                        confidence_level=self.confidence_level,
                        forecast_horizon=self.forecast_horizon,
                        feature_engineering=None,
                        scaler_type=self.scaler_type,
                        feature_selection=False,
                        use_pca=False,
                        random_state=self.random_state
                    )
                )
            else:
                logger.warning(f"Unknown model type: {model_type}")
        
        return models
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the ensemble model to the data.
        
        Parameters:
        -----------
        X : np.ndarray
            Features array
        y : np.ndarray
            Target array
        """
        # Initialize base models
        self.base_models = self._initialize_models()
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Fit base models
        for model in self.base_models:
            model.model = model._build_model(X.shape[1]) if hasattr(model, '_build_model') else None
            model._fit_model(X_train, y_train)
        
        # Get predictions from base models
        val_predictions = np.zeros((len(X_val), len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            for j in range(len(X_val)):
                val_predictions[j, i] = model._predict(X_val[j:j+1])
        
        # Determine ensemble method
        if self.ensemble_method == 'average':
            # Simple average
            self.model_weights = np.ones(len(self.base_models)) / len(self.base_models)
            
            logger.info(f"Using average ensemble with equal weights")
        elif self.ensemble_method == 'weighted':
            # Calculate weights based on validation performance
            errors = np.zeros(len(self.base_models))
            
            for i, model in enumerate(self.base_models):
                errors[i] = mean_squared_error(y_val, val_predictions[:, i])
            
            # Inverse error weighting
            weights = 1 / errors
            self.model_weights = weights / np.sum(weights)
            
            logger.info(f"Using weighted ensemble with weights: {self.model_weights}")
        elif self.ensemble_method == 'meta':
            # Train meta-model
            if self.meta_model_type == 'linear':
                self.meta_model = LinearRegression()
            elif self.meta_model_type == 'ridge':
                self.meta_model = Ridge(alpha=1.0, random_state=self.random_state)
            elif self.meta_model_type == 'lasso':
                self.meta_model = Lasso(alpha=0.1, random_state=self.random_state)
            elif self.meta_model_type == 'elasticnet':
                self.meta_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state)
            else:
                logger.warning(f"Unknown meta-model type: {self.meta_model_type}, using linear")
                self.meta_model = LinearRegression()
            
            # Fit meta-model
            self.meta_model.fit(val_predictions, y_val)
            
            # Log meta-model coefficients
            if hasattr(self.meta_model, 'coef_'):
                logger.info(f"Meta-model coefficients: {self.meta_model.coef_}")
            
            logger.info(f"Using meta-model ensemble with {self.meta_model_type}")
        else:
            logger.warning(f"Unknown ensemble method: {self.ensemble_method}, using average")
            self.model_weights = np.ones(len(self.base_models)) / len(self.base_models)
    
    def _predict(self, X: np.ndarray) -> float:
        """
        Make prediction with the ensemble model.
        
        Parameters:
        -----------
        X : np.ndarray
            Features array
            
        Returns:
        --------
        float
            Prediction
        """
        # Get predictions from base models
        predictions = np.zeros(len(self.base_models))
        
        for i, model in enumerate(self.base_models):
            predictions[i] = model._predict(X)
        
        # Combine predictions
        if self.ensemble_method == 'meta' and self.meta_model is not None:
            # Use meta-model
            ensemble_pred = self.meta_model.predict(predictions.reshape(1, -1))[0]
        else:
            # Use weighted average
            ensemble_pred = np.sum(predictions * self.model_weights)
        
        return ensemble_pred


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
    
    # Create feature engineering
    feature_eng = FeatureEngineering(
        include_technical=True,
        include_statistical=True,
        include_volatility=True,
        include_correlation=True,
        include_tail=True
    )
    
    # Create ML risk models
    rf_model = RandomForestRiskModel(
        name="RandomForest",
        target='var',
        confidence_level=0.95,
        forecast_horizon=21,
        feature_engineering=feature_eng
    )
    
    gbm_model = GradientBoostingRiskModel(
        name="GradientBoosting",
        target='var',
        confidence_level=0.95,
        forecast_horizon=21,
        feature_engineering=feature_eng
    )
    
    nn_model = NeuralNetworkRiskModel(
        name="NeuralNetwork",
        target='var',
        confidence_level=0.95,
        forecast_horizon=21,
        feature_engineering=feature_eng
    )
    
    ensemble_model = EnsembleMLRiskModel(
        name="EnsembleML",
        target='var',
        confidence_level=0.95,
        forecast_horizon=21,
        feature_engineering=feature_eng,
        models=['rf', 'gbm', 'nn'],
        ensemble_method='weighted'
    )
    
    # Fit models
    rf_model.fit(returns, prices)
    gbm_model.fit(returns, prices)
    nn_model.fit(returns, prices)
    ensemble_model.fit(returns, prices)
    
    # Equal weights for portfolio
    weights = np.ones(len(tickers)) / len(tickers)
    
    # Get predictions
    rf_var = rf_model.predict_var(returns, weights, 0.95, prices)
    gbm_var = gbm_model.predict_var(returns, weights, 0.95, prices)
    nn_var = nn_model.predict_var(returns, weights, 0.95, prices)
    ensemble_var = ensemble_model.predict_var(returns, weights, 0.95, prices)
    
    print(f"Random Forest VaR: {rf_var:.4f}")
    print(f"Gradient Boosting VaR: {gbm_var:.4f}")
    print(f"Neural Network VaR: {nn_var:.4f}")
    print(f"Ensemble VaR: {ensemble_var:.4f}")
    
    # Plot feature importance
    rf_importance = rf_model.plot_feature_importance()
    rf_importance.savefig('rf_importance.png')
    
    gbm_importance = gbm_model.plot_feature_importance()
    gbm_importance.savefig('gbm_importance.png')
