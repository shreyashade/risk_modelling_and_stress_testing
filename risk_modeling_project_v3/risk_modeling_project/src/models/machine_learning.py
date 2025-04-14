"""
Machine Learning Module for Risk Modeling Framework.

This module provides functionality for machine learning-based risk modeling.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class MachineLearningRiskModel:
    """
    Machine Learning Risk Model class for risk modeling.
    
    This class provides methods for using machine learning to model and predict
    risk metrics.
    """
    
    def __init__(self):
        """
        Initialize the machine learning risk model.
        """
        self.returns = None
        self.features = None
        self.target = None
        self.model_type = 'random_forest'
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.lookback_period = 20
        self.prediction_horizon = 5
        self.calibrated = False
    
    def calibrate(self, historical_data, target_metric='volatility', model_type='random_forest', regime=None):
        """
        Calibrate the machine learning risk model with historical data.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        target_metric : str, default='volatility'
            Target risk metric to predict ('volatility', 'var', 'es')
        model_type : str, default='random_forest'
            Type of machine learning model to use
            ('random_forest', 'gradient_boosting', 'neural_network', 'lstm')
        regime : dict, optional
            Current market regime information
        """
        self.returns = historical_data
        self.model_type = model_type
        
        # Prepare features and target
        self._prepare_data(historical_data, target_metric)
        
        # Train model
        self._train_model(regime)
        
        self.calibrated = True
    
    def _prepare_data(self, historical_data, target_metric):
        """
        Prepare data for machine learning model.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        target_metric : str
            Target risk metric to predict ('volatility', 'var', 'es')
        """
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(historical_data.shape[1]) / historical_data.shape[1]
        portfolio_returns = historical_data.dot(portfolio_weights)
        
        # Create features
        self.features = self._create_features(historical_data, portfolio_returns)
        
        # Create target
        self.target = self._create_target(portfolio_returns, target_metric)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, shuffle=False
        )
        
        # Scale features and target
        self.X_train_scaled = self.scaler_X.fit_transform(X_train)
        self.X_test_scaled = self.scaler_X.transform(X_test)
        self.y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        self.y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # Reshape data for LSTM if needed
        if self.model_type == 'lstm' or self.model_type == 'gru':
            self.X_train_lstm = self.X_train_scaled.reshape(
                (self.X_train_scaled.shape[0], 1, self.X_train_scaled.shape[1])
            )
            self.X_test_lstm = self.X_test_scaled.reshape(
                (self.X_test_scaled.shape[0], 1, self.X_test_scaled.shape[1])
            )
    
    def _create_features(self, historical_data, portfolio_returns):
        """
        Create features for machine learning model.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        portfolio_returns : pandas.Series
            Portfolio returns
            
        Returns:
        --------
        numpy.ndarray
            Features for machine learning model
        """
        # Initialize list of features
        features_list = []
        
        # Add lagged returns
        for lag in range(1, self.lookback_period + 1):
            features_list.append(portfolio_returns.shift(lag).values[self.lookback_period:])
        
        # Add rolling statistics
        for window in [5, 10, 20, 60]:
            # Rolling mean
            features_list.append(
                portfolio_returns.rolling(window=window).mean().values[self.lookback_period:]
            )
            
            # Rolling standard deviation
            features_list.append(
                portfolio_returns.rolling(window=window).std().values[self.lookback_period:]
            )
            
            # Rolling skewness
            features_list.append(
                portfolio_returns.rolling(window=window).skew().values[self.lookback_period:]
            )
            
            # Rolling kurtosis
            features_list.append(
                portfolio_returns.rolling(window=window).kurt().values[self.lookback_period:]
            )
        
        # Add EWMA volatility
        for alpha in [0.94, 0.97]:
            features_list.append(
                portfolio_returns.ewm(alpha=alpha).std().values[self.lookback_period:]
            )
        
        # Add correlation features
        if historical_data.shape[1] > 1:
            # Pairwise correlations
            for i in range(historical_data.shape[1]):
                for j in range(i+1, historical_data.shape[1]):
                    rolling_corr = historical_data.iloc[:, i].rolling(window=20).corr(
                        historical_data.iloc[:, j]
                    )
                    features_list.append(rolling_corr.values[self.lookback_period:])
        
        # Add technical indicators
        # RSI
        delta = portfolio_returns.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        features_list.append(rsi.values[self.lookback_period:])
        
        # MACD
        ema12 = portfolio_returns.ewm(span=12).mean()
        ema26 = portfolio_returns.ewm(span=26).mean()
        macd = ema12 - ema26
        features_list.append(macd.values[self.lookback_period:])
        
        # Stack features
        features = np.column_stack(features_list)
        
        # Remove NaN values
        features = features[~np.isnan(features).any(axis=1)]
        
        return features
    
    def _create_target(self, portfolio_returns, target_metric):
        """
        Create target for machine learning model.
        
        Parameters:
        -----------
        portfolio_returns : pandas.Series
            Portfolio returns
        target_metric : str
            Target risk metric to predict ('volatility', 'var', 'es')
            
        Returns:
        --------
        numpy.ndarray
            Target for machine learning model
        """
        if target_metric == 'volatility':
            # Future realized volatility
            target = portfolio_returns.rolling(window=self.prediction_horizon).std().shift(-self.prediction_horizon)
            target = target.values[self.lookback_period:]
        
        elif target_metric == 'var':
            # Future realized VaR (95%)
            target = np.zeros(len(portfolio_returns) - self.lookback_period)
            
            for i in range(len(target)):
                future_returns = portfolio_returns.iloc[i + self.lookback_period:i + self.lookback_period + self.prediction_horizon]
                target[i] = -np.percentile(future_returns, 5)
        
        elif target_metric == 'es':
            # Future realized ES (95%)
            target = np.zeros(len(portfolio_returns) - self.lookback_period)
            
            for i in range(len(target)):
                future_returns = portfolio_returns.iloc[i + self.lookback_period:i + self.lookback_period + self.prediction_horizon]
                var = -np.percentile(future_returns, 5)
                target[i] = -future_returns[future_returns <= -var].mean()
        
        else:
            raise ValueError(f"Unknown target metric: {target_metric}")
        
        # Remove NaN values
        target = target[~np.isnan(target)]
        
        return target
    
    def _train_model(self, regime=None):
        """
        Train machine learning model.
        
        Parameters:
        -----------
        regime : dict, optional
            Current market regime information
        """
        if self.model_type == 'random_forest':
            # Train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.model.fit(self.X_train_scaled, self.y_train_scaled)
        
        elif self.model_type == 'gradient_boosting':
            # Train Gradient Boosting model
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.model.fit(self.X_train_scaled, self.y_train_scaled)
        
        elif self.model_type == 'neural_network':
            # Train Neural Network model
            self.model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
            self.model.fit(self.X_train_scaled, self.y_train_scaled)
        
        elif self.model_type == 'lstm':
            # Train LSTM model
            self.model = Sequential()
            self.model.add(LSTM(50, return_sequences=True, input_shape=(1, self.X_train_scaled.shape[1])))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(50))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(1))
            
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            self.model.fit(
                self.X_train_lstm,
                self.y_train_scaled,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
        
        elif self.model_type == 'gru':
            # Train GRU model
            self.model = Sequential()
            self.model.add(GRU(50, return_sequences=True, input_shape=(1, self.X_train_scaled.shape[1])))
            self.model.add(Dropout(0.2))
            self.model.add(GRU(50))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(1))
            
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            self.model.fit(
                self.X_train_lstm,
                self.y_train_scaled,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Evaluate model
        self._evaluate_model()
    
    def _evaluate_model(self):
        """
        Evaluate machine learning model.
        """
        # Make predictions
        if self.model_type in ['lstm', 'gru']:
            y_pred_scaled = self.model.predict(self.X_test_lstm, verbose=0).flatten()
        else:
            y_pred_scaled = self.model.predict(self.X_test_scaled)
        
        # Inverse transform predictions
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test = self.scaler_y.inverse_transform(self.y_test_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store evaluation metrics
        self.evaluation_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def predict(self, new_data):
        """
        Predict risk metrics using machine learning model.
        
        Parameters:
        -----------
        new_data : pandas.DataFrame
            New data for prediction
            
        Returns:
        --------
        float
            Predicted risk metric
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Prepare features
        portfolio_weights = np.ones(new_data.shape[1]) / new_data.shape[1]
        portfolio_returns = new_data.dot(portfolio_weights)
        
        features = self._create_features(new_data, portfolio_returns)
        
        # Scale features
        features_scaled = self.scaler_X.transform(features)
        
        # Make prediction
        if self.model_type in ['lstm', 'gru']:
            features_lstm = features_scaled.reshape((features_scaled.shape[0], 1, features_scaled.shape[1]))
            prediction_scaled = self.model.predict(features_lstm, verbose=0).flatten()
        else:
            prediction_scaled = self.model.predict(features_scaled)
        
        # Inverse transform prediction
        prediction = self.scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()
        
        return prediction
    
    def estimate_risk(self, confidence_level=0.95, new_data=None):
        """
        Estimate risk metrics using machine learning model.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        new_data : pandas.DataFrame, optional
            New data for prediction (if None, use last available data)
            
        Returns:
        --------
        dict
            Dictionary containing risk metrics
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Make prediction
        if new_data is not None:
            prediction = self.predict(new_data)
        else:
            # Use last prediction from test set
            if self.model_type in ['lstm', 'gru']:
                prediction_scaled = self.model.predict(
                    self.X_test_lstm[-1:], verbose=0
                ).flatten()
            else:
                prediction_scaled = self.model.predict(self.X_test_scaled[-1:])
            
            prediction = self.scaler_y.inverse_transform(
                prediction_scaled.reshape(-1, 1)
            ).flatten()
        
        # Return risk metrics
        return {
            'prediction': prediction[-1] if len(prediction) > 0 else prediction,
            'confidence_level': confidence_level,
            'model_type': self.model_type,
            'evaluation_metrics': self.evaluation_metrics
        }
    
    def plot_predictions(self, figsize=(12, 6)):
        """
        Plot actual vs. predicted values.
        
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
        
        # Make predictions
        if self.model_type in ['lstm', 'gru']:
            y_pred_scaled = self.model.predict(self.X_test_lstm, verbose=0).flatten()
        else:
            y_pred_scaled = self.model.predict(self.X_test_scaled)
        
        # Inverse transform predictions
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test = self.scaler_y.inverse_transform(self.y_test_scaled.reshape(-1, 1)).flatten()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot actual vs. predicted values
        ax.plot(y_test, label='Actual')
        ax.plot(y_pred, label='Predicted')
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'Actual vs. Predicted Values ({self.model_type.replace("_", " ").title()})')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance(self, figsize=(12, 8)):
        """
        Plot feature importance.
        
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
        
        if self.model_type not in ['random_forest', 'gradient_boosting']:
            raise ValueError(f"Feature importance not available for {self.model_type}")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create feature names
        feature_names = [f'Feature {i}' for i in range(len(importance))]
        
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot feature importance
        ax.bar(range(len(indices)), importance[indices])
        
        # Set labels and title
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance')
        ax.set_title('Feature Importance')
        
        # Set x-axis ticks
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
