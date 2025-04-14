"""
Adversarial Training Module for Risk Modeling Framework.

This module provides functionality for adversarial training to improve robustness to market shocks.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
import gymnasium as gym  # Using gymnasium instead of gym
from gymnasium import spaces
from collections import deque
import random


class AdversarialTraining:
    """
    Adversarial Training class for risk modeling.
    
    This class provides methods for using adversarial training to improve
    robustness to market shocks.
    """
    
    def __init__(self):
        """
        Initialize the adversarial training model.
        """
        self.returns = None
        self.model = None
        self.adversarial_model = None
        self.state_size = 20  # Default: 20 features
        self.action_size = 5  # Default: 5 actions (risk levels)
        self.epsilon = 0.1  # Perturbation magnitude
        self.calibrated = False
    
    def calibrate(self, historical_data, state_size=None, action_size=None, epsilon=None):
        """
        Calibrate the adversarial training model with historical data.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        state_size : int, optional
            Size of state space
        action_size : int, optional
            Size of action space
        epsilon : float, optional
            Perturbation magnitude
        """
        self.returns = historical_data
        
        # Set parameters
        if state_size is not None:
            self.state_size = state_size
        
        if action_size is not None:
            self.action_size = action_size
        
        if epsilon is not None:
            self.epsilon = epsilon
        
        # Create base model
        self._create_base_model()
        
        # Create adversarial model
        self._create_adversarial_model()
        
        # Train models
        self._train_models()
        
        self.calibrated = True
    
    def _create_base_model(self):
        """
        Create base model.
        """
        # Create model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
    
    def _create_adversarial_model(self):
        """
        Create adversarial model.
        """
        # Create model
        self.adversarial_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.state_size, activation='tanh')
        ])
        
        # Compile model
        self.adversarial_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
    
    def _train_models(self, epochs=100, batch_size=32):
        """
        Train base and adversarial models.
        
        Parameters:
        -----------
        epochs : int, default=100
            Number of epochs to train
        batch_size : int, default=32
            Batch size
        """
        # Get training data
        X, y = self._get_training_data()
        
        # Training history
        self.training_history = {
            'base_loss': [],
            'adversarial_loss': []
        }
        
        for epoch in range(epochs):
            # Train base model
            base_loss = self._train_base_model(X, y, batch_size)
            
            # Train adversarial model
            adversarial_loss = self._train_adversarial_model(X, y, batch_size)
            
            # Update training history
            self.training_history['base_loss'].append(base_loss)
            self.training_history['adversarial_loss'].append(adversarial_loss)
    
    def _get_training_data(self):
        """
        Get training data.
        
        Returns:
        --------
        tuple
            (X, y)
        """
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        portfolio_returns = self.returns.dot(portfolio_weights)
        
        # Create features and targets
        X = []
        y = []
        
        for i in range(len(portfolio_returns) - self.state_size - 1):
            # Get state
            state = portfolio_returns.iloc[i:i+self.state_size].values
            
            # Get target (next return)
            target = portfolio_returns.iloc[i+self.state_size]
            
            # Map target to action
            action = self._map_return_to_action(target)
            
            # Add to training data
            X.append(state)
            y.append(action)
        
        return np.array(X), np.array(y)
    
    def _map_return_to_action(self, return_value):
        """
        Map return to action.
        
        Parameters:
        -----------
        return_value : float
            Return value
            
        Returns:
        --------
        numpy.ndarray
            One-hot encoded action
        """
        # Define return bins
        bins = np.linspace(-0.1, 0.1, self.action_size + 1)
        
        # Find bin index
        bin_index = np.digitize(return_value, bins) - 1
        
        # Clip bin index
        bin_index = np.clip(bin_index, 0, self.action_size - 1)
        
        # Create one-hot encoded action
        action = np.zeros(self.action_size)
        action[bin_index] = 1
        
        return action
    
    def _train_base_model(self, X, y, batch_size):
        """
        Train base model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features
        y : numpy.ndarray
            Targets
        batch_size : int
            Batch size
            
        Returns:
        --------
        float
            Loss
        """
        # Generate adversarial examples
        X_adv = self._generate_adversarial_examples(X)
        
        # Combine original and adversarial examples
        X_combined = np.vstack([X, X_adv])
        y_combined = np.vstack([y, y])
        
        # Train model
        history = self.model.fit(
            X_combined, y_combined,
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )
        
        return history.history['loss'][0]
    
    def _train_adversarial_model(self, X, y, batch_size):
        """
        Train adversarial model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features
        y : numpy.ndarray
            Targets
        batch_size : int
            Batch size
            
        Returns:
        --------
        float
            Loss
        """
        # Define adversarial loss function
        def adversarial_loss(y_true, y_pred):
            # Get base model predictions for original examples
            base_preds_orig = self.model(X)
            
            # Get perturbed examples
            X_perturbed = X + self.epsilon * y_pred
            
            # Get base model predictions for perturbed examples
            base_preds_pert = self.model(X_perturbed)
            
            # Calculate loss (maximize difference between predictions)
            return -tf.reduce_mean(tf.square(base_preds_pert - base_preds_orig))
        
        # Compile adversarial model with custom loss
        self.adversarial_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=adversarial_loss
        )
        
        # Train model
        history = self.adversarial_model.fit(
            X, np.zeros((len(X), self.state_size)),
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )
        
        return history.history['loss'][0]
    
    def _generate_adversarial_examples(self, X):
        """
        Generate adversarial examples.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Original examples
            
        Returns:
        --------
        numpy.ndarray
            Adversarial examples
        """
        # Get perturbations
        perturbations = self.adversarial_model.predict(X, verbose=0)
        
        # Generate adversarial examples
        X_adv = X + self.epsilon * perturbations
        
        return X_adv
    
    def predict(self, state, use_adversarial=False):
        """
        Predict action for a given state.
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
        use_adversarial : bool, default=False
            Whether to use adversarial examples
            
        Returns:
        --------
        numpy.ndarray
            Predicted action probabilities
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Reshape state
        state = np.reshape(state, (1, -1))
        
        if use_adversarial:
            # Generate adversarial example
            perturbation = self.adversarial_model.predict(state, verbose=0)
            state_adv = state + self.epsilon * perturbation
            
            # Predict action
            return self.model.predict(state_adv, verbose=0)[0]
        else:
            # Predict action
            return self.model.predict(state, verbose=0)[0]
    
    def get_robust_action(self, state):
        """
        Get robust action for a given state.
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
            
        Returns:
        --------
        int
            Robust action
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Get predictions for original and adversarial examples
        pred_orig = self.predict(state, use_adversarial=False)
        pred_adv = self.predict(state, use_adversarial=True)
        
        # Calculate robust action probabilities
        robust_probs = 0.5 * (pred_orig + pred_adv)
        
        # Get robust action
        return np.argmax(robust_probs)
    
    def evaluate_robustness(self, test_data):
        """
        Evaluate model robustness.
        
        Parameters:
        -----------
        test_data : pandas.DataFrame
            Test data
            
        Returns:
        --------
        dict
            Dictionary containing robustness metrics
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(test_data.shape[1]) / test_data.shape[1]
        portfolio_returns = test_data.dot(portfolio_weights)
        
        # Create features and targets
        X = []
        y = []
        
        for i in range(len(portfolio_returns) - self.state_size - 1):
            # Get state
            state = portfolio_returns.iloc[i:i+self.state_size].values
            
            # Get target (next return)
            target = portfolio_returns.iloc[i+self.state_size]
            
            # Map target to action
            action = self._map_return_to_action(target)
            
            # Add to test data
            X.append(state)
            y.append(action)
        
        X = np.array(X)
        y = np.array(y)
        
        # Generate adversarial examples
        X_adv = self._generate_adversarial_examples(X)
        
        # Evaluate on original examples
        loss_orig = self.model.evaluate(X, y, verbose=0)
        
        # Evaluate on adversarial examples
        loss_adv = self.model.evaluate(X_adv, y, verbose=0)
        
        # Calculate robustness metrics
        robustness_metrics = {
            'original_loss': loss_orig,
            'adversarial_loss': loss_adv,
            'robustness_ratio': loss_orig / loss_adv if loss_adv > 0 else float('inf')
        }
        
        return robustness_metrics
    
    def plot_training_history(self, figsize=(12, 6)):
        """
        Plot training history.
        
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
        
        # Plot training history
        ax.plot(self.training_history['base_loss'], label='Base Model Loss')
        ax.plot(self.training_history['adversarial_loss'], label='Adversarial Model Loss')
        
        # Set labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_adversarial_examples(self, test_data, num_examples=5, figsize=(15, 10)):
        """
        Plot adversarial examples.
        
        Parameters:
        -----------
        test_data : pandas.DataFrame
            Test data
        num_examples : int, default=5
            Number of examples to plot
        figsize : tuple, default=(15, 10)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(test_data.shape[1]) / test_data.shape[1]
        portfolio_returns = test_data.dot(portfolio_weights)
        
        # Get random states
        indices = np.random.choice(
            len(portfolio_returns) - self.state_size,
            size=num_examples,
            replace=False
        )
        
        # Create figure
        fig, axes = plt.subplots(num_examples, 2, figsize=figsize)
        
        for i, idx in enumerate(indices):
            # Get state
            state = portfolio_returns.iloc[idx:idx+self.state_size].values
            
            # Generate adversarial example
            perturbation = self.adversarial_model.predict(
                np.reshape(state, (1, -1)),
                verbose=0
            )[0]
            state_adv = state + self.epsilon * perturbation
            
            # Plot original state
            axes[i, 0].plot(state)
            axes[i, 0].set_title(f'Original State {i+1}')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot adversarial state
            axes[i, 1].plot(state_adv)
            axes[i, 1].set_title(f'Adversarial State {i+1}')
            axes[i, 1].grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_robustness_comparison(self, test_data, figsize=(12, 8)):
        """
        Plot robustness comparison.
        
        Parameters:
        -----------
        test_data : pandas.DataFrame
            Test data
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
        portfolio_weights = np.ones(test_data.shape[1]) / test_data.shape[1]
        portfolio_returns = test_data.dot(portfolio_weights)
        
        # Create features
        X = []
        
        for i in range(len(portfolio_returns) - self.state_size - 1):
            # Get state
            state = portfolio_returns.iloc[i:i+self.state_size].values
            
            # Add to test data
            X.append(state)
        
        X = np.array(X)
        
        # Generate adversarial examples
        X_adv = self._generate_adversarial_examples(X)
        
        # Get predictions
        preds_orig = self.model.predict(X, verbose=0)
        preds_adv = self.model.predict(X_adv, verbose=0)
        
        # Calculate prediction differences
        pred_diffs = np.abs(preds_orig - preds_adv).mean(axis=1)
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot prediction differences
        axes[0].hist(pred_diffs, bins=30, alpha=0.7)
        axes[0].set_xlabel('Prediction Difference')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Prediction Differences')
        axes[0].grid(True, alpha=0.3)
        
        # Plot cumulative distribution
        axes[1].hist(pred_diffs, bins=30, alpha=0.7, cumulative=True, density=True)
        axes[1].set_xlabel('Prediction Difference')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].set_title('Cumulative Distribution of Prediction Differences')
        axes[1].grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
