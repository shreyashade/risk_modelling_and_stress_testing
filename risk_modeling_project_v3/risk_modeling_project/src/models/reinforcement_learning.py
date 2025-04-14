"""
Reinforcement Learning Module for Risk Modeling Framework.

This module provides functionality for reinforcement learning-based risk management.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random


class RLRiskManager:
    """
    Reinforcement Learning Risk Manager class for risk modeling.
    
    This class provides methods for using reinforcement learning to dynamically
    adjust risk parameters.
    """
    
    def __init__(self):
        """
        Initialize the RL risk manager.
        """
        self.returns = None
        self.model = None
        self.env = None
        self.agent = None
        self.state_size = 20  # Default: 20 features
        self.action_size = 5  # Default: 5 actions (risk levels)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.calibrated = False
    
    def calibrate(self, historical_data, state_size=None, action_size=None):
        """
        Calibrate the RL risk manager with historical data.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        state_size : int, optional
            Size of state space
        action_size : int, optional
            Size of action space
        """
        self.returns = historical_data
        
        # Set parameters
        if state_size is not None:
            self.state_size = state_size
        
        if action_size is not None:
            self.action_size = action_size
        
        # Create environment
        self._create_environment()
        
        # Create agent
        self._create_agent()
        
        # Train agent
        self._train_agent()
        
        self.calibrated = True
    
    def _create_environment(self):
        """
        Create reinforcement learning environment.
        """
        self.env = RiskManagementEnv(self.returns, self.state_size, self.action_size)
    
    def _create_agent(self):
        """
        Create reinforcement learning agent.
        """
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            memory=self.memory,
            gamma=self.gamma,
            epsilon=self.epsilon,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            learning_rate=self.learning_rate
        )
    
    def _train_agent(self, episodes=100):
        """
        Train reinforcement learning agent.
        
        Parameters:
        -----------
        episodes : int, default=100
            Number of episodes to train
        """
        # Training history
        self.training_history = {
            'episode': [],
            'reward': [],
            'epsilon': []
        }
        
        for episode in range(episodes):
            # Reset environment
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            
            # Initialize variables
            done = False
            total_reward = 0
            
            while not done:
                # Choose action
                action = self.agent.act(state)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                
                # Remember experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                
                # Update total reward
                total_reward += reward
                
                # Train agent
                if len(self.agent.memory) > self.batch_size:
                    self.agent.replay(self.batch_size)
            
            # Update training history
            self.training_history['episode'].append(episode)
            self.training_history['reward'].append(total_reward)
            self.training_history['epsilon'].append(self.agent.epsilon)
    
    def get_optimal_action(self, state):
        """
        Get optimal action for a given state.
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
            
        Returns:
        --------
        int
            Optimal action
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Reshape state
        state = np.reshape(state, [1, self.state_size])
        
        # Get optimal action
        return self.agent.act(state, use_epsilon=False)
    
    def get_optimal_risk_parameters(self, state):
        """
        Get optimal risk parameters for a given state.
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
            
        Returns:
        --------
        dict
            Dictionary containing optimal risk parameters
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Get optimal action
        action = self.get_optimal_action(state)
        
        # Map action to risk parameters
        risk_parameters = self.env.action_to_risk_parameters(action)
        
        return risk_parameters
    
    def plot_training_history(self, figsize=(12, 8)):
        """
        Plot training history.
        
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
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot rewards
        axes[0].plot(self.training_history['episode'], self.training_history['reward'])
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Rewards during Training')
        axes[0].grid(True, alpha=0.3)
        
        # Plot epsilon
        axes[1].plot(self.training_history['episode'], self.training_history['epsilon'])
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Epsilon')
        axes[1].set_title('Exploration Rate during Training')
        axes[1].grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_action_distribution(self, figsize=(12, 6)):
        """
        Plot action distribution.
        
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
        
        # Get states
        states = self.env.get_all_states()
        
        # Get actions
        actions = [self.get_optimal_action(state) for state in states]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot action distribution
        ax.hist(actions, bins=self.action_size, alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('Action')
        ax.set_ylabel('Frequency')
        ax.set_title('Action Distribution')
        
        # Set x-ticks
        ax.set_xticks(range(self.action_size))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig


class RiskManagementEnv(gym.Env):
    """
    Risk Management Environment for reinforcement learning.
    """
    
    def __init__(self, returns, state_size, action_size):
        """
        Initialize the risk management environment.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            Historical returns data
        state_size : int
            Size of state space
        action_size : int
            Size of action space
        """
        super(RiskManagementEnv, self).__init__()
        
        self.returns = returns
        self.state_size = state_size
        self.action_size = action_size
        
        # Define action and observation space
        self.action_space = spaces.Discrete(action_size)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        
        # Initialize state
        self.current_step = 0
        self.max_steps = len(returns) - state_size - 1
        self.state = None
    
    def reset(self):
        """
        Reset the environment.
        
        Returns:
        --------
        numpy.ndarray
            Initial state
        """
        # Reset current step
        self.current_step = 0
        
        # Get initial state
        self.state = self._get_state()
        
        return self.state
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Parameters:
        -----------
        action : int
            Action to take
            
        Returns:
        --------
        tuple
            (next_state, reward, done, info)
        """
        # Get current risk parameters
        risk_params = self.action_to_risk_parameters(action)
        
        # Calculate reward
        reward = self._calculate_reward(risk_params)
        
        # Update current step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        # Get next state
        if not done:
            next_state = self._get_state()
        else:
            next_state = self.state
        
        # Update state
        self.state = next_state
        
        return next_state, reward, done, {}
    
    def _get_state(self):
        """
        Get current state.
        
        Returns:
        --------
        numpy.ndarray
            Current state
        """
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        portfolio_returns = self.returns.dot(portfolio_weights)
        
        # Get state features
        state = []
        
        # Add recent returns
        for i in range(self.current_step, self.current_step + self.state_size):
            state.append(portfolio_returns.iloc[i])
        
        return np.array(state)
    
    def action_to_risk_parameters(self, action):
        """
        Map action to risk parameters.
        
        Parameters:
        -----------
        action : int
            Action
            
        Returns:
        --------
        dict
            Dictionary containing risk parameters
        """
        # Define risk parameter ranges
        var_confidence_levels = np.linspace(0.9, 0.99, self.action_size)
        lookback_windows = np.linspace(63, 252, self.action_size).astype(int)
        
        # Map action to risk parameters
        risk_params = {
            'var_confidence_level': var_confidence_levels[action],
            'lookback_window': lookback_windows[action]
        }
        
        return risk_params
    
    def _calculate_reward(self, risk_params):
        """
        Calculate reward for current action.
        
        Parameters:
        -----------
        risk_params : dict
            Dictionary containing risk parameters
            
        Returns:
        --------
        float
            Reward
        """
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        portfolio_returns = self.returns.dot(portfolio_weights)
        
        # Get lookback window
        lookback_window = risk_params['lookback_window']
        
        # Get confidence level
        confidence_level = risk_params['var_confidence_level']
        
        # Get historical returns for VaR calculation
        historical_returns = portfolio_returns.iloc[
            self.current_step - lookback_window:self.current_step
        ]
        
        # Calculate VaR
        var = -np.percentile(historical_returns, 100 * (1 - confidence_level))
        
        # Get actual return
        actual_return = portfolio_returns.iloc[self.current_step]
        
        # Calculate VaR violation
        var_violation = 1 if actual_return < -var else 0
        
        # Calculate expected violation rate
        expected_violation_rate = 1 - confidence_level
        
        # Calculate reward components
        accuracy_reward = -abs(var_violation - expected_violation_rate)
        efficiency_reward = -var / abs(portfolio_returns.mean())
        
        # Calculate total reward
        reward = accuracy_reward + 0.5 * efficiency_reward
        
        return reward
    
    def get_all_states(self):
        """
        Get all possible states.
        
        Returns:
        --------
        list
            List of all possible states
        """
        states = []
        
        for i in range(self.max_steps):
            self.current_step = i
            states.append(self._get_state())
        
        return states


class DQNAgent:
    """
    Deep Q-Network Agent for reinforcement learning.
    """
    
    def __init__(self, state_size, action_size, memory, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.001):
        """
        Initialize the DQN agent.
        
        Parameters:
        -----------
        state_size : int
            Size of state space
        action_size : int
            Size of action space
        memory : collections.deque
            Replay memory
        gamma : float, default=0.95
            Discount factor
        epsilon : float, default=1.0
            Exploration rate
        epsilon_min : float, default=0.01
            Minimum exploration rate
        epsilon_decay : float, default=0.995
            Exploration rate decay
        learning_rate : float, default=0.001
            Learning rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = memory
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build neural network model.
        
        Returns:
        --------
        tensorflow.keras.models.Sequential
            Neural network model
        """
        # Create model
        model = tf.keras.models.Sequential()
        
        # Add layers
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        
        # Compile model
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """
        Remember experience.
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : numpy.ndarray
            Next state
        done : bool
            Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, use_epsilon=True):
        """
        Choose action.
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
        use_epsilon : bool, default=True
            Whether to use epsilon-greedy policy
            
        Returns:
        --------
        int
            Chosen action
        """
        if use_epsilon and np.random.rand() <= self.epsilon:
            # Explore
            return np.random.randint(self.action_size)
        else:
            # Exploit
            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """
        Train model using experience replay.
        
        Parameters:
        -----------
        batch_size : int
            Batch size
        """
        # Sample batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            # Calculate target
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state, verbose=0)[0]
                )
            
            # Get current Q-values
            target_f = self.model.predict(state, verbose=0)
            
            # Update Q-value for chosen action
            target_f[0][action] = target
            
            # Train model
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
