"""
Multi-Agent System Module for Risk Modeling Framework.

This module provides functionality for multi-agent systems for complex portfolio optimization.
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


class MultiAgentSystem:
    """
    Multi-Agent System class for risk modeling.
    
    This class provides methods for using multi-agent systems for complex
    portfolio optimization.
    """
    
    def __init__(self):
        """
        Initialize the multi-agent system.
        """
        self.returns = None
        self.agents = []
        self.num_agents = 3  # Default: 3 agents (conservative, balanced, aggressive)
        self.state_size = 20  # Default: 20 features
        self.action_size = 5  # Default: 5 actions (allocation levels)
        self.calibrated = False
    
    def calibrate(self, historical_data, num_agents=None, state_size=None, action_size=None):
        """
        Calibrate the multi-agent system with historical data.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        num_agents : int, optional
            Number of agents
        state_size : int, optional
            Size of state space
        action_size : int, optional
            Size of action space
        """
        self.returns = historical_data
        
        # Set parameters
        if num_agents is not None:
            self.num_agents = num_agents
        
        if state_size is not None:
            self.state_size = state_size
        
        if action_size is not None:
            self.action_size = action_size
        
        # Create agents
        self._create_agents()
        
        # Create environment
        self._create_environment()
        
        # Train agents
        self._train_agents()
        
        self.calibrated = True
    
    def _create_agents(self):
        """
        Create agents.
        """
        self.agents = []
        
        for i in range(self.num_agents):
            # Create agent
            agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                memory=deque(maxlen=2000),
                gamma=0.95,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.995,
                learning_rate=0.001
            )
            
            # Add agent to list
            self.agents.append(agent)
    
    def _create_environment(self):
        """
        Create environment.
        """
        self.env = PortfolioOptimizationEnv(
            self.returns,
            self.state_size,
            self.action_size,
            self.num_agents
        )
    
    def _train_agents(self, episodes=100):
        """
        Train agents.
        
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
            states = self.env.reset()
            states = [np.reshape(state, [1, self.state_size]) for state in states]
            
            # Initialize variables
            done = False
            total_reward = 0
            
            while not done:
                # Choose actions
                actions = [agent.act(state) for agent, state in zip(self.agents, states)]
                
                # Take actions
                next_states, rewards, done, _ = self.env.step(actions)
                next_states = [np.reshape(state, [1, self.state_size]) for state in next_states]
                
                # Remember experiences
                for i, agent in enumerate(self.agents):
                    agent.remember(states[i], actions[i], rewards[i], next_states[i], done)
                
                # Update states
                states = next_states
                
                # Update total reward
                total_reward += sum(rewards)
                
                # Train agents
                for agent in self.agents:
                    if len(agent.memory) > agent.batch_size:
                        agent.replay(agent.batch_size)
            
            # Update training history
            self.training_history['episode'].append(episode)
            self.training_history['reward'].append(total_reward)
            self.training_history['epsilon'].append(self.agents[0].epsilon)
    
    def get_optimal_actions(self, states):
        """
        Get optimal actions for given states.
        
        Parameters:
        -----------
        states : list
            List of states
            
        Returns:
        --------
        list
            List of optimal actions
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Reshape states
        states = [np.reshape(state, [1, self.state_size]) for state in states]
        
        # Get optimal actions
        return [agent.act(state, use_epsilon=False) for agent, state in zip(self.agents, states)]
    
    def get_optimal_portfolio_allocation(self, states):
        """
        Get optimal portfolio allocation for given states.
        
        Parameters:
        -----------
        states : list
            List of states
            
        Returns:
        --------
        numpy.ndarray
            Optimal portfolio allocation
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Get optimal actions
        actions = self.get_optimal_actions(states)
        
        # Map actions to portfolio allocation
        return self.env.actions_to_allocation(actions)
    
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
    
    def plot_agent_allocations(self, test_data, figsize=(12, 8)):
        """
        Plot agent allocations.
        
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
        
        # Get states
        states = []
        
        for i in range(self.num_agents):
            # Calculate agent-specific features
            agent_features = self._calculate_agent_features(test_data, i)
            
            # Add to states
            states.append(agent_features)
        
        # Get optimal actions
        actions = self.get_optimal_actions(states)
        
        # Map actions to portfolio allocation
        allocation = self.env.actions_to_allocation(actions)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot allocation
        ax.bar(range(len(allocation)), allocation)
        
        # Set labels and title
        ax.set_xlabel('Asset')
        ax.set_ylabel('Allocation')
        ax.set_title('Optimal Portfolio Allocation')
        
        # Set x-ticks
        ax.set_xticks(range(len(allocation)))
        ax.set_xticklabels([f'Asset {i+1}' for i in range(len(allocation))])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def _calculate_agent_features(self, data, agent_index):
        """
        Calculate agent-specific features.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data
        agent_index : int
            Agent index
            
        Returns:
        --------
        numpy.ndarray
            Agent-specific features
        """
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(data.shape[1]) / data.shape[1]
        portfolio_returns = data.dot(portfolio_weights)
        
        # Calculate features based on agent type
        if agent_index == 0:
            # Conservative agent (focus on downside risk)
            features = self._calculate_conservative_features(portfolio_returns)
        elif agent_index == 1:
            # Balanced agent (focus on risk-adjusted returns)
            features = self._calculate_balanced_features(portfolio_returns)
        else:
            # Aggressive agent (focus on upside potential)
            features = self._calculate_aggressive_features(portfolio_returns)
        
        return features
    
    def _calculate_conservative_features(self, returns):
        """
        Calculate conservative agent features.
        
        Parameters:
        -----------
        returns : pandas.Series
            Portfolio returns
            
        Returns:
        --------
        numpy.ndarray
            Conservative agent features
        """
        # Calculate features
        features = []
        
        # Add recent returns
        features.extend(returns.iloc[-self.state_size//2:].values)
        
        # Add downside risk measures
        for window in [21, 63, 126]:
            # Calculate downside deviation
            downside_returns = returns.iloc[-window:][returns.iloc[-window:] < 0]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
            features.append(downside_deviation)
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + returns.iloc[-window:]).cumprod()
            max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
            features.append(max_drawdown)
            
            # Calculate VaR
            var_95 = np.percentile(returns.iloc[-window:], 5)
            features.append(var_95)
            
            # Calculate CVaR
            cvar_95 = returns.iloc[-window:][returns.iloc[-window:] <= var_95].mean()
            features.append(cvar_95)
        
        # Pad features if necessary
        if len(features) < self.state_size:
            features.extend([0] * (self.state_size - len(features)))
        
        return np.array(features[:self.state_size])
    
    def _calculate_balanced_features(self, returns):
        """
        Calculate balanced agent features.
        
        Parameters:
        -----------
        returns : pandas.Series
            Portfolio returns
            
        Returns:
        --------
        numpy.ndarray
            Balanced agent features
        """
        # Calculate features
        features = []
        
        # Add recent returns
        features.extend(returns.iloc[-self.state_size//2:].values)
        
        # Add risk-adjusted return measures
        for window in [21, 63, 126]:
            # Calculate mean return
            mean_return = returns.iloc[-window:].mean()
            features.append(mean_return)
            
            # Calculate volatility
            volatility = returns.iloc[-window:].std()
            features.append(volatility)
            
            # Calculate Sharpe ratio
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            features.append(sharpe_ratio)
            
            # Calculate Sortino ratio
            downside_returns = returns.iloc[-window:][returns.iloc[-window:] < 0]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
            sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
            features.append(sortino_ratio)
        
        # Pad features if necessary
        if len(features) < self.state_size:
            features.extend([0] * (self.state_size - len(features)))
        
        return np.array(features[:self.state_size])
    
    def _calculate_aggressive_features(self, returns):
        """
        Calculate aggressive agent features.
        
        Parameters:
        -----------
        returns : pandas.Series
            Portfolio returns
            
        Returns:
        --------
        numpy.ndarray
            Aggressive agent features
        """
        # Calculate features
        features = []
        
        # Add recent returns
        features.extend(returns.iloc[-self.state_size//2:].values)
        
        # Add upside potential measures
        for window in [21, 63, 126]:
            # Calculate mean return
            mean_return = returns.iloc[-window:].mean()
            features.append(mean_return)
            
            # Calculate upside deviation
            upside_returns = returns.iloc[-window:][returns.iloc[-window:] > 0]
            upside_deviation = np.std(upside_returns) if len(upside_returns) > 0 else 0
            features.append(upside_deviation)
            
            # Calculate maximum return
            max_return = returns.iloc[-window:].max()
            features.append(max_return)
            
            # Calculate upside potential ratio
            upside_potential = upside_returns.mean() if len(upside_returns) > 0 else 0
            downside_returns = returns.iloc[-window:][returns.iloc[-window:] < 0]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
            upside_potential_ratio = upside_potential / downside_deviation if downside_deviation > 0 else 0
            features.append(upside_potential_ratio)
        
        # Pad features if necessary
        if len(features) < self.state_size:
            features.extend([0] * (self.state_size - len(features)))
        
        return np.array(features[:self.state_size])


class PortfolioOptimizationEnv(gym.Env):
    """
    Portfolio Optimization Environment for multi-agent reinforcement learning.
    """
    
    def __init__(self, returns, state_size, action_size, num_agents):
        """
        Initialize the portfolio optimization environment.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            Historical returns data
        state_size : int
            Size of state space
        action_size : int
            Size of action space
        num_agents : int
            Number of agents
        """
        super(PortfolioOptimizationEnv, self).__init__()
        
        self.returns = returns
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.num_assets = returns.shape[1]
        
        # Define action and observation space
        self.action_space = spaces.Tuple([spaces.Discrete(action_size) for _ in range(num_agents)])
        self.observation_space = spaces.Tuple([
            spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)
            for _ in range(num_agents)
        ])
        
        # Initialize state
        self.current_step = 0
        self.max_steps = len(returns) - state_size - 1
        self.states = None
    
    def reset(self):
        """
        Reset the environment.
        
        Returns:
        --------
        list
            Initial states
        """
        # Reset current step
        self.current_step = 0
        
        # Get initial states
        self.states = self._get_states()
        
        return self.states
    
    def step(self, actions):
        """
        Take a step in the environment.
        
        Parameters:
        -----------
        actions : list
            List of actions
            
        Returns:
        --------
        tuple
            (next_states, rewards, done, info)
        """
        # Map actions to portfolio allocation
        allocation = self.actions_to_allocation(actions)
        
        # Calculate portfolio return
        portfolio_return = self.returns.iloc[self.current_step].dot(allocation)
        
        # Calculate rewards
        rewards = self._calculate_rewards(actions, portfolio_return)
        
        # Update current step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        # Get next states
        if not done:
            next_states = self._get_states()
        else:
            next_states = self.states
        
        # Update states
        self.states = next_states
        
        return next_states, rewards, done, {}
    
    def _get_states(self):
        """
        Get current states.
        
        Returns:
        --------
        list
            List of states
        """
        states = []
        
        for i in range(self.num_agents):
            # Calculate agent-specific features
            agent_features = self._calculate_agent_features(i)
            
            # Add to states
            states.append(agent_features)
        
        return states
    
    def _calculate_agent_features(self, agent_index):
        """
        Calculate agent-specific features.
        
        Parameters:
        -----------
        agent_index : int
            Agent index
            
        Returns:
        --------
        numpy.ndarray
            Agent-specific features
        """
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        portfolio_returns = self.returns.dot(portfolio_weights)
        
        # Calculate features based on agent type
        if agent_index == 0:
            # Conservative agent (focus on downside risk)
            features = self._calculate_conservative_features(portfolio_returns)
        elif agent_index == 1:
            # Balanced agent (focus on risk-adjusted returns)
            features = self._calculate_balanced_features(portfolio_returns)
        else:
            # Aggressive agent (focus on upside potential)
            features = self._calculate_aggressive_features(portfolio_returns)
        
        return features
    
    def _calculate_conservative_features(self, returns):
        """
        Calculate conservative agent features.
        
        Parameters:
        -----------
        returns : pandas.Series
            Portfolio returns
            
        Returns:
        --------
        numpy.ndarray
            Conservative agent features
        """
        # Calculate features
        features = []
        
        # Add recent returns
        start_idx = max(0, self.current_step - self.state_size//2)
        end_idx = self.current_step
        features.extend(returns.iloc[start_idx:end_idx].values)
        
        # Add downside risk measures
        for window in [21, 63, 126]:
            start_idx = max(0, self.current_step - window)
            end_idx = self.current_step
            
            # Calculate downside deviation
            downside_returns = returns.iloc[start_idx:end_idx][returns.iloc[start_idx:end_idx] < 0]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
            features.append(downside_deviation)
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + returns.iloc[start_idx:end_idx]).cumprod()
            max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
            features.append(max_drawdown)
            
            # Calculate VaR
            var_95 = np.percentile(returns.iloc[start_idx:end_idx], 5)
            features.append(var_95)
            
            # Calculate CVaR
            cvar_95 = returns.iloc[start_idx:end_idx][returns.iloc[start_idx:end_idx] <= var_95].mean()
            features.append(cvar_95)
        
        # Pad features if necessary
        if len(features) < self.state_size:
            features.extend([0] * (self.state_size - len(features)))
        
        return np.array(features[:self.state_size])
    
    def _calculate_balanced_features(self, returns):
        """
        Calculate balanced agent features.
        
        Parameters:
        -----------
        returns : pandas.Series
            Portfolio returns
            
        Returns:
        --------
        numpy.ndarray
            Balanced agent features
        """
        # Calculate features
        features = []
        
        # Add recent returns
        start_idx = max(0, self.current_step - self.state_size//2)
        end_idx = self.current_step
        features.extend(returns.iloc[start_idx:end_idx].values)
        
        # Add risk-adjusted return measures
        for window in [21, 63, 126]:
            start_idx = max(0, self.current_step - window)
            end_idx = self.current_step
            
            # Calculate mean return
            mean_return = returns.iloc[start_idx:end_idx].mean()
            features.append(mean_return)
            
            # Calculate volatility
            volatility = returns.iloc[start_idx:end_idx].std()
            features.append(volatility)
            
            # Calculate Sharpe ratio
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            features.append(sharpe_ratio)
            
            # Calculate Sortino ratio
            downside_returns = returns.iloc[start_idx:end_idx][returns.iloc[start_idx:end_idx] < 0]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
            sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
            features.append(sortino_ratio)
        
        # Pad features if necessary
        if len(features) < self.state_size:
            features.extend([0] * (self.state_size - len(features)))
        
        return np.array(features[:self.state_size])
    
    def _calculate_aggressive_features(self, returns):
        """
        Calculate aggressive agent features.
        
        Parameters:
        -----------
        returns : pandas.Series
            Portfolio returns
            
        Returns:
        --------
        numpy.ndarray
            Aggressive agent features
        """
        # Calculate features
        features = []
        
        # Add recent returns
        start_idx = max(0, self.current_step - self.state_size//2)
        end_idx = self.current_step
        features.extend(returns.iloc[start_idx:end_idx].values)
        
        # Add upside potential measures
        for window in [21, 63, 126]:
            start_idx = max(0, self.current_step - window)
            end_idx = self.current_step
            
            # Calculate mean return
            mean_return = returns.iloc[start_idx:end_idx].mean()
            features.append(mean_return)
            
            # Calculate upside deviation
            upside_returns = returns.iloc[start_idx:end_idx][returns.iloc[start_idx:end_idx] > 0]
            upside_deviation = np.std(upside_returns) if len(upside_returns) > 0 else 0
            features.append(upside_deviation)
            
            # Calculate maximum return
            max_return = returns.iloc[start_idx:end_idx].max()
            features.append(max_return)
            
            # Calculate upside potential ratio
            upside_potential = upside_returns.mean() if len(upside_returns) > 0 else 0
            downside_returns = returns.iloc[start_idx:end_idx][returns.iloc[start_idx:end_idx] < 0]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
            upside_potential_ratio = upside_potential / downside_deviation if downside_deviation > 0 else 0
            features.append(upside_potential_ratio)
        
        # Pad features if necessary
        if len(features) < self.state_size:
            features.extend([0] * (self.state_size - len(features)))
        
        return np.array(features[:self.state_size])
    
    def actions_to_allocation(self, actions):
        """
        Map actions to portfolio allocation.
        
        Parameters:
        -----------
        actions : list
            List of actions
            
        Returns:
        --------
        numpy.ndarray
            Portfolio allocation
        """
        # Initialize allocation
        allocation = np.zeros(self.num_assets)
        
        # Map actions to allocation weights
        for i, action in enumerate(actions):
            # Define allocation weights based on action
            weights = np.zeros(self.num_assets)
            
            if i == 0:
                # Conservative agent (focus on low-volatility assets)
                asset_volatilities = self.returns.std().values
                asset_ranks = np.argsort(asset_volatilities)
                num_assets_to_allocate = max(1, int(self.num_assets * (action + 1) / self.action_size))
                selected_assets = asset_ranks[:num_assets_to_allocate]
                weights[selected_assets] = 1 / num_assets_to_allocate
            elif i == 1:
                # Balanced agent (focus on risk-adjusted returns)
                asset_returns = self.returns.mean().values
                asset_volatilities = self.returns.std().values
                asset_sharpe = asset_returns / asset_volatilities
                asset_ranks = np.argsort(-asset_sharpe)  # Descending order
                num_assets_to_allocate = max(1, int(self.num_assets * (action + 1) / self.action_size))
                selected_assets = asset_ranks[:num_assets_to_allocate]
                weights[selected_assets] = 1 / num_assets_to_allocate
            else:
                # Aggressive agent (focus on high-return assets)
                asset_returns = self.returns.mean().values
                asset_ranks = np.argsort(-asset_returns)  # Descending order
                num_assets_to_allocate = max(1, int(self.num_assets * (action + 1) / self.action_size))
                selected_assets = asset_ranks[:num_assets_to_allocate]
                weights[selected_assets] = 1 / num_assets_to_allocate
            
            # Add to allocation
            allocation += weights / self.num_agents
        
        # Normalize allocation
        if np.sum(allocation) > 0:
            allocation = allocation / np.sum(allocation)
        else:
            # Equal weight if all allocations are zero
            allocation = np.ones(self.num_assets) / self.num_assets
        
        return allocation
    
    def _calculate_rewards(self, actions, portfolio_return):
        """
        Calculate rewards for each agent.
        
        Parameters:
        -----------
        actions : list
            List of actions
        portfolio_return : float
            Portfolio return
            
        Returns:
        --------
        list
            List of rewards
        """
        # Calculate rewards
        rewards = []
        
        for i in range(self.num_agents):
            if i == 0:
                # Conservative agent (focus on downside risk)
                # Higher reward for avoiding losses, lower reward for missing gains
                if portfolio_return < 0:
                    reward = -2 * portfolio_return  # Penalize losses more
                else:
                    reward = 0.5 * portfolio_return  # Reward gains less
            elif i == 1:
                # Balanced agent (focus on risk-adjusted returns)
                # Equal reward for gains and penalty for losses
                reward = portfolio_return
            else:
                # Aggressive agent (focus on upside potential)
                # Higher reward for gains, lower penalty for losses
                if portfolio_return > 0:
                    reward = 2 * portfolio_return  # Reward gains more
                else:
                    reward = 0.5 * portfolio_return  # Penalize losses less
            
            # Add to rewards
            rewards.append(reward)
        
        return rewards


class DQNAgent:
    """
    Deep Q-Network Agent for reinforcement learning.
    """
    
    def __init__(self, state_size, action_size, memory, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.001, batch_size=32):
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
        batch_size : int, default=32
            Batch size
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = memory
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
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
