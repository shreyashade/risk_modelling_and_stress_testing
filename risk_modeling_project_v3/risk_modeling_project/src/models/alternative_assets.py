"""
Alternative Assets Risk Model Module for Risk Modeling Framework.

This module provides functionality for risk modeling of alternative assets
such as cryptocurrencies, private equity, and other non-traditional investments.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class AlternativeAssetsRiskModel:
    """
    Alternative Assets Risk Model class for risk modeling.
    
    This class provides methods for risk modeling of alternative assets
    such as cryptocurrencies, private equity, and other non-traditional investments.
    """
    
    def __init__(self):
        """
        Initialize the alternative assets risk model.
        """
        self.returns = None
        self.asset_types = None
        self.correlation_network = None
        self.tail_dependencies = None
        self.regime_probabilities = None
        self.calibrated = False
    
    def calibrate(self, historical_data, asset_types=None):
        """
        Calibrate the alternative assets risk model with historical data.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data for alternative assets
        asset_types : dict, optional
            Dictionary mapping asset names to asset types
            (e.g., 'crypto', 'private_equity', 'real_estate', etc.)
        """
        self.returns = historical_data
        
        # Set asset types
        if asset_types is not None:
            self.asset_types = asset_types
        else:
            # Default: all assets are treated as 'alternative'
            self.asset_types = {col: 'alternative' for col in historical_data.columns}
        
        # Calculate correlation network
        self._calculate_correlation_network()
        
        # Calculate tail dependencies
        self._calculate_tail_dependencies()
        
        # Detect regimes
        self._detect_regimes()
        
        self.calibrated = True
    
    def _calculate_correlation_network(self):
        """
        Calculate correlation network for alternative assets.
        """
        # Calculate correlation matrix
        correlation_matrix = self.returns.corr()
        
        # Create network
        self.correlation_network = nx.Graph()
        
        # Add nodes
        for asset in self.returns.columns:
            self.correlation_network.add_node(
                asset,
                type=self.asset_types.get(asset, 'alternative')
            )
        
        # Add edges
        for i, asset_i in enumerate(self.returns.columns):
            for j, asset_j in enumerate(self.returns.columns):
                if i < j:  # Avoid duplicate edges
                    correlation = correlation_matrix.iloc[i, j]
                    if abs(correlation) > 0.3:  # Only add significant correlations
                        self.correlation_network.add_edge(
                            asset_i,
                            asset_j,
                            weight=correlation
                        )
    
    def _calculate_tail_dependencies(self):
        """
        Calculate tail dependencies for alternative assets.
        """
        # Initialize tail dependencies
        self.tail_dependencies = pd.DataFrame(
            index=self.returns.columns,
            columns=self.returns.columns,
            data=0.0
        )
        
        # Calculate tail dependencies
        for i, asset_i in enumerate(self.returns.columns):
            for j, asset_j in enumerate(self.returns.columns):
                if i < j:  # Avoid duplicate calculations
                    # Calculate lower tail dependency
                    lower_tail = self._calculate_tail_dependency(
                        self.returns[asset_i],
                        self.returns[asset_j],
                        tail='lower'
                    )
                    
                    # Calculate upper tail dependency
                    upper_tail = self._calculate_tail_dependency(
                        self.returns[asset_i],
                        self.returns[asset_j],
                        tail='upper'
                    )
                    
                    # Use maximum of lower and upper tail dependencies
                    tail_dependency = max(lower_tail, upper_tail)
                    
                    # Store tail dependency
                    self.tail_dependencies.iloc[i, j] = tail_dependency
                    self.tail_dependencies.iloc[j, i] = tail_dependency
    
    def _calculate_tail_dependency(self, x, y, tail='lower', q=0.05):
        """
        Calculate tail dependency between two return series.
        
        Parameters:
        -----------
        x : pandas.Series
            First return series
        y : pandas.Series
            Second return series
        tail : str, default='lower'
            Tail to calculate dependency for ('lower' or 'upper')
        q : float, default=0.05
            Quantile for tail dependency
            
        Returns:
        --------
        float
            Tail dependency
        """
        if tail == 'lower':
            # Calculate lower tail dependency
            x_q = np.percentile(x, 100 * q)
            y_q = np.percentile(y, 100 * q)
            
            p_x_below = np.mean(x <= x_q)
            p_y_below = np.mean(y <= y_q)
            p_both_below = np.mean((x <= x_q) & (y <= y_q))
            
            if p_x_below * p_y_below > 0:
                return p_both_below / (p_x_below * p_y_below)
            else:
                return 0.0
        else:
            # Calculate upper tail dependency
            x_q = np.percentile(x, 100 * (1 - q))
            y_q = np.percentile(y, 100 * (1 - q))
            
            p_x_above = np.mean(x >= x_q)
            p_y_above = np.mean(y >= y_q)
            p_both_above = np.mean((x >= x_q) & (y >= y_q))
            
            if p_x_above * p_y_above > 0:
                return p_both_above / (p_x_above * p_y_above)
            else:
                return 0.0
    
    def _detect_regimes(self, num_regimes=3):
        """
        Detect regimes for alternative assets.
        
        Parameters:
        -----------
        num_regimes : int, default=3
            Number of regimes to detect
        """
        # Calculate portfolio returns (equal weight)
        portfolio_weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        portfolio_returns = self.returns.dot(portfolio_weights)
        
        # Calculate rolling volatility
        rolling_vol = portfolio_returns.rolling(window=21).std()
        
        # Calculate rolling correlation
        rolling_corr = self.returns.rolling(window=63).corr()
        
        # Calculate average correlation
        avg_corr = pd.Series(index=rolling_corr.index.levels[0])
        for date in rolling_corr.index.levels[0]:
            corr_matrix = rolling_corr.loc[date]
            avg_corr[date] = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        # Combine features
        features = pd.DataFrame({
            'volatility': rolling_vol,
            'correlation': avg_corr
        }).dropna()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Cluster regimes
        kmeans = KMeans(n_clusters=num_regimes, random_state=42)
        regimes = kmeans.fit_predict(features_scaled)
        
        # Create regime probabilities
        self.regime_probabilities = pd.DataFrame(
            index=features.index,
            columns=[f'Regime {i}' for i in range(num_regimes)],
            data=0.0
        )
        
        for i, regime in enumerate(regimes):
            self.regime_probabilities.iloc[i, regime] = 1.0
    
    def calculate_var(self, confidence_level=0.95, horizon=1, method='historical'):
        """
        Calculate Value at Risk (VaR) for alternative assets.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for VaR
        horizon : int, default=1
            Forecast horizon
        method : str, default='historical'
            Method for VaR calculation ('historical', 'parametric', or 'monte_carlo')
            
        Returns:
        --------
        pandas.Series
            VaR for each asset
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Initialize VaR
        var = pd.Series(index=self.returns.columns)
        
        if method == 'historical':
            # Calculate historical VaR
            for asset in self.returns.columns:
                var[asset] = -np.percentile(
                    self.returns[asset],
                    100 * (1 - confidence_level)
                ) * np.sqrt(horizon)
        elif method == 'parametric':
            # Calculate parametric VaR
            for asset in self.returns.columns:
                mean = self.returns[asset].mean()
                std = self.returns[asset].std()
                var[asset] = -(mean * horizon + std * np.sqrt(horizon) * stats.norm.ppf(confidence_level))
        elif method == 'monte_carlo':
            # Calculate Monte Carlo VaR
            for asset in self.returns.columns:
                # Fit normal distribution
                mean = self.returns[asset].mean()
                std = self.returns[asset].std()
                
                # Generate scenarios
                np.random.seed(42)
                scenarios = np.random.normal(
                    loc=mean * horizon,
                    scale=std * np.sqrt(horizon),
                    size=10000
                )
                
                # Calculate VaR
                var[asset] = -np.percentile(scenarios, 100 * (1 - confidence_level))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return var
    
    def calculate_expected_shortfall(self, confidence_level=0.95, horizon=1, method='historical'):
        """
        Calculate Expected Shortfall (ES) for alternative assets.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for ES
        horizon : int, default=1
            Forecast horizon
        method : str, default='historical'
            Method for ES calculation ('historical', 'parametric', or 'monte_carlo')
            
        Returns:
        --------
        pandas.Series
            ES for each asset
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Initialize ES
        es = pd.Series(index=self.returns.columns)
        
        if method == 'historical':
            # Calculate historical ES
            for asset in self.returns.columns:
                var = -np.percentile(
                    self.returns[asset],
                    100 * (1 - confidence_level)
                )
                es[asset] = -self.returns[asset][self.returns[asset] <= -var].mean() * np.sqrt(horizon)
        elif method == 'parametric':
            # Calculate parametric ES
            for asset in self.returns.columns:
                mean = self.returns[asset].mean()
                std = self.returns[asset].std()
                var = -(mean * horizon + std * np.sqrt(horizon) * stats.norm.ppf(confidence_level))
                es[asset] = -(mean * horizon + std * np.sqrt(horizon) * stats.norm.pdf(stats.norm.ppf(confidence_level)) / (1 - confidence_level))
        elif method == 'monte_carlo':
            # Calculate Monte Carlo ES
            for asset in self.returns.columns:
                # Fit normal distribution
                mean = self.returns[asset].mean()
                std = self.returns[asset].std()
                
                # Generate scenarios
                np.random.seed(42)
                scenarios = np.random.normal(
                    loc=mean * horizon,
                    scale=std * np.sqrt(horizon),
                    size=10000
                )
                
                # Calculate VaR
                var = -np.percentile(scenarios, 100 * (1 - confidence_level))
                
                # Calculate ES
                es[asset] = -scenarios[scenarios <= var].mean()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return es
    
    def calculate_portfolio_risk(self, weights, confidence_level=0.95, horizon=1, method='historical'):
        """
        Calculate portfolio risk for alternative assets.
        
        Parameters:
        -----------
        weights : pandas.Series or dict
            Portfolio weights
        confidence_level : float, default=0.95
            Confidence level for risk measures
        horizon : int, default=1
            Forecast horizon
        method : str, default='historical'
            Method for risk calculation ('historical', 'parametric', or 'monte_carlo')
            
        Returns:
        --------
        dict
            Dictionary containing portfolio risk measures
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Convert weights to Series if dict
        if isinstance(weights, dict):
            weights = pd.Series(weights)
        
        # Calculate portfolio returns
        portfolio_returns = self.returns.dot(weights)
        
        # Calculate risk measures
        if method == 'historical':
            # Calculate historical risk measures
            var = -np.percentile(
                portfolio_returns,
                100 * (1 - confidence_level)
            ) * np.sqrt(horizon)
            
            es = -portfolio_returns[portfolio_returns <= -var / np.sqrt(horizon)].mean() * np.sqrt(horizon)
            
            volatility = portfolio_returns.std() * np.sqrt(horizon)
            
            max_drawdown = (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min()
        elif method == 'parametric':
            # Calculate parametric risk measures
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            
            var = -(mean * horizon + std * np.sqrt(horizon) * stats.norm.ppf(confidence_level))
            
            es = -(mean * horizon + std * np.sqrt(horizon) * stats.norm.pdf(stats.norm.ppf(confidence_level)) / (1 - confidence_level))
            
            volatility = std * np.sqrt(horizon)
            
            max_drawdown = -3 * volatility  # Approximation
        elif method == 'monte_carlo':
            # Calculate Monte Carlo risk measures
            # Fit normal distribution
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            
            # Generate scenarios
            np.random.seed(42)
            scenarios = np.random.normal(
                loc=mean * horizon,
                scale=std * np.sqrt(horizon),
                size=10000
            )
            
            # Calculate risk measures
            var = -np.percentile(scenarios, 100 * (1 - confidence_level))
            
            es = -scenarios[scenarios <= var].mean()
            
            volatility = std * np.sqrt(horizon)
            
            # Simulate path
            path = np.cumsum(np.random.normal(
                loc=mean,
                scale=std,
                size=(10000, 252)
            ), axis=1)
            
            max_drawdown = np.min(path - np.maximum.accumulate(path, axis=1), axis=1).mean()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate additional risk measures
        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        
        sortino_ratio = portfolio_returns.mean() / portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
        
        # Return risk measures
        return {
            'var': var,
            'expected_shortfall': es,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio
        }
    
    def stress_test_portfolio(self, weights, scenarios):
        """
        Stress test portfolio for alternative assets.
        
        Parameters:
        -----------
        weights : pandas.Series or dict
            Portfolio weights
        scenarios : dict
            Dictionary mapping scenario names to scenario parameters
            
        Returns:
        --------
        pandas.DataFrame
            Stress test results
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Convert weights to Series if dict
        if isinstance(weights, dict):
            weights = pd.Series(weights)
        
        # Initialize results
        results = pd.DataFrame(
            index=scenarios.keys(),
            columns=['portfolio_return', 'var_95', 'expected_shortfall_95']
        )
        
        # Run stress tests
        for scenario_name, scenario_params in scenarios.items():
            # Generate stressed returns
            stressed_returns = self._generate_stressed_returns(scenario_params)
            
            # Calculate portfolio return
            portfolio_return = stressed_returns.dot(weights).sum()
            
            # Calculate risk measures
            var_95 = -np.percentile(stressed_returns.dot(weights), 5)
            es_95 = -stressed_returns.dot(weights)[stressed_returns.dot(weights) <= -var_95].mean()
            
            # Store results
            results.loc[scenario_name, 'portfolio_return'] = portfolio_return
            results.loc[scenario_name, 'var_95'] = var_95
            results.loc[scenario_name, 'expected_shortfall_95'] = es_95
        
        return results
    
    def _generate_stressed_returns(self, scenario_params):
        """
        Generate stressed returns for a scenario.
        
        Parameters:
        -----------
        scenario_params : dict
            Scenario parameters
            
        Returns:
        --------
        pandas.DataFrame
            Stressed returns
        """
        # Copy returns
        stressed_returns = self.returns.copy()
        
        # Apply scenario parameters
        if 'volatility_multiplier' in scenario_params:
            # Increase volatility
            vol_mult = scenario_params['volatility_multiplier']
            
            for asset in stressed_returns.columns:
                mean = stressed_returns[asset].mean()
                std = stressed_returns[asset].std()
                
                stressed_returns[asset] = np.random.normal(
                    loc=mean,
                    scale=std * vol_mult,
                    size=len(stressed_returns)
                )
        
        if 'correlation_multiplier' in scenario_params:
            # Increase correlation
            corr_mult = scenario_params['correlation_multiplier']
            
            # Calculate correlation matrix
            correlation_matrix = stressed_returns.corr()
            
            # Apply multiplier
            stressed_correlation_matrix = correlation_matrix * corr_mult
            np.fill_diagonal(stressed_correlation_matrix.values, 1.0)
            
            # Ensure positive semi-definite
            eigenvalues, eigenvectors = np.linalg.eigh(stressed_correlation_matrix)
            eigenvalues = np.maximum(eigenvalues, 0)
            stressed_correlation_matrix = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)
            
            # Generate correlated returns
            means = stressed_returns.mean().values
            stds = stressed_returns.std().values
            
            # Generate uncorrelated random variables
            uncorrelated = np.random.normal(size=(len(stressed_returns), len(stressed_returns.columns)))
            
            # Apply correlation
            cholesky = np.linalg.cholesky(stressed_correlation_matrix)
            correlated = uncorrelated.dot(cholesky.T)
            
            # Apply means and stds
            for i, asset in enumerate(stressed_returns.columns):
                stressed_returns[asset] = means[i] + stds[i] * correlated[:, i]
        
        if 'market_shock' in scenario_params:
            # Apply market shock
            shock = scenario_params['market_shock']
            
            # Apply shock to all assets
            for asset in stressed_returns.columns:
                stressed_returns[asset] = stressed_returns[asset] + shock
        
        if 'asset_specific_shocks' in scenario_params:
            # Apply asset-specific shocks
            asset_shocks = scenario_params['asset_specific_shocks']
            
            for asset, shock in asset_shocks.items():
                if asset in stressed_returns.columns:
                    stressed_returns[asset] = stressed_returns[asset] + shock
        
        return stressed_returns
    
    def plot_correlation_network(self, figsize=(12, 12)):
        """
        Plot correlation network for alternative assets.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 12)
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
        
        # Get node positions
        pos = nx.spring_layout(self.correlation_network, seed=42)
        
        # Get node colors
        node_colors = []
        for node in self.correlation_network.nodes():
            if self.asset_types.get(node) == 'crypto':
                node_colors.append('blue')
            elif self.asset_types.get(node) == 'private_equity':
                node_colors.append('green')
            elif self.asset_types.get(node) == 'real_estate':
                node_colors.append('red')
            else:
                node_colors.append('gray')
        
        # Get edge colors
        edge_colors = []
        for u, v, data in self.correlation_network.edges(data=True):
            if data['weight'] > 0:
                edge_colors.append('green')
            else:
                edge_colors.append('red')
        
        # Get edge widths
        edge_widths = []
        for u, v, data in self.correlation_network.edges(data=True):
            edge_widths.append(abs(data['weight']) * 3)
        
        # Draw network
        nx.draw_networkx(
            self.correlation_network,
            pos=pos,
            node_color=node_colors,
            edge_color=edge_colors,
            width=edge_widths,
            with_labels=True,
            font_size=10,
            node_size=500,
            alpha=0.8,
            ax=ax
        )
        
        # Set title
        ax.set_title('Correlation Network for Alternative Assets')
        
        # Remove axis
        ax.axis('off')
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_tail_dependencies(self, figsize=(12, 10)):
        """
        Plot tail dependencies for alternative assets.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 10)
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
        
        # Plot tail dependencies
        sns.heatmap(
            self.tail_dependencies,
            cmap='YlOrRd',
            vmin=0,
            vmax=2,
            annot=True,
            fmt='.2f',
            ax=ax
        )
        
        # Set title and labels
        ax.set_title('Tail Dependencies for Alternative Assets')
        ax.set_xlabel('Asset')
        ax.set_ylabel('Asset')
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_regime_probabilities(self, figsize=(12, 6)):
        """
        Plot regime probabilities for alternative assets.
        
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
        
        # Plot regime probabilities
        self.regime_probabilities.plot(ax=ax)
        
        # Set title and labels
        ax.set_title('Regime Probabilities for Alternative Assets')
        ax.set_xlabel('Date')
        ax.set_ylabel('Probability')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_risk_measures(self, confidence_level=0.95, horizon=1, method='historical', figsize=(12, 8)):
        """
        Plot risk measures for alternative assets.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for risk measures
        horizon : int, default=1
            Forecast horizon
        method : str, default='historical'
            Method for risk calculation ('historical', 'parametric', or 'monte_carlo')
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        # Calculate risk measures
        var = self.calculate_var(confidence_level, horizon, method)
        es = self.calculate_expected_shortfall(confidence_level, horizon, method)
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot VaR
        var.plot(kind='bar', ax=axes[0])
        axes[0].set_title(f'Value at Risk ({confidence_level*100:.0f}%) for Alternative Assets')
        axes[0].set_xlabel('Asset')
        axes[0].set_ylabel('VaR')
        axes[0].grid(True, alpha=0.3)
        
        # Plot ES
        es.plot(kind='bar', ax=axes[1])
        axes[1].set_title(f'Expected Shortfall ({confidence_level*100:.0f}%) for Alternative Assets')
        axes[1].set_xlabel('Asset')
        axes[1].set_ylabel('ES')
        axes[1].grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_stress_test_results(self, results, figsize=(12, 8)):
        """
        Plot stress test results for alternative assets.
        
        Parameters:
        -----------
        results : pandas.DataFrame
            Stress test results
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
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Plot portfolio return
        results['portfolio_return'].plot(kind='bar', ax=axes[0])
        axes[0].set_title('Portfolio Return under Stress Scenarios')
        axes[0].set_xlabel('Scenario')
        axes[0].set_ylabel('Return')
        axes[0].grid(True, alpha=0.3)
        
        # Plot VaR
        results['var_95'].plot(kind='bar', ax=axes[1])
        axes[1].set_title('Value at Risk (95%) under Stress Scenarios')
        axes[1].set_xlabel('Scenario')
        axes[1].set_ylabel('VaR')
        axes[1].grid(True, alpha=0.3)
        
        # Plot ES
        results['expected_shortfall_95'].plot(kind='bar', ax=axes[2])
        axes[2].set_title('Expected Shortfall (95%) under Stress Scenarios')
        axes[2].set_xlabel('Scenario')
        axes[2].set_ylabel('ES')
        axes[2].grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig


class CryptoRiskModel(AlternativeAssetsRiskModel):
    """
    Cryptocurrency Risk Model class for risk modeling.
    
    This class provides methods for risk modeling of cryptocurrencies.
    """
    
    def __init__(self):
        """
        Initialize the cryptocurrency risk model.
        """
        super().__init__()
        self.blockchain_metrics = None
        self.sentiment_scores = None
    
    def calibrate(self, historical_data, blockchain_metrics=None, sentiment_scores=None, asset_types=None):
        """
        Calibrate the cryptocurrency risk model with historical data.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data for cryptocurrencies
        blockchain_metrics : pandas.DataFrame, optional
            Blockchain metrics data (e.g., transaction volume, active addresses)
        sentiment_scores : pandas.DataFrame, optional
            Sentiment scores data
        asset_types : dict, optional
            Dictionary mapping asset names to asset types
        """
        # Set blockchain metrics
        self.blockchain_metrics = blockchain_metrics
        
        # Set sentiment scores
        self.sentiment_scores = sentiment_scores
        
        # Set default asset types if not provided
        if asset_types is None:
            asset_types = {col: 'crypto' for col in historical_data.columns}
        
        # Call parent calibrate method
        super().calibrate(historical_data, asset_types)
    
    def calculate_var(self, confidence_level=0.95, horizon=1, method='historical', include_sentiment=True):
        """
        Calculate Value at Risk (VaR) for cryptocurrencies.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for VaR
        horizon : int, default=1
            Forecast horizon
        method : str, default='historical'
            Method for VaR calculation ('historical', 'parametric', or 'monte_carlo')
        include_sentiment : bool, default=True
            Whether to include sentiment in VaR calculation
            
        Returns:
        --------
        pandas.Series
            VaR for each cryptocurrency
        """
        # Calculate base VaR
        var = super().calculate_var(confidence_level, horizon, method)
        
        # Adjust VaR based on sentiment if available
        if include_sentiment and self.sentiment_scores is not None:
            for asset in var.index:
                if asset in self.sentiment_scores.columns:
                    # Get latest sentiment score
                    sentiment = self.sentiment_scores[asset].iloc[-1]
                    
                    # Adjust VaR based on sentiment
                    if sentiment < -0.5:  # Very negative sentiment
                        var[asset] *= 1.5
                    elif sentiment < 0:  # Negative sentiment
                        var[asset] *= 1.2
                    elif sentiment > 0.5:  # Very positive sentiment
                        var[asset] *= 0.8
                    elif sentiment > 0:  # Positive sentiment
                        var[asset] *= 0.9
        
        return var
    
    def plot_blockchain_metrics(self, figsize=(12, 8)):
        """
        Plot blockchain metrics for cryptocurrencies.
        
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
        
        if self.blockchain_metrics is None:
            raise ValueError("Blockchain metrics not available")
        
        # Create figure
        fig, axes = plt.subplots(len(self.blockchain_metrics.columns), 1, figsize=figsize)
        
        # Plot blockchain metrics
        for i, metric in enumerate(self.blockchain_metrics.columns):
            self.blockchain_metrics[metric].plot(ax=axes[i])
            axes[i].set_title(f'{metric} for Cryptocurrencies')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_sentiment_scores(self, figsize=(12, 6)):
        """
        Plot sentiment scores for cryptocurrencies.
        
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
        
        if self.sentiment_scores is None:
            raise ValueError("Sentiment scores not available")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot sentiment scores
        self.sentiment_scores.plot(ax=ax)
        
        # Set title and labels
        ax.set_title('Sentiment Scores for Cryptocurrencies')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig


class PrivateEquityRiskModel(AlternativeAssetsRiskModel):
    """
    Private Equity Risk Model class for risk modeling.
    
    This class provides methods for risk modeling of private equity investments.
    """
    
    def __init__(self):
        """
        Initialize the private equity risk model.
        """
        super().__init__()
        self.cash_flow_projections = None
        self.j_curves = None
    
    def calibrate(self, historical_data, cash_flow_projections=None, asset_types=None):
        """
        Calibrate the private equity risk model with historical data.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data for private equity investments
        cash_flow_projections : pandas.DataFrame, optional
            Cash flow projections data
        asset_types : dict, optional
            Dictionary mapping asset names to asset types
        """
        # Set cash flow projections
        self.cash_flow_projections = cash_flow_projections
        
        # Set default asset types if not provided
        if asset_types is None:
            asset_types = {col: 'private_equity' for col in historical_data.columns}
        
        # Call parent calibrate method
        super().calibrate(historical_data, asset_types)
        
        # Calculate J-curves
        if self.cash_flow_projections is not None:
            self._calculate_j_curves()
    
    def _calculate_j_curves(self):
        """
        Calculate J-curves for private equity investments.
        """
        # Initialize J-curves
        self.j_curves = pd.DataFrame(
            index=self.cash_flow_projections.index,
            columns=self.cash_flow_projections.columns
        )
        
        # Calculate cumulative cash flows
        for asset in self.cash_flow_projections.columns:
            self.j_curves[asset] = self.cash_flow_projections[asset].cumsum()
    
    def calculate_irr(self):
        """
        Calculate Internal Rate of Return (IRR) for private equity investments.
        
        Returns:
        --------
        pandas.Series
            IRR for each private equity investment
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        if self.cash_flow_projections is None:
            raise ValueError("Cash flow projections not available")
        
        # Initialize IRR
        irr = pd.Series(index=self.cash_flow_projections.columns)
        
        # Calculate IRR
        for asset in self.cash_flow_projections.columns:
            cash_flows = self.cash_flow_projections[asset].values
            
            # Calculate IRR
            try:
                irr[asset] = np.irr(cash_flows)
            except:
                irr[asset] = np.nan
        
        return irr
    
    def calculate_tvpi(self):
        """
        Calculate Total Value to Paid-In (TVPI) for private equity investments.
        
        Returns:
        --------
        pandas.Series
            TVPI for each private equity investment
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated")
        
        if self.cash_flow_projections is None:
            raise ValueError("Cash flow projections not available")
        
        # Initialize TVPI
        tvpi = pd.Series(index=self.cash_flow_projections.columns)
        
        # Calculate TVPI
        for asset in self.cash_flow_projections.columns:
            cash_flows = self.cash_flow_projections[asset].values
            
            # Calculate paid-in capital
            paid_in = -np.sum(cash_flows[cash_flows < 0])
            
            # Calculate distributions
            distributions = np.sum(cash_flows[cash_flows > 0])
            
            # Calculate TVPI
            if paid_in > 0:
                tvpi[asset] = distributions / paid_in
            else:
                tvpi[asset] = np.nan
        
        return tvpi
    
    def plot_j_curves(self, figsize=(12, 6)):
        """
        Plot J-curves for private equity investments.
        
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
        
        if self.j_curves is None:
            raise ValueError("J-curves not available")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot J-curves
        self.j_curves.plot(ax=ax)
        
        # Set title and labels
        ax.set_title('J-Curves for Private Equity Investments')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Cash Flow')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_cash_flow_projections(self, figsize=(12, 6)):
        """
        Plot cash flow projections for private equity investments.
        
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
        
        if self.cash_flow_projections is None:
            raise ValueError("Cash flow projections not available")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot cash flow projections
        self.cash_flow_projections.plot(kind='bar', ax=ax)
        
        # Set title and labels
        ax.set_title('Cash Flow Projections for Private Equity Investments')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cash Flow')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
