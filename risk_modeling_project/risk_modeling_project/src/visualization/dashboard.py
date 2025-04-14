"""
Visualization and Dashboard for Risk Modeling Framework

This module implements various visualization tools and dashboard components
for presenting risk modeling results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple, Callable
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set default theme for plotly
pio.templates.default = "plotly_white"


class RiskVisualization:
    """Base class for risk visualization tools"""
    
    def __init__(self, title: str = "Risk Analysis"):
        """
        Initialize a Risk Visualization object
        
        Parameters:
        -----------
        title : str, default="Risk Analysis"
            Title for the visualization
        """
        self.title = title
        self.fig = None
    
    def plot(self, **kwargs) -> None:
        """
        Create and display the plot
        
        Parameters:
        -----------
        **kwargs : dict
            Additional parameters for plotting
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save(self, filepath: str, **kwargs) -> None:
        """
        Save the plot to a file
        
        Parameters:
        -----------
        filepath : str
            Path to save the plot
        **kwargs : dict
            Additional parameters for saving
        """
        if self.fig is None:
            raise ValueError("No plot to save. Call plot() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the plot
        self.fig.savefig(filepath, **kwargs)
        logger.info(f"Plot saved to {filepath}")


class PortfolioReturnDistribution(RiskVisualization):
    """Visualization of portfolio return distribution"""
    
    def __init__(self, title: str = "Portfolio Return Distribution"):
        """
        Initialize a Portfolio Return Distribution visualization
        
        Parameters:
        -----------
        title : str, default="Portfolio Return Distribution"
            Title for the visualization
        """
        super().__init__(title)
    
    def plot(self, returns: pd.Series, var_level: float = 0.05, 
           es_level: float = 0.05, figsize: Tuple[int, int] = (10, 6),
           bins: int = 50, **kwargs) -> None:
        """
        Create and display the plot
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        var_level : float, default=0.05
            Confidence level for VaR (e.g., 0.05 for 95% VaR)
        es_level : float, default=0.05
            Confidence level for ES (e.g., 0.05 for 95% ES)
        figsize : tuple, default=(10, 6)
            Figure size
        bins : int, default=50
            Number of bins for histogram
        **kwargs : dict
            Additional parameters for plotting
        """
        # Create figure
        self.fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        sns.histplot(returns, bins=bins, kde=True, ax=ax)
        
        # Calculate VaR and ES
        var = returns.quantile(var_level)
        es = returns[returns <= var].mean()
        
        # Add VaR and ES lines
        ax.axvline(var, color='red', linestyle='--', 
                 label=f'VaR ({(1-var_level)*100:.0f}%): {var:.2%}')
        ax.axvline(es, color='darkred', linestyle='--', 
                 label=f'ES ({(1-es_level)*100:.0f}%): {es:.2%}')
        
        # Add mean line
        mean = returns.mean()
        ax.axvline(mean, color='green', linestyle='-', 
                 label=f'Mean: {mean:.2%}')
        
        # Add normal distribution fit
        x = np.linspace(returns.min(), returns.max(), 100)
        y = stats.norm.pdf(x, returns.mean(), returns.std())
        ax.plot(x, y * len(returns) * (returns.max() - returns.min()) / bins, 
              'b--', label='Normal Fit')
        
        # Add labels and title
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.set_title(self.title)
        ax.legend()
        
        # Show plot
        plt.tight_layout()
        plt.show()


class ReturnDistributionComparison(RiskVisualization):
    """Comparison of return distributions"""
    
    def __init__(self, title: str = "Return Distribution Comparison"):
        """
        Initialize a Return Distribution Comparison visualization
        
        Parameters:
        -----------
        title : str, default="Return Distribution Comparison"
            Title for the visualization
        """
        super().__init__(title)
    
    def plot(self, returns_dict: Dict[str, pd.Series], 
           figsize: Tuple[int, int] = (12, 8), bins: int = 30,
           kde: bool = True, **kwargs) -> None:
        """
        Create and display the plot
        
        Parameters:
        -----------
        returns_dict : dict
            Dictionary mapping names to return series
        figsize : tuple, default=(12, 8)
            Figure size
        bins : int, default=30
            Number of bins for histogram
        kde : bool, default=True
            Whether to plot kernel density estimate
        **kwargs : dict
            Additional parameters for plotting
        """
        # Create figure
        self.fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histograms
        for name, returns in returns_dict.items():
            sns.histplot(returns, bins=bins, kde=kde, ax=ax, alpha=0.5, label=name)
        
        # Add labels and title
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.set_title(self.title)
        ax.legend()
        
        # Show plot
        plt.tight_layout()
        plt.show()


class TimeSeriesVisualization(RiskVisualization):
    """Visualization of time series data"""
    
    def __init__(self, title: str = "Time Series Analysis"):
        """
        Initialize a Time Series Visualization
        
        Parameters:
        -----------
        title : str, default="Time Series Analysis"
            Title for the visualization
        """
        super().__init__(title)
    
    def plot(self, data: Union[pd.Series, pd.DataFrame], 
           figsize: Tuple[int, int] = (12, 6), **kwargs) -> None:
        """
        Create and display the plot
        
        Parameters:
        -----------
        data : pd.Series or pd.DataFrame
            Time series data to plot
        figsize : tuple, default=(12, 6)
            Figure size
        **kwargs : dict
            Additional parameters for plotting
        """
        # Create figure
        self.fig, ax = plt.subplots(figsize=figsize)
        
        # Plot time series
        if isinstance(data, pd.Series):
            data.plot(ax=ax)
        else:
            data.plot(ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(self.title)
        ax.legend()
        
        # Show plot
        plt.tight_layout()
        plt.show()


class RollingRiskMetrics(RiskVisualization):
    """Visualization of rolling risk metrics"""
    
    def __init__(self, title: str = "Rolling Risk Metrics"):
        """
        Initialize a Rolling Risk Metrics visualization
        
        Parameters:
        -----------
        title : str, default="Rolling Risk Metrics"
            Title for the visualization
        """
        super().__init__(title)
    
    def plot(self, returns: pd.Series, window: int = 60, 
           figsize: Tuple[int, int] = (12, 10), **kwargs) -> None:
        """
        Create and display the plot
        
        Parameters:
        -----------
        returns : pd.Series
            Return series
        window : int, default=60
            Rolling window size
        figsize : tuple, default=(12, 10)
            Figure size
        **kwargs : dict
            Additional parameters for plotting
        """
        # Calculate rolling metrics
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        rolling_var = returns.rolling(window=window).quantile(0.05)
        
        # Calculate rolling ES
        rolling_es = pd.Series(index=returns.index, dtype=float)
        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i-window:i]
            var = window_returns.quantile(0.05)
            es = window_returns[window_returns <= var].mean()
            rolling_es.iloc[i-1] = es
        
        # Create figure
        self.fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Plot rolling mean
        rolling_mean.plot(ax=axes[0])
        axes[0].set_ylabel('Mean')
        axes[0].set_title('Rolling Mean')
        axes[0].grid(True)
        
        # Plot rolling standard deviation
        rolling_std.plot(ax=axes[1])
        axes[1].set_ylabel('Std Dev')
        axes[1].set_title('Rolling Volatility')
        axes[1].grid(True)
        
        # Plot rolling VaR
        rolling_var.plot(ax=axes[2])
        axes[2].set_ylabel('VaR')
        axes[2].set_title('Rolling VaR (95%)')
        axes[2].grid(True)
        
        # Plot rolling ES
        rolling_es.plot(ax=axes[3])
        axes[3].set_ylabel('ES')
        axes[3].set_title('Rolling ES (95%)')
        axes[3].grid(True)
        
        # Add main title
        plt.suptitle(self.title, fontsize=16)
        
        # Show plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()


class CorrelationHeatmap(RiskVisualization):
    """Visualization of correlation matrix as heatmap"""
    
    def __init__(self, title: str = "Correlation Heatmap"):
        """
        Initialize a Correlation Heatmap visualization
        
        Parameters:
        -----------
        title : str, default="Correlation Heatmap"
            Title for the visualization
        """
        super().__init__(title)
    
    def plot(self, returns: pd.DataFrame, figsize: Tuple[int, int] = (10, 8),
           cmap: str = 'coolwarm', **kwargs) -> None:
        """
        Create and display the plot
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Return series for multiple assets
        figsize : tuple, default=(10, 8)
            Figure size
        cmap : str, default='coolwarm'
            Colormap for heatmap
        **kwargs : dict
            Additional parameters for plotting
        """
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Create figure
        self.fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, ax=ax, 
                  vmin=-1, vmax=1, center=0, fmt='.2f')
        
        # Add title
        ax.set_title(self.title)
        
        # Show plot
        plt.tight_layout()
        plt.show()


class RiskContributionPie(RiskVisualization):
    """Visualization of risk contribution as pie chart"""
    
    def __init__(self, title: str = "Risk Contribution"):
        """
        Initialize a Risk Contribution Pie visualization
        
        Parameters:
        -----------
        title : str, default="Risk Contribution"
            Title for the visualization
        """
        super().__init__(title)
    
    def plot(self, returns: pd.DataFrame, weights: Optional[np.ndarray] = None,
           figsize: Tuple[int, int] = (10, 8), **kwargs) -> None:
        """
        Create and display the plot
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Return series for multiple assets
        weights : np.ndarray, optional
            Portfolio weights (if None, equal weights are used)
        figsize : tuple, default=(10, 8)
            Figure size
        **kwargs : dict
            Additional parameters for plotting
        """
        # Set default weights if not provided
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Calculate portfolio variance
        portfolio_variance = weights @ cov_matrix @ weights
        
        # Calculate marginal contribution to risk
        mcr = cov_matrix @ weights
        
        # Calculate risk contribution
        rc = weights * mcr / np.sqrt(portfolio_variance)
        
        # Calculate percentage contribution
        pct_contrib = rc / np.sum(rc) * 100
        
        # Create figure
        self.fig, ax = plt.subplots(figsize=figsize)
        
        # Plot pie chart
        ax.pie(pct_contrib, labels=returns.columns, autopct='%1.1f%%', 
             startangle=90, shadow=True)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Add title
        ax.set_title(self.title)
        
        # Show plot
        plt.tight_layout()
        plt.show()


class StressTestVisualization(RiskVisualization):
    """Visualization of stress test results"""
    
    def __init__(self, title: str = "Stress Test Results"):
        """
        Initialize a Stress Test Visualization
        
        Parameters:
        -----------
        title : str, default="Stress Test Results"
            Title for the visualization
        """
        super().__init__(title)
    
    def plot(self, original_data: pd.DataFrame, stressed_data: pd.DataFrame,
           figsize: Tuple[int, int] = (12, 8), **kwargs) -> None:
        """
        Create and display the plot
        
        Parameters:
        -----------
        original_data : pd.DataFrame
            Original time series data
        stressed_data : pd.DataFrame
            Stressed time series data
        figsize : tuple, default=(12, 8)
            Figure size
        **kwargs : dict
            Additional parameters for plotting
        """
        # Create figure
        self.fig, axes = plt.subplots(len(original_data.columns), 1, 
                                    figsize=figsize, sharex=True)
        
        # Ensure axes is a list even with a single subplot
        if len(original_data.columns) == 1:
            axes = [axes]
        
        # Plot each asset
        for i, col in enumerate(original_data.columns):
            axes[i].plot(original_data.index, original_data[col], 'b-', label='Original')
            axes[i].plot(stressed_data.index, stressed_data[col], 'r-', label='Stressed')
            axes[i].set_title(col)
            axes[i].grid(True)
            axes[i].legend()
        
        # Add main title
        plt.suptitle(self.title, fontsize=16)
        
        # Show plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()


class RegimeVisualization(RiskVisualization):
    """Visualization of regime analysis results"""
    
    def __init__(self, title: str = "Regime Analysis"):
        """
        Initialize a Regime Visualization
        
        Parameters:
        -----------
        title : str, default="Regime Analysis"
            Title for the visualization
        """
        super().__init__(title)
    
    def plot(self, returns: pd.Series, regimes: pd.Series,
           figsize: Tuple[int, int] = (12, 8), **kwargs) -> None:
        """
        Create and display the plot
        
        Parameters:
        -----------
        returns : pd.Series
            Return series
        regimes : pd.Series
            Regime classifications
        figsize : tuple, default=(12, 8)
            Figure size
        **kwargs : dict
            Additional parameters for plotting
        """
        # Create figure
        self.fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot returns
        axes[0].plot(returns.index, returns.values)
        axes[0].set_title('Returns')
        axes[0].grid(True)
        
        # Plot regimes
        unique_regimes = regimes.unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_regimes)))
        
        for i, regime in enumerate(unique_regimes):
            regime_data = returns[regimes == regime]
            axes[1].scatter(regime_data.index, regime_data.values, 
                          color=colors[i], label=f'Regime {regime}')
        
        axes[1].set_title('Regimes')
        axes[1].grid(True)
        axes[1].legend()
        
        # Add main title
        plt.suptitle(self.title, fontsize=16)
        
        # Show plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()


class EVTTailVisualization(RiskVisualization):
    """Visualization of EVT tail analysis"""
    
    def __init__(self, title: str = "EVT Tail Analysis"):
        """
        Initialize an EVT Tail Visualization
        
        Parameters:
        -----------
        title : str, default="EVT Tail Analysis"
            Title for the visualization
        """
        super().__init__(title)
    
    def plot(self, returns: pd.Series, threshold: Optional[float] = None,
           figsize: Tuple[int, int] = (12, 10), **kwargs) -> None:
        """
        Create and display the plot
        
        Parameters:
        -----------
        returns : pd.Series
            Return series
        threshold : float, optional
            Threshold for tail (if None, uses 5% quantile)
        figsize : tuple, default=(12, 10)
            Figure size
        **kwargs : dict
            Additional parameters for plotting
        """
        # Set default threshold if not provided
        if threshold is None:
            threshold = returns.quantile(0.05)
        
        # Extract tail observations
        tail_returns = returns[returns <= threshold]
        
        # Create figure
        self.fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2)
        
        # Plot 1: Full return distribution
        ax1 = plt.subplot(gs[0, 0])
        sns.histplot(returns, bins=50, kde=True, ax=ax1)
        ax1.axvline(threshold, color='red', linestyle='--', 
                  label=f'Threshold: {threshold:.2%}')
        ax1.set_title('Full Return Distribution')
        ax1.set_xlabel('Return')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Plot 2: Tail distribution
        ax2 = plt.subplot(gs[0, 1])
        sns.histplot(tail_returns, bins=30, kde=True, ax=ax2)
        ax2.set_title('Tail Distribution')
        ax2.set_xlabel('Return')
        ax2.set_ylabel('Frequency')
        
        # Plot 3: QQ plot
        ax3 = plt.subplot(gs[1, 0])
        from scipy import stats
        stats.probplot(tail_returns, dist='expon', plot=ax3)
        ax3.set_title('Exponential QQ Plot of Tail')
        
        # Plot 4: Mean excess plot
        ax4 = plt.subplot(gs[1, 1])
        
        # Calculate mean excess function
        thresholds = np.linspace(returns.min(), returns.quantile(0.2), 50)
        mean_excess = []
        
        for u in thresholds:
            exceedances = returns[returns <= u] - u
            if len(exceedances) > 0:
                mean_excess.append(np.abs(exceedances.mean()))
            else:
                mean_excess.append(np.nan)
        
        ax4.plot(thresholds, mean_excess, 'o-')
        ax4.set_title('Mean Excess Plot')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Mean Excess')
        ax4.grid(True)
        
        # Add main title
        plt.suptitle(self.title, fontsize=16)
        
        # Show plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()


class RiskDashboard:
    """Interactive dashboard for risk analysis using Plotly"""
    
    def __init__(self, title: str = "Risk Analysis Dashboard"):
        """
        Initialize a Risk Dashboard
        
        Parameters:
        -----------
        title : str, default="Risk Analysis Dashboard"
            Title for the dashboard
        """
        self.title = title
        self.fig = None
    
    def create_dashboard(self, returns: pd.DataFrame, 
                       weights: Optional[np.ndarray] = None,
                       var_level: float = 0.05,
                       height: int = 1000, width: int = 1200) -> None:
        """
        Create an interactive dashboard
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Return series for multiple assets
        weights : np.ndarray, optional
            Portfolio weights (if None, equal weights are used)
        var_level : float, default=0.05
            Confidence level for VaR (e.g., 0.05 for 95% VaR)
        height : int, default=1000
            Dashboard height
        width : int, default=1200
            Dashboard width
        """
        # Set default weights if not provided
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)
        
        # Calculate portfolio returns
        portfolio_returns = returns @ weights
        
        # Create dashboard
        self.fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Portfolio Value Over Time',
                'Return Distribution',
                'Rolling Volatility',
                'Rolling VaR',
                'Asset Correlation Heatmap',
                'Risk Contribution'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "pie"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Plot 1: Portfolio Value Over Time
        portfolio_value = (1 + portfolio_returns).cumprod()
        self.fig.add_trace(
            go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value.values,
                mode='lines',
                name='Portfolio Value'
            ),
            row=1, col=1
        )
        
        # Plot 2: Return Distribution
        self.fig.add_trace(
            go.Histogram(
                x=portfolio_returns.values,
                nbinsx=50,
                name='Returns'
            ),
            row=1, col=2
        )
        
        # Add VaR line
        var = portfolio_returns.quantile(var_level)
        self.fig.add_vline(
            x=var,
            line_width=2,
            line_dash="dash",
            line_color="red",
            row=1, col=2
        )
        
        # Add annotation for VaR
        self.fig.add_annotation(
            x=var,
            y=0.9,
            text=f"VaR ({(1-var_level)*100:.0f}%): {var:.2%}",
            showarrow=True,
            arrowhead=1,
            row=1, col=2
        )
        
        # Plot 3: Rolling Volatility
        window = min(60, len(returns) // 4)
        rolling_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        self.fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='Rolling Volatility'
            ),
            row=2, col=1
        )
        
        # Plot 4: Rolling VaR
        rolling_var = portfolio_returns.rolling(window=window).quantile(var_level) * np.sqrt(252)  # Annualized
        
        self.fig.add_trace(
            go.Scatter(
                x=rolling_var.index,
                y=rolling_var.values,
                mode='lines',
                name='Rolling VaR'
            ),
            row=2, col=2
        )
        
        # Plot 5: Asset Correlation Heatmap
        corr_matrix = returns.corr()
        
        self.fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate='%{text}',
                colorbar=dict(title='Correlation')
            ),
            row=3, col=1
        )
        
        # Plot 6: Risk Contribution Pie Chart
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Calculate portfolio variance
        portfolio_variance = weights @ cov_matrix @ weights
        
        # Calculate marginal contribution to risk
        mcr = cov_matrix @ weights
        
        # Calculate risk contribution
        rc = weights * mcr / np.sqrt(portfolio_variance)
        
        # Calculate percentage contribution
        pct_contrib = rc / np.sum(rc) * 100
        
        self.fig.add_trace(
            go.Pie(
                labels=returns.columns,
                values=pct_contrib,
                textinfo='label+percent',
                hole=0.3
            ),
            row=3, col=2
        )
        
        # Update layout
        self.fig.update_layout(
            title=self.title,
            height=height,
            width=width,
            showlegend=False
        )
        
        # Show dashboard
        self.fig.show()
    
    def save_html(self, filepath: str) -> None:
        """
        Save the dashboard as an HTML file
        
        Parameters:
        -----------
        filepath : str
            Path to save the HTML file
        """
        if self.fig is None:
            raise ValueError("No dashboard to save. Call create_dashboard() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the dashboard
        self.fig.write_html(filepath)
        logger.info(f"Dashboard saved to {filepath}")


class StressTestDashboard:
    """Interactive dashboard for stress testing using Plotly"""
    
    def __init__(self, title: str = "Stress Test Dashboard"):
        """
        Initialize a Stress Test Dashboard
        
        Parameters:
        -----------
        title : str, default="Stress Test Dashboard"
            Title for the dashboard
        """
        self.title = title
        self.fig = None
    
    def create_dashboard(self, original_data: pd.DataFrame, 
                       stressed_data: Dict[str, pd.DataFrame],
                       weights: Optional[np.ndarray] = None,
                       height: int = 1000, width: int = 1200) -> None:
        """
        Create an interactive dashboard
        
        Parameters:
        -----------
        original_data : pd.DataFrame
            Original time series data
        stressed_data : dict
            Dictionary mapping scenario names to stressed data
        weights : np.ndarray, optional
            Portfolio weights (if None, equal weights are used)
        height : int, default=1000
            Dashboard height
        width : int, default=1200
            Dashboard width
        """
        # Set default weights if not provided
        if weights is None:
            weights = np.ones(len(original_data.columns)) / len(original_data.columns)
        
        # Calculate portfolio values
        original_portfolio = (original_data @ weights)
        stressed_portfolios = {name: (data @ weights) for name, data in stressed_data.items()}
        
        # Create dashboard
        self.fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Portfolio Value Comparison',
                'Drawdown Comparison',
                'Asset Impact Heatmap',
                'Scenario Impact Comparison',
                'Return Distribution Comparison',
                'Risk Metrics Comparison'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "bar"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Plot 1: Portfolio Value Comparison
        self.fig.add_trace(
            go.Scatter(
                x=original_portfolio.index,
                y=original_portfolio.values,
                mode='lines',
                name='Original'
            ),
            row=1, col=1
        )
        
        for name, portfolio in stressed_portfolios.items():
            self.fig.add_trace(
                go.Scatter(
                    x=portfolio.index,
                    y=portfolio.values,
                    mode='lines',
                    name=name
                ),
                row=1, col=1
            )
        
        # Plot 2: Drawdown Comparison
        # Calculate drawdowns
        original_dd = self._calculate_drawdown(original_portfolio)
        stressed_dds = {name: self._calculate_drawdown(portfolio) 
                      for name, portfolio in stressed_portfolios.items()}
        
        self.fig.add_trace(
            go.Scatter(
                x=original_dd.index,
                y=original_dd.values,
                mode='lines',
                name='Original'
            ),
            row=1, col=2
        )
        
        for name, dd in stressed_dds.items():
            self.fig.add_trace(
                go.Scatter(
                    x=dd.index,
                    y=dd.values,
                    mode='lines',
                    name=name
                ),
                row=1, col=2
            )
        
        # Plot 3: Asset Impact Heatmap
        # Calculate impact for each asset and scenario
        impact_data = {}
        
        for name, data in stressed_data.items():
            # Calculate relative change at the end
            rel_change = (data.iloc[-1] / original_data.iloc[-1] - 1) * 100
            impact_data[name] = rel_change
        
        impact_df = pd.DataFrame(impact_data)
        
        self.fig.add_trace(
            go.Heatmap(
                z=impact_df.values,
                x=impact_df.columns,
                y=impact_df.index,
                colorscale='RdYlGn_r',
                zmid=0,
                text=impact_df.round(2).values,
                texttemplate='%{text:.2f}%',
                colorbar=dict(title='Impact (%)')
            ),
            row=2, col=1
        )
        
        # Plot 4: Scenario Impact Comparison
        # Calculate maximum drawdown for each scenario
        max_dds = {'Original': original_dd.min()}
        max_dds.update({name: dd.min() for name, dd in stressed_dds.items()})
        
        self.fig.add_trace(
            go.Bar(
                x=list(max_dds.keys()),
                y=list(max_dds.values()),
                name='Maximum Drawdown'
            ),
            row=2, col=2
        )
        
        # Plot 5: Return Distribution Comparison
        # Calculate returns
        original_returns = original_portfolio.pct_change().dropna()
        stressed_returns = {name: portfolio.pct_change().dropna() 
                          for name, portfolio in stressed_portfolios.items()}
        
        self.fig.add_trace(
            go.Histogram(
                x=original_returns.values,
                nbinsx=30,
                opacity=0.7,
                name='Original'
            ),
            row=3, col=1
        )
        
        for name, returns in stressed_returns.items():
            self.fig.add_trace(
                go.Histogram(
                    x=returns.values,
                    nbinsx=30,
                    opacity=0.7,
                    name=name
                ),
                row=3, col=1
            )
        
        # Plot 6: Risk Metrics Comparison
        # Calculate risk metrics
        risk_metrics = {}
        
        # Original data
        risk_metrics['Original'] = {
            'Volatility': original_returns.std() * np.sqrt(252),
            'VaR (95%)': original_returns.quantile(0.05) * np.sqrt(252),
            'ES (95%)': original_returns[original_returns <= original_returns.quantile(0.05)].mean() * np.sqrt(252)
        }
        
        # Stressed data
        for name, returns in stressed_returns.items():
            risk_metrics[name] = {
                'Volatility': returns.std() * np.sqrt(252),
                'VaR (95%)': returns.quantile(0.05) * np.sqrt(252),
                'ES (95%)': returns[returns <= returns.quantile(0.05)].mean() * np.sqrt(252)
            }
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(risk_metrics)
        
        # Plot each metric
        for i, metric in enumerate(metrics_df.index):
            self.fig.add_trace(
                go.Bar(
                    x=metrics_df.columns,
                    y=metrics_df.loc[metric],
                    name=metric
                ),
                row=3, col=2
            )
        
        # Update layout
        self.fig.update_layout(
            title=self.title,
            height=height,
            width=width,
            barmode='group'
        )
        
        # Show dashboard
        self.fig.show()
    
    def _calculate_drawdown(self, data: pd.Series) -> pd.Series:
        """
        Calculate drawdown series
        
        Parameters:
        -----------
        data : pd.Series
            Time series data
            
        Returns:
        --------
        pd.Series
            Drawdown series
        """
        # Calculate cumulative maximum
        cummax = data.cummax()
        
        # Calculate drawdown
        drawdown = (data / cummax - 1)
        
        return drawdown
    
    def save_html(self, filepath: str) -> None:
        """
        Save the dashboard as an HTML file
        
        Parameters:
        -----------
        filepath : str
            Path to save the HTML file
        """
        if self.fig is None:
            raise ValueError("No dashboard to save. Call create_dashboard() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the dashboard
        self.fig.write_html(filepath)
        logger.info(f"Dashboard saved to {filepath}")


class RegimeAnalysisDashboard:
    """Interactive dashboard for regime analysis using Plotly"""
    
    def __init__(self, title: str = "Regime Analysis Dashboard"):
        """
        Initialize a Regime Analysis Dashboard
        
        Parameters:
        -----------
        title : str, default="Regime Analysis Dashboard"
            Title for the dashboard
        """
        self.title = title
        self.fig = None
    
    def create_dashboard(self, returns: pd.DataFrame, 
                       regimes: pd.Series,
                       weights: Optional[np.ndarray] = None,
                       height: int = 1000, width: int = 1200) -> None:
        """
        Create an interactive dashboard
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Return series for multiple assets
        regimes : pd.Series
            Regime classifications
        weights : np.ndarray, optional
            Portfolio weights (if None, equal weights are used)
        height : int, default=1000
            Dashboard height
        width : int, default=1200
            Dashboard width
        """
        # Set default weights if not provided
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)
        
        # Calculate portfolio returns
        portfolio_returns = returns @ weights
        
        # Create dashboard
        self.fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Portfolio Returns by Regime',
                'Regime Distribution',
                'Return Distribution by Regime',
                'Volatility by Regime',
                'Correlation Heatmap by Regime',
                'Risk Metrics by Regime'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "bar"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Plot 1: Portfolio Returns by Regime
        unique_regimes = regimes.unique()
        colors = px.colors.qualitative.Plotly[:len(unique_regimes)]
        
        for i, regime in enumerate(unique_regimes):
            regime_returns = portfolio_returns[regimes == regime]
            
            self.fig.add_trace(
                go.Scatter(
                    x=regime_returns.index,
                    y=regime_returns.values,
                    mode='markers',
                    marker=dict(color=colors[i]),
                    name=f'Regime {regime}'
                ),
                row=1, col=1
            )
        
        # Plot 2: Regime Distribution
        regime_counts = regimes.value_counts()
        
        self.fig.add_trace(
            go.Pie(
                labels=[f'Regime {r}' for r in regime_counts.index],
                values=regime_counts.values,
                textinfo='label+percent',
                marker=dict(colors=colors)
            ),
            row=1, col=2
        )
        
        # Plot 3: Return Distribution by Regime
        for i, regime in enumerate(unique_regimes):
            regime_returns = portfolio_returns[regimes == regime]
            
            self.fig.add_trace(
                go.Histogram(
                    x=regime_returns.values,
                    nbinsx=30,
                    opacity=0.7,
                    marker=dict(color=colors[i]),
                    name=f'Regime {regime}'
                ),
                row=2, col=1
            )
        
        # Plot 4: Volatility by Regime
        regime_vols = {}
        
        for regime in unique_regimes:
            regime_returns = returns[regimes == regime]
            regime_vols[f'Regime {regime}'] = regime_returns.std() * np.sqrt(252)  # Annualized
        
        vol_df = pd.DataFrame(regime_vols)
        
        for col in vol_df.columns:
            self.fig.add_trace(
                go.Bar(
                    x=vol_df.index,
                    y=vol_df[col],
                    name=col
                ),
                row=2, col=2
            )
        
        # Plot 5: Correlation Heatmap by Regime
        # Use the first regime for the heatmap
        first_regime = unique_regimes[0]
        regime_returns = returns[regimes == first_regime]
        corr_matrix = regime_returns.corr()
        
        self.fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate='%{text}',
                colorbar=dict(title=f'Correlation (Regime {first_regime})')
            ),
            row=3, col=1
        )
        
        # Add dropdown for regime selection
        buttons = []
        
        for i, regime in enumerate(unique_regimes):
            regime_returns = returns[regimes == regime]
            corr_matrix = regime_returns.corr()
            
            buttons.append(
                dict(
                    method='update',
                    label=f'Regime {regime}',
                    args=[
                        {'z': [corr_matrix.values],
                         'text': [corr_matrix.round(2).values]},
                        {'colorbar.title': f'Correlation (Regime {regime})'}
                    ]
                )
            )
        
        self.fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction='down',
                    showactive=True,
                    x=0.1,
                    y=0.27
                )
            ]
        )
        
        # Plot 6: Risk Metrics by Regime
        # Calculate risk metrics for each regime
        risk_metrics = {}
        
        for regime in unique_regimes:
            regime_portfolio_returns = portfolio_returns[regimes == regime]
            
            risk_metrics[f'Regime {regime}'] = {
                'Mean': regime_portfolio_returns.mean() * 252,  # Annualized
                'Volatility': regime_portfolio_returns.std() * np.sqrt(252),  # Annualized
                'VaR (95%)': regime_portfolio_returns.quantile(0.05) * np.sqrt(252),  # Annualized
                'ES (95%)': regime_portfolio_returns[regime_portfolio_returns <= regime_portfolio_returns.quantile(0.05)].mean() * np.sqrt(252)  # Annualized
            }
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(risk_metrics)
        
        # Plot each metric
        for i, metric in enumerate(metrics_df.index):
            self.fig.add_trace(
                go.Bar(
                    x=metrics_df.columns,
                    y=metrics_df.loc[metric],
                    name=metric
                ),
                row=3, col=2
            )
        
        # Update layout
        self.fig.update_layout(
            title=self.title,
            height=height,
            width=width,
            barmode='group'
        )
        
        # Show dashboard
        self.fig.show()
    
    def save_html(self, filepath: str) -> None:
        """
        Save the dashboard as an HTML file
        
        Parameters:
        -----------
        filepath : str
            Path to save the HTML file
        """
        if self.fig is None:
            raise ValueError("No dashboard to save. Call create_dashboard() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the dashboard
        self.fig.write_html(filepath)
        logger.info(f"Dashboard saved to {filepath}")


class ComprehensiveRiskDashboard:
    """Comprehensive interactive dashboard for risk analysis using Plotly"""
    
    def __init__(self, title: str = "Comprehensive Risk Analysis Dashboard"):
        """
        Initialize a Comprehensive Risk Dashboard
        
        Parameters:
        -----------
        title : str, default="Comprehensive Risk Analysis Dashboard"
            Title for the dashboard
        """
        self.title = title
        self.fig = None
    
    def create_dashboard(self, returns: pd.DataFrame, 
                       stressed_data: Optional[Dict[str, pd.DataFrame]] = None,
                       regimes: Optional[pd.Series] = None,
                       evt_results: Optional[Dict] = None,
                       weights: Optional[np.ndarray] = None,
                       height: int = 1500, width: int = 1200) -> None:
        """
        Create an interactive dashboard
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Return series for multiple assets
        stressed_data : dict, optional
            Dictionary mapping scenario names to stressed returns
        regimes : pd.Series, optional
            Regime classifications
        evt_results : dict, optional
            EVT analysis results
        weights : np.ndarray, optional
            Portfolio weights (if None, equal weights are used)
        height : int, default=1500
            Dashboard height
        width : int, default=1200
            Dashboard width
        """
        # Set default weights if not provided
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)
        
        # Calculate portfolio returns
        portfolio_returns = returns @ weights
        
        # Create dashboard
        self.fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Portfolio Value Over Time',
                'Return Distribution',
                'Rolling Volatility',
                'Rolling VaR',
                'Asset Correlation Heatmap',
                'Risk Contribution',
                'Stress Test Comparison',
                'Regime Analysis',
                'EVT Tail Analysis',
                'Risk Metrics Comparison',
                'Drawdown Analysis',
                'Performance Metrics'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "heatmap"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "table"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Plot 1: Portfolio Value Over Time
        portfolio_value = (1 + portfolio_returns).cumprod()
        self.fig.add_trace(
            go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value.values,
                mode='lines',
                name='Portfolio Value'
            ),
            row=1, col=1
        )
        
        # Plot 2: Return Distribution
        self.fig.add_trace(
            go.Histogram(
                x=portfolio_returns.values,
                nbinsx=50,
                name='Returns'
            ),
            row=1, col=2
        )
        
        # Add VaR line
        var_level = 0.05
        var = portfolio_returns.quantile(var_level)
        self.fig.add_vline(
            x=var,
            line_width=2,
            line_dash="dash",
            line_color="red",
            row=1, col=2
        )
        
        # Add annotation for VaR
        self.fig.add_annotation(
            x=var,
            y=0.9,
            text=f"VaR ({(1-var_level)*100:.0f}%): {var:.2%}",
            showarrow=True,
            arrowhead=1,
            row=1, col=2
        )
        
        # Plot 3: Rolling Volatility
        window = min(60, len(returns) // 4)
        rolling_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        self.fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='Rolling Volatility'
            ),
            row=1, col=3
        )
        
        # Plot 4: Rolling VaR
        rolling_var = portfolio_returns.rolling(window=window).quantile(var_level) * np.sqrt(252)  # Annualized
        
        self.fig.add_trace(
            go.Scatter(
                x=rolling_var.index,
                y=rolling_var.values,
                mode='lines',
                name='Rolling VaR'
            ),
            row=2, col=1
        )
        
        # Plot 5: Asset Correlation Heatmap
        corr_matrix = returns.corr()
        
        self.fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate='%{text}',
                colorbar=dict(title='Correlation')
            ),
            row=2, col=2
        )
        
        # Plot 6: Risk Contribution Pie Chart
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Calculate portfolio variance
        portfolio_variance = weights @ cov_matrix @ weights
        
        # Calculate marginal contribution to risk
        mcr = cov_matrix @ weights
        
        # Calculate risk contribution
        rc = weights * mcr / np.sqrt(portfolio_variance)
        
        # Calculate percentage contribution
        pct_contrib = rc / np.sum(rc) * 100
        
        self.fig.add_trace(
            go.Pie(
                labels=returns.columns,
                values=pct_contrib,
                textinfo='label+percent',
                hole=0.3
            ),
            row=2, col=3
        )
        
        # Plot 7: Stress Test Comparison
        if stressed_data is not None:
            # Calculate portfolio values under stress
            stressed_portfolios = {}
            
            for name, data in stressed_data.items():
                # Calculate portfolio returns
                stressed_returns = data @ weights
                
                # Calculate portfolio value
                stressed_value = (1 + stressed_returns).cumprod()
                stressed_portfolios[name] = stressed_value
            
            # Plot original portfolio value
            self.fig.add_trace(
                go.Scatter(
                    x=portfolio_value.index,
                    y=portfolio_value.values,
                    mode='lines',
                    name='Original'
                ),
                row=3, col=1
            )
            
            # Plot stressed portfolio values
            for name, value in stressed_portfolios.items():
                self.fig.add_trace(
                    go.Scatter(
                        x=value.index,
                        y=value.values,
                        mode='lines',
                        name=name
                    ),
                    row=3, col=1
                )
        
        # Plot 8: Regime Analysis
        if regimes is not None:
            unique_regimes = regimes.unique()
            colors = px.colors.qualitative.Plotly[:len(unique_regimes)]
            
            for i, regime in enumerate(unique_regimes):
                regime_returns = portfolio_returns[regimes == regime]
                
                self.fig.add_trace(
                    go.Scatter(
                        x=regime_returns.index,
                        y=regime_returns.values,
                        mode='markers',
                        marker=dict(color=colors[i]),
                        name=f'Regime {regime}'
                    ),
                    row=3, col=2
                )
        
        # Plot 9: EVT Tail Analysis
        if evt_results is not None:
            # Extract tail observations
            threshold = evt_results.get('threshold', portfolio_returns.quantile(0.05))
            tail_returns = portfolio_returns[portfolio_returns <= threshold]
            
            self.fig.add_trace(
                go.Histogram(
                    x=tail_returns.values,
                    nbinsx=30,
                    name='Tail Returns'
                ),
                row=3, col=3
            )
            
            # Add threshold line
            self.fig.add_vline(
                x=threshold,
                line_width=2,
                line_dash="dash",
                line_color="red",
                row=3, col=3
            )
        
        # Plot 10: Risk Metrics Comparison
        # Calculate risk metrics
        risk_metrics = {
            'Volatility': portfolio_returns.std() * np.sqrt(252),
            'VaR (95%)': portfolio_returns.quantile(0.05) * np.sqrt(252),
            'ES (95%)': portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean() * np.sqrt(252),
            'Skewness': portfolio_returns.skew(),
            'Kurtosis': portfolio_returns.kurtosis()
        }
        
        self.fig.add_trace(
            go.Bar(
                x=list(risk_metrics.keys()),
                y=list(risk_metrics.values()),
                name='Risk Metrics'
            ),
            row=4, col=1
        )
        
        # Plot 11: Drawdown Analysis
        # Calculate drawdown
        drawdown = self._calculate_drawdown(portfolio_value)
        
        self.fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.3)'
            ),
            row=4, col=2
        )
        
        # Plot 12: Performance Metrics
        # Calculate performance metrics
        performance_metrics = {
            'Metric': [
                'Annualized Return',
                'Annualized Volatility',
                'Sharpe Ratio',
                'Maximum Drawdown',
                'Calmar Ratio',
                'Sortino Ratio',
                'VaR (95%)',
                'ES (95%)'
            ],
            'Value': [
                f"{portfolio_returns.mean() * 252:.2%}",
                f"{portfolio_returns.std() * np.sqrt(252):.2%}",
                f"{(portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)):.2f}",
                f"{drawdown.min():.2%}",
                f"{(portfolio_returns.mean() * 252) / abs(drawdown.min()):.2f}",
                f"{(portfolio_returns.mean() * 252) / (portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)):.2f}",
                f"{portfolio_returns.quantile(0.05) * np.sqrt(252):.2%}",
                f"{portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean() * np.sqrt(252):.2%}"
            ]
        }
        
        self.fig.add_trace(
            go.Table(
                header=dict(
                    values=list(performance_metrics.keys()),
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=list(performance_metrics.values()),
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=4, col=3
        )
        
        # Update layout
        self.fig.update_layout(
            title=self.title,
            height=height,
            width=width,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Show dashboard
        self.fig.show()
    
    def _calculate_drawdown(self, data: pd.Series) -> pd.Series:
        """
        Calculate drawdown series
        
        Parameters:
        -----------
        data : pd.Series
            Time series data
            
        Returns:
        --------
        pd.Series
            Drawdown series
        """
        # Calculate cumulative maximum
        cummax = data.cummax()
        
        # Calculate drawdown
        drawdown = (data / cummax - 1)
        
        return drawdown
    
    def save_html(self, filepath: str) -> None:
        """
        Save the dashboard as an HTML file
        
        Parameters:
        -----------
        filepath : str
            Path to save the HTML file
        """
        if self.fig is None:
            raise ValueError("No dashboard to save. Call create_dashboard() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the dashboard
        self.fig.write_html(filepath)
        logger.info(f"Dashboard saved to {filepath}")


# Helper functions for visualization

def plot_correlation_network(returns: pd.DataFrame, threshold: float = 0.5, 
                          figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot correlation network
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Return series for multiple assets
    threshold : float, default=0.5
        Correlation threshold for drawing edges
    figsize : tuple, default=(10, 8)
        Figure size
    """
    try:
        import networkx as nx
    except ImportError:
        logger.error("networkx is required for correlation network visualization")
        return
    
    # Calculate correlation matrix
    corr_matrix = returns.corr()
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for asset in returns.columns:
        G.add_node(asset)
    
    # Add edges
    for i in range(len(returns.columns)):
        for j in range(i+1, len(returns.columns)):
            asset_i = returns.columns[i]
            asset_j = returns.columns[j]
            correlation = corr_matrix.iloc[i, j]
            
            if abs(correlation) > threshold:
                G.add_edge(asset_i, asset_j, weight=correlation)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Set node positions
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    
    # Draw edges with colors based on correlation
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Create colormap
    cmap = plt.cm.RdBu
    vmin = -1
    vmax = 1
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2,
                         edge_color=weights, edge_cmap=cmap,
                         edge_vmin=vmin, edge_vmax=vmax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, label='Correlation')
    
    # Set title and layout
    plt.title('Asset Correlation Network')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_risk_contribution_treemap(returns: pd.DataFrame, weights: Optional[np.ndarray] = None) -> None:
    """
    Plot risk contribution treemap using Plotly
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Return series for multiple assets
    weights : np.ndarray, optional
        Portfolio weights (if None, equal weights are used)
    """
    # Set default weights if not provided
    if weights is None:
        weights = np.ones(len(returns.columns)) / len(returns.columns)
    
    # Calculate covariance matrix
    cov_matrix = returns.cov()
    
    # Calculate portfolio variance
    portfolio_variance = weights @ cov_matrix @ weights
    
    # Calculate marginal contribution to risk
    mcr = cov_matrix @ weights
    
    # Calculate risk contribution
    rc = weights * mcr / np.sqrt(portfolio_variance)
    
    # Calculate percentage contribution
    pct_contrib = rc / np.sum(rc) * 100
    
    # Create treemap
    fig = px.treemap(
        names=returns.columns,
        values=pct_contrib,
        title='Risk Contribution Treemap'
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25)
    )
    
    # Show plot
    fig.show()

def plot_efficient_frontier(returns: pd.DataFrame, risk_free_rate: float = 0.0,
                         n_portfolios: int = 1000) -> None:
    """
    Plot efficient frontier using Plotly
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Return series for multiple assets
    risk_free_rate : float, default=0.0
        Risk-free rate
    n_portfolios : int, default=1000
        Number of portfolios to simulate
    """
    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean() * 252  # Annualized
    cov_matrix = returns.cov() * 252  # Annualized
    
    # Generate random portfolios
    results = np.zeros((n_portfolios, 3))
    weights_record = np.zeros((n_portfolios, len(returns.columns)))
    
    for i in range(n_portfolios):
        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)
        weights_record[i] = weights
        
        # Calculate portfolio return and volatility
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
        
        # Calculate Sharpe ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        results[i, 0] = portfolio_volatility
        results[i, 1] = portfolio_return
        results[i, 2] = sharpe_ratio
    
    # Find portfolio with maximum Sharpe ratio
    max_sharpe_idx = np.argmax(results[:, 2])
    max_sharpe_volatility = results[max_sharpe_idx, 0]
    max_sharpe_return = results[max_sharpe_idx, 1]
    
    # Find portfolio with minimum volatility
    min_vol_idx = np.argmin(results[:, 0])
    min_vol_volatility = results[min_vol_idx, 0]
    min_vol_return = results[min_vol_idx, 1]
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add random portfolios
    fig.add_trace(
        go.Scatter(
            x=results[:, 0],
            y=results[:, 1],
            mode='markers',
            marker=dict(
                size=5,
                color=results[:, 2],
                colorscale='Viridis',
                colorbar=dict(title='Sharpe Ratio')
            ),
            text=[f"Sharpe: {sharpe:.2f}" for sharpe in results[:, 2]],
            name='Portfolios'
        )
    )
    
    # Add maximum Sharpe ratio portfolio
    fig.add_trace(
        go.Scatter(
            x=[max_sharpe_volatility],
            y=[max_sharpe_return],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='star'
            ),
            name='Maximum Sharpe Ratio'
        )
    )
    
    # Add minimum volatility portfolio
    fig.add_trace(
        go.Scatter(
            x=[min_vol_volatility],
            y=[min_vol_return],
            mode='markers',
            marker=dict(
                size=15,
                color='green',
                symbol='star'
            ),
            name='Minimum Volatility'
        )
    )
    
    # Add capital market line
    x_range = np.linspace(0, max(results[:, 0]) * 1.2, 100)
    y_values = risk_free_rate + (max_sharpe_return - risk_free_rate) / max_sharpe_volatility * x_range
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_values,
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Capital Market Line'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Efficient Frontier',
        xaxis=dict(title='Annualized Volatility'),
        yaxis=dict(title='Annualized Return'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # Show plot
    fig.show()

def create_risk_report(returns: pd.DataFrame, weights: Optional[np.ndarray] = None,
                     output_file: str = 'risk_report.html') -> None:
    """
    Create a comprehensive risk report
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Return series for multiple assets
    weights : np.ndarray, optional
        Portfolio weights (if None, equal weights are used)
    output_file : str, default='risk_report.html'
        Output file path
    """
    # Set default weights if not provided
    if weights is None:
        weights = np.ones(len(returns.columns)) / len(returns.columns)
    
    # Calculate portfolio returns
    portfolio_returns = returns @ weights
    
    # Create dashboard
    dashboard = RiskDashboard("Risk Analysis Dashboard")
    dashboard.create_dashboard(returns, weights)
    
    # Save dashboard
    dashboard.save_html(output_file)
    
    logger.info(f"Risk report saved to {output_file}")

def create_stress_test_report(original_data: pd.DataFrame, 
                           stressed_data: Dict[str, pd.DataFrame],
                           weights: Optional[np.ndarray] = None,
                           output_file: str = 'stress_test_report.html') -> None:
    """
    Create a comprehensive stress test report
    
    Parameters:
    -----------
    original_data : pd.DataFrame
        Original time series data
    stressed_data : dict
        Dictionary mapping scenario names to stressed data
    weights : np.ndarray, optional
        Portfolio weights (if None, equal weights are used)
    output_file : str, default='stress_test_report.html'
        Output file path
    """
    # Create dashboard
    dashboard = StressTestDashboard("Stress Test Dashboard")
    dashboard.create_dashboard(original_data, stressed_data, weights)
    
    # Save dashboard
    dashboard.save_html(output_file)
    
    logger.info(f"Stress test report saved to {output_file}")

def create_regime_analysis_report(returns: pd.DataFrame, 
                               regimes: pd.Series,
                               weights: Optional[np.ndarray] = None,
                               output_file: str = 'regime_analysis_report.html') -> None:
    """
    Create a comprehensive regime analysis report
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Return series for multiple assets
    regimes : pd.Series
        Regime classifications
    weights : np.ndarray, optional
        Portfolio weights (if None, equal weights are used)
    output_file : str, default='regime_analysis_report.html'
        Output file path
    """
    # Create dashboard
    dashboard = RegimeAnalysisDashboard("Regime Analysis Dashboard")
    dashboard.create_dashboard(returns, regimes, weights)
    
    # Save dashboard
    dashboard.save_html(output_file)
    
    logger.info(f"Regime analysis report saved to {output_file}")

def create_comprehensive_report(returns: pd.DataFrame, 
                             stressed_data: Optional[Dict[str, pd.DataFrame]] = None,
                             regimes: Optional[pd.Series] = None,
                             evt_results: Optional[Dict] = None,
                             weights: Optional[np.ndarray] = None,
                             output_file: str = 'comprehensive_report.html') -> None:
    """
    Create a comprehensive risk analysis report
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Return series for multiple assets
    stressed_data : dict, optional
        Dictionary mapping scenario names to stressed returns
    regimes : pd.Series, optional
        Regime classifications
    evt_results : dict, optional
        EVT analysis results
    weights : np.ndarray, optional
        Portfolio weights (if None, equal weights are used)
    output_file : str, default='comprehensive_report.html'
        Output file path
    """
    # Create dashboard
    dashboard = ComprehensiveRiskDashboard("Comprehensive Risk Analysis Dashboard")
    dashboard.create_dashboard(returns, stressed_data, regimes, evt_results, weights)
    
    # Save dashboard
    dashboard.save_html(output_file)
    
    logger.info(f"Comprehensive report saved to {output_file}")
