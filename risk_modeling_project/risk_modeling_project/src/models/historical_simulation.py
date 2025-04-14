"""
Historical Simulation Framework for Risk Modeling and Stress Testing

This module implements historical simulation methods for risk modeling,
including bootstrapping, filtered historical simulation, and scenario generation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Union, Optional, Tuple, Callable
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalSimulator:
    """Base class for historical simulation methods"""
    
    def __init__(self, name: str):
        """
        Initialize a HistoricalSimulator object
        
        Parameters:
        -----------
        name : str
            Name of the historical simulation method
        """
        self.name = name
    
    def simulate(self, historical_data: pd.DataFrame, n_simulations: int, 
                horizon: int, params: Optional[Dict] = None) -> np.ndarray:
        """
        Generate simulated paths using historical data
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            DataFrame with historical returns or prices
        n_simulations : int
            Number of simulations to generate
        horizon : int
            Simulation horizon (number of steps)
        params : dict, optional
            Additional parameters for the simulation
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_simulations, horizon) with simulated paths
        """
        raise NotImplementedError("Subclasses must implement this method")


class StandardHistoricalSimulation(HistoricalSimulator):
    """Standard historical simulation using random sampling with replacement"""
    
    def __init__(self):
        """Initialize a Standard Historical Simulation object"""
        super().__init__("Standard Historical Simulation")
    
    def simulate(self, historical_data: pd.DataFrame, n_simulations: int, 
                horizon: int, params: Optional[Dict] = None) -> np.ndarray:
        """
        Generate simulated paths using standard historical simulation
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            DataFrame with historical returns
        n_simulations : int
            Number of simulations to generate
        horizon : int
            Simulation horizon (number of steps)
        params : dict, optional
            Additional parameters:
            - initial_value: Initial value for the simulated paths (default: 1.0)
            - return_type: Type of returns in historical_data ('simple' or 'log', default: 'simple')
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_simulations, horizon+1) with simulated paths
        """
        if params is None:
            params = {}
            
        initial_value = params.get('initial_value', 1.0)
        return_type = params.get('return_type', 'simple')
        
        # Extract historical returns
        if isinstance(historical_data, pd.DataFrame):
            if historical_data.shape[1] > 1:
                logger.warning("Multiple columns in historical_data, using the first column")
            historical_returns = historical_data.iloc[:, 0].values
        else:
            historical_returns = historical_data
        
        # Initialize paths array
        paths = np.zeros((n_simulations, horizon + 1))
        paths[:, 0] = initial_value
        
        # Generate random indices for sampling
        indices = np.random.randint(0, len(historical_returns), (n_simulations, horizon))
        
        # Generate paths
        for t in range(1, horizon + 1):
            if return_type == 'simple':
                paths[:, t] = paths[:, t-1] * (1 + historical_returns[indices[:, t-1]])
            else:  # log returns
                paths[:, t] = paths[:, t-1] * np.exp(historical_returns[indices[:, t-1]])
        
        return paths


class BlockBootstrapSimulation(HistoricalSimulator):
    """Block bootstrap historical simulation preserving serial dependence"""
    
    def __init__(self):
        """Initialize a Block Bootstrap Simulation object"""
        super().__init__("Block Bootstrap Simulation")
    
    def simulate(self, historical_data: pd.DataFrame, n_simulations: int, 
                horizon: int, params: Optional[Dict] = None) -> np.ndarray:
        """
        Generate simulated paths using block bootstrap
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            DataFrame with historical returns
        n_simulations : int
            Number of simulations to generate
        horizon : int
            Simulation horizon (number of steps)
        params : dict, optional
            Additional parameters:
            - initial_value: Initial value for the simulated paths (default: 1.0)
            - return_type: Type of returns in historical_data ('simple' or 'log', default: 'simple')
            - block_size: Size of blocks for bootstrap (default: 10)
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_simulations, horizon+1) with simulated paths
        """
        if params is None:
            params = {}
            
        initial_value = params.get('initial_value', 1.0)
        return_type = params.get('return_type', 'simple')
        block_size = params.get('block_size', 10)
        
        # Extract historical returns
        if isinstance(historical_data, pd.DataFrame):
            if historical_data.shape[1] > 1:
                logger.warning("Multiple columns in historical_data, using the first column")
            historical_returns = historical_data.iloc[:, 0].values
        else:
            historical_returns = historical_data
        
        # Initialize paths array
        paths = np.zeros((n_simulations, horizon + 1))
        paths[:, 0] = initial_value
        
        # Calculate number of blocks needed
        n_blocks_needed = int(np.ceil(horizon / block_size))
        
        # Generate paths
        for i in range(n_simulations):
            # Generate blocks for this path
            blocks = []
            for _ in range(n_blocks_needed):
                # Randomly select starting point for block
                start_idx = np.random.randint(0, len(historical_returns) - block_size + 1)
                block = historical_returns[start_idx:start_idx + block_size]
                blocks.append(block)
            
            # Concatenate blocks
            sampled_returns = np.concatenate(blocks)[:horizon]
            
            # Generate path
            for t in range(1, horizon + 1):
                if return_type == 'simple':
                    paths[i, t] = paths[i, t-1] * (1 + sampled_returns[t-1])
                else:  # log returns
                    paths[i, t] = paths[i, t-1] * np.exp(sampled_returns[t-1])
        
        return paths


class StationnaryBootstrapSimulation(HistoricalSimulator):
    """Stationary bootstrap with random block lengths"""
    
    def __init__(self):
        """Initialize a Stationary Bootstrap Simulation object"""
        super().__init__("Stationary Bootstrap Simulation")
    
    def simulate(self, historical_data: pd.DataFrame, n_simulations: int, 
                horizon: int, params: Optional[Dict] = None) -> np.ndarray:
        """
        Generate simulated paths using stationary bootstrap
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            DataFrame with historical returns
        n_simulations : int
            Number of simulations to generate
        horizon : int
            Simulation horizon (number of steps)
        params : dict, optional
            Additional parameters:
            - initial_value: Initial value for the simulated paths (default: 1.0)
            - return_type: Type of returns in historical_data ('simple' or 'log', default: 'simple')
            - expected_block_size: Expected size of blocks (default: 10)
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_simulations, horizon+1) with simulated paths
        """
        if params is None:
            params = {}
            
        initial_value = params.get('initial_value', 1.0)
        return_type = params.get('return_type', 'simple')
        expected_block_size = params.get('expected_block_size', 10)
        
        # Probability of starting a new block
        p = 1 / expected_block_size
        
        # Extract historical returns
        if isinstance(historical_data, pd.DataFrame):
            if historical_data.shape[1] > 1:
                logger.warning("Multiple columns in historical_data, using the first column")
            historical_returns = historical_data.iloc[:, 0].values
        else:
            historical_returns = historical_data
        
        n_hist = len(historical_returns)
        
        # Initialize paths array
        paths = np.zeros((n_simulations, horizon + 1))
        paths[:, 0] = initial_value
        
        # Generate paths
        for i in range(n_simulations):
            # Generate sampled returns for this path
            sampled_returns = np.zeros(horizon)
            t = 0
            
            while t < horizon:
                # Randomly select starting point
                start_idx = np.random.randint(0, n_hist)
                
                # Generate block length from geometric distribution
                block_length = np.random.geometric(p)
                
                # Add returns from this block
                for j in range(block_length):
                    if t + j < horizon:
                        idx = (start_idx + j) % n_hist  # Wrap around if needed
                        sampled_returns[t + j] = historical_returns[idx]
                    else:
                        break
                
                t += block_length
            
            # Generate path
            for t in range(1, horizon + 1):
                if return_type == 'simple':
                    paths[i, t] = paths[i, t-1] * (1 + sampled_returns[t-1])
                else:  # log returns
                    paths[i, t] = paths[i, t-1] * np.exp(sampled_returns[t-1])
        
        return paths


class FilteredHistoricalSimulation(HistoricalSimulator):
    """Filtered historical simulation with volatility scaling"""
    
    def __init__(self):
        """Initialize a Filtered Historical Simulation object"""
        super().__init__("Filtered Historical Simulation")
    
    def simulate(self, historical_data: pd.DataFrame, n_simulations: int, 
                horizon: int, params: Optional[Dict] = None) -> np.ndarray:
        """
        Generate simulated paths using filtered historical simulation
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            DataFrame with historical returns and optionally volatility
        n_simulations : int
            Number of simulations to generate
        horizon : int
            Simulation horizon (number of steps)
        params : dict, optional
            Additional parameters:
            - initial_value: Initial value for the simulated paths (default: 1.0)
            - return_type: Type of returns in historical_data ('simple' or 'log', default: 'simple')
            - current_volatility: Current volatility estimate
            - volatility_column: Column name for historical volatility (if not provided, will be estimated)
            - returns_column: Column name for historical returns (default: first column)
            - volatility_model: Function to forecast volatility for horizon steps
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_simulations, horizon+1) with simulated paths
        """
        if params is None:
            params = {}
            
        initial_value = params.get('initial_value', 1.0)
        return_type = params.get('return_type', 'simple')
        current_volatility = params.get('current_volatility')
        volatility_column = params.get('volatility_column')
        returns_column = params.get('returns_column')
        volatility_model = params.get('volatility_model')
        
        # Extract historical returns
        if isinstance(historical_data, pd.DataFrame):
            if returns_column is None:
                returns_column = historical_data.columns[0]
            
            historical_returns = historical_data[returns_column].values
            
            # Extract or estimate historical volatility
            if volatility_column is not None:
                historical_volatility = historical_data[volatility_column].values
            else:
                # Estimate historical volatility using rolling window
                window = params.get('volatility_window', 21)
                rolling_vol = historical_data[returns_column].rolling(window=window).std().values
                historical_volatility = rolling_vol[window-1:]
                historical_returns = historical_returns[window-1:]
        else:
            historical_returns = historical_data
            # Estimate historical volatility using rolling window
            window = params.get('volatility_window', 21)
            rolling_vol = pd.Series(historical_data).rolling(window=window).std().values
            historical_volatility = rolling_vol[window-1:]
            historical_returns = historical_returns[window-1:]
        
        # If current volatility is not provided, use the last historical volatility
        if current_volatility is None:
            current_volatility = historical_volatility[-1]
        
        # Forecast volatility for horizon steps
        if volatility_model is not None:
            forecasted_volatility = volatility_model(current_volatility, horizon)
        else:
            # Simple constant volatility forecast
            forecasted_volatility = np.full(horizon, current_volatility)
        
        # Initialize paths array
        paths = np.zeros((n_simulations, horizon + 1))
        paths[:, 0] = initial_value
        
        # Generate random indices for sampling
        indices = np.random.randint(0, len(historical_returns), (n_simulations, horizon))
        
        # Generate paths with volatility scaling
        for t in range(1, horizon + 1):
            # Get historical returns for this step
            step_returns = historical_returns[indices[:, t-1]]
            
            # Get historical volatility for these returns
            step_volatility = historical_volatility[indices[:, t-1]]
            
            # Scale returns by volatility ratio
            volatility_ratio = forecasted_volatility[t-1] / step_volatility
            scaled_returns = step_returns * volatility_ratio
            
            if return_type == 'simple':
                paths[:, t] = paths[:, t-1] * (1 + scaled_returns)
            else:  # log returns
                paths[:, t] = paths[:, t-1] * np.exp(scaled_returns)
        
        return paths


class HistoricalScenarioSimulation(HistoricalSimulator):
    """Historical simulation based on specific historical scenarios"""
    
    def __init__(self):
        """Initialize a Historical Scenario Simulation object"""
        super().__init__("Historical Scenario Simulation")
    
    def simulate(self, historical_data: pd.DataFrame, n_simulations: int, 
                horizon: int, params: Optional[Dict] = None) -> np.ndarray:
        """
        Generate simulated paths using historical scenarios
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            DataFrame with historical returns, must have DatetimeIndex
        n_simulations : int
            Number of simulations to generate (ignored if scenarios are provided)
        horizon : int
            Simulation horizon (number of steps)
        params : dict, optional
            Additional parameters:
            - initial_value: Initial value for the simulated paths (default: 1.0)
            - return_type: Type of returns in historical_data ('simple' or 'log', default: 'simple')
            - scenarios: List of scenario periods as (start_date, end_date) tuples
            - scenario_weights: Weights for sampling scenarios (default: equal weights)
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_scenarios or n_simulations, horizon+1) with simulated paths
        """
        if params is None:
            params = {}
            
        initial_value = params.get('initial_value', 1.0)
        return_type = params.get('return_type', 'simple')
        scenarios = params.get('scenarios')
        scenario_weights = params.get('scenario_weights')
        
        if not isinstance(historical_data.index, pd.DatetimeIndex):
            raise ValueError("historical_data must have a DatetimeIndex for scenario-based simulation")
        
        # If scenarios are not provided, use random windows
        if scenarios is None:
            # Generate random windows of length 'horizon'
            max_start = len(historical_data) - horizon
            if max_start <= 0:
                raise ValueError("historical_data is too short for the specified horizon")
            
            start_indices = np.random.randint(0, max_start, n_simulations)
            scenarios = [(historical_data.index[i], historical_data.index[i + horizon - 1]) 
                       for i in start_indices]
        
        n_scenarios = len(scenarios)
        
        # If weights are not provided, use equal weights
        if scenario_weights is None:
            scenario_weights = np.ones(n_scenarios) / n_scenarios
        
        # Initialize paths array
        paths = np.zeros((n_scenarios, horizon + 1))
        paths[:, 0] = initial_value
        
        # Generate paths for each scenario
        for i, (start_date, end_date) in enumerate(scenarios):
            # Extract returns for this scenario
            scenario_returns = historical_data.loc[start_date:end_date].iloc[:horizon]
            
            if len(scenario_returns) < horizon:
                logger.warning(f"Scenario {i} has fewer than {horizon} data points, padding with zeros")
                padding_length = horizon - len(scenario_returns)
                padding = pd.DataFrame(0, index=range(padding_length), columns=scenario_returns.columns)
                scenario_returns = pd.concat([scenario_returns, padding])
            
            # Generate path
            for t in range(1, horizon + 1):
                if return_type == 'simple':
                    paths[i, t] = paths[i, t-1] * (1 + scenario_returns.iloc[t-1, 0])
                else:  # log returns
                    paths[i, t] = paths[i, t-1] * np.exp(scenario_returns.iloc[t-1, 0])
        
        return paths


class HistoricalRiskModel:
    """Main class for historical simulation risk modeling"""
    
    def __init__(self):
        """Initialize a Historical Risk Model"""
        self.simulators = {}
        self.results = {}
    
    def add_simulator(self, name: str, simulator: HistoricalSimulator) -> None:
        """
        Add a historical simulator
        
        Parameters:
        -----------
        name : str
            Name of the simulator
        simulator : HistoricalSimulator
            Simulator object
        """
        self.simulators[name] = simulator
    
    def run_simulation(self, simulator_name: str, historical_data: pd.DataFrame, 
                      n_simulations: int, horizon: int, params: Optional[Dict] = None) -> str:
        """
        Run a historical simulation
        
        Parameters:
        -----------
        simulator_name : str
            Name of the simulator to use
        historical_data : pd.DataFrame
            DataFrame with historical data
        n_simulations : int
            Number of simulations to generate
        horizon : int
            Simulation horizon (number of steps)
        params : dict, optional
            Additional parameters for the simulation
            
        Returns:
        --------
        str
            Result ID for retrieving simulation results
        """
        if simulator_name not in self.simulators:
            raise ValueError(f"Simulator '{simulator_name}' not found")
        
        simulator = self.simulators[simulator_name]
        
        # Run simulation
        paths = simulator.simulate(
            historical_data=historical_data,
            n_simulations=n_simulations,
            horizon=horizon,
            params=params
        )
        
        # Store results
        result_id = f"{simulator_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.results[result_id] = {
            'paths': paths,
            'simulator': simulator_name,
            'params': params
        }
        
        return result_id
    
    def calculate_var(self, result_id: str, alpha: float = 0.05, relative: bool = True) -> float:
        """
        Calculate Value at Risk (VaR) from simulation results
        
        Parameters:
        -----------
        result_id : str
            Result ID from a previous simulation
        alpha : float, default=0.05
            Confidence level (e.g., 0.05 for 95% VaR)
        relative : bool, default=True
            If True, calculate relative VaR (percentage), otherwise absolute VaR
            
        Returns:
        --------
        float
            Value at Risk
        """
        if result_id not in self.results:
            raise ValueError(f"Result '{result_id}' not found")
        
        paths = self.results[result_id]['paths']
        
        # Calculate returns from initial to final value
        if relative:
            returns = paths[:, -1] / paths[:, 0] - 1
            var = -np.percentile(returns, alpha * 100)
        else:
            final_values = paths[:, -1]
            var = paths[:, 0].mean() - np.percentile(final_values, alpha * 100)
        
        return var
    
    def calculate_expected_shortfall(self, result_id: str, alpha: float = 0.05, relative: bool = True) -> float:
        """
        Calculate Expected Shortfall (ES) from simulation results
        
        Parameters:
        -----------
        result_id : str
            Result ID from a previous simulation
        alpha : float, default=0.05
            Confidence level (e.g., 0.05 for 95% ES)
        relative : bool, default=True
            If True, calculate relative ES (percentage), otherwise absolute ES
            
        Returns:
        --------
        float
            Expected Shortfall
        """
        if result_id not in self.results:
            raise ValueError(f"Result '{result_id}' not found")
        
        paths = self.results[result_id]['paths']
        
        # Calculate returns from initial to final value
        if relative:
            returns = paths[:, -1] / paths[:, 0] - 1
            var = -np.percentile(returns, alpha * 100)
            es = -np.mean(returns[returns <= -var])
        else:
            final_values = paths[:, -1]
            var_value = np.percentile(final_values, alpha * 100)
            es = paths[:, 0].mean() - np.mean(final_values[final_values <= var_value])
        
        return es
    
    def calculate_risk_metrics(self, result_id: str) -> Dict:
        """
        Calculate various risk metrics from simulation results
        
        Parameters:
        -----------
        result_id : str
            Result ID from a previous simulation
            
        Returns:
        --------
        dict
            Dictionary with risk metrics
        """
        if result_id not in self.results:
            raise ValueError(f"Result '{result_id}' not found")
        
        paths = self.results[result_id]['paths']
        
        # Calculate returns from initial to final value
        returns = paths[:, -1] / paths[:, 0] - 1
        
        # Calculate risk metrics
        metrics = {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'median': np.median(returns),
            'min': np.min(returns),
            'max': np.max(returns),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'var_95': self.calculate_var(result_id, alpha=0.05),
            'var_99': self.calculate_var(result_id, alpha=0.01),
            'es_95': self.calculate_expected_shortfall(result_id, alpha=0.05),
            'es_99': self.calculate_expected_shortfall(result_id, alpha=0.01)
        }
        
        return metrics
    
    def plot_paths(self, result_id: str, title: str = "Simulated Paths", 
                 xlabel: str = "Time Steps", ylabel: str = "Value", 
                 figsize: Tuple[int, int] = (10, 6), n_display_paths: int = 100) -> None:
        """
        Plot simulated paths
        
        Parameters:
        -----------
        result_id : str
            Result ID from a previous simulation
        title : str, default="Simulated Paths"
            Plot title
        xlabel : str, default="Time Steps"
            X-axis label
        ylabel : str, default="Value"
            Y-axis label
        figsize : tuple, default=(10, 6)
            Figure size
        n_display_paths : int, default=100
            Number of paths to display (to avoid overcrowding)
        """
        if result_id not in self.results:
            raise ValueError(f"Result '{result_id}' not found")
        
        paths = self.results[result_id]['paths']
        
        plt.figure(figsize=figsize)
        
        n_paths = min(paths.shape[0], n_display_paths)
        indices = np.random.choice(paths.shape[0], n_paths, replace=False)
        
        for i in indices:
            plt.plot(paths[i, :], 'b-', alpha=0.1)
        
        # Plot mean path
        mean_path = np.mean(paths, axis=0)
        plt.plot(mean_path, 'r-', linewidth=2, label="Mean")
        
        # Plot 5th and 95th percentiles
        p5 = np.percentile(paths, 5, axis=0)
        p95 = np.percentile(paths, 95, axis=0)
        plt.plot(p5, 'g--', linewidth=1.5, label="5th percentile")
        plt.plot(p95, 'g--', linewidth=1.5, label="95th percentile")
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_histogram(self, result_id: str, title: str = "Return Distribution", 
                      xlabel: str = "Return", ylabel: str = "Frequency", 
                      figsize: Tuple[int, int] = (10, 6), bins: int = 50) -> None:
        """
        Plot histogram of final returns
        
        Parameters:
        -----------
        result_id : str
            Result ID from a previous simulation
        title : str, default="Return Distribution"
            Plot title
        xlabel : str, default="Return"
            X-axis label
        ylabel : str, default="Frequency"
            Y-axis label
        figsize : tuple, default=(10, 6)
            Figure size
        bins : int, default=50
            Number of histogram bins
        """
        if result_id not in self.results:
            raise ValueError(f"Result '{result_id}' not found")
        
        paths = self.results[result_id]['paths']
        
        plt.figure(figsize=figsize)
        
        # Calculate returns from initial to final value
        returns = paths[:, -1] / paths[:, 0] - 1
        
        # Plot histogram
        plt.hist(returns, bins=bins, density=True, alpha=0.7)
        
        # Plot normal distribution for comparison
        x = np.linspace(min(returns), max(returns), 1000)
        plt.plot(x, stats.norm.pdf(x, np.mean(returns), np.std(returns)), 
                'r-', linewidth=2, label="Normal Distribution")
        
        # Calculate and display risk metrics
        var_95 = self.calculate_var(result_id, alpha=0.05)
        var_99 = self.calculate_var(result_id, alpha=0.01)
        es_95 = self.calculate_expected_shortfall(result_id, alpha=0.05)
        
        # Add vertical lines for VaR and ES
        plt.axvline(-var_95, color='g', linestyle='--', linewidth=1.5, 
                   label=f"95% VaR: {var_95:.2%}")
        plt.axvline(-var_99, color='y', linestyle='--', linewidth=1.5, 
                   label=f"99% VaR: {var_99:.2%}")
        plt.axvline(-es_95, color='r', linestyle='--', linewidth=1.5, 
                   label=f"95% ES: {es_95:.2%}")
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def compare_simulators(self, result_ids: List[str], metric: str = 'var_95',
                         figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Compare results from different simulators
        
        Parameters:
        -----------
        result_ids : list of str
            Result IDs to compare
        metric : str, default='var_95'
            Risk metric to compare ('var_95', 'var_99', 'es_95', 'es_99', 'mean', 'std')
        figsize : tuple, default=(10, 6)
            Figure size
        """
        for result_id in result_ids:
            if result_id not in self.results:
                raise ValueError(f"Result '{result_id}' not found")
        
        plt.figure(figsize=figsize)
        
        labels = []
        values = []
        
        for result_id in result_ids:
            simulator_name = self.results[result_id]['simulator']
            metrics = self.calculate_risk_metrics(result_id)
            
            labels.append(simulator_name)
            values.append(metrics[metric])
        
        plt.bar(labels, values)
        plt.title(f"Comparison of {metric.upper()} across Simulators")
        plt.ylabel(metric.upper())
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()
    
    def save_results(self, result_id: str, filepath: str) -> None:
        """
        Save simulation results to file
        
        Parameters:
        -----------
        result_id : str
            Result ID to save
        filepath : str
            Path to save the results
        """
        if result_id not in self.results:
            raise ValueError(f"Result '{result_id}' not found")
        
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.results[result_id], f)
    
    @classmethod
    def load_results(cls, filepath: str) -> Dict:
        """
        Load simulation results from file
        
        Parameters:
        -----------
        filepath : str
            Path to load the results from
            
        Returns:
        --------
        dict
            Loaded results
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class HistoricalDataProcessor:
    """Class for processing historical data for simulation"""
    
    @staticmethod
    def calculate_returns(prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Parameters:
        -----------
        prices : pd.DataFrame
            DataFrame with price data
        method : str, default='simple'
            Method to calculate returns ('simple' or 'log')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with calculated returns
        """
        if method == 'simple':
            returns = prices / prices.shift(1) - 1
        elif method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError("Method must be either 'simple' or 'log'")
        
        return returns.dropna()
    
    @staticmethod
    def calculate_volatility(returns: pd.DataFrame, window: int = 21, 
                           annualize: bool = True, trading_days: int = 252) -> pd.DataFrame:
        """
        Calculate rolling volatility
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame with return data
        window : int, default=21
            Window for rolling calculation
        annualize : bool, default=True
            Whether to annualize the volatility
        trading_days : int, default=252
            Number of trading days in a year
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with calculated volatilities
        """
        # Calculate rolling standard deviation
        vol = returns.rolling(window=window).std()
        
        # Annualize if requested
        if annualize:
            vol = vol * np.sqrt(trading_days)
        
        return vol
    
    @staticmethod
    def identify_market_regimes(returns: pd.DataFrame, n_regimes: int = 2, 
                              window: int = 63) -> pd.DataFrame:
        """
        Identify market regimes based on volatility
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame with return data
        n_regimes : int, default=2
            Number of regimes to identify
        window : int, default=63
            Window for volatility calculation
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with regime labels
        """
        from sklearn.cluster import KMeans
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=window).std().dropna()
        
        # Prepare data for clustering
        X = volatility.values.reshape(-1, 1)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Create DataFrame with regime labels
        regimes = pd.DataFrame(labels, index=volatility.index, columns=['regime'])
        
        # Sort regimes by volatility (0 = low vol, 1 = high vol, etc.)
        regime_volatility = {}
        for i in range(n_regimes):
            regime_volatility[i] = np.mean(X[labels == i])
        
        # Create mapping from original labels to sorted labels
        sorted_regimes = sorted(regime_volatility.items(), key=lambda x: x[1])
        mapping = {orig: i for i, (orig, _) in enumerate(sorted_regimes)}
        
        # Apply mapping
        regimes['regime'] = regimes['regime'].map(mapping)
        
        return regimes
    
    @staticmethod
    def extract_regime_data(returns: pd.DataFrame, regimes: pd.DataFrame, 
                          regime_label: int) -> pd.DataFrame:
        """
        Extract data for a specific market regime
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame with return data
        regimes : pd.DataFrame
            DataFrame with regime labels
        regime_label : int
            Label of the regime to extract
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with returns for the specified regime
        """
        # Align indices
        aligned_returns = returns.loc[returns.index.isin(regimes.index)]
        aligned_regimes = regimes.loc[regimes.index.isin(returns.index)]
        
        # Extract data for the specified regime
        mask = aligned_regimes['regime'] == regime_label
        regime_returns = aligned_returns.loc[mask]
        
        return regime_returns
    
    @staticmethod
    def identify_stress_periods(returns: pd.DataFrame, window: int = 21, 
                              threshold: float = 2.0) -> pd.DataFrame:
        """
        Identify stress periods based on volatility
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame with return data
        window : int, default=21
            Window for volatility calculation
        threshold : float, default=2.0
            Threshold in standard deviations above mean volatility
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with stress period indicators
        """
        # Calculate rolling volatility
        volatility = returns.rolling(window=window).std().dropna()
        
        # Calculate mean and standard deviation of volatility
        mean_vol = volatility.mean()
        std_vol = volatility.std()
        
        # Identify stress periods
        stress = volatility > (mean_vol + threshold * std_vol)
        
        return stress
    
    @staticmethod
    def extract_historical_scenarios(returns: pd.DataFrame, 
                                   scenario_definitions: Dict[str, Tuple[str, str]]) -> Dict[str, pd.DataFrame]:
        """
        Extract historical scenarios
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame with return data and DatetimeIndex
        scenario_definitions : dict
            Dictionary mapping scenario names to (start_date, end_date) tuples
            
        Returns:
        --------
        dict
            Dictionary mapping scenario names to DataFrames with scenario returns
        """
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("returns must have a DatetimeIndex for scenario extraction")
        
        scenarios = {}
        
        for name, (start_date, end_date) in scenario_definitions.items():
            scenario_returns = returns.loc[start_date:end_date]
            scenarios[name] = scenario_returns
        
        return scenarios
    
    @staticmethod
    def create_custom_scenario(base_returns: pd.DataFrame, 
                             modifications: Dict[int, float]) -> pd.DataFrame:
        """
        Create a custom scenario by modifying historical returns
        
        Parameters:
        -----------
        base_returns : pd.DataFrame
            DataFrame with base return data
        modifications : dict
            Dictionary mapping time indices to return modifications
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with modified returns
        """
        custom_returns = base_returns.copy()
        
        for idx, modification in modifications.items():
            if idx < len(custom_returns):
                custom_returns.iloc[idx] = modification
        
        return custom_returns


class BootstrapAggregation:
    """Class for bootstrap aggregation (bagging) of historical simulations"""
    
    def __init__(self, n_bootstraps: int = 10, sample_size: float = 0.8):
        """
        Initialize a BootstrapAggregation object
        
        Parameters:
        -----------
        n_bootstraps : int, default=10
            Number of bootstrap samples
        sample_size : float, default=0.8
            Size of each bootstrap sample as a fraction of the original data
        """
        self.n_bootstraps = n_bootstraps
        self.sample_size = sample_size
        self.bootstrap_samples = []
        self.simulation_results = []
    
    def generate_samples(self, historical_data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Generate bootstrap samples
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            DataFrame with historical data
            
        Returns:
        --------
        list
            List of bootstrap samples
        """
        n_samples = int(len(historical_data) * self.sample_size)
        
        self.bootstrap_samples = []
        
        for _ in range(self.n_bootstraps):
            # Sample with replacement
            indices = np.random.choice(len(historical_data), n_samples, replace=True)
            sample = historical_data.iloc[indices].copy()
            self.bootstrap_samples.append(sample)
        
        return self.bootstrap_samples
    
    def run_simulations(self, simulator: HistoricalSimulator, n_simulations: int, 
                       horizon: int, params: Optional[Dict] = None) -> List[np.ndarray]:
        """
        Run simulations on bootstrap samples
        
        Parameters:
        -----------
        simulator : HistoricalSimulator
            Simulator to use
        n_simulations : int
            Number of simulations per bootstrap sample
        horizon : int
            Simulation horizon
        params : dict, optional
            Additional parameters for the simulation
            
        Returns:
        --------
        list
            List of simulation results
        """
        if not self.bootstrap_samples:
            raise ValueError("No bootstrap samples generated. Call generate_samples first.")
        
        self.simulation_results = []
        
        for sample in self.bootstrap_samples:
            paths = simulator.simulate(
                historical_data=sample,
                n_simulations=n_simulations,
                horizon=horizon,
                params=params
            )
            self.simulation_results.append(paths)
        
        return self.simulation_results
    
    def aggregate_results(self) -> np.ndarray:
        """
        Aggregate simulation results
        
        Returns:
        --------
        np.ndarray
            Aggregated simulation paths
        """
        if not self.simulation_results:
            raise ValueError("No simulation results. Call run_simulations first.")
        
        # Concatenate all paths
        all_paths = np.vstack([result for result in self.simulation_results])
        
        return all_paths
    
    def calculate_var(self, alpha: float = 0.05, relative: bool = True) -> float:
        """
        Calculate Value at Risk (VaR) from aggregated results
        
        Parameters:
        -----------
        alpha : float, default=0.05
            Confidence level (e.g., 0.05 for 95% VaR)
        relative : bool, default=True
            If True, calculate relative VaR (percentage), otherwise absolute VaR
            
        Returns:
        --------
        float
            Value at Risk
        """
        aggregated_paths = self.aggregate_results()
        
        # Calculate returns from initial to final value
        if relative:
            returns = aggregated_paths[:, -1] / aggregated_paths[:, 0] - 1
            var = -np.percentile(returns, alpha * 100)
        else:
            final_values = aggregated_paths[:, -1]
            var = aggregated_paths[:, 0].mean() - np.percentile(final_values, alpha * 100)
        
        return var
    
    def calculate_expected_shortfall(self, alpha: float = 0.05, relative: bool = True) -> float:
        """
        Calculate Expected Shortfall (ES) from aggregated results
        
        Parameters:
        -----------
        alpha : float, default=0.05
            Confidence level (e.g., 0.05 for 95% ES)
        relative : bool, default=True
            If True, calculate relative ES (percentage), otherwise absolute ES
            
        Returns:
        --------
        float
            Expected Shortfall
        """
        aggregated_paths = self.aggregate_results()
        
        # Calculate returns from initial to final value
        if relative:
            returns = aggregated_paths[:, -1] / aggregated_paths[:, 0] - 1
            var = -np.percentile(returns, alpha * 100)
            es = -np.mean(returns[returns <= -var])
        else:
            final_values = aggregated_paths[:, -1]
            var_value = np.percentile(final_values, alpha * 100)
            es = aggregated_paths[:, 0].mean() - np.mean(final_values[final_values <= var_value])
        
        return es
    
    def calculate_confidence_intervals(self, metric: str = 'var_95', 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence intervals for a risk metric
        
        Parameters:
        -----------
        metric : str, default='var_95'
            Risk metric ('var_95', 'var_99', 'es_95', 'es_99')
        confidence : float, default=0.95
            Confidence level for the interval
            
        Returns:
        --------
        tuple
            (lower_bound, upper_bound) of the confidence interval
        """
        if not self.simulation_results:
            raise ValueError("No simulation results. Call run_simulations first.")
        
        # Calculate metric for each bootstrap sample
        metric_values = []
        
        for paths in self.simulation_results:
            if metric.startswith('var'):
                alpha = float(metric.split('_')[1]) / 100
                value = self._calculate_var_for_paths(paths, alpha)
            elif metric.startswith('es'):
                alpha = float(metric.split('_')[1]) / 100
                value = self._calculate_es_for_paths(paths, alpha)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            metric_values.append(value)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(metric_values, lower_percentile)
        upper_bound = np.percentile(metric_values, upper_percentile)
        
        return lower_bound, upper_bound
    
    def _calculate_var_for_paths(self, paths: np.ndarray, alpha: float) -> float:
        """Calculate VaR for a set of paths"""
        returns = paths[:, -1] / paths[:, 0] - 1
        var = -np.percentile(returns, alpha * 100)
        return var
    
    def _calculate_es_for_paths(self, paths: np.ndarray, alpha: float) -> float:
        """Calculate ES for a set of paths"""
        returns = paths[:, -1] / paths[:, 0] - 1
        var = -np.percentile(returns, alpha * 100)
        es = -np.mean(returns[returns <= -var])
        return es
