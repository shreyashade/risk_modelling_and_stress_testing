"""
Monte Carlo Simulation Framework for Risk Modeling and Stress Testing

This module implements various Monte Carlo simulation methods for risk modeling,
including different stochastic processes, correlation models, and scenario generation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky
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


class StochasticProcess:
    """Base class for stochastic processes used in Monte Carlo simulations"""
    
    def __init__(self, name: str):
        """
        Initialize a StochasticProcess object
        
        Parameters:
        -----------
        name : str
            Name of the stochastic process
        """
        self.name = name
    
    def generate_paths(self, initial_value: float, time_steps: int, n_paths: int, dt: float,
                      params: Dict) -> np.ndarray:
        """
        Generate sample paths for the stochastic process
        
        Parameters:
        -----------
        initial_value : float
            Initial value of the process
        time_steps : int
            Number of time steps
        n_paths : int
            Number of paths to generate
        dt : float
            Time step size
        params : dict
            Parameters for the process
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_paths, time_steps+1) with generated paths
        """
        raise NotImplementedError("Subclasses must implement this method")


class GeometricBrownianMotion(StochasticProcess):
    """Geometric Brownian Motion process"""
    
    def __init__(self):
        """Initialize a Geometric Brownian Motion process"""
        super().__init__("Geometric Brownian Motion")
    
    def generate_paths(self, initial_value: float, time_steps: int, n_paths: int, dt: float,
                      params: Dict) -> np.ndarray:
        """
        Generate sample paths for Geometric Brownian Motion
        
        Parameters:
        -----------
        initial_value : float
            Initial value of the process
        time_steps : int
            Number of time steps
        n_paths : int
            Number of paths to generate
        dt : float
            Time step size
        params : dict
            Parameters for the process:
            - mu: Drift parameter (annualized)
            - sigma: Volatility parameter (annualized)
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_paths, time_steps+1) with generated paths
        """
        mu = params.get('mu', 0)
        sigma = params.get('sigma', 0.2)
        
        # Initialize array for paths
        paths = np.zeros((n_paths, time_steps + 1))
        paths[:, 0] = initial_value
        
        # Generate random normal variables
        Z = np.random.normal(0, 1, (n_paths, time_steps))
        
        # Generate paths
        for t in range(1, time_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
        
        return paths


class JumpDiffusionProcess(StochasticProcess):
    """Jump Diffusion process (Merton model)"""
    
    def __init__(self):
        """Initialize a Jump Diffusion process"""
        super().__init__("Jump Diffusion Process")
    
    def generate_paths(self, initial_value: float, time_steps: int, n_paths: int, dt: float,
                      params: Dict) -> np.ndarray:
        """
        Generate sample paths for Jump Diffusion process
        
        Parameters:
        -----------
        initial_value : float
            Initial value of the process
        time_steps : int
            Number of time steps
        n_paths : int
            Number of paths to generate
        dt : float
            Time step size
        params : dict
            Parameters for the process:
            - mu: Drift parameter (annualized)
            - sigma: Volatility parameter (annualized)
            - lambda: Jump intensity (average number of jumps per year)
            - mu_j: Average jump size
            - sigma_j: Jump size volatility
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_paths, time_steps+1) with generated paths
        """
        mu = params.get('mu', 0)
        sigma = params.get('sigma', 0.2)
        lambda_j = params.get('lambda', 1)  # Jump intensity
        mu_j = params.get('mu_j', -0.1)     # Average jump size
        sigma_j = params.get('sigma_j', 0.1) # Jump size volatility
        
        # Initialize array for paths
        paths = np.zeros((n_paths, time_steps + 1))
        paths[:, 0] = initial_value
        
        # Generate diffusion component
        Z = np.random.normal(0, 1, (n_paths, time_steps))
        
        # Generate jump component
        for t in range(1, time_steps + 1):
            # Diffusion component
            diffusion = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
            
            # Jump component
            jump_count = np.random.poisson(lambda_j * dt, n_paths)
            jump_sizes = np.zeros(n_paths)
            
            for i in range(n_paths):
                if jump_count[i] > 0:
                    jumps = np.random.normal(mu_j, sigma_j, jump_count[i])
                    jump_sizes[i] = np.sum(jumps)
            
            # Combine diffusion and jump components
            paths[:, t] = paths[:, t-1] * np.exp(diffusion + jump_sizes)
        
        return paths


class HestonModel(StochasticProcess):
    """Heston stochastic volatility model"""
    
    def __init__(self):
        """Initialize a Heston model"""
        super().__init__("Heston Model")
    
    def generate_paths(self, initial_value: float, time_steps: int, n_paths: int, dt: float,
                      params: Dict) -> np.ndarray:
        """
        Generate sample paths for Heston model
        
        Parameters:
        -----------
        initial_value : float
            Initial value of the process
        time_steps : int
            Number of time steps
        n_paths : int
            Number of paths to generate
        dt : float
            Time step size
        params : dict
            Parameters for the process:
            - mu: Drift parameter (annualized)
            - kappa: Mean reversion speed for variance
            - theta: Long-term variance
            - sigma_v: Volatility of variance
            - rho: Correlation between asset returns and variance
            - v0: Initial variance
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_paths, time_steps+1) with generated paths
        """
        mu = params.get('mu', 0)
        kappa = params.get('kappa', 2)
        theta = params.get('theta', 0.04)
        sigma_v = params.get('sigma_v', 0.3)
        rho = params.get('rho', -0.7)
        v0 = params.get('v0', 0.04)
        
        # Initialize arrays for price and variance paths
        price_paths = np.zeros((n_paths, time_steps + 1))
        variance_paths = np.zeros((n_paths, time_steps + 1))
        
        price_paths[:, 0] = initial_value
        variance_paths[:, 0] = v0
        
        # Generate correlated random variables
        for t in range(1, time_steps + 1):
            Z1 = np.random.normal(0, 1, n_paths)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n_paths)
            
            # Update variance (with full truncation to ensure positivity)
            variance_paths[:, t] = np.maximum(
                variance_paths[:, t-1] + kappa * (theta - np.maximum(0, variance_paths[:, t-1])) * dt
                + sigma_v * np.sqrt(np.maximum(0, variance_paths[:, t-1]) * dt) * Z2,
                0
            )
            
            # Update price
            price_paths[:, t] = price_paths[:, t-1] * np.exp(
                (mu - 0.5 * variance_paths[:, t-1]) * dt
                + np.sqrt(variance_paths[:, t-1] * dt) * Z1
            )
        
        return price_paths


class GARCHProcess(StochasticProcess):
    """GARCH(1,1) process for volatility modeling"""
    
    def __init__(self):
        """Initialize a GARCH process"""
        super().__init__("GARCH Process")
    
    def generate_paths(self, initial_value: float, time_steps: int, n_paths: int, dt: float,
                      params: Dict) -> np.ndarray:
        """
        Generate sample paths for GARCH(1,1) process
        
        Parameters:
        -----------
        initial_value : float
            Initial value of the process
        time_steps : int
            Number of time steps
        n_paths : int
            Number of paths to generate
        dt : float
            Time step size
        params : dict
            Parameters for the process:
            - mu: Drift parameter (annualized)
            - omega: Constant term in variance equation
            - alpha: ARCH parameter (impact of past squared returns)
            - beta: GARCH parameter (impact of past variance)
            - initial_variance: Initial variance value
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_paths, time_steps+1) with generated paths
        """
        mu = params.get('mu', 0)
        omega = params.get('omega', 0.00001)
        alpha = params.get('alpha', 0.1)
        beta = params.get('beta', 0.8)
        initial_variance = params.get('initial_variance', 0.04)
        
        # Initialize arrays for price and variance paths
        price_paths = np.zeros((n_paths, time_steps + 1))
        variance_paths = np.zeros((n_paths, time_steps + 1))
        
        price_paths[:, 0] = initial_value
        variance_paths[:, 0] = initial_variance
        
        # Generate paths
        for t in range(1, time_steps + 1):
            # Generate random shocks
            Z = np.random.normal(0, 1, n_paths)
            
            # Calculate returns
            returns = mu * dt + np.sqrt(variance_paths[:, t-1] * dt) * Z
            
            # Update variance using GARCH(1,1) formula
            variance_paths[:, t] = (omega 
                                  + alpha * (returns**2) / dt
                                  + beta * variance_paths[:, t-1])
            
            # Update price
            price_paths[:, t] = price_paths[:, t-1] * np.exp(returns)
        
        return price_paths


class CorrelationModel:
    """Base class for correlation models used in Monte Carlo simulations"""
    
    def __init__(self, name: str):
        """
        Initialize a CorrelationModel object
        
        Parameters:
        -----------
        name : str
            Name of the correlation model
        """
        self.name = name
    
    def generate_correlated_returns(self, n_assets: int, time_steps: int, n_paths: int,
                                   params: Dict) -> np.ndarray:
        """
        Generate correlated returns for multiple assets
        
        Parameters:
        -----------
        n_assets : int
            Number of assets
        time_steps : int
            Number of time steps
        n_paths : int
            Number of paths to generate
        params : dict
            Parameters for the model
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_paths, time_steps, n_assets) with correlated returns
        """
        raise NotImplementedError("Subclasses must implement this method")


class ConstantCorrelationModel(CorrelationModel):
    """Constant correlation model using Cholesky decomposition"""
    
    def __init__(self):
        """Initialize a Constant Correlation model"""
        super().__init__("Constant Correlation Model")
    
    def generate_correlated_returns(self, n_assets: int, time_steps: int, n_paths: int,
                                   params: Dict) -> np.ndarray:
        """
        Generate correlated returns using constant correlation matrix
        
        Parameters:
        -----------
        n_assets : int
            Number of assets
        time_steps : int
            Number of time steps
        n_paths : int
            Number of paths to generate
        params : dict
            Parameters for the model:
            - correlation_matrix: Correlation matrix (n_assets x n_assets)
            - volatilities: Annualized volatilities for each asset
            - means: Annualized mean returns for each asset
            - dt: Time step size
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_paths, time_steps, n_assets) with correlated returns
        """
        correlation_matrix = params.get('correlation_matrix')
        volatilities = params.get('volatilities')
        means = params.get('means')
        dt = params.get('dt', 1/252)  # Default to daily
        
        if correlation_matrix is None or volatilities is None or means is None:
            raise ValueError("Correlation matrix, volatilities, and means must be provided")
        
        if correlation_matrix.shape != (n_assets, n_assets):
            raise ValueError(f"Correlation matrix shape {correlation_matrix.shape} does not match n_assets {n_assets}")
        
        if len(volatilities) != n_assets or len(means) != n_assets:
            raise ValueError("Volatilities and means must have length equal to n_assets")
        
        # Create covariance matrix from correlation matrix and volatilities
        volatilities_diag = np.diag(volatilities)
        covariance_matrix = volatilities_diag @ correlation_matrix @ volatilities_diag
        
        # Cholesky decomposition of covariance matrix
        try:
            chol_matrix = cholesky(covariance_matrix, lower=True)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use nearest positive definite matrix
            logger.warning("Correlation matrix is not positive definite. Using nearest PD matrix.")
            covariance_matrix = self._nearest_positive_definite(covariance_matrix)
            chol_matrix = cholesky(covariance_matrix, lower=True)
        
        # Generate independent normal random variables
        Z = np.random.normal(0, 1, (n_paths, time_steps, n_assets))
        
        # Transform to correlated returns
        correlated_Z = np.zeros((n_paths, time_steps, n_assets))
        for p in range(n_paths):
            for t in range(time_steps):
                correlated_Z[p, t, :] = chol_matrix @ Z[p, t, :]
        
        # Calculate returns with drift and volatility
        returns = np.zeros((n_paths, time_steps, n_assets))
        for i in range(n_assets):
            returns[:, :, i] = means[i] * dt + correlated_Z[:, :, i] * np.sqrt(dt)
        
        return returns
    
    def _nearest_positive_definite(self, A: np.ndarray) -> np.ndarray:
        """
        Find the nearest positive definite matrix to A
        
        Parameters:
        -----------
        A : np.ndarray
            Input matrix
            
        Returns:
        --------
        np.ndarray
            Nearest positive definite matrix
        """
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        
        if self._is_positive_definite(A3):
            return A3
        
        # If still not positive definite, add a small diagonal matrix
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self._is_positive_definite(A3):
            A3 += I * k * spacing
            k *= 2
        
        return A3
    
    def _is_positive_definite(self, A: np.ndarray) -> bool:
        """
        Check if a matrix is positive definite
        
        Parameters:
        -----------
        A : np.ndarray
            Input matrix
            
        Returns:
        --------
        bool
            True if positive definite, False otherwise
        """
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False


class DynamicCorrelationModel(CorrelationModel):
    """Dynamic correlation model with time-varying correlations"""
    
    def __init__(self):
        """Initialize a Dynamic Correlation model"""
        super().__init__("Dynamic Correlation Model")
    
    def generate_correlated_returns(self, n_assets: int, time_steps: int, n_paths: int,
                                   params: Dict) -> np.ndarray:
        """
        Generate correlated returns using dynamic correlation matrix
        
        Parameters:
        -----------
        n_assets : int
            Number of assets
        time_steps : int
            Number of time steps
        n_paths : int
            Number of paths to generate
        params : dict
            Parameters for the model:
            - correlation_matrices: List of correlation matrices for each time step
            - volatilities: List of volatilities for each asset and time step
            - means: Annualized mean returns for each asset
            - dt: Time step size
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_paths, time_steps, n_assets) with correlated returns
        """
        correlation_matrices = params.get('correlation_matrices')
        volatilities = params.get('volatilities')
        means = params.get('means')
        dt = params.get('dt', 1/252)  # Default to daily
        
        if correlation_matrices is None or volatilities is None or means is None:
            raise ValueError("Correlation matrices, volatilities, and means must be provided")
        
        if len(correlation_matrices) != time_steps:
            raise ValueError(f"Number of correlation matrices {len(correlation_matrices)} does not match time_steps {time_steps}")
        
        if volatilities.shape != (time_steps, n_assets):
            raise ValueError(f"Volatilities shape {volatilities.shape} does not match (time_steps, n_assets) {(time_steps, n_assets)}")
        
        if len(means) != n_assets:
            raise ValueError("Means must have length equal to n_assets")
        
        # Initialize returns array
        returns = np.zeros((n_paths, time_steps, n_assets))
        
        # Generate correlated returns for each time step
        for t in range(time_steps):
            # Create covariance matrix from correlation matrix and volatilities at time t
            volatilities_diag = np.diag(volatilities[t, :])
            covariance_matrix = volatilities_diag @ correlation_matrices[t] @ volatilities_diag
            
            # Cholesky decomposition of covariance matrix
            try:
                chol_matrix = cholesky(covariance_matrix, lower=True)
            except np.linalg.LinAlgError:
                # If Cholesky fails, use nearest positive definite matrix
                logger.warning(f"Correlation matrix at time {t} is not positive definite. Using nearest PD matrix.")
                covariance_matrix = self._nearest_positive_definite(covariance_matrix)
                chol_matrix = cholesky(covariance_matrix, lower=True)
            
            # Generate independent normal random variables
            Z = np.random.normal(0, 1, (n_paths, n_assets))
            
            # Transform to correlated returns
            for p in range(n_paths):
                correlated_Z = chol_matrix @ Z[p, :]
                returns[p, t, :] = means * dt + correlated_Z * np.sqrt(dt)
        
        return returns
    
    def _nearest_positive_definite(self, A: np.ndarray) -> np.ndarray:
        """
        Find the nearest positive definite matrix to A
        
        Parameters:
        -----------
        A : np.ndarray
            Input matrix
            
        Returns:
        --------
        np.ndarray
            Nearest positive definite matrix
        """
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        
        if self._is_positive_definite(A3):
            return A3
        
        # If still not positive definite, add a small diagonal matrix
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self._is_positive_definite(A3):
            A3 += I * k * spacing
            k *= 2
        
        return A3
    
    def _is_positive_definite(self, A: np.ndarray) -> bool:
        """
        Check if a matrix is positive definite
        
        Parameters:
        -----------
        A : np.ndarray
            Input matrix
            
        Returns:
        --------
        bool
            True if positive definite, False otherwise
        """
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False


class CopulaCorrelationModel(CorrelationModel):
    """Correlation model using copulas for more flexible dependence structures"""
    
    def __init__(self, copula_type: str = 'gaussian'):
        """
        Initialize a Copula Correlation model
        
        Parameters:
        -----------
        copula_type : str, default='gaussian'
            Type of copula to use ('gaussian', 't', 'clayton', 'gumbel', 'frank')
        """
        super().__init__(f"{copula_type.capitalize()} Copula Model")
        self.copula_type = copula_type
    
    def generate_correlated_returns(self, n_assets: int, time_steps: int, n_paths: int,
                                   params: Dict) -> np.ndarray:
        """
        Generate correlated returns using copula
        
        Parameters:
        -----------
        n_assets : int
            Number of assets
        time_steps : int
            Number of time steps
        n_paths : int
            Number of paths to generate
        params : dict
            Parameters for the model:
            - correlation_matrix: Correlation matrix for Gaussian or t copula
            - df: Degrees of freedom for t copula
            - theta: Parameter for Clayton, Gumbel, or Frank copula
            - marginals: List of marginal distributions for each asset
            - means: Annualized mean returns for each asset
            - volatilities: Annualized volatilities for each asset
            - dt: Time step size
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_paths, time_steps, n_assets) with correlated returns
        """
        correlation_matrix = params.get('correlation_matrix')
        df = params.get('df', 3)  # Degrees of freedom for t copula
        theta = params.get('theta', 2)  # Parameter for Clayton, Gumbel, or Frank copula
        marginals = params.get('marginals', ['normal'] * n_assets)
        means = params.get('means')
        volatilities = params.get('volatilities')
        dt = params.get('dt', 1/252)  # Default to daily
        
        if means is None or volatilities is None:
            raise ValueError("Means and volatilities must be provided")
        
        if len(means) != n_assets or len(volatilities) != n_assets:
            raise ValueError("Means and volatilities must have length equal to n_assets")
        
        # Initialize returns array
        returns = np.zeros((n_paths, time_steps, n_assets))
        
        # Generate uniform marginals using the specified copula
        if self.copula_type == 'gaussian':
            if correlation_matrix is None:
                raise ValueError("Correlation matrix must be provided for Gaussian copula")
            
            # Generate correlated normal random variables
            uniform_samples = self._gaussian_copula(correlation_matrix, n_paths * time_steps, n_assets)
        
        elif self.copula_type == 't':
            if correlation_matrix is None:
                raise ValueError("Correlation matrix must be provided for t copula")
            
            # Generate correlated t random variables
            uniform_samples = self._t_copula(correlation_matrix, df, n_paths * time_steps, n_assets)
        
        elif self.copula_type == 'clayton':
            # Generate Clayton copula samples
            uniform_samples = self._clayton_copula(theta, n_paths * time_steps, n_assets)
        
        elif self.copula_type == 'gumbel':
            # Generate Gumbel copula samples
            uniform_samples = self._gumbel_copula(theta, n_paths * time_steps, n_assets)
        
        elif self.copula_type == 'frank':
            # Generate Frank copula samples
            uniform_samples = self._frank_copula(theta, n_paths * time_steps, n_assets)
        
        else:
            raise ValueError(f"Unsupported copula type: {self.copula_type}")
        
        # Reshape to (n_paths, time_steps, n_assets)
        uniform_samples = uniform_samples.reshape(n_paths, time_steps, n_assets)
        
        # Transform uniform samples to returns using inverse CDF of marginal distributions
        for i in range(n_assets):
            if marginals[i] == 'normal':
                # Standard normal inverse CDF
                Z = stats.norm.ppf(uniform_samples[:, :, i])
                returns[:, :, i] = means[i] * dt + volatilities[i] * Z * np.sqrt(dt)
            
            elif marginals[i] == 't':
                # t distribution inverse CDF
                df_marginal = params.get('df_marginals', [3] * n_assets)[i]
                Z = stats.t.ppf(uniform_samples[:, :, i], df_marginal)
                returns[:, :, i] = means[i] * dt + volatilities[i] * Z * np.sqrt(dt)
            
            elif marginals[i] == 'skewed_t':
                # Skewed t distribution inverse CDF
                df_marginal = params.get('df_marginals', [3] * n_assets)[i]
                skew = params.get('skew_marginals', [0] * n_assets)[i]
                Z = self._skewed_t_ppf(uniform_samples[:, :, i], df_marginal, skew)
                returns[:, :, i] = means[i] * dt + volatilities[i] * Z * np.sqrt(dt)
            
            else:
                raise ValueError(f"Unsupported marginal distribution: {marginals[i]}")
        
        return returns
    
    def _gaussian_copula(self, correlation_matrix: np.ndarray, n_samples: int, n_dim: int) -> np.ndarray:
        """
        Generate samples from a Gaussian copula
        
        Parameters:
        -----------
        correlation_matrix : np.ndarray
            Correlation matrix
        n_samples : int
            Number of samples to generate
        n_dim : int
            Dimension of the copula
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_samples, n_dim) with uniform samples
        """
        # Generate multivariate normal samples
        Z = np.random.multivariate_normal(np.zeros(n_dim), correlation_matrix, n_samples)
        
        # Transform to uniform using the normal CDF
        U = stats.norm.cdf(Z)
        
        return U
    
    def _t_copula(self, correlation_matrix: np.ndarray, df: int, n_samples: int, n_dim: int) -> np.ndarray:
        """
        Generate samples from a t copula
        
        Parameters:
        -----------
        correlation_matrix : np.ndarray
            Correlation matrix
        df : int
            Degrees of freedom
        n_samples : int
            Number of samples to generate
        n_dim : int
            Dimension of the copula
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_samples, n_dim) with uniform samples
        """
        # Generate multivariate normal samples
        Z = np.random.multivariate_normal(np.zeros(n_dim), correlation_matrix, n_samples)
        
        # Generate chi-square random variable with df degrees of freedom
        chi_square = np.random.chisquare(df, n_samples) / df
        
        # Transform to multivariate t
        T = Z / np.sqrt(chi_square)[:, np.newaxis]
        
        # Transform to uniform using the t CDF
        U = stats.t.cdf(T, df)
        
        return U
    
    def _clayton_copula(self, theta: float, n_samples: int, n_dim: int) -> np.ndarray:
        """
        Generate samples from a Clayton copula
        
        Parameters:
        -----------
        theta : float
            Copula parameter (theta > 0)
        n_samples : int
            Number of samples to generate
        n_dim : int
            Dimension of the copula
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_samples, n_dim) with uniform samples
        """
        if theta <= 0:
            raise ValueError("Clayton copula parameter theta must be positive")
        
        # Generate gamma random variables with mean 1 and variance 1/theta
        gamma_samples = np.random.gamma(1/theta, 1, n_samples)
        
        # Generate independent uniform random variables
        U = np.random.uniform(0, 1, (n_samples, n_dim))
        
        # Transform to Clayton copula samples
        V = np.zeros((n_samples, n_dim))
        for i in range(n_dim):
            V[:, i] = (1 + gamma_samples * (-np.log(U[:, i]))**(-theta))**(-1/theta)
        
        return V
    
    def _gumbel_copula(self, theta: float, n_samples: int, n_dim: int) -> np.ndarray:
        """
        Generate samples from a Gumbel copula
        
        Parameters:
        -----------
        theta : float
            Copula parameter (theta >= 1)
        n_samples : int
            Number of samples to generate
        n_dim : int
            Dimension of the copula
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_samples, n_dim) with uniform samples
        """
        if theta < 1:
            raise ValueError("Gumbel copula parameter theta must be >= 1")
        
        # Generate stable random variables
        alpha = 1/theta
        stable_samples = self._sample_stable(alpha, 1, n_samples)
        
        # Generate independent uniform random variables
        U = np.random.uniform(0, 1, (n_samples, n_dim))
        
        # Transform to Gumbel copula samples
        V = np.zeros((n_samples, n_dim))
        for i in range(n_dim):
            V[:, i] = np.exp(-(-np.log(U[:, i]))**theta / stable_samples)
        
        return V
    
    def _frank_copula(self, theta: float, n_samples: int, n_dim: int) -> np.ndarray:
        """
        Generate samples from a Frank copula
        
        Parameters:
        -----------
        theta : float
            Copula parameter (theta != 0)
        n_samples : int
            Number of samples to generate
        n_dim : int
            Dimension of the copula
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_samples, n_dim) with uniform samples
        """
        if theta == 0:
            raise ValueError("Frank copula parameter theta must not be 0")
        
        # Generate independent uniform random variables
        U = np.random.uniform(0, 1, (n_samples, n_dim))
        
        # For bivariate case, use conditional distribution method
        if n_dim == 2:
            V = np.zeros((n_samples, 2))
            V[:, 0] = U[:, 0]
            
            # Conditional distribution method
            t = np.random.uniform(0, 1, n_samples)
            V[:, 1] = -np.log(1 + t * (np.exp(-theta * U[:, 0]) - 1) / (np.exp(-theta) - 1)) / theta
            
            return V
        
        # For higher dimensions, use the algorithm from Marshall and Olkin
        # This is an approximation for the Frank copula
        gamma_samples = np.random.exponential(1, n_samples)
        
        V = np.zeros((n_samples, n_dim))
        for i in range(n_dim):
            V[:, i] = -np.log(1 - (1 - np.exp(-theta)) / (1 + (np.exp(-theta * gamma_samples) - 1) * U[:, i])) / theta
        
        return V
    
    def _sample_stable(self, alpha: float, beta: float, n_samples: int) -> np.ndarray:
        """
        Generate samples from a stable distribution
        
        Parameters:
        -----------
        alpha : float
            Stability parameter (0 < alpha <= 2)
        beta : float
            Skewness parameter (-1 <= beta <= 1)
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_samples,) with stable samples
        """
        if alpha <= 0 or alpha > 2:
            raise ValueError("Stability parameter alpha must be in (0, 2]")
        
        if beta < -1 or beta > 1:
            raise ValueError("Skewness parameter beta must be in [-1, 1]")
        
        # Generate uniform and exponential random variables
        U = np.random.uniform(-np.pi/2, np.pi/2, n_samples)
        W = np.random.exponential(1, n_samples)
        
        # Compute stable random variables
        if alpha == 1 and beta == 0:
            # Cauchy distribution
            return np.tan(U)
        
        # General case using Chambers-Mallows-Stuck method
        gamma = beta * np.tan(np.pi * alpha / 2)
        delta = alpha != 1
        
        term1 = np.sin(alpha * (U + np.arctan(gamma) / alpha))
        term2 = (np.cos(U) ** (1/alpha))
        term3 = (np.cos(U - alpha * (U + np.arctan(gamma) / alpha)) / W) ** ((1-alpha)/alpha)
        
        return term1 * term2 * term3
    
    def _skewed_t_ppf(self, u: np.ndarray, df: float, skew: float) -> np.ndarray:
        """
        Inverse CDF (percent point function) of the skewed t distribution
        
        Parameters:
        -----------
        u : np.ndarray
            Uniform random variables
        df : float
            Degrees of freedom
        skew : float
            Skewness parameter
            
        Returns:
        --------
        np.ndarray
            Skewed t random variables
        """
        # This is a simplified implementation of Hansen's skewed t distribution
        a = 4 * skew * ((df-2)/(df-1))
        b = np.sqrt(1 + a**2)
        
        # Split calculation based on quantile
        q = np.zeros_like(u)
        mask = u < 0.5
        
        # For u < 0.5
        q[mask] = stats.t.ppf(u[mask] * 2, df)
        q[mask] = (q[mask] + a) / b
        
        # For u >= 0.5
        q[~mask] = stats.t.ppf(1 - (1 - u[~mask]) * 2, df)
        q[~mask] = -(q[~mask] + a) / b
        
        return q


class MonteCarloSimulator:
    """Class for performing Monte Carlo simulations for risk modeling"""
    
    def __init__(self, process_model: StochasticProcess, correlation_model: Optional[CorrelationModel] = None):
        """
        Initialize a MonteCarloSimulator object
        
        Parameters:
        -----------
        process_model : StochasticProcess
            Stochastic process model to use for simulations
        correlation_model : CorrelationModel, optional
            Correlation model to use for multi-asset simulations
        """
        self.process_model = process_model
        self.correlation_model = correlation_model
    
    def simulate_single_asset(self, initial_value: float, time_horizon: float, n_paths: int,
                            n_steps: Optional[int] = None, dt: Optional[float] = None,
                            process_params: Optional[Dict] = None) -> np.ndarray:
        """
        Simulate paths for a single asset
        
        Parameters:
        -----------
        initial_value : float
            Initial value of the asset
        time_horizon : float
            Time horizon in years
        n_paths : int
            Number of paths to simulate
        n_steps : int, optional
            Number of time steps (if None, calculated from dt)
        dt : float, optional
            Time step size in years (if None, calculated from n_steps)
        process_params : dict, optional
            Parameters for the stochastic process
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_paths, n_steps+1) with simulated paths
        """
        if process_params is None:
            process_params = {}
        
        # Determine time step size and number of steps
        if dt is None and n_steps is None:
            # Default to daily steps
            dt = 1/252
            n_steps = int(time_horizon / dt)
        elif dt is None:
            dt = time_horizon / n_steps
        elif n_steps is None:
            n_steps = int(time_horizon / dt)
        
        # Generate paths
        paths = self.process_model.generate_paths(
            initial_value=initial_value,
            time_steps=n_steps,
            n_paths=n_paths,
            dt=dt,
            params=process_params
        )
        
        return paths
    
    def simulate_portfolio(self, initial_values: List[float], weights: List[float],
                         time_horizon: float, n_paths: int, n_steps: Optional[int] = None,
                         dt: Optional[float] = None, process_params: Optional[Dict] = None,
                         correlation_params: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths for a portfolio of assets
        
        Parameters:
        -----------
        initial_values : list of float
            Initial values of the assets
        weights : list of float
            Portfolio weights for each asset
        time_horizon : float
            Time horizon in years
        n_paths : int
            Number of paths to simulate
        n_steps : int, optional
            Number of time steps (if None, calculated from dt)
        dt : float, optional
            Time step size in years (if None, calculated from n_steps)
        process_params : dict, optional
            Parameters for the stochastic process
        correlation_params : dict, optional
            Parameters for the correlation model
            
        Returns:
        --------
        tuple
            (asset_paths, portfolio_paths)
            asset_paths: Array of shape (n_paths, n_steps+1, n_assets) with simulated asset paths
            portfolio_paths: Array of shape (n_paths, n_steps+1) with simulated portfolio paths
        """
        if self.correlation_model is None:
            raise ValueError("Correlation model must be provided for portfolio simulation")
        
        if process_params is None:
            process_params = {}
        
        if correlation_params is None:
            correlation_params = {}
        
        n_assets = len(initial_values)
        
        if len(weights) != n_assets:
            raise ValueError("Number of weights must match number of initial values")
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Determine time step size and number of steps
        if dt is None and n_steps is None:
            # Default to daily steps
            dt = 1/252
            n_steps = int(time_horizon / dt)
        elif dt is None:
            dt = time_horizon / n_steps
        elif n_steps is None:
            n_steps = int(time_horizon / dt)
        
        # Add dt to correlation parameters
        correlation_params['dt'] = dt
        
        # Generate correlated returns
        returns = self.correlation_model.generate_correlated_returns(
            n_assets=n_assets,
            time_steps=n_steps,
            n_paths=n_paths,
            params=correlation_params
        )
        
        # Convert returns to asset paths
        asset_paths = np.zeros((n_paths, n_steps + 1, n_assets))
        for i in range(n_assets):
            asset_paths[:, 0, i] = initial_values[i]
            
            for t in range(1, n_steps + 1):
                asset_paths[:, t, i] = asset_paths[:, t-1, i] * np.exp(returns[:, t-1, i])
        
        # Calculate portfolio paths
        portfolio_paths = np.zeros((n_paths, n_steps + 1))
        for t in range(n_steps + 1):
            portfolio_paths[:, t] = np.sum(asset_paths[:, t, :] * weights, axis=1)
        
        return asset_paths, portfolio_paths
    
    def calculate_var(self, paths: np.ndarray, alpha: float = 0.05, relative: bool = True) -> float:
        """
        Calculate Value at Risk (VaR) from simulated paths
        
        Parameters:
        -----------
        paths : np.ndarray
            Array of simulated paths
        alpha : float, default=0.05
            Confidence level (e.g., 0.05 for 95% VaR)
        relative : bool, default=True
            If True, calculate relative VaR (percentage), otherwise absolute VaR
            
        Returns:
        --------
        float
            Value at Risk
        """
        # Calculate returns from initial to final value
        if relative:
            returns = paths[:, -1] / paths[:, 0] - 1
            var = -np.percentile(returns, alpha * 100)
        else:
            final_values = paths[:, -1]
            var = paths[:, 0].mean() - np.percentile(final_values, alpha * 100)
        
        return var
    
    def calculate_expected_shortfall(self, paths: np.ndarray, alpha: float = 0.05, relative: bool = True) -> float:
        """
        Calculate Expected Shortfall (ES) from simulated paths
        
        Parameters:
        -----------
        paths : np.ndarray
            Array of simulated paths
        alpha : float, default=0.05
            Confidence level (e.g., 0.05 for 95% ES)
        relative : bool, default=True
            If True, calculate relative ES (percentage), otherwise absolute ES
            
        Returns:
        --------
        float
            Expected Shortfall
        """
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
    
    def calculate_risk_metrics(self, paths: np.ndarray) -> Dict:
        """
        Calculate various risk metrics from simulated paths
        
        Parameters:
        -----------
        paths : np.ndarray
            Array of simulated paths
            
        Returns:
        --------
        dict
            Dictionary with risk metrics
        """
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
            'var_95': self.calculate_var(paths, alpha=0.05),
            'var_99': self.calculate_var(paths, alpha=0.01),
            'es_95': self.calculate_expected_shortfall(paths, alpha=0.05),
            'es_99': self.calculate_expected_shortfall(paths, alpha=0.01)
        }
        
        return metrics
    
    def plot_paths(self, paths: np.ndarray, title: str = "Simulated Paths", 
                 xlabel: str = "Time Steps", ylabel: str = "Value", 
                 figsize: Tuple[int, int] = (10, 6), n_display_paths: int = 100) -> None:
        """
        Plot simulated paths
        
        Parameters:
        -----------
        paths : np.ndarray
            Array of simulated paths
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
    
    def plot_histogram(self, paths: np.ndarray, title: str = "Return Distribution", 
                      xlabel: str = "Return", ylabel: str = "Frequency", 
                      figsize: Tuple[int, int] = (10, 6), bins: int = 50) -> None:
        """
        Plot histogram of final returns
        
        Parameters:
        -----------
        paths : np.ndarray
            Array of simulated paths
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
        var_95 = self.calculate_var(paths, alpha=0.05)
        var_99 = self.calculate_var(paths, alpha=0.01)
        es_95 = self.calculate_expected_shortfall(paths, alpha=0.05)
        
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


class StressTestScenario:
    """Base class for stress test scenarios in Monte Carlo simulations"""
    
    def __init__(self, name: str):
        """
        Initialize a StressTestScenario object
        
        Parameters:
        -----------
        name : str
            Name of the stress test scenario
        """
        self.name = name
    
    def apply(self, paths: np.ndarray, params: Dict) -> np.ndarray:
        """
        Apply stress test scenario to simulated paths
        
        Parameters:
        -----------
        paths : np.ndarray
            Array of simulated paths
        params : dict
            Parameters for the stress test
            
        Returns:
        --------
        np.ndarray
            Array with stressed paths
        """
        raise NotImplementedError("Subclasses must implement this method")


class MarketCrashScenario(StressTestScenario):
    """Stress test scenario simulating a market crash"""
    
    def __init__(self):
        """Initialize a Market Crash scenario"""
        super().__init__("Market Crash Scenario")
    
    def apply(self, paths: np.ndarray, params: Dict) -> np.ndarray:
        """
        Apply market crash scenario to simulated paths
        
        Parameters:
        -----------
        paths : np.ndarray
            Array of simulated paths
        params : dict
            Parameters for the stress test:
            - crash_time: Time step at which crash occurs
            - crash_magnitude: Magnitude of the crash (e.g., 0.2 for 20% drop)
            - recovery_rate: Rate of recovery after crash (e.g., 0.05 for 5% per step)
            - affected_paths: Fraction of paths affected by the crash (e.g., 0.8 for 80%)
            
        Returns:
        --------
        np.ndarray
            Array with stressed paths
        """
        crash_time = params.get('crash_time', paths.shape[1] // 2)
        crash_magnitude = params.get('crash_magnitude', 0.2)
        recovery_rate = params.get('recovery_rate', 0.05)
        affected_paths = params.get('affected_paths', 1.0)
        
        # Make a copy of the original paths
        stressed_paths = paths.copy()
        
        # Determine which paths are affected
        n_paths = paths.shape[0]
        n_affected = int(n_paths * affected_paths)
        affected_indices = np.random.choice(n_paths, n_affected, replace=False)
        
        # Apply crash and recovery
        for i in affected_indices:
            # Apply crash
            stressed_paths[i, crash_time] = paths[i, crash_time-1] * (1 - crash_magnitude)
            
            # Apply recovery
            for t in range(crash_time + 1, paths.shape[1]):
                recovery_factor = 1 + recovery_rate
                stressed_paths[i, t] = stressed_paths[i, t-1] * recovery_factor
                
                # Cap recovery at original path level
                if stressed_paths[i, t] > paths[i, t]:
                    stressed_paths[i, t] = paths[i, t]
        
        return stressed_paths


class InterestRateShockScenario(StressTestScenario):
    """Stress test scenario simulating an interest rate shock"""
    
    def __init__(self):
        """Initialize an Interest Rate Shock scenario"""
        super().__init__("Interest Rate Shock Scenario")
    
    def apply(self, paths: np.ndarray, params: Dict) -> np.ndarray:
        """
        Apply interest rate shock scenario to simulated paths
        
        Parameters:
        -----------
        paths : np.ndarray
            Array of simulated paths
        params : dict
            Parameters for the stress test:
            - shock_time: Time step at which shock occurs
            - rate_change: Change in interest rate (e.g., 0.02 for 2% increase)
            - equity_impact: Impact on equity values per 1% rate change
            - bond_impact: Impact on bond values per 1% rate change
            - asset_type: Type of asset ('equity', 'bond', or 'mixed')
            - bond_duration: Duration of bonds (for bond or mixed assets)
            
        Returns:
        --------
        np.ndarray
            Array with stressed paths
        """
        shock_time = params.get('shock_time', paths.shape[1] // 2)
        rate_change = params.get('rate_change', 0.02)  # 2% increase
        equity_impact = params.get('equity_impact', -0.05)  # 5% drop per 1% rate increase
        bond_impact = params.get('bond_impact', -0.07)  # 7% drop per 1% rate increase
        asset_type = params.get('asset_type', 'equity')
        bond_duration = params.get('bond_duration', 5)  # 5-year duration
        
        # Make a copy of the original paths
        stressed_paths = paths.copy()
        
        # Determine impact based on asset type
        if asset_type == 'equity':
            impact = equity_impact * rate_change * 100
        elif asset_type == 'bond':
            impact = -bond_duration * rate_change * 100  # Duration-based impact
        elif asset_type == 'mixed':
            equity_weight = params.get('equity_weight', 0.6)
            bond_weight = 1 - equity_weight
            impact = (equity_weight * equity_impact + bond_weight * (-bond_duration)) * rate_change * 100
        else:
            raise ValueError(f"Unsupported asset type: {asset_type}")
        
        # Apply shock
        for i in range(paths.shape[0]):
            # Apply immediate impact
            stressed_paths[i, shock_time] = paths[i, shock_time-1] * (1 + impact/100)
            
            # Propagate impact to future time steps
            for t in range(shock_time + 1, paths.shape[1]):
                # Calculate relative change from original path
                relative_change = paths[i, t] / paths[i, t-1]
                stressed_paths[i, t] = stressed_paths[i, t-1] * relative_change
        
        return stressed_paths


class VolatilityShockScenario(StressTestScenario):
    """Stress test scenario simulating a volatility spike"""
    
    def __init__(self):
        """Initialize a Volatility Shock scenario"""
        super().__init__("Volatility Shock Scenario")
    
    def apply(self, paths: np.ndarray, params: Dict) -> np.ndarray:
        """
        Apply volatility shock scenario to simulated paths
        
        Parameters:
        -----------
        paths : np.ndarray
            Array of simulated paths
        params : dict
            Parameters for the stress test:
            - shock_time: Time step at which shock occurs
            - vol_multiplier: Factor by which volatility increases
            - shock_duration: Duration of the shock in time steps
            - mean_reversion: Rate at which volatility reverts to normal
            
        Returns:
        --------
        np.ndarray
            Array with stressed paths
        """
        shock_time = params.get('shock_time', paths.shape[1] // 3)
        vol_multiplier = params.get('vol_multiplier', 3)  # Triple volatility
        shock_duration = params.get('shock_duration', paths.shape[1] // 6)  # 1/6 of simulation period
        mean_reversion = params.get('mean_reversion', 0.1)  # 10% reversion per step
        
        # Make a copy of the original paths
        stressed_paths = paths.copy()
        
        # Calculate returns from the original paths
        returns = np.zeros_like(paths)
        for t in range(1, paths.shape[1]):
            returns[:, t] = paths[:, t] / paths[:, t-1] - 1
        
        # Calculate normal volatility (standard deviation of returns)
        normal_vol = np.std(returns[:, 1:shock_time])
        
        # Apply volatility shock
        for t in range(shock_time, min(shock_time + shock_duration, paths.shape[1])):
            # Calculate current volatility multiplier
            if t > shock_time:
                # Gradually revert to normal volatility
                steps_from_shock = t - shock_time
                current_multiplier = vol_multiplier * (1 - mean_reversion) ** steps_from_shock
                if current_multiplier < 1:
                    current_multiplier = 1
            else:
                current_multiplier = vol_multiplier
            
            # Generate new returns with increased volatility
            new_returns = np.random.normal(
                np.mean(returns[:, 1:shock_time]),
                normal_vol * current_multiplier,
                paths.shape[0]
            )
            
            # Apply new returns
            stressed_paths[:, t] = stressed_paths[:, t-1] * (1 + new_returns)
        
        # Continue with original relative changes after shock period
        for t in range(shock_time + shock_duration, paths.shape[1]):
            relative_change = paths[:, t] / paths[:, t-1]
            stressed_paths[:, t] = stressed_paths[:, t-1] * relative_change
        
        return stressed_paths


class CustomScenario(StressTestScenario):
    """Custom stress test scenario using user-defined functions"""
    
    def __init__(self, name: str, scenario_func: Callable):
        """
        Initialize a Custom scenario
        
        Parameters:
        -----------
        name : str
            Name of the custom scenario
        scenario_func : callable
            Function that applies the scenario to paths
        """
        super().__init__(name)
        self.scenario_func = scenario_func
    
    def apply(self, paths: np.ndarray, params: Dict) -> np.ndarray:
        """
        Apply custom scenario to simulated paths
        
        Parameters:
        -----------
        paths : np.ndarray
            Array of simulated paths
        params : dict
            Parameters for the stress test
            
        Returns:
        --------
        np.ndarray
            Array with stressed paths
        """
        return self.scenario_func(paths, params)


class TariffScenario(StressTestScenario):
    """Stress test scenario simulating the impact of tariff announcements"""
    
    def __init__(self):
        """Initialize a Tariff Impact scenario"""
        super().__init__("Tariff Impact Scenario")
    
    def apply(self, paths: np.ndarray, params: Dict) -> np.ndarray:
        """
        Apply tariff impact scenario to simulated paths
        
        Parameters:
        -----------
        paths : np.ndarray
            Array of simulated paths
        params : dict
            Parameters for the stress test:
            - announcement_time: Time step at which tariff is announced
            - implementation_time: Time step at which tariff is implemented
            - tariff_rate: Tariff rate as a percentage
            - domestic_impact: Impact on domestic companies (percentage)
            - foreign_impact: Impact on foreign companies (percentage)
            - sector_impacts: Dictionary mapping sectors to impact multipliers
            - company_type: Type of company ('domestic', 'foreign', or 'multinational')
            - sector: Sector of the company
            - recovery_period: Number of time steps for market to adjust to new tariffs
            
        Returns:
        --------
        np.ndarray
            Array with stressed paths
        """
        announcement_time = params.get('announcement_time', paths.shape[1] // 4)
        implementation_time = params.get('implementation_time', paths.shape[1] // 2)
        tariff_rate = params.get('tariff_rate', 0.25)  # 25% tariff
        domestic_impact = params.get('domestic_impact', 0.02)  # 2% positive impact
        foreign_impact = params.get('foreign_impact', -0.15)  # 15% negative impact
        sector_impacts = params.get('sector_impacts', {
            'technology': 1.2,
            'consumer_goods': 1.5,
            'industrial': 1.3,
            'healthcare': 0.7,
            'financial': 0.5
        })
        company_type = params.get('company_type', 'multinational')
        sector = params.get('sector', 'technology')
        recovery_period = params.get('recovery_period', paths.shape[1] // 8)
        
        # Make a copy of the original paths
        stressed_paths = paths.copy()
        
        # Determine base impact based on company type
        if company_type == 'domestic':
            base_impact = domestic_impact
        elif company_type == 'foreign':
            base_impact = foreign_impact
        elif company_type == 'multinational':
            domestic_weight = params.get('domestic_weight', 0.6)
            base_impact = domestic_weight * domestic_impact + (1 - domestic_weight) * foreign_impact
        else:
            raise ValueError(f"Unsupported company type: {company_type}")
        
        # Apply sector-specific multiplier
        sector_multiplier = sector_impacts.get(sector, 1.0)
        impact = base_impact * sector_multiplier * tariff_rate / 0.25  # Scale by tariff rate
        
        # Apply announcement effect
        for i in range(paths.shape[0]):
            # Immediate reaction to announcement
            stressed_paths[i, announcement_time] = paths[i, announcement_time-1] * (1 + impact * 0.5)
            
            # Period between announcement and implementation
            for t in range(announcement_time + 1, implementation_time):
                # Gradual adjustment with some volatility
                adjustment_factor = 0.5 + 0.5 * (t - announcement_time) / (implementation_time - announcement_time)
                noise = np.random.normal(0, 0.01)  # Add some noise/uncertainty
                step_impact = impact * adjustment_factor + noise
                
                relative_change = paths[i, t] / paths[i, t-1]
                stressed_paths[i, t] = stressed_paths[i, t-1] * relative_change * (1 + step_impact)
            
            # Implementation effect
            stressed_paths[i, implementation_time] = stressed_paths[i, implementation_time-1] * (1 + impact * 0.7)
            
            # Post-implementation adjustment
            for t in range(implementation_time + 1, min(implementation_time + recovery_period, paths.shape[1])):
                # Market adjusts to new normal
                progress = (t - implementation_time) / recovery_period
                adjustment = 1 - np.exp(-3 * progress)  # Exponential adjustment
                
                # Long-term impact after full adjustment
                long_term_impact = impact * 1.2  # Could be different from initial impact
                
                # Current impact during adjustment period
                current_impact = impact * (1 - adjustment) + long_term_impact * adjustment
                
                relative_change = paths[i, t] / paths[i, t-1]
                stressed_paths[i, t] = stressed_paths[i, t-1] * relative_change * (1 + current_impact * 0.1)
            
            # Continue with adjusted path after recovery period
            for t in range(implementation_time + recovery_period, paths.shape[1]):
                relative_change = paths[i, t] / paths[i, t-1]
                stressed_paths[i, t] = stressed_paths[i, t-1] * relative_change
        
        return stressed_paths


class MonteCarloRiskModel:
    """Main class for Monte Carlo risk modeling and stress testing"""
    
    def __init__(self):
        """Initialize a Monte Carlo Risk Model"""
        self.simulators = {}
        self.stress_scenarios = {}
        self.results = {}
    
    def add_simulator(self, name: str, simulator: MonteCarloSimulator) -> None:
        """
        Add a Monte Carlo simulator
        
        Parameters:
        -----------
        name : str
            Name of the simulator
        simulator : MonteCarloSimulator
            Simulator object
        """
        self.simulators[name] = simulator
    
    def add_stress_scenario(self, scenario: StressTestScenario) -> None:
        """
        Add a stress test scenario
        
        Parameters:
        -----------
        scenario : StressTestScenario
            Stress test scenario object
        """
        self.stress_scenarios[scenario.name] = scenario
    
    def run_simulation(self, simulator_name: str, simulation_params: Dict) -> str:
        """
        Run a Monte Carlo simulation
        
        Parameters:
        -----------
        simulator_name : str
            Name of the simulator to use
        simulation_params : dict
            Parameters for the simulation
            
        Returns:
        --------
        str
            Result ID for retrieving simulation results
        """
        if simulator_name not in self.simulators:
            raise ValueError(f"Simulator '{simulator_name}' not found")
        
        simulator = self.simulators[simulator_name]
        
        # Extract simulation parameters
        is_portfolio = simulation_params.get('is_portfolio', False)
        
        if is_portfolio:
            # Portfolio simulation
            initial_values = simulation_params.get('initial_values')
            weights = simulation_params.get('weights')
            time_horizon = simulation_params.get('time_horizon')
            n_paths = simulation_params.get('n_paths', 10000)
            n_steps = simulation_params.get('n_steps')
            dt = simulation_params.get('dt')
            process_params = simulation_params.get('process_params', {})
            correlation_params = simulation_params.get('correlation_params', {})
            
            # Run portfolio simulation
            asset_paths, portfolio_paths = simulator.simulate_portfolio(
                initial_values=initial_values,
                weights=weights,
                time_horizon=time_horizon,
                n_paths=n_paths,
                n_steps=n_steps,
                dt=dt,
                process_params=process_params,
                correlation_params=correlation_params
            )
            
            # Store results
            result_id = f"{simulator_name}_portfolio_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.results[result_id] = {
                'type': 'portfolio',
                'asset_paths': asset_paths,
                'portfolio_paths': portfolio_paths,
                'params': simulation_params
            }
        
        else:
            # Single asset simulation
            initial_value = simulation_params.get('initial_value')
            time_horizon = simulation_params.get('time_horizon')
            n_paths = simulation_params.get('n_paths', 10000)
            n_steps = simulation_params.get('n_steps')
            dt = simulation_params.get('dt')
            process_params = simulation_params.get('process_params', {})
            
            # Run single asset simulation
            paths = simulator.simulate_single_asset(
                initial_value=initial_value,
                time_horizon=time_horizon,
                n_paths=n_paths,
                n_steps=n_steps,
                dt=dt,
                process_params=process_params
            )
            
            # Store results
            result_id = f"{simulator_name}_single_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.results[result_id] = {
                'type': 'single',
                'paths': paths,
                'params': simulation_params
            }
        
        return result_id
    
    def apply_stress_test(self, result_id: str, scenario_name: str, scenario_params: Dict) -> str:
        """
        Apply a stress test scenario to simulation results
        
        Parameters:
        -----------
        result_id : str
            Result ID from a previous simulation
        scenario_name : str
            Name of the stress scenario to apply
        scenario_params : dict
            Parameters for the stress scenario
            
        Returns:
        --------
        str
            Result ID for retrieving stress test results
        """
        if result_id not in self.results:
            raise ValueError(f"Result '{result_id}' not found")
        
        if scenario_name not in self.stress_scenarios:
            raise ValueError(f"Stress scenario '{scenario_name}' not found")
        
        result = self.results[result_id]
        scenario = self.stress_scenarios[scenario_name]
        
        # Apply stress scenario
        if result['type'] == 'portfolio':
            # Apply to portfolio paths
            original_paths = result['portfolio_paths']
            stressed_paths = scenario.apply(original_paths, scenario_params)
            
            # Store results
            stress_result_id = f"{result_id}_{scenario_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.results[stress_result_id] = {
                'type': 'stressed_portfolio',
                'original_paths': original_paths,
                'stressed_paths': stressed_paths,
                'scenario': scenario_name,
                'params': scenario_params
            }
        
        else:
            # Apply to single asset paths
            original_paths = result['paths']
            stressed_paths = scenario.apply(original_paths, scenario_params)
            
            # Store results
            stress_result_id = f"{result_id}_{scenario_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.results[stress_result_id] = {
                'type': 'stressed_single',
                'original_paths': original_paths,
                'stressed_paths': stressed_paths,
                'scenario': scenario_name,
                'params': scenario_params
            }
        
        return stress_result_id
    
    def get_risk_metrics(self, result_id: str) -> Dict:
        """
        Calculate risk metrics for simulation results
        
        Parameters:
        -----------
        result_id : str
            Result ID from a previous simulation or stress test
            
        Returns:
        --------
        dict
            Dictionary with risk metrics
        """
        if result_id not in self.results:
            raise ValueError(f"Result '{result_id}' not found")
        
        result = self.results[result_id]
        
        # Determine which paths to use
        if result['type'] in ['single', 'portfolio']:
            paths = result['paths'] if result['type'] == 'single' else result['portfolio_paths']
            is_stressed = False
        elif result['type'] in ['stressed_single', 'stressed_portfolio']:
            paths = result['stressed_paths']
            is_stressed = True
        else:
            raise ValueError(f"Unsupported result type: {result['type']}")
        
        # Calculate returns from initial to final value
        returns = paths[:, -1] / paths[:, 0] - 1
        
        # Calculate basic statistics
        metrics = {
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_dev': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns)
        }
        
        # Calculate VaR at different confidence levels
        for alpha in [0.01, 0.025, 0.05, 0.1]:
            var = -np.percentile(returns, alpha * 100)
            metrics[f'var_{int(alpha*100)}'] = var
        
        # Calculate Expected Shortfall at different confidence levels
        for alpha in [0.01, 0.025, 0.05, 0.1]:
            var = -np.percentile(returns, alpha * 100)
            es = -np.mean(returns[returns <= -var])
            metrics[f'es_{int(alpha*100)}'] = es
        
        # Calculate maximum drawdown
        max_drawdown = 0
        peak = paths[:, 0]
        
        for t in range(1, paths.shape[1]):
            # Calculate drawdown for each path
            drawdown = (paths[:, t] - peak) / peak
            max_drawdown = min(max_drawdown, np.min(drawdown))
            
            # Update peak
            peak = np.maximum(peak, paths[:, t])
        
        metrics['max_drawdown'] = max_drawdown
        
        # Add scenario information if stressed
        if is_stressed:
            metrics['scenario'] = result['scenario']
            
            # Calculate impact of stress
            original_paths = result['original_paths']
            original_returns = original_paths[:, -1] / original_paths[:, 0] - 1
            stressed_returns = returns
            
            metrics['stress_impact_mean'] = np.mean(stressed_returns) - np.mean(original_returns)
            metrics['stress_impact_var_95'] = (-np.percentile(stressed_returns, 95) - 
                                             (-np.percentile(original_returns, 95)))
            metrics['stress_impact_es_95'] = (
                -np.mean(stressed_returns[stressed_returns <= -np.percentile(stressed_returns, 95)]) - 
                -np.mean(original_returns[original_returns <= -np.percentile(original_returns, 95)])
            )
        
        return metrics
    
    def compare_scenarios(self, base_result_id: str, scenario_result_ids: List[str]) -> Dict:
        """
        Compare multiple stress test scenarios
        
        Parameters:
        -----------
        base_result_id : str
            Result ID for the base simulation
        scenario_result_ids : list of str
            Result IDs for stress test scenarios
            
        Returns:
        --------
        dict
            Dictionary with comparison metrics
        """
        if base_result_id not in self.results:
            raise ValueError(f"Base result '{base_result_id}' not found")
        
        for result_id in scenario_result_ids:
            if result_id not in self.results:
                raise ValueError(f"Scenario result '{result_id}' not found")
        
        # Get base metrics
        base_metrics = self.get_risk_metrics(base_result_id)
        
        # Get scenario metrics
        scenario_metrics = {result_id: self.get_risk_metrics(result_id) for result_id in scenario_result_ids}
        
        # Prepare comparison
        comparison = {
            'base': {
                'mean_return': base_metrics['mean_return'],
                'var_95': base_metrics['var_95'],
                'es_95': base_metrics['es_95'],
                'max_drawdown': base_metrics['max_drawdown']
            },
            'scenarios': {}
        }
        
        for result_id, metrics in scenario_metrics.items():
            scenario_name = self.results[result_id]['scenario']
            comparison['scenarios'][scenario_name] = {
                'mean_return': metrics['mean_return'],
                'var_95': metrics['var_95'],
                'es_95': metrics['es_95'],
                'max_drawdown': metrics['max_drawdown'],
                'impact_mean': metrics['mean_return'] - base_metrics['mean_return'],
                'impact_var_95': metrics['var_95'] - base_metrics['var_95'],
                'impact_es_95': metrics['es_95'] - base_metrics['es_95'],
                'impact_max_drawdown': metrics['max_drawdown'] - base_metrics['max_drawdown']
            }
        
        return comparison
    
    def plot_scenario_comparison(self, base_result_id: str, scenario_result_ids: List[str],
                               figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot comparison of multiple stress test scenarios
        
        Parameters:
        -----------
        base_result_id : str
            Result ID for the base simulation
        scenario_result_ids : list of str
            Result IDs for stress test scenarios
        figsize : tuple, default=(12, 8)
            Figure size
        """
        if base_result_id not in self.results:
            raise ValueError(f"Base result '{base_result_id}' not found")
        
        for result_id in scenario_result_ids:
            if result_id not in self.results:
                raise ValueError(f"Scenario result '{result_id}' not found")
        
        # Get paths
        base_result = self.results[base_result_id]
        base_paths = base_result['paths'] if base_result['type'] == 'single' else base_result['portfolio_paths']
        
        scenario_paths = []
        scenario_names = []
        
        for result_id in scenario_result_ids:
            result = self.results[result_id]
            paths = result['stressed_paths']
            scenario_name = result['scenario']
            
            scenario_paths.append(paths)
            scenario_names.append(scenario_name)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot mean paths
        base_mean = np.mean(base_paths, axis=0)
        plt.plot(base_mean, 'k-', linewidth=2, label="Base Scenario")
        
        colors = ['r', 'g', 'b', 'm', 'c', 'y']
        
        for i, (paths, name) in enumerate(zip(scenario_paths, scenario_names)):
            mean_path = np.mean(paths, axis=0)
            plt.plot(mean_path, f"{colors[i % len(colors)]}-", linewidth=2, label=name)
        
        plt.title("Comparison of Stress Test Scenarios")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Create histogram of final returns
        plt.figure(figsize=figsize)
        
        base_returns = base_paths[:, -1] / base_paths[:, 0] - 1
        plt.hist(base_returns, bins=50, alpha=0.3, label="Base Scenario")
        
        for i, (paths, name) in enumerate(zip(scenario_paths, scenario_names)):
            returns = paths[:, -1] / paths[:, 0] - 1
            plt.hist(returns, bins=50, alpha=0.3, label=name)
        
        plt.title("Distribution of Returns under Different Scenarios")
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Create bar chart of risk metrics
        metrics = ['VaR (95%)', 'ES (95%)', 'Max Drawdown']
        base_values = [
            self.get_risk_metrics(base_result_id)['var_95'],
            self.get_risk_metrics(base_result_id)['es_95'],
            -self.get_risk_metrics(base_result_id)['max_drawdown']
        ]
        
        scenario_values = []
        for result_id in scenario_result_ids:
            metrics_dict = self.get_risk_metrics(result_id)
            scenario_values.append([
                metrics_dict['var_95'],
                metrics_dict['es_95'],
                -metrics_dict['max_drawdown']
            ])
        
        plt.figure(figsize=figsize)
        
        x = np.arange(len(metrics))
        width = 0.8 / (len(scenario_result_ids) + 1)
        
        plt.bar(x - 0.4 + width/2, base_values, width, label="Base Scenario")
        
        for i, (values, name) in enumerate(zip(scenario_values, scenario_names)):
            plt.bar(x - 0.4 + (i + 1.5) * width, values, width, label=name)
        
        plt.title("Risk Metrics Comparison")
        plt.xticks(x, metrics)
        plt.ylabel("Value")
        plt.legend()
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
