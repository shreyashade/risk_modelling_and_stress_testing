"""
Hierarchical Bayesian Models for Multi-Asset Portfolio Risk Analysis

This module implements hierarchical Bayesian models for analyzing and modeling
dependencies between multiple assets in a portfolio.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pymc as pm
import arviz as az
import theano.tensor as tt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class HierarchicalBayesianModel:
    """
    A class implementing hierarchical Bayesian models for multi-asset portfolios
    with complex dependencies.
    
    This class provides methods for:
    1. Multi-level modeling of asset returns
    2. Bayesian factor models with time-varying loadings
    3. Hierarchical correlation structures
    """
    
    def __init__(self, name="HierarchicalBayesianModel", random_state=None):
        """
        Initialize the Hierarchical Bayesian Model.
        
        Parameters:
        -----------
        name : str, default="HierarchicalBayesianModel"
            Name of the model instance
        random_state : int, optional
            Random seed for reproducibility
        """
        self.name = name
        self.random_state = random_state
        self.models = {}
        self.traces = {}
        self.parameters = {}
        self.factor_loadings = {}
        self.correlations = {}
        
        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
    
    def fit_multi_level(self, returns, group_mapping=None, samples=2000, tune=1000, 
                        chains=2, target_accept=0.8, return_inferencedata=True):
        """
        Fit a multi-level Bayesian model for asset returns.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            Asset returns time series with assets as columns
        group_mapping : dict, optional
            Mapping of assets to groups (e.g., sectors, countries)
            If None, each asset is treated as its own group
        samples : int, default=2000
            Number of samples to draw
        tune : int, default=1000
            Number of tuning steps
        chains : int, default=2
            Number of MCMC chains
        target_accept : float, default=0.8
            Target acceptance rate
        return_inferencedata : bool, default=True
            Whether to return inference data object
            
        Returns:
        --------
        dict
            Dictionary containing model, trace, and parameters
        """
        # Ensure returns is a DataFrame
        if not isinstance(returns, pd.DataFrame):
            raise ValueError("Returns must be a pandas DataFrame with assets as columns")
        
        # Create group mapping if not provided
        if group_mapping is None:
            group_mapping = {asset: i for i, asset in enumerate(returns.columns)}
        
        # Extract unique groups
        unique_groups = sorted(set(group_mapping.values()))
        group_indices = {group: i for i, group in enumerate(unique_groups)}
        
        # Create asset to group index mapping
        asset_to_group_idx = np.array([group_indices[group_mapping[asset]] for asset in returns.columns])
        
        # Standardize returns
        scaler = StandardScaler()
        returns_standardized = scaler.fit_transform(returns)
        
        n_assets = returns.shape[1]
        n_groups = len(unique_groups)
        n_obs = returns.shape[0]
        
        # Create PyMC model
        with pm.Model() as multi_level_model:
            # Group-level parameters
            group_mu = pm.Normal('group_mu', mu=0, sigma=0.1, shape=n_groups)
            group_sigma = pm.HalfNormal('group_sigma', sigma=0.1, shape=n_groups)
            
            # Asset-level parameters
            asset_mu_offset = pm.Normal('asset_mu_offset', mu=0, sigma=0.05, shape=n_assets)
            asset_sigma_ratio = pm.HalfNormal('asset_sigma_ratio', sigma=0.5, shape=n_assets)
            
            # Calculate asset-level parameters
            asset_mu = pm.Deterministic('asset_mu', group_mu[asset_to_group_idx] + asset_mu_offset)
            asset_sigma = pm.Deterministic('asset_sigma', group_sigma[asset_to_group_idx] * asset_sigma_ratio)
            
            # Observations
            obs = pm.Normal('obs', mu=asset_mu, sigma=asset_sigma, observed=returns_standardized)
            
            # Sample from posterior
            trace = pm.sample(samples, tune=tune, chains=chains, target_accept=target_accept,
                             return_inferencedata=return_inferencedata, random_seed=self.random_state)
        
        # Store model and trace
        self.models['multi_level'] = multi_level_model
        self.traces['multi_level'] = trace
        
        # Extract parameters
        params = {
            'group_mu': trace.posterior['group_mu'].mean(dim=('chain', 'draw')).values,
            'group_sigma': trace.posterior['group_sigma'].mean(dim=('chain', 'draw')).values,
            'asset_mu': trace.posterior['asset_mu'].mean(dim=('chain', 'draw')).values,
            'asset_sigma': trace.posterior['asset_sigma'].mean(dim=('chain', 'draw')).values,
            'group_mapping': group_mapping,
            'group_indices': group_indices,
            'asset_names': returns.columns.tolist()
        }
        
        self.parameters['multi_level'] = params
        
        # Return results
        return {
            'model': multi_level_model,
            'trace': trace,
            'parameters': params
        }
    
    def fit_factor_model(self, returns, n_factors=3, time_varying=True, samples=2000, tune=1000, 
                         chains=2, target_accept=0.8, return_inferencedata=True):
        """
        Fit a Bayesian factor model with optional time-varying loadings.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            Asset returns time series with assets as columns
        n_factors : int, default=3
            Number of latent factors
        time_varying : bool, default=True
            Whether to use time-varying factor loadings
        samples : int, default=2000
            Number of samples to draw
        tune : int, default=1000
            Number of tuning steps
        chains : int, default=2
            Number of MCMC chains
        target_accept : float, default=0.8
            Target acceptance rate
        return_inferencedata : bool, default=True
            Whether to return inference data object
            
        Returns:
        --------
        dict
            Dictionary containing model, trace, and factor loadings
        """
        # Ensure returns is a DataFrame
        if not isinstance(returns, pd.DataFrame):
            raise ValueError("Returns must be a pandas DataFrame with assets as columns")
        
        # Standardize returns
        scaler = StandardScaler()
        returns_standardized = scaler.fit_transform(returns)
        
        n_assets = returns.shape[1]
        n_obs = returns.shape[0]
        
        # Create PyMC model
        with pm.Model() as factor_model:
            # Factor time series
            factors = pm.Normal('factors', mu=0, sigma=1, shape=(n_obs, n_factors))
            
            if time_varying:
                # Time-varying factor loadings
                # We model loadings as a random walk
                loadings_init = pm.Normal('loadings_init', mu=0, sigma=1, shape=(n_assets, n_factors))
                loadings_innovation = pm.Normal('loadings_innovation', mu=0, sigma=0.05, 
                                               shape=(n_obs-1, n_assets, n_factors))
                
                # Accumulate innovations
                loadings = pm.Deterministic('loadings', 
                                           tt.concatenate([
                                               loadings_init.reshape((1, n_assets, n_factors)),
                                               loadings_init.reshape((1, n_assets, n_factors)) + 
                                               tt.cumsum(loadings_innovation, axis=0)
                                           ], axis=0))
                
                # Idiosyncratic volatility (asset-specific)
                idio_vol = pm.HalfNormal('idio_vol', sigma=0.1, shape=n_assets)
                
                # Expected returns
                expected_returns = pm.Deterministic('expected_returns',
                                                  tt.sum(loadings * factors.reshape((n_obs, 1, n_factors)), axis=2))
                
                # Observations
                obs = pm.Normal('obs', mu=expected_returns, sigma=idio_vol, observed=returns_standardized)
                
            else:
                # Static factor loadings
                loadings = pm.Normal('loadings', mu=0, sigma=1, shape=(n_assets, n_factors))
                
                # Idiosyncratic volatility (asset-specific)
                idio_vol = pm.HalfNormal('idio_vol', sigma=0.1, shape=n_assets)
                
                # Expected returns
                expected_returns = pm.Deterministic('expected_returns',
                                                  tt.dot(factors, loadings.T))
                
                # Observations
                obs = pm.Normal('obs', mu=expected_returns, sigma=idio_vol, observed=returns_standardized)
            
            # Sample from posterior
            trace = pm.sample(samples, tune=tune, chains=chains, target_accept=target_accept,
                             return_inferencedata=return_inferencedata, random_seed=self.random_state)
        
        # Store model and trace
        model_name = 'factor_model_tv' if time_varying else 'factor_model'
        self.models[model_name] = factor_model
        self.traces[model_name] = trace
        
        # Extract factor loadings
        if time_varying:
            loadings = trace.posterior['loadings'].mean(dim=('chain', 'draw')).values
        else:
            loadings = trace.posterior['loadings'].mean(dim=('chain', 'draw')).values
            # Expand to time dimension for consistency
            loadings = np.tile(loadings, (n_obs, 1, 1))
        
        # Extract factors
        factors = trace.posterior['factors'].mean(dim=('chain', 'draw')).values
        
        # Extract idiosyncratic volatility
        idio_vol = trace.posterior['idio_vol'].mean(dim=('chain', 'draw')).values
        
        # Store factor loadings
        self.factor_loadings[model_name] = {
            'loadings': loadings,
            'factors': factors,
            'idio_vol': idio_vol,
            'asset_names': returns.columns.tolist(),
            'time_varying': time_varying
        }
        
        # Return results
        return {
            'model': factor_model,
            'trace': trace,
            'factor_loadings': self.factor_loadings[model_name]
        }
    
    def fit_hierarchical_correlation(self, returns, group_mapping=None, samples=2000, tune=1000, 
                                    chains=2, target_accept=0.8, return_inferencedata=True):
        """
        Fit a hierarchical correlation model for asset returns.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            Asset returns time series with assets as columns
        group_mapping : dict, optional
            Mapping of assets to groups (e.g., sectors, countries)
            If None, each asset is treated as its own group
        samples : int, default=2000
            Number of samples to draw
        tune : int, default=1000
            Number of tuning steps
        chains : int, default=2
            Number of MCMC chains
        target_accept : float, default=0.8
            Target acceptance rate
        return_inferencedata : bool, default=True
            Whether to return inference data object
            
        Returns:
        --------
        dict
            Dictionary containing model, trace, and correlation matrices
        """
        # Ensure returns is a DataFrame
        if not isinstance(returns, pd.DataFrame):
            raise ValueError("Returns must be a pandas DataFrame with assets as columns")
        
        # Create group mapping if not provided
        if group_mapping is None:
            group_mapping = {asset: i for i, asset in enumerate(returns.columns)}
        
        # Extract unique groups
        unique_groups = sorted(set(group_mapping.values()))
        group_indices = {group: i for i, group in enumerate(unique_groups)}
        
        # Create asset to group index mapping
        asset_to_group_idx = np.array([group_indices[group_mapping[asset]] for asset in returns.columns])
        
        # Create group to assets mapping
        group_to_assets = {}
        for asset, group in group_mapping.items():
            if group not in group_to_assets:
                group_to_assets[group] = []
            group_to_assets[group].append(asset)
        
        # Standardize returns
        scaler = StandardScaler()
        returns_standardized = scaler.fit_transform(returns)
        
        n_assets = returns.shape[1]
        n_groups = len(unique_groups)
        n_obs = returns.shape[0]
        
        # Create PyMC model
        with pm.Model() as corr_model:
            # Group-level correlation matrices
            group_corr_chol = []
            for g in range(n_groups):
                n_assets_in_group = len(group_to_assets[unique_groups[g]])
                if n_assets_in_group > 1:
                    # LKJ prior for correlation matrices
                    group_corr_chol.append(
                        pm.LKJCholeskyCov(f'group_{g}_corr_chol', n=n_assets_in_group, eta=2)
                    )
                else:
                    # For groups with only one asset, use identity matrix
                    group_corr_chol.append(np.eye(1))
            
            # Between-group correlation matrix
            between_group_corr_chol = pm.LKJCholeskyCov('between_group_corr_chol', n=n_groups, eta=2)
            
            # Asset-level volatilities
            asset_sigma = pm.HalfNormal('asset_sigma', sigma=0.1, shape=n_assets)
            
            # Group-level volatilities
            group_sigma = pm.HalfNormal('group_sigma', sigma=0.1, shape=n_groups)
            
            # Asset-level means
            asset_mu = pm.Normal('asset_mu', mu=0, sigma=0.05, shape=n_assets)
            
            # Construct full covariance matrix
            # This is a complex operation that would require custom Theano code
            # For simplicity, we'll use a multivariate normal with a full covariance matrix
            
            # Convert Cholesky factors to correlation matrices
            between_group_corr = pm.math.dot(between_group_corr_chol, between_group_corr_chol.T)
            
            # Create covariance matrix
            cov = pm.Normal('cov', mu=0, sigma=1, shape=(n_assets, n_assets))
            
            # Observations
            obs = pm.MvNormal('obs', mu=asset_mu, cov=cov, observed=returns_standardized)
            
            # Sample from posterior
            trace = pm.sample(samples, tune=tune, chains=chains, target_accept=target_accept,
                             return_inferencedata=return_inferencedata, random_seed=self.random_state)
        
        # Store model and trace
        self.models['hierarchical_correlation'] = corr_model
        self.traces['hierarchical_correlation'] = trace
        
        # Extract correlation matrices
        # This is a simplified approach since the full hierarchical correlation
        # structure is complex to extract from the trace
        
        # Calculate empirical correlation matrix
        empirical_corr = np.corrcoef(returns_standardized.T)
        
        # Store correlation matrices
        self.correlations['hierarchical_correlation'] = {
            'empirical_corr': empirical_corr,
            'asset_names': returns.columns.tolist(),
            'group_mapping': group_mapping,
            'group_indices': group_indices
        }
        
        # Return results
        return {
            'model': corr_model,
            'trace': trace,
            'correlations': self.correlations['hierarchical_correlation']
        }
    
    def predict_returns(self, model_type='multi_level', n_periods=1, 
                        return_samples=False, n_samples=1000):
        """
        Predict future returns using the fitted model.
        
        Parameters:
        -----------
        model_type : str, default='multi_level'
            Type of model to use for prediction
            ('multi_level', 'factor_model', 'factor_model_tv', 'hierarchical_correlation')
        n_periods : int, default=1
            Number of periods to predict
        return_samples : bool, default=False
            Whether to return samples from the predictive distribution
        n_samples : int, default=1000
            Number of samples to draw if return_samples is True
            
        Returns:
        --------
        dict
            Dictionary containing predicted returns and optionally samples
        """
        if model_type not in self.models:
            raise ValueError(f"Model '{model_type}' not fitted. Call fit_{model_type} first.")
        
        # Get trace
        trace = self.traces[model_type]
        
        # Predict based on model type
        if model_type == 'multi_level':
            predictions = self._predict_multi_level(trace, n_periods, return_samples, n_samples)
        elif model_type in ['factor_model', 'factor_model_tv']:
            predictions = self._predict_factor_model(trace, model_type, n_periods, return_samples, n_samples)
        elif model_type == 'hierarchical_correlation':
            predictions = self._predict_hierarchical_correlation(trace, n_periods, return_samples, n_samples)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return predictions
    
    def _predict_multi_level(self, trace, n_periods, return_samples, n_samples):
        """
        Predict future returns using the multi-level model.
        
        Parameters:
        -----------
        trace : arviz.InferenceData
            MCMC trace
        n_periods : int
            Number of periods to predict
        return_samples : bool
            Whether to return samples from the predictive distribution
        n_samples : int
            Number of samples to draw if return_samples is True
            
        Returns:
        --------
        dict
            Dictionary containing predicted returns and optionally samples
        """
        # Extract parameters
        params = self.parameters['multi_level']
        asset_mu = params['asset_mu']
        asset_sigma = params['asset_sigma']
        asset_names = params['asset_names']
        
        n_assets = len(asset_names)
        
        # Generate predictions
        if return_samples:
            # Draw samples from the posterior predictive distribution
            samples_idx = np.random.choice(trace.posterior['asset_mu'].shape[0] * 
                                          trace.posterior['asset_mu'].shape[1],
                                          size=n_samples, replace=True)
            
            chain_idx = samples_idx // trace.posterior['asset_mu'].shape[1]
            draw_idx = samples_idx % trace.posterior['asset_mu'].shape[1]
            
            mu_samples = trace.posterior['asset_mu'].values[chain_idx, draw_idx, :]
            sigma_samples = trace.posterior['asset_sigma'].values[chain_idx, draw_idx, :]
            
            # Generate return samples
            return_samples = np.zeros((n_samples, n_periods, n_assets))
            
            for i in range(n_samples):
                for j in range(n_periods):
                    return_samples[i, j, :] = np.random.normal(mu_samples[i], sigma_samples[i])
            
            # Calculate mean prediction
            mean_prediction = return_samples.mean(axis=0)
            
            # Create DataFrame
            mean_pred_df = pd.DataFrame(mean_prediction, columns=asset_names)
            
            return {
                'mean_prediction': mean_pred_df,
                'samples': return_samples,
                'asset_names': asset_names
            }
        else:
            # Generate mean prediction
            mean_prediction = np.tile(asset_mu, (n_periods, 1))
            
            # Create DataFrame
            mean_pred_df = pd.DataFrame(mean_prediction, columns=asset_names)
            
            return {
                'mean_prediction': mean_pred_df,
                'asset_names': asset_names
            }
    
    def _predict_factor_model(self, trace, model_type, n_periods, return_samples, n_samples):
        """
        Predict future returns using the factor model.
        
        Parameters:
        -----------
        trace : arviz.InferenceData
            MCMC trace
        model_type : str
            Type of factor model ('factor_model' or 'factor_model_tv')
        n_periods : int
            Number of periods to predict
        return_samples : bool
            Whether to return samples from the predictive distribution
        n_samples : int
            Number of samples to draw if return_samples is True
            
        Returns:
        --------
        dict
            Dictionary containing predicted returns and optionally samples
        """
        # Extract factor loadings
        factor_loadings = self.factor_loadings[model_type]
        loadings = factor_loadings['loadings']
        factors = factor_loadings['factors']
        idio_vol = factor_loadings['idio_vol']
        asset_names = factor_loadings['asset_names']
        time_varying = factor_loadings['time_varying']
        
        n_assets = len(asset_names)
        n_factors = loadings.shape[-1]
        
        # Generate predictions
        if return_samples:
            # Draw samples from the posterior predictive distribution
            samples_idx = np.random.choice(trace.posterior['factors'].shape[0] * 
                                          trace.posterior['factors'].shape[1],
                                          size=n_samples, replace=True)
            
            chain_idx = samples_idx // trace.posterior['factors'].shape[1]
            draw_idx = samples_idx % trace.posterior['factors'].shape[1]
            
            # For time-varying loadings, use the last time point
            if time_varying:
                loadings_samples = trace.posterior['loadings'].values[chain_idx, draw_idx, -1, :, :]
            else:
                loadings_samples = trace.posterior['loadings'].values[chain_idx, draw_idx, :, :]
            
            idio_vol_samples = trace.posterior['idio_vol'].values[chain_idx, draw_idx, :]
            
            # Generate factor samples (random walk from last factors)
            last_factors = trace.posterior['factors'].values[chain_idx, draw_idx, -1, :]
            factor_samples = np.zeros((n_samples, n_periods, n_factors))
            factor_samples[:, 0, :] = last_factors
            
            for i in range(1, n_periods):
                factor_samples[:, i, :] = factor_samples[:, i-1, :] + np.random.normal(0, 0.1, (n_samples, n_factors))
            
            # Generate return samples
            return_samples = np.zeros((n_samples, n_periods, n_assets))
            
            for i in range(n_samples):
                for j in range(n_periods):
                    # Expected returns from factors
                    expected_returns = np.sum(loadings_samples[i] * factor_samples[i, j, :].reshape(1, -1), axis=1)
                    
                    # Add idiosyncratic noise
                    return_samples[i, j, :] = expected_returns + np.random.normal(0, idio_vol_samples[i])
            
            # Calculate mean prediction
            mean_prediction = return_samples.mean(axis=0)
            
            # Create DataFrame
            mean_pred_df = pd.DataFrame(mean_prediction, columns=asset_names)
            
            return {
                'mean_prediction': mean_pred_df,
                'samples': return_samples,
                'asset_names': asset_names
            }
        else:
            # For time-varying loadings, use the last time point
            if time_varying:
                last_loadings = loadings[-1]
            else:
                last_loadings = loadings
            
            # Generate factor predictions (random walk from last factors)
            last_factors = factors[-1]
            factor_predictions = np.zeros((n_periods, n_factors))
            factor_predictions[0] = last_factors
            
            for i in range(1, n_periods):
                factor_predictions[i] = factor_predictions[i-1]  # Assume factors stay constant
            
            # Generate return predictions
            return_predictions = np.zeros((n_periods, n_assets))
            
            for i in range(n_periods):
                # Expected returns from factors
                return_predictions[i] = np.sum(last_loadings * factor_predictions[i].reshape(1, -1), axis=1)
            
            # Create DataFrame
            mean_pred_df = pd.DataFrame(return_predictions, columns=asset_names)
            
            return {
                'mean_prediction': mean_pred_df,
                'asset_names': asset_names
            }
    
    def _predict_hierarchical_correlation(self, trace, n_periods, return_samples, n_samples):
        """
        Predict future returns using the hierarchical correlation model.
        
        Parameters:
        -----------
        trace : arviz.InferenceData
            MCMC trace
        n_periods : int
            Number of periods to predict
        return_samples : bool
            Whether to return samples from the predictive distribution
        n_samples : int
            Number of samples to draw if return_samples is True
            
        Returns:
        --------
        dict
            Dictionary containing predicted returns and optionally samples
        """
        # Extract parameters
        correlations = self.correlations['hierarchical_correlation']
        empirical_corr = correlations['empirical_corr']
        asset_names = correlations['asset_names']
        
        n_assets = len(asset_names)
        
        # Extract asset means and volatilities
        asset_mu = trace.posterior['asset_mu'].mean(dim=('chain', 'draw')).values
        asset_sigma = trace.posterior['asset_sigma'].mean(dim=('chain', 'draw')).values
        
        # Generate predictions
        if return_samples:
            # Draw samples from the posterior predictive distribution
            samples_idx = np.random.choice(trace.posterior['asset_mu'].shape[0] * 
                                          trace.posterior['asset_mu'].shape[1],
                                          size=n_samples, replace=True)
            
            chain_idx = samples_idx // trace.posterior['asset_mu'].shape[1]
            draw_idx = samples_idx % trace.posterior['asset_mu'].shape[1]
            
            mu_samples = trace.posterior['asset_mu'].values[chain_idx, draw_idx, :]
            sigma_samples = trace.posterior['asset_sigma'].values[chain_idx, draw_idx, :]
            
            # Generate return samples
            return_samples = np.zeros((n_samples, n_periods, n_assets))
            
            for i in range(n_samples):
                # Create covariance matrix
                cov = np.diag(sigma_samples[i]) @ empirical_corr @ np.diag(sigma_samples[i])
                
                # Generate multivariate normal samples
                for j in range(n_periods):
                    return_samples[i, j, :] = np.random.multivariate_normal(mu_samples[i], cov)
            
            # Calculate mean prediction
            mean_prediction = return_samples.mean(axis=0)
            
            # Create DataFrame
            mean_pred_df = pd.DataFrame(mean_prediction, columns=asset_names)
            
            return {
                'mean_prediction': mean_pred_df,
                'samples': return_samples,
                'asset_names': asset_names
            }
        else:
            # Generate mean prediction
            mean_prediction = np.tile(asset_mu, (n_periods, 1))
            
            # Create DataFrame
            mean_pred_df = pd.DataFrame(mean_prediction, columns=asset_names)
            
            return {
                'mean_prediction': mean_pred_df,
                'asset_names': asset_names
            }
    
    def predict_covariance(self, model_type='factor_model', n_samples=1000):
        """
        Predict the covariance matrix using the fitted model.
        
        Parameters:
        -----------
        model_type : str, default='factor_model'
            Type of model to use for prediction
            ('multi_level', 'factor_model', 'factor_model_tv', 'hierarchical_correlation')
        n_samples : int, default=1000
            Number of samples to draw for covariance estimation
            
        Returns:
        --------
        dict
            Dictionary containing predicted covariance matrix and correlation matrix
        """
        if model_type not in self.models:
            raise ValueError(f"Model '{model_type}' not fitted. Call fit_{model_type} first.")
        
        # Get trace
        trace = self.traces[model_type]
        
        # Predict based on model type
        if model_type == 'multi_level':
            predictions = self._predict_multi_level_covariance(trace, n_samples)
        elif model_type in ['factor_model', 'factor_model_tv']:
            predictions = self._predict_factor_model_covariance(trace, model_type, n_samples)
        elif model_type == 'hierarchical_correlation':
            predictions = self._predict_hierarchical_correlation_covariance(trace, n_samples)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return predictions
    
    def _predict_multi_level_covariance(self, trace, n_samples):
        """
        Predict covariance matrix using the multi-level model.
        
        Parameters:
        -----------
        trace : arviz.InferenceData
            MCMC trace
        n_samples : int
            Number of samples to draw for covariance estimation
            
        Returns:
        --------
        dict
            Dictionary containing predicted covariance matrix and correlation matrix
        """
        # Extract parameters
        params = self.parameters['multi_level']
        asset_mu = params['asset_mu']
        asset_sigma = params['asset_sigma']
        asset_names = params['asset_names']
        group_mapping = params['group_mapping']
        
        n_assets = len(asset_names)
        
        # Draw samples from the posterior predictive distribution
        samples_idx = np.random.choice(trace.posterior['asset_mu'].shape[0] * 
                                      trace.posterior['asset_mu'].shape[1],
                                      size=n_samples, replace=True)
        
        chain_idx = samples_idx // trace.posterior['asset_mu'].shape[1]
        draw_idx = samples_idx % trace.posterior['asset_mu'].shape[1]
        
        mu_samples = trace.posterior['asset_mu'].values[chain_idx, draw_idx, :]
        sigma_samples = trace.posterior['asset_sigma'].values[chain_idx, draw_idx, :]
        
        # Generate return samples
        return_samples = np.zeros((n_samples, n_assets))
        
        for i in range(n_samples):
            return_samples[i, :] = np.random.normal(mu_samples[i], sigma_samples[i])
        
        # Calculate covariance matrix
        cov_matrix = np.cov(return_samples, rowvar=False)
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(return_samples, rowvar=False)
        
        # Create DataFrames
        cov_df = pd.DataFrame(cov_matrix, index=asset_names, columns=asset_names)
        corr_df = pd.DataFrame(corr_matrix, index=asset_names, columns=asset_names)
        
        return {
            'covariance': cov_df,
            'correlation': corr_df,
            'asset_names': asset_names
        }
    
    def _predict_factor_model_covariance(self, trace, model_type, n_samples):
        """
        Predict covariance matrix using the factor model.
        
        Parameters:
        -----------
        trace : arviz.InferenceData
            MCMC trace
        model_type : str
            Type of factor model ('factor_model' or 'factor_model_tv')
        n_samples : int
            Number of samples to draw for covariance estimation
            
        Returns:
        --------
        dict
            Dictionary containing predicted covariance matrix and correlation matrix
        """
        # Extract factor loadings
        factor_loadings = self.factor_loadings[model_type]
        loadings = factor_loadings['loadings']
        idio_vol = factor_loadings['idio_vol']
        asset_names = factor_loadings['asset_names']
        time_varying = factor_loadings['time_varying']
        
        n_assets = len(asset_names)
        
        # For time-varying loadings, use the last time point
        if time_varying:
            loadings = loadings[-1]
        
        # Draw samples from the posterior predictive distribution
        samples_idx = np.random.choice(trace.posterior['loadings'].shape[0] * 
                                      trace.posterior['loadings'].shape[1],
                                      size=n_samples, replace=True)
        
        chain_idx = samples_idx // trace.posterior['loadings'].shape[1]
        draw_idx = samples_idx % trace.posterior['loadings'].shape[1]
        
        if time_varying:
            loadings_samples = trace.posterior['loadings'].values[chain_idx, draw_idx, -1, :, :]
        else:
            loadings_samples = trace.posterior['loadings'].values[chain_idx, draw_idx, :, :]
        
        idio_vol_samples = trace.posterior['idio_vol'].values[chain_idx, draw_idx, :]
        
        # Calculate factor covariance (identity for standard factors)
        factor_cov = np.eye(loadings.shape[1])
        
        # Calculate covariance matrix
        cov_matrix = np.zeros((n_assets, n_assets))
        
        for i in range(n_samples):
            # Factor contribution to covariance
            factor_contrib = loadings_samples[i] @ factor_cov @ loadings_samples[i].T
            
            # Add idiosyncratic variance
            total_cov = factor_contrib + np.diag(idio_vol_samples[i]**2)
            
            cov_matrix += total_cov
        
        cov_matrix /= n_samples
        
        # Calculate correlation matrix
        std_devs = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
        
        # Create DataFrames
        cov_df = pd.DataFrame(cov_matrix, index=asset_names, columns=asset_names)
        corr_df = pd.DataFrame(corr_matrix, index=asset_names, columns=asset_names)
        
        return {
            'covariance': cov_df,
            'correlation': corr_df,
            'asset_names': asset_names
        }
    
    def _predict_hierarchical_correlation_covariance(self, trace, n_samples):
        """
        Predict covariance matrix using the hierarchical correlation model.
        
        Parameters:
        -----------
        trace : arviz.InferenceData
            MCMC trace
        n_samples : int
            Number of samples to draw for covariance estimation
            
        Returns:
        --------
        dict
            Dictionary containing predicted covariance matrix and correlation matrix
        """
        # Extract parameters
        correlations = self.correlations['hierarchical_correlation']
        empirical_corr = correlations['empirical_corr']
        asset_names = correlations['asset_names']
        
        n_assets = len(asset_names)
        
        # Extract asset volatilities
        asset_sigma = trace.posterior['asset_sigma'].mean(dim=('chain', 'draw')).values
        
        # Calculate covariance matrix
        cov_matrix = np.diag(asset_sigma) @ empirical_corr @ np.diag(asset_sigma)
        
        # Create DataFrames
        cov_df = pd.DataFrame(cov_matrix, index=asset_names, columns=asset_names)
        corr_df = pd.DataFrame(empirical_corr, index=asset_names, columns=asset_names)
        
        return {
            'covariance': cov_df,
            'correlation': corr_df,
            'asset_names': asset_names
        }
    
    def plot_factor_loadings(self, model_type='factor_model', figsize=(12, 8)):
        """
        Plot factor loadings from the factor model.
        
        Parameters:
        -----------
        model_type : str, default='factor_model'
            Type of factor model ('factor_model' or 'factor_model_tv')
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if model_type not in ['factor_model', 'factor_model_tv']:
            raise ValueError(f"Model '{model_type}' is not a factor model.")
        
        if model_type not in self.factor_loadings:
            raise ValueError(f"Model '{model_type}' not fitted. Call fit_factor_model first.")
        
        # Extract factor loadings
        factor_loadings = self.factor_loadings[model_type]
        loadings = factor_loadings['loadings']
        asset_names = factor_loadings['asset_names']
        time_varying = factor_loadings['time_varying']
        
        n_assets = len(asset_names)
        n_factors = loadings.shape[-1]
        
        # Create figure
        if time_varying:
            # Plot time-varying loadings
            fig, axes = plt.subplots(n_factors, 1, figsize=figsize, sharex=True)
            
            if n_factors == 1:
                axes = [axes]
            
            for i in range(n_factors):
                for j in range(n_assets):
                    axes[i].plot(loadings[:, j, i], label=asset_names[j])
                
                axes[i].set_title(f'Factor {i+1} Loadings')
                axes[i].legend()
                axes[i].grid(True)
            
            plt.tight_layout()
            
        else:
            # Plot static loadings
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create heatmap
            sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0,
                       xticklabels=[f'Factor {i+1}' for i in range(n_factors)],
                       yticklabels=asset_names, ax=ax)
            
            ax.set_title('Factor Loadings')
            
        return fig
    
    def plot_correlation_matrix(self, model_type='hierarchical_correlation', figsize=(10, 8)):
        """
        Plot correlation matrix from the model.
        
        Parameters:
        -----------
        model_type : str, default='hierarchical_correlation'
            Type of model to use
            ('multi_level', 'factor_model', 'factor_model_tv', 'hierarchical_correlation')
        figsize : tuple, default=(10, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Predict covariance matrix
        prediction = self.predict_covariance(model_type)
        
        # Extract correlation matrix
        corr_matrix = prediction['correlation']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   vmin=-1, vmax=1, ax=ax)
        
        ax.set_title(f'Correlation Matrix ({model_type})')
        
        return fig
    
    def plot_group_parameters(self, figsize=(12, 8)):
        """
        Plot group-level parameters from the multi-level model.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if 'multi_level' not in self.parameters:
            raise ValueError("Multi-level model not fitted. Call fit_multi_level first.")
        
        # Extract parameters
        params = self.parameters['multi_level']
        group_mu = params['group_mu']
        group_sigma = params['group_sigma']
        group_indices = params['group_indices']
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Sort groups by mean
        sorted_indices = np.argsort(group_mu)
        sorted_groups = [list(group_indices.keys())[list(group_indices.values()).index(i)] 
                        for i in sorted_indices]
        
        # Plot group means
        axes[0].bar(range(len(sorted_groups)), group_mu[sorted_indices])
        axes[0].set_xticks(range(len(sorted_groups)))
        axes[0].set_xticklabels(sorted_groups, rotation=45, ha='right')
        axes[0].set_title('Group Mean Returns')
        axes[0].set_ylabel('Mean')
        axes[0].grid(True, axis='y')
        
        # Plot group volatilities
        axes[1].bar(range(len(sorted_groups)), group_sigma[sorted_indices])
        axes[1].set_xticks(range(len(sorted_groups)))
        axes[1].set_xticklabels(sorted_groups, rotation=45, ha='right')
        axes[1].set_title('Group Volatilities')
        axes[1].set_ylabel('Volatility')
        axes[1].grid(True, axis='y')
        
        plt.tight_layout()
        
        return fig
    
    def plot_asset_parameters(self, figsize=(12, 8)):
        """
        Plot asset-level parameters from the multi-level model.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if 'multi_level' not in self.parameters:
            raise ValueError("Multi-level model not fitted. Call fit_multi_level first.")
        
        # Extract parameters
        params = self.parameters['multi_level']
        asset_mu = params['asset_mu']
        asset_sigma = params['asset_sigma']
        asset_names = params['asset_names']
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Sort assets by mean
        sorted_indices = np.argsort(asset_mu)
        sorted_assets = [asset_names[i] for i in sorted_indices]
        
        # Plot asset means
        axes[0].bar(range(len(sorted_assets)), asset_mu[sorted_indices])
        axes[0].set_xticks(range(len(sorted_assets)))
        axes[0].set_xticklabels(sorted_assets, rotation=90)
        axes[0].set_title('Asset Mean Returns')
        axes[0].set_ylabel('Mean')
        axes[0].grid(True, axis='y')
        
        # Plot asset volatilities
        axes[1].bar(range(len(sorted_assets)), asset_sigma[sorted_indices])
        axes[1].set_xticks(range(len(sorted_assets)))
        axes[1].set_xticklabels(sorted_assets, rotation=90)
        axes[1].set_title('Asset Volatilities')
        axes[1].set_ylabel('Volatility')
        axes[1].grid(True, axis='y')
        
        plt.tight_layout()
        
        return fig
    
    def plot_return_distribution(self, model_type='multi_level', n_samples=1000, figsize=(12, 8)):
        """
        Plot predicted return distributions.
        
        Parameters:
        -----------
        model_type : str, default='multi_level'
            Type of model to use for prediction
            ('multi_level', 'factor_model', 'factor_model_tv', 'hierarchical_correlation')
        n_samples : int, default=1000
            Number of samples to draw
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Predict returns
        predictions = self.predict_returns(model_type, n_periods=1, return_samples=True, n_samples=n_samples)
        
        # Extract samples and asset names
        samples = predictions['samples'][:, 0, :]  # First period only
        asset_names = predictions['asset_names']
        
        n_assets = len(asset_names)
        
        # Create figure
        fig, axes = plt.subplots(min(n_assets, 9), 1, figsize=figsize)
        
        if n_assets == 1:
            axes = [axes]
        
        # Plot distributions for up to 9 assets
        for i in range(min(n_assets, 9)):
            sns.histplot(samples[:, i], kde=True, ax=axes[i])
            axes[i].set_title(f'{asset_names[i]} Return Distribution')
            axes[i].axvline(samples[:, i].mean(), color='r', linestyle='--', 
                           label=f'Mean: {samples[:, i].mean():.4f}')
            axes[i].axvline(np.percentile(samples[:, i], 5), color='g', linestyle='--',
                           label=f'5% VaR: {np.percentile(samples[:, i], 5):.4f}')
            axes[i].legend()
        
        plt.tight_layout()
        
        return fig
    
    def calculate_risk_metrics(self, model_type='multi_level', confidence_levels=[0.95, 0.99], 
                              n_samples=10000):
        """
        Calculate risk metrics using the fitted model.
        
        Parameters:
        -----------
        model_type : str, default='multi_level'
            Type of model to use for prediction
            ('multi_level', 'factor_model', 'factor_model_tv', 'hierarchical_correlation')
        confidence_levels : list, default=[0.95, 0.99]
            Confidence levels for VaR and ES
        n_samples : int, default=10000
            Number of samples to draw
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing risk metrics
        """
        # Predict returns
        predictions = self.predict_returns(model_type, n_periods=1, return_samples=True, n_samples=n_samples)
        
        # Extract samples and asset names
        samples = predictions['samples'][:, 0, :]  # First period only
        asset_names = predictions['asset_names']
        
        n_assets = len(asset_names)
        
        # Calculate risk metrics
        risk_metrics = []
        
        for i in range(n_assets):
            asset_samples = samples[:, i]
            
            metrics = {
                'Asset': asset_names[i],
                'Mean': np.mean(asset_samples),
                'Std Dev': np.std(asset_samples),
                'Skewness': stats.skew(asset_samples),
                'Kurtosis': stats.kurtosis(asset_samples)
            }
            
            # Calculate VaR and ES for each confidence level
            for cl in confidence_levels:
                var = np.percentile(asset_samples, 100 * (1 - cl))
                es = np.mean(asset_samples[asset_samples <= var])
                
                metrics[f'VaR ({cl*100:.0f}%)'] = -var  # Negative to represent loss
                metrics[f'ES ({cl*100:.0f}%)'] = -es  # Negative to represent loss
            
            risk_metrics.append(metrics)
        
        # Create DataFrame
        risk_df = pd.DataFrame(risk_metrics)
        
        return risk_df
    
    def calculate_portfolio_risk(self, weights, model_type='multi_level', confidence_levels=[0.95, 0.99], 
                                n_samples=10000):
        """
        Calculate portfolio risk metrics using the fitted model.
        
        Parameters:
        -----------
        weights : dict or numpy.ndarray
            Portfolio weights (dict mapping asset names to weights or array)
        model_type : str, default='multi_level'
            Type of model to use for prediction
            ('multi_level', 'factor_model', 'factor_model_tv', 'hierarchical_correlation')
        confidence_levels : list, default=[0.95, 0.99]
            Confidence levels for VaR and ES
        n_samples : int, default=10000
            Number of samples to draw
            
        Returns:
        --------
        dict
            Dictionary containing portfolio risk metrics
        """
        # Predict returns
        predictions = self.predict_returns(model_type, n_periods=1, return_samples=True, n_samples=n_samples)
        
        # Extract samples and asset names
        samples = predictions['samples'][:, 0, :]  # First period only
        asset_names = predictions['asset_names']
        
        n_assets = len(asset_names)
        
        # Convert weights to array
        if isinstance(weights, dict):
            weight_array = np.zeros(n_assets)
            for i, asset in enumerate(asset_names):
                if asset in weights:
                    weight_array[i] = weights[asset]
        else:
            weight_array = weights
        
        # Normalize weights
        weight_array = weight_array / np.sum(weight_array)
        
        # Calculate portfolio returns
        portfolio_returns = samples @ weight_array
        
        # Calculate risk metrics
        metrics = {
            'Mean': np.mean(portfolio_returns),
            'Std Dev': np.std(portfolio_returns),
            'Sharpe Ratio': np.mean(portfolio_returns) / np.std(portfolio_returns),
            'Skewness': stats.skew(portfolio_returns),
            'Kurtosis': stats.kurtosis(portfolio_returns)
        }
        
        # Calculate VaR and ES for each confidence level
        for cl in confidence_levels:
            var = np.percentile(portfolio_returns, 100 * (1 - cl))
            es = np.mean(portfolio_returns[portfolio_returns <= var])
            
            metrics[f'VaR ({cl*100:.0f}%)'] = -var  # Negative to represent loss
            metrics[f'ES ({cl*100:.0f}%)'] = -es  # Negative to represent loss
        
        return metrics
    
    def plot_portfolio_distribution(self, weights, model_type='multi_level', n_samples=10000, figsize=(10, 6)):
        """
        Plot portfolio return distribution.
        
        Parameters:
        -----------
        weights : dict or numpy.ndarray
            Portfolio weights (dict mapping asset names to weights or array)
        model_type : str, default='multi_level'
            Type of model to use for prediction
            ('multi_level', 'factor_model', 'factor_model_tv', 'hierarchical_correlation')
        n_samples : int, default=10000
            Number of samples to draw
        figsize : tuple, default=(10, 6)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Predict returns
        predictions = self.predict_returns(model_type, n_periods=1, return_samples=True, n_samples=n_samples)
        
        # Extract samples and asset names
        samples = predictions['samples'][:, 0, :]  # First period only
        asset_names = predictions['asset_names']
        
        n_assets = len(asset_names)
        
        # Convert weights to array
        if isinstance(weights, dict):
            weight_array = np.zeros(n_assets)
            for i, asset in enumerate(asset_names):
                if asset in weights:
                    weight_array[i] = weights[asset]
        else:
            weight_array = weights
        
        # Normalize weights
        weight_array = weight_array / np.sum(weight_array)
        
        # Calculate portfolio returns
        portfolio_returns = samples @ weight_array
        
        # Calculate risk metrics
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        es_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        es_99 = np.mean(portfolio_returns[portfolio_returns <= var_99])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot distribution
        sns.histplot(portfolio_returns, kde=True, ax=ax)
        
        # Add vertical lines for risk metrics
        ax.axvline(np.mean(portfolio_returns), color='b', linestyle='-', 
                  label=f'Mean: {np.mean(portfolio_returns):.4f}')
        ax.axvline(var_95, color='g', linestyle='--', 
                  label=f'95% VaR: {-var_95:.4f}')
        ax.axvline(var_99, color='r', linestyle='--', 
                  label=f'99% VaR: {-var_99:.4f}')
        ax.axvline(es_95, color='g', linestyle=':', 
                  label=f'95% ES: {-es_95:.4f}')
        ax.axvline(es_99, color='r', linestyle=':', 
                  label=f'99% ES: {-es_99:.4f}')
        
        ax.set_title('Portfolio Return Distribution')
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        return fig
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        import pickle
        
        # Create a dictionary with model attributes
        model_dict = {
            'name': self.name,
            'random_state': self.random_state,
            'parameters': self.parameters,
            'factor_loadings': self.factor_loadings,
            'correlations': self.correlations
        }
        
        # Save traces separately if they exist
        for model_type, trace in self.traces.items():
            trace_path = f"{filepath}_{model_type}_trace"
            az.to_netcdf(trace, trace_path)
        
        # Save model dictionary
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to load the model from
            
        Returns:
        --------
        HierarchicalBayesianModel
            Loaded model
        """
        import pickle
        import os
        
        # Load model dictionary
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        
        # Create new instance
        model = cls(name=model_dict['name'], random_state=model_dict['random_state'])
        
        # Set attributes
        model.parameters = model_dict['parameters']
        model.factor_loadings = model_dict['factor_loadings']
        model.correlations = model_dict['correlations']
        
        # Load traces if they exist
        model_types = ['multi_level', 'factor_model', 'factor_model_tv', 'hierarchical_correlation']
        for model_type in model_types:
            trace_path = f"{filepath}_{model_type}_trace"
            if os.path.exists(trace_path):
                model.traces[model_type] = az.from_netcdf(trace_path)
        
        return model


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    # Create a multi-asset portfolio
    n_assets = 10
    n_obs = 500
    
    # Asset groups (sectors)
    sectors = ['Technology', 'Finance', 'Energy', 'Healthcare', 'Consumer']
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    
    # Assign assets to sectors
    asset_sectors = np.random.choice(sectors, size=n_assets)
    group_mapping = {asset_names[i]: asset_sectors[i] for i in range(n_assets)}
    
    # Generate sector-specific parameters
    sector_params = {
        'Technology': {'mu': 0.001, 'sigma': 0.02},
        'Finance': {'mu': 0.0005, 'sigma': 0.015},
        'Energy': {'mu': 0.0002, 'sigma': 0.025},
        'Healthcare': {'mu': 0.0008, 'sigma': 0.018},
        'Consumer': {'mu': 0.0006, 'sigma': 0.016}
    }
    
    # Generate asset-specific parameters
    asset_params = []
    for i in range(n_assets):
        sector = asset_sectors[i]
        sector_mu = sector_params[sector]['mu']
        sector_sigma = sector_params[sector]['sigma']
        
        # Add asset-specific variation
        asset_mu = sector_mu + np.random.normal(0, 0.0002)
        asset_sigma = sector_sigma * np.random.uniform(0.8, 1.2)
        
        asset_params.append({'mu': asset_mu, 'sigma': asset_sigma})
    
    # Generate correlated returns
    # Create correlation matrix with sector-based block structure
    corr_matrix = np.eye(n_assets)
    
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            # Higher correlation within sector
            if asset_sectors[i] == asset_sectors[j]:
                corr = np.random.uniform(0.6, 0.8)
            else:
                corr = np.random.uniform(0.2, 0.4)
            
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    
    # Create covariance matrix
    sigma_vector = np.array([params['sigma'] for params in asset_params])
    cov_matrix = np.diag(sigma_vector) @ corr_matrix @ np.diag(sigma_vector)
    
    # Generate returns
    mu_vector = np.array([params['mu'] for params in asset_params])
    returns = np.random.multivariate_normal(mu_vector, cov_matrix, size=n_obs)
    
    # Create DataFrame
    returns_df = pd.DataFrame(returns, columns=asset_names)
    
    # Create model
    model = HierarchicalBayesianModel(random_state=42)
    
    # Fit multi-level model
    multi_level_results = model.fit_multi_level(returns_df, group_mapping, samples=500, tune=200)
    
    # Plot group parameters
    fig = model.plot_group_parameters()
    plt.savefig('group_parameters.png')
    
    # Fit factor model
    factor_model_results = model.fit_factor_model(returns_df, n_factors=3, time_varying=False, 
                                                 samples=500, tune=200)
    
    # Plot factor loadings
    fig = model.plot_factor_loadings()
    plt.savefig('factor_loadings.png')
    
    # Calculate risk metrics
    risk_metrics = model.calculate_risk_metrics()
    print(risk_metrics)
    
    # Calculate portfolio risk
    equal_weights = np.ones(n_assets) / n_assets
    portfolio_risk = model.calculate_portfolio_risk(equal_weights)
    print(portfolio_risk)
    
    # Plot portfolio distribution
    fig = model.plot_portfolio_distribution(equal_weights)
    plt.savefig('portfolio_distribution.png')
