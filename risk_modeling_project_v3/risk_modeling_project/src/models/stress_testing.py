"""
Stress Testing Module for Risk Modeling Framework.

This module provides functionality for stress testing portfolios under various
market scenarios and conditions.
"""

import numpy as np
import pandas as pd
from scipy import stats


class StressTesting:
    """
    Stress Testing class for risk modeling.
    
    This class provides methods for stress testing portfolios under various
    market scenarios and conditions.
    """
    
    def __init__(self):
        """
        Initialize the stress testing model.
        """
        self.scenarios = {}
        self.results = {}
        self.portfolio = None
        self.historical_data = None
    
    def calibrate(self, historical_data, portfolio=None, regime=None):
        """
        Calibrate the stress testing model with historical data.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical returns data
        portfolio : object, optional
            Portfolio object
        regime : str, optional
            Current market regime
        """
        self.historical_data = historical_data
        self.portfolio = portfolio
        
        # Define standard scenarios
        self.define_standard_scenarios()
    
    def define_standard_scenarios(self):
        """
        Define standard stress testing scenarios.
        """
        # Financial crisis scenario (2008-style)
        self.scenarios['financial_crisis'] = {
            'description': '2008-style financial crisis',
            'shocks': {
                'equities': -0.40,  # 40% drop in equities
                'bonds': 0.05,      # 5% increase in bonds (flight to safety)
                'commodities': -0.30,  # 30% drop in commodities
                'real_estate': -0.35,  # 35% drop in real estate
                'volatility': 3.0   # 3x increase in volatility
            },
            'correlation_changes': {
                'equities_bonds': -0.7,  # Strong negative correlation
                'equities_commodities': 0.8  # Strong positive correlation
            },
            'duration': 60  # 60 days
        }
        
        # Interest rate shock scenario
        self.scenarios['interest_rate_shock'] = {
            'description': 'Sudden interest rate hike',
            'shocks': {
                'equities': -0.15,  # 15% drop in equities
                'bonds': -0.10,     # 10% drop in bonds
                'commodities': 0.05,  # 5% increase in commodities
                'real_estate': -0.20,  # 20% drop in real estate
                'volatility': 1.5   # 1.5x increase in volatility
            },
            'correlation_changes': {
                'equities_bonds': 0.6,  # Positive correlation
                'equities_commodities': 0.3  # Weak positive correlation
            },
            'duration': 30  # 30 days
        }
        
        # Inflation shock scenario
        self.scenarios['inflation_shock'] = {
            'description': 'Sudden inflation spike',
            'shocks': {
                'equities': -0.10,  # 10% drop in equities
                'bonds': -0.15,     # 15% drop in bonds
                'commodities': 0.25,  # 25% increase in commodities
                'real_estate': 0.05,  # 5% increase in real estate
                'volatility': 1.8   # 1.8x increase in volatility
            },
            'correlation_changes': {
                'equities_bonds': 0.5,  # Positive correlation
                'equities_commodities': -0.4  # Negative correlation
            },
            'duration': 45  # 45 days
        }
        
        # Geopolitical crisis scenario
        self.scenarios['geopolitical_crisis'] = {
            'description': 'Major geopolitical crisis',
            'shocks': {
                'equities': -0.25,  # 25% drop in equities
                'bonds': 0.10,      # 10% increase in bonds (flight to safety)
                'commodities': 0.30,  # 30% increase in commodities (oil shock)
                'real_estate': -0.15,  # 15% drop in real estate
                'volatility': 2.5   # 2.5x increase in volatility
            },
            'correlation_changes': {
                'equities_bonds': -0.6,  # Strong negative correlation
                'equities_commodities': -0.5  # Negative correlation
            },
            'duration': 40  # 40 days
        }
        
        # Pandemic scenario (COVID-19 style)
        self.scenarios['pandemic'] = {
            'description': 'Global pandemic (COVID-19 style)',
            'shocks': {
                'equities': -0.35,  # 35% drop in equities
                'bonds': 0.08,      # 8% increase in bonds (flight to safety)
                'commodities': -0.40,  # 40% drop in commodities (demand collapse)
                'real_estate': -0.25,  # 25% drop in real estate
                'volatility': 4.0   # 4x increase in volatility
            },
            'correlation_changes': {
                'equities_bonds': -0.8,  # Strong negative correlation
                'equities_commodities': 0.9  # Strong positive correlation
            },
            'duration': 70  # 70 days
        }
        
        # Tech bubble burst scenario
        self.scenarios['tech_bubble'] = {
            'description': 'Technology sector bubble burst',
            'shocks': {
                'equities': -0.30,  # 30% drop in equities
                'tech_equities': -0.60,  # 60% drop in tech equities
                'bonds': 0.05,      # 5% increase in bonds (flight to safety)
                'commodities': -0.10,  # 10% drop in commodities
                'real_estate': -0.15,  # 15% drop in real estate
                'volatility': 2.8   # 2.8x increase in volatility
            },
            'correlation_changes': {
                'equities_bonds': -0.5,  # Negative correlation
                'equities_commodities': 0.4  # Positive correlation
            },
            'duration': 50  # 50 days
        }
        
        # Tariff war scenario
        self.scenarios['tariff_war'] = {
            'description': 'Global tariff war',
            'shocks': {
                'equities': -0.20,  # 20% drop in equities
                'bonds': 0.03,      # 3% increase in bonds
                'commodities': -0.15,  # 15% drop in commodities
                'real_estate': -0.10,  # 10% drop in real estate
                'volatility': 2.0   # 2x increase in volatility
            },
            'correlation_changes': {
                'equities_bonds': -0.4,  # Negative correlation
                'equities_commodities': 0.6  # Positive correlation
            },
            'duration': 90  # 90 days
        }
    
    def add_custom_scenario(self, name, description, shocks, correlation_changes=None, duration=30):
        """
        Add a custom stress testing scenario.
        
        Parameters:
        -----------
        name : str
            Scenario name
        description : str
            Scenario description
        shocks : dict
            Dictionary of asset class shocks
        correlation_changes : dict, optional
            Dictionary of correlation changes
        duration : int, default=30
            Scenario duration in days
        """
        self.scenarios[name] = {
            'description': description,
            'shocks': shocks,
            'correlation_changes': correlation_changes if correlation_changes is not None else {},
            'duration': duration
        }
    
    def run_scenario(self, scenario_name, portfolio_weights=None, confidence_level=0.95, num_simulations=10000):
        """
        Run a stress testing scenario.
        
        Parameters:
        -----------
        scenario_name : str
            Name of the scenario to run
        portfolio_weights : array-like, optional
            Portfolio weights (if None, use equal weights)
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        num_simulations : int, default=10000
            Number of Monte Carlo simulations
            
        Returns:
        --------
        dict
            Dictionary containing stress test results
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        scenario = self.scenarios[scenario_name]
        
        # Get historical data
        returns = self.historical_data
        
        # Set portfolio weights if not provided
        if portfolio_weights is None:
            portfolio_weights = np.ones(returns.shape[1]) / returns.shape[1]
        
        # Calculate historical portfolio returns
        historical_portfolio_returns = returns.dot(portfolio_weights)
        
        # Calculate historical risk metrics
        historical_var = -np.percentile(historical_portfolio_returns, 100 * (1 - confidence_level))
        historical_es = -historical_portfolio_returns[historical_portfolio_returns <= -historical_var].mean()
        historical_volatility = historical_portfolio_returns.std()
        
        # Apply shocks to returns
        shocked_returns = returns.copy()
        
        # Apply asset class shocks
        for asset_class, shock in scenario['shocks'].items():
            if asset_class == 'volatility':
                # Volatility shock is handled separately
                continue
            
            # Find columns matching the asset class
            matching_columns = [col for col in returns.columns if asset_class.lower() in col.lower()]
            
            if not matching_columns and asset_class in returns.columns:
                matching_columns = [asset_class]
            
            # Apply shock to matching columns
            for col in matching_columns:
                shocked_returns[col] = returns[col] * (1 + shock)
        
        # Apply volatility shock
        if 'volatility' in scenario['shocks']:
            volatility_multiplier = scenario['shocks']['volatility']
            
            # Calculate current volatilities
            current_volatilities = returns.std()
            
            # Generate random returns with increased volatility
            np.random.seed(42)  # For reproducibility
            
            # Create empty DataFrame for simulated returns
            simulated_returns = pd.DataFrame(index=range(num_simulations), columns=returns.columns)
            
            # Generate simulated returns with shocked volatility
            for col in returns.columns:
                mean_return = returns[col].mean()
                shocked_volatility = current_volatilities[col] * volatility_multiplier
                simulated_returns[col] = np.random.normal(mean_return, shocked_volatility, num_simulations)
            
            # Calculate simulated portfolio returns
            simulated_portfolio_returns = simulated_returns.dot(portfolio_weights)
            
            # Calculate stressed risk metrics
            stressed_var = -np.percentile(simulated_portfolio_returns, 100 * (1 - confidence_level))
            stressed_es = -simulated_portfolio_returns[simulated_portfolio_returns <= -stressed_var].mean()
            stressed_volatility = simulated_portfolio_returns.std()
        else:
            # Calculate stressed portfolio returns
            stressed_portfolio_returns = shocked_returns.dot(portfolio_weights)
            
            # Calculate stressed risk metrics
            stressed_var = -np.percentile(stressed_portfolio_returns, 100 * (1 - confidence_level))
            stressed_es = -stressed_portfolio_returns[stressed_portfolio_returns <= -stressed_var].mean()
            stressed_volatility = stressed_portfolio_returns.std()
        
        # Calculate changes in risk metrics
        var_change = (stressed_var - historical_var) / historical_var * 100
        es_change = (stressed_es - historical_es) / historical_es * 100
        volatility_change = (stressed_volatility - historical_volatility) / historical_volatility * 100
        
        # Store results
        results = {
            'scenario': scenario_name,
            'description': scenario['description'],
            'historical_var': historical_var,
            'stressed_var': stressed_var,
            'var_change_pct': var_change,
            'historical_es': historical_es,
            'stressed_es': stressed_es,
            'es_change_pct': es_change,
            'historical_volatility': historical_volatility,
            'stressed_volatility': stressed_volatility,
            'volatility_change_pct': volatility_change,
            'confidence_level': confidence_level,
            'duration': scenario['duration']
        }
        
        # Store results
        self.results[scenario_name] = results
        
        return results
    
    def run_all_scenarios(self, portfolio_weights=None, confidence_level=0.95, num_simulations=10000):
        """
        Run all defined stress testing scenarios.
        
        Parameters:
        -----------
        portfolio_weights : array-like, optional
            Portfolio weights (if None, use equal weights)
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        num_simulations : int, default=10000
            Number of Monte Carlo simulations
            
        Returns:
        --------
        dict
            Dictionary containing all stress test results
        """
        all_results = {}
        
        for scenario_name in self.scenarios:
            results = self.run_scenario(
                scenario_name,
                portfolio_weights=portfolio_weights,
                confidence_level=confidence_level,
                num_simulations=num_simulations
            )
            
            all_results[scenario_name] = results
        
        return all_results
    
    def get_worst_case_scenario(self, metric='var_change_pct'):
        """
        Get the worst-case scenario based on a specific metric.
        
        Parameters:
        -----------
        metric : str, default='var_change_pct'
            Metric to use for comparison ('var_change_pct', 'es_change_pct', 'volatility_change_pct')
            
        Returns:
        --------
        dict
            Worst-case scenario results
        """
        if not self.results:
            raise ValueError("No stress test results available")
        
        worst_scenario = None
        worst_value = -float('inf')
        
        for scenario_name, results in self.results.items():
            if metric in results and results[metric] > worst_value:
                worst_value = results[metric]
                worst_scenario = scenario_name
        
        if worst_scenario is None:
            raise ValueError(f"Metric '{metric}' not found in results")
        
        return self.results[worst_scenario]
    
    def estimate_risk(self, confidence_level=0.95, scenario=None):
        """
        Estimate risk metrics under a specific scenario.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for risk metrics
        scenario : str, optional
            Scenario name (if None, use worst-case scenario)
            
        Returns:
        --------
        dict
            Dictionary containing risk metrics
        """
        if not self.results:
            raise ValueError("No stress test results available")
        
        if scenario is None:
            # Use worst-case scenario
            scenario_results = self.get_worst_case_scenario()
        elif scenario in self.results:
            # Use specified scenario
            scenario_results = self.results[scenario]
        else:
            raise ValueError(f"Scenario '{scenario}' not found in results")
        
        # Extract risk metrics
        var = scenario_results['stressed_var']
        es = scenario_results['stressed_es']
        
        return {
            'VaR': var,
            'ES': es,
            'confidence_level': confidence_level,
            'scenario': scenario_results['scenario'],
            'description': scenario_results['description']
        }
