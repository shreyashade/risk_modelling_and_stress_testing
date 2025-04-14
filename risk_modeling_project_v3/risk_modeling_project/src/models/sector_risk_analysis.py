"""
Sector-Specific Risk Analysis

This module implements specialized risk analysis methods for different market sectors,
focusing on the unique risk characteristics and factors affecting technology,
energy, and financial sectors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings
warnings.filterwarnings('ignore')

class SectorRiskAnalysis:
    """
    Performs sector-specific risk analysis for different market sectors.
    
    This class provides specialized methods for analyzing risk factors,
    sensitivities, and characteristics unique to specific market sectors.
    """
    
    def __init__(self, returns_data, metadata, economic_data=None, sector_mapping=None):
        """
        Initialize the sector risk analysis.
        
        Parameters:
        -----------
        returns_data : pandas.DataFrame
            DataFrame containing asset returns
        metadata : dict
            Dictionary containing asset metadata
        economic_data : pandas.DataFrame, optional
            DataFrame containing economic indicator data
        sector_mapping : dict, optional
            Dictionary mapping assets to sectors
            If None, use sector information from metadata
        """
        self.returns_data = returns_data
        self.metadata = metadata
        self.economic_data = economic_data
        
        # Create sector mapping if not provided
        if sector_mapping is None:
            self.sector_mapping = {}
            for ticker, info in metadata.items():
                if ticker in returns_data.columns:
                    sector = info.get('sector', 'Unknown')
                    self.sector_mapping[ticker] = sector
        else:
            self.sector_mapping = sector_mapping
        
        # Create sector groups
        self.sector_groups = {}
        for ticker, sector in self.sector_mapping.items():
            if sector not in self.sector_groups:
                self.sector_groups[sector] = []
            self.sector_groups[sector].append(ticker)
        
        # Calculate sector returns
        self.sector_returns = self._calculate_sector_returns()
    
    def _calculate_sector_returns(self):
        """
        Calculate returns for each sector.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing sector returns
        """
        sector_returns = pd.DataFrame(index=self.returns_data.index)
        
        for sector, tickers in self.sector_groups.items():
            if tickers:
                # Calculate equal-weighted sector returns
                sector_returns[sector] = self.returns_data[tickers].mean(axis=1)
        
        return sector_returns
    
    def analyze_tech_sector(self, tech_tickers=None, factors=None):
        """
        Perform specialized analysis for the technology sector.
        
        Parameters:
        -----------
        tech_tickers : list, optional
            List of technology sector tickers
            If None, use tickers mapped to 'Technology' sector
        factors : dict, optional
            Dictionary of factor data for factor analysis
            
        Returns:
        --------
        dict
            Dictionary containing technology sector analysis results
        """
        # Get technology sector tickers
        if tech_tickers is None:
            tech_tickers = self.sector_groups.get('Technology', [])
            
            # If no tickers mapped to 'Technology', try to find tech tickers
            if not tech_tickers:
                tech_keywords = ['tech', 'software', 'hardware', 'semiconductor', 'internet']
                for ticker, info in self.metadata.items():
                    if ticker in self.returns_data.columns:
                        sector = info.get('sector', '').lower()
                        industry = info.get('industry', '').lower()
                        
                        # Check if sector or industry contains tech keywords
                        if any(keyword in sector for keyword in tech_keywords) or \
                           any(keyword in industry for keyword in tech_keywords):
                            tech_tickers.append(ticker)
        
        if not tech_tickers:
            print("No technology sector tickers found.")
            return {}
        
        print(f"Analyzing technology sector with {len(tech_tickers)} tickers.")
        
        # Extract tech returns
        tech_returns = self.returns_data[tech_tickers]
        
        # Calculate tech sector statistics
        tech_stats = self._calculate_sector_statistics(tech_returns)
        
        # Perform tech-specific analyses
        
        # 1. Innovation cycle sensitivity
        # For tech companies, analyze sensitivity to innovation cycles
        # This is a simplified approach using rolling volatility as a proxy
        rolling_vol = tech_returns.rolling(window=60).std() * np.sqrt(252)
        innovation_sensitivity = rolling_vol.mean()
        
        # 2. Growth vs. value segmentation
        # Segment tech stocks into growth vs. value categories
        # This is a simplified approach using return patterns
        growth_value_scores = {}
        
        for ticker in tech_tickers:
            # Calculate metrics that differentiate growth from value
            returns = tech_returns[ticker]
            
            # Growth stocks typically have higher volatility
            volatility = returns.std() * np.sqrt(252)
            
            # Growth stocks typically have more positive skew
            skew = stats.skew(returns.dropna())
            
            # Combine into a growth score (higher = more growth-oriented)
            growth_score = volatility * (1 + skew)
            growth_value_scores[ticker] = growth_score
        
        # Normalize scores to 0-1 range
        min_score = min(growth_value_scores.values())
        max_score = max(growth_value_scores.values())
        score_range = max_score - min_score
        
        if score_range > 0:
            normalized_scores = {ticker: (score - min_score) / score_range 
                               for ticker, score in growth_value_scores.items()}
        else:
            normalized_scores = {ticker: 0.5 for ticker in growth_value_scores}
        
        # Classify as growth (>0.6), blend (0.4-0.6), or value (<0.4)
        classifications = {}
        for ticker, score in normalized_scores.items():
            if score > 0.6:
                classifications[ticker] = 'Growth'
            elif score < 0.4:
                classifications[ticker] = 'Value'
            else:
                classifications[ticker] = 'Blend'
        
        # 3. Technology subsector analysis
        # Group tech stocks into subsectors
        subsectors = {}
        
        for ticker in tech_tickers:
            if ticker in self.metadata:
                industry = self.metadata[ticker].get('industry', 'Unknown')
                if industry not in subsectors:
                    subsectors[industry] = []
                subsectors[industry].append(ticker)
        
        # Calculate subsector returns
        subsector_returns = pd.DataFrame(index=tech_returns.index)
        
        for subsector, tickers in subsectors.items():
            if tickers:
                subsector_returns[subsector] = tech_returns[tickers].mean(axis=1)
        
        # Calculate subsector statistics
        subsector_stats = {}
        
        for subsector in subsectors:
            if subsector in subsector_returns.columns:
                subsector_stats[subsector] = self._calculate_return_statistics(subsector_returns[subsector])
        
        # 4. Technology factor exposures
        # Analyze exposure to common tech factors
        factor_exposures = {}
        
        if factors is not None and self.economic_data is not None:
            # Align economic data with returns
            aligned_data = self.economic_data.reindex(tech_returns.index)
            
            # For each tech stock, calculate factor exposures
            for ticker in tech_tickers:
                # Run regression of returns on factors
                y = tech_returns[ticker].dropna()
                X = aligned_data.loc[y.index].dropna()
                
                if not X.empty and len(X) == len(y):
                    # Add constant
                    X = sm.add_constant(X)
                    
                    # Run regression
                    model = sm.OLS(y, X).fit()
                    
                    # Store factor exposures
                    factor_exposures[ticker] = model.params.to_dict()
        
        # 5. Disruption risk analysis
        # Analyze vulnerability to technological disruption
        # This is a simplified approach using return correlations
        disruption_risk = {}
        
        # Calculate correlation matrix
        corr_matrix = tech_returns.corr()
        
        for ticker in tech_tickers:
            # Calculate average correlation with other tech stocks
            correlations = corr_matrix[ticker].drop(ticker)
            avg_corr = correlations.mean()
            
            # Lower correlation might indicate more unique business model
            # which could be less vulnerable to common disruption
            disruption_risk[ticker] = 1 - avg_corr
        
        # Return results
        return {
            'sector_stats': tech_stats,
            'innovation_sensitivity': innovation_sensitivity,
            'growth_value_scores': normalized_scores,
            'classifications': classifications,
            'subsector_returns': subsector_returns,
            'subsector_stats': subsector_stats,
            'factor_exposures': factor_exposures,
            'disruption_risk': disruption_risk
        }
    
    def analyze_energy_sector(self, energy_tickers=None, include_commodity_data=True):
        """
        Perform specialized analysis for the energy sector.
        
        Parameters:
        -----------
        energy_tickers : list, optional
            List of energy sector tickers
            If None, use tickers mapped to 'Energy' sector
        include_commodity_data : bool, default=True
            Whether to include commodity price data in analysis
            
        Returns:
        --------
        dict
            Dictionary containing energy sector analysis results
        """
        # Get energy sector tickers
        if energy_tickers is None:
            energy_tickers = self.sector_groups.get('Energy', [])
            
            # If no tickers mapped to 'Energy', try to find energy tickers
            if not energy_tickers:
                energy_keywords = ['energy', 'oil', 'gas', 'petroleum', 'drilling', 'coal']
                for ticker, info in self.metadata.items():
                    if ticker in self.returns_data.columns:
                        sector = info.get('sector', '').lower()
                        industry = info.get('industry', '').lower()
                        
                        # Check if sector or industry contains energy keywords
                        if any(keyword in sector for keyword in energy_keywords) or \
                           any(keyword in industry for keyword in energy_keywords):
                            energy_tickers.append(ticker)
        
        if not energy_tickers:
            print("No energy sector tickers found.")
            return {}
        
        print(f"Analyzing energy sector with {len(energy_tickers)} tickers.")
        
        # Extract energy returns
        energy_returns = self.returns_data[energy_tickers]
        
        # Calculate energy sector statistics
        energy_stats = self._calculate_sector_statistics(energy_returns)
        
        # Perform energy-specific analyses
        
        # 1. Commodity price sensitivity
        commodity_sensitivity = {}
        
        if include_commodity_data and self.economic_data is not None:
            # Look for commodity price data in economic data
            commodity_cols = [col for col in self.economic_data.columns 
                             if any(c in col.lower() for c in ['oil', 'gas', 'wti', 'brent', 'crude'])]
            
            if commodity_cols:
                # Align commodity data with returns
                aligned_data = self.economic_data[commodity_cols].reindex(energy_returns.index)
                
                # For each energy stock, calculate commodity price sensitivity
                for ticker in energy_tickers:
                    # Run regression of returns on commodity prices
                    y = energy_returns[ticker].dropna()
                    X = aligned_data.loc[y.index].dropna()
                    
                    if not X.empty and len(X) == len(y):
                        # Add constant
                        X = sm.add_constant(X)
                        
                        # Run regression
                        model = sm.OLS(y, X).fit()
                        
                        # Store commodity sensitivities
                        commodity_sensitivity[ticker] = model.params.to_dict()
        
        # 2. Energy subsector analysis
        # Group energy stocks into subsectors
        subsectors = {}
        
        for ticker in energy_tickers:
            if ticker in self.metadata:
                industry = self.metadata[ticker].get('industry', 'Unknown')
                if industry not in subsectors:
                    subsectors[industry] = []
                subsectors[industry].append(ticker)
        
        # Calculate subsector returns
        subsector_returns = pd.DataFrame(index=energy_returns.index)
        
        for subsector, tickers in subsectors.items():
            if tickers:
                subsector_returns[subsector] = energy_returns[tickers].mean(axis=1)
        
        # Calculate subsector statistics
        subsector_stats = {}
        
        for subsector in subsectors:
            if subsector in subsector_returns.columns:
                subsector_stats[subsector] = self._calculate_return_statistics(subsector_returns[subsector])
        
        # 3. Geopolitical risk exposure
        # This is a simplified approach using volatility during known geopolitical events
        geopolitical_events = {
            'Ukraine Invasion': ('2022-02-20', '2022-03-20'),
            'OPEC+ Dispute': ('2020-03-01', '2020-04-15'),
            'Iran Tensions': ('2019-09-15', '2019-10-15'),
            'Saudi Oil Attack': ('2019-09-14', '2019-09-30')
        }
        
        geopolitical_exposure = {}
        
        for ticker in energy_tickers:
            event_volatilities = []
            
            for event, (start_date, end_date) in geopolitical_events.items():
                # Check if dates are in returns data
                if start_date in energy_returns.index and end_date in energy_returns.index:
                    # Extract returns during event
                    event_returns = energy_returns.loc[start_date:end_date, ticker]
                    
                    # Calculate volatility during event
                    event_vol = event_returns.std() * np.sqrt(252)
                    event_volatilities.append(event_vol)
            
            if event_volatilities:
                # Average volatility during geopolitical events
                geopolitical_exposure[ticker] = np.mean(event_volatilities)
            else:
                geopolitical_exposure[ticker] = np.nan
        
        # 4. Energy transition risk
        # Analyze vulnerability to energy transition
        # This is a simplified approach using business descriptions
        transition_risk = {}
        
        for ticker in energy_tickers:
            if ticker in self.metadata:
                # Look for keywords related to renewable energy
                description = str(self.metadata[ticker].get('longBusinessSummary', '')).lower()
                
                # Count renewable energy keywords
                renewable_keywords = ['renewable', 'solar', 'wind', 'clean energy', 'sustainable']
                renewable_count = sum(description.count(keyword) for keyword in renewable_keywords)
                
                # Count fossil fuel keywords
                fossil_keywords = ['oil', 'gas', 'petroleum', 'coal', 'drilling', 'refining']
                fossil_count = sum(description.count(keyword) for keyword in fossil_keywords)
                
                # Calculate transition risk score
                if renewable_count + fossil_count > 0:
                    # Higher score = higher risk (more fossil fuel exposure)
                    transition_risk[ticker] = fossil_count / (renewable_count + fossil_count)
                else:
                    transition_risk[ticker] = 0.5  # Default value
            else:
                transition_risk[ticker] = 0.5  # Default value
        
        # 5. Seasonal patterns
        # Analyze seasonal patterns in energy returns
        monthly_returns = {}
        
        for ticker in energy_tickers:
            # Group returns by month
            returns = energy_returns[ticker]
            returns_by_month = returns.groupby(returns.index.month)
            
            # Calculate average return for each month
            monthly_avg = returns_by_month.mean() * 100  # Convert to percentage
            
            monthly_returns[ticker] = monthly_avg
        
        # Calculate sector-wide monthly returns
        sector_monthly = pd.DataFrame(monthly_returns).mean(axis=1)
        
        # Return results
        return {
            'sector_stats': energy_stats,
            'commodity_sensitivity': commodity_sensitivity,
            'subsector_returns': subsector_returns,
            'subsector_stats': subsector_stats,
            'geopolitical_exposure': geopolitical_exposure,
            'transition_risk': transition_risk,
            'monthly_returns': monthly_returns,
            'sector_monthly_returns': sector_monthly
        }
    
    def analyze_financial_sector(self, financial_tickers=None, include_rate_data=True):
        """
        Perform specialized analysis for the financial sector.
        
        Parameters:
        -----------
        financial_tickers : list, optional
            List of financial sector tickers
            If None, use tickers mapped to 'Financial' or 'Financials' sector
        include_rate_data : bool, default=True
            Whether to include interest rate data in analysis
            
        Returns:
        --------
        dict
            Dictionary containing financial sector analysis results
        """
        # Get financial sector tickers
        if financial_tickers is None:
            financial_tickers = self.sector_groups.get('Financial', [])
            
            # Also check for 'Financials' (plural form)
            if 'Financials' in self.sector_groups:
                financial_tickers.extend(self.sector_groups['Financials'])
            
            # If no tickers mapped to financial sectors, try to find financial tickers
            if not financial_tickers:
                financial_keywords = ['bank', 'financial', 'insurance', 'asset management', 'broker']
                for ticker, info in self.metadata.items():
                    if ticker in self.returns_data.columns:
                        sector = info.get('sector', '').lower()
                        industry = info.get('industry', '').lower()
                        
                        # Check if sector or industry contains financial keywords
                        if any(keyword in sector for keyword in financial_keywords) or \
                           any(keyword in industry for keyword in financial_keywords):
                            financial_tickers.append(ticker)
        
        if not financial_tickers:
            print("No financial sector tickers found.")
            return {}
        
        print(f"Analyzing financial sector with {len(financial_tickers)} tickers.")
        
        # Extract financial returns
        financial_returns = self.returns_data[financial_tickers]
        
        # Calculate financial sector statistics
        financial_stats = self._calculate_sector_statistics(financial_returns)
        
        # Perform financial-specific analyses
        
        # 1. Interest rate sensitivity
        rate_sensitivity = {}
        
        if include_rate_data and self.economic_data is not None:
            # Look for interest rate data in economic data
            rate_cols = [col for col in self.economic_data.columns 
                        if any(r in col.lower() for r in ['rate', 'yield', 'treasury', 'libor', 'sofr'])]
            
            if rate_cols:
                # Align rate data with returns
                aligned_data = self.economic_data[rate_cols].reindex(financial_returns.index)
                
                # Calculate changes in rates
                rate_changes = aligned_data.pct_change()
                
                # For each financial stock, calculate rate sensitivity
                for ticker in financial_tickers:
                    # Run regression of returns on rate changes
                    y = financial_returns[ticker].dropna()
                    X = rate_changes.loc[y.index].dropna()
                    
                    if not X.empty and len(X) == len(y):
                        # Add constant
                        X = sm.add_constant(X)
                        
                        # Run regression
                        model = sm.OLS(y, X).fit()
                        
                        # Store rate sensitivities
                        rate_sensitivity[ticker] = model.params.to_dict()
        
        # 2. Yield curve sensitivity
        # Analyze sensitivity to yield curve changes
        yield_curve_sensitivity = {}
        
        if include_rate_data and self.economic_data is not None:
            # Look for treasury yield data
            short_term_cols = [col for col in self.economic_data.columns 
                              if any(t in col.lower() for t in ['3m', '6m', '1y', 'short'])]
            
            long_term_cols = [col for col in self.economic_data.columns 
                             if any(t in col.lower() for t in ['10y', '20y', '30y', 'long'])]
            
            if short_term_cols and long_term_cols:
                # Calculate yield curve slope
                short_rates = self.economic_data[short_term_cols].mean(axis=1)
                long_rates = self.economic_data[long_term_cols].mean(axis=1)
                yield_curve = long_rates - short_rates
                
                # Align yield curve with returns
                aligned_curve = yield_curve.reindex(financial_returns.index)
                
                # Calculate changes in yield curve
                curve_changes = aligned_curve.diff()
                
                # For each financial stock, calculate yield curve sensitivity
                for ticker in financial_tickers:
                    # Run regression of returns on yield curve changes
                    y = financial_returns[ticker].dropna()
                    X = pd.DataFrame({'yield_curve': curve_changes.loc[y.index].dropna()})
                    
                    if not X.empty and len(X) == len(y):
                        # Add constant
                        X = sm.add_constant(X)
                        
                        # Run regression
                        model = sm.OLS(y, X).fit()
                        
                        # Store yield curve sensitivity
                        yield_curve_sensitivity[ticker] = model.params.get('yield_curve', 0)
        
        # 3. Financial subsector analysis
        # Group financial stocks into subsectors
        subsectors = {}
        
        for ticker in financial_tickers:
            if ticker in self.metadata:
                industry = self.metadata[ticker].get('industry', 'Unknown')
                if industry not in subsectors:
                    subsectors[industry] = []
                subsectors[industry].append(ticker)
        
        # Calculate subsector returns
        subsector_returns = pd.DataFrame(index=financial_returns.index)
        
        for subsector, tickers in subsectors.items():
            if tickers:
                subsector_returns[subsector] = financial_returns[tickers].mean(axis=1)
        
        # Calculate subsector statistics
        subsector_stats = {}
        
        for subsector in subsectors:
            if subsector in subsector_returns.columns:
                subsector_stats[subsector] = self._calculate_return_statistics(subsector_returns[subsector])
        
        # 4. Credit cycle sensitivity
        # Analyze sensitivity to credit cycles
        # This is a simplified approach using economic indicators
        credit_sensitivity = {}
        
        if self.economic_data is not None:
            # Look for credit-related indicators
            credit_cols = [col for col in self.economic_data.columns 
                          if any(c in col.lower() for c in ['credit', 'spread', 'default', 'loan'])]
            
            if credit_cols:
                # Align credit data with returns
                aligned_data = self.economic_data[credit_cols].reindex(financial_returns.index)
                
                # For each financial stock, calculate credit sensitivity
                for ticker in financial_tickers:
                    # Run regression of returns on credit indicators
                    y = financial_returns[ticker].dropna()
                    X = aligned_data.loc[y.index].dropna()
                    
                    if not X.empty and len(X) == len(y):
                        # Add constant
                        X = sm.add_constant(X)
                        
                        # Run regression
                        model = sm.OLS(y, X).fit()
                        
                        # Store credit sensitivities
                        credit_sensitivity[ticker] = model.params.to_dict()
        
        # 5. Regulatory risk analysis
        # Analyze vulnerability to regulatory changes
        # This is a simplified approach using size as a proxy
        regulatory_risk = {}
        
        for ticker in financial_tickers:
            if ticker in self.metadata:
                # Use market cap as a proxy for regulatory risk
                # Larger institutions often face more regulatory scrutiny
                market_cap = self.metadata[ticker].get('marketCap', 0)
                
                if market_cap:
                    # Normalize to 0-1 scale
                    regulatory_risk[ticker] = market_cap
        
        # Normalize regulatory risk scores
        if regulatory_risk:
            min_risk = min(regulatory_risk.values())
            max_risk = max(regulatory_risk.values())
            risk_range = max_risk - min_risk
            
            if risk_range > 0:
                regulatory_risk = {ticker: (risk - min_risk) / risk_range 
                                 for ticker, risk in regulatory_risk.items()}
        
        # 6. Systemic risk contribution
        # Analyze contribution to systemic risk
        # This is a simplified approach using correlations
        systemic_risk = {}
        
        # Calculate correlation with market
        market_returns = self.returns_data.mean(axis=1)
        
        for ticker in financial_tickers:
            # Calculate correlation with market
            correlation = financial_returns[ticker].corr(market_returns)
            
            # Calculate beta
            cov_with_market = financial_returns[ticker].cov(market_returns)
            market_var = market_returns.var()
            
            if market_var > 0:
                beta = cov_with_market / market_var
            else:
                beta = 1.0
            
            # Combine correlation and beta for systemic risk score
            systemic_risk[ticker] = correlation * beta
        
        # Return results
        return {
            'sector_stats': financial_stats,
            'rate_sensitivity': rate_sensitivity,
            'yield_curve_sensitivity': yield_curve_sensitivity,
            'subsector_returns': subsector_returns,
            'subsector_stats': subsector_stats,
            'credit_sensitivity': credit_sensitivity,
            'regulatory_risk': regulatory_risk,
            'systemic_risk': systemic_risk
        }
    
    def _calculate_sector_statistics(self, returns_data):
        """
        Calculate comprehensive statistics for a sector.
        
        Parameters:
        -----------
        returns_data : pandas.DataFrame
            DataFrame containing sector returns
            
        Returns:
        --------
        dict
            Dictionary containing sector statistics
        """
        # Calculate portfolio return (equal-weighted)
        portfolio_returns = returns_data.mean(axis=1)
        
        # Calculate basic statistics
        basic_stats = self._calculate_return_statistics(portfolio_returns)
        
        # Calculate correlation matrix
        correlation = returns_data.corr()
        
        # Calculate principal components
        pca = PCA()
        pca.fit(returns_data.dropna())
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        
        # Calculate factor loadings
        loadings = pca.components_
        
        # Return statistics
        return {
            'basic_stats': basic_stats,
            'correlation': correlation,
            'explained_variance': explained_variance,
            'factor_loadings': loadings
        }
    
    def _calculate_return_statistics(self, returns):
        """
        Calculate comprehensive return statistics.
        
        Parameters:
        -----------
        returns : pandas.Series
            Series containing returns
            
        Returns:
        --------
        dict
            Dictionary containing return statistics
        """
        # Drop NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            return {}
        
        # Calculate basic statistics
        mean_return = returns.mean()
        median_return = returns.median()
        std_dev = returns.std()
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Calculate annualized statistics
        ann_return = mean_return * 252
        ann_volatility = std_dev * np.sqrt(252)
        sharpe_ratio = ann_return / ann_volatility if ann_volatility > 0 else 0
        
        # Calculate drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate downside risk
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = ann_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Calculate autocorrelation
        if len(returns) > 1:
            autocorr = returns.autocorr(lag=1)
        else:
            autocorr = 0
        
        # Return statistics
        return {
            'mean_return': mean_return,
            'median_return': median_return,
            'std_dev': std_dev,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'ann_return': ann_return,
            'ann_volatility': ann_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'downside_deviation': downside_deviation,
            'sortino_ratio': sortino_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'autocorr': autocorr
        }
    
    def plot_sector_comparison(self, figsize=(12, 10), title='Sector Comparison'):
        """
        Plot comparison of different sectors.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 10)
            Figure size
        title : str, default='Sector Comparison'
            Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot sector returns
        cum_returns = (1 + self.sector_returns).cumprod()
        cum_returns.plot(ax=axes[0, 0])
        
        # Set labels and title for first subplot
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].set_title('Sector Returns')
        axes[0, 0].legend(loc='upper left')
        axes[0, 0].grid(True)
        
        # Calculate sector statistics
        sector_stats = {}
        
        for sector in self.sector_returns.columns:
            sector_stats[sector] = self._calculate_return_statistics(self.sector_returns[sector])
        
        # Extract statistics for plotting
        sectors = list(sector_stats.keys())
        returns = [sector_stats[sector]['ann_return'] for sector in sectors]
        volatilities = [sector_stats[sector]['ann_volatility'] for sector in sectors]
        sharpes = [sector_stats[sector]['sharpe_ratio'] for sector in sectors]
        drawdowns = [sector_stats[sector]['max_drawdown'] for sector in sectors]
        
        # Plot risk-return scatter
        axes[0, 1].scatter(volatilities, returns)
        
        # Add sector labels
        for i, sector in enumerate(sectors):
            axes[0, 1].annotate(sector, (volatilities[i], returns[i]))
        
        # Set labels and title for second subplot
        axes[0, 1].set_xlabel('Volatility')
        axes[0, 1].set_ylabel('Return')
        axes[0, 1].set_title('Risk-Return by Sector')
        axes[0, 1].grid(True)
        
        # Plot Sharpe ratios
        axes[1, 0].bar(sectors, sharpes)
        
        # Set labels and title for third subplot
        axes[1, 0].set_xlabel('Sector')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].set_title('Sector Sharpe Ratios')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True)
        
        # Plot maximum drawdowns
        axes[1, 1].bar(sectors, drawdowns)
        
        # Set labels and title for fourth subplot
        axes[1, 1].set_xlabel('Sector')
        axes[1, 1].set_ylabel('Maximum Drawdown')
        axes[1, 1].set_title('Sector Maximum Drawdowns')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        return fig
    
    def plot_tech_analysis(self, tech_analysis, figsize=(12, 15), title='Technology Sector Analysis'):
        """
        Plot technology sector analysis results.
        
        Parameters:
        -----------
        tech_analysis : dict
            Dictionary containing technology sector analysis results
        figsize : tuple, default=(12, 15)
            Figure size
        title : str, default='Technology Sector Analysis'
            Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not tech_analysis:
            print("No technology sector analysis results to plot.")
            return None
        
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # Plot subsector returns
        if 'subsector_returns' in tech_analysis:
            subsector_returns = tech_analysis['subsector_returns']
            cum_returns = (1 + subsector_returns).cumprod()
            cum_returns.plot(ax=axes[0, 0])
            
            # Set labels and title
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Cumulative Return')
            axes[0, 0].set_title('Technology Subsector Returns')
            axes[0, 0].legend(loc='upper left')
            axes[0, 0].grid(True)
        
        # Plot growth vs. value classification
        if 'classifications' in tech_analysis:
            classifications = tech_analysis['classifications']
            
            # Count stocks in each category
            categories = {'Growth': 0, 'Blend': 0, 'Value': 0}
            for _, category in classifications.items():
                categories[category] += 1
            
            # Plot as pie chart
            axes[0, 1].pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%')
            axes[0, 1].set_title('Growth vs. Value Classification')
        
        # Plot innovation sensitivity
        if 'innovation_sensitivity' in tech_analysis:
            innovation_sensitivity = tech_analysis['innovation_sensitivity']
            
            # Sort by sensitivity
            sorted_sensitivity = innovation_sensitivity.sort_values(ascending=False)
            
            # Plot as bar chart
            axes[1, 0].bar(sorted_sensitivity.index, sorted_sensitivity.values)
            
            # Set labels and title
            axes[1, 0].set_xlabel('Ticker')
            axes[1, 0].set_ylabel('Innovation Sensitivity')
            axes[1, 0].set_title('Innovation Cycle Sensitivity')
            axes[1, 0].tick_params(axis='x', rotation=90)
            axes[1, 0].grid(True, axis='y')
        
        # Plot disruption risk
        if 'disruption_risk' in tech_analysis:
            disruption_risk = tech_analysis['disruption_risk']
            
            # Sort by risk
            sorted_risk = pd.Series(disruption_risk).sort_values(ascending=False)
            
            # Plot as bar chart
            axes[1, 1].bar(sorted_risk.index, sorted_risk.values)
            
            # Set labels and title
            axes[1, 1].set_xlabel('Ticker')
            axes[1, 1].set_ylabel('Disruption Risk')
            axes[1, 1].set_title('Technological Disruption Risk')
            axes[1, 1].tick_params(axis='x', rotation=90)
            axes[1, 1].grid(True, axis='y')
        
        # Plot subsector statistics
        if 'subsector_stats' in tech_analysis:
            subsector_stats = tech_analysis['subsector_stats']
            
            # Extract statistics for plotting
            subsectors = list(subsector_stats.keys())
            returns = [subsector_stats[subsector]['ann_return'] for subsector in subsectors]
            volatilities = [subsector_stats[subsector]['ann_volatility'] for subsector in subsectors]
            
            # Plot as scatter
            axes[2, 0].scatter(volatilities, returns)
            
            # Add subsector labels
            for i, subsector in enumerate(subsectors):
                axes[2, 0].annotate(subsector, (volatilities[i], returns[i]))
            
            # Set labels and title
            axes[2, 0].set_xlabel('Volatility')
            axes[2, 0].set_ylabel('Return')
            axes[2, 0].set_title('Risk-Return by Technology Subsector')
            axes[2, 0].grid(True)
        
        # Plot factor exposures
        if 'factor_exposures' in tech_analysis:
            factor_exposures = tech_analysis['factor_exposures']
            
            if factor_exposures:
                # Extract first ticker's factors as example
                first_ticker = list(factor_exposures.keys())[0]
                factors = list(factor_exposures[first_ticker].keys())
                
                # Remove constant
                if 'const' in factors:
                    factors.remove('const')
                
                if factors:
                    # Calculate average exposure to each factor
                    avg_exposures = {}
                    
                    for factor in factors:
                        exposures = [exposure.get(factor, 0) for exposure in factor_exposures.values()]
                        avg_exposures[factor] = np.mean(exposures)
                    
                    # Sort by absolute exposure
                    sorted_exposures = pd.Series(avg_exposures).sort_values(key=abs, ascending=False)
                    
                    # Plot as bar chart
                    axes[2, 1].bar(sorted_exposures.index, sorted_exposures.values)
                    
                    # Set labels and title
                    axes[2, 1].set_xlabel('Factor')
                    axes[2, 1].set_ylabel('Average Exposure')
                    axes[2, 1].set_title('Technology Sector Factor Exposures')
                    axes[2, 1].tick_params(axis='x', rotation=45)
                    axes[2, 1].grid(True, axis='y')
        
        plt.tight_layout()
        
        return fig
    
    def plot_energy_analysis(self, energy_analysis, figsize=(12, 15), title='Energy Sector Analysis'):
        """
        Plot energy sector analysis results.
        
        Parameters:
        -----------
        energy_analysis : dict
            Dictionary containing energy sector analysis results
        figsize : tuple, default=(12, 15)
            Figure size
        title : str, default='Energy Sector Analysis'
            Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not energy_analysis:
            print("No energy sector analysis results to plot.")
            return None
        
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # Plot subsector returns
        if 'subsector_returns' in energy_analysis:
            subsector_returns = energy_analysis['subsector_returns']
            cum_returns = (1 + subsector_returns).cumprod()
            cum_returns.plot(ax=axes[0, 0])
            
            # Set labels and title
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Cumulative Return')
            axes[0, 0].set_title('Energy Subsector Returns')
            axes[0, 0].legend(loc='upper left')
            axes[0, 0].grid(True)
        
        # Plot commodity sensitivity
        if 'commodity_sensitivity' in energy_analysis:
            commodity_sensitivity = energy_analysis['commodity_sensitivity']
            
            if commodity_sensitivity:
                # Extract first ticker's factors as example
                first_ticker = list(commodity_sensitivity.keys())[0]
                commodities = list(commodity_sensitivity[first_ticker].keys())
                
                # Remove constant
                if 'const' in commodities:
                    commodities.remove('const')
                
                if commodities:
                    # Calculate average sensitivity to each commodity
                    avg_sensitivities = {}
                    
                    for commodity in commodities:
                        sensitivities = [sensitivity.get(commodity, 0) for sensitivity in commodity_sensitivity.values()]
                        avg_sensitivities[commodity] = np.mean(sensitivities)
                    
                    # Sort by absolute sensitivity
                    sorted_sensitivities = pd.Series(avg_sensitivities).sort_values(key=abs, ascending=False)
                    
                    # Plot as bar chart
                    axes[0, 1].bar(sorted_sensitivities.index, sorted_sensitivities.values)
                    
                    # Set labels and title
                    axes[0, 1].set_xlabel('Commodity')
                    axes[0, 1].set_ylabel('Average Sensitivity')
                    axes[0, 1].set_title('Commodity Price Sensitivity')
                    axes[0, 1].tick_params(axis='x', rotation=45)
                    axes[0, 1].grid(True, axis='y')
        
        # Plot geopolitical exposure
        if 'geopolitical_exposure' in energy_analysis:
            geopolitical_exposure = energy_analysis['geopolitical_exposure']
            
            # Sort by exposure
            sorted_exposure = pd.Series(geopolitical_exposure).sort_values(ascending=False)
            
            # Plot as bar chart
            axes[1, 0].bar(sorted_exposure.index, sorted_exposure.values)
            
            # Set labels and title
            axes[1, 0].set_xlabel('Ticker')
            axes[1, 0].set_ylabel('Geopolitical Exposure')
            axes[1, 0].set_title('Geopolitical Risk Exposure')
            axes[1, 0].tick_params(axis='x', rotation=90)
            axes[1, 0].grid(True, axis='y')
        
        # Plot transition risk
        if 'transition_risk' in energy_analysis:
            transition_risk = energy_analysis['transition_risk']
            
            # Sort by risk
            sorted_risk = pd.Series(transition_risk).sort_values(ascending=False)
            
            # Plot as bar chart
            axes[1, 1].bar(sorted_risk.index, sorted_risk.values)
            
            # Set labels and title
            axes[1, 1].set_xlabel('Ticker')
            axes[1, 1].set_ylabel('Transition Risk')
            axes[1, 1].set_title('Energy Transition Risk')
            axes[1, 1].tick_params(axis='x', rotation=90)
            axes[1, 1].grid(True, axis='y')
        
        # Plot subsector statistics
        if 'subsector_stats' in energy_analysis:
            subsector_stats = energy_analysis['subsector_stats']
            
            # Extract statistics for plotting
            subsectors = list(subsector_stats.keys())
            returns = [subsector_stats[subsector]['ann_return'] for subsector in subsectors]
            volatilities = [subsector_stats[subsector]['ann_volatility'] for subsector in subsectors]
            
            # Plot as scatter
            axes[2, 0].scatter(volatilities, returns)
            
            # Add subsector labels
            for i, subsector in enumerate(subsectors):
                axes[2, 0].annotate(subsector, (volatilities[i], returns[i]))
            
            # Set labels and title
            axes[2, 0].set_xlabel('Volatility')
            axes[2, 0].set_ylabel('Return')
            axes[2, 0].set_title('Risk-Return by Energy Subsector')
            axes[2, 0].grid(True)
        
        # Plot seasonal patterns
        if 'sector_monthly_returns' in energy_analysis:
            monthly_returns = energy_analysis['sector_monthly_returns']
            
            # Plot as bar chart
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            axes[2, 1].bar(month_names, monthly_returns.values)
            
            # Set labels and title
            axes[2, 1].set_xlabel('Month')
            axes[2, 1].set_ylabel('Average Return (%)')
            axes[2, 1].set_title('Seasonal Patterns in Energy Returns')
            axes[2, 1].grid(True, axis='y')
        
        plt.tight_layout()
        
        return fig
    
    def plot_financial_analysis(self, financial_analysis, figsize=(12, 15), title='Financial Sector Analysis'):
        """
        Plot financial sector analysis results.
        
        Parameters:
        -----------
        financial_analysis : dict
            Dictionary containing financial sector analysis results
        figsize : tuple, default=(12, 15)
            Figure size
        title : str, default='Financial Sector Analysis'
            Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not financial_analysis:
            print("No financial sector analysis results to plot.")
            return None
        
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # Plot subsector returns
        if 'subsector_returns' in financial_analysis:
            subsector_returns = financial_analysis['subsector_returns']
            cum_returns = (1 + subsector_returns).cumprod()
            cum_returns.plot(ax=axes[0, 0])
            
            # Set labels and title
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Cumulative Return')
            axes[0, 0].set_title('Financial Subsector Returns')
            axes[0, 0].legend(loc='upper left')
            axes[0, 0].grid(True)
        
        # Plot interest rate sensitivity
        if 'rate_sensitivity' in financial_analysis:
            rate_sensitivity = financial_analysis['rate_sensitivity']
            
            if rate_sensitivity:
                # Extract first ticker's factors as example
                first_ticker = list(rate_sensitivity.keys())[0]
                rates = list(rate_sensitivity[first_ticker].keys())
                
                # Remove constant
                if 'const' in rates:
                    rates.remove('const')
                
                if rates:
                    # Calculate average sensitivity to each rate
                    avg_sensitivities = {}
                    
                    for rate in rates:
                        sensitivities = [sensitivity.get(rate, 0) for sensitivity in rate_sensitivity.values()]
                        avg_sensitivities[rate] = np.mean(sensitivities)
                    
                    # Sort by absolute sensitivity
                    sorted_sensitivities = pd.Series(avg_sensitivities).sort_values(key=abs, ascending=False)
                    
                    # Plot as bar chart
                    axes[0, 1].bar(sorted_sensitivities.index, sorted_sensitivities.values)
                    
                    # Set labels and title
                    axes[0, 1].set_xlabel('Interest Rate')
                    axes[0, 1].set_ylabel('Average Sensitivity')
                    axes[0, 1].set_title('Interest Rate Sensitivity')
                    axes[0, 1].tick_params(axis='x', rotation=45)
                    axes[0, 1].grid(True, axis='y')
        
        # Plot yield curve sensitivity
        if 'yield_curve_sensitivity' in financial_analysis:
            yield_curve_sensitivity = financial_analysis['yield_curve_sensitivity']
            
            # Sort by sensitivity
            sorted_sensitivity = pd.Series(yield_curve_sensitivity).sort_values()
            
            # Plot as bar chart
            axes[1, 0].bar(sorted_sensitivity.index, sorted_sensitivity.values)
            
            # Set labels and title
            axes[1, 0].set_xlabel('Ticker')
            axes[1, 0].set_ylabel('Yield Curve Sensitivity')
            axes[1, 0].set_title('Yield Curve Sensitivity')
            axes[1, 0].tick_params(axis='x', rotation=90)
            axes[1, 0].grid(True, axis='y')
        
        # Plot regulatory risk
        if 'regulatory_risk' in financial_analysis:
            regulatory_risk = financial_analysis['regulatory_risk']
            
            # Sort by risk
            sorted_risk = pd.Series(regulatory_risk).sort_values(ascending=False)
            
            # Plot as bar chart
            axes[1, 1].bar(sorted_risk.index, sorted_risk.values)
            
            # Set labels and title
            axes[1, 1].set_xlabel('Ticker')
            axes[1, 1].set_ylabel('Regulatory Risk')
            axes[1, 1].set_title('Regulatory Risk')
            axes[1, 1].tick_params(axis='x', rotation=90)
            axes[1, 1].grid(True, axis='y')
        
        # Plot subsector statistics
        if 'subsector_stats' in financial_analysis:
            subsector_stats = financial_analysis['subsector_stats']
            
            # Extract statistics for plotting
            subsectors = list(subsector_stats.keys())
            returns = [subsector_stats[subsector]['ann_return'] for subsector in subsectors]
            volatilities = [subsector_stats[subsector]['ann_volatility'] for subsector in subsectors]
            
            # Plot as scatter
            axes[2, 0].scatter(volatilities, returns)
            
            # Add subsector labels
            for i, subsector in enumerate(subsectors):
                axes[2, 0].annotate(subsector, (volatilities[i], returns[i]))
            
            # Set labels and title
            axes[2, 0].set_xlabel('Volatility')
            axes[2, 0].set_ylabel('Return')
            axes[2, 0].set_title('Risk-Return by Financial Subsector')
            axes[2, 0].grid(True)
        
        # Plot systemic risk
        if 'systemic_risk' in financial_analysis:
            systemic_risk = financial_analysis['systemic_risk']
            
            # Sort by risk
            sorted_risk = pd.Series(systemic_risk).sort_values(ascending=False)
            
            # Plot as bar chart
            axes[2, 1].bar(sorted_risk.index, sorted_risk.values)
            
            # Set labels and title
            axes[2, 1].set_xlabel('Ticker')
            axes[2, 1].set_ylabel('Systemic Risk Contribution')
            axes[2, 1].set_title('Systemic Risk Contribution')
            axes[2, 1].tick_params(axis='x', rotation=90)
            axes[2, 1].grid(True, axis='y')
        
        plt.tight_layout()
        
        return fig


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    # Create a multi-asset portfolio
    n_assets = 15
    n_days = 1000
    
    # Asset names and sectors
    tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    energy_tickers = ['XOM', 'CVX', 'COP', 'EOG', 'SLB']
    financial_tickers = ['JPM', 'BAC', 'GS', 'MS', 'WFC']
    
    all_tickers = tech_tickers + energy_tickers + financial_tickers
    
    # Create sector mapping
    sector_mapping = {}
    for ticker in tech_tickers:
        sector_mapping[ticker] = 'Technology'
    for ticker in energy_tickers:
        sector_mapping[ticker] = 'Energy'
    for ticker in financial_tickers:
        sector_mapping[ticker] = 'Financial'
    
    # Create metadata
    metadata = {}
    for ticker in tech_tickers:
        metadata[ticker] = {
            'name': f"{ticker} Inc.",
            'sector': 'Technology',
            'industry': np.random.choice(['Software', 'Hardware', 'Internet', 'Semiconductor']),
            'market_cap': np.random.randint(100, 1000) * 1e9
        }
    
    for ticker in energy_tickers:
        metadata[ticker] = {
            'name': f"{ticker} Corp.",
            'sector': 'Energy',
            'industry': np.random.choice(['Oil & Gas', 'Drilling', 'Refining', 'Equipment']),
            'market_cap': np.random.randint(50, 500) * 1e9
        }
    
    for ticker in financial_tickers:
        metadata[ticker] = {
            'name': f"{ticker} Group",
            'sector': 'Financial',
            'industry': np.random.choice(['Banking', 'Insurance', 'Asset Management', 'Brokerage']),
            'market_cap': np.random.randint(50, 500) * 1e9
        }
    
    # Generate correlated returns
    # Different mean returns and volatilities for each sector
    tech_mu = np.array([0.0008, 0.0007, 0.0009, 0.0008, 0.0007])  # Higher growth
    tech_sigma = np.array([0.018, 0.016, 0.020, 0.022, 0.019])  # Higher volatility
    
    energy_mu = np.array([0.0005, 0.0006, 0.0004, 0.0005, 0.0006])  # Moderate growth
    energy_sigma = np.array([0.016, 0.017, 0.015, 0.018, 0.014])  # Moderate volatility
    
    financial_mu = np.array([0.0006, 0.0005, 0.0007, 0.0006, 0.0005])  # Moderate growth
    financial_sigma = np.array([0.015, 0.014, 0.016, 0.015, 0.013])  # Moderate volatility
    
    # Combine means and volatilities
    mu = np.concatenate([tech_mu, energy_mu, financial_mu])
    sigma = np.concatenate([tech_sigma, energy_sigma, financial_sigma])
    
    # Create block correlation matrix
    # Higher correlation within sectors, lower between sectors
    corr = np.zeros((n_assets, n_assets))
    
    # Set diagonal to 1
    np.fill_diagonal(corr, 1.0)
    
    # Set within-sector correlations
    for i in range(5):
        for j in range(5):
            if i != j:
                # Tech sector (higher correlation)
                corr[i, j] = 0.7
                
                # Energy sector (higher correlation)
                corr[i+5, j+5] = 0.6
                
                # Financial sector (higher correlation)
                corr[i+10, j+10] = 0.8
    
    # Set between-sector correlations
    for i in range(5):
        for j in range(5):
            # Tech-Energy
            corr[i, j+5] = 0.3
            corr[j+5, i] = 0.3
            
            # Tech-Financial
            corr[i, j+10] = 0.4
            corr[j+10, i] = 0.4
            
            # Energy-Financial
            corr[i+5, j+10] = 0.5
            corr[j+10, i+5] = 0.5
    
    # Create covariance matrix
    cov = np.diag(sigma) @ corr @ np.diag(sigma)
    
    # Generate returns
    returns = np.random.multivariate_normal(mu, cov, size=n_days)
    
    # Create DataFrame
    returns_df = pd.DataFrame(returns, columns=all_tickers)
    
    # Add date index
    start_date = pd.Timestamp('2020-01-01')
    date_range = pd.date_range(start=start_date, periods=n_days, freq='B')
    returns_df.index = date_range
    
    # Create economic data
    economic_data = pd.DataFrame(index=date_range)
    
    # Add interest rates
    economic_data['3M_Treasury'] = np.cumsum(np.random.normal(0.0001, 0.0005, n_days))
    economic_data['10Y_Treasury'] = economic_data['3M_Treasury'] + np.random.normal(0.02, 0.005, n_days)
    
    # Add oil prices
    economic_data['WTI_Crude'] = 50 + np.cumsum(np.random.normal(0.0002, 0.02, n_days))
    
    # Add economic indicators
    economic_data['GDP_Growth'] = np.random.normal(0.02, 0.005, n_days)
    economic_data['Inflation'] = np.random.normal(0.02, 0.003, n_days)
    economic_data['Unemployment'] = 5 + np.cumsum(np.random.normal(0, 0.01, n_days))
    
    # Create sector risk analysis
    sector_analysis = SectorRiskAnalysis(returns_df, metadata, economic_data, sector_mapping)
    
    # Analyze each sector
    tech_results = sector_analysis.analyze_tech_sector()
    energy_results = sector_analysis.analyze_energy_sector()
    financial_results = sector_analysis.analyze_financial_sector()
    
    # Plot results
    fig1 = sector_analysis.plot_sector_comparison()
    fig1.savefig('sector_comparison.png')
    
    fig2 = sector_analysis.plot_tech_analysis(tech_results)
    fig2.savefig('tech_analysis.png')
    
    fig3 = sector_analysis.plot_energy_analysis(energy_results)
    fig3.savefig('energy_analysis.png')
    
    fig4 = sector_analysis.plot_financial_analysis(financial_results)
    fig4.savefig('financial_analysis.png')
    
    plt.close('all')
    
    print("Sector risk analysis complete. Results saved to PNG files.")
