"""
Data Collection for Real-World Case Studies

This module implements data collection and preprocessing functions for real-world
case studies to demonstrate the risk modeling framework's capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import requests
import json
import os
import sys
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataCollector:
    """
    Collects and preprocesses financial data for real-world case studies.
    
    This class provides methods for collecting data from various sources,
    including market data, economic indicators, and alternative data sources.
    """
    
    def __init__(self, data_dir='data', cache=True, cache_expiry_days=7):
        """
        Initialize the data collector.
        
        Parameters:
        -----------
        data_dir : str, default='data'
            Directory to store collected data
        cache : bool, default=True
            Whether to cache data locally
        cache_expiry_days : int, default=7
            Number of days after which cached data expires
        """
        self.data_dir = data_dir
        self.cache = cache
        self.cache_expiry_days = cache_expiry_days
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Create subdirectories for different data types
        os.makedirs(os.path.join(data_dir, 'market'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'economic'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'alternative'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)
    
    def get_market_data(self, tickers, start_date, end_date=None, interval='1d', 
                        include_dividends=True, adjust_prices=True):
        """
        Get market data for specified tickers.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
            If None, use current date
        interval : str, default='1d'
            Data interval ('1d', '1wk', '1mo')
        include_dividends : bool, default=True
            Whether to include dividend data
        adjust_prices : bool, default=True
            Whether to use adjusted prices
            
        Returns:
        --------
        dict
            Dictionary containing price data and metadata
        """
        # Set end date to current date if not specified
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check cache
        cache_file = os.path.join(self.data_dir, 'market', 
                                 f"market_data_{'-'.join(tickers)}_{start_date}_{end_date}_{interval}.csv")
        
        if self.cache and os.path.exists(cache_file):
            # Check if cache is expired
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_time).days < self.cache_expiry_days:
                # Load from cache
                print(f"Loading market data from cache: {cache_file}")
                price_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                
                # Load metadata
                metadata_file = cache_file.replace('.csv', '_metadata.json')
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {}
                
                return {'data': price_data, 'metadata': metadata}
        
        # Download data
        print(f"Downloading market data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Use yfinance to download data
        data = yf.download(tickers, start=start_date, end=end_date, 
                          interval=interval, group_by='ticker', 
                          auto_adjust=adjust_prices, actions=include_dividends)
        
        # Process data
        if len(tickers) == 1:
            # Single ticker
            ticker = tickers[0]
            price_data = pd.DataFrame()
            
            # Extract price columns
            if adjust_prices:
                price_data[f"{ticker}_Open"] = data['Open']
                price_data[f"{ticker}_High"] = data['High']
                price_data[f"{ticker}_Low"] = data['Low']
                price_data[f"{ticker}_Close"] = data['Close']
                price_data[f"{ticker}_Volume"] = data['Volume']
            else:
                price_data[f"{ticker}_Open"] = data['Open']
                price_data[f"{ticker}_High"] = data['High']
                price_data[f"{ticker}_Low"] = data['Low']
                price_data[f"{ticker}_Close"] = data['Close']
                price_data[f"{ticker}_Adj Close"] = data['Adj Close']
                price_data[f"{ticker}_Volume"] = data['Volume']
            
            # Extract dividend data if available
            if include_dividends and 'Dividends' in data.columns:
                price_data[f"{ticker}_Dividends"] = data['Dividends']
        else:
            # Multiple tickers
            price_data = pd.DataFrame()
            
            for ticker in tickers:
                # Extract price columns
                if adjust_prices:
                    price_data[f"{ticker}_Open"] = data[ticker]['Open']
                    price_data[f"{ticker}_High"] = data[ticker]['High']
                    price_data[f"{ticker}_Low"] = data[ticker]['Low']
                    price_data[f"{ticker}_Close"] = data[ticker]['Close']
                    price_data[f"{ticker}_Volume"] = data[ticker]['Volume']
                else:
                    price_data[f"{ticker}_Open"] = data[ticker]['Open']
                    price_data[f"{ticker}_High"] = data[ticker]['High']
                    price_data[f"{ticker}_Low"] = data[ticker]['Low']
                    price_data[f"{ticker}_Close"] = data[ticker]['Close']
                    price_data[f"{ticker}_Adj Close"] = data[ticker]['Adj Close']
                    price_data[f"{ticker}_Volume"] = data[ticker]['Volume']
                
                # Extract dividend data if available
                if include_dividends and 'Dividends' in data[ticker].columns:
                    price_data[f"{ticker}_Dividends"] = data[ticker]['Dividends']
        
        # Get metadata
        metadata = {}
        for ticker in tickers:
            try:
                ticker_info = yf.Ticker(ticker).info
                metadata[ticker] = {
                    'name': ticker_info.get('shortName', ticker),
                    'sector': ticker_info.get('sector', 'Unknown'),
                    'industry': ticker_info.get('industry', 'Unknown'),
                    'market_cap': ticker_info.get('marketCap', None),
                    'currency': ticker_info.get('currency', 'USD')
                }
            except Exception as e:
                print(f"Error getting metadata for {ticker}: {e}")
                metadata[ticker] = {
                    'name': ticker,
                    'sector': 'Unknown',
                    'industry': 'Unknown',
                    'market_cap': None,
                    'currency': 'USD'
                }
        
        # Cache data
        if self.cache:
            price_data.to_csv(cache_file)
            
            # Cache metadata
            metadata_file = cache_file.replace('.csv', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
        
        return {'data': price_data, 'metadata': metadata}
    
    def get_economic_indicators(self, indicators, start_date, end_date=None, source='fred'):
        """
        Get economic indicator data.
        
        Parameters:
        -----------
        indicators : list
            List of indicator codes
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
            If None, use current date
        source : str, default='fred'
            Data source ('fred', 'worldbank', 'imf')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing economic indicator data
        """
        # Set end date to current date if not specified
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check cache
        cache_file = os.path.join(self.data_dir, 'economic', 
                                 f"economic_data_{'-'.join(indicators)}_{start_date}_{end_date}_{source}.csv")
        
        if self.cache and os.path.exists(cache_file):
            # Check if cache is expired
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_time).days < self.cache_expiry_days:
                # Load from cache
                print(f"Loading economic data from cache: {cache_file}")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Download data
        print(f"Downloading economic data for {len(indicators)} indicators from {start_date} to {end_date}")
        
        # Initialize DataFrame
        econ_data = pd.DataFrame()
        
        if source == 'fred':
            try:
                # Try to use pandas_datareader for FRED data
                import pandas_datareader.data as web
                
                for indicator in indicators:
                    try:
                        # Get data from FRED
                        data = web.DataReader(indicator, 'fred', start_date, end_date)
                        econ_data[indicator] = data[indicator]
                    except Exception as e:
                        print(f"Error getting data for {indicator}: {e}")
            except ImportError:
                print("pandas_datareader not installed. Using alternative method.")
                
                # Alternative method using yfinance
                for indicator in indicators:
                    try:
                        # Construct FRED ticker
                        fred_ticker = f"{indicator}=F"
                        data = yf.download(fred_ticker, start=start_date, end=end_date, 
                                          interval='1d', progress=False)
                        if not data.empty:
                            econ_data[indicator] = data['Close']
                    except Exception as e:
                        print(f"Error getting data for {indicator}: {e}")
        
        elif source == 'worldbank':
            print("World Bank data source not implemented yet.")
            # This would use the World Bank API
        
        elif source == 'imf':
            print("IMF data source not implemented yet.")
            # This would use the IMF API
        
        else:
            raise ValueError(f"Unknown source: {source}")
        
        # Cache data
        if self.cache and not econ_data.empty:
            econ_data.to_csv(cache_file)
        
        return econ_data
    
    def get_crypto_data(self, coins, start_date, end_date=None, interval='1d'):
        """
        Get cryptocurrency data.
        
        Parameters:
        -----------
        coins : list
            List of cryptocurrency symbols (e.g., 'BTC', 'ETH')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
            If None, use current date
        interval : str, default='1d'
            Data interval ('1d', '1h', '15m')
            
        Returns:
        --------
        dict
            Dictionary containing price data and metadata
        """
        # Set end date to current date if not specified
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check cache
        cache_file = os.path.join(self.data_dir, 'alternative', 
                                 f"crypto_data_{'-'.join(coins)}_{start_date}_{end_date}_{interval}.csv")
        
        if self.cache and os.path.exists(cache_file):
            # Check if cache is expired
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_time).days < self.cache_expiry_days:
                # Load from cache
                print(f"Loading crypto data from cache: {cache_file}")
                price_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                
                # Load metadata
                metadata_file = cache_file.replace('.csv', '_metadata.json')
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {}
                
                return {'data': price_data, 'metadata': metadata}
        
        # Download data
        print(f"Downloading crypto data for {len(coins)} coins from {start_date} to {end_date}")
        
        # Initialize DataFrame
        price_data = pd.DataFrame()
        metadata = {}
        
        # Convert coin symbols to yfinance format
        yf_tickers = [f"{coin}-USD" for coin in coins]
        
        # Use yfinance to download data
        data = yf.download(yf_tickers, start=start_date, end=end_date, 
                          interval=interval, group_by='ticker', progress=False)
        
        # Process data
        if len(coins) == 1:
            # Single coin
            coin = coins[0]
            ticker = yf_tickers[0]
            
            # Extract price columns
            price_data[f"{coin}_Open"] = data['Open']
            price_data[f"{coin}_High"] = data['High']
            price_data[f"{coin}_Low"] = data['Low']
            price_data[f"{coin}_Close"] = data['Close']
            price_data[f"{coin}_Volume"] = data['Volume']
            
            # Get metadata
            try:
                ticker_info = yf.Ticker(ticker).info
                metadata[coin] = {
                    'name': ticker_info.get('name', coin),
                    'market_cap': ticker_info.get('marketCap', None),
                    'currency': 'USD'
                }
            except Exception as e:
                print(f"Error getting metadata for {coin}: {e}")
                metadata[coin] = {
                    'name': coin,
                    'market_cap': None,
                    'currency': 'USD'
                }
        else:
            # Multiple coins
            for i, coin in enumerate(coins):
                ticker = yf_tickers[i]
                
                # Extract price columns
                price_data[f"{coin}_Open"] = data[ticker]['Open']
                price_data[f"{coin}_High"] = data[ticker]['High']
                price_data[f"{coin}_Low"] = data[ticker]['Low']
                price_data[f"{coin}_Close"] = data[ticker]['Close']
                price_data[f"{coin}_Volume"] = data[ticker]['Volume']
                
                # Get metadata
                try:
                    ticker_info = yf.Ticker(ticker).info
                    metadata[coin] = {
                        'name': ticker_info.get('name', coin),
                        'market_cap': ticker_info.get('marketCap', None),
                        'currency': 'USD'
                    }
                except Exception as e:
                    print(f"Error getting metadata for {coin}: {e}")
                    metadata[coin] = {
                        'name': coin,
                        'market_cap': None,
                        'currency': 'USD'
                    }
        
        # Cache data
        if self.cache and not price_data.empty:
            price_data.to_csv(cache_file)
            
            # Cache metadata
            metadata_file = cache_file.replace('.csv', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
        
        return {'data': price_data, 'metadata': metadata}
    
    def get_private_equity_data(self, funds=None, start_date=None, end_date=None):
        """
        Get private equity data.
        
        Note: This is a placeholder method. In a real implementation, this would
        connect to a private equity database or API.
        
        Parameters:
        -----------
        funds : list, optional
            List of fund identifiers
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        dict
            Dictionary containing fund data and metadata
        """
        print("Private equity data collection not implemented.")
        print("This would typically require access to specialized databases.")
        
        # Create synthetic data for demonstration purposes
        if start_date is None:
            start_date = '2018-01-01'
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if funds is None:
            funds = ['PE_Fund_1', 'PE_Fund_2', 'PE_Fund_3']
        
        # Generate dates (quarterly data is typical for PE)
        date_range = pd.date_range(start=start_date, end=end_date, freq='Q')
        
        # Generate synthetic data
        np.random.seed(42)
        data = pd.DataFrame(index=date_range)
        
        for fund in funds:
            # Generate quarterly returns (typically 2-4% per quarter for PE)
            returns = np.random.normal(0.03, 0.02, len(date_range))
            data[f"{fund}_Return"] = returns
            
            # Generate NAV (starting at 100, growing with returns)
            nav = 100 * np.cumprod(1 + returns)
            data[f"{fund}_NAV"] = nav
            
            # Generate capital calls and distributions
            data[f"{fund}_Capital_Call"] = np.random.exponential(5, len(date_range)) * np.exp(-0.1 * np.arange(len(date_range)))
            data[f"{fund}_Distribution"] = np.random.exponential(2, len(date_range)) * np.exp(0.1 * np.arange(len(date_range)))
        
        # Generate metadata
        metadata = {}
        for fund in funds:
            metadata[fund] = {
                'name': fund,
                'vintage': str(int(np.random.randint(2010, 2020))),
                'strategy': np.random.choice(['Buyout', 'Venture', 'Growth', 'Distressed']),
                'geography': np.random.choice(['North America', 'Europe', 'Asia', 'Global']),
                'fund_size': np.random.randint(100, 1000) * 1e6
            }
        
        return {'data': data, 'metadata': metadata}
    
    def create_returns_dataset(self, price_data, tickers=None, price_col='Close', 
                              return_type='log', period='daily', dropna=True):
        """
        Create a returns dataset from price data.
        
        Parameters:
        -----------
        price_data : pandas.DataFrame
            DataFrame containing price data
        tickers : list, optional
            List of tickers to include
            If None, use all tickers in price_data
        price_col : str, default='Close'
            Price column to use for calculating returns
        return_type : str, default='log'
            Type of returns to calculate ('log', 'simple')
        period : str, default='daily'
            Return period ('daily', 'weekly', 'monthly')
        dropna : bool, default=True
            Whether to drop rows with NaN values
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing returns
        """
        # Identify tickers in price_data
        if tickers is None:
            # Extract tickers from column names
            all_cols = price_data.columns
            tickers = set()
            for col in all_cols:
                if '_' in col:
                    ticker = col.split('_')[0]
                    tickers.add(ticker)
            tickers = sorted(list(tickers))
        
        # Create returns DataFrame
        returns_data = pd.DataFrame(index=price_data.index)
        
        for ticker in tickers:
            # Get price column
            price_col_name = f"{ticker}_{price_col}"
            
            if price_col_name in price_data.columns:
                prices = price_data[price_col_name]
                
                # Resample if needed
                if period == 'weekly':
                    prices = prices.resample('W').last()
                elif period == 'monthly':
                    prices = prices.resample('M').last()
                
                # Calculate returns
                if return_type == 'log':
                    returns = np.log(prices / prices.shift(1))
                else:  # simple returns
                    returns = prices.pct_change()
                
                # Add to returns DataFrame
                returns_data[ticker] = returns
        
        # Drop NaN values if requested
        if dropna:
            returns_data = returns_data.dropna()
        
        return returns_data
    
    def create_market_regime_dataset(self, returns_data, window=60, n_regimes=3, method='volatility'):
        """
        Create a dataset with market regime labels.
        
        Parameters:
        -----------
        returns_data : pandas.DataFrame
            DataFrame containing returns
        window : int, default=60
            Window size for calculating regime indicators
        n_regimes : int, default=3
            Number of regimes to identify
        method : str, default='volatility'
            Method for identifying regimes ('volatility', 'momentum', 'correlation')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing returns and regime labels
        """
        # Create copy of returns data
        regime_data = returns_data.copy()
        
        if method == 'volatility':
            # Calculate rolling volatility
            portfolio_returns = returns_data.mean(axis=1)
            rolling_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(252)
            
            # Classify regimes based on volatility
            regime_data['volatility'] = rolling_vol
            
            # Use quantiles to classify regimes
            quantiles = rolling_vol.quantile([i/n_regimes for i in range(1, n_regimes)]).values
            
            # Initialize regime column
            regime_data['regime'] = 0
            
            # Assign regimes
            for i in range(n_regimes - 1):
                regime_data.loc[rolling_vol > quantiles[i], 'regime'] = i + 1
        
        elif method == 'momentum':
            # Calculate rolling returns
            portfolio_returns = returns_data.mean(axis=1)
            rolling_returns = portfolio_returns.rolling(window=window).mean() * 252
            
            # Classify regimes based on momentum
            regime_data['momentum'] = rolling_returns
            
            # Use quantiles to classify regimes
            quantiles = rolling_returns.quantile([i/n_regimes for i in range(1, n_regimes)]).values
            
            # Initialize regime column
            regime_data['regime'] = 0
            
            # Assign regimes
            for i in range(n_regimes - 1):
                regime_data.loc[rolling_returns > quantiles[i], 'regime'] = i + 1
        
        elif method == 'correlation':
            # Calculate rolling correlation
            corr_values = []
            
            for i in range(window, len(returns_data)):
                window_data = returns_data.iloc[i-window:i]
                corr_matrix = window_data.corr()
                avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                corr_values.append(avg_corr)
            
            # Create Series with correlation values
            rolling_corr = pd.Series(corr_values, index=returns_data.index[window:])
            
            # Classify regimes based on correlation
            regime_data = regime_data.iloc[window:]
            regime_data['correlation'] = rolling_corr
            
            # Use quantiles to classify regimes
            quantiles = rolling_corr.quantile([i/n_regimes for i in range(1, n_regimes)]).values
            
            # Initialize regime column
            regime_data['regime'] = 0
            
            # Assign regimes
            for i in range(n_regimes - 1):
                regime_data.loc[rolling_corr > quantiles[i], 'regime'] = i + 1
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return regime_data
    
    def create_stress_test_scenarios(self, returns_data, scenarios=None, method='historical'):
        """
        Create stress test scenarios.
        
        Parameters:
        -----------
        returns_data : pandas.DataFrame
            DataFrame containing returns
        scenarios : dict, optional
            Dictionary of predefined scenarios
            If None, use default scenarios
        method : str, default='historical'
            Method for creating scenarios ('historical', 'monte_carlo', 'parametric')
            
        Returns:
        --------
        dict
            Dictionary containing stress test scenarios
        """
        # Define default scenarios if not provided
        if scenarios is None:
            scenarios = {
                'financial_crisis_2008': {
                    'start_date': '2008-09-01',
                    'end_date': '2009-03-31',
                    'description': '2008 Financial Crisis'
                },
                'covid_crash_2020': {
                    'start_date': '2020-02-15',
                    'end_date': '2020-03-31',
                    'description': 'COVID-19 Market Crash'
                },
                'tech_bubble_2000': {
                    'start_date': '2000-03-01',
                    'end_date': '2000-05-31',
                    'description': 'Dot-com Bubble Burst'
                },
                'taper_tantrum_2013': {
                    'start_date': '2013-05-01',
                    'end_date': '2013-06-30',
                    'description': 'Taper Tantrum'
                },
                'china_slowdown_2015': {
                    'start_date': '2015-08-01',
                    'end_date': '2015-09-30',
                    'description': 'China Slowdown'
                },
                'trump_tariffs_2023': {
                    'start_date': '2023-03-01',
                    'end_date': '2023-04-30',
                    'description': 'Trump Tariff Announcements'
                }
            }
        
        # Initialize results
        stress_scenarios = {}
        
        if method == 'historical':
            # Use historical scenarios
            for scenario_name, scenario_info in scenarios.items():
                start_date = scenario_info['start_date']
                end_date = scenario_info['end_date']
                
                # Check if dates are in returns_data
                if pd.Timestamp(start_date) in returns_data.index and pd.Timestamp(end_date) in returns_data.index:
                    # Extract returns for scenario period
                    scenario_returns = returns_data.loc[start_date:end_date]
                    
                    # Calculate cumulative returns
                    cumulative_returns = (1 + scenario_returns).prod() - 1
                    
                    # Calculate volatility
                    volatility = scenario_returns.std() * np.sqrt(252)
                    
                    # Calculate maximum drawdown
                    cum_returns = (1 + scenario_returns).cumprod()
                    running_max = cum_returns.cummax()
                    drawdown = (cum_returns - running_max) / running_max
                    max_drawdown = drawdown.min()
                    
                    # Store scenario data
                    stress_scenarios[scenario_name] = {
                        'description': scenario_info['description'],
                        'start_date': start_date,
                        'end_date': end_date,
                        'returns': scenario_returns,
                        'cumulative_returns': cumulative_returns,
                        'volatility': volatility,
                        'max_drawdown': max_drawdown
                    }
                else:
                    print(f"Warning: Scenario {scenario_name} dates not in returns_data")
        
        elif method == 'monte_carlo':
            # Generate Monte Carlo scenarios
            for scenario_name, scenario_info in scenarios.items():
                # Get parameters from scenario_info or use defaults
                n_days = scenario_info.get('n_days', 60)
                n_scenarios = scenario_info.get('n_scenarios', 1000)
                shock_factor = scenario_info.get('shock_factor', 2.0)
                
                # Calculate mean and covariance
                mean_returns = returns_data.mean().values
                cov_matrix = returns_data.cov().values
                
                # Apply shock factor to covariance
                shocked_cov = cov_matrix * shock_factor
                
                # Generate scenarios
                np.random.seed(42)  # For reproducibility
                scenario_returns_list = []
                
                for _ in range(n_scenarios):
                    # Generate random returns
                    random_returns = np.random.multivariate_normal(mean_returns, shocked_cov, n_days)
                    
                    # Convert to DataFrame
                    scenario_df = pd.DataFrame(random_returns, columns=returns_data.columns)
                    
                    # Calculate cumulative returns
                    cumulative_returns = (1 + scenario_df).prod() - 1
                    
                    # Store scenario
                    scenario_returns_list.append(cumulative_returns)
                
                # Find worst scenario
                scenario_returns_array = np.array(scenario_returns_list)
                portfolio_returns = scenario_returns_array.mean(axis=1)
                worst_idx = portfolio_returns.argmin()
                
                # Store worst scenario
                worst_scenario = scenario_returns_list[worst_idx]
                
                # Store scenario data
                stress_scenarios[scenario_name] = {
                    'description': scenario_info.get('description', scenario_name),
                    'n_days': n_days,
                    'n_scenarios': n_scenarios,
                    'shock_factor': shock_factor,
                    'worst_scenario': pd.Series(worst_scenario, index=returns_data.columns),
                    'all_scenarios': scenario_returns_list
                }
        
        elif method == 'parametric':
            # Generate parametric scenarios
            for scenario_name, scenario_info in scenarios.items():
                # Get parameters from scenario_info
                shocks = scenario_info.get('shocks', {})
                correlations = scenario_info.get('correlations', {})
                
                # Apply shocks to mean returns
                mean_returns = returns_data.mean().values
                shocked_returns = mean_returns.copy()
                
                for asset, shock in shocks.items():
                    if asset in returns_data.columns:
                        idx = returns_data.columns.get_loc(asset)
                        shocked_returns[idx] = shock
                
                # Apply correlation changes
                cov_matrix = returns_data.cov().values
                shocked_cov = cov_matrix.copy()
                
                for asset_pair, corr in correlations.items():
                    asset1, asset2 = asset_pair
                    if asset1 in returns_data.columns and asset2 in returns_data.columns:
                        idx1 = returns_data.columns.get_loc(asset1)
                        idx2 = returns_data.columns.get_loc(asset2)
                        
                        # Calculate new covariance
                        vol1 = np.sqrt(shocked_cov[idx1, idx1])
                        vol2 = np.sqrt(shocked_cov[idx2, idx2])
                        new_cov = corr * vol1 * vol2
                        
                        # Update covariance matrix
                        shocked_cov[idx1, idx2] = new_cov
                        shocked_cov[idx2, idx1] = new_cov
                
                # Store scenario data
                stress_scenarios[scenario_name] = {
                    'description': scenario_info.get('description', scenario_name),
                    'shocked_returns': pd.Series(shocked_returns, index=returns_data.columns),
                    'shocked_cov': pd.DataFrame(shocked_cov, index=returns_data.columns, columns=returns_data.columns),
                    'shocks': shocks,
                    'correlations': correlations
                }
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return stress_scenarios
    
    def create_sector_analysis_dataset(self, returns_data, metadata):
        """
        Create a dataset for sector-specific analysis.
        
        Parameters:
        -----------
        returns_data : pandas.DataFrame
            DataFrame containing returns
        metadata : dict
            Dictionary containing asset metadata
            
        Returns:
        --------
        dict
            Dictionary containing sector analysis data
        """
        # Extract sector information from metadata
        sectors = {}
        for ticker, info in metadata.items():
            if ticker in returns_data.columns:
                sector = info.get('sector', 'Unknown')
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(ticker)
        
        # Create sector returns
        sector_returns = pd.DataFrame(index=returns_data.index)
        
        for sector, tickers in sectors.items():
            if tickers:
                # Calculate equal-weighted sector returns
                sector_returns[sector] = returns_data[tickers].mean(axis=1)
        
        # Calculate sector statistics
        sector_stats = {}
        
        for sector in sectors:
            if sector in sector_returns.columns:
                # Calculate annualized return
                ann_return = sector_returns[sector].mean() * 252
                
                # Calculate annualized volatility
                ann_vol = sector_returns[sector].std() * np.sqrt(252)
                
                # Calculate Sharpe ratio
                sharpe = ann_return / ann_vol
                
                # Calculate maximum drawdown
                cum_returns = (1 + sector_returns[sector]).cumprod()
                running_max = cum_returns.cummax()
                drawdown = (cum_returns - running_max) / running_max
                max_drawdown = drawdown.min()
                
                # Store statistics
                sector_stats[sector] = {
                    'annualized_return': ann_return,
                    'annualized_volatility': ann_vol,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'n_assets': len(sectors[sector])
                }
        
        # Calculate sector correlations
        sector_corr = sector_returns.corr()
        
        # Return sector analysis data
        return {
            'sector_returns': sector_returns,
            'sector_stats': sector_stats,
            'sector_correlations': sector_corr,
            'sector_assets': sectors
        }
    
    def plot_returns(self, returns_data, figsize=(12, 6), title='Asset Returns'):
        """
        Plot asset returns.
        
        Parameters:
        -----------
        returns_data : pandas.DataFrame
            DataFrame containing returns
        figsize : tuple, default=(12, 6)
            Figure size
        title : str, default='Asset Returns'
            Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot cumulative returns
        cum_returns = (1 + returns_data).cumprod()
        cum_returns.plot(ax=ax)
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True)
        
        return fig
    
    def plot_regime_analysis(self, regime_data, figsize=(12, 8), title='Market Regime Analysis'):
        """
        Plot market regime analysis.
        
        Parameters:
        -----------
        regime_data : pandas.DataFrame
            DataFrame containing returns and regime labels
        figsize : tuple, default=(12, 8)
            Figure size
        title : str, default='Market Regime Analysis'
            Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot cumulative returns
        returns_cols = [col for col in regime_data.columns if col not in ['regime', 'volatility', 'momentum', 'correlation']]
        cum_returns = (1 + regime_data[returns_cols]).cumprod()
        cum_returns.plot(ax=axes[0])
        
        # Set labels and title for first subplot
        axes[0].set_ylabel('Cumulative Return')
        axes[0].set_title(title)
        axes[0].legend(loc='upper left')
        axes[0].grid(True)
        
        # Plot regime indicator
        if 'regime' in regime_data.columns:
            # Get regime indicator
            regimes = regime_data['regime']
            
            # Plot regime as background color
            unique_regimes = sorted(regimes.unique())
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_regimes)))
            
            for i, regime in enumerate(unique_regimes):
                regime_periods = regime_data.index[regimes == regime]
                if len(regime_periods) > 0:
                    for start_idx in range(len(regime_periods)):
                        if start_idx == 0 or regime_periods[start_idx] != regime_periods[start_idx-1] + pd.Timedelta(days=1):
                            # Start of a new regime period
                            start_date = regime_periods[start_idx]
                            
                            # Find end of period
                            end_idx = start_idx
                            while end_idx + 1 < len(regime_periods) and regime_periods[end_idx+1] == regime_periods[end_idx] + pd.Timedelta(days=1):
                                end_idx += 1
                            
                            end_date = regime_periods[end_idx]
                            
                            # Add colored background
                            axes[0].axvspan(start_date, end_date, alpha=0.2, color=colors[i])
            
            # Plot regime indicator
            axes[1].plot(regime_data.index, regimes, 'o-', label='Regime')
            
            # Plot regime indicator (volatility, momentum, or correlation)
            if 'volatility' in regime_data.columns:
                axes[1].plot(regime_data.index, regime_data['volatility'], 'r--', label='Volatility')
            elif 'momentum' in regime_data.columns:
                axes[1].plot(regime_data.index, regime_data['momentum'], 'g--', label='Momentum')
            elif 'correlation' in regime_data.columns:
                axes[1].plot(regime_data.index, regime_data['correlation'], 'b--', label='Correlation')
            
            # Set labels and title for second subplot
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Regime')
            axes[1].set_title('Market Regimes')
            axes[1].legend(loc='upper left')
            axes[1].grid(True)
        
        plt.tight_layout()
        
        return fig
    
    def plot_stress_test_results(self, stress_scenarios, figsize=(12, 8), title='Stress Test Results'):
        """
        Plot stress test results.
        
        Parameters:
        -----------
        stress_scenarios : dict
            Dictionary containing stress test scenarios
        figsize : tuple, default=(12, 8)
            Figure size
        title : str, default='Stress Test Results'
            Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot cumulative returns for historical scenarios
        historical_scenarios = {}
        
        for scenario_name, scenario_data in stress_scenarios.items():
            if 'returns' in scenario_data:
                # Historical scenario
                returns = scenario_data['returns']
                cum_returns = (1 + returns).cumprod()
                
                # Normalize to start at 1
                cum_returns = cum_returns / cum_returns.iloc[0]
                
                # Store for plotting
                historical_scenarios[scenario_name] = cum_returns
        
        # Plot historical scenarios
        if historical_scenarios:
            for scenario_name, cum_returns in historical_scenarios.items():
                # Calculate portfolio return (equal-weighted)
                portfolio_return = cum_returns.mean(axis=1)
                
                # Plot portfolio return
                axes[0].plot(portfolio_return.index, portfolio_return, label=scenario_name)
            
            # Set labels and title for first subplot
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Cumulative Return')
            axes[0].set_title('Historical Scenarios')
            axes[0].legend(loc='upper left')
            axes[0].grid(True)
        
        # Plot scenario impact
        scenario_names = []
        scenario_impacts = []
        
        for scenario_name, scenario_data in stress_scenarios.items():
            if 'cumulative_returns' in scenario_data:
                # Historical scenario
                cum_returns = scenario_data['cumulative_returns']
                
                # Calculate portfolio impact (equal-weighted)
                portfolio_impact = cum_returns.mean()
                
                # Store for plotting
                scenario_names.append(scenario_name)
                scenario_impacts.append(portfolio_impact)
            elif 'worst_scenario' in scenario_data:
                # Monte Carlo scenario
                worst_scenario = scenario_data['worst_scenario']
                
                # Calculate portfolio impact (equal-weighted)
                portfolio_impact = worst_scenario.mean()
                
                # Store for plotting
                scenario_names.append(scenario_name)
                scenario_impacts.append(portfolio_impact)
            elif 'shocked_returns' in scenario_data:
                # Parametric scenario
                shocked_returns = scenario_data['shocked_returns']
                
                # Calculate portfolio impact (equal-weighted)
                portfolio_impact = shocked_returns.mean()
                
                # Store for plotting
                scenario_names.append(scenario_name)
                scenario_impacts.append(portfolio_impact)
        
        # Plot scenario impacts
        if scenario_names:
            # Sort by impact
            sorted_indices = np.argsort(scenario_impacts)
            sorted_names = [scenario_names[i] for i in sorted_indices]
            sorted_impacts = [scenario_impacts[i] for i in sorted_indices]
            
            # Plot as horizontal bar chart
            axes[1].barh(sorted_names, sorted_impacts)
            
            # Add data labels
            for i, impact in enumerate(sorted_impacts):
                axes[1].text(impact, i, f'{impact:.2%}', va='center')
            
            # Set labels and title for second subplot
            axes[1].set_xlabel('Portfolio Impact')
            axes[1].set_ylabel('Scenario')
            axes[1].set_title('Scenario Impact on Portfolio')
            axes[1].grid(True)
        
        plt.tight_layout()
        
        return fig
    
    def plot_sector_analysis(self, sector_data, figsize=(12, 10), title='Sector Analysis'):
        """
        Plot sector analysis results.
        
        Parameters:
        -----------
        sector_data : dict
            Dictionary containing sector analysis data
        figsize : tuple, default=(12, 10)
            Figure size
        title : str, default='Sector Analysis'
            Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot sector returns
        sector_returns = sector_data['sector_returns']
        cum_returns = (1 + sector_returns).cumprod()
        cum_returns.plot(ax=axes[0, 0])
        
        # Set labels and title for first subplot
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].set_title('Sector Returns')
        axes[0, 0].legend(loc='upper left')
        axes[0, 0].grid(True)
        
        # Plot sector statistics
        sector_stats = sector_data['sector_stats']
        sectors = list(sector_stats.keys())
        
        # Extract statistics
        returns = [sector_stats[sector]['annualized_return'] for sector in sectors]
        volatilities = [sector_stats[sector]['annualized_volatility'] for sector in sectors]
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
        
        # Plot sector correlations
        sector_corr = sector_data['sector_correlations']
        sns.heatmap(sector_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        
        # Set title for fourth subplot
        axes[1, 1].set_title('Sector Correlations')
        
        plt.tight_layout()
        
        return fig


# Example usage
if __name__ == "__main__":
    # Create data collector
    collector = DataCollector(data_dir='data')
    
    # Define tickers for different case studies
    tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    finance_tickers = ['JPM', 'BAC', 'GS', 'MS', 'WFC']
    energy_tickers = ['XOM', 'CVX', 'COP', 'EOG', 'SLB']
    crypto_coins = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']
    
    # Define time periods
    start_date = '2018-01-01'
    end_date = '2023-12-31'
    
    # Get market data for tech sector
    tech_data = collector.get_market_data(tech_tickers, start_date, end_date)
    
    # Create returns dataset
    tech_returns = collector.create_returns_dataset(tech_data['data'], tech_tickers)
    
    # Create market regime dataset
    tech_regimes = collector.create_market_regime_dataset(tech_returns, window=60, n_regimes=3, method='volatility')
    
    # Create stress test scenarios
    tech_stress = collector.create_stress_test_scenarios(tech_returns)
    
    # Create sector analysis dataset
    sector_analysis = collector.create_sector_analysis_dataset(tech_returns, tech_data['metadata'])
    
    # Plot results
    fig1 = collector.plot_returns(tech_returns, title='Tech Sector Returns')
    fig1.savefig('tech_returns.png')
    
    fig2 = collector.plot_regime_analysis(tech_regimes, title='Tech Sector Regime Analysis')
    fig2.savefig('tech_regimes.png')
    
    fig3 = collector.plot_stress_test_results(tech_stress, title='Tech Sector Stress Tests')
    fig3.savefig('tech_stress.png')
    
    fig4 = collector.plot_sector_analysis(sector_analysis, title='Tech Sector Analysis')
    fig4.savefig('tech_sector_analysis.png')
    
    plt.close('all')
    
    print("Data collection and analysis complete. Results saved to PNG files.")
