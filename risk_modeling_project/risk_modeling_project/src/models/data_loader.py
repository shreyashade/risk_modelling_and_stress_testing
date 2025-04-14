"""
Data Loader Module for Risk Modeling and Stress Testing Framework

This module provides functionality for loading financial data from various sources,
including market data providers and CSV files.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Tuple
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Base class for loading financial data"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize a DataLoader object
        
        Parameters:
        -----------
        cache_dir : str, optional
            Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def load_data(self, tickers: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Load data for the specified tickers and date range
        
        Parameters:
        -----------
        tickers : list of str
            List of ticker symbols
        start_date : datetime
            Start date for the data
        end_date : datetime
            End date for the data
            
        Returns:
        --------
        dict
            Dictionary mapping tickers to DataFrames with price data
        """
        raise NotImplementedError("Subclasses must implement this method")


class YahooFinanceLoader(DataLoader):
    """Class for loading data from Yahoo Finance"""
    
    def __init__(self, cache_dir: Optional[str] = None, use_cache: bool = True):
        """
        Initialize a YahooFinanceLoader object
        
        Parameters:
        -----------
        cache_dir : str, optional
            Directory to cache downloaded data
        use_cache : bool, default=True
            Whether to use cached data if available
        """
        super().__init__(cache_dir)
        self.use_cache = use_cache
    
    def load_data(self, tickers: List[str], start_date: datetime, end_date: datetime,
                 interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Load data from Yahoo Finance
        
        Parameters:
        -----------
        tickers : list of str
            List of ticker symbols
        start_date : datetime
            Start date for the data
        end_date : datetime
            End date for the data
        interval : str, default="1d"
            Data interval (1d, 1wk, 1mo, etc.)
            
        Returns:
        --------
        dict
            Dictionary mapping tickers to DataFrames with price data
        """
        result = {}
        
        for ticker in tickers:
            cache_file = None
            if self.cache_dir and self.use_cache:
                cache_file = os.path.join(
                    self.cache_dir, 
                    f"{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{interval}.csv"
                )
                
                if os.path.exists(cache_file):
                    logger.info(f"Loading cached data for {ticker}")
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    result[ticker] = df
                    continue
            
            try:
                logger.info(f"Downloading data for {ticker}")
                data = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date, 
                    interval=interval,
                    progress=False
                )
                
                if data.empty:
                    logger.warning(f"No data found for {ticker}")
                    continue
                
                # Keep only the Adj Close column and rename it
                df = data[['Adj Close']].copy()
                df.columns = [ticker]
                
                # Save to cache if enabled
                if cache_file:
                    df.to_csv(cache_file)
                
                result[ticker] = df
                
            except Exception as e:
                logger.error(f"Error downloading data for {ticker}: {str(e)}")
        
        return result


class CSVDataLoader(DataLoader):
    """Class for loading data from CSV files"""
    
    def load_data(self, file_paths: Dict[str, str], date_column: str = "Date",
                 price_column: str = "Adj Close") -> Dict[str, pd.DataFrame]:
        """
        Load data from CSV files
        
        Parameters:
        -----------
        file_paths : dict
            Dictionary mapping tickers to file paths
        date_column : str, default="Date"
            Name of the date column
        price_column : str, default="Adj Close"
            Name of the price column
            
        Returns:
        --------
        dict
            Dictionary mapping tickers to DataFrames with price data
        """
        result = {}
        
        for ticker, file_path in file_paths.items():
            try:
                df = pd.read_csv(file_path)
                
                # Convert date column to datetime
                df[date_column] = pd.to_datetime(df[date_column])
                
                # Set date as index
                df = df.set_index(date_column)
                
                # Keep only the price column and rename it
                df = df[[price_column]].copy()
                df.columns = [ticker]
                
                result[ticker] = df
                
            except Exception as e:
                logger.error(f"Error loading data from {file_path}: {str(e)}")
        
        return result


class DataProcessor:
    """Class for processing financial data"""
    
    @staticmethod
    def align_data(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align multiple DataFrames on their index
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping tickers to DataFrames
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with aligned data for all tickers
        """
        if not data_dict:
            return pd.DataFrame()
        
        # Concatenate all DataFrames
        result = pd.concat(data_dict.values(), axis=1)
        
        # Forward fill missing values (for non-trading days)
        result = result.fillna(method='ffill')
        
        # Backward fill any remaining missing values at the beginning
        result = result.fillna(method='bfill')
        
        return result
    
    @staticmethod
    def calculate_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Parameters:
        -----------
        prices : pd.DataFrame
            DataFrame with price data
        method : str, default='log'
            Method to calculate returns ('log' or 'simple')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with calculated returns
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        elif method == 'simple':
            returns = prices / prices.shift(1) - 1
        else:
            raise ValueError("Method must be either 'log' or 'simple'")
        
        # Drop first row with NaN values
        returns = returns.dropna()
        
        return returns
    
    @staticmethod
    def calculate_volatility(returns: pd.DataFrame, window: int = 21, annualize: bool = True,
                            trading_days: int = 252) -> pd.DataFrame:
        """
        Calculate rolling volatility
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame with return data
        window : int, default=21
            Window for rolling calculation (21 days = approximately 1 month)
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
    def calculate_correlation(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
        """
        Calculate rolling correlation matrix
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame with return data
        window : int, default=63
            Window for rolling calculation (63 days = approximately 3 months)
            
        Returns:
        --------
        dict
            Dictionary with dates as keys and correlation matrices as values
        """
        # Initialize result dictionary
        result = {}
        
        # Calculate rolling correlation
        for i in range(window, len(returns) + 1):
            date = returns.index[i - 1]
            window_returns = returns.iloc[i - window:i]
            corr_matrix = window_returns.corr()
            result[date] = corr_matrix
        
        return result
    
    @staticmethod
    def detect_outliers(returns: pd.DataFrame, n_std: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in return data
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame with return data
        n_std : float, default=3.0
            Number of standard deviations to use as threshold
            
        Returns:
        --------
        pd.DataFrame
            Boolean DataFrame with True for outliers
        """
        # Calculate mean and standard deviation
        mean = returns.mean()
        std = returns.std()
        
        # Identify outliers
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        
        outliers = (returns < lower_bound) | (returns > upper_bound)
        
        return outliers
    
    @staticmethod
    def winsorize_returns(returns: pd.DataFrame, limits: Tuple[float, float] = (0.01, 0.01)) -> pd.DataFrame:
        """
        Winsorize return data to handle outliers
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame with return data
        limits : tuple, default=(0.01, 0.01)
            Tuple of (lower, upper) percentages to winsorize
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with winsorized returns
        """
        result = returns.copy()
        
        for col in result.columns:
            series = result[col]
            lower_bound = series.quantile(limits[0])
            upper_bound = series.quantile(1 - limits[1])
            
            result[col] = np.where(series < lower_bound, lower_bound, series)
            result[col] = np.where(result[col] > upper_bound, upper_bound, result[col])
        
        return result
    
    @staticmethod
    def calculate_drawdowns(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Calculate drawdowns from price data
        
        Parameters:
        -----------
        prices : pd.DataFrame
            DataFrame with price data
            
        Returns:
        --------
        tuple
            (drawdowns, drawdown_periods, max_drawdown)
        """
        # Calculate running maximum
        running_max = prices.cummax()
        
        # Calculate drawdowns
        drawdowns = (prices / running_max - 1) * 100
        
        # Identify drawdown periods
        is_drawdown = drawdowns < 0
        
        # Calculate maximum drawdown
        max_drawdown = drawdowns.min()
        
        return drawdowns, is_drawdown, max_drawdown


class MarketDataManager:
    """Class for managing market data for risk modeling"""
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize a MarketDataManager object
        
        Parameters:
        -----------
        data_loader : DataLoader
            Data loader to use for fetching data
        """
        self.data_loader = data_loader
        self.price_data = {}
        self.return_data = {}
        self.volatility_data = {}
        self.correlation_data = {}
        self.processor = DataProcessor()
    
    def load_market_data(self, tickers: List[str], start_date: datetime, end_date: datetime,
                        return_method: str = 'log') -> None:
        """
        Load market data for the specified tickers and date range
        
        Parameters:
        -----------
        tickers : list of str
            List of ticker symbols
        start_date : datetime
            Start date for the data
        end_date : datetime
            End date for the data
        return_method : str, default='log'
            Method to calculate returns ('log' or 'simple')
        """
        # Load price data
        self.price_data = self.data_loader.load_data(tickers, start_date, end_date)
        
        # Align price data
        aligned_prices = self.processor.align_data(self.price_data)
        
        # Calculate returns
        self.return_data = self.processor.calculate_returns(aligned_prices, method=return_method)
        
        # Calculate volatility
        self.volatility_data = self.processor.calculate_volatility(self.return_data)
        
        # Calculate correlation
        self.correlation_data = self.processor.calculate_correlation(self.return_data)
    
    def get_latest_correlation_matrix(self) -> pd.DataFrame:
        """
        Get the latest correlation matrix
        
        Returns:
        --------
        pd.DataFrame
            Latest correlation matrix
        """
        if not self.correlation_data:
            raise ValueError("No correlation data available. Call load_market_data first.")
        
        # Get the latest date
        latest_date = max(self.correlation_data.keys())
        
        return self.correlation_data[latest_date]
    
    def get_price_data(self, aligned: bool = True) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Get price data
        
        Parameters:
        -----------
        aligned : bool, default=True
            Whether to return aligned data
            
        Returns:
        --------
        Union[Dict[str, pd.DataFrame], pd.DataFrame]
            Price data
        """
        if not self.price_data:
            raise ValueError("No price data available. Call load_market_data first.")
        
        if aligned:
            return self.processor.align_data(self.price_data)
        else:
            return self.price_data
    
    def get_return_data(self) -> pd.DataFrame:
        """
        Get return data
        
        Returns:
        --------
        pd.DataFrame
            Return data
        """
        if self.return_data is None or self.return_data.empty:
            raise ValueError("No return data available. Call load_market_data first.")
        
        return self.return_data
    
    def get_volatility_data(self) -> pd.DataFrame:
        """
        Get volatility data
        
        Returns:
        --------
        pd.DataFrame
            Volatility data
        """
        if self.volatility_data is None or self.volatility_data.empty:
            raise ValueError("No volatility data available. Call load_market_data first.")
        
        return self.volatility_data
    
    def calculate_covariance_matrix(self, window: int = 63) -> pd.DataFrame:
        """
        Calculate covariance matrix
        
        Parameters:
        -----------
        window : int, default=63
            Window for calculation (63 days = approximately 3 months)
            
        Returns:
        --------
        pd.DataFrame
            Covariance matrix
        """
        if self.return_data is None or self.return_data.empty:
            raise ValueError("No return data available. Call load_market_data first.")
        
        # Use the last 'window' days of return data
        recent_returns = self.return_data.iloc[-window:]
        
        # Calculate covariance matrix (annualized)
        cov_matrix = recent_returns.cov() * 252
        
        return cov_matrix
    
    def save_data(self, directory: str) -> None:
        """
        Save all data to CSV files
        
        Parameters:
        -----------
        directory : str
            Directory to save the data
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save aligned price data
        aligned_prices = self.processor.align_data(self.price_data)
        aligned_prices.to_csv(os.path.join(directory, "price_data.csv"))
        
        # Save return data
        if self.return_data is not None and not self.return_data.empty:
            self.return_data.to_csv(os.path.join(directory, "return_data.csv"))
        
        # Save volatility data
        if self.volatility_data is not None and not self.volatility_data.empty:
            self.volatility_data.to_csv(os.path.join(directory, "volatility_data.csv"))
        
        # Save latest correlation matrix
        if self.correlation_data:
            latest_date = max(self.correlation_data.keys())
            self.correlation_data[latest_date].to_csv(os.path.join(directory, "correlation_matrix.csv"))
    
    @classmethod
    def load_from_directory(cls, directory: str) -> 'MarketDataManager':
        """
        Load data from CSV files
        
        Parameters:
        -----------
        directory : str
            Directory to load the data from
            
        Returns:
        --------
        MarketDataManager
            Loaded market data manager
        """
        # Create a dummy data loader
        data_loader = DataLoader()
        manager = cls(data_loader)
        
        # Load price data
        price_path = os.path.join(directory, "price_data.csv")
        if os.path.exists(price_path):
            prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
            
            # Convert to dictionary of DataFrames
            manager.price_data = {col: prices[[col]] for col in prices.columns}
        
        # Load return data
        return_path = os.path.join(directory, "return_data.csv")
        if os.path.exists(return_path):
            manager.return_data = pd.read_csv(return_path, index_col=0, parse_dates=True)
        
        # Load volatility data
        vol_path = os.path.join(directory, "volatility_data.csv")
        if os.path.exists(vol_path):
            manager.volatility_data = pd.read_csv(vol_path, index_col=0, parse_dates=True)
        
        # Load correlation matrix
        corr_path = os.path.join(directory, "correlation_matrix.csv")
        if os.path.exists(corr_path):
            corr_matrix = pd.read_csv(corr_path, index_col=0)
            
            # Add to correlation data with current date
            manager.correlation_data = {datetime.now(): corr_matrix}
        
        return manager
