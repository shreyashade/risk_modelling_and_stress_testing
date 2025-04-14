"""
Portfolio Structure and Data Model for Risk Modeling and Stress Testing Framework

This module defines the portfolio structure for equities and mixed assets,
including classes for assets, positions, and portfolios.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Union, Optional, Tuple
from enum import Enum


class AssetType(Enum):
    """Enumeration of supported asset types"""
    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    FOREX = "forex"
    OPTION = "option"
    FUTURE = "future"
    ETF = "etf"
    CRYPTO = "crypto"


class Asset:
    """Base class for all financial assets"""
    
    def __init__(
        self,
        ticker: str,
        asset_type: AssetType,
        name: Optional[str] = None,
        currency: str = "USD",
        sector: Optional[str] = None,
        country: Optional[str] = None,
        exchange: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize an Asset object
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol or identifier of the asset
        asset_type : AssetType
            Type of the asset (equity, bond, etc.)
        name : str, optional
            Full name of the asset
        currency : str, default="USD"
            Currency in which the asset is denominated
        sector : str, optional
            Industry sector for the asset
        country : str, optional
            Country of domicile
        exchange : str, optional
            Exchange where the asset is traded
        metadata : dict, optional
            Additional metadata for the asset
        """
        self.ticker = ticker
        self.asset_type = asset_type
        self.name = name if name else ticker
        self.currency = currency
        self.sector = sector
        self.country = country
        self.exchange = exchange
        self.metadata = metadata if metadata else {}
        self._price_history = None
        self._returns_history = None
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ticker='{self.ticker}', type={self.asset_type.value})"
    
    def load_price_history(self, data: pd.DataFrame) -> None:
        """
        Load historical price data for the asset
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing price history with DatetimeIndex
        """
        self._price_history = data
        
    def calculate_returns(self, method: str = 'log') -> pd.DataFrame:
        """
        Calculate returns from price history
        
        Parameters:
        -----------
        method : str, default='log'
            Method to calculate returns ('log' or 'simple')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing calculated returns
        """
        if self._price_history is None:
            raise ValueError("Price history not loaded. Call load_price_history first.")
        
        if method == 'log':
            self._returns_history = np.log(self._price_history / self._price_history.shift(1))
        elif method == 'simple':
            self._returns_history = self._price_history / self._price_history.shift(1) - 1
        else:
            raise ValueError("Method must be either 'log' or 'simple'")
            
        # Drop NaN values (first row)
        self._returns_history = self._returns_history.dropna()
        
        return self._returns_history
    
    @property
    def price_history(self) -> pd.DataFrame:
        """Get the price history"""
        if self._price_history is None:
            raise ValueError("Price history not loaded")
        return self._price_history
    
    @property
    def returns_history(self) -> pd.DataFrame:
        """Get the returns history"""
        if self._returns_history is None:
            raise ValueError("Returns not calculated. Call calculate_returns first.")
        return self._returns_history


class Equity(Asset):
    """Class representing equity assets"""
    
    def __init__(
        self,
        ticker: str,
        name: Optional[str] = None,
        currency: str = "USD",
        sector: Optional[str] = None,
        country: Optional[str] = None,
        exchange: Optional[str] = None,
        market_cap: Optional[float] = None,
        beta: Optional[float] = None,
        dividend_yield: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize an Equity object
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol of the equity
        name : str, optional
            Full name of the company
        currency : str, default="USD"
            Currency in which the equity is denominated
        sector : str, optional
            Industry sector for the equity
        country : str, optional
            Country of domicile
        exchange : str, optional
            Exchange where the equity is traded
        market_cap : float, optional
            Market capitalization in millions
        beta : float, optional
            Beta coefficient (market sensitivity)
        dividend_yield : float, optional
            Dividend yield as a decimal (e.g., 0.02 for 2%)
        metadata : dict, optional
            Additional metadata for the equity
        """
        super().__init__(
            ticker=ticker,
            asset_type=AssetType.EQUITY,
            name=name,
            currency=currency,
            sector=sector,
            country=country,
            exchange=exchange,
            metadata=metadata
        )
        self.market_cap = market_cap
        self.beta = beta
        self.dividend_yield = dividend_yield


class Bond(Asset):
    """Class representing fixed income assets"""
    
    def __init__(
        self,
        ticker: str,
        name: Optional[str] = None,
        currency: str = "USD",
        issuer: Optional[str] = None,
        issuer_type: Optional[str] = None,
        country: Optional[str] = None,
        coupon: Optional[float] = None,
        maturity_date: Optional[datetime] = None,
        credit_rating: Optional[str] = None,
        duration: Optional[float] = None,
        convexity: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize a Bond object
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol or identifier of the bond
        name : str, optional
            Full name of the bond
        currency : str, default="USD"
            Currency in which the bond is denominated
        issuer : str, optional
            Name of the bond issuer
        issuer_type : str, optional
            Type of issuer (e.g., 'government', 'corporate')
        country : str, optional
            Country of the issuer
        coupon : float, optional
            Annual coupon rate as a decimal (e.g., 0.05 for 5%)
        maturity_date : datetime, optional
            Maturity date of the bond
        credit_rating : str, optional
            Credit rating of the bond (e.g., 'AAA', 'BBB')
        duration : float, optional
            Modified duration of the bond
        convexity : float, optional
            Convexity of the bond
        metadata : dict, optional
            Additional metadata for the bond
        """
        super().__init__(
            ticker=ticker,
            asset_type=AssetType.BOND,
            name=name,
            currency=currency,
            country=country,
            metadata=metadata
        )
        self.issuer = issuer
        self.issuer_type = issuer_type
        self.coupon = coupon
        self.maturity_date = maturity_date
        self.credit_rating = credit_rating
        self.duration = duration
        self.convexity = convexity


class Position:
    """Class representing a position in a financial asset"""
    
    def __init__(
        self,
        asset: Asset,
        quantity: float,
        entry_price: float,
        entry_date: datetime = None,
        currency: str = "USD"
    ):
        """
        Initialize a Position object
        
        Parameters:
        -----------
        asset : Asset
            The underlying asset
        quantity : float
            Quantity of the asset held
        entry_price : float
            Average entry price per unit
        entry_date : datetime, optional
            Date when the position was entered
        currency : str, default="USD"
            Currency of the position
        """
        self.asset = asset
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_date = entry_date if entry_date else datetime.now()
        self.currency = currency
        
    def __repr__(self) -> str:
        return f"Position(asset='{self.asset.ticker}', quantity={self.quantity})"
    
    @property
    def market_value(self) -> float:
        """
        Calculate the current market value of the position
        
        Returns:
        --------
        float
            Current market value
        """
        # In a real implementation, this would fetch the current price
        # For now, we'll use the last price from price history if available
        current_price = self.entry_price
        if hasattr(self.asset, 'price_history') and self.asset._price_history is not None:
            current_price = self.asset.price_history.iloc[-1].values[0]
            
        return self.quantity * current_price
    
    def calculate_pnl(self, current_price: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate profit and loss for the position
        
        Parameters:
        -----------
        current_price : float, optional
            Current price of the asset. If None, uses the last price from price history
            
        Returns:
        --------
        tuple
            (absolute P&L, percentage P&L)
        """
        if current_price is None:
            if hasattr(self.asset, 'price_history') and self.asset._price_history is not None:
                current_price = self.asset.price_history.iloc[-1].values[0]
            else:
                raise ValueError("Current price not provided and price history not available")
        
        absolute_pnl = self.quantity * (current_price - self.entry_price)
        percentage_pnl = (current_price / self.entry_price - 1) * 100
        
        return absolute_pnl, percentage_pnl


class Portfolio:
    """Class representing a portfolio of positions"""
    
    def __init__(
        self,
        name: str,
        positions: Optional[List[Position]] = None,
        base_currency: str = "USD",
        creation_date: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize a Portfolio object
        
        Parameters:
        -----------
        name : str
            Name of the portfolio
        positions : list of Position, optional
            Initial positions in the portfolio
        base_currency : str, default="USD"
            Base currency of the portfolio
        creation_date : datetime, optional
            Creation date of the portfolio
        metadata : dict, optional
            Additional metadata for the portfolio
        """
        self.name = name
        self.positions = positions if positions else []
        self.base_currency = base_currency
        self.creation_date = creation_date if creation_date else datetime.now()
        self.metadata = metadata if metadata else {}
        self._position_history = []
        
    def __repr__(self) -> str:
        return f"Portfolio(name='{self.name}', positions={len(self.positions)})"
    
    def add_position(self, position: Position) -> None:
        """
        Add a position to the portfolio
        
        Parameters:
        -----------
        position : Position
            Position to add
        """
        self.positions.append(position)
        self._position_history.append({
            'date': datetime.now(),
            'action': 'add',
            'ticker': position.asset.ticker,
            'quantity': position.quantity,
            'price': position.entry_price
        })
    
    def remove_position(self, position_index: int) -> Position:
        """
        Remove a position from the portfolio
        
        Parameters:
        -----------
        position_index : int
            Index of the position to remove
            
        Returns:
        --------
        Position
            The removed position
        """
        if position_index < 0 or position_index >= len(self.positions):
            raise IndexError("Position index out of range")
            
        position = self.positions.pop(position_index)
        self._position_history.append({
            'date': datetime.now(),
            'action': 'remove',
            'ticker': position.asset.ticker,
            'quantity': position.quantity,
            'price': None  # Exit price not specified here
        })
        
        return position
    
    def update_position(self, position_index: int, new_quantity: float, price: float) -> None:
        """
        Update the quantity of an existing position
        
        Parameters:
        -----------
        position_index : int
            Index of the position to update
        new_quantity : float
            New quantity of the asset
        price : float
            Price at which the update occurs
        """
        if position_index < 0 or position_index >= len(self.positions):
            raise IndexError("Position index out of range")
            
        position = self.positions[position_index]
        old_quantity = position.quantity
        position.quantity = new_quantity
        
        self._position_history.append({
            'date': datetime.now(),
            'action': 'update',
            'ticker': position.asset.ticker,
            'quantity_change': new_quantity - old_quantity,
            'price': price
        })
    
    @property
    def total_value(self) -> float:
        """
        Calculate the total value of the portfolio
        
        Returns:
        --------
        float
            Total portfolio value
        """
        return sum(position.market_value for position in self.positions)
    
    def get_weights(self) -> Dict[str, float]:
        """
        Calculate the weight of each position in the portfolio
        
        Returns:
        --------
        dict
            Dictionary mapping asset tickers to their weights
        """
        total = self.total_value
        if total == 0:
            return {position.asset.ticker: 0 for position in self.positions}
            
        return {position.asset.ticker: position.market_value / total 
                for position in self.positions}
    
    def get_asset_allocation(self) -> Dict[str, float]:
        """
        Calculate the allocation by asset type
        
        Returns:
        --------
        dict
            Dictionary mapping asset types to their allocation percentages
        """
        total = self.total_value
        if total == 0:
            return {asset_type.value: 0 for asset_type in AssetType}
            
        allocation = {}
        for asset_type in AssetType:
            type_value = sum(position.market_value for position in self.positions 
                            if position.asset.asset_type == asset_type)
            allocation[asset_type.value] = type_value / total
            
        return allocation
    
    def get_sector_allocation(self) -> Dict[str, float]:
        """
        Calculate the allocation by sector for equity positions
        
        Returns:
        --------
        dict
            Dictionary mapping sectors to their allocation percentages
        """
        equity_value = sum(position.market_value for position in self.positions 
                          if position.asset.asset_type == AssetType.EQUITY)
        
        if equity_value == 0:
            return {}
            
        sectors = {}
        for position in self.positions:
            if position.asset.asset_type == AssetType.EQUITY and position.asset.sector:
                if position.asset.sector not in sectors:
                    sectors[position.asset.sector] = 0
                sectors[position.asset.sector] += position.market_value / equity_value
                
        return sectors
    
    def calculate_returns(self, start_date: datetime, end_date: datetime = None) -> pd.Series:
        """
        Calculate historical returns for the portfolio
        
        Parameters:
        -----------
        start_date : datetime
            Start date for the calculation
        end_date : datetime, optional
            End date for the calculation. If None, uses current date
            
        Returns:
        --------
        pd.Series
            Series of portfolio returns
        """
        if end_date is None:
            end_date = datetime.now()
            
        # Check if all assets have returns history
        for position in self.positions:
            if position.asset._returns_history is None:
                position.asset.calculate_returns()
                
        # Get all unique dates across all assets
        all_dates = set()
        for position in self.positions:
            dates = position.asset.returns_history.index
            dates = [d for d in dates if start_date <= d <= end_date]
            all_dates.update(dates)
            
        all_dates = sorted(all_dates)
        
        # Calculate weighted returns for each date
        portfolio_returns = pd.Series(index=all_dates, dtype=float)
        weights = self.get_weights()
        
        for date in all_dates:
            daily_return = 0
            for position in self.positions:
                ticker = position.asset.ticker
                if ticker in weights and weights[ticker] > 0:
                    asset_returns = position.asset.returns_history
                    if date in asset_returns.index:
                        daily_return += weights[ticker] * asset_returns.loc[date].values[0]
            
            portfolio_returns[date] = daily_return
            
        return portfolio_returns
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert portfolio to a DataFrame
        
        Returns:
        --------
        pd.DataFrame
            DataFrame representation of the portfolio
        """
        data = []
        for position in self.positions:
            data.append({
                'ticker': position.asset.ticker,
                'name': position.asset.name,
                'asset_type': position.asset.asset_type.value,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_value': position.market_value,
                'weight': self.get_weights().get(position.asset.ticker, 0),
                'currency': position.currency
            })
            
        return pd.DataFrame(data)
    
    def save_to_csv(self, filepath: str) -> None:
        """
        Save portfolio to CSV file
        
        Parameters:
        -----------
        filepath : str
            Path to save the CSV file
        """
        self.to_dataframe().to_csv(filepath, index=False)
    
    @classmethod
    def load_from_csv(cls, filepath: str, assets_dict: Dict[str, Asset]) -> 'Portfolio':
        """
        Load portfolio from CSV file
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
        assets_dict : dict
            Dictionary mapping tickers to Asset objects
            
        Returns:
        --------
        Portfolio
            Loaded portfolio
        """
        df = pd.read_csv(filepath)
        portfolio = cls(name=f"Portfolio_from_{filepath.split('/')[-1]}")
        
        for _, row in df.iterrows():
            ticker = row['ticker']
            if ticker in assets_dict:
                position = Position(
                    asset=assets_dict[ticker],
                    quantity=row['quantity'],
                    entry_price=row['entry_price'],
                    currency=row['currency']
                )
                portfolio.add_position(position)
                
        return portfolio


class PortfolioManager:
    """Class for managing multiple portfolios and performing operations on them"""
    
    def __init__(self):
        """Initialize a PortfolioManager object"""
        self.portfolios = {}
        self.assets = {}
        
    def add_portfolio(self, portfolio: Portfolio) -> None:
        """
        Add a portfolio to the manager
        
        Parameters:
        -----------
        portfolio : Portfolio
            Portfolio to add
        """
        self.portfolios[portfolio.name] = portfolio
        
    def remove_portfolio(self, portfolio_name: str) -> Portfolio:
        """
        Remove a portfolio from the manager
        
        Parameters:
        -----------
        portfolio_name : str
            Name of the portfolio to remove
            
        Returns:
        --------
        Portfolio
            The removed portfolio
        """
        if portfolio_name not in self.portfolios:
            raise KeyError(f"Portfolio '{portfolio_name}' not found")
            
        return self.portfolios.pop(portfolio_name)
    
    def add_asset(self, asset: Asset) -> None:
        """
        Add an asset to the manager
        
        Parameters:
        -----------
        asset : Asset
            Asset to add
        """
        self.assets[asset.ticker] = asset
        
    def create_portfolio(self, name: str, positions_data: List[Dict]) -> Portfolio:
        """
        Create a new portfolio from positions data
        
        Parameters:
        -----------
        name : str
            Name of the portfolio
        positions_data : list of dict
            List of dictionaries with position data
            
        Returns:
        --------
        Portfolio
            The created portfolio
        """
        portfolio = Portfolio(name=name)
        
        for pos_data in positions_data:
            ticker = pos_data['ticker']
            if ticker not in self.assets:
                raise KeyError(f"Asset '{ticker}' not found")
                
            position = Position(
                asset=self.assets[ticker],
                quantity=pos_data['quantity'],
                entry_price=pos_data['entry_price'],
                entry_date=pos_data.get('entry_date'),
                currency=pos_data.get('currency', 'USD')
            )
            portfolio.add_position(position)
            
        self.add_portfolio(portfolio)
        return portfolio
    
    def calculate_correlation_matrix(self, start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
        """
        Calculate correlation matrix for all assets
        
        Parameters:
        -----------
        start_date : datetime
            Start date for the calculation
        end_date : datetime, optional
            End date for the calculation
            
        Returns:
        --------
        pd.DataFrame
            Correlation matrix
        """
        if end_date is None:
            end_date = datetime.now()
            
        # Ensure all assets have returns calculated
        for ticker, asset in self.assets.items():
            if asset._returns_history is None:
                try:
                    asset.calculate_returns()
                except ValueError:
                    print(f"Warning: No price history for {ticker}")
        
        # Collect all returns in a single DataFrame
        all_returns = pd.DataFrame()
        
        for ticker, asset in self.assets.items():
            if asset._returns_history is not None:
                returns = asset.returns_history
                returns = returns[(returns.index >= start_date) & (returns.index <= end_date)]
                if not returns.empty:
                    all_returns[ticker] = returns.iloc[:, 0]
        
        # Calculate correlation matrix
        return all_returns.corr()
    
    def save_state(self, directory: str) -> None:
        """
        Save the state of the portfolio manager
        
        Parameters:
        -----------
        directory : str
            Directory to save the state
        """
        import os
        import pickle
        
        os.makedirs(directory, exist_ok=True)
        
        # Save portfolios
        for name, portfolio in self.portfolios.items():
            portfolio.save_to_csv(os.path.join(directory, f"{name}.csv"))
            
        # Save assets
        with open(os.path.join(directory, "assets.pkl"), "wb") as f:
            pickle.dump(self.assets, f)
    
    @classmethod
    def load_state(cls, directory: str) -> 'PortfolioManager':
        """
        Load the state of the portfolio manager
        
        Parameters:
        -----------
        directory : str
            Directory to load the state from
            
        Returns:
        --------
        PortfolioManager
            Loaded portfolio manager
        """
        import os
        import pickle
        
        manager = cls()
        
        # Load assets
        assets_path = os.path.join(directory, "assets.pkl")
        if os.path.exists(assets_path):
            with open(assets_path, "rb") as f:
                manager.assets = pickle.load(f)
        
        # Load portfolios
        for filename in os.listdir(directory):
            if filename.endswith(".csv") and filename != "assets.csv":
                portfolio_name = filename[:-4]  # Remove .csv extension
                portfolio_path = os.path.join(directory, filename)
                portfolio = Portfolio.load_from_csv(portfolio_path, manager.assets)
                portfolio.name = portfolio_name
                manager.add_portfolio(portfolio)
                
        return manager
