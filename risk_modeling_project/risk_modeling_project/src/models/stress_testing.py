"""
Stress Testing Scenarios for Risk Modeling Framework

This module implements various stress testing scenarios for risk modeling,
including market crashes, interest rate shocks, volatility spikes, and tariff impacts.
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


class StressScenario:
    """Base class for stress testing scenarios"""
    
    def __init__(self, name: str):
        """
        Initialize a StressScenario object
        
        Parameters:
        -----------
        name : str
            Name of the stress scenario
        """
        self.name = name
    
    def apply(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply stress scenario to data
        
        Parameters:
        -----------
        data : pd.DataFrame or np.ndarray
            Data to apply the stress scenario to
        **kwargs : dict
            Additional parameters for the scenario
            
        Returns:
        --------
        pd.DataFrame or np.ndarray
            Stressed data
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_description(self) -> str:
        """
        Get a description of the stress scenario
        
        Returns:
        --------
        str
            Description of the scenario
        """
        return f"Stress scenario: {self.name}"


class HistoricalScenario(StressScenario):
    """Stress scenario based on a historical market event"""
    
    def __init__(self, name: str, event_period: Tuple[str, str], data_source: Optional[pd.DataFrame] = None):
        """
        Initialize a Historical Scenario
        
        Parameters:
        -----------
        name : str
            Name of the scenario
        event_period : tuple of str
            (Start date, end date) of the historical event
        data_source : pd.DataFrame, optional
            Historical data source with DatetimeIndex
        """
        super().__init__(name)
        self.event_period = event_period
        self.data_source = data_source
        self.returns = None
        
        # Extract returns for the event period if data_source is provided
        if data_source is not None:
            self._extract_event_returns()
    
    def _extract_event_returns(self) -> None:
        """Extract returns for the event period from the data source"""
        if not isinstance(self.data_source.index, pd.DatetimeIndex):
            raise ValueError("data_source must have a DatetimeIndex")
        
        start_date, end_date = self.event_period
        
        # Extract data for the event period
        event_data = self.data_source.loc[start_date:end_date]
        
        if len(event_data) == 0:
            raise ValueError(f"No data found for event period {start_date} to {end_date}")
        
        # Calculate returns
        self.returns = event_data.pct_change().dropna()
    
    def apply(self, data: pd.DataFrame, method: str = 'absolute', 
             scale_factor: float = 1.0, **kwargs) -> pd.DataFrame:
        """
        Apply historical scenario to data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to apply the scenario to
        method : str, default='absolute'
            Method to apply returns ('absolute', 'relative', 'cumulative')
        scale_factor : float, default=1.0
            Factor to scale the historical returns
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        pd.DataFrame
            Stressed data
        """
        if self.returns is None:
            raise ValueError("Event returns not available. Provide data_source or set returns manually.")
        
        if len(self.returns) > len(data):
            logger.warning(f"Event period ({len(self.returns)} days) is longer than data ({len(data)} days). Truncating.")
            event_returns = self.returns.iloc[:len(data)]
        else:
            event_returns = self.returns
        
        # Create a copy of the data
        stressed_data = data.copy()
        
        # Apply the scenario based on the method
        if method == 'absolute':
            # Apply the absolute returns from the event
            for i in range(len(event_returns)):
                if i < len(stressed_data):
                    stressed_data.iloc[i] = stressed_data.iloc[0] * (1 + event_returns.iloc[i] * scale_factor)
        
        elif method == 'relative':
            # Apply the returns relative to the previous day
            for i in range(len(event_returns)):
                if i < len(stressed_data):
                    if i == 0:
                        stressed_data.iloc[i] = stressed_data.iloc[0]
                    else:
                        stressed_data.iloc[i] = stressed_data.iloc[i-1] * (1 + event_returns.iloc[i] * scale_factor)
        
        elif method == 'cumulative':
            # Apply the cumulative effect of the event
            cumulative_returns = (1 + event_returns).cumprod() - 1
            for i in range(len(cumulative_returns)):
                if i < len(stressed_data):
                    stressed_data.iloc[i] = stressed_data.iloc[0] * (1 + cumulative_returns.iloc[i] * scale_factor)
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return stressed_data
    
    def get_description(self) -> str:
        """
        Get a description of the historical scenario
        
        Returns:
        --------
        str
            Description of the scenario
        """
        start_date, end_date = self.event_period
        return (f"Historical scenario: {self.name}\n"
                f"Period: {start_date} to {end_date}\n"
                f"Duration: {len(self.returns) if self.returns is not None else 'Unknown'} days")


class MarketCrashScenario(StressScenario):
    """Stress scenario simulating a market crash"""
    
    def __init__(self, name: str = "Market Crash"):
        """Initialize a Market Crash scenario"""
        super().__init__(name)
    
    def apply(self, data: pd.DataFrame, crash_magnitude: float = 0.15, 
             crash_duration: int = 5, recovery_duration: int = 20,
             recovery_type: str = 'exponential', start_day: int = 10, **kwargs) -> pd.DataFrame:
        """
        Apply market crash scenario to data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to apply the scenario to
        crash_magnitude : float, default=0.15
            Magnitude of the crash (e.g., 0.15 for 15% drop)
        crash_duration : int, default=5
            Duration of the crash in days
        recovery_duration : int, default=20
            Duration of the recovery in days
        recovery_type : str, default='exponential'
            Type of recovery ('linear', 'exponential', 'logarithmic', 'none')
        start_day : int, default=10
            Day on which the crash starts
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        pd.DataFrame
            Stressed data
        """
        # Create a copy of the data
        stressed_data = data.copy()
        
        # Check if start_day is valid
        if start_day >= len(data):
            raise ValueError(f"start_day ({start_day}) must be less than data length ({len(data)})")
        
        # Calculate crash and recovery periods
        crash_end = min(start_day + crash_duration, len(data))
        recovery_end = min(crash_end + recovery_duration, len(data))
        
        # Apply crash
        crash_factor = (1 - crash_magnitude) ** (1 / crash_duration)
        for i in range(start_day, crash_end):
            day_in_crash = i - start_day
            stressed_data.iloc[i] = stressed_data.iloc[start_day] * crash_factor ** (day_in_crash + 1)
        
        # Apply recovery
        if recovery_type != 'none' and crash_end < len(data):
            # Calculate start and end values for recovery
            recovery_start_value = stressed_data.iloc[crash_end - 1]
            
            # Target recovery value (partial recovery)
            recovery_target = data.iloc[crash_end - 1] * (1 - crash_magnitude * 0.3)  # Recover 70% of the loss
            
            # Apply recovery based on the specified type
            for i in range(crash_end, recovery_end):
                progress = (i - crash_end) / recovery_duration
                
                if recovery_type == 'linear':
                    recovery_factor = progress
                elif recovery_type == 'exponential':
                    recovery_factor = 1 - np.exp(-3 * progress)
                elif recovery_type == 'logarithmic':
                    recovery_factor = np.log(1 + 9 * progress) / np.log(10)
                else:
                    raise ValueError(f"Unsupported recovery_type: {recovery_type}")
                
                # Apply recovery factor
                for col in stressed_data.columns:
                    stressed_data.loc[stressed_data.index[i], col] = (
                        recovery_start_value[col] + 
                        recovery_factor * (recovery_target[col] - recovery_start_value[col])
                    )
        
        return stressed_data
    
    def get_description(self) -> str:
        """
        Get a description of the market crash scenario
        
        Returns:
        --------
        str
            Description of the scenario
        """
        return (f"Market crash scenario: {self.name}\n"
                f"Simulates a sudden market decline followed by a recovery period.")


class InterestRateShockScenario(StressScenario):
    """Stress scenario simulating an interest rate shock"""
    
    def __init__(self, name: str = "Interest Rate Shock"):
        """Initialize an Interest Rate Shock scenario"""
        super().__init__(name)
    
    def apply(self, data: pd.DataFrame, rate_change: float = 0.01, 
             equity_impact: Dict[str, float] = None,
             bond_impact: Dict[str, float] = None,
             shock_day: int = 10, **kwargs) -> pd.DataFrame:
        """
        Apply interest rate shock scenario to data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to apply the scenario to
        rate_change : float, default=0.01
            Change in interest rate (e.g., 0.01 for 1% increase)
        equity_impact : dict, optional
            Dictionary mapping equity tickers to impact factors
        bond_impact : dict, optional
            Dictionary mapping bond tickers to impact factors (duration)
        shock_day : int, default=10
            Day on which the shock occurs
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        pd.DataFrame
            Stressed data
        """
        # Create a copy of the data
        stressed_data = data.copy()
        
        # Check if shock_day is valid
        if shock_day >= len(data):
            raise ValueError(f"shock_day ({shock_day}) must be less than data length ({len(data)})")
        
        # Default impact factors if not provided
        if equity_impact is None:
            equity_impact = {col: -5.0 for col in data.columns}  # Default: 5% drop per 1% rate increase
        
        if bond_impact is None:
            bond_impact = {col: -7.0 for col in data.columns}  # Default: 7% drop per 1% rate increase (duration ~7)
        
        # Apply shock
        for col in stressed_data.columns:
            # Determine impact based on asset type
            if col in equity_impact:
                impact = equity_impact[col] * rate_change
            elif col in bond_impact:
                impact = bond_impact[col] * rate_change
            else:
                logger.warning(f"No impact factor specified for {col}. Using default equity impact.")
                impact = -5.0 * rate_change
            
            # Apply immediate impact
            stressed_data.loc[stressed_data.index[shock_day], col] = (
                stressed_data.iloc[shock_day - 1][col] * (1 + impact)
            )
            
            # Propagate impact to future days
            for i in range(shock_day + 1, len(stressed_data)):
                # Calculate relative change from original data
                if data.iloc[i - 1][col] != 0:
                    relative_change = data.iloc[i][col] / data.iloc[i - 1][col]
                else:
                    relative_change = 1.0
                
                stressed_data.loc[stressed_data.index[i], col] = (
                    stressed_data.iloc[i - 1][col] * relative_change
                )
        
        return stressed_data
    
    def get_description(self) -> str:
        """
        Get a description of the interest rate shock scenario
        
        Returns:
        --------
        str
            Description of the scenario
        """
        return (f"Interest rate shock scenario: {self.name}\n"
                f"Simulates the impact of a sudden change in interest rates on different asset classes.")


class VolatilityShockScenario(StressScenario):
    """Stress scenario simulating a volatility spike"""
    
    def __init__(self, name: str = "Volatility Shock"):
        """Initialize a Volatility Shock scenario"""
        super().__init__(name)
    
    def apply(self, data: pd.DataFrame, vol_multiplier: float = 3.0,
             shock_duration: int = 10, start_day: int = 10,
             mean_reversion_speed: float = 0.2, **kwargs) -> pd.DataFrame:
        """
        Apply volatility shock scenario to data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to apply the scenario to
        vol_multiplier : float, default=3.0
            Factor by which volatility increases
        shock_duration : int, default=10
            Duration of the shock in days
        start_day : int, default=10
            Day on which the shock starts
        mean_reversion_speed : float, default=0.2
            Speed at which volatility reverts to normal (0-1)
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        pd.DataFrame
            Stressed data
        """
        # Create a copy of the data
        stressed_data = data.copy()
        
        # Check if start_day is valid
        if start_day >= len(data):
            raise ValueError(f"start_day ({start_day}) must be less than data length ({len(data)})")
        
        # Calculate shock end day
        shock_end = min(start_day + shock_duration, len(data))
        
        # Calculate normal volatility for each asset
        normal_vol = {}
        for col in data.columns:
            if start_day > 1:
                # Use pre-shock data to estimate normal volatility
                normal_vol[col] = data.iloc[:start_day][col].pct_change().std()
            else:
                # Use a default value if not enough pre-shock data
                normal_vol[col] = 0.01  # 1% daily volatility
        
        # Apply volatility shock
        for i in range(start_day, shock_end):
            # Calculate current volatility multiplier (decreases over time)
            days_from_start = i - start_day
            current_multiplier = vol_multiplier * ((1 - mean_reversion_speed) ** days_from_start)
            
            for col in stressed_data.columns:
                # Generate new return with increased volatility
                if i > 0:
                    # Calculate mean return from original data
                    mean_return = 0
                    if i > 5:
                        mean_return = data.iloc[i-5:i][col].pct_change().mean()
                    
                    # Generate random return with increased volatility
                    shock_return = np.random.normal(
                        mean_return,
                        normal_vol[col] * current_multiplier
                    )
                    
                    # Apply the shock return
                    stressed_data.loc[stressed_data.index[i], col] = (
                        stressed_data.iloc[i - 1][col] * (1 + shock_return)
                    )
        
        # Continue with original relative changes after shock period
        for i in range(shock_end, len(stressed_data)):
            for col in stressed_data.columns:
                if data.iloc[i - 1][col] != 0:
                    relative_change = data.iloc[i][col] / data.iloc[i - 1][col]
                else:
                    relative_change = 1.0
                
                stressed_data.loc[stressed_data.index[i], col] = (
                    stressed_data.iloc[i - 1][col] * relative_change
                )
        
        return stressed_data
    
    def get_description(self) -> str:
        """
        Get a description of the volatility shock scenario
        
        Returns:
        --------
        str
            Description of the scenario
        """
        return (f"Volatility shock scenario: {self.name}\n"
                f"Simulates a period of increased market volatility followed by gradual normalization.")


class LiquidityCrisisScenario(StressScenario):
    """Stress scenario simulating a liquidity crisis"""
    
    def __init__(self, name: str = "Liquidity Crisis"):
        """Initialize a Liquidity Crisis scenario"""
        super().__init__(name)
    
    def apply(self, data: pd.DataFrame, liquidity_impact: Dict[str, float] = None,
             crisis_magnitude: float = 0.2, crisis_duration: int = 15,
             start_day: int = 10, recovery_duration: int = 30, **kwargs) -> pd.DataFrame:
        """
        Apply liquidity crisis scenario to data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to apply the scenario to
        liquidity_impact : dict, optional
            Dictionary mapping tickers to liquidity impact factors (0-1)
        crisis_magnitude : float, default=0.2
            Base magnitude of the crisis (e.g., 0.2 for 20% impact)
        crisis_duration : int, default=15
            Duration of the acute crisis in days
        start_day : int, default=10
            Day on which the crisis starts
        recovery_duration : int, default=30
            Duration of the recovery in days
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        pd.DataFrame
            Stressed data
        """
        # Create a copy of the data
        stressed_data = data.copy()
        
        # Check if start_day is valid
        if start_day >= len(data):
            raise ValueError(f"start_day ({start_day}) must be less than data length ({len(data)})")
        
        # Default liquidity impact if not provided
        if liquidity_impact is None:
            liquidity_impact = {col: 1.0 for col in data.columns}  # Default: full impact
        
        # Calculate crisis and recovery periods
        crisis_end = min(start_day + crisis_duration, len(data))
        recovery_end = min(crisis_end + recovery_duration, len(data))
        
        # Apply crisis
        for i in range(start_day, crisis_end):
            # Calculate crisis progress (0 to 1)
            progress = (i - start_day) / crisis_duration
            
            # Crisis impact increases over time to peak at the end of the crisis period
            impact_factor = progress * 2 if progress < 0.5 else 2 - progress * 2
            
            for col in stressed_data.columns:
                # Apply impact based on asset liquidity
                asset_impact = crisis_magnitude * liquidity_impact.get(col, 1.0) * impact_factor
                
                if i == start_day:
                    # Initial impact
                    stressed_data.loc[stressed_data.index[i], col] = (
                        stressed_data.iloc[i - 1][col] * (1 - asset_impact)
                    )
                else:
                    # Continued impact with some mean reversion
                    prev_return = (stressed_data.iloc[i - 1][col] / stressed_data.iloc[i - 2][col]) - 1
                    new_return = prev_return * 0.7 - asset_impact * 0.3  # Partial mean reversion
                    
                    stressed_data.loc[stressed_data.index[i], col] = (
                        stressed_data.iloc[i - 1][col] * (1 + new_return)
                    )
        
        # Apply recovery
        if crisis_end < len(data):
            # Calculate start values for recovery
            recovery_start = stressed_data.iloc[crisis_end - 1]
            
            # Target recovery values (partial recovery)
            recovery_target = data.iloc[crisis_end - 1] * 0.9  # Recover to 90% of original value
            
            for i in range(crisis_end, recovery_end):
                # Calculate recovery progress (0 to 1)
                progress = (i - crisis_end) / recovery_duration
                
                # Recovery factor (logarithmic recovery - faster at first, then slowing)
                recovery_factor = np.log(1 + 9 * progress) / np.log(10)
                
                for col in stressed_data.columns:
                    # Apply recovery based on asset liquidity
                    asset_recovery = recovery_factor * liquidity_impact.get(col, 1.0)
                    
                    stressed_data.loc[stressed_data.index[i], col] = (
                        recovery_start[col] + 
                        asset_recovery * (recovery_target[col] - recovery_start[col])
                    )
        
        # Continue with original relative changes after recovery period
        for i in range(recovery_end, len(stressed_data)):
            for col in stressed_data.columns:
                if data.iloc[i - 1][col] != 0:
                    relative_change = data.iloc[i][col] / data.iloc[i - 1][col]
                else:
                    relative_change = 1.0
                
                stressed_data.loc[stressed_data.index[i], col] = (
                    stressed_data.iloc[i - 1][col] * relative_change
                )
        
        return stressed_data
    
    def get_description(self) -> str:
        """
        Get a description of the liquidity crisis scenario
        
        Returns:
        --------
        str
            Description of the scenario
        """
        return (f"Liquidity crisis scenario: {self.name}\n"
                f"Simulates a market-wide liquidity crunch with varying impacts based on asset liquidity.")


class TariffImpactScenario(StressScenario):
    """Stress scenario simulating the impact of tariff announcements"""
    
    def __init__(self, name: str = "Tariff Impact"):
        """Initialize a Tariff Impact scenario"""
        super().__init__(name)
    
    def apply(self, data: pd.DataFrame, 
             announcement_day: int = 5,
             implementation_day: int = 20,
             tariff_rate: float = 0.25,  # 25% tariff
             company_types: Dict[str, str] = None,
             sectors: Dict[str, str] = None,
             domestic_impact: float = 0.02,  # 2% positive impact
             foreign_impact: float = -0.15,  # 15% negative impact
             sector_impacts: Dict[str, float] = None,
             adjustment_period: int = 30,
             **kwargs) -> pd.DataFrame:
        """
        Apply tariff impact scenario to data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to apply the scenario to
        announcement_day : int, default=5
            Day on which the tariff is announced
        implementation_day : int, default=20
            Day on which the tariff is implemented
        tariff_rate : float, default=0.25
            Tariff rate as a percentage (e.g., 0.25 for 25%)
        company_types : dict, optional
            Dictionary mapping tickers to company types ('domestic', 'foreign', 'multinational')
        sectors : dict, optional
            Dictionary mapping tickers to sectors
        domestic_impact : float, default=0.02
            Base impact on domestic companies (percentage)
        foreign_impact : float, default=-0.15
            Base impact on foreign companies (percentage)
        sector_impacts : dict, optional
            Dictionary mapping sectors to impact multipliers
        adjustment_period : int, default=30
            Number of days for market to adjust to new tariffs
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        pd.DataFrame
            Stressed data
        """
        # Create a copy of the data
        stressed_data = data.copy()
        
        # Check if days are valid
        if announcement_day >= len(data) or implementation_day >= len(data):
            raise ValueError(f"announcement_day and implementation_day must be less than data length ({len(data)})")
        
        if implementation_day <= announcement_day:
            raise ValueError("implementation_day must be after announcement_day")
        
        # Default company types if not provided
        if company_types is None:
            company_types = {col: 'multinational' for col in data.columns}
        
        # Default sectors if not provided
        if sectors is None:
            sectors = {col: 'technology' for col in data.columns}
        
        # Default sector impacts if not provided
        if sector_impacts is None:
            sector_impacts = {
                'technology': 1.2,
                'consumer_goods': 1.5,
                'industrial': 1.3,
                'healthcare': 0.7,
                'financial': 0.5,
                'energy': 0.8,
                'utilities': 0.4,
                'materials': 1.4,
                'communication': 0.9,
                'real_estate': 0.6
            }
        
        # Calculate adjustment period end
        adjustment_end = min(implementation_day + adjustment_period, len(data))
        
        # Apply scenario
        for col in stressed_data.columns:
            # Determine base impact based on company type
            company_type = company_types.get(col, 'multinational')
            sector = sectors.get(col, 'technology')
            
            if company_type == 'domestic':
                base_impact = domestic_impact
            elif company_type == 'foreign':
                base_impact = foreign_impact
            elif company_type == 'multinational':
                # For multinationals, impact depends on domestic vs. foreign exposure
                domestic_weight = kwargs.get('domestic_weight', 0.6)
                base_impact = domestic_weight * domestic_impact + (1 - domestic_weight) * foreign_impact
            else:
                logger.warning(f"Unknown company type: {company_type}. Using multinational.")
                domestic_weight = kwargs.get('domestic_weight', 0.6)
                base_impact = domestic_weight * domestic_impact + (1 - domestic_weight) * foreign_impact
            
            # Apply sector-specific multiplier
            sector_multiplier = sector_impacts.get(sector, 1.0)
            impact = base_impact * sector_multiplier * tariff_rate / 0.25  # Scale by tariff rate
            
            # Apply announcement effect
            announcement_impact = impact * 0.5  # Initial reaction is 50% of full impact
            stressed_data.loc[stressed_data.index[announcement_day], col] = (
                stressed_data.iloc[announcement_day - 1][col] * (1 + announcement_impact)
            )
            
            # Period between announcement and implementation
            for i in range(announcement_day + 1, implementation_day):
                # Gradual adjustment with some volatility
                progress = (i - announcement_day) / (implementation_day - announcement_day)
                adjustment_factor = 0.5 + 0.5 * progress
                noise = np.random.normal(0, 0.01)  # Add some noise/uncertainty
                step_impact = impact * adjustment_factor + noise
                
                # Calculate relative change from original data
                if data.iloc[i - 1][col] != 0:
                    relative_change = data.iloc[i][col] / data.iloc[i - 1][col]
                else:
                    relative_change = 1.0
                
                stressed_data.loc[stressed_data.index[i], col] = (
                    stressed_data.iloc[i - 1][col] * relative_change * (1 + step_impact * 0.1)
                )
            
            # Implementation effect
            implementation_impact = impact * 0.7  # Implementation reaction
            stressed_data.loc[stressed_data.index[implementation_day], col] = (
                stressed_data.iloc[implementation_day - 1][col] * (1 + implementation_impact)
            )
            
            # Post-implementation adjustment
            for i in range(implementation_day + 1, adjustment_end):
                # Market adjusts to new normal
                progress = (i - implementation_day) / adjustment_period
                adjustment = 1 - np.exp(-3 * progress)  # Exponential adjustment
                
                # Long-term impact after full adjustment
                long_term_impact = impact * 1.2  # Could be different from initial impact
                
                # Current impact during adjustment period
                current_impact = impact * (1 - adjustment) + long_term_impact * adjustment
                
                # Calculate relative change from original data
                if data.iloc[i - 1][col] != 0:
                    relative_change = data.iloc[i][col] / data.iloc[i - 1][col]
                else:
                    relative_change = 1.0
                
                stressed_data.loc[stressed_data.index[i], col] = (
                    stressed_data.iloc[i - 1][col] * relative_change * (1 + current_impact * 0.1)
                )
            
            # Continue with adjusted path after adjustment period
            for i in range(adjustment_end, len(stressed_data)):
                # Calculate relative change from original data
                if data.iloc[i - 1][col] != 0:
                    relative_change = data.iloc[i][col] / data.iloc[i - 1][col]
                else:
                    relative_change = 1.0
                
                stressed_data.loc[stressed_data.index[i], col] = (
                    stressed_data.iloc[i - 1][col] * relative_change
                )
        
        return stressed_data
    
    def get_description(self) -> str:
        """
        Get a description of the tariff impact scenario
        
        Returns:
        --------
        str
            Description of the scenario
        """
        return (f"Tariff impact scenario: {self.name}\n"
                f"Simulates the market reaction to tariff announcements and implementation, "
                f"with different impacts based on company type and sector.")


class CombinedScenario(StressScenario):
    """Combination of multiple stress scenarios"""
    
    def __init__(self, name: str, scenarios: List[StressScenario], weights: Optional[List[float]] = None):
        """
        Initialize a Combined Scenario
        
        Parameters:
        -----------
        name : str
            Name of the combined scenario
        scenarios : list of StressScenario
            List of scenarios to combine
        weights : list of float, optional
            Weights for each scenario (if None, equal weights are used)
        """
        super().__init__(name)
        self.scenarios = scenarios
        
        # Set weights
        if weights is None:
            self.weights = [1.0 / len(scenarios)] * len(scenarios)
        else:
            if len(weights) != len(scenarios):
                raise ValueError("Number of weights must match number of scenarios")
            
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def apply(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply combined scenario to data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to apply the scenario to
        **kwargs : dict
            Additional parameters passed to individual scenarios
            
        Returns:
        --------
        pd.DataFrame
            Stressed data
        """
        # Create a copy of the data
        original_data = data.copy()
        
        # Apply each scenario and combine results
        combined_data = pd.DataFrame(0, index=data.index, columns=data.columns)
        
        for i, (scenario, weight) in enumerate(zip(self.scenarios, self.weights)):
            # Apply scenario
            scenario_result = scenario.apply(original_data, **kwargs)
            
            # Add weighted result to combined data
            combined_data += scenario_result * weight
        
        return combined_data
    
    def get_description(self) -> str:
        """
        Get a description of the combined scenario
        
        Returns:
        --------
        str
            Description of the scenario
        """
        description = f"Combined scenario: {self.name}\n"
        description += "Combines the following scenarios with weights:\n"
        
        for scenario, weight in zip(self.scenarios, self.weights):
            description += f"- {scenario.name} (weight: {weight:.2f})\n"
        
        return description


class CustomScenario(StressScenario):
    """Custom stress scenario using user-defined functions"""
    
    def __init__(self, name: str, scenario_func: Callable):
        """
        Initialize a Custom Scenario
        
        Parameters:
        -----------
        name : str
            Name of the scenario
        scenario_func : callable
            Function that applies the scenario to data
        """
        super().__init__(name)
        self.scenario_func = scenario_func
    
    def apply(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply custom scenario to data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to apply the scenario to
        **kwargs : dict
            Additional parameters passed to the scenario function
            
        Returns:
        --------
        pd.DataFrame
            Stressed data
        """
        return self.scenario_func(data, **kwargs)
    
    def get_description(self) -> str:
        """
        Get a description of the custom scenario
        
        Returns:
        --------
        str
            Description of the scenario
        """
        return f"Custom scenario: {self.name}"


class StressTestManager:
    """Manager for stress testing scenarios"""
    
    def __init__(self):
        """Initialize a Stress Test Manager"""
        self.scenarios = {}
        self.results = {}
    
    def add_scenario(self, scenario: StressScenario) -> None:
        """
        Add a stress scenario
        
        Parameters:
        -----------
        scenario : StressScenario
            Stress scenario to add
        """
        self.scenarios[scenario.name] = scenario
    
    def create_historical_scenario(self, name: str, event_period: Tuple[str, str], 
                                 data_source: pd.DataFrame) -> None:
        """
        Create and add a historical scenario
        
        Parameters:
        -----------
        name : str
            Name of the scenario
        event_period : tuple of str
            (Start date, end date) of the historical event
        data_source : pd.DataFrame
            Historical data source with DatetimeIndex
        """
        scenario = HistoricalScenario(name, event_period, data_source)
        self.add_scenario(scenario)
    
    def create_market_crash_scenario(self, name: str = "Market Crash", 
                                   crash_magnitude: float = 0.15,
                                   crash_duration: int = 5,
                                   recovery_duration: int = 20) -> None:
        """
        Create and add a market crash scenario
        
        Parameters:
        -----------
        name : str, default="Market Crash"
            Name of the scenario
        crash_magnitude : float, default=0.15
            Magnitude of the crash (e.g., 0.15 for 15% drop)
        crash_duration : int, default=5
            Duration of the crash in days
        recovery_duration : int, default=20
            Duration of the recovery in days
        """
        scenario = MarketCrashScenario(name)
        self.add_scenario(scenario)
    
    def create_interest_rate_shock_scenario(self, name: str = "Interest Rate Shock",
                                         rate_change: float = 0.01) -> None:
        """
        Create and add an interest rate shock scenario
        
        Parameters:
        -----------
        name : str, default="Interest Rate Shock"
            Name of the scenario
        rate_change : float, default=0.01
            Change in interest rate (e.g., 0.01 for 1% increase)
        """
        scenario = InterestRateShockScenario(name)
        self.add_scenario(scenario)
    
    def create_volatility_shock_scenario(self, name: str = "Volatility Shock",
                                      vol_multiplier: float = 3.0,
                                      shock_duration: int = 10) -> None:
        """
        Create and add a volatility shock scenario
        
        Parameters:
        -----------
        name : str, default="Volatility Shock"
            Name of the scenario
        vol_multiplier : float, default=3.0
            Factor by which volatility increases
        shock_duration : int, default=10
            Duration of the shock in days
        """
        scenario = VolatilityShockScenario(name)
        self.add_scenario(scenario)
    
    def create_liquidity_crisis_scenario(self, name: str = "Liquidity Crisis",
                                      crisis_magnitude: float = 0.2,
                                      crisis_duration: int = 15) -> None:
        """
        Create and add a liquidity crisis scenario
        
        Parameters:
        -----------
        name : str, default="Liquidity Crisis"
            Name of the scenario
        crisis_magnitude : float, default=0.2
            Base magnitude of the crisis (e.g., 0.2 for 20% impact)
        crisis_duration : int, default=15
            Duration of the acute crisis in days
        """
        scenario = LiquidityCrisisScenario(name)
        self.add_scenario(scenario)
    
    def create_tariff_impact_scenario(self, name: str = "Tariff Impact",
                                    tariff_rate: float = 0.25,
                                    company_types: Dict[str, str] = None,
                                    sectors: Dict[str, str] = None) -> None:
        """
        Create and add a tariff impact scenario
        
        Parameters:
        -----------
        name : str, default="Tariff Impact"
            Name of the scenario
        tariff_rate : float, default=0.25
            Tariff rate as a percentage (e.g., 0.25 for 25%)
        company_types : dict, optional
            Dictionary mapping tickers to company types ('domestic', 'foreign', 'multinational')
        sectors : dict, optional
            Dictionary mapping tickers to sectors
        """
        scenario = TariffImpactScenario(name)
        self.add_scenario(scenario)
    
    def create_combined_scenario(self, name: str, scenario_names: List[str], 
                               weights: Optional[List[float]] = None) -> None:
        """
        Create and add a combined scenario
        
        Parameters:
        -----------
        name : str
            Name of the combined scenario
        scenario_names : list of str
            Names of scenarios to combine
        weights : list of float, optional
            Weights for each scenario (if None, equal weights are used)
        """
        # Check if all scenarios exist
        scenarios = []
        for scenario_name in scenario_names:
            if scenario_name not in self.scenarios:
                raise ValueError(f"Scenario '{scenario_name}' not found")
            
            scenarios.append(self.scenarios[scenario_name])
        
        # Create combined scenario
        scenario = CombinedScenario(name, scenarios, weights)
        self.add_scenario(scenario)
    
    def run_stress_test(self, scenario_name: str, data: pd.DataFrame, **kwargs) -> str:
        """
        Run a stress test
        
        Parameters:
        -----------
        scenario_name : str
            Name of the scenario to run
        data : pd.DataFrame
            Data to apply the scenario to
        **kwargs : dict
            Additional parameters for the scenario
            
        Returns:
        --------
        str
            Result ID for retrieving stress test results
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        scenario = self.scenarios[scenario_name]
        
        # Run stress test
        stressed_data = scenario.apply(data, **kwargs)
        
        # Store results
        result_id = f"{scenario_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.results[result_id] = {
            'scenario': scenario_name,
            'original_data': data,
            'stressed_data': stressed_data,
            'params': kwargs
        }
        
        return result_id
    
    def get_results(self, result_id: str) -> Dict:
        """
        Get stress test results
        
        Parameters:
        -----------
        result_id : str
            Result ID from a previous stress test
            
        Returns:
        --------
        dict
            Dictionary with stress test results
        """
        if result_id not in self.results:
            raise ValueError(f"Result '{result_id}' not found")
        
        return self.results[result_id]
    
    def calculate_impact(self, result_id: str) -> pd.DataFrame:
        """
        Calculate impact of a stress test
        
        Parameters:
        -----------
        result_id : str
            Result ID from a previous stress test
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with impact metrics
        """
        if result_id not in self.results:
            raise ValueError(f"Result '{result_id}' not found")
        
        result = self.results[result_id]
        original_data = result['original_data']
        stressed_data = result['stressed_data']
        
        # Calculate absolute and relative changes
        absolute_change = stressed_data - original_data
        relative_change = absolute_change / original_data
        
        # Calculate summary statistics
        impact = pd.DataFrame(index=original_data.columns)
        
        # Maximum drawdown
        for col in original_data.columns:
            original_max_drawdown = self._calculate_max_drawdown(original_data[col])
            stressed_max_drawdown = self._calculate_max_drawdown(stressed_data[col])
            impact.loc[col, 'Original Max Drawdown'] = original_max_drawdown
            impact.loc[col, 'Stressed Max Drawdown'] = stressed_max_drawdown
            impact.loc[col, 'Max Drawdown Change'] = stressed_max_drawdown - original_max_drawdown
        
        # Final impact
        impact['Final Absolute Change'] = absolute_change.iloc[-1]
        impact['Final Relative Change'] = relative_change.iloc[-1]
        
        # Maximum impact
        impact['Max Absolute Change'] = absolute_change.min()
        impact['Max Relative Change'] = relative_change.min()
        
        # Volatility change
        original_vol = original_data.pct_change().std()
        stressed_vol = stressed_data.pct_change().std()
        impact['Original Volatility'] = original_vol
        impact['Stressed Volatility'] = stressed_vol
        impact['Volatility Change'] = stressed_vol - original_vol
        impact['Volatility Ratio'] = stressed_vol / original_vol
        
        return impact
    
    def _calculate_max_drawdown(self, series: pd.Series) -> float:
        """Calculate maximum drawdown for a series"""
        # Calculate cumulative maximum
        cummax = series.cummax()
        
        # Calculate drawdown
        drawdown = (series / cummax - 1)
        
        # Return maximum drawdown
        return drawdown.min()
    
    def compare_scenarios(self, result_ids: List[str], metric: str = 'Max Drawdown Change') -> pd.DataFrame:
        """
        Compare results from different scenarios
        
        Parameters:
        -----------
        result_ids : list of str
            Result IDs to compare
        metric : str, default='Max Drawdown Change'
            Metric to compare
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with comparison results
        """
        comparison = pd.DataFrame()
        
        for result_id in result_ids:
            if result_id not in self.results:
                raise ValueError(f"Result '{result_id}' not found")
            
            result = self.results[result_id]
            scenario_name = result['scenario']
            
            # Calculate impact
            impact = self.calculate_impact(result_id)
            
            # Add to comparison
            if metric in impact.columns:
                comparison[scenario_name] = impact[metric]
            else:
                raise ValueError(f"Metric '{metric}' not found in impact metrics")
        
        return comparison
    
    def plot_scenario(self, result_id: str, tickers: Optional[List[str]] = None,
                    figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot stress test results
        
        Parameters:
        -----------
        result_id : str
            Result ID from a previous stress test
        tickers : list of str, optional
            Tickers to plot (if None, all tickers are plotted)
        figsize : tuple, default=(12, 8)
            Figure size
        """
        if result_id not in self.results:
            raise ValueError(f"Result '{result_id}' not found")
        
        result = self.results[result_id]
        original_data = result['original_data']
        stressed_data = result['stressed_data']
        scenario_name = result['scenario']
        
        # Select tickers to plot
        if tickers is None:
            tickers = original_data.columns
        else:
            # Check if all tickers exist
            for ticker in tickers:
                if ticker not in original_data.columns:
                    raise ValueError(f"Ticker '{ticker}' not found in data")
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot each ticker
        for ticker in tickers:
            plt.subplot(len(tickers), 1, list(tickers).index(ticker) + 1)
            
            # Plot original and stressed data
            plt.plot(original_data.index, original_data[ticker], 'b-', label='Original')
            plt.plot(stressed_data.index, stressed_data[ticker], 'r-', label='Stressed')
            
            plt.title(f"{ticker} - {scenario_name}")
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_comparison(self, result_ids: List[str], ticker: str,
                      figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot comparison of multiple scenarios for a single ticker
        
        Parameters:
        -----------
        result_ids : list of str
            Result IDs to compare
        ticker : str
            Ticker to plot
        figsize : tuple, default=(12, 6)
            Figure size
        """
        plt.figure(figsize=figsize)
        
        # Plot original data
        original_data = None
        
        for result_id in result_ids:
            if result_id not in self.results:
                raise ValueError(f"Result '{result_id}' not found")
            
            result = self.results[result_id]
            
            if ticker not in result['original_data'].columns:
                raise ValueError(f"Ticker '{ticker}' not found in data")
            
            if original_data is None:
                original_data = result['original_data'][ticker]
                plt.plot(original_data.index, original_data, 'k-', label='Original')
            
            # Plot stressed data
            scenario_name = result['scenario']
            stressed_data = result['stressed_data'][ticker]
            
            plt.plot(stressed_data.index, stressed_data, label=scenario_name)
        
        plt.title(f"{ticker} - Scenario Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_impact_heatmap(self, result_id: str, metric: str = 'Final Relative Change',
                          figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot impact heatmap
        
        Parameters:
        -----------
        result_id : str
            Result ID from a previous stress test
        metric : str, default='Final Relative Change'
            Metric to plot
        figsize : tuple, default=(10, 8)
            Figure size
        """
        if result_id not in self.results:
            raise ValueError(f"Result '{result_id}' not found")
        
        # Calculate impact
        impact = self.calculate_impact(result_id)
        
        if metric not in impact.columns:
            raise ValueError(f"Metric '{metric}' not found in impact metrics")
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot heatmap
        plt.imshow([impact[metric]], cmap='RdYlGn', aspect='auto')
        
        # Add colorbar
        plt.colorbar(label=metric)
        
        # Add labels
        plt.yticks([0], [self.results[result_id]['scenario']])
        plt.xticks(range(len(impact.index)), impact.index, rotation=90)
        
        plt.title(f"Impact Heatmap - {metric}")
        plt.tight_layout()
        plt.show()
    
    def save_results(self, result_id: str, filepath: str) -> None:
        """
        Save stress test results to file
        
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
        Load stress test results from file
        
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


# Define some pre-configured historical scenarios
def create_financial_crisis_scenario(data_source: pd.DataFrame) -> HistoricalScenario:
    """
    Create a scenario based on the 2008 financial crisis
    
    Parameters:
    -----------
    data_source : pd.DataFrame
        Historical data source with DatetimeIndex
        
    Returns:
    --------
    HistoricalScenario
        Financial crisis scenario
    """
    return HistoricalScenario(
        name="2008 Financial Crisis",
        event_period=("2008-09-01", "2009-03-31"),
        data_source=data_source
    )

def create_covid_crash_scenario(data_source: pd.DataFrame) -> HistoricalScenario:
    """
    Create a scenario based on the COVID-19 market crash
    
    Parameters:
    -----------
    data_source : pd.DataFrame
        Historical data source with DatetimeIndex
        
    Returns:
    --------
    HistoricalScenario
        COVID-19 crash scenario
    """
    return HistoricalScenario(
        name="COVID-19 Crash",
        event_period=("2020-02-15", "2020-04-15"),
        data_source=data_source
    )

def create_tech_bubble_scenario(data_source: pd.DataFrame) -> HistoricalScenario:
    """
    Create a scenario based on the dot-com bubble burst
    
    Parameters:
    -----------
    data_source : pd.DataFrame
        Historical data source with DatetimeIndex
        
    Returns:
    --------
    HistoricalScenario
        Dot-com bubble scenario
    """
    return HistoricalScenario(
        name="Dot-Com Bubble Burst",
        event_period=("2000-03-01", "2000-12-31"),
        data_source=data_source
    )

def create_black_monday_scenario(data_source: pd.DataFrame) -> HistoricalScenario:
    """
    Create a scenario based on Black Monday
    
    Parameters:
    -----------
    data_source : pd.DataFrame
        Historical data source with DatetimeIndex
        
    Returns:
    --------
    HistoricalScenario
        Black Monday scenario
    """
    return HistoricalScenario(
        name="Black Monday",
        event_period=("1987-10-14", "1987-11-30"),
        data_source=data_source
    )

def create_2018_correction_scenario(data_source: pd.DataFrame) -> HistoricalScenario:
    """
    Create a scenario based on the 2018 market correction
    
    Parameters:
    -----------
    data_source : pd.DataFrame
        Historical data source with DatetimeIndex
        
    Returns:
    --------
    HistoricalScenario
        2018 correction scenario
    """
    return HistoricalScenario(
        name="2018 Market Correction",
        event_period=("2018-10-01", "2018-12-31"),
        data_source=data_source
    )

def create_trump_tariff_scenario(data_source: pd.DataFrame) -> HistoricalScenario:
    """
    Create a scenario based on Trump's tariff announcements
    
    Parameters:
    -----------
    data_source : pd.DataFrame
        Historical data source with DatetimeIndex
        
    Returns:
    --------
    HistoricalScenario
        Trump tariff scenario
    """
    return HistoricalScenario(
        name="Trump Tariff Announcement",
        event_period=("2018-03-01", "2018-04-30"),
        data_source=data_source
    )
