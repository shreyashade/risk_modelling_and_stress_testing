# Advanced Risk Modeling and Stress Testing Framework

This project implements a sophisticated risk modeling and stress testing framework for quantitative analysis of financial portfolios. The framework combines multiple advanced techniques including Monte Carlo simulation, historical simulation, Extreme Value Theory (EVT), stress testing, and regime analysis to provide a comprehensive risk assessment toolkit.

## Key Features

- **Dual Simulation Approach**: Implements both Monte Carlo and historical simulation methods for comparative analysis
- **Advanced Tail Risk Modeling**: Utilizes Extreme Value Theory (EVT) for precise modeling of extreme market events
- **Comprehensive Stress Testing**: Includes various stress scenarios including market crashes, interest rate shocks, and volatility spikes
- **Regime Analysis**: Analyzes portfolio performance across different correlation and volatility regimes
- **Interactive Visualizations**: Provides sophisticated dashboards for risk metrics visualization
- **Regulatory Compliance**: Incorporates Basel III and other regulatory frameworks

## Project Structure

```
risk_modeling_project/
├── data/                  # Data storage directory
├── docs/                  # Documentation files
├── notebooks/             # Jupyter notebooks for analysis and examples
├── src/                   # Source code
│   ├── models/            # Risk modeling implementations
│   │   ├── portfolio.py           # Portfolio structure and management
│   │   ├── data_loader.py         # Data loading and preprocessing
│   │   ├── monte_carlo.py         # Monte Carlo simulation
│   │   ├── historical_simulation.py # Historical simulation
│   │   ├── extreme_value_theory.py # EVT implementation
│   │   ├── stress_testing.py      # Stress testing scenarios
│   │   └── regime_analysis.py     # Correlation and volatility regime analysis
│   ├── visualization/      # Visualization tools
│   │   └── dashboard.py           # Interactive dashboards
│   └── research/           # Research documents
│       ├── risk_modeling_techniques.md
│       └── extreme_value_theory.md
├── tests/                 # Unit tests
└── README.md              # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/shreyashade/risk-modelling-and-stress-testing.git
cd risk-modeling-project

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Portfolio Risk Analysis

```python
from src.models.portfolio import Portfolio
from src.models.data_loader import DataLoader
from src.models.monte_carlo import MonteCarloSimulation
from src.visualization.dashboard import RiskDashboard

# Load data
data_loader = DataLoader()
returns = data_loader.load_returns(['AAPL', 'MSFT', 'GOOGL', 'AMZN'], 
                                  start_date='2018-01-01')

# Create portfolio
portfolio = Portfolio(returns)
portfolio.set_weights([0.25, 0.25, 0.25, 0.25])

# Run Monte Carlo simulation
mc_sim = MonteCarloSimulation(portfolio)
mc_results = mc_sim.run_simulation(n_simulations=10000, horizon=20)

# Visualize results
dashboard = RiskDashboard()
dashboard.create_dashboard(returns, portfolio.weights)
dashboard.save_html('risk_dashboard.html')
```

### Stress Testing

```python
from src.models.stress_testing import StressTesting
from src.visualization.dashboard import StressTestDashboard

# Create stress testing scenarios
stress_test = StressTesting(portfolio)
stressed_data = stress_test.run_scenarios([
    'market_crash',
    'interest_rate_shock',
    'volatility_spike'
])

# Visualize stress test results
stress_dashboard = StressTestDashboard()
stress_dashboard.create_dashboard(returns, stressed_data, portfolio.weights)
stress_dashboard.save_html('stress_test_dashboard.html')
```

### Regime Analysis

```python
from src.models.regime_analysis import RegimeAnalysis
from src.visualization.dashboard import RegimeAnalysisDashboard

# Perform regime analysis
regime_analyzer = RegimeAnalysis(returns)
regimes = regime_analyzer.detect_regimes(method='markov_switching')

# Visualize regime analysis results
regime_dashboard = RegimeAnalysisDashboard()
regime_dashboard.create_dashboard(returns, regimes, portfolio.weights)
regime_dashboard.save_html('regime_analysis_dashboard.html')
```

### Extreme Value Theory Analysis

```python
from src.models.extreme_value_theory import EVTAnalysis
from src.visualization.dashboard import EVTTailVisualization

# Perform EVT analysis
evt_analyzer = EVTAnalysis(portfolio.returns)
evt_results = evt_analyzer.fit_gpd(threshold=0.05)

# Visualize EVT results
evt_viz = EVTTailVisualization()
evt_viz.plot(portfolio.returns, evt_results['threshold'])
```

## Advanced Features

### Comprehensive Risk Dashboard

```python
from src.visualization.dashboard import ComprehensiveRiskDashboard

# Create comprehensive dashboard with all analyses
dashboard = ComprehensiveRiskDashboard()
dashboard.create_dashboard(
    returns=returns,
    stressed_data=stressed_data,
    regimes=regimes,
    evt_results=evt_results,
    weights=portfolio.weights
)
dashboard.save_html('comprehensive_risk_report.html')
```

### Custom Stress Scenarios

```python
# Define custom stress scenario
custom_scenario = {
    'name': 'tariff_war',
    'shocks': {
        'AAPL': -0.15,  # 15% drop
        'MSFT': -0.10,  # 10% drop
        'GOOGL': -0.12, # 12% drop
        'AMZN': -0.18   # 18% drop
    },
    'volatility_multiplier': 1.5,
    'correlation_adjustment': 0.2  # Increase correlations
}

# Apply custom scenario
stress_test = StressTesting(portfolio)
custom_results = stress_test.apply_custom_scenario(custom_scenario)
```

## Regulatory Compliance

The framework incorporates Basel III requirements for risk assessment and reporting:

- Value at Risk (VaR) at 99% confidence level
- Expected Shortfall (ES) calculations
- Stress testing under prescribed scenarios
- Liquidity risk assessment
- Counterparty risk evaluation

## Performance Considerations

For large portfolios or extensive simulations, consider:

- Using parallel processing for Monte Carlo simulations
- Implementing GPU acceleration for matrix operations
- Optimizing data storage for historical simulations
- Using incremental calculation methods for rolling metrics

## Contributing

Contributions to enhance the framework are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Financial risk management literature and research papers
- Open-source financial analysis libraries
- Quantitative finance community

## Contact

For questions or feedback, please contact [shreyashade@yahoo.com](mailto:shreyashade@yahoo.com)
