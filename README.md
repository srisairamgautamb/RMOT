# Rough Martingale Optimal Transport (RMOT)

A production-ready implementation for exotic option pricing using entropy-regularized optimal transport with rough volatility priors. This framework provides finite, tight price bounds for options under realistic market conditions where volatility exhibits rough path behavior.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Mathematical Foundation](#mathematical-foundation)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Modules](#modules)
- [Multi-Asset Extension](#multi-asset-extension)
- [FRTB Compliance](#frtb-compliance)
- [Testing](#testing)
- [References](#references)
- [License](#license)

---

## Overview

Traditional option pricing approaches face fundamental limitations:
- **Black-Scholes**: Assumes constant volatility, which contradicts empirical market data
- **Classical Martingale Optimal Transport (MOT)**: Provides model-free bounds but yields infinitely wide intervals for deep out-of-the-money options

**RMOT addresses both limitations** by:
- Incorporating rough volatility (Hurst exponent H < 0.5), which accurately captures observed market behavior
- Providing finite, mathematically rigorous bounds through entropy regularization

Market data demonstrates that implied volatility follows rough paths with Hurst exponent H approximately 0.1, rather than the smooth H = 0.5 assumed by classical models.

---

## Key Features

- **Rough Heston Simulation**: Hybrid scheme for simulating rough volatility paths with fractional kernels
- **RMOT Dual Solver**: L-BFGS-B optimization for computing optimal transport bounds
- **Malliavin Calculus Engine**: Greeks computation via pathwise and likelihood ratio methods
- **Fisher Information Analysis**: Parameter identifiability and Cramer-Rao bounds
- **FRTB Compliance**: Non-Modelable Risk Factor (NMRF) capital calculations
- **Multi-Asset Extension**: Correlation identification and basket option pricing
- **Real Market Validation**: Live data integration via yfinance

---

## Mathematical Foundation

### Rough Heston Model

The single-asset rough Heston model is defined by:

**Price Process:**
```
dS_t = S_t * sqrt(v_t) * dW_t^S
```

**Rough Volatility Process:**
```
v_t = v_0 + (1/Gamma(H+1/2)) * integral_0^t (t-s)^(H-1/2) * [kappa*(theta-v_s)*ds + eta*sqrt(v_s)*dW_s^v]
```

### Model Parameters

| Parameter | Symbol | Typical Range | Description |
|-----------|--------|---------------|-------------|
| Hurst Exponent | H | [0.02, 0.45] | Path roughness (lower = rougher) |
| Vol-of-Vol | eta | [0.05, 0.40] | Volatility fluctuation intensity |
| Spot-Vol Correlation | rho | [-0.95, 0] | Leverage effect |
| Initial Variance | xi_0 | [0.01, 0.50] | Starting volatility level |
| Mean Reversion | kappa | [0.5, 5.0] | Speed of reversion |
| Long-term Variance | theta | [0.01, 0.50] | Equilibrium level |

### RMOT Dual Problem

Given rough Heston prior P_rough, the RMOT problem finds the optimal measure P*:

**Primal Problem:**
```
min_{P : P calibrated to liquid strikes} D_KL(P || P_rough)
```

**Dual Problem (solved numerically):**
```
max_lambda { -log E_P[exp(sum lambda_i * (S_T - K_i)_+)] + sum lambda_i * C_market(K_i) }
```

**Error Bound (Theorem 3.8):**
```
|C_RMOT(K) - C_market(K)| <= C * T^(2H) * exp(k - I(k)/(2*T^(2H)))
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- NumPy, SciPy, Numba
- pytest (for testing)
- yfinance (for real market data)

### Setup

```bash
git clone https://github.com/srisairamgautamb/RMOT.git
cd RMOT
pip install numpy scipy numba pytest yfinance
```

---

## Project Structure

```
RMOT/
├── src/
│   ├── simulation/
│   │   └── rough_heston.py          # Rough Heston path simulator
│   ├── pricing/
│   │   └── rmot_solver.py           # Core RMOT pricing engine
│   ├── calibration/
│   │   └── fisher_information.py    # Parameter identifiability analysis
│   ├── sensitivity/
│   │   └── malliavin.py             # Greeks via Malliavin calculus
│   ├── frtb/
│   │   └── compliance.py            # FRTB capital calculations
│   └── validation/
│       ├── convergence_test.py      # O(1/sqrt(N)) verification
│       └── validate_real_market.py  # Real market calibration
├── tests/
│   ├── test_rmot_solver.py
│   ├── test_rough_heston.py
│   ├── test_fisher.py
│   ├── test_malliavin.py
│   ├── test_frtb.py
│   └── verify_solver_correctness.py
├── MULTI/                           # Multi-asset extension
│   ├── src/
│   │   ├── psi_functional.py        # Correlation kernel computation
│   │   ├── correlation_copula.py    # Rough martingale copula
│   │   ├── basket_pricing.py        # Basket option pricer
│   │   ├── frtb_bounds.py           # Multi-asset FRTB bounds
│   │   └── pipeline.py              # End-to-end orchestration
│   └── tests/
│       ├── benchmark_suite.py
│       └── comprehensive_stress_test.py
├── rmot_solver.py                   # Standalone solver module
├── main.py                          # CLI entry point
├── run_all_tests.py                 # Master test runner
└── README.md
```

---

## Usage

### Quick Start

```python
import numpy as np
from rmot_solver import compute_bounds, call_payoff

# Define market data
S0 = 100.0
strikes = np.array([90.0, 100.0, 110.0])
prices = np.array([15.0, 8.0, 4.0])

# Compute bounds for an exotic call option
exotic_strike = 105.0
result = compute_bounds(
    payoff_func=call_payoff(exotic_strike),
    S0=S0,
    strikes=strikes,
    prices=prices,
    H=0.1,
    eta=0.3
)

print(f"Price bounds: [{result['P_min']:.4f}, {result['P_max']:.4f}]")
print(f"Bound width: {result['width']:.4f}")
```

### Using the Pricing Engine

```python
from src.simulation.rough_heston import RoughHestonSimulator, RoughHestonParams
from src.pricing.rmot_solver import RMOTPricingEngine

# Initialize rough Heston parameters
params = RoughHestonParams(
    H=0.1,
    eta=0.3,
    rho=-0.7,
    xi0=0.04,
    kappa=1.0,
    theta=0.04,
    S0=100.0,
    r=0.0
)

# Create simulator and pricing engine
simulator = RoughHestonSimulator(params)
engine = RMOTPricingEngine(simulator, lambda_reg=0.01)

# Solve RMOT dual problem
result = engine.solve_dual_rmot(
    liquid_strikes=np.array([90, 95, 100, 105, 110]),
    liquid_prices=np.array([12.5, 9.0, 6.5, 4.5, 3.0]),
    T=0.25,
    target_strike=115.0,
    n_samples=100000
)

print(f"Target price: {result['target_price']:.4f}")
print(f"Error bound: {result['error_bound']['bound']:.4f}")
```

### Computing Greeks

```python
from src.sensitivity.malliavin import MalliavinEngine

malliavin = MalliavinEngine(simulator)
greeks = malliavin.compute_greeks(
    strikes=np.array([95, 100, 105]),
    T=0.25,
    n_paths=100000
)

print(f"Delta: {greeks['delta']}")
print(f"Gamma: {greeks['gamma']}")
print(f"Vega: {greeks['vega']}")
```

### Command Line Interface

```bash
# Run system verification
python main.py --mode verify

# Run FRTB compliance check
python main.py --mode frtb --portfolio portfolio.csv --market market_data.csv
```

---

## Modules

### Rough Heston Simulator (`src/simulation/rough_heston.py`)

Implements the hybrid scheme for rough Heston simulation with:
- Riemann-Liouville fractional kernel weights
- Numba-accelerated path generation
- European call option pricing via Monte Carlo

### RMOT Pricing Engine (`src/pricing/rmot_solver.py`)

Core pricing functionality including:
- Dual RMOT problem formulation
- L-BFGS-B optimization with adaptive tolerances
- Extrapolation error bounds (Theorem 3.8)
- Calibration constraint verification

### Fisher Information Analyzer (`src/calibration/fisher_information.py`)

Parameter identifiability analysis:
- Fisher information matrix computation
- Effective dimension calculation
- Cramer-Rao bound estimation
- Data sufficiency validation

### Malliavin Calculus Engine (`src/sensitivity/malliavin.py`)

Greeks computation:
- Delta via pathwise method
- Gamma via finite differences
- Vega and rough volatility sensitivities
- Hurst parameter sensitivity (dC/dH)

### FRTB Compliance Engine (`src/frtb/compliance.py`)

Regulatory capital calculations:
- NMRF position identification
- Conservative bound computation
- Capital charge calculation
- Portfolio-level aggregation

---

## Multi-Asset Extension

The `MULTI/` directory contains the novel multi-asset extension implementing:

### Psi Functional (`psi_functional.py`)

The correlation identification kernel defined as:
```
Psi_ij(u_i, u_j; H_i, H_j, T) = u_i * u_j * (eta_i * eta_j) / (4*Gamma(H_i+1/2)*Gamma(H_j+1/2))
    * integral_[0,T]^2 (T-s)^(H_i-1/2) * (T-t)^(H_j-1/2) * E[sqrt(v_s^i * v_t^j)] * Sigma(s,t) ds dt
```

### Rough Martingale Copula (`correlation_copula.py`)

Pilot-calibrated amplification scheme for proper correlation enforcement:
- Target vs realized correlation analysis
- Amplification factor computation
- Multi-asset path simulation

### Basket Option Pricer (`basket_pricing.py`)

Basket option pricing with:
- Weighted basket computation
- Monte Carlo pricing with standard errors
- Strike ordering validation (ITM > ATM > OTM)

### Running Multi-Asset Experiments

```bash
cd MULTI

# Run benchmark suite (21 unit tests)
python -m tests.benchmark_suite

# Run comprehensive stress tests (47 tests)
python tests/comprehensive_stress_test.py

# Run real data experiment
python run_experiment.py --mode batch --tickers SPY QQQ
```

---

## FRTB Compliance

This implementation supports FRTB Non-Modelable Risk Factor (NMRF) capital calculations:

### Key Benefits

- **Data Sufficiency Validation**: Fisher information analysis for parameter identifiability
- **Conservative Bounds**: RMOT provides finite bounds with mathematical guarantees
- **Capital Efficiency**: Reduces capital requirements from worst-case assumptions

### Example Capital Calculation

```python
from src.frtb.compliance import FRTBComplianceEngine, FRTBPosition

# Process portfolio
result = engine.process_portfolio(
    positions=positions,
    liquid_strikes=liquid_strikes,
    liquid_prices=liquid_prices,
    T=0.5
)

print(f"Total Capital Charge: ${result['total_capital_charge']:,.2f}")
print(f"Capital Ratio: {result['capital_ratio']*100:.2f}%")
```

---

## Testing

### Run All Tests

```bash
# Single-asset tests
python run_all_tests.py

# Multi-asset tests
cd MULTI
python tests/comprehensive_stress_test.py
```

### Test Coverage

| Test Suite | Tests | Description |
|------------|-------|-------------|
| Single-Asset Unit | 21 | Core solver functionality |
| Multi-Asset Unit | 21 | Multi-asset modules |
| Stress Tests | 47 | Edge cases and stability |

### Validation Results

| Validation | Result |
|------------|--------|
| Convergence Rate | O(1/sqrt(N)), R-squared = 0.997 |
| Fisher Information | 5x5 matrix positive definite |
| Real Market (SPX) | Calibration error 1.8% |
| Correlation Enforcement | Error = 0.0087 |
| FRTB Scaling | Slope = 0.2000, R-squared = 1.0000 |

---

## References

### Theoretical Foundation

1. Bayer, C., Friz, P., Gassiat, P., Martin, J., and Stemper, B. (2019). "A regularity structure for rough volatility." *Mathematical Finance*, 30(3), 782-832.

2. Gatheral, J., Jaisson, T., and Rosenbaum, M. (2018). "Volatility is rough." *Quantitative Finance*, 18(6), 933-949.

3. Acciaio, B. and Dolinsky, Y. (2020). "Model-free pricing and hedging of exotic claims." *SIAM Journal on Financial Mathematics*, 11(4), 1094-1117.

4. El Euch, O. and Rosenbaum, M. (2019). "The characteristic function of rough Heston models." *Mathematical Finance*, 29(1), 3-38.

### Implementation References

5. Fukasawa, M. (2017). "Short-time at-the-money skew and rough fractional volatility." *Quantitative Finance*, 17(2), 189-198.

6. Bennedsen, M., Lunde, A., and Pakkanen, M. (2017). "Hybrid scheme for Brownian semistationary processes." *Finance and Stochastics*, 21(4), 931-965.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Citation

If you use this software in your research, please cite:

```bibtex
@software{rmot2025,
  author = {Srisairam Gautam B},
  title = {RMOT: Rough Martingale Optimal Transport},
  year = {2025},
  url = {https://github.com/srisairamgautamb/RMOT}
}
```

---

## Contact

For questions or collaboration inquiries, please open an issue on GitHub or contact the maintainer directly.
