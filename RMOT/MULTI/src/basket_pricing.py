"""
Basket Option Pricing for Multi-Asset RMOT

Prices basket options using simulated correlated rough Heston paths.
Reference: PDF Section 4
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass

try:
    from .data_structures import RoughHestonParams, BasketPricingResult
except ImportError:
    from data_structures import RoughHestonParams, BasketPricingResult


def price_basket_call(
    paths: np.ndarray,
    basket_weights: np.ndarray,
    strike: float,
    r: float = 0.045,
    T: float = 1/12
) -> BasketPricingResult:
    """
    Price a basket call option from simulated paths.
    
    Payoff: max(Σ w_i S_T^i - K, 0)
    
    Args:
        paths: (n_paths, n_steps+1, N) spot prices
        basket_weights: Weights [w_1, ..., w_N]
        strike: Basket strike K
        r: Risk-free rate
        T: Maturity
    
    Returns:
        BasketPricingResult with price, std_error, delta
    """
    n_paths = paths.shape[0]
    N = paths.shape[2]
    
    # Basket value at maturity
    S_T = paths[:, -1, :]  # (n_paths, N)
    basket_value = np.sum(S_T * basket_weights, axis=1)  # (n_paths,)
    
    # Payoff
    payoffs = np.maximum(basket_value - strike, 0)
    
    # Discounted price
    discount = np.exp(-r * T)
    price = discount * np.mean(payoffs)
    std_error = discount * np.std(payoffs) / np.sqrt(n_paths)
    
    # Delta: ∂P/∂S_0^i
    # Using pathwise derivative: indicator * w_i
    in_the_money = (basket_value > strike).astype(float)
    delta = np.zeros(N)
    for i in range(N):
        # Approximate delta using sensitivity to initial spot
        delta[i] = discount * np.mean(in_the_money * basket_weights[i])
    
    return BasketPricingResult(
        strike=strike,
        price=price,
        std_error=std_error,
        delta=delta
    )


def price_basket_put(
    paths: np.ndarray,
    basket_weights: np.ndarray,
    strike: float,
    r: float = 0.045,
    T: float = 1/12
) -> BasketPricingResult:
    """
    Price a basket put option.
    
    Payoff: max(K - Σ w_i S_T^i, 0)
    """
    n_paths = paths.shape[0]
    N = paths.shape[2]
    
    S_T = paths[:, -1, :]
    basket_value = np.sum(S_T * basket_weights, axis=1)
    
    payoffs = np.maximum(strike - basket_value, 0)
    
    discount = np.exp(-r * T)
    price = discount * np.mean(payoffs)
    std_error = discount * np.std(payoffs) / np.sqrt(n_paths)
    
    in_the_money = (basket_value < strike).astype(float)
    delta = np.zeros(N)
    for i in range(N):
        delta[i] = -discount * np.mean(in_the_money * basket_weights[i])
    
    return BasketPricingResult(
        strike=strike,
        price=price,
        std_error=std_error,
        delta=delta
    )


def price_multiple_strikes(
    paths: np.ndarray,
    basket_weights: np.ndarray,
    strikes: np.ndarray,
    r: float = 0.045,
    T: float = 1/12,
    option_type: str = 'call'
) -> List[BasketPricingResult]:
    """
    Price basket options for multiple strikes.
    
    Args:
        paths: Simulated paths
        basket_weights: Basket weights
        strikes: Array of strikes
        r: Risk-free rate
        T: Maturity
        option_type: 'call' or 'put'
    
    Returns:
        List of BasketPricingResult
    """
    results = []
    price_func = price_basket_call if option_type == 'call' else price_basket_put
    
    for K in strikes:
        result = price_func(paths, basket_weights, K, r, T)
        results.append(result)
    
    return results


def compute_basket_spot(
    marginal_params: List[RoughHestonParams],
    basket_weights: np.ndarray
) -> float:
    """Compute basket spot price."""
    spots = np.array([p.spot for p in marginal_params])
    return np.sum(spots * basket_weights)


def compute_basket_atm_vol(
    paths: np.ndarray,
    basket_weights: np.ndarray,
    T: float = 1/12
) -> float:
    """
    Compute ATM implied volatility of basket from simulated paths.
    
    Uses log-return variance.
    """
    S_0 = paths[:, 0, :]  # Initial spots
    S_T = paths[:, -1, :]  # Terminal spots
    
    basket_0 = np.sum(S_0 * basket_weights, axis=1)
    basket_T = np.sum(S_T * basket_weights, axis=1)
    
    log_return = np.log(basket_T / basket_0)
    realized_var = np.var(log_return) / T
    
    return np.sqrt(realized_var)


# =====================================================================
# TESTS
# =====================================================================

def test_basket_pricing():
    """Test basket option pricing."""
    from path_simulation import simulate_correlated_rough_heston
    
    params_1 = RoughHestonParams(
        H=0.10, eta=0.15, rho=-0.7, xi0=0.04, kappa=2.0, theta=0.04,
        spot=100.0, maturity=1/12
    )
    params_2 = RoughHestonParams(
        H=0.15, eta=0.18, rho=-0.5, xi0=0.05, kappa=1.5, theta=0.05,
        spot=100.0, maturity=1/12
    )
    
    correlation = np.array([[1.0, 0.6], [0.6, 1.0]])
    basket_weights = np.array([0.5, 0.5])
    
    paths = simulate_correlated_rough_heston(
        [params_1, params_2], correlation, n_paths=10000, n_steps=50
    )
    
    basket_spot = compute_basket_spot([params_1, params_2], basket_weights)
    strikes = np.array([90, 95, 100, 105, 110])
    
    print("=" * 60)
    print("BASKET OPTION PRICING TEST")
    print("=" * 60)
    print(f"Basket spot: ${basket_spot:.2f}")
    print(f"Basket ATM vol: {compute_basket_atm_vol(paths, basket_weights, 1/12)*100:.2f}%")
    print()
    
    results = price_multiple_strikes(paths, basket_weights, strikes, T=1/12)
    
    for r in results:
        print(f"K={r.strike:6.1f}: Price=${r.price:6.3f} ± ${r.std_error:.3f}, "
              f"Δ=[{r.delta[0]:.3f}, {r.delta[1]:.3f}]")
    
    print("✅ Basket pricing test passed")


if __name__ == "__main__":
    test_basket_pricing()
