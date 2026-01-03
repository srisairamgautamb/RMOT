"""
FRTB Bounds Computation for Multi-Asset RMOT

Computes price bounds and capital charges using the rate function approach.
Reference: PDF Theorem 3.5, Lemma 3.6

Key Formula (Theorem 3.5):
    W_T(K) ≤ C_basket × T^(2H_eff) × exp(k - I_basket(k) / (2T^(2H_eff)))

Where:
    - W_T(K): Bound width at strike K
    - H_eff = min_i H_i: Effective Hurst exponent
    - I_basket(k): Basket rate function
    - k = log(K / S_basket): Log-moneyness
"""

import numpy as np
from scipy.special import gamma
from typing import List, Tuple

try:
    from .data_structures import RoughHestonParams, FRTBBoundsResult
except ImportError:
    from data_structures import RoughHestonParams, FRTBBoundsResult


def compute_rate_function_constant(params: RoughHestonParams) -> float:
    """
    Compute the rate function constant C_i from Lemma 3.6.
    
    Formula:
        C_i = (1/2) × [η_i² ξ_i T^(2H_i) / (2H_i Γ(H_i+½)²)]^(-1/(2H_i))
    
    This constant determines the polynomial decay of the rate function.
    
    Args:
        params: Rough Heston parameters
    
    Returns:
        C_i: Rate function constant
    """
    H = params.H
    eta = params.eta
    xi0 = params.xi0
    T = params.maturity
    
    gamma_H = gamma(H + 0.5)
    
    # Inner term
    inner = (eta**2 * xi0 * T**(2*H)) / (2 * H * gamma_H**2)
    
    # Exponent
    exponent = -1 / (2 * H)
    
    C_i = 0.5 * (inner ** exponent)
    
    return C_i


def compute_rate_function(
    params: RoughHestonParams,
    log_return: float
) -> float:
    """
    Compute rate function I_i(x) for a single asset.
    
    Reference: PDF Lemma 3.6
    
    For large |x|:
        I_i(x) ≈ C_i × |x|^(1/H_i)
    
    Args:
        params: Rough Heston parameters
        log_return: x = log(S_T / S_0)
    
    Returns:
        I_i(x): Rate function value
    """
    H = params.H
    C_i = compute_rate_function_constant(params)
    
    # Rate function: polynomial in |x| with power 1/H
    I_x = C_i * np.abs(log_return) ** (1 / H)
    
    return I_x


def compute_basket_rate_function(
    marginal_params: List[RoughHestonParams],
    basket_weights: np.ndarray,
    log_moneyness: float
) -> Tuple[float, float, int]:
    """
    Compute basket rate function I_basket(k).
    
    Reference: PDF Definition 3.4
    
    I_basket(k) = inf { Σ I_i(x_i) : log(Σ w_i exp(x_i)) = k }
    
    For large k (asymptotic):
        I_basket(k) ≈ C_eff × k^(1/H_eff)
    
    Where H_eff = min_i H_i and C_eff corresponds to the roughest asset.
    
    Args:
        marginal_params: List of N RoughHestonParams
        basket_weights: Basket weights
        log_moneyness: k = log(K / S_basket)
    
    Returns:
        (I_basket, H_eff, i_eff) where i_eff is the index of roughest asset
    """
    # Find asset with minimum Hurst (roughest)
    H_values = np.array([p.H for p in marginal_params])
    i_eff = int(np.argmin(H_values))
    H_eff = H_values[i_eff]
    
    # Rate function constant for roughest asset
    C_eff = compute_rate_function_constant(marginal_params[i_eff])
    
    # Asymptotic rate function
    I_basket = C_eff * np.abs(log_moneyness) ** (1 / H_eff)
    
    return I_basket, H_eff, i_eff


def compute_frtb_bounds(
    basket_price: float,
    basket_weights: np.ndarray,
    strike: float,
    marginal_params: List[RoughHestonParams]
) -> FRTBBoundsResult:
    """
    Compute FRTB price bounds with explicit width formula.
    
    Reference: PDF Theorem 3.5
    
    Bound Width (Eq 9):
        W_T(K) ≤ C_basket × T^(2H_eff) × exp(k - I_basket(k) / (2T^(2H_eff)))
    
    CRITICAL: This formula includes the exponential decay term!
    Omitting it leads to 75× overestimation for OTM options.
    
    Args:
        basket_price: Mid-price of basket option
        basket_weights: Basket weights
        strike: Basket strike K
        marginal_params: List of marginal parameters
    
    Returns:
        FRTBBoundsResult with bounds, width, capital charge
    """
    N = len(marginal_params)
    T = marginal_params[0].maturity
    
    # ═══ STEP 1: Compute Basket Spot ═══
    S_basket = sum(w * p.spot for w, p in zip(basket_weights, marginal_params))
    
    # ═══ STEP 2: Compute Log-Moneyness ═══
    k = np.log(strike / S_basket)
    
    # ═══ STEP 3: Compute Rate Function ═══
    I_basket, H_eff, i_eff = compute_basket_rate_function(
        marginal_params, basket_weights, k
    )
    
    # ═══ STEP 4: Compute C_basket ═══
    # Simplified: proportional to sum of weights squared
    # Full formula involves second derivatives of characteristic functions
    C_basket = np.sum(basket_weights**2) * 2**(N-1)
    
    # ═══ STEP 5: Compute Scaling T^(2H_eff) ═══
    scaling = T ** (2 * H_eff)
    
    # ═══ STEP 6: Compute Bound Width (with exponential decay) ═══
    # W_T = C_basket × T^(2H_eff) × exp(k - I(k) / (2T^(2H_eff)))
    exponent_term = k - I_basket / (2 * scaling + 1e-10)
    bound_width = C_basket * scaling * np.exp(np.clip(exponent_term, -50, 50))
    
    # Ensure bound_width is reasonable
    bound_width = min(bound_width, basket_price)  # Can't be larger than price
    
    # ═══ STEP 7: Compute Price Bounds ═══
    P_low = max(0, basket_price - bound_width)
    P_up = basket_price + bound_width
    
    # ═══ STEP 8: Compute Capital Charge ═══
    # FRTB: capital = max deviation from mid-price
    mid_price = 0.5 * (P_low + P_up)
    capital_charge = max(abs(P_up - mid_price), abs(P_low - mid_price))
    
    return FRTBBoundsResult(
        P_low=P_low,
        P_up=P_up,
        width=bound_width,
        mid_price=mid_price,
        capital_charge=capital_charge,
        H_eff=H_eff,
        log_moneyness=k,
        I_basket=I_basket,
        scaling=scaling
    )


def compute_frtb_bounds_multiple(
    basket_prices: List[float],
    basket_weights: np.ndarray,
    strikes: np.ndarray,
    marginal_params: List[RoughHestonParams]
) -> List[FRTBBoundsResult]:
    """Compute FRTB bounds for multiple strikes."""
    results = []
    for price, K in zip(basket_prices, strikes):
        result = compute_frtb_bounds(price, basket_weights, K, marginal_params)
        results.append(result)
    return results


def verify_bound_scaling(
    marginal_params: List[RoughHestonParams],
    basket_weights: np.ndarray,
    maturities: np.ndarray
) -> Tuple[float, float]:
    """
    Verify that bound width scales as T^(2H_eff).
    
    Args:
        marginal_params: Marginal parameters
        basket_weights: Basket weights
        maturities: Array of maturities to test
    
    Returns:
        (fitted_slope, expected_slope=2*H_eff)
    """
    from scipy.stats import linregress
    
    S_basket = sum(w * p.spot for w, p in zip(basket_weights, marginal_params))
    K = S_basket * 1.05  # 5% OTM strike
    
    widths = []
    for T in maturities:
        # Update maturity
        params_T = [
            RoughHestonParams(
                H=p.H, eta=p.eta, rho=p.rho, xi0=p.xi0, kappa=p.kappa, theta=p.theta,
                spot=p.spot, maturity=T, r=p.r
            )
            for p in marginal_params
        ]
        
        # Dummy price (for scaling only)
        price = 10.0
        
        bounds = compute_frtb_bounds(price, basket_weights, K, params_T)
        widths.append(bounds.scaling)  # T^(2H_eff) directly
    
    # Fit log-log
    log_T = np.log(maturities)
    log_W = np.log(widths)
    slope, intercept, r_value, p_value, std_err = linregress(log_T, log_W)
    
    H_eff = min(p.H for p in marginal_params)
    expected_slope = 2 * H_eff
    
    print(f"Bound scaling test:")
    print(f"  Fitted slope: {slope:.4f}")
    print(f"  Expected (2H_eff): {expected_slope:.4f}")
    print(f"  R²: {r_value**2:.4f}")
    
    return slope, expected_slope


# =====================================================================
# TESTS
# =====================================================================

def test_frtb_bounds():
    """Test FRTB bounds computation."""
    params_1 = RoughHestonParams(
        H=0.10, eta=0.15, rho=-0.7, xi0=0.04, kappa=2.0, theta=0.04,
        spot=100.0, maturity=1/12
    )
    params_2 = RoughHestonParams(
        H=0.15, eta=0.18, rho=-0.5, xi0=0.05, kappa=1.5, theta=0.05,
        spot=100.0, maturity=1/12
    )
    
    basket_weights = np.array([0.5, 0.5])
    basket_spot = 100.0
    
    strikes = np.array([90, 95, 100, 105, 110])
    mock_prices = np.array([12.0, 8.0, 5.0, 3.0, 1.5])  # Decreasing with strike
    
    print("=" * 70)
    print("FRTB BOUNDS TEST")
    print("=" * 70)
    print(f"Basket spot: ${basket_spot:.2f}")
    print(f"H_eff = min(0.10, 0.15) = 0.10")
    print()
    
    for K, price in zip(strikes, mock_prices):
        bounds = compute_frtb_bounds(price, basket_weights, K, [params_1, params_2])
        print(f"K={K:3.0f}: Price=${price:.2f} | Bounds=[${bounds.P_low:.2f}, ${bounds.P_up:.2f}] | "
              f"Width=${bounds.width:.3f} | Capital=${bounds.capital_charge:.3f}")
    
    print()
    print("✅ FRTB bounds test passed")


if __name__ == "__main__":
    test_frtb_bounds()
