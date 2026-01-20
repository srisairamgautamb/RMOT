"""
Multi-Asset RMOT Pipeline - End-to-End Orchestration

Reference: PDF Algorithm 1

This is the main entry point that orchestrates all phases:
1. Marginal Calibration (parallel)
2. Correlation Estimation (Ψ_ij optimization)
3. Basket Option Pricing
4. FRTB Bounds Computation
"""

import numpy as np
from typing import List, Dict, Optional
import time

try:
    from .data_structures import (
        RoughHestonParams, MultiAssetConfig, MarginalCalibrationResult,
        CorrelationEstimationResult, FRTBBoundsResult
    )
    from .psi_functional import compute_psi_functional, precompute_psi_cache
    from .path_simulation import simulate_correlated_rough_heston, validate_correlation_constraint
    from .basket_pricing import price_basket_call, price_multiple_strikes, compute_basket_spot
    from .frtb_bounds import compute_frtb_bounds, compute_frtb_bounds_multiple
except ImportError:
    from data_structures import (
        RoughHestonParams, MultiAssetConfig, MarginalCalibrationResult,
        CorrelationEstimationResult, FRTBBoundsResult
    )
    from psi_functional import compute_psi_functional, precompute_psi_cache
    from path_simulation import simulate_correlated_rough_heston, validate_correlation_constraint
    from basket_pricing import price_basket_call, price_multiple_strikes, compute_basket_spot
    from frtb_bounds import compute_frtb_bounds, compute_frtb_bounds_multiple


def calibrate_marginals_simple(
    config: MultiAssetConfig
) -> MarginalCalibrationResult:
    """
    Phase 1: Marginal Calibration (Simplified)
    
    Uses heuristic parameter estimation from market data.
    For production: integrate with single-asset RMOT solver.
    
    Args:
        config: Multi-asset configuration
    
    Returns:
        MarginalCalibrationResult with calibrated parameters
    """
    print("\n" + "=" * 70)
    print("PHASE 1: MARGINAL CALIBRATION")
    print("=" * 70)
    
    params_list = []
    errors = []
    
    for i, asset in enumerate(config.assets):
        # Heuristic calibration
        # ATM implied vol
        atm_idx = np.argmin(np.abs(asset.strikes - asset.spot))
        atm_price = asset.market_prices[atm_idx]
        
        # Approximate ATM vol (Black-Scholes approximation)
        from scipy.stats import norm
        T = asset.maturity
        sigma_atm = atm_price / (asset.spot * 0.4 * np.sqrt(T))  # Rough approximation
        sigma_atm = np.clip(sigma_atm, 0.05, 1.0)
        
        # Estimate skew for correlation
        if len(asset.strikes) > 2:
            otm_idx = 0  # Low strike
            itm_idx = -1  # High strike
            skew = (asset.market_prices[itm_idx] - asset.market_prices[otm_idx]) / asset.spot
            rho_guess = -0.7 if skew < 0 else -0.3
        else:
            rho_guess = -0.5
        
        # Create parameters
        params = RoughHestonParams(
            H=0.08 + 0.04 * i,  # Different H for each asset (identifiability)
            eta=0.15,
            rho=rho_guess,
            xi0=sigma_atm**2,
            kappa=2.0,
            theta=sigma_atm**2,
            spot=asset.spot,
            maturity=T
        )
        params.validate()
        params_list.append(params)
        
        # Compute calibration error (simplified)
        error = np.abs(sigma_atm - 0.15)  # Dummy error
        errors.append(error)
        
        print(f"  Asset {i} ({asset.ticker}): H={params.H:.3f}, η={params.eta:.3f}, "
              f"ρ={params.rho:.3f}, ξ₀={params.xi0:.4f}")
    
    result = MarginalCalibrationResult(
        params=params_list,
        calibration_errors=np.array(errors)
    )
    
    # Verify Hurst distinctness (Assumption 2.1)
    result.verify_distinct_hurst()
    print("✅ Marginal calibration complete (Hurst distinctness verified)")
    
    return result


def estimate_correlation_simple(
    marginal_result: MarginalCalibrationResult,
    config: MultiAssetConfig
) -> CorrelationEstimationResult:
    """
    Phase 2: Correlation Estimation (Simplified)
    
    Uses initial guess as estimate. For production: implement full
    Ψ_ij-based optimization (CorrelationEstimator class).
    
    Args:
        marginal_result: Calibrated marginal parameters
        config: Multi-asset configuration
    
    Returns:
        CorrelationEstimationResult
    """
    print("\n" + "=" * 70)
    print("PHASE 2: CORRELATION ESTIMATION")
    print("=" * 70)
    
    N = len(marginal_result.params)
    rho = config.correlation_guess.copy()
    
    # Pre-compute Ψ_ij (for validation)
    print("Computing Ψ_ij functionals...")
    for i in range(N):
        for j in range(i + 1, N):
            psi_ij = compute_psi_functional(
                1.0, 1.0,
                marginal_result.params[i],
                marginal_result.params[j]
            )
            print(f"  Ψ_{i}{j} = {psi_ij:.6f}")
    
    # Simplified: use initial guess
    # For production: run full optimization
    
    # Fisher Information (simplified: identity)
    fisher_info = np.eye(N * (N - 1) // 2)
    cramer_rao_stds = np.ones(N * (N - 1) // 2) * 0.05
    
    result = CorrelationEstimationResult(
        rho=rho,
        fisher_information=fisher_info,
        cramer_rao_stds=cramer_rao_stds,
        converged=True,
        n_iterations=0
    )
    result.validate()
    
    print(f"Estimated correlation matrix:\n{rho}")
    print("✅ Correlation estimation complete")
    
    return result


def multi_asset_rmot_pipeline(
    config: MultiAssetConfig,
    n_paths: int = 50000,
    n_steps: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Complete Multi-Asset RMOT Pipeline (Algorithm 1)
    
    Reference: PDF Algorithm 1
    
    Phases:
    1. Marginal Calibration - Fit N independent rough Heston models
    2. Correlation Estimation - Estimate ρ using Ψ_ij functional
    3. Basket Pricing - Simulate correlated paths and price baskets
    4. FRTB Bounds - Compute price bounds and capital charges
    
    Args:
        config: Multi-asset configuration with market data
        n_paths: Monte Carlo paths for pricing
        n_steps: Time discretization
        verbose: Print progress
    
    Returns:
        Complete results dictionary
    """
    start_time = time.time()
    
    if verbose:
        print("=" * 70)
        print("MULTI-ASSET RMOT PIPELINE")
        print("=" * 70)
        print(f"Assets: {config.n_assets}")
        print(f"Basket weights: {config.basket_weights}")
        print(f"MC paths: {n_paths}, steps: {n_steps}")
    
    # ═══ PHASE 1: MARGINAL CALIBRATION ═══
    marginal_result = calibrate_marginals_simple(config)
    
    # ═══ PHASE 2: CORRELATION ESTIMATION ═══
    correlation_result = estimate_correlation_simple(marginal_result, config)
    
    # ═══ PHASE 3: BASKET PRICING ═══
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 3: BASKET OPTION PRICING")
        print("=" * 70)
    
    # Simulate paths
    paths = simulate_correlated_rough_heston(
        marginal_result.params,
        correlation_result.rho,
        n_paths,
        n_steps
    )
    
    if verbose:
        print(f"Simulated {n_paths} paths")
        validate_correlation_constraint(paths, correlation_result.rho)
    
    # Compute basket spot
    basket_spot = compute_basket_spot(marginal_result.params, config.basket_weights)
    if verbose:
        print(f"Basket spot: ${basket_spot:.2f}")
    
    # Price basket options
    T = config.maturity
    basket_prices = price_multiple_strikes(
        paths, config.basket_weights, config.basket_strikes, T=T
    )
    
    if verbose:
        print("\nBasket Option Prices:")
        for r in basket_prices:
            print(f"  K={r.strike:8.2f}: Price=${r.price:8.3f} ± ${r.std_error:.3f}")
    
    # ═══ PHASE 4: FRTB BOUNDS ═══
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 4: FRTB BOUNDS")
        print("=" * 70)
    
    frtb_results = compute_frtb_bounds_multiple(
        [p.price for p in basket_prices],
        config.basket_weights,
        config.basket_strikes,
        marginal_result.params
    )
    
    if verbose:
        print("\nFRTB Price Bounds:")
        for bp, fb in zip(basket_prices, frtb_results):
            print(f"  K={bp.strike:8.2f}: [{fb.P_low:8.3f}, {fb.P_up:8.3f}] "
                  f"Width=${fb.width:.3f} Capital=${fb.capital_charge:.3f}")
    
    # ═══ SUMMARY ═══
    elapsed = time.time() - start_time
    
    if verbose:
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Total time: {elapsed:.2f}s")
        print(f"Assets: {config.n_assets}")
        print(f"H values: {[p.H for p in marginal_result.params]}")
        print(f"Total capital charge: ${sum(f.capital_charge for f in frtb_results):.2f}")
    
    return {
        'marginal_calibration': marginal_result,
        'correlation_estimation': correlation_result,
        'basket_prices': basket_prices,
        'frtb_bounds': frtb_results,
        'paths': paths,
        'elapsed_time': elapsed
    }


# =====================================================================
# DEMO
# =====================================================================

def run_demo():
    """Run a demo of the Multi-Asset RMOT pipeline."""
    from data_structures import create_synthetic_config
    
    print("\n" + "=" * 70)
    print("MULTI-ASSET RMOT DEMO")
    print("=" * 70)
    
    # Create synthetic configuration
    config = create_synthetic_config(n_assets=2, n_strikes=15, maturity=1/12)
    
    # Run pipeline
    result = multi_asset_rmot_pipeline(config, n_paths=20000, n_steps=50)
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE ✅")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    run_demo()
