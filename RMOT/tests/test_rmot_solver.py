
import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.simulation.rough_heston import RoughHestonParams, RoughHestonSimulator
from src.pricing.rmot_solver import RMOTPricingEngine, price_with_rmot

def test_rmot_calibration_convergence():
    """Test that RMOT calibrates to liquid prices"""
    # Ground truth model
    params = RoughHestonParams(
        H=0.1, eta=1.0, rho=-0.5, xi0=0.04, 
        kappa=1.0, theta=0.04, S0=100.0, r=0.0
    )
    sim = RoughHestonSimulator(params)
    
    # Generate "Market Prices" using MC
    T = 0.5
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    market_prices = []
    for K in strikes:
        res = sim.price_european_call(K, T, n_paths=50000)
        market_prices.append(res['price'])
    market_prices = np.array(market_prices)
    
    # RMOT Engine
    # Use same simulator as prior (Perfect Model Match case)
    # Ideally should calibrate perfectly with lambda -> 0.
    # With lambda=0.01, it should be close.
    
    rmot = RMOTPricingEngine(sim, lambda_reg=0.01) # Reasonable regularization
    
    # Use fewer samples for speed in test
    result = rmot.solve_dual_rmot(strikes, market_prices, T, n_samples=20000)
    
    assert result['optimization_success']
    calib = result['calibration_error']
    print(f"Calibration Error: {calib}")
    
    # Tolerances
    assert calib['max_abs_error'] < 0.05
    assert calib['rmse'] < 0.05

def test_price_with_rmot_wrapper():
    """Test high-level API"""
    params = RoughHestonParams(
        H=0.1, eta=1.0, rho=-0.5, xi0=0.04, 
        kappa=1.0, theta=0.04, S0=100.0, r=0.0
    )
    
    liquid_strikes = np.array([95.0, 100.0, 105.0])
    liquid_prices = np.array([8.0, 5.0, 3.0]) # Dummy prices approx
    target_strike = 120.0
    T = 0.5
    
    # Use smaller n_samples for test
    start_res = price_with_rmot(
        liquid_strikes, liquid_prices, target_strike, T, params,
        lambda_reg=0.01, n_samples=10000
    )
    
    assert 'price' in start_res
    assert 'error_bound' in start_res
    assert start_res['price'] >= 0

def test_error_bound_monotonicity():
    """Verify error bound decreases with strike distance (or behaves as expected)"""
    params = RoughHestonParams(
        H=0.1, eta=1.0, rho=-0.5, xi0=0.04, 
        kappa=1.0, theta=0.04, S0=100.0, r=0.0
    )
    sim = RoughHestonSimulator(params)
    rmot = RMOTPricingEngine(sim)
    
    # dummy multipliers
    multipliers = np.zeros(2) 
    
    # K=300 vs K=350 (Deep OTM to see decay)
    b1 = rmot.compute_error_bound(300.0, 0.5, multipliers)
    b2 = rmot.compute_error_bound(350.0, 0.5, multipliers)
    
    # Bound should decrease as K increases (further OTM)?
    # Bound term `S0 exp(k) exp(-I(k))`.
    # k = log(K/S0).
    # I(k) ~ k^(1/H) ~ k^10.
    # exp(k - c*k^10). Dominant term is -k^10.
    # So bound should decrease rapidly.
    
    print(f"Bound K=120: {b1['bound']}")
    print(f"Bound K=130: {b2['bound']}")
    
    assert b2['bound'] < b1['bound']
