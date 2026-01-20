
import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.simulation.rough_heston import RoughHestonParams, RoughHestonSimulator
from src.sensitivity.malliavin import MalliavinEngine, compute_malliavin_weights_H

def test_malliavin_weights_derivative():
    """Test finite difference derivative of weights"""
    H = 0.1
    dH = 1e-4
    n_steps = 50
    dt = 0.01
    
    weights_H, dweights_dH = compute_malliavin_weights_H(n_steps, H, dt, dH)
    
    # Check shape
    assert weights_H.shape == (n_steps, n_steps)
    assert dweights_dH.shape == (n_steps, n_steps)
    
    # Check values
    # Manual FD check
    from src.simulation.rough_heston import compute_weights
    w1 = compute_weights(n_steps, H, dt)
    w2 = compute_weights(n_steps, H + dH, dt)
    fd = (w2 - w1) / dH
    
    # Tolerances
    assert np.allclose(dweights_dH, fd, atol=1e-8)

def test_sensitivity_computation():
    """Test dC/dH computation"""
    params = RoughHestonParams(
        H=0.1, eta=1.9, rho=-0.7, xi0=0.04, 
        kappa=1.5, theta=0.04, S0=100.0, r=0.0
    )
    sim = RoughHestonSimulator(params)
    engine = MalliavinEngine(sim)
    
    strikes = np.array([90.0, 100.0, 110.0])
    T = 0.5 # Longer maturity to see H effect
    
    greeks = engine.compute_greeks(strikes, T, n_paths=20000, n_steps=50)
    
    # Check output keys
    assert 'delta' in greeks
    assert 'dC_dH' in greeks
    
    # Check Delta values (should be roughly N(d1))
    # ATM Delta ~ 0.5. With rough vol eta=1.9, can be higher.
    assert 0.3 < greeks['delta'][1] < 0.85
    
    # Check dC/dH
    # Rough volatility (low H) usually creates steeper smile.
    # Increasing H -> Smoother smile -> Lower ATM vol for same params? Or different?
    # Usually sensitivity is non-zero.
    print(f"dC/dH: {greeks['dC_dH']}")
    
    # Just check it runs and produces finite values for now
    assert np.all(np.isfinite(greeks['dC_dH']))
    assert not np.all(greeks['dC_dH'] == 0)

def test_delta_correctness():
    """Verify Delta against finite difference"""
    params = RoughHestonParams(
        H=0.1, eta=0.5, rho=-0.5, xi0=0.04,
        kappa=1.0, theta=0.04, S0=100.0, r=0.0
    )
    sim = RoughHestonSimulator(params)
    engine = MalliavinEngine(sim)
    
    strikes = np.array([100.0])
    T = 0.5
    
    # Malliavin / Pathwise Delta
    greeks = engine.compute_greeks(strikes, T, n_paths=50000, n_steps=50)
    delta_malliavin = greeks['delta'][0]
    
    # Finite Difference Delta
    dS = 1.0
    sim_plus = RoughHestonSimulator(RoughHestonParams(
        H=0.1, eta=0.5, rho=-0.5, xi0=0.04,
        kappa=1.0, theta=0.04, S0=100.0 + dS, r=0.0
    ))
    sim_minus = RoughHestonSimulator(RoughHestonParams(
        H=0.1, eta=0.5, rho=-0.5, xi0=0.04,
        kappa=1.0, theta=0.04, S0=100.0 - dS, r=0.0
    ))
    
    # Use same seed for better FD convergence?
    # The simulator generates seed internally if not provided.
    # I should expose seed in simulator.simulate or set np.random.seed before.
    # But simulate sets seed if provided.
    
    price_plus = sim_plus.price_european_call(100.0, T, n_paths=50000, n_steps=50)['price']
    price_minus = sim_minus.price_european_call(100.0, T, n_paths=50000, n_steps=50)['price']
    
    delta_fd = (price_plus - price_minus) / (2 * dS)
    
    print(f"Delta Malliavin: {delta_malliavin}, Delta FD: {delta_fd}")
    assert np.abs(delta_malliavin - delta_fd) < 0.05
