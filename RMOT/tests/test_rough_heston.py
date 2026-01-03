
import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.simulation.rough_heston import RoughHestonParams, RoughHestonSimulator, compute_weights

def test_weight_computation():
    """Test Riemann-Liouville weights sum correctly"""
    H = 0.1
    n_steps = 100
    dt = 0.01
    
    weights = compute_weights(n_steps, H, dt)
    
    # Weights should be positive
    assert np.all(weights >= 0)
    
    # Diagonal should be largest (most recent impact) usually
    # For H < 0.5 (rough), the kernel is singular at 0, so close memory is stronger.
    for i in range(10, n_steps):
        assert weights[i, i] >= weights[i, i-1]

def test_simulation_convergence():
    """Test Monte Carlo convergence for ATM call"""
    params = RoughHestonParams(
        H=0.1, eta=1.9, rho=-0.7, xi0=0.04, 
        kappa=1.5, theta=0.04, S0=100.0, r=0.0
    )
    
    sim = RoughHestonSimulator(params)
    
    # ATM call with T=1/12
    # This is a very short tenor.
    result = sim.price_european_call(K=100.0, T=1/12, n_paths=100000, n_steps=50)
    
    # ATM call roughly sigma * sqrt(T) * 0.4
    # sigma approx 0.2 (sqrt(0.04))
    # price ~ 100 * 0.2 * sqrt(1/12) * 0.4 ~ 100 * 0.2 * 0.288 * 0.4 ~ 2.3
    # Spec says "Assertion: 1.5 < price < 3.0"
    
    print(f"Computed Price: {result['price']}")
    assert 1.5 < result['price'] < 3.0
    assert result['stderr'] < 0.02  # Precision check

def test_parameter_validation():
    """Test parameter validation catches errors"""
    
    with pytest.raises(AssertionError, match="H="):
        RoughHestonParams(H=0.6, eta=1.0, rho=0.0, xi0=0.04, 
                         kappa=1.0, theta=0.04, S0=100.0, r=0.0)
    
    with pytest.raises(AssertionError, match="rho="):
        RoughHestonParams(H=0.1, eta=1.0, rho=1.5, xi0=0.04,
                         kappa=1.0, theta=0.04, S0=100.0, r=0.0)

def test_martingale_property():
    """Verify S_t is a martingale (mean should be close to S0)"""
    params = RoughHestonParams(
        H=0.1, eta=0.5, rho=-0.5, xi0=0.04,
        kappa=1.0, theta=0.04, S0=100.0, r=0.0
    )
    sim = RoughHestonSimulator(params)
    S_paths = sim.simulate(T=1.0, n_steps=100, n_paths=50000)
    
    S_T_mean = np.mean(S_paths[:, -1])
    # With r=0, E[S_T] = S0
    assert np.abs(S_T_mean - 100.0) < 0.5  # 0.5% error tolerance
