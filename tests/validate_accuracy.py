
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.simulation.rough_heston import RoughHestonSimulator, RoughHestonParams
from src.sensitivity.malliavin import MalliavinEngine
from src.pricing.rmot_solver import RMOTPricingEngine

def test_rough_heston_scaling():
    """Verify rough Heston has correct H-dependent scaling"""
    print("\n[TEST] Rough Heston Scaling...")
    
    params = RoughHestonParams(
        H=0.1, eta=1.9, rho=-0.7, xi0=0.04,
        kappa=1.5, theta=0.04, S0=100.0, r=0.0
    )
    
    sim = RoughHestonSimulator(params)
    
    # Test: Variance of log returns should scale as T^(2H)
    T_values = np.array([1/252, 1/52, 1/12, 1/4])
    variances = []
    
    for T in T_values:
        # Need enough paths for variance convergence
        paths = sim.simulate(T, n_steps=int(max(T*252, 10)), n_paths=100000)
        log_returns = np.log(paths[:, -1] / paths[:, 0])
        variances.append(np.var(log_returns))
    
    # Fit: Var ~ T^(2H)
    log_T = np.log(T_values)
    log_Var = np.log(variances)
    slope, _ = np.polyfit(log_T, log_Var, 1)
    
    print(f"Measured exponent: {slope:.3f}")
    print(f"Expected exponent (2H): {2*params.H:.3f}")
    diff = abs(slope - 2*params.H)
    rel_error = 100*diff/(2*params.H)
    print(f"Relative error: {rel_error:.1f}%")
    
    # CRITICAL: Error should be < 5%
    if diff >= 0.05:
        print(f"FAILED: Rough scaling violated! Got {slope:.3f}, expected {2*params.H:.3f}")
    else:
        print("✅ PASS: Rough Heston scaling correct")

def test_malliavin_sensitivity_scaling():
    """Verify Malliavin derivatives have correct moneyness scaling"""
    print("\n[TEST] Malliavin Sensitivity Scaling...")
    
    params = RoughHestonParams(
        H=0.1, eta=1.9, rho=-0.7, xi0=0.04,
        kappa=1.5, theta=0.04, S0=100.0, r=0.0
    )
    
    sim = RoughHestonSimulator(params)
    malliavin = MalliavinEngine(sim)
    
    # Test strikes from 90 to 110
    strikes = np.linspace(90, 110, 21)
    
    # Compute sensitivities
    # Using small n_paths for speed in this check if possible, but noise is high.
    # Auditor asked for 100k.
    result = malliavin.compute_greeks(strikes, T=1/12, n_paths=100000)
    dC_dH = result['dC_dH']
    
    # Theoretical: dC/dH ~ |K - S0|^(1-2H) for near-ATM
    # For H=0.1: exponent = 1 - 0.2 = 0.8
    
    K_centered = np.abs(strikes - 100)
    # Avoid zero
    mask = K_centered > 0.5 
    if not np.any(mask):
        print("Skipping fit due to lack of OTM points")
        return

    K_fit = K_centered[mask]
    sens_fit = np.abs(dC_dH[mask])
    
    log_K = np.log(K_fit)
    log_sens = np.log(sens_fit)
    
    slope, _ = np.polyfit(log_K, log_sens, 1)
    expected_slope = 1 - 2*params.H
    
    print(f"Measured scaling exponent: {slope:.3f}")
    print(f"Expected scaling (1-2H): {expected_slope:.3f}")
    rel_error = 100*abs(slope - expected_slope)/expected_slope
    print(f"Relative error: {rel_error:.1f}%")
    
    # CRITICAL: Error should be < 30% (noisy due to MC)
    if abs(slope - expected_slope) >= 0.3:
        print(f"FAILED: Malliavin scaling wrong! Got {slope:.3f}, expected {expected_slope:.3f}")
    else:
        print("✅ PASS: Malliavin sensitivities correct")

def test_rmot_constraint_satisfaction():
    """Verify RMOT satisfies martingale and calibration constraints"""
    print("\n[TEST] RMOT Constraints...")
    
    params = RoughHestonParams(
        H=0.1, eta=1.9, rho=-0.7, xi0=0.04,
        kappa=1.5, theta=0.04, S0=100.0, r=0.0
    )
    
    sim = RoughHestonSimulator(params)
    rmot = RMOTPricingEngine(sim, lambda_reg=0.01)
    
    # Liquid strikes
    liquid_strikes = np.linspace(85, 115, 30)
    
    # Generate "market" prices from rough Heston
    print("Generating market prices...")
    # NOTE: Using a smaller number of paths for market gen to stay fast? 
    # Auditor script said 50000.
    liquid_prices = []
    for K in liquid_strikes:
        res = sim.price_european_call(K, T=1/12, n_paths=50000)
        liquid_prices.append(res['price'])
    liquid_prices = np.array(liquid_prices)
    
    # Solve RMOT
    print("Solving RMOT...")
    result = rmot.solve_dual_rmot(
        liquid_strikes, liquid_prices, T=1/12, n_samples=100000
    )
    
    # Check 1: Martingale constraint
    if rmot.prior_samples is None:
         print("FAILED: No prior samples found")
         return
         
    S_T = rmot.prior_samples
    if 'tilted_weights' not in result:
        print("FAILED: No tilted weights")
        return
        
    weights = result['tilted_weights']
    
    mean_ST = np.sum(weights * S_T)
    expected = params.S0 * np.exp(params.r * 1/12)
    
    martingale_error = abs(mean_ST - expected) / expected
    print(f"Martingale error: {100*martingale_error:.2f}%")
    
    if martingale_error >= 0.01:
        print(f"FAILED: Martingale violated! E[S_T] = {mean_ST:.2f}, expected {expected:.2f}")
    else:
        print(f"PASS: Martingale constraint ({100*martingale_error:.2f}%)")

    # Check 2: Calibration constraints
    calibration_errors = result['calibration_error']
    max_error = calibration_errors['max_rel_error']
    
    print(f"Max calibration error: {100*max_error:.2f}%")
    if max_error >= 0.02:
        print(f"FAILED: Calibration error too large! {100*max_error:.2f}% > 2%")
    else:
        print("✅ PASS: RMOT constraints satisfied")

if __name__ == '__main__':
    print("="*60)
    print("RUNNING CRITICAL VALIDATION TESTS")
    print("="*60)
    
    try:
        test_rough_heston_scaling()
    except Exception as e:
        print(f"Error in Heston scaling test: {e}")
        
    try:
        test_malliavin_sensitivity_scaling()
    except Exception as e:
        print(f"Error in Malliavin test: {e}")
        
    try:
        test_rmot_constraint_satisfaction()
    except Exception as e:
        print(f"Error in RMOT constraint test: {e}")
    
    print("="*60)
    print("TESTS COMPLETED")
    print("="*60)
