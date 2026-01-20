
import numpy as np
import sys
import os
import time

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.rough_heston import RoughHestonParams, RoughHestonSimulator
from src.pricing.rmot_solver import RMOTPricingEngine

def run_self_consistency_test():
    print("Running RMOT Solver Self-Consistency Test...")
    
    # 1. Define Model
    params = RoughHestonParams(
        H=0.1, eta=1.0, rho=-0.5, xi0=0.04,
        kappa=1.0, theta=0.04, S0=100.0, r=0.0
    )
    sim = RoughHestonSimulator(params)
    
    T = 0.5
    n_paths_truth = 50_000 # Reduced for faster testing
    n_steps = 100
    
    # 2. Generate Ground Truth Prices (The "Market")
    print(f"Generating Ground Truth Prices ({n_paths_truth} paths)...")
    np.random.seed(42)
    S_paths, _, _ = sim.simulate(T=T, n_steps=n_steps, n_paths=n_paths_truth, return_variance=True, return_noise=True)
    S_T = S_paths[:, -1]
    
    strikes = np.linspace(80, 120, 21) # 80, 82, ..., 120
    true_prices = []
    for K in strikes:
        payoff = np.maximum(S_T - K, 0)
        price = np.mean(payoff) * np.exp(-params.r * T)
        true_prices.append(price)
    true_prices = np.array(true_prices)
    
    print("Ground Truth Prices generated.")
    print(f"ATM Price (K=100): {true_prices[10]:.4f}")
    
    # 3. Initialize Solver with NEW samples (Independent Prior)
    print("\nInitializing Solver (Prior)...")
    # Solver uses its own simulation internally (defaults to 100k usually, we can specify)
    # Using lambda_reg=0.01 (Standard)
    rmot = RMOTPricingEngine(sim, lambda_reg=0.01)
    
    # 4. Calibrate
    print("Calibrating to Ground Truth...")
    start_time = time.time()
    result = rmot.solve_dual_rmot(
        strikes, true_prices, T=T,
        n_samples=50_000, # Reduced for faster testing
        target_strike=None
    )
    end_time = time.time()
    
    print(f"\nCalibration Complete in {end_time - start_time:.4f}s")
    print(f"Optimization Success: {result['optimization_success']}")
    
    # 5. Check Error
    cal_err = result['calibration_error']
    max_rel_err = cal_err['max_rel_error'] * 100
    rmse = cal_err['rmse']
    
    print("\nResults:")
    print(f"  Max Relative Error: {max_rel_err:.4f}%")
    print(f"  RMSE: {rmse:.6f}")
    
    if max_rel_err < 2.0:
        print("\n✅ PASS: Solver successfully recovered Ground Truth distribution (< 2%).")
        print("   This confirms the Solver logic is CORRECT.")
    else:
        print(f"\n❌ FAIL: Error {max_rel_err:.4f}% > 2%.")
        print("   Potential Issue: Solver logic or sample variance.")
        
if __name__ == "__main__":
    run_self_consistency_test()
