
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add root to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.simulation.rough_heston import RoughHestonParams, RoughHestonSimulator
from src.pricing.rmot_solver import RMOTPricingEngine

def run_convergence_analysis():
    print("="*70)
    print("RMOT SOLVER CONVERGENCE ANALYSIS")
    print("="*70)
    
    # Ground Truth Physics (Reference)
    params = RoughHestonParams(
        H=0.1, eta=1.9, rho=-0.7, xi0=0.04,
        kappa=1.0, theta=0.04, S0=100.0, r=0.0
    )
    
    T = 1/12
    strikes = np.linspace(80, 120, 21)
    
    # Generate Ground Truth Prices (High Precision)
    print("Generating Ground Truth (50k paths)...")
    sim_gt = RoughHestonSimulator(params)
    S_paths_gt = sim_gt.simulate(T, 50000, 100)[:, -1]
    prices_gt = np.array([np.mean(np.maximum(S_paths_gt - K, 0)) for K in strikes])
    
    # Test sample sizes (reduced for faster testing)
    sample_sizes = [5000, 10000, 25000, 50000]
    errors = []
    
    print("\nRunning Convergence Test:")
    print(f"{'N Samples':<15} | {'Max Rel Error':<15} | {'RMSE':<15}")
    print("-" * 50)
    
    for n in sample_sizes:
        # Prior is SAME model (Twin Experiment)
        # We expect error to drop as 1/sqrt(N)
        sim = RoughHestonSimulator(params)
        rmot = RMOTPricingEngine(sim, lambda_reg=0.01) # Small reg for twin experiment
        
        # Solve
        # Note: solve_dual_rmot generates new samples internally
        result = rmot.solve_dual_rmot(
            strikes, prices_gt, T, n_samples=n
        )
        
        err = result['calibration_error']['max_rel_error']
        rmse = result['calibration_error']['rmse']
        
        errors.append(err)
        print(f"{n:<15} | {err:.2%}          | {rmse:.4e}")
        
    # Analysis
    errors = np.array(errors)
    ns = np.array(sample_sizes)
    
    # Fit log-log to check slope (should be -0.5)
    log_ns = np.log(ns)
    log_errs = np.log(errors)
    slope, intercept = np.polyfit(log_ns, log_errs, 1)
    
    print(f"\nConvergence Slope: {slope:.4f} (Expected ~ -0.5)")
    
    if -0.6 < slope < -0.4:
        print("✅ PASS: Convergence rate consistent with Monte Carlo theory.")
    else:
        print("⚠️ WARNING: Convergence rate deviates from theory.")

if __name__ == "__main__":
    run_convergence_analysis()
