
import time
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.simulation.rough_heston import RoughHestonSimulator, RoughHestonParams

def verify_problem_sizes():
    """Ensure benchmark used correct problem sizes"""
    
    print("Verifying benchmark problem sizes...")
    
    # Path Simulation
    print("\n1. PATH SIMULATION:")
    print("   Expected: 10^6 paths × 100 steps")
    # We will simulate and check shape
    params = RoughHestonParams(
        H=0.1, eta=1.0, rho=-0.5, xi0=0.04,
        kappa=1.0, theta=0.04, S0=100.0, r=0.0
    )
    sim = RoughHestonSimulator(params)
    paths = sim.simulate(T=0.5, n_steps=100, n_paths=100) # Small sample
    print(f"   Shape 100 paths: {paths.shape} (Matches 100+1 columns?)")
    
    # Malliavin
    print("\n2. MALLIAVIN WEIGHTS:")
    print("   Expected: 50 strikes × 10^6 paths")
    
    # Fisher Matrix
    print("\n3. FISHER MATRIX:")
    print("   Expected: Compute 50 MC prices + 250 sensitivities")
    
    # RMOT
    print("\n4. RMOT CALIBRATION:")
    print("   Expected: 100k prior samples + dual optimization")


def measure_minimum_mc_time():
    """Measure theoretical minimum for MC operations"""
    
    params = RoughHestonParams(
        H=0.1, eta=1.9, rho=-0.7, xi0=0.04,
        kappa=1.5, theta=0.04, S0=100.0, r=0.0
    )
    
    sim = RoughHestonSimulator(params)
    
    # Minimum time for 100k paths (needed for RMOT)
    print("\nMeasuring MC overhead...")
    
    start = time.time()
    # 21 steps = 1 month / daily roughly
    paths = sim.simulate(T=1/12, n_steps=21, n_paths=100000) 
    elapsed = time.time() - start
    
    print(f"100k paths × 21 steps: {elapsed:.3f} sec")
    print(f"This is the MINIMUM for RMOT calibration")
    print(f"Your RMOT time: 0.08 sec (from previous report)")
    
    if elapsed > 0.08:
        print(f"🔴 IMPOSSIBLE: Your RMOT is faster than MC generation!")
        print(f"   Ratio: {0.08/elapsed:.2f}× (should be ≥ 1.0)")
        return False
    else:
        print(f"✅ Plausible (MC is fast enough)")
        return True

if __name__ == '__main__':
    verify_problem_sizes()
    measure_minimum_mc_time()
