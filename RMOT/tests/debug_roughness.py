
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.simulation.rough_heston import RoughHestonSimulator, RoughHestonParams

def check_volatility_roughness():
    print("Checking Volatility Roughness (E[(v_t - v_0)^2] ~ t^2H)...")
    
    params = RoughHestonParams(
        H=0.1, eta=1.0, rho=0.0, xi0=0.04,
        kappa=0.001, theta=0.04, S0=100.0, r=0.0
    )
    # Kappa=0 simplifies to pure fractional Brownian motion driven volatility
    
    sim = RoughHestonSimulator(params)
    T = 1.0
    n_steps = 1000
    n_paths = 5000
    
    print("Simulating...")
    S, v = sim.simulate(T, n_steps, n_paths, return_variance=True)
    
    # Check scaling of v increments or v_t from v_0
    # E[(v_t - v_0)^2] vs t
    
    times = np.linspace(0, T, n_steps+1)
    variogram = np.mean((v - params.xi0)**2, axis=0) # Average over paths
    
    # Skip t=0
    times = times[1:]
    variogram = variogram[1:]
    
    log_t = np.log(times)
    log_var = np.log(variogram)
    
    # Fit line
    slope, intercept = np.polyfit(log_t, log_var, 1)
    
    print(f"Meausred Slope of E[(v_t - v0)^2]: {slope:.4f}")
    print(f"Expected Slope (2H): {2 * params.H:.4f}")
    
    if abs(slope - 2*params.H) < 0.1:
        print("✅ Volatility IS rough (Simulator is correct)")
    else:
        print("❌ Volatility is NOT rough (Simulator is wrong)")

    # Also check Auditor's S scaling
    log_ret_var = np.var(np.log(S[:, -1] / S[:, 0]))
    # For Heston, Var(log S) ~ T * v0
    # We used T=1.
    print(f"Var(log S_T): {log_ret_var:.4f}")
    print("If we change T, does it scale as T^1 or T^2H?")

if __name__ == "__main__":
    check_volatility_roughness()
