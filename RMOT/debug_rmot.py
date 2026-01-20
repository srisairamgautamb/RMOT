
import numpy as np
import warnings
from src.simulation.rough_heston import RoughHestonParams, RoughHestonSimulator
from src.pricing.rmot_solver import RMOTPricingEngine

# Do not suppress warnings
# warnings.filterwarnings('ignore')

def debug_rmot():
    print("DEBUG: Starting RMOT Calibration")
    
    params = RoughHestonParams(
        H=0.1, eta=1.0, rho=-0.5, xi0=0.04,
        kappa=1.0, theta=0.04, S0=100.0, r=0.0
    )
    sim = RoughHestonSimulator(params)
    rmot = RMOTPricingEngine(sim, lambda_reg=0.01)
    
    n_strikes = 50
    liquid_strikes = np.linspace(80, 120, n_strikes)
    # Same prices as benchmark
    liquid_prices = np.maximum(100.0 - liquid_strikes, 0) + 3.0 * np.exp(-0.01 * (liquid_strikes - 100)**2)
    
    print(f"DEBUG: Running solve_dual_rmot with {n_strikes} strikes...")
    
    # Run with less samples for speed, but enough to trigger issue
    result = rmot.solve_dual_rmot(
        liquid_strikes, liquid_prices, T=0.5, 
        target_strike=130.0, n_samples=50000
    )
    
    print(f"DEBUG: Optimization Success: {result['optimization_success']}")
    if not result['optimization_success']:
        print("DEBUG: FAILURE MESSAGE captured manually")
        # Ensure we print it if it wasn't warned
    
    print(f"Calibration RMSE: {result['calibration_error']['rmse']}")

if __name__ == "__main__":
    debug_rmot()
