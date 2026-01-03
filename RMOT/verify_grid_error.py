
import numpy as np
from rmot_solver import generate_rBergomi_prior, construct_adaptive_grid, DEFAULT_PARAMS

def check_grid_error():
    S0 = 100.0
    N_paths = 100000 # Stress test size
    seed = 42
    
    print("Generating Paths...")
    S_paths = generate_rBergomi_prior(S0=S0, N_paths=N_paths, seed=seed)
    print(f"Path Mean: {np.mean(S_paths):.6f} (Should be 100.0)")
    print(f"Path Max:  {np.max(S_paths):.2f}")
    
    print("Constructing Grid...")
    # Using default M_base=200
    S_grid, p_prior = construct_adaptive_grid(S_paths, strikes=np.array([90, 100, 110]), M_base=200)
    
    discrete_mean = np.dot(S_grid, p_prior)
    error = discrete_mean - S0
    
    print(f"Grid Mean: {discrete_mean:.6f}")
    print(f"Grid Error: {error:.6f}")
    
    if abs(error) > 0.1:
        print("FAIL: Discretization error is too large. Solver cannot recover.")
    else:
        print("SUCCESS: Grid is accurate enough.")

if __name__ == "__main__":
    check_grid_error()
