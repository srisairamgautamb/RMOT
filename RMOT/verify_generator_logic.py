
import numpy as np
from rmot_solver import generate_rBergomi_prior, DEFAULT_PARAMS

def check_prior():
    S0 = 100.0
    N_paths = 100000
    seed = 42
    
    print("Testing generate_rBergomi_prior with Loop implementation...")
    # Manually check Sigma construction
    T=0.25; H=0.1; N_t=100
    t = np.linspace(0, T, N_t + 1)
    t_pos = t[1:]
    n = len(t_pos)
    Sigma = np.zeros((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            ti, tj = t_pos[i], t_pos[j]
            Sigma[i, j] = 0.5 * (ti**(2*H) + tj**(2*H) - np.abs(ti - tj)**(2*H))
            Sigma[n + i, n + j] = min(ti, tj)
            
    print(f"Sigma Max: {np.max(Sigma)}")
    print(f"Sigma Min: {np.min(Sigma)}")
    
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(2*n))
    print(f"L Max: {np.max(L)}")
    
    S_terminal = generate_rBergomi_prior(S0=S0, N_paths=N_paths, seed=seed)
    
    print(f"Stats for Seed {seed}:")
    print(f"Mean: {np.mean(S_terminal)}")
    print(f"Min:  {np.min(S_terminal)}")
    print(f"Max:  {np.max(S_terminal)}")
    print(f"Std:  {np.std(S_terminal)}")
    
    strikes = [90, 100, 110]
    prices = []
    for K in strikes:
        p = np.mean(np.maximum(S_terminal - K, 0))
        prices.append(p)
    print(f"Call Prices [90, 100, 110]: {prices}")
    
    # Sanity checks
    # For SPX parameters, prices should be roughly 15, 8, 4
    if prices[1] > 20:
        print("FAIL: Prices seem insanely high. Volatility implementation likely broken.")
    elif prices[1] < 1:
        print("FAIL: Prices too low.")
    else:
        print("SUCCESS: Prices look reasonable.")

if __name__ == "__main__":
    check_prior()
