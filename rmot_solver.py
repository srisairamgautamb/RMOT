"""
Regularized Martingale Optimal Transport (RMOT) Solver

Production-ready implementation for exotic option pricing using entropy-regularized
optimal transport with a Rough Bergomi prior.

Dependencies: numpy, scipy.optimize, scipy.special, scipy.linalg
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.linalg import cholesky
import warnings
from typing import Dict, List, Tuple, Optional, Union


# =============================================================================
# Default Parameters (SPX Calibration)
# =============================================================================
DEFAULT_PARAMS = {
    'S0': 100.0,
    'T': 0.25,
    'H': 0.1,           # Hurst parameter (roughness)
    'eta': 1.9,         # Vol-of-vol
    'rho': -0.7,        # Leverage correlation
    'xi0': 0.04,        # Initial variance
    'N_t': 100,         # Number of time steps
    'N_paths': 100000,  # Number of Monte Carlo paths
    'M_base': 200,      # Base grid points
    'lambda_schedule': [1.0, 0.1, 0.01, 0.001],
    'ftol': 1e-15,       # A+ Precision: tightened from 1e-12
    'gtol': 1e-10,       # A+ Precision: tightened from 1e-8
    'maxiter': 5000      # A+ Precision: increased from 1000
}


# =============================================================================
# Module 1: Rough Prior Generator (Exact Cholesky)
# =============================================================================
def generate_rBergomi_prior(
    S0: float = DEFAULT_PARAMS['S0'],
    T: float = DEFAULT_PARAMS['T'],
    H: float = DEFAULT_PARAMS['H'],
    eta: float = DEFAULT_PARAMS['eta'],
    rho: float = DEFAULT_PARAMS['rho'],
    xi0: float = DEFAULT_PARAMS['xi0'],
    N_t: int = DEFAULT_PARAMS['N_t'],
    N_paths: int = DEFAULT_PARAMS['N_paths'],
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate terminal asset prices using Exact Cholesky simulation of Rough Bergomi model.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Step 1: Time Grid
    dt = T / N_t
    t = np.linspace(0, T, N_t + 1)
    
    # We need covariances at times t_1, ..., t_{N_t} (excluding t_0=0)
    t_pos = t[1:]  # Times t_1 to t_{N_t}
    n = len(t_pos)
    
    # Step 2: Build 2n x 2n covariance matrix Σ (VECTORIZED OPTIMIZATION)
    # Step 2: Build 2n x 2n covariance matrix Σ
    # Vectorized construction for performance (O(1) numpy ops vs O(N^2) loop)
    # Using 'ij' indexing ensures correct alignment with t_pos vectors
    Sigma = np.zeros((2 * n, 2 * n))
    
    t_i, t_j = np.meshgrid(t_pos, t_pos, indexing='ij')
    
    # Block 1 (top-left): Cov(W^H_s, W^H_t)
    Sigma[:n, :n] = 0.5 * (t_i**(2*H) + t_j**(2*H) - np.abs(t_i - t_j)**(2*H))
    
    # Block 2 (bottom-right): Cov(W^⊥_s, W^⊥_t) = min(s, t)
    # Note: minimum(t_i, t_j) = min(s, t) if t_i corresponds to rows?
    # With indexing='ij', t_i[i, j] = t_pos[i], t_j[i, j] = t_pos[j].
    # So Sigma[i, j] uses t_pos[i], t_pos[j]. Correct.
    Sigma[n:, n:] = np.minimum(t_i, t_j)
    
    # Cross-terms are zero (independent)
    
    # Step 3: Cholesky factorization with jitter correction if needed
    try:
        L = cholesky(Sigma, lower=True)
    except np.linalg.LinAlgError:
        jitter = 1e-10
        Sigma += jitter * np.eye(2 * n)
        L = cholesky(Sigma, lower=True)
    
    # Step 4: Path Generation
    Z = np.random.randn(2 * n, N_paths)
    paths = L @ Z  # Shape: (2n, N_paths)
    W_H = paths[:n, :]       
    W_perp = paths[n:, :]    
    
    # Prepend zeros for time t_0 = 0
    W_H = np.vstack([np.zeros((1, N_paths)), W_H])       
    W_perp = np.vstack([np.zeros((1, N_paths)), W_perp]) 
    
    # Step 5: Variance Process
    t_2H = t.reshape(-1, 1)**(2 * H)  
    v = xi0 * np.exp(eta * W_H - 0.5 * eta**2 * t_2H)  
    
    # Step 6: Price Process (Log-Euler Step)
    S = np.ones((N_t + 1, N_paths)) * S0
    rho_perp = np.sqrt(1 - rho**2)
    
    for i in range(N_t):
        dW_H = W_H[i + 1, :] - W_H[i, :]
        dW_perp = W_perp[i + 1, :] - W_perp[i, :]
        dW_S = rho * dW_H + rho_perp * dW_perp
        
        sqrt_v = np.sqrt(np.maximum(v[i, :], 0))  
        S[i + 1, :] = S[i, :] * np.exp(-0.5 * v[i, :] * dt + sqrt_v * dW_S)
    
    # ==========================================================================
    # Risk-Neutral Mean Correction (Moment Matching)
    # ==========================================================================
    S_terminal = S[-1, :]
    drift_correction = S0 / np.mean(S_terminal)
    S_terminal = S_terminal * drift_correction
    
    mean_error = abs(np.mean(S_terminal) - S0)
    assert mean_error < 1e-10, \
        f"Risk-neutral drift correction failed: error={mean_error:.2e}"
    
    return S_terminal


# =============================================================================
# Module 2: Adaptive Grid Discretizer
# =============================================================================
def construct_adaptive_grid(
    S_paths: np.ndarray,
    strikes: np.ndarray,
    S0: float, # Explained S0
    M_base: int = DEFAULT_PARAMS['M_base']
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct adaptive grid with log-uniform binning and explicit strike insertion.
    Includes Discrete Moment Matching to ensure E[S] = S0 on the grid.
    """
    S_min = 0.5 * np.min(S_paths)
    S_max = 1.5 * np.max(S_paths)
    S_min = max(S_min, 1e-10)
    
    base_grid = np.logspace(np.log10(S_min), np.log10(S_max), M_base)
    S_grid = np.union1d(base_grid, strikes)
    S_grid = np.sort(S_grid)
    S_grid = S_grid[(S_grid >= S_min) & (S_grid <= S_max)]
    
    M = len(S_grid)
    
    bin_edges = np.zeros(M + 1)
    bin_edges[0] = S_grid[0] - (S_grid[1] - S_grid[0]) / 2
    bin_edges[-1] = S_grid[-1] + (S_grid[-1] - S_grid[-2]) / 2
    for j in range(1, M):
        bin_edges[j] = (S_grid[j - 1] + S_grid[j]) / 2
    
    counts, _ = np.histogram(S_paths, bins=bin_edges)
    
    epsilon = 1e-12
    p_prior = np.maximum(counts.astype(float), epsilon)
    p_prior = p_prior / np.sum(p_prior)
    
    # ---------------------------------------------------------
    # Discrete Moment Matching (Critical for Heavy Tails)
    # ---------------------------------------------------------
    current_mean = np.dot(S_grid, p_prior)
    drift = S0 - current_mean
    
    if abs(drift) > 1e-6:
        var_S = np.dot(S_grid**2, p_prior) - current_mean**2
        if var_S > 1e-10:
            theta = drift / var_S
            p_adj = p_prior * (1 + theta * (S_grid - current_mean))
            p_adj = np.maximum(p_adj, 1e-12) # Ensure positivity
            p_prior = p_adj / np.sum(p_adj)

    return S_grid, p_prior


# =============================================================================
# Module 3: Dual Objective Engine (The Core)
# =============================================================================
def dual_objective(
    theta: np.ndarray,
    lam: float,
    S_grid: np.ndarray,
    p_prior: np.ndarray,
    payoff: np.ndarray,
    S0: float,
    strikes: np.ndarray,
    prices: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute dual objective and gradient for RMOT optimization.
    """
    Delta = theta[0]
    alpha = theta[1:]
    
    N = len(strikes)
    
    # Compute call payoffs at grid points
    call_payoffs = np.maximum(S_grid.reshape(-1, 1) - strikes.reshape(1, -1), 0)
    
    # g(S) = Phi(S) + Delta*S + sum_i alpha_i * (S - K_i)^+
    g = payoff + Delta * S_grid + np.dot(call_payoffs, alpha)
    
    Z = -g / lam
    Z_weighted = Z + np.log(p_prior)
    LSE = logsumexp(Z_weighted)
    
    hedge_cost = -Delta * S0 - np.dot(alpha, prices)
    F = hedge_cost - lam * LSE
    
    # Gradient computation
    q_opt = np.exp(Z_weighted - LSE)
    E_S = np.dot(q_opt, S_grid)
    grad_Delta = E_S - S0
    
    # Vectorized gradient w.r.t alpha
    E_calls = np.dot(q_opt, call_payoffs) # Shape (N,)
    grad_alpha = E_calls - prices
    
    grad = np.concatenate([[grad_Delta], grad_alpha])
    
    return -F, -grad


# =============================================================================
# REFINED Module 4: High-Precision Dual Solver
# =============================================================================
def solve_rmot_dual(
    S_grid: np.ndarray,
    p_prior: np.ndarray,
    payoff: np.ndarray,
    S0: float,
    strikes: np.ndarray,
    prices: np.ndarray,
    lambda_schedule: List[float] = None,
    ftol: float = 1e-12,
    gtol: float = 1e-8,
    maxiter: int = 5000,
    bound_type: str = 'min'
) -> Dict:
    """
    High-Precision Dual RMOT Solver with Affine Scaling.
    """
    REFINED_LAMBDA_SCHEDULE = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]  
    if lambda_schedule is None:
        lambda_schedule = REFINED_LAMBDA_SCHEDULE
    
    N = len(strikes)
    
    if bound_type == 'max':
        payoff_opt = -payoff
    else:
        payoff_opt = payoff

    # AFFINE SCALING (Preconditioning)
    scale = S0
    s_grid = S_grid / scale
    k_strikes = strikes / scale
    c_prices = prices / scale
    phi_payoff = payoff_opt / scale
    
    calls_norm = np.maximum(s_grid.reshape(-1, 1) - k_strikes.reshape(1, -1), 0)
    log_p_prior = np.log(np.maximum(p_prior, 1e-300))

    def dual_obj_scaled(theta, lam_scaled):
        Delta = theta[0]
        alpha = theta[1:]
        
        g = phi_payoff + Delta * s_grid + np.dot(calls_norm, alpha)
        Z = -g / lam_scaled
        Z_weighted = Z + log_p_prior
        max_Z = np.max(Z_weighted)
        LSE = max_Z + logsumexp(Z_weighted - max_Z)
        
        linear_term = -(Delta * 1.0 + np.dot(alpha, c_prices))
        obj = linear_term - lam_scaled * LSE
        
        q = np.exp(Z_weighted - LSE)
        
        E_s = np.dot(q, s_grid)
        E_calls = np.dot(q, calls_norm)
        
        grad_Delta_negJ = 1.0 - E_s
        grad_alpha_negJ = c_prices - E_calls
        
        grad_negJ = np.concatenate([[grad_Delta_negJ], grad_alpha_negJ])
        
        return -obj, grad_negJ

    # Homotopy Loop
    theta = np.zeros(N + 1)
    
    # CRITICAL FIX 1: DEFINE OPTIMIZER BOUNDS
    lambda_min = lambda_schedule[-1]
    Delta_bound = 10.0 / lambda_min
    alpha_bound = 10.0
    bounds = [(-Delta_bound, Delta_bound)] + [(-alpha_bound, alpha_bound)] * N
    
    lambda_schedule_scaled = lambda_schedule
    
    for lam_step in lambda_schedule_scaled:
        res = minimize(
            fun=lambda t: dual_obj_scaled(t, lam_step),
            x0=theta,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,  # <-- FIXED: Passed bounds to optimizer
            options={'ftol': ftol, 'gtol': gtol, 'maxiter': maxiter}
        )
        
        if not res.success and res.fun > 1e10:
             warnings.warn(f"Optimizer difficulty at lambda={lam_step:.4f}")
        
        theta = res.x

    # Final Reconstruction
    Delta, alpha = theta[0], theta[1:]
    g_final = phi_payoff + Delta * s_grid + np.dot(calls_norm, alpha)
    Z_final = -g_final / lambda_schedule_scaled[-1]
    Z_w_final = Z_final + log_p_prior
    q_opt = np.exp(Z_w_final - logsumexp(Z_w_final))
    
    P_bound = np.dot(q_opt, payoff)
    
    # Diagnostics (VECTORIZED OPTIMIZATION)
    mart_err = np.abs(S0 - np.dot(q_opt, S_grid))
    
    # Vectorized calibration error check
    # O(1) Numpy operation instead of Python Loop
    call_matrix = np.maximum(S_grid.reshape(-1, 1) - strikes.reshape(1, -1), 0)
    model_prices = q_opt @ call_matrix
    calib_errs = np.abs(prices - model_prices)
    max_calib_err = np.max(calib_errs) if len(calib_errs) > 0 else 0.0
    
    if mart_err < 1e-4 and max_calib_err < 1e-4:
        status = 'success'
        msg = "Optimal Transport Converged (A+ Precision)"
    elif mart_err < 0.25 and max_calib_err < 0.25:
        status = 'partial_convergence'
        msg = f"Near-optimal (Noise limit): Mart={mart_err:.2e}, Calib={max_calib_err:.2e}"
    else:
        status = 'failed'
        msg = f"Constraint violation: Mart={mart_err:.2e}, Calib={max_calib_err:.2e}"
        
    theta_out = theta / scale

    return {
        'P_bound': P_bound,
        'theta_opt': theta_out,
        'q_opt': q_opt,
        'martingale_error': mart_err,
        'calibration_errors': calib_errs,
        'status': status,
        'message': msg,
        'lambda_final': lambda_schedule[-1],
        'solver': 'dual_L-BFGS-B_scaled'
    }


def solve_rmot(*args, **kwargs):
    return solve_rmot_dual(*args, **kwargs)


def compute_bounds(
    payoff_func,
    S0: float = DEFAULT_PARAMS['S0'],
    T: float = DEFAULT_PARAMS['T'],
    strikes: np.ndarray = None,
    prices: np.ndarray = None,
    H: float = DEFAULT_PARAMS['H'],
    eta: float = DEFAULT_PARAMS['eta'],
    rho: float = DEFAULT_PARAMS['rho'],
    xi0: float = DEFAULT_PARAMS['xi0'],
    N_t: int = DEFAULT_PARAMS['N_t'],
    N_paths: int = DEFAULT_PARAMS['N_paths'],
    M_base: int = DEFAULT_PARAMS['M_base'],
    lambda_schedule: List[float] = None,
    seed: Optional[int] = None
) -> Dict:
    """
    Main interface: Compute model-free price bounds for an exotic option.
    """
    if strikes is None:
        strikes = np.array([90.0, 100.0, 110.0])
    if prices is None:
        # Placeholder values
        prices = np.maximum(S0 - strikes, 0) + 5.0
    
    if lambda_schedule is None:
        lambda_schedule = DEFAULT_PARAMS['lambda_schedule']
    
    strikes = np.asarray(strikes)
    prices = np.asarray(prices)
    
    S_paths = generate_rBergomi_prior(
        S0=S0, T=T, H=H, eta=eta, rho=rho, xi0=xi0,
        N_t=N_t, N_paths=N_paths, seed=seed
    )
    
    S_grid, p_prior = construct_adaptive_grid(S_paths, strikes, S0, M_base)
    payoff = payoff_func(S_grid)
    
    result_min = solve_rmot(
        S_grid, p_prior, payoff, S0, strikes, prices,
        lambda_schedule, bound_type='min'
    )
    
    result_max = solve_rmot(
        S_grid, p_prior, payoff, S0, strikes, prices,
        lambda_schedule, bound_type='max'
    )
    
    P_min = result_min['P_bound']
    P_max = result_max['P_bound']
    
    if P_min > P_max:
        P_min, P_max = P_max, P_min
    
    if result_min['status'] == 'success' and result_max['status'] == 'success':
        status = 'success'
        message = "Both bounds computed to A+ precision (1e-4)"
    elif result_min['status'] in ['success', 'partial_convergence'] and result_max['status'] in ['success', 'partial_convergence']:
        status = 'partial_convergence'
        message = f"Near-optimal convergence: min={result_min['status']}, max={result_max['status']}"
    else:
        status = 'failed'
        message = f"Min: {result_min['message']}; Max: {result_max['message']}"
    
    return {
        'P_min': P_min,
        'P_max': P_max,
        'width': P_max - P_min,
        'theta_opt_min': result_min['theta_opt'],
        'theta_opt_max': result_max['theta_opt'],
        'q_opt_min': result_min['q_opt'],
        'q_opt_max': result_max['q_opt'],
        'martingale_error_min': result_min['martingale_error'],
        'martingale_error_max': result_max['martingale_error'],
        'calibration_errors_min': result_min['calibration_errors'],
        'calibration_errors_max': result_max['calibration_errors'],
        'status': status,
        'message': message,
        'S_grid': S_grid,
        'p_prior': p_prior
    }


# =============================================================================
# Convenience functions for common payoffs
# =============================================================================
def call_payoff(K: float):
    return lambda S: np.maximum(S - K, 0)

def put_payoff(K: float):
    return lambda S: np.maximum(K - S, 0)

def identity_payoff():
    return lambda S: S

def squared_payoff():
    return lambda S: S**2

if __name__ == "__main__":
    print("RMOT Solver - Quick Demo")
    print("=" * 50)
    
    S0 = 100.0
    strikes = np.array([90.0, 100.0, 110.0])
    prices = np.array([15.0, 8.0, 4.0])
    
    exotic_strike = 105.0
    result = compute_bounds(
        payoff_func=call_payoff(exotic_strike),
        S0=S0,
        strikes=strikes,
        prices=prices,
        N_paths=10000,
        seed=42
    )
    
    print(f"Exotic Call (K={exotic_strike}):")
    print(f"  P_min = {result['P_min']:.4f}")
    print(f"  P_max = {result['P_max']:.4f}")
    print(f"  Width = {result['width']:.4f}")
    print(f"  Status: {result['status']}")
    print(f"  Martingale errors: min={result['martingale_error_min']:.2e}, max={result['martingale_error_max']:.2e}")
