"""
Path Simulation for Multi-Asset Rough Heston

Simulates N-dimensional correlated rough Heston paths using
the Rough Martingale Copula construction (PDF Definition 2.8).

Reference: PDF Equations (6)-(7)
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.special import gamma

try:
    from .data_structures import RoughHestonParams
except ImportError:
    from data_structures import RoughHestonParams


def cholesky_decomposition_safe(correlation: np.ndarray) -> np.ndarray:
    """
    Safe Cholesky decomposition with fallback for near-PSD matrices.
    
    Args:
        correlation: N×N correlation matrix
    
    Returns:
        L: Lower triangular Cholesky factor
    """
    try:
        return np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError:
        # Fallback: eigendecomposition for near-PSD matrix
        eigvals, eigvecs = np.linalg.eigh(correlation)
        eigvals = np.maximum(eigvals, 1e-8)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
        return L


def simulate_rough_volatility(
    params: RoughHestonParams,
    n_paths: int,
    n_steps: int,
    dW_vol: np.ndarray
) -> np.ndarray:
    """
    Simulate rough volatility process ν_t.
    
    Reference: PDF Equation (2)
    
    ν_t = ξ_0(t) + (1/Γ(H+½)) ∫_0^t (t-s)^(H-½) [κ(θ-ν_s)ds + η√ν_s dW_s^V]
    
    Simplified implementation using Euler-Maruyama with fractional weighting.
    
    Args:
        params: Rough Heston parameters
        n_paths: Number of MC paths
        n_steps: Time discretization
        dW_vol: Brownian increments for vol process (n_paths, n_steps)
    
    Returns:
        vol_paths: (n_paths, n_steps+1) variance paths
    """
    H = params.H
    eta = params.eta
    kappa = params.kappa
    theta = params.theta
    xi0 = params.xi0
    T = params.maturity
    dt = T / n_steps
    
    # Fractional kernel normalization
    kernel_norm = 1.0 / gamma(H + 0.5)
    
    vol_paths = np.zeros((n_paths, n_steps + 1))
    vol_paths[:, 0] = xi0
    
    for t in range(n_steps):
        # Current variance
        v_t = np.maximum(vol_paths[:, t], 1e-10)
        
        # Mean reversion drift
        drift = kappa * (theta - v_t)
        
        # Vol-of-vol diffusion
        diffusion = eta * np.sqrt(v_t)
        
        # Fractional integration weight (simplified)
        frac_weight = kernel_norm * (dt) ** (H - 0.5)
        
        # Update
        dv = frac_weight * (drift * dt + diffusion * dW_vol[:, t])
        vol_paths[:, t + 1] = v_t + dv
        
        # Ensure positivity
        vol_paths[:, t + 1] = np.maximum(vol_paths[:, t + 1], 1e-10)
    
    return vol_paths


def simulate_correlated_rough_heston(
    marginal_params: List[RoughHestonParams],
    correlation: np.ndarray,
    n_paths: int = 10000,
    n_steps: int = 100,
    use_copula: bool = True
) -> np.ndarray:
    """
    Simulate N-dimensional correlated rough Heston paths.
    
    Reference: PDF Definition 2.8 (Rough Martingale Copula)
    
    The correlation matrix ρ governs the instantaneous quadratic
    covariation between log-price processes:
    
        d⟨X^i, X^j⟩_t = ρ_ij √(ν_t^i ν_t^j) dt
    
    Args:
        marginal_params: List of N RoughHestonParams
        correlation: N×N correlation matrix
        n_paths: Number of Monte Carlo paths
        n_steps: Time discretization
        use_copula: If True, use RoughMartingaleCopula (better correlation enforcement)
    
    Returns:
        paths: (n_paths, n_steps+1, N) array of spot prices
               paths[p, t, i] = S_t^i for path p, time t, asset i
    """
    # Try to use the improved copula implementation
    if use_copula:
        try:
            from .correlation_copula import RoughMartingaleCopula
            copula = RoughMartingaleCopula(marginal_params, correlation, calibrate_amplification=True)
            spot_paths, vol_paths, _ = copula.simulate(n_paths, n_steps)
            return spot_paths
        except ImportError:
            pass  # Fall back to original implementation
    
    # Original implementation (fallback)
    N = len(marginal_params)
    T = marginal_params[0].maturity
    dt = T / n_steps
    
    # ═══ STEP 1: Generate Correlated Brownian Motions ═══
    L = cholesky_decomposition_safe(correlation)
    
    # Independent Brownian increments for spot processes
    dW_spot_uncorr = np.random.randn(n_paths, n_steps, N) * np.sqrt(dt)
    
    # Correlate: dW_corr = dW_uncorr @ L^T
    dW_spot_corr = np.einsum('ptn,mn->ptm', dW_spot_uncorr, L.T)
    
    # Independent Brownian increments for vol processes
    dW_vol = np.random.randn(n_paths, n_steps, N) * np.sqrt(dt)
    
    # ═══ STEP 2: Simulate Volatility for Each Asset ═══
    vol_paths = np.zeros((n_paths, n_steps + 1, N))
    for i, params in enumerate(marginal_params):
        # Correlate vol Brownian with spot Brownian (via rho_i)
        dW_vol_i = params.rho * dW_spot_corr[:, :, i] + np.sqrt(1 - params.rho**2) * dW_vol[:, :, i]
        vol_paths[:, :, i] = simulate_rough_volatility(params, n_paths, n_steps, dW_vol_i)
    
    # ═══ STEP 3: Simulate Log-Price Paths ═══
    log_paths = np.zeros((n_paths, n_steps + 1, N))
    for i, params in enumerate(marginal_params):
        log_paths[:, 0, i] = np.log(params.spot)
    
    for t in range(n_steps):
        for i, params in enumerate(marginal_params):
            v_t = vol_paths[:, t, i]
            sqrt_v = np.sqrt(np.maximum(v_t, 1e-10))
            
            # Log-price dynamics: dX = (r - ½ν)dt + √ν dW^S
            drift = (params.r - 0.5 * v_t) * dt
            diffusion = sqrt_v * dW_spot_corr[:, t, i]
            
            log_paths[:, t + 1, i] = log_paths[:, t, i] + drift + diffusion
    
    # ═══ STEP 4: Convert to Spot Prices ═══
    paths = np.exp(log_paths)
    
    return paths


def validate_correlation_constraint(
    paths: np.ndarray,
    target_correlation: np.ndarray,
    tolerance: float = 0.15
) -> Tuple[bool, np.ndarray]:
    """
    Validate that simulated paths satisfy correlation constraints.
    
    Reference: PDF Equation (7)
    
    Computes empirical correlation from log-returns and compares
    to target correlation matrix.
    
    Args:
        paths: (n_paths, n_steps+1, N) spot prices
        target_correlation: N×N target correlation matrix
        tolerance: Maximum acceptable |ρ_empirical - ρ_target|
    
    Returns:
        (is_valid, empirical_correlation)
    """
    N = paths.shape[2]
    
    # Compute log-returns
    log_returns = np.diff(np.log(paths), axis=1)
    
    # Compute empirical correlation for each asset pair
    empirical_corr = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # Correlation of cumulative log-returns
            returns_i = log_returns[:, :, i].flatten()
            returns_j = log_returns[:, :, j].flatten()
            empirical_corr[i, j] = np.corrcoef(returns_i, returns_j)[0, 1]
    
    # Check deviation
    max_error = np.max(np.abs(empirical_corr - target_correlation))
    is_valid = max_error < tolerance
    
    if not is_valid:
        print(f"⚠️  Correlation validation failed: max error = {max_error:.4f}")
        print(f"  Target:\n{target_correlation}")
        print(f"  Empirical:\n{empirical_corr}")
    
    return is_valid, empirical_corr


# =====================================================================
# TESTS
# =====================================================================

def test_path_simulation():
    """Test basic path simulation."""
    params_1 = RoughHestonParams(
        H=0.10, eta=0.15, rho=-0.7, xi0=0.04, kappa=2.0, theta=0.04,
        spot=100.0, maturity=1/12
    )
    params_2 = RoughHestonParams(
        H=0.15, eta=0.18, rho=-0.5, xi0=0.05, kappa=1.5, theta=0.05,
        spot=100.0, maturity=1/12
    )
    
    correlation = np.array([[1.0, 0.6], [0.6, 1.0]])
    
    paths = simulate_correlated_rough_heston(
        [params_1, params_2], correlation, n_paths=5000, n_steps=50
    )
    
    print(f"✅ Path simulation: shape = {paths.shape}")
    print(f"  Asset 1: S_0 = {paths[:, 0, 0].mean():.2f}, S_T = {paths[:, -1, 0].mean():.2f}")
    print(f"  Asset 2: S_0 = {paths[:, 0, 1].mean():.2f}, S_T = {paths[:, -1, 1].mean():.2f}")
    
    is_valid, emp_corr = validate_correlation_constraint(paths, correlation)
    print(f"  Correlation validation: {'✅' if is_valid else '❌'}")
    print(f"  Empirical correlation:\n{emp_corr}")


if __name__ == "__main__":
    test_path_simulation()
