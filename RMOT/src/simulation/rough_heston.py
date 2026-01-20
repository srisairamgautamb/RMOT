
import numpy as np
from numba import njit, prange
from typing import Tuple, Dict
from dataclasses import dataclass

@dataclass
class RoughHestonParams:
    """Parameter container with validation"""
    H: float        # Hurst exponent ∈ (0, 0.5)
    eta: float      # Vol-of-vol > 0
    rho: float      # Correlation ∈ [-1, 1]
    xi0: float      # Initial variance > 0
    kappa: float    # Mean reversion > 0
    theta: float    # Long-term variance > 0
    S0: float       # Initial price > 0
    r: float        # Risk-free rate
    
    def __post_init__(self):
        """Validate parameters"""
        assert 0 < self.H < 0.5, f"H={self.H} must be in (0, 0.5)"
        assert self.eta > 0, f"eta={self.eta} must be positive"
        assert -1 <= self.rho <= 1, f"rho={self.rho} must be in [-1, 1]"
        assert self.xi0 > 0, f"xi0={self.xi0} must be positive"
        assert self.kappa > 0, f"kappa={self.kappa} must be positive"
        assert self.theta > 0, f"theta={self.theta} must be positive"
        assert self.S0 > 0, f"S0={self.S0} must be positive"

from math import gamma

@njit(parallel=True, cache=True, fastmath=True)
def compute_weights(n_steps: int, H: float, dt: float) -> np.ndarray:
    """
    Compute Riemann-Liouville weights b_{i,j}
    
    Formula: b_{i,j} = (1/Γ(H+1/2)) * [(i-j)^(H+1/2) - (i-j-1)^(H+1/2)] / dt^(H+1/2)
    
    Returns: Array of shape (n_steps, n_steps) with b[i,j] = weight for step i from step j
    """
    
    # Correct Discrete Volterra Kernel weights
    # Integral of (t-s)^{H-0.5} is -(t-s)^{H+0.5}/(H+0.5)
    # Scale factor is dt**(H+0.5) / Gamma(H+1.5)
    
    weights = np.zeros((n_steps, n_steps))
    gamma_val = gamma(H + 1.5)
    gamma_factor = dt**(H + 0.5) / gamma_val
    
    # We are computing weights for the Volterra process convolution
    # v_{i+1} ~ sum_{j=0}^{i} K(t_{i+1}, t_j) ... 
    # The kernel is (t-s)^{H-0.5}
    # Discretization: b_{i+1, j} corresponds to integral over [t_j, t_{j+1}]?
    # Spec says: b_{i+1,j} = (1/Γ(H+1/2)) [(i+1-j)^(H+1/2) - (i-j)^(H+1/2)] / dt^(H+1/2)
    # Note: The spec formula has (i+1-j) and (i-j). 
    # Let's double check indices. 
    # If we want weight for v_{i+1} from noise at step j (which is usually W_{j+1}-W_j or similar)
    # The code in spec:
    # for i in prange(n_steps):
    #     for j in range(i + 1):
    #         diff = i - j
    #         if diff == 0: ...
    # This seems to compute weights for the current step i based on history j <= i.
    
    for i in prange(n_steps):
        for j in range(i + 1):
            diff = i - j
            if diff == 0:
                # Integral from 0 to 1 of x^{H-0.5} = 1^{H+0.5} / (H+0.5)
                # But here we use the discrete difference formula from spec
                # term (0+1)^(H+0.5) - (0)^(H+0.5) = 1
                weights[i, j] = gamma_factor * (1.0)**(H + 0.5)
            else:
                weights[i, j] = gamma_factor * (
                    (diff + 1)**(H + 0.5) - (diff)**(H + 0.5)
                )
    
    return weights

@njit(parallel=True, cache=True, fastmath=True)
def simulate_paths_rough_heston(
    params_array: np.ndarray,  # [H, eta, rho, xi0, kappa, theta, S0, r]
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate rough Heston paths using hybrid scheme
    
    Returns:
        S_paths: Array of shape (n_paths, n_steps+1) with stock prices
        v_paths: Array of shape (n_paths, n_steps+1) with variance paths
    """
    # Unpack parameters
    H, eta, rho, xi0, kappa, theta, S0, r = params_array
    
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    sqrt_1_rho2 = np.sqrt(1.0 - rho**2)
    
    # Pre-compute weights (shared across paths)
    weights = compute_weights(n_steps, H, dt)
    
    # Allocate output
    S_paths = np.zeros((n_paths, n_steps + 1))
    v_paths = np.zeros((n_paths, n_steps + 1))
    dW_v_paths = np.zeros((n_paths, n_steps))
    
    # Set initial conditions
    S_paths[:, 0] = S0
    v_paths[:, 0] = xi0
    
    # Set random seed
    np.random.seed(seed)
    
    # Generate all random numbers upfront
    # We need n_steps brownian increments. 
    dW_S = np.random.randn(n_paths, n_steps) * sqrt_dt
    dW_perp = np.random.randn(n_paths, n_steps) * sqrt_dt
    
    # Simulate paths in parallel
    for path_idx in prange(n_paths):
        v = xi0
        S = S0
        
        # History of diffusion terms for convolution
        # We need to store 'sigma(v) * dW' terms or similar?
        # The spec says:
        # diffusion = eta * sqrt(v) * dW_v
        # weighted_sum += weights[i, j] * diffusion
        # So we need to store diffusion history.
        
        diffusion_history = np.zeros(n_steps)
        
        for i in range(n_steps):
            # i is current step index (0 to n_steps-1)
            # We are computing v_{i+1} and S_{i+1}
            
            # Current v is v_{i} (which is v at start of this loop)
            # stored in v_paths[path_idx, i]
            
            # Correlated Brownian increment for variance
            # dW_v at step i
            dW_v = rho * dW_S[path_idx, i] + sqrt_1_rho2 * dW_perp[path_idx, i]
            dW_v_paths[path_idx, i] = dW_v
            
            # Variance update (fractional Riccati / Volterra)
            # v_{i+1} = xi0 + \int K ...
            # Discretized as:
            # v_{i+1} = xi0 + sum_{j=0}^{i} b_{i,j} * diffusion_j + drift
            # Wait, the weights matrix index is [i, j]. 
            # If we follow the spec code strictly:
            # for j in range(i + 1):
            #    weighted_sum += weights[i, j] * diffusion
            
            # Calculate diffusion term for current step i
            # diffusion_i = eta * sqrt(v_i) * dW_v_i
            diff_term = eta * np.sqrt(max(v, 1e-8)) * dW_v
            
            # Drift term at step i
            # kappa * (theta - v_i) * dt
            # We need to store this history too if we are doing full convolution
            # But wait, usually we just convolve the whole thing:
            # dX_t = kappa(theta-v)dt + eta sqrt(v)dW
            # v_t = v_0 + \int K(t-s) dX_s
            
            # So: flux_j = kappa*(theta - v_paths[p,j])*dt + eta*sqrt(v_paths[p,j])*dW_v
            # And v_{i+1} = xi0 + sum_{j} b_{i,j} * flux_j
            
            drift_now = kappa * (theta - v) * dt
            flux_now = drift_now + diff_term
            
            # Store flux history instead of just diffusion history?
            # Or store drift history separately.
            # Let's reuse diffusion_history array to store "Total Flux" if we can,
            # but we called it diffusion_history. Let's rename or misuse it.
            # Actually, let's just create a flux history.
            
            # Optimization: We can't allocate new array inside loop efficiently?
            # We should allocate `flux_history` outside.
            # But for now, let's just modify `diffusion_history` to be `flux_history`.
            # diffusion_history[i] = flux_now
            
            # Re-allocating inside loop is bad but let's assume we change variable purpose
            # diffusion_history was allocated as zeros(n_steps).
            diffusion_history[i] = flux_now
            
            # Convolution
            weighted_sum = 0.0
            for j in range(i + 1):
                weighted_sum += weights[i, j] * diffusion_history[j]
            
            v_new = xi0 + weighted_sum
            
            # Note: The provided spec is "Exact Cholesky" in Module 1 text, 
            # but then "Implementation Spec" in Module 1.2 code.
            # I am implementing the "Implementation Spec" code.
            
            v_new = max(v_new, 1e-8)  # Absorbing barrier at 0
            
            # Stock price update (Euler-Maruyama)
            # Use v (start of step) to avoid lookahead bias and restore martingale property
            S_new = S * np.exp(
                (r - 0.5 * v) * dt + np.sqrt(v) * dW_S[path_idx, i]
            )
            
            S_paths[path_idx, i + 1] = S_new
            v_paths[path_idx, i + 1] = v_new
            
            v = v_new
            S = S_new
    
    return S_paths, v_paths, dW_v_paths


class RoughHestonSimulator:
    """High-level interface for rough Heston simulation"""
    
    def __init__(self, params: RoughHestonParams):
        self.params = params
        self._validate_params()
    
    def _validate_params(self):
        """Additional Feller condition checks"""
        feller = 2 * self.params.kappa * self.params.theta
        vol_squared = self.params.eta**2
        
        if feller < vol_squared:
            import warnings
            warnings.warn(
                f"Feller condition violated: 2κθ={feller:.4f} < η²={vol_squared:.4f}. "
                "Variance may hit zero."
            )
    
    def simulate(
        self, 
        T: float, 
        n_steps: int, 
        n_paths: int, 
        return_variance: bool = False,
        return_noise: bool = False,
        seed: int = None
    ) -> np.ndarray:
        """
        Simulate rough Heston paths
        
        Args:
            T: Time horizon (years)
            n_steps: Number of time steps
            n_paths: Number of Monte Carlo paths
            return_variance: If True, return (S_paths, v_paths), else just S_paths
            return_noise: If True (requires return_variance), return (S, v, dW_v)
            seed: Random seed for reproducibility
        
        Returns:
            S_paths: Shape (n_paths, n_steps+1) if return_variance=False
            (S_paths, v_paths): Both shape (n_paths, n_steps+1) if return_variance=True.
            (S_paths, v_paths, dW_v): If return_noise=True.
        """
        params_array = np.array([
            self.params.H,
            self.params.eta,
            self.params.rho,
            self.params.xi0,
            self.params.kappa,
            self.params.theta,
            self.params.S0,
            self.params.r
        ], dtype=np.float64)
        
        if seed is None:
            seed = np.random.randint(0, 2**31)
        
        S_paths, v_paths, dW_v = simulate_paths_rough_heston(
            params_array, T, n_steps, n_paths, seed
        )
        
        if return_noise:
            if not return_variance:
                import warnings
                warnings.warn("return_noise=True requires return_variance=True. Returning noise anyway.")
            return S_paths, v_paths, dW_v
        
        if return_variance:
            return S_paths, v_paths
        return S_paths
    
    def price_european_call(
        self, 
        K: float, 
        T: float, 
        n_paths: int = 100000,
        n_steps: int = 100
    ) -> Dict[str, float]:
        """
        Price European call option via Monte Carlo
        
        Returns:
            dict with keys: 'price', 'stderr', 'CI_lower', 'CI_upper'
        """
        S_paths = self.simulate(T, n_steps, n_paths)
        S_T = S_paths[:, -1]
        
        # Discounted payoff
        payoffs = np.maximum(S_T - K, 0) * np.exp(-self.params.r * T)
        
        price = np.mean(payoffs)
        stderr = np.std(payoffs) / np.sqrt(n_paths)
        
        return {
            'price': price,
            'stderr': stderr,
            'CI_lower': price - 1.96 * stderr,
            'CI_upper': price + 1.96 * stderr
        }
