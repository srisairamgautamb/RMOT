
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from typing import Dict, Tuple, List, Optional
import warnings

from src.simulation.rough_heston import RoughHestonSimulator, RoughHestonParams

class RMOTPricingEngine:
    """
    Rough Volatility Martingale Optimal Transport Pricing
    """
    
    def __init__(
        self,
        rough_simulator: RoughHestonSimulator,
        lambda_reg: float = 0.01
    ):
        self.simulator = rough_simulator
        self.lambda_reg = lambda_reg
        self.prior_samples = None
        self.weights_prior = None
    
    def generate_prior_samples(
        self,
        T: float,
        n_samples: int = 100000,
        cache: bool = True
    ) -> np.ndarray:
        """
        Generate samples from rough volatility prior P_rough
        
        Args:
            T: Maturity
            n_samples: Number of Monte Carlo samples
            cache: If True, store samples for reuse
        
        Returns:
            S_T: Terminal prices of shape (n_samples,)
        """
        # Daily steps: T * 252
        n_steps = int(max(T * 252, 50)) 
        
        S_paths = self.simulator.simulate(
            T=T,
            n_steps=n_steps,
            n_paths=n_samples
        )
        S_T = S_paths[:, -1]
        
        if cache:
            self.prior_samples = S_T
            self.weights_prior = np.ones(n_samples) / n_samples
        
        return S_T
    
    def solve_dual_rmot(
        self,
        liquid_strikes: np.ndarray,
        liquid_prices: np.ndarray,
        T: float,
        target_strike: float = None,
        n_samples: int = 100000
    ) -> Dict:
        """
        Solve dual RMOT problem to find optimal measure P*_λ
        
        Args:
            liquid_strikes: Array of liquid strikes K_i
            liquid_prices: Array of market prices C(K_i)
            T: Maturity
            target_strike: Strike to price (if None, just calibrate)
            n_samples: Prior samples
        
        Returns:
            dict with 'multipliers', 'tilted_weights', 'target_price', 'error_bound'
        """
        # Generate or use cached prior samples
        if self.prior_samples is None or len(self.prior_samples) != n_samples:
            S_T = self.generate_prior_samples(T, n_samples)
        else:
            S_T = self.prior_samples
        
        # 1. Pre-Conditioning (Affine Scaling)
        S_0 = self.simulator.params.S0
        scale = S_0
        if scale < 1e-4: scale = 1.0 # Safety
        
        S_norm = S_T / scale
        K_norm = liquid_strikes / scale
        C_norm = liquid_prices / scale
        S0_norm = 1.0
        
        m = len(liquid_strikes)
        
        # Pre-compute payoff matrix for vectorization: (n_samples, m)
        # Memory usage: 100k * 50 * 8 bytes ≈ 40 MB (Acceptable)
        # S_norm[:, None] is (N, 1), K_norm[None, :] is (1, M)
        payoff_matrix = np.maximum(S_norm[:, None] - K_norm[None, :], 0)
        
        def compute_g_norm_vectorized(multipliers: np.ndarray) -> np.ndarray:
            Delta = multipliers[0]
            alphas = multipliers[1:]
            
            # g = Delta * S + sum(alpha * payoff)
            # (N,) + (N, M) @ (M,) -> (N,)
            term2 = payoff_matrix @ alphas
            return Delta * S_norm + term2
        
        def objective(multipliers: np.ndarray) -> float:
            g_vals = compute_g_norm_vectorized(multipliers)
            
            # Log-sum-exp trick
            log_weights_unnorm = -g_vals / self.lambda_reg
            
            # log(mean(exp(x)))
            log_expect = logsumexp(log_weights_unnorm) - np.log(len(S_norm))
            
            term1 = self.lambda_reg * log_expect
            
            Delta = multipliers[0]
            alphas = multipliers[1:]
            term2 = Delta * S0_norm + np.sum(alphas * C_norm)
            
            return term1 - term2 # Revert to original
        
        def gradient(multipliers: np.ndarray) -> np.ndarray:
            g_vals = compute_g_norm_vectorized(multipliers)
            log_weights_unnorm = -g_vals / self.lambda_reg
            
            max_val = np.max(log_weights_unnorm)
            weights = np.exp(log_weights_unnorm - max_val)
            weights /= np.sum(weights)
            
            grad = np.zeros(m + 1)
            
            # ∂/∂Δ = E_P* [S] - S0
            grad[0] = np.sum(weights * S_norm) - S0_norm
            
            # ∂/∂αᵢ = E_P* [(S-K)+] - C_market
            grad[1:] = (weights @ payoff_matrix) - C_norm
            
            return grad
            
        # Optimize in normalized space
        initial_multipliers = np.zeros(m + 1)
        
        # Increased maxiter and maxfun for robustness
        # Optimize in normalized space
        initial_multipliers = np.zeros(m + 1)
        
        result = minimize(
            objective,
            initial_multipliers,
            method='L-BFGS-B',
            jac=gradient,
            options={
                'maxiter': 10000, 
                'maxfun': 50000, 
                'ftol': 1e-5,  # Relaxed tolerance for MC noise
                'gtol': 1e-5
            }
        )
        
        if not result.success:
            print(f"Optimization Failed: {result.message}")
            
        optimal_multipliers = result.x
        
        # Check convergence
        final_grad = gradient(result.x)
        grad_norm = np.linalg.norm(final_grad)
        print(f"Final Grad Norm: {grad_norm:.4e}")
        
        if not result.success:
            print(f"Stage 2 Failed: {result.message}")
            
        optimal_multipliers = result.x
        
        # Check convergence
        final_grad = gradient(result.x)
        grad_norm = np.linalg.norm(final_grad)
        print(f"Final Grad Norm: {grad_norm:.4e}")
        
        # Compute tilted weights using optimal multipliers (normalized space)
        g_vals = compute_g_norm_vectorized(optimal_multipliers)
        log_weights_unnorm = -g_vals / self.lambda_reg
        max_val = np.max(log_weights_unnorm)
        tilted_weights = np.exp(log_weights_unnorm - max_val)
        tilted_weights /= np.sum(tilted_weights)
        
        # Price target strike (re-scale output)
        target_price = None
        error_bound = None
        
        if target_strike is not None:
            # Scale target strike too
            K_target_norm = target_strike / scale
            target_payoff_norm = np.maximum(S_norm - K_target_norm, 0)
            
            # Price in normalized units
            price_norm = np.sum(tilted_weights * target_payoff_norm)
            
            # Scale back
            target_price = price_norm * scale * np.exp(-self.simulator.params.r * T)
            
            error_bound = self.compute_error_bound(
                target_strike, T, optimal_multipliers
            )
        
        # Validation: Check constraints (use original units)
        calibration_error = self._check_calibration(
            liquid_strikes, liquid_prices, S_T, tilted_weights, T
        )
        
        return {
            'multipliers': optimal_multipliers, # Normalized multipliers
            'tilted_weights': tilted_weights,
            'target_price': target_price,
            'error_bound': error_bound,
            'calibration_error': calibration_error,
            'optimization_success': result.success
        }
    
    def compute_error_bound(
        self,
        K: float,
        T: float,
        multipliers: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute extrapolation error bound (Theorem 3.8)
        
        Error ≤ (√(2C/λ) + √(2η)) · S₀ exp(k) · exp(-I(k)/(2T^(2H)))
        """
        S_0 = self.simulator.params.S0
        H = self.simulator.params.H
        k = np.log(K / S_0)
        
        # Large deviations rate function I(k) for rough vol
        # Approximation: I(k) ~ k^2 / (2 * V) ?
        # Spec says: "I(k) ≈ c · k^(1/H) for large k"
        # "c = 5.0 # Calibrated constant for H=0.1"
        c = 5.0
        I_k = c * (np.abs(k) ** (1.0 / H)) if np.abs(k) > 1e-6 else 0.0
        
        # KL divergence bound: D_KL(P*_λ || P_rough) ≤ C/λ
        # C is usually assumed or estimated. Spec uses "C_KL = 1.0"
        C_KL = 1.0
        
        # Model error
        eta_model = 0.1
        
        tilt_error_factor = np.sqrt(2 * C_KL / self.lambda_reg)
        model_error_factor = np.sqrt(2 * eta_model)
        
        # Exponential decay term
        # exp(k - I_k / (2 * T^(2H)))
        # Note: k can be negative for OTM put or ITM call?
        # Usually k is log-moneyness.
        # Check sign in spec: "exp(k)". If k is positive (OTM Call K>S0).
        # Decay is driven by I_k.
        
        exponent = k - I_k / (2 * T**(2 * H))
        exponential_decay = np.exp(exponent)
        
        # Bound
        bound = (tilt_error_factor + model_error_factor) * S_0 * exponential_decay
        
        return {
            'bound': bound,
            'k': k,
            'I_k': I_k,
            'exponential_decay': exponential_decay
        }
    
    def _check_calibration(
        self,
        strikes: np.ndarray,
        target_prices: np.ndarray,
        S_T: np.ndarray,
        weights: np.ndarray,
        T: float
    ) -> Dict[str, float]:
        """Verify calibration constraints are satisfied"""
        computed_prices = np.zeros(len(strikes))
        r = self.simulator.params.r
        df = np.exp(-r * T)
        
        for i, K in enumerate(strikes):
            payoff = np.maximum(S_T - K, 0)
            computed_prices[i] = np.sum(weights * payoff) * df
        
        abs_errors = np.abs(computed_prices - target_prices)
        rel_errors = abs_errors / (target_prices + 1e-10)
        
        return {
            'max_abs_error': np.max(abs_errors),
            'max_rel_error': np.max(rel_errors),
            'rmse': np.sqrt(np.mean(abs_errors**2))
        }


def price_with_rmot(
    liquid_strikes: np.ndarray,
    liquid_prices: np.ndarray,
    target_strike: float,
    T: float,
    rough_params: RoughHestonParams,
    lambda_reg: float = 0.01,
    n_samples: int = 100000
) -> Dict:
    """
    Price a deep OTM option using RMOT with rough volatility prior
    """
    simulator = RoughHestonSimulator(rough_params)
    rmot_engine = RMOTPricingEngine(simulator, lambda_reg)
    
    result = rmot_engine.solve_dual_rmot(
        liquid_strikes, liquid_prices, T, target_strike, n_samples
    )
    
    return {
        'price': result['target_price'],
        'error_bound': result['error_bound']['bound'],
        'confidence_interval': [
            result['target_price'] - result['error_bound']['bound'],
            result['target_price'] + result['error_bound']['bound']
        ],
        'calibration_quality': result['calibration_error']
    }
