import numpy as np
from scipy.special import gamma, digamma
from typing import Dict, Tuple, Any
import copy
import pandas as pd
from src.sensitivity.malliavin import MalliavinEngine
# Import Simulator for numerical derivatives
from src.simulation.rough_heston import RoughHestonSimulator

def compute_malliavin_derivative_H(S_paths, v_paths, dW_v_paths, params, dt):
    """
    Compute Malliavin derivative D_H v_t using recursive formula
    formula: D_H v_n = η·Σ w_i^H √v_i ΔW_i + η·Σ K_i (D_H v_i)/(2√v_i) ΔW_i
    """
    H, eta = params.H, params.eta
    n_paths, n_steps = v_paths.shape
    # v_paths has shape (N, steps+1). So n_steps variable is actually steps+1.
    # We want actual steps.
    n_steps = n_steps - 1
    
    # Initialize Malliavin derivatives
    D_H_v = np.zeros_like(v_paths)
    D_H_S = np.zeros_like(S_paths)
    
    # Helper for kernel derivatives
    def kernel_derivative_H(n, i, H, dt):
        if i >= n: return 0.0, 0.0
        
        tau1 = (n - i) * dt
        tau2 = (n - i - 1) * dt if i < n-1 else 0.0
        
        if tau1 > 0:
            K1 = tau1**(H-0.5)
            dK1_dH = K1 * (np.log(tau1) - digamma(H+0.5))
        else: K1, dK1_dH = 0.0, 0.0
        
        if tau2 > 0:
            K2 = tau2**(H-0.5)
            dK2_dH = K2 * (np.log(tau2) - digamma(H+0.5))
        else: K2, dK2_dH = 0.0, 0.0
        
        denom = (H + 0.5) * np.sqrt(dt)
        K_i = (K1 - K2) / denom
        w_i = (dK1_dH - dK2_dH) / denom - K_i / (H + 0.5)
        
        return w_i, K_i
    
    # Recursive computation
    for n in range(1, n_steps):
        for i in range(n):
            w_i, K_i = kernel_derivative_H(n, i, H, dt)
            
            sqrt_v_i = np.sqrt(np.maximum(v_paths[:, i], 1e-8))
            term1 = eta * w_i * sqrt_v_i * dW_v_paths[:, i]
            
            if D_H_v[:, i].any():
                term2 = eta * K_i * (D_H_v[:, i] / (2 * sqrt_v_i)) * dW_v_paths[:, i]
            else:
                term2 = 0.0
            
            D_H_v[:, n] += term1 + term2
    
    # Spot Malliavin derivative
    for n in range(1, n_steps):
        integral_term = -0.5 * np.sum(D_H_v[:, :n], axis=1) * dt
        with np.errstate(divide='ignore', invalid='ignore'):
            term2 = np.sum(D_H_v[:, :n] / (2 * np.sqrt(v_paths[:, :n] + 1e-8)) * dW_v_paths[:, :n], axis=1)
        
        D_H_S[:, n] = S_paths[:, n] * (integral_term + term2)
    
    return D_H_v, D_H_S

def compute_malliavin_derivative_eta(S_paths, v_paths, dW_v_paths, params, dt):
    """Compute ∂v/∂η and ∂S/∂η"""
    H, eta = params.H, params.eta
    n_paths, n_steps = v_paths.shape
    n_steps = n_steps - 1
    D_eta_v = np.zeros_like(v_paths)
    D_eta_S = np.zeros_like(S_paths)
    
    for n in range(1, n_steps):
        for i in range(n):
            tau = (n - i) * dt
            if tau > 0:
                K_i = tau**(H-0.5) / ((H + 0.5) * np.sqrt(dt))
            else:
                K_i = 0.0
                
            sqrt_v_i = np.sqrt(np.maximum(v_paths[:, i], 1e-8))
            term1 = K_i * sqrt_v_i * dW_v_paths[:, i]
            
            if D_eta_v[:, i].any():
                term2 = eta * K_i * (D_eta_v[:, i] / (2 * sqrt_v_i)) * dW_v_paths[:, i]
            else:
                term2 = 0.0
                
            D_eta_v[:, n] += term1 + term2
            
    for n in range(1, n_steps):
        integral_term = -0.5 * np.sum(D_eta_v[:, :n], axis=1) * dt
        with np.errstate(divide='ignore'):
             term2 = np.sum(D_eta_v[:, :n] / (2 * np.sqrt(v_paths[:, :n] + 1e-8)) * dW_v_paths[:, :n], axis=1)
        D_eta_S[:, n] = S_paths[:, n] * (integral_term + term2)
        
    return D_eta_v, D_eta_S

def compute_malliavin_derivative_rho(S_paths, v_paths, dW_S_paths, dW_v_paths, params, dt):
    """Compute ∂S/∂ρ"""
    rho = params.rho
    n_paths, n_steps = S_paths.shape
    n_steps = n_steps - 1
    D_rho_S = np.zeros_like(S_paths)
    
    # Recover dZ
    denom = np.sqrt(1 - rho**2 + 1e-10)
    dZ = (dW_S_paths - rho * dW_v_paths) / denom
    
    for n in range(1, n_steps):
        d_dWS_drho = dW_v_paths[:, n] - (rho / denom) * dZ[:, n]
        sqrt_v = np.sqrt(np.maximum(v_paths[:, n], 1e-8))
        D_rho_S[:, n] = D_rho_S[:, n-1] + S_paths[:, n] * sqrt_v * d_dWS_drho
        
    return None, D_rho_S

def compute_option_sensitivity(strikes, S_T, D_theta_S_T, r, T):
    df = np.exp(-r * T)
    sens = np.zeros(len(strikes))
    for k, K in enumerate(strikes):
        indicator = (S_T > K).astype(float)
        sens[k] = np.mean(indicator * D_theta_S_T) * df
    return sens

class FisherInformationAnalyzer:
    """
    Compute and analyze Fisher Information for rough volatility calibration
    """
    
    def __init__(self, malliavin_engine=None):
        self.malliavin_engine = malliavin_engine
    
    def compute_fisher_matrix(
        self,
        strikes: np.ndarray,
        T: float,
        noise_std: float = 0.02,
        n_paths: int = 50000,
        params=None
    ) -> np.ndarray:
        if params is None and self.malliavin_engine:
            params = self.malliavin_engine.simulator.params
            
        if params is None:
            raise ValueError("Must provide params or initialized malliavin_engine")
            
        # print(f"Computing Fisher Information (N={n_paths})...")
        
        sim = RoughHestonSimulator(params)
        dt = T / 100
        S_paths, v_paths, dW_v_paths = sim.simulate(
            T=T, n_paths=n_paths, n_steps=100, return_noise=True
        )
        S_T = S_paths[:, -1]
        
        # Reconstruct dW_S from S_path
        # S_{i+1} = S_i * exp( (r - 0.5*v)*dt + sqrt(v)*dW_S )
        # log(S_{i+1}/S_i) = (r - 0.5*v)*dt + sqrt(v)*dW_S
        # dW_S = (log(S_{i+1}/S_i) - (r - 0.5*v)*dt) / sqrt(v)
        
        n_steps = 100
        dW_S_paths = np.zeros_like(dW_v_paths)
        
        for i in range(n_steps):
            S_i = S_paths[:, i]
            S_next = S_paths[:, i+1]
            v_i = np.maximum(v_paths[:, i], 1e-8)
            
            log_ret = np.log(S_next / S_i)
            drift = (params.r - 0.5 * v_i) * dt
            
            dW_S_paths[:, i] = (log_ret - drift) / np.sqrt(v_i)
        
        n_params = 5
        sensitivities = np.zeros((len(strikes), n_params))
        
        # 1. H
        _, D_H_S = compute_malliavin_derivative_H(S_paths, v_paths, dW_v_paths, params, dt)
        sensitivities[:, 0] = compute_option_sensitivity(strikes, S_T, D_H_S[:, -1], params.r, T)
        
        # 2. eta
        _, D_eta_S = compute_malliavin_derivative_eta(S_paths, v_paths, dW_v_paths, params, dt)
        sensitivities[:, 1] = compute_option_sensitivity(strikes, S_T, D_eta_S[:, -1], params.r, T)
        
        # 3. rho
        _, D_rho_S = compute_malliavin_derivative_rho(S_paths, v_paths, dW_S_paths, dW_v_paths, params, dt)
        sensitivities[:, 2] = compute_option_sensitivity(strikes, S_T, D_rho_S[:, -1], params.r, T)
        
        # 4. xi0 (Finite Diff)
        dtheta = 1e-5
        p_pert = copy.deepcopy(params)
        p_pert.xi0 += dtheta
        sim_pert = RoughHestonSimulator(p_pert)
        S_perturb = sim_pert.simulate(T, n_paths, 100, return_noise=False)[:, -1]
        base_prices = np.array([np.mean(np.maximum(S_T - K, 0)) for K in strikes])
        pert_prices = np.array([np.mean(np.maximum(S_perturb - K, 0)) for K in strikes])
        sensitivities[:, 3] = (pert_prices - base_prices) / dtheta
        
        # 5. kappa (Finite Diff)
        p_pert = copy.deepcopy(params)
        p_pert.kappa += dtheta
        sim_pert = RoughHestonSimulator(p_pert)
        S_perturb = sim_pert.simulate(T, n_paths, 100, return_noise=False)[:, -1]
        pert_prices = np.array([np.mean(np.maximum(S_perturb - K, 0)) for K in strikes])
        sensitivities[:, 4] = (pert_prices - base_prices) / dtheta
        
        # DF
        df = np.exp(-params.r * T)
        sensitivities[:, 3:] *= df
        
        # Fisher Info
        fisher = np.zeros((n_params, n_params))
        for k in range(len(strikes)):
            grad = sensitivities[k, :]
            fisher += np.outer(grad, grad) / noise_std**2
            
        fisher /= len(strikes)
        return fisher

    def compute_effective_dimension(self, fisher_matrix, threshold=0.01):
        eigenvalues, _ = np.linalg.eigh(fisher_matrix)
        eigenvalues = eigenvalues[::-1]
        if eigenvalues[0] > 1e-12:
            d_eff = int(np.sum(eigenvalues/eigenvalues[0] > threshold))
        else: d_eff = 0
        return d_eff, eigenvalues

    def cramèr_rao_bounds(self, fisher_matrix):
        try:
            cov = np.linalg.inv(fisher_matrix + 1e-10 * np.eye(5))
        except:
            cov = np.linalg.pinv(fisher_matrix)
        stds = np.sqrt(np.maximum(np.diag(cov), 0.0))
        return {
            'std_H': stds[0], 'std_eta': stds[1], 'std_rho': stds[2],
            'std_xi0': stds[3], 'std_kappa': stds[4]
        }

    def validate_identifiability(self, strikes, T, min_strikes_required=50, params=None, n_paths=50000):
        fisher = self.compute_fisher_matrix(strikes, T, params=params, n_paths=n_paths)
        d_eff, ev = self.compute_effective_dimension(fisher)
        cr = self.cramèr_rao_bounds(fisher)
        
        return {
            'n_strikes': len(strikes),
            'd_eff': d_eff,
            'eigenvalues': ev,
            'cramèr_rao_bounds': cr,
            'recommendation': "PASS" if d_eff >= 3 else "WARNING"
        }
