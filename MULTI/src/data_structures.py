"""
Core Data Structures for Multi-Asset RMOT

Reference: MULTI_ASSET_RMOT_ENGINEERING_SPEC Section 1.2
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class RoughHestonParams:
    """
    Rough Heston parameters for a single asset.
    
    Reference: PDF Equations (1)-(2)
    
    Model:
        dS_t/S_t = √(ν_t) dW_t^S
        ν_t = ξ_0(t) + ∫_0^t K_H(t-s) [κ(θ-ν_s)ds + η√(ν_s)dW_s^V]
    
    Where K_H(t) = t^(H-1/2) / Γ(H+1/2) is the fractional kernel.
    """
    H: float           # Hurst exponent ∈ (0, 0.5) - roughness parameter
    eta: float         # Vol-of-vol (small, ~0.1-0.2 per Assumption 2.3)
    rho: float         # Spot-vol correlation ∈ (-1, 1)
    xi0: float         # Initial/forward variance
    kappa: float       # Mean reversion speed
    theta: float       # Long-term variance
    spot: float        # S_0 (initial spot price)
    maturity: float    # T (option maturity)
    r: float = 0.045   # Risk-free rate
    
    def validate(self) -> None:
        """
        Enforce mathematical constraints.
        
        Critical constraints:
        - H ∈ (0, 0.5): Rough regime (smoother for H > 0.5)
        - η < 0.3: Small vol-of-vol (Assumption 2.3 for Ψ_ij expansion)
        - ρ ∈ [-1, 1]: Valid correlation
        """
        if not (0 < self.H < 0.5):
            raise ValueError(f"Hurst must be in (0, 0.5), got {self.H}")
        if not (0 < self.eta < 0.5):
            raise ValueError(f"Vol-of-vol too large: {self.eta}, need η < 0.5")
        if not (-1 <= self.rho <= 1):
            raise ValueError(f"Correlation out of bounds: {self.rho}")
        if self.xi0 <= 0:
            raise ValueError(f"Initial variance must be positive: {self.xi0}")
        if self.spot <= 0:
            raise ValueError(f"Spot price must be positive: {self.spot}")
        if self.maturity <= 0:
            raise ValueError(f"Maturity must be positive: {self.maturity}")
    
    def to_dict(self) -> dict:
        return {
            'H': self.H, 'eta': self.eta, 'rho': self.rho,
            'xi0': self.xi0, 'kappa': self.kappa, 'theta': self.theta,
            'spot': self.spot, 'maturity': self.maturity, 'r': self.r
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'RoughHestonParams':
        return cls(**d)


@dataclass
class AssetConfig:
    """Configuration for a single asset's market data."""
    ticker: str                    # 'SPX', 'NDX', etc.
    spot: float                    # S_0^i
    strikes: np.ndarray            # K ∈ [0.8*S_0, 1.2*S_0]
    market_prices: np.ndarray      # Observed call prices C_i(K)
    price_errors: Optional[np.ndarray] = None  # σ_i for each observation
    maturity: float = 1/12         # T (default 1 month)
    
    def __post_init__(self):
        self.strikes = np.asarray(self.strikes)
        self.market_prices = np.asarray(self.market_prices)
        if self.price_errors is None:
            # Default: 1% of mid-price as error
            self.price_errors = 0.01 * np.abs(self.market_prices)
        else:
            self.price_errors = np.asarray(self.price_errors)
        
        # Validate
        assert len(self.strikes) == len(self.market_prices), \
            f"Strikes ({len(self.strikes)}) and prices ({len(self.market_prices)}) must match"


@dataclass
class MultiAssetConfig:
    """Multi-asset basket configuration."""
    assets: List[AssetConfig]      # N assets
    basket_weights: np.ndarray     # w = [w_1, ..., w_N], sum to 1
    correlation_guess: np.ndarray  # Initial ρ (e.g., historical)
    basket_strikes: np.ndarray     # K for basket options
    
    def __post_init__(self):
        self.basket_weights = np.asarray(self.basket_weights)
        self.correlation_guess = np.asarray(self.correlation_guess)
        self.basket_strikes = np.asarray(self.basket_strikes)
        
        N = len(self.assets)
        
        # Validate weights
        if len(self.basket_weights) != N:
            raise ValueError(f"Weights ({len(self.basket_weights)}) must match assets ({N})")
        if not np.isclose(self.basket_weights.sum(), 1.0):
            raise ValueError(f"Weights must sum to 1, got {self.basket_weights.sum()}")
        
        # Validate correlation matrix
        if self.correlation_guess.shape != (N, N):
            raise ValueError(f"Correlation matrix must be {N}×{N}")
        if not np.allclose(self.correlation_guess, self.correlation_guess.T):
            raise ValueError("Correlation matrix must be symmetric")
        if not np.allclose(np.diag(self.correlation_guess), 1.0):
            raise ValueError("Correlation matrix diagonal must be 1")
    
    @property
    def n_assets(self) -> int:
        return len(self.assets)
    
    @property
    def maturity(self) -> float:
        """Common maturity (assuming all assets have same T)."""
        return self.assets[0].maturity


@dataclass
class MarginalCalibrationResult:
    """Complete result from Phase 1: Marginal Calibration."""
    params: List[RoughHestonParams]    # N parameter sets
    calibration_errors: np.ndarray     # RMSE for each asset
    
    def __post_init__(self):
        self.calibration_errors = np.asarray(self.calibration_errors)
    
    def verify_distinct_hurst(self, tolerance: float = 0.02) -> None:
        """
        Validates Assumption 2.1: H_i ≠ H_j
        
        Critical for identifiability (Theorem 3.1).
        If H_i ≈ H_j, the correlation estimation becomes ill-conditioned.
        
        Args:
            tolerance: Minimum |H_i - H_j| required
        
        Raises:
            ValueError if any pair is too close
        """
        H_values = [p.H for p in self.params]
        N = len(H_values)
        
        for i in range(N):
            for j in range(i + 1, N):
                if abs(H_values[i] - H_values[j]) < tolerance:
                    raise ValueError(
                        f"Assets {i} and {j} have nearly identical Hurst: "
                        f"H_{i}={H_values[i]:.4f}, H_{j}={H_values[j]:.4f}. "
                        f"Correlation estimation will be ill-conditioned! "
                        f"(Need |H_i - H_j| ≥ {tolerance})"
                    )
    
    def get_hurst_values(self) -> np.ndarray:
        return np.array([p.H for p in self.params])
    
    def summary(self) -> str:
        lines = ["Marginal Calibration Results:"]
        for i, (p, err) in enumerate(zip(self.params, self.calibration_errors)):
            lines.append(f"  Asset {i}: H={p.H:.4f}, η={p.eta:.4f}, ρ={p.rho:.4f}, RMSE={err:.4f}")
        return "\n".join(lines)


@dataclass
class CorrelationEstimationResult:
    """Output of Phase 2: Correlation Optimization."""
    rho: np.ndarray                    # N×N correlation matrix
    fisher_information: np.ndarray     # I(ρ) for Cramér-Rao bounds
    cramer_rao_stds: np.ndarray       # Lower bounds on std(ρ_ij)
    optimization_history: List[dict] = field(default_factory=list)
    lambda_dual: Optional[np.ndarray] = None  # Optimal dual variables
    converged: bool = True
    n_iterations: int = 0
    
    def validate(self) -> None:
        """Ensure ρ is a valid correlation matrix."""
        N = self.rho.shape[0]
        
        # Symmetric
        if not np.allclose(self.rho, self.rho.T):
            raise ValueError("Correlation matrix is not symmetric")
        
        # Diagonal is 1
        if not np.allclose(np.diag(self.rho), 1.0):
            raise ValueError("Correlation matrix diagonal must be 1")
        
        # Positive semi-definite
        eigvals = np.linalg.eigvalsh(self.rho)
        if np.any(eigvals < -1e-8):
            raise ValueError(f"Correlation matrix is not PSD: min eigenvalue = {eigvals.min()}")
    
    def get_correlation(self, i: int, j: int) -> float:
        """Get ρ_ij with Cramér-Rao bound."""
        return self.rho[i, j]
    
    def summary(self) -> str:
        lines = [
            "Correlation Estimation Results:",
            f"  Converged: {self.converged} ({self.n_iterations} iterations)",
            f"  Correlation Matrix:\n{self.rho}",
            f"  Cramér-Rao Std Bounds: {self.cramer_rao_stds}"
        ]
        return "\n".join(lines)


@dataclass
class BasketPricingResult:
    """Output of Phase 3: Basket Pricing."""
    strike: float
    price: float
    std_error: float
    delta: np.ndarray  # ∂P/∂S_0^i for each asset
    gamma: Optional[np.ndarray] = None


@dataclass 
class FRTBBoundsResult:
    """Output of Phase 4: FRTB Bounds."""
    P_low: float           # Lower price bound
    P_up: float            # Upper price bound
    width: float           # P_up - P_low
    mid_price: float       # (P_low + P_up) / 2
    capital_charge: float  # Max deviation from mid
    H_eff: float           # Effective Hurst (min H_i)
    log_moneyness: float   # k = log(K / S_basket)
    I_basket: float        # Rate function value
    scaling: float         # T^(2H_eff)


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def project_to_correlation_matrix(M: np.ndarray) -> np.ndarray:
    """
    Project arbitrary matrix to nearest valid correlation matrix.
    
    Method: Alternating projections (Higham 2002)
    
    Ensures:
    1. Symmetry
    2. Unit diagonal
    3. Positive semi-definiteness
    """
    # Ensure symmetry
    M = 0.5 * (M + M.T)
    
    # Set diagonal to 1
    np.fill_diagonal(M, 1.0)
    
    # Project to positive semi-definite cone
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0)  # Clip negative eigenvalues
    M = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Re-normalize diagonal to 1
    D = np.diag(1.0 / np.sqrt(np.maximum(np.diag(M), 1e-10)))
    M = D @ M @ D
    
    # Final symmetry enforcement
    M = 0.5 * (M + M.T)
    np.fill_diagonal(M, 1.0)
    
    return M


def create_synthetic_config(
    n_assets: int = 2,
    n_strikes: int = 20,
    maturity: float = 1/12
) -> MultiAssetConfig:
    """Create synthetic market data for testing."""
    
    assets = []
    spots = [4500, 15000, 3000, 2000, 500][:n_assets]  # SPX, NDX, RUT, etc.
    
    for i in range(n_assets):
        spot = spots[i]
        strikes = np.linspace(0.85 * spot, 1.15 * spot, n_strikes)
        
        # Synthetic Black-Scholes prices (for testing)
        from scipy.stats import norm
        sigma = 0.15 + 0.05 * i  # Different IV per asset
        d1 = (np.log(spot / strikes) + 0.5 * sigma**2 * maturity) / (sigma * np.sqrt(maturity))
        d2 = d1 - sigma * np.sqrt(maturity)
        prices = spot * norm.cdf(d1) - strikes * np.exp(-0.045 * maturity) * norm.cdf(d2)
        
        assets.append(AssetConfig(
            ticker=f"ASSET{i}",
            spot=spot,
            strikes=strikes,
            market_prices=prices,
            maturity=maturity
        ))
    
    # Basket weights (equal weighted)
    weights = np.ones(n_assets) / n_assets
    
    # Initial correlation guess (identity-like with some structure)
    rho_guess = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            rho_guess[i, j] = rho_guess[j, i] = 0.5 * np.exp(-0.5 * abs(i - j))
    
    # Basket strikes
    basket_spot = np.sum(weights * np.array(spots[:n_assets]))
    basket_strikes = np.linspace(0.9 * basket_spot, 1.1 * basket_spot, 5)
    
    return MultiAssetConfig(
        assets=assets,
        basket_weights=weights,
        correlation_guess=rho_guess,
        basket_strikes=basket_strikes
    )
