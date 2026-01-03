# Multi-Asset RMOT Package
"""
Multi-Asset Rough Martingale Optimal Transport (RMOT)

Extends single-asset RMOT to handle:
- N correlated rough Heston assets
- Basket option pricing
- FRTB capital charge bounds

Reference: Multi_asset_RMOT.pdf
"""

from .data_structures import (
    RoughHestonParams,
    AssetConfig,
    MultiAssetConfig,
    MarginalCalibrationResult,
    CorrelationEstimationResult
)

__version__ = "1.0.0"
__all__ = [
    "RoughHestonParams",
    "AssetConfig", 
    "MultiAssetConfig",
    "MarginalCalibrationResult",
    "CorrelationEstimationResult"
]
