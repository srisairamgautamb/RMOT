
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

from src.simulation.rough_heston import RoughHestonParams, RoughHestonSimulator
from src.pricing.rmot_solver import RMOTPricingEngine

# Assume Fisher is available
from src.calibration.fisher_information import FisherInformationAnalyzer, MalliavinEngine

@dataclass
class FRTBPosition:
    """Represents a trading book position"""
    position_id: str
    notional: float
    strike: float
    maturity: float
    option_type: str  # 'call' or 'put'
    is_liquid: bool

class FRTBComplianceEngine:
    """
    FRTB Non-Modelable Risk Factor (NMRF) valuation engine
    """
    
    def __init__(
        self,
        rmot_engine: RMOTPricingEngine,
        fisher_analyzer: FisherInformationAnalyzer
    ):
        self.rmot_engine = rmot_engine
        self.fisher_analyzer = fisher_analyzer
    
    def validate_data_sufficiency(
        self,
        liquid_strikes: np.ndarray,
        T: float
    ) -> Dict:
        """
        Step 1: Validate data sufficiency (Algorithm 1, Line 5-6)
        """
        validation = self.fisher_analyzer.validate_identifiability(
            liquid_strikes, T
        )
        
        if validation['n_strikes'] < 50:
            return {
                'status': 'WARNING',
                'message': validation['recommendation'],
                'proceed': True,  # Can proceed with warning
                'validation': validation
            }
        
        return {
            'status': 'PASS',
            'message': 'Sufficient data for calibration',
            'proceed': True,
            'validation': validation
        }
    
    def compute_nmrf_bounds(
        self,
        position: FRTBPosition,
        liquid_strikes: np.ndarray,
        liquid_prices: np.ndarray
    ) -> Dict:
        """
        Step 2-3: Compute RMOT bounds with error quantification
        """
        # Find nearest liquid price for reference
        idx = np.argmin(np.abs(liquid_strikes - position.strike))
        nearest_price = liquid_prices[idx]
        
        if position.is_liquid:
            # Liquid position - use market price (approximate by nearest liquid if exact match not found)
            # In production, we'd lookup exact. Here we approximate or assume liquid_strikes contains it.
            return {
                'lower_bound': nearest_price,
                'upper_bound': nearest_price,
                'mid_price': nearest_price,
                'is_nmrf': False
            }
        
        # Non-modelable position - use RMOT
        # Solve dual RMOT for the target strike
        rmot_result = self.rmot_engine.solve_dual_rmot(
            liquid_strikes,
            liquid_prices,
            position.maturity,
            target_strike=position.strike
        )
        
        mid_price = rmot_result['target_price']
        error_bound = rmot_result['error_bound']['bound']
        
        return {
            'lower_bound': mid_price - error_bound,
            'upper_bound': mid_price + error_bound,
            'mid_price': mid_price,
            'error_bound': error_bound,
            'error_components': rmot_result['error_bound'],
            'is_nmrf': True,
            'optimization_success': rmot_result['optimization_success']
        }
    
    def calculate_capital_charge(
        self,
        position: FRTBPosition,
        bounds: Dict
    ) -> Dict:
        """
        Step 4: Capital calculation (Algorithm 1, Line 17)
        
        Capital = max(|P_upper - P_mid|, |P_lower - P_mid|) × Notional
        """
        if not bounds['is_nmrf']:
            # Liquid position - minimal capital (or specific risk charge, here 0 for NMRF part)
            capital_charge = 0.0
        else:
            # NMRF position - conservative valuation
            # RMOT price is P_mid. Bound is symmetric [P_mid - err, P_mid + err].
            # So max deviation IS error_bound.
            
            error = bounds['error_bound']
            capital_charge = error * abs(position.notional) # Notional can be negative? usually magnitude.
            # Spec says "Notional".
        
        return {
            'capital_charge': capital_charge,
            'capital_charge_pct': capital_charge / abs(position.notional) if position.notional != 0 else 0,
            'position_id': position.position_id,
            'is_nmrf': bounds['is_nmrf']
        }
    
    def process_portfolio(
        self,
        positions: List[FRTBPosition],
        liquid_strikes: np.ndarray,
        liquid_prices: np.ndarray,
        T: float
    ) -> Dict:
        """
        Complete FRTB workflow for entire portfolio
        """
        # Step 1: Data validation
        data_validation = self.validate_data_sufficiency(liquid_strikes, T)
        
        if not data_validation['proceed']:
            return {
                'status': 'FAILED',
                'message': data_validation['message'],
                'results': None
            }
        
        # Step 2-4: Process each position
        results = []
        total_capital = 0.0
        total_notional = 0.0
        
        for position in positions:
            bounds = self.compute_nmrf_bounds(
                position, liquid_strikes, liquid_prices
            )
            
            capital = self.calculate_capital_charge(position, bounds)
            
            results.append({
                'position': position,
                'bounds': bounds,
                'capital': capital
            })
            
            total_capital += capital['capital_charge']
            total_notional += abs(position.notional)
        
        # Aggregate statistics
        nmrf_positions = [r for r in results if r['bounds']['is_nmrf']]
        
        return {
            'status': 'SUCCESS',
            'data_validation': data_validation,
            'total_capital_charge': total_capital,
            'total_notional': total_notional,
            'capital_ratio': total_capital / total_notional if total_notional > 0 else 0,
            'n_positions': len(positions),
            'n_nmrf': len(nmrf_positions),
            'results': results,
            'summary': self._generate_summary(results, total_capital, total_notional)
        }
    
    def _generate_summary(
        self,
        results: List[Dict],
        total_capital: float,
        total_notional: float
    ) -> str:
        """Generate human-readable summary"""
        n_nmrf = sum(1 for r in results if r['bounds']['is_nmrf'])
        
        summary = f"""
FRTB COMPLIANCE REPORT
======================
Total Positions: {len(results)}
Non-Modelable (NMRF): {n_nmrf}
Liquid: {len(results) - n_nmrf}

CAPITAL CALCULATION
-------------------
Total Notional: ${total_notional/1e9:.2f}B
Total Capital Charge: ${total_capital/1e6:.2f}M
Capital Ratio: {100*total_capital/total_notional:.2f}%
        """
        return summary.strip()

# Helper for integration
def create_sample_portfolio():
    return [
        FRTBPosition("POS_001", 1000000, 100.0, 0.5, 'call', True), # Liquid
        FRTBPosition("POS_002", 5000000, 120.0, 0.5, 'call', False), # NMRF
        FRTBPosition("POS_003", 2000000, 80.0, 0.5, 'put', False)   # NMRF
    ]
