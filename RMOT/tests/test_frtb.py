
import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.simulation.rough_heston import RoughHestonParams, RoughHestonSimulator
from src.pricing.rmot_solver import RMOTPricingEngine
from src.sensitivity.malliavin import MalliavinEngine
from src.calibration.fisher_information import FisherInformationAnalyzer
from src.frtb.compliance import FRTBComplianceEngine, FRTBPosition, create_sample_portfolio

def test_frtb_portfolio_processing():
    """Test full FRTB workflow on sample portfolio"""
    # Setup Engines
    params = RoughHestonParams(
        H=0.1, eta=1.0, rho=-0.5, xi0=0.04, 
        kappa=1.0, theta=0.04, S0=100.0, r=0.0
    )
    sim = RoughHestonSimulator(params)
    rmot = RMOTPricingEngine(sim, lambda_reg=0.01)
    malliavin = MalliavinEngine(sim)
    fisher = FisherInformationAnalyzer(malliavin)
    
    frtb = FRTBComplianceEngine(rmot, fisher)
    
    # Mock Market Data (Liquid Strikes)
    liquid_strikes = np.linspace(90, 110, 21) # 21 strikes
    # Dummy prices: C = max(S-K, 0) approx
    liquid_prices = np.maximum(100.0 - liquid_strikes, 0) + 2.0 # Add some premium
    T = 0.5
    
    # Portfolio
    positions = create_sample_portfolio()
    # Ensure maturities match T for test simplicity
    for p in positions:
        p.maturity = T
    
    # Run
    # Use small n_samples for speed
    # Note: simulate() caches samples in rmot_engine if T is same.
    # We need to make sure rmot generates samples.
    
    # Hack: Inject samples manually or just run (it calls generate_prior_samples)
    
    report = frtb.process_portfolio(positions, liquid_strikes, liquid_prices, T)
    
    print(report['summary'])
    
    assert report['status'] == 'SUCCESS'
    assert report['total_notional'] > 0
    # POS_001 is liquid, Capital 0?
    assert report['results'][0]['capital']['capital_charge'] == 0.0
    # POS_002 is NMRF, Capital > 0
    assert report['results'][1]['capital']['capital_charge'] > 0.0

def test_data_sufficiency_warning():
    """Test that low strike count returns warning"""
    params = RoughHestonParams(H=0.1, eta=1.0, rho=-0.5, xi0=0.04, kappa=1.0, theta=0.04, S0=100.0, r=0.0)
    sim = RoughHestonSimulator(params)
    rmot = RMOTPricingEngine(sim)
    fisher = FisherInformationAnalyzer(MalliavinEngine(sim))
    frtb = FRTBComplianceEngine(rmot, fisher)
    
    liquid_strikes = np.linspace(90, 110, 10) # 10 strikes (Low)
    T = 0.5
    
    val = frtb.validate_data_sufficiency(liquid_strikes, T)
    assert val['status'] == 'WARNING'
