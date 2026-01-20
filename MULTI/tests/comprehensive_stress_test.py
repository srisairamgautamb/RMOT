#!/usr/bin/env python3
"""
COMPREHENSIVE SYSTEM STRESS TEST
Multi-Asset RMOT - Full End-to-End Validation

This script tests the ENTIRE system holistically:
1. All modules working together
2. Stress tests under extreme conditions
3. Benchmark comparisons
4. Publication-quality validation
"""

import numpy as np
import time
import sys
from datetime import datetime
from typing import Dict, List, Tuple

# Add path
sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/RMOT/RMOT/MULTI')

from src.data_structures import (
    RoughHestonParams, AssetConfig, MultiAssetConfig,
    MarginalCalibrationResult, CorrelationEstimationResult,
    project_to_correlation_matrix
)
from src.psi_functional import compute_psi_functional
from src.psi_functional_gauss_jacobi import compute_psi_functional_gauss_jacobi
from src.correlation_copula import RoughMartingaleCopula, test_copula_correlation
from src.path_simulation import simulate_correlated_rough_heston
from src.basket_pricing import price_basket_call, price_multiple_strikes
from src.frtb_bounds import compute_frtb_bounds, compute_frtb_bounds_multiple
from src.pipeline import multi_asset_rmot_pipeline
from src.real_time_data import RealTimeDataStream
from src.monitoring import ResearchMonitor, PipelineMetrics


class ComprehensiveTestSuite:
    """Comprehensive stress testing of entire system."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.n_passed = 0
        self.n_failed = 0
        self.n_total = 0
    
    def run_all(self):
        """Run all comprehensive tests."""
        print("\n" + "=" * 80)
        print("🔬 COMPREHENSIVE MULTI-ASSET RMOT STRESS TEST")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Testing: ALL COMPONENTS HOLISTICALLY")
        
        # 1. Data Structures Validation
        self._test_data_structures()
        
        # 2. Ψ_ij Functional Tests
        self._test_psi_functional()
        
        # 3. Correlation Copula Tests
        self._test_correlation_copula()
        
        # 4. Path Simulation Tests
        self._test_path_simulation()
        
        # 5. Basket Pricing Tests
        self._test_basket_pricing()
        
        # 6. FRTB Bounds Tests
        self._test_frtb_bounds()
        
        # 7. Full Pipeline Tests
        self._test_full_pipeline()
        
        # 8. Real Market Data Tests
        self._test_real_market_data()
        
        # 9. Stress Tests
        self._test_stress_conditions()
        
        # 10. Performance Benchmarks
        self._test_performance()
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _record(self, name: str, passed: bool, details: str = ""):
        """Record test result."""
        self.n_total += 1
        if passed:
            self.n_passed += 1
            status = "✅ PASS"
        else:
            self.n_failed += 1
            status = "❌ FAIL"
        
        self.results[name] = {
            'passed': passed,
            'details': details
        }
        print(f"  {name}: {status}" + (f" ({details})" if details else ""))
    
    # ═══════════════════════════════════════════════════════════════════
    # TEST CATEGORIES
    # ═══════════════════════════════════════════════════════════════════
    
    def _test_data_structures(self):
        """Test data structure validation."""
        print(f"\n{'─'*70}")
        print("1. DATA STRUCTURES VALIDATION")
        print(f"{'─'*70}")
        
        # Test RoughHestonParams validation
        try:
            params = RoughHestonParams(
                H=0.10, eta=0.15, rho=-0.7, xi0=0.04,
                kappa=2.0, theta=0.04, spot=100.0, maturity=1/12
            )
            params.validate()
            self._record("RoughHestonParams validation", True)
        except Exception as e:
            self._record("RoughHestonParams validation", False, str(e))
        
        # Test invalid H rejection
        try:
            invalid = RoughHestonParams(H=0.6, eta=0.15, rho=-0.7, xi0=0.04,
                                        kappa=2.0, theta=0.04, spot=100.0, maturity=1/12)
            invalid.validate()
            self._record("Invalid H rejection (H=0.6)", False, "Should have raised")
        except ValueError:
            self._record("Invalid H rejection (H=0.6)", True)
        
        # Test correlation projection
        non_psd = np.array([[1.0, 1.2], [1.2, 1.0]])  # Invalid
        projected = project_to_correlation_matrix(non_psd)
        eigvals = np.linalg.eigvalsh(projected)
        self._record("Non-PSD projection", eigvals.min() >= -1e-10, f"min_eig={eigvals.min():.2e}")
        
        # Test MarginalCalibrationResult Hurst check
        try:
            params1 = RoughHestonParams(H=0.10, eta=0.15, rho=-0.7, xi0=0.04,
                                        kappa=2.0, theta=0.04, spot=100.0, maturity=1/12)
            params2 = RoughHestonParams(H=0.10, eta=0.15, rho=-0.7, xi0=0.04,
                                        kappa=2.0, theta=0.04, spot=100.0, maturity=1/12)
            result = MarginalCalibrationResult(params=[params1, params2], calibration_errors=np.array([0.01, 0.01]))
            result.verify_distinct_hurst()
            self._record("Identical Hurst rejection", False, "Should have raised")
        except ValueError:
            self._record("Identical Hurst rejection", True)
    
    def _test_psi_functional(self):
        """Test Ψ_ij functional."""
        print(f"\n{'─'*70}")
        print("2. Ψ_ij FUNCTIONAL TESTS")
        print(f"{'─'*70}")
        
        params_1 = RoughHestonParams(H=0.10, eta=0.15, rho=-0.7, xi0=0.04,
                                     kappa=2.0, theta=0.04, spot=100.0, maturity=1/12)
        params_2 = RoughHestonParams(H=0.15, eta=0.18, rho=-0.5, xi0=0.05,
                                     kappa=1.5, theta=0.05, spot=100.0, maturity=1/12)
        
        # Symmetry test
        psi_12 = compute_psi_functional(1.0, 1.0, params_1, params_2)
        psi_21 = compute_psi_functional(1.0, 1.0, params_2, params_1)
        sym_error = abs(psi_12 - psi_21)
        self._record("Ψ_ij symmetry", sym_error < 1e-10, f"|Ψ₁₂-Ψ₂₁|={sym_error:.2e}")
        
        # Linearity test
        psi_1 = compute_psi_functional(1.0, 1.0, params_1, params_1)
        psi_2 = compute_psi_functional(2.0, 1.0, params_1, params_1)
        lin_error = abs(psi_2 - 2*psi_1)
        self._record("Ψ_ij linearity", lin_error < 1e-10, f"|Ψ(2u,v)-2Ψ(u,v)|={lin_error:.2e}")
        
        # Bilinearity test
        psi_a = compute_psi_functional(1.0, 1.0, params_1, params_1)
        psi_b = compute_psi_functional(2.0, 1.0, params_1, params_1)
        psi_sum = compute_psi_functional(3.0, 1.0, params_1, params_1)
        bi_error = abs(psi_sum - (psi_a + psi_b))
        self._record("Ψ_ij bilinearity", bi_error < 1e-10, f"error={bi_error:.2e}")
        
        # Gauss-Jacobi comparison
        psi_naive = compute_psi_functional(1.0, 1.0, params_1, params_2, n_grid=200)
        psi_gj = compute_psi_functional_gauss_jacobi(1.0, 1.0, params_1, params_2, n_points=32)
        rel_diff = abs(psi_gj - psi_naive) / max(abs(psi_naive), 1e-10)
        self._record("Gauss-Jacobi accuracy", rel_diff < 0.20, f"rel_diff={rel_diff:.2%}")
        
        # Gauss-Jacobi speedup
        n_calls = 50
        t0 = time.time()
        for _ in range(n_calls):
            compute_psi_functional(1.0, 1.0, params_1, params_2, n_grid=200)
        naive_time = (time.time() - t0) / n_calls * 1000
        
        t0 = time.time()
        for _ in range(n_calls):
            compute_psi_functional_gauss_jacobi(1.0, 1.0, params_1, params_2, n_points=32)
        gj_time = (time.time() - t0) / n_calls * 1000
        
        speedup = naive_time / gj_time
        self._record("Gauss-Jacobi speedup", speedup > 2.0, f"{speedup:.1f}× (naive={naive_time:.2f}ms, GJ={gj_time:.2f}ms)")
    
    def _test_correlation_copula(self):
        """Test correlation copula."""
        print(f"\n{'─'*70}")
        print("3. CORRELATION COPULA TESTS")
        print(f"{'─'*70}")
        
        params_1 = RoughHestonParams(H=0.10, eta=0.15, rho=-0.7, xi0=0.04,
                                     kappa=2.0, theta=0.04, spot=100.0, maturity=1/12)
        params_2 = RoughHestonParams(H=0.15, eta=0.18, rho=-0.5, xi0=0.05,
                                     kappa=1.5, theta=0.05, spot=100.0, maturity=1/12)
        
        target_rho = np.array([[1.0, 0.85], [0.85, 1.0]])
        
        # Test copula initialization
        try:
            copula = RoughMartingaleCopula([params_1, params_2], target_rho)
            self._record("Copula initialization", True)
        except Exception as e:
            self._record("Copula initialization", False, str(e))
            return
        
        # Test amplification factor
        amp = copula.amplification
        self._record("Amplification calibrated", 1.0 <= amp <= 1.5, f"α={amp:.3f}")
        
        # Test simulation
        spot_paths, vol_paths, _ = copula.simulate(n_paths=30000, n_steps=50, seed=42)
        self._record("Copula simulation", spot_paths.shape == (30000, 51, 2))
        
        # Test correlation enforcement
        is_valid, rho_realized, max_error = copula.validate_correlation(spot_paths, tolerance=0.05)
        self._record("Correlation enforcement", max_error < 0.05, f"|ρ_realized-ρ_target|={max_error:.4f}")
    
    def _test_path_simulation(self):
        """Test path simulation."""
        print(f"\n{'─'*70}")
        print("4. PATH SIMULATION TESTS")
        print(f"{'─'*70}")
        
        params_1 = RoughHestonParams(H=0.10, eta=0.15, rho=-0.7, xi0=0.04,
                                     kappa=2.0, theta=0.04, spot=100.0, maturity=1/12)
        params_2 = RoughHestonParams(H=0.15, eta=0.18, rho=-0.5, xi0=0.05,
                                     kappa=1.5, theta=0.05, spot=100.0, maturity=1/12)
        
        rho = np.array([[1.0, 0.85], [0.85, 1.0]])
        
        # Simulate paths
        paths = simulate_correlated_rough_heston([params_1, params_2], rho, n_paths=20000, n_steps=50)
        
        # Shape check
        self._record("Path shape", paths.shape == (20000, 51, 2))
        
        # No NaN/Inf
        self._record("No NaN paths", not np.any(np.isnan(paths)))
        self._record("No Inf paths", not np.any(np.isinf(paths)))
        
        # Terminal distribution
        S_T = paths[:, -1, :]
        mean_ratio = np.mean(S_T, axis=0) / np.array([params_1.spot, params_2.spot])
        self._record("Terminal mean reasonable", np.all((mean_ratio > 0.8) & (mean_ratio < 1.2)), 
                    f"ratios={mean_ratio}")
    
    def _test_basket_pricing(self):
        """Test basket pricing."""
        print(f"\n{'─'*70}")
        print("5. BASKET PRICING TESTS")
        print(f"{'─'*70}")
        
        params_1 = RoughHestonParams(H=0.10, eta=0.15, rho=-0.7, xi0=0.04,
                                     kappa=2.0, theta=0.04, spot=100.0, maturity=1/12)
        params_2 = RoughHestonParams(H=0.15, eta=0.18, rho=-0.5, xi0=0.05,
                                     kappa=1.5, theta=0.05, spot=100.0, maturity=1/12)
        
        weights = np.array([0.5, 0.5])
        rho = np.array([[1.0, 0.85], [0.85, 1.0]])
        
        paths = simulate_correlated_rough_heston([params_1, params_2], rho, n_paths=30000, n_steps=50)
        
        basket_spot = 0.5 * 100 + 0.5 * 100
        strikes = np.array([95, 100, 105])
        
        results = price_multiple_strikes(paths, weights, strikes, T=1/12)
        
        # ITM > ATM > OTM
        prices = [r.price for r in results]
        self._record("ITM > ATM > OTM ordering", prices[0] > prices[1] > prices[2],
                    f"prices={[f'${p:.2f}' for p in prices]}")
        
        # Positive prices
        self._record("All prices positive", all(p > 0 for p in prices))
        
        # Standard errors reasonable
        se = [r.std_error for r in results]
        self._record("Standard errors < 5%", all(e/p < 0.05 for e, p in zip(se, prices)),
                    f"SE%={[f'{100*e/p:.1f}%' for e, p in zip(se, prices)]}")
    
    def _test_frtb_bounds(self):
        """Test FRTB bounds."""
        print(f"\n{'─'*70}")
        print("6. FRTB BOUNDS TESTS")
        print(f"{'─'*70}")
        
        params_1 = RoughHestonParams(H=0.10, eta=0.15, rho=-0.7, xi0=0.04,
                                     kappa=2.0, theta=0.04, spot=100.0, maturity=1/12)
        params_2 = RoughHestonParams(H=0.15, eta=0.18, rho=-0.5, xi0=0.05,
                                     kappa=1.5, theta=0.05, spot=100.0, maturity=1/12)
        
        weights = np.array([0.5, 0.5])
        basket_spot = 100.0
        
        # Test at different strikes
        for K, price_est in [(95, 8.0), (100, 4.0), (105, 1.5)]:
            bounds = compute_frtb_bounds(price_est, weights, K, [params_1, params_2])
            
            # P_low <= price <= P_up
            valid = bounds.P_low <= price_est <= bounds.P_up
            self._record(f"Bounds contain price (K={K})", valid,
                        f"[{bounds.P_low:.2f}, {bounds.P_up:.2f}] ∋ {price_est}")
            
            # Width is finite
            self._record(f"Finite width (K={K})", np.isfinite(bounds.width),
                        f"width=${bounds.width:.4f}")
        
        # Scaling test
        from scipy.stats import linregress
        maturities = np.array([1/52, 1/12, 3/12, 6/12])
        scalings = []
        for T in maturities:
            p1 = RoughHestonParams(H=0.10, eta=0.15, rho=-0.7, xi0=0.04,
                                   kappa=2.0, theta=0.04, spot=100.0, maturity=T)
            p2 = RoughHestonParams(H=0.15, eta=0.18, rho=-0.5, xi0=0.05,
                                   kappa=1.5, theta=0.05, spot=100.0, maturity=T)
            bounds = compute_frtb_bounds(5.0, weights, 100.0, [p1, p2])
            scalings.append(bounds.scaling)
        
        log_T = np.log(maturities)
        log_S = np.log(scalings)
        slope, _, r_value, _, _ = linregress(log_T, log_S)
        expected = 2 * 0.10  # 2 * H_eff
        
        self._record("Scaling exponent", abs(slope - expected) < 0.05,
                    f"slope={slope:.4f}, expected={expected:.4f}, R²={r_value**2:.4f}")
    
    def _test_full_pipeline(self):
        """Test full pipeline integration."""
        print(f"\n{'─'*70}")
        print("7. FULL PIPELINE TESTS")
        print(f"{'─'*70}")
        
        # Create synthetic config
        assets = []
        for i, (ticker, spot) in enumerate([('A', 100.0), ('B', 100.0)]):
            strikes = np.linspace(90, 110, 10)
            prices = np.maximum(spot - strikes, 0) + 5  # Synthetic
            assets.append(AssetConfig(ticker=ticker, spot=spot, strikes=strikes,
                                      market_prices=prices, maturity=1/12))
        
        weights = np.array([0.5, 0.5])
        rho_guess = np.array([[1.0, 0.8], [0.8, 1.0]])
        basket_strikes = np.array([95, 100, 105])
        
        config = MultiAssetConfig(assets=assets, basket_weights=weights,
                                  correlation_guess=rho_guess, basket_strikes=basket_strikes)
        
        # Run pipeline
        try:
            start = time.time()
            result = multi_asset_rmot_pipeline(config, n_paths=20000, n_steps=50, verbose=False)
            elapsed = time.time() - start
            self._record("Pipeline completes", True, f"{elapsed:.2f}s")
        except Exception as e:
            self._record("Pipeline completes", False, str(e))
            return
        
        # Check outputs
        self._record("Marginal calibration exists", 'marginal_calibration' in result)
        self._record("Correlation estimation exists", 'correlation_estimation' in result)
        self._record("Basket prices exist", 'basket_prices' in result and len(result['basket_prices']) > 0)
        self._record("FRTB bounds exist", 'frtb_bounds' in result and len(result['frtb_bounds']) > 0)
        
        # Check Hurst distinctness
        H_values = [p.H for p in result['marginal_calibration'].params]
        self._record("Distinct Hurst values", len(set(H_values)) == len(H_values),
                    f"H={H_values}")
    
    def _test_real_market_data(self):
        """Test real market data integration."""
        print(f"\n{'─'*70}")
        print("8. REAL MARKET DATA TESTS")
        print(f"{'─'*70}")
        
        try:
            stream = RealTimeDataStream()
            config = stream.fetch_live_data(['SPY', 'QQQ'])
            self._record("Fetch SPY+QQQ", True, f"SPY=${config.assets[0].spot:.2f}, QQQ=${config.assets[1].spot:.2f}")
        except Exception as e:
            self._record("Fetch SPY+QQQ", False, str(e))
            return
        
        # Run pipeline on real data
        try:
            start = time.time()
            result = multi_asset_rmot_pipeline(config, n_paths=20000, n_steps=50, verbose=False)
            elapsed = time.time() - start
            self._record("Pipeline on real data", True, f"{elapsed:.2f}s")
        except Exception as e:
            self._record("Pipeline on real data", False, str(e))
            return
        
        # Validate outputs
        basket_prices = [p.price for p in result['basket_prices']]
        self._record("Valid basket prices", all(np.isfinite(p) and p > 0 for p in basket_prices[:3]),
                    f"prices={[f'${p:.2f}' for p in basket_prices[:3]]}")
        
        bounds = result['frtb_bounds']
        self._record("Valid FRTB bounds", all(np.isfinite(b.width) for b in bounds[:3]),
                    f"widths={[f'${b.width:.4f}' for b in bounds[:3]]}")
    
    def _test_stress_conditions(self):
        """Test under stress conditions."""
        print(f"\n{'─'*70}")
        print("9. STRESS CONDITION TESTS")
        print(f"{'─'*70}")
        
        # Extreme correlations
        for rho_val in [0.99, -0.95, 0.0]:
            params = [
                RoughHestonParams(H=0.10, eta=0.15, rho=-0.7, xi0=0.04, kappa=2.0, theta=0.04, spot=100.0, maturity=1/12),
                RoughHestonParams(H=0.15, eta=0.15, rho=-0.7, xi0=0.04, kappa=2.0, theta=0.04, spot=100.0, maturity=1/12)
            ]
            rho = np.array([[1.0, rho_val], [rho_val, 1.0]])
            try:
                paths = simulate_correlated_rough_heston(params, rho, n_paths=5000, n_steps=25, use_copula=False)
                valid = not np.any(np.isnan(paths)) and not np.any(np.isinf(paths))
                self._record(f"Extreme ρ={rho_val}", valid)
            except Exception as e:
                self._record(f"Extreme ρ={rho_val}", False, str(e))
        
        # Very short maturity
        params = [
            RoughHestonParams(H=0.10, eta=0.15, rho=-0.7, xi0=0.04, kappa=2.0, theta=0.04, spot=100.0, maturity=1/365),
            RoughHestonParams(H=0.15, eta=0.15, rho=-0.7, xi0=0.04, kappa=2.0, theta=0.04, spot=100.0, maturity=1/365)
        ]
        try:
            rho = np.array([[1.0, 0.8], [0.8, 1.0]])
            paths = simulate_correlated_rough_heston(params, rho, n_paths=5000, n_steps=25, use_copula=False)
            valid = not np.any(np.isnan(paths))
            self._record("T=1 day", valid)
        except Exception as e:
            self._record("T=1 day", False, str(e))
        
        # Very long maturity
        params = [
            RoughHestonParams(H=0.10, eta=0.15, rho=-0.7, xi0=0.04, kappa=2.0, theta=0.04, spot=100.0, maturity=2.0),
            RoughHestonParams(H=0.15, eta=0.15, rho=-0.7, xi0=0.04, kappa=2.0, theta=0.04, spot=100.0, maturity=2.0)
        ]
        try:
            paths = simulate_correlated_rough_heston(params, rho, n_paths=5000, n_steps=25, use_copula=False)
            valid = not np.any(np.isnan(paths))
            self._record("T=2 years", valid)
        except Exception as e:
            self._record("T=2 years", False, str(e))
        
        # High vol-of-vol
        params = [
            RoughHestonParams(H=0.10, eta=0.40, rho=-0.7, xi0=0.04, kappa=2.0, theta=0.04, spot=100.0, maturity=1/12),
            RoughHestonParams(H=0.15, eta=0.40, rho=-0.7, xi0=0.04, kappa=2.0, theta=0.04, spot=100.0, maturity=1/12)
        ]
        try:
            paths = simulate_correlated_rough_heston(params, rho, n_paths=5000, n_steps=25, use_copula=False)
            valid = not np.any(np.isnan(paths))
            self._record("High η=0.40", valid)
        except Exception as e:
            self._record("High η=0.40", False, str(e))
        
        # 5 assets
        params = [
            RoughHestonParams(H=0.08+0.04*i, eta=0.15, rho=-0.7, xi0=0.04, kappa=2.0, theta=0.04, spot=100.0, maturity=1/12)
            for i in range(5)
        ]
        rho = np.eye(5)
        for i in range(5):
            for j in range(i+1, 5):
                rho[i,j] = rho[j,i] = 0.7
        try:
            paths = simulate_correlated_rough_heston(params, rho, n_paths=5000, n_steps=25, use_copula=False)
            valid = paths.shape == (5000, 26, 5) and not np.any(np.isnan(paths))
            self._record("5 assets", valid, f"shape={paths.shape}")
        except Exception as e:
            self._record("5 assets", False, str(e))
    
    def _test_performance(self):
        """Test performance benchmarks."""
        print(f"\n{'─'*70}")
        print("10. PERFORMANCE BENCHMARKS")
        print(f"{'─'*70}")
        
        params_1 = RoughHestonParams(H=0.10, eta=0.15, rho=-0.7, xi0=0.04,
                                     kappa=2.0, theta=0.04, spot=100.0, maturity=1/12)
        params_2 = RoughHestonParams(H=0.15, eta=0.18, rho=-0.5, xi0=0.05,
                                     kappa=1.5, theta=0.05, spot=100.0, maturity=1/12)
        rho = np.array([[1.0, 0.85], [0.85, 1.0]])
        
        # Ψ_ij speed
        n_calls = 100
        t0 = time.time()
        for _ in range(n_calls):
            compute_psi_functional_gauss_jacobi(1.0, 1.0, params_1, params_2, n_points=32)
        psi_time = (time.time() - t0) / n_calls * 1000
        self._record("Ψ_ij < 1ms", psi_time < 1.0, f"{psi_time:.2f}ms")
        
        # Simulation speed
        t0 = time.time()
        paths = simulate_correlated_rough_heston([params_1, params_2], rho, n_paths=50000, n_steps=50, use_copula=False)
        sim_time = time.time() - t0
        self._record("50k paths < 2s", sim_time < 2.0, f"{sim_time:.2f}s")
        
        # Full pipeline speed
        assets = [AssetConfig(ticker='A', spot=100.0, strikes=np.linspace(90,110,10),
                             market_prices=np.maximum(100-np.linspace(90,110,10),0)+5, maturity=1/12),
                  AssetConfig(ticker='B', spot=100.0, strikes=np.linspace(90,110,10),
                             market_prices=np.maximum(100-np.linspace(90,110,10),0)+5, maturity=1/12)]
        config = MultiAssetConfig(assets=assets, basket_weights=np.array([0.5,0.5]),
                                  correlation_guess=rho, basket_strikes=np.array([95,100,105]))
        
        t0 = time.time()
        result = multi_asset_rmot_pipeline(config, n_paths=20000, n_steps=50, verbose=False)
        pipe_time = time.time() - t0
        self._record("Pipeline < 5s", pipe_time < 5.0, f"{pipe_time:.2f}s")
    
    def _print_summary(self):
        """Print test summary."""
        elapsed = time.time() - self.start_time
        
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total tests: {self.n_total}")
        print(f"Passed: {self.n_passed} ✅")
        print(f"Failed: {self.n_failed} ❌")
        print(f"Pass rate: {100*self.n_passed/self.n_total:.1f}%")
        print(f"Total time: {elapsed:.1f}s")
        
        if self.n_failed == 0:
            print(f"\n{'='*80}")
            print("🎉 ALL TESTS PASSED - SYSTEM FULLY VALIDATED")
            print(f"{'='*80}")
        else:
            print(f"\n⚠️  {self.n_failed} TESTS FAILED:")
            for name, result in self.results.items():
                if not result['passed']:
                    print(f"   ❌ {name}: {result['details']}")


if __name__ == "__main__":
    suite = ComprehensiveTestSuite()
    suite.run_all()
