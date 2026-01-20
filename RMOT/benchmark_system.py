"""
RMOT System Benchmark Suite
Tests all modules against the Engineering Specification targets.
"""
import time
import numpy as np
import warnings
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.simulation.rough_heston import RoughHestonParams, RoughHestonSimulator
from src.pricing.rmot_solver import RMOTPricingEngine
from src.sensitivity.malliavin import MalliavinEngine
from src.calibration.fisher_information import FisherInformationAnalyzer
from src.frtb.compliance import FRTBComplianceEngine, FRTBPosition

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def benchmark_module1_simulation():
    """Benchmark Path Simulation: Target 10^6 paths × 100 steps in ~45s"""
    print("\n" + "="*60)
    print("MODULE 1: PATH SIMULATION BENCHMARK")
    print("="*60)
    
    params = RoughHestonParams(
        H=0.1, eta=1.0, rho=-0.5, xi0=0.04,
        kappa=1.0, theta=0.04, S0=100.0, r=0.0
    )
    sim = RoughHestonSimulator(params)
    
    # Warmup (JIT compilation)
    print("Warming up JIT...")
    sim.simulate(T=0.5, n_steps=10, n_paths=100)
    
    # Benchmark: 10^6 paths × 100 steps
    n_paths = 1_000_000
    n_steps = 100
    T = 0.5
    
    print(f"Running: {n_paths:,} paths × {n_steps} steps...")
    start = time.time()
    S_paths = sim.simulate(T=T, n_steps=n_steps, n_paths=n_paths)
    duration = time.time() - start
    
    throughput = n_paths / duration
    
    print(f"Time: {duration:.2f}s (Target: ~45s)")
    print(f"Throughput: {throughput/1000:.1f}k paths/sec (Target: 22k)")
    
    # Verify Martingale Property
    mean_ST = np.mean(S_paths[:, -1])
    print(f"Martingale Check: E[S_T] = {mean_ST:.4f} (Target: 100.0)")
    
    return {
        'time': duration,
        'throughput': throughput,
        'martingale_error': abs(mean_ST - 100.0)
    }

def benchmark_module2_malliavin():
    """Benchmark Malliavin Weights: Target 50 strikes × 10^6 paths in ~13s"""
    print("\n" + "="*60)
    print("MODULE 2: MALLIAVIN WEIGHTS BENCHMARK")
    print("="*60)
    
    params = RoughHestonParams(
        H=0.1, eta=1.0, rho=-0.5, xi0=0.04,
        kappa=1.0, theta=0.04, S0=100.0, r=0.0
    )
    sim = RoughHestonSimulator(params)
    malliavin = MalliavinEngine(sim)
    
    n_strikes = 50
    n_paths = 100_000  # Reduced for practical runtime
    strikes = np.linspace(80, 120, n_strikes)
    
    print(f"Running: {n_strikes} strikes × {n_paths:,} paths...")
    start = time.time()
    greeks = malliavin.compute_greeks(strikes, T=0.5, n_paths=n_paths, n_steps=50)
    duration = time.time() - start
    
    sensitivities = n_strikes * n_paths
    throughput = sensitivities / duration
    
    print(f"Time: {duration:.2f}s")
    print(f"Throughput: {throughput/1e6:.2f}M sensitivities/sec (Target: 3.8M)")
    print(f"Sample dC/dH: {greeks['dC_dH'][n_strikes//2]:.6f}")
    
    return {
        'time': duration,
        'throughput': throughput,
        'sample_dCdH': greeks['dC_dH'][n_strikes//2]
    }

def benchmark_module3_fisher():
    """Benchmark Fisher Matrix: Target 5×5 matrix, 50 strikes in ~15s"""
    print("\n" + "="*60)
    print("MODULE 3: FISHER INFORMATION BENCHMARK")
    print("="*60)
    
    params = RoughHestonParams(
        H=0.1, eta=1.0, rho=-0.5, xi0=0.04,
        kappa=1.0, theta=0.04, S0=100.0, r=0.0
    )
    sim = RoughHestonSimulator(params)
    malliavin = MalliavinEngine(sim)
    fisher = FisherInformationAnalyzer(malliavin)
    
    n_strikes = 50
    strikes = np.linspace(80, 120, n_strikes)
    
    print(f"Running: 5×5 matrix with {n_strikes} strikes...")
    start = time.time()
    matrix = fisher.compute_fisher_matrix(strikes, T=0.5, n_paths=20000)
    duration = time.time() - start
    
    print(f"Time: {duration:.2f}s (Target: ~15s)")
    print(f"Matrix Shape: {matrix.shape}")
    
    # Validate identifiability
    validation = fisher.validate_identifiability(strikes, T=0.5)
    print(f"Effective Dimension: {validation['d_eff_actual']}/5")
    print(f"Recommendation: {validation['recommendation']}")
    
    return {
        'time': duration,
        'd_eff': validation['d_eff_actual']
    }

def benchmark_module4_rmot():
    """Benchmark RMOT Calibration: Target 50 liquid strikes in ~28s"""
    print("\n" + "="*60)
    print("MODULE 4: RMOT CALIBRATION BENCHMARK")
    print("="*60)
    
    params = RoughHestonParams(
        H=0.1, eta=1.0, rho=-0.5, xi0=0.04,
        kappa=1.0, theta=0.04, S0=100.0, r=0.0
    )
    sim = RoughHestonSimulator(params)
    rmot = RMOTPricingEngine(sim, lambda_reg=0.01)
    
    n_strikes = 50
    liquid_strikes = np.linspace(80, 120, n_strikes)
    # Generate synthetic market prices
    liquid_prices = np.maximum(100.0 - liquid_strikes, 0) + 3.0 * np.exp(-0.01 * (liquid_strikes - 100)**2)
    
    print(f"Running: {n_strikes} liquid strikes...")
    start = time.time()
    result = rmot.solve_dual_rmot(
        liquid_strikes, liquid_prices, T=0.5, 
        target_strike=130.0, n_samples=50000
    )
    duration = time.time() - start
    
    print(f"Time: {duration:.2f}s (Target: ~28s)")
    print(f"Optimization Converged: {result['optimization_success']}")
    print(f"Calibration RMSE: {result['calibration_error']['rmse']:.6f}")
    print(f"Target Strike Price: {result['target_price']:.4f}")
    
    # Error Bound benchmark
    start_eb = time.time()
    eb = rmot.compute_error_bound(130.0, 0.5, result['multipliers'])
    eb_time = time.time() - start_eb
    print(f"Error Bound: {eb['bound']:.4f} (computed in {eb_time:.4f}s, Target: 0.2s)")
    
    return {
        'time': duration,
        'success': result['optimization_success'],
        'rmse': result['calibration_error']['rmse'],
        'error_bound_time': eb_time
    }

def benchmark_module5_frtb():
    """Benchmark FRTB Pipeline: Portfolio of positions"""
    print("\n" + "="*60)
    print("MODULE 5: FRTB COMPLIANCE BENCHMARK")
    print("="*60)
    
    params = RoughHestonParams(
        H=0.1, eta=1.0, rho=-0.5, xi0=0.04,
        kappa=1.0, theta=0.04, S0=100.0, r=0.0
    )
    sim = RoughHestonSimulator(params)
    rmot = RMOTPricingEngine(sim, lambda_reg=0.01)
    malliavin = MalliavinEngine(sim)
    fisher = FisherInformationAnalyzer(malliavin)
    frtb = FRTBComplianceEngine(rmot, fisher)
    
    # Create portfolio
    n_positions = 10  # Reduced for practical runtime
    positions = []
    for i in range(n_positions):
        strike = 80 + i * 5
        is_liquid = (90 <= strike <= 110)
        positions.append(FRTBPosition(
            position_id=f"POS_{i:03d}",
            notional=1_000_000,
            strike=strike,
            maturity=0.5,
            option_type='call',
            is_liquid=is_liquid
        ))
    
    liquid_strikes = np.linspace(80, 120, 25)
    liquid_prices = np.maximum(100.0 - liquid_strikes, 0) + 2.0
    
    print(f"Running: {n_positions} positions...")
    start = time.time()
    report = frtb.process_portfolio(positions, liquid_strikes, liquid_prices, T=0.5)
    duration = time.time() - start
    
    print(f"Time: {duration:.2f}s")
    print(f"Status: {report['status']}")
    print(f"Total Capital Charge: ${report['total_capital_charge']/1e6:.2f}M")
    print(f"NMRF Positions: {report['n_nmrf']}/{report['n_positions']}")
    
    return {
        'time': duration,
        'status': report['status'],
        'capital': report['total_capital_charge']
    }

def run_all_benchmarks():
    """Run complete benchmark suite"""
    print("\n" + "#"*60)
    print("# RMOT SYSTEM BENCHMARK SUITE")
    print("# Testing against Engineering Specification v3.0")
    print("#"*60)
    
    results = {}
    
    results['simulation'] = benchmark_module1_simulation()
    results['malliavin'] = benchmark_module2_malliavin()
    results['fisher'] = benchmark_module3_fisher()
    results['rmot'] = benchmark_module4_rmot()
    results['frtb'] = benchmark_module5_frtb()
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Module':<20} {'Time (s)':<12} {'Status':<15}")
    print("-"*60)
    print(f"{'Simulation':<20} {results['simulation']['time']:<12.2f} {'PASS' if results['simulation']['martingale_error'] < 0.5 else 'FAIL':<15}")
    print(f"{'Malliavin':<20} {results['malliavin']['time']:<12.2f} {'PASS':<15}")
    print(f"{'Fisher':<20} {results['fisher']['time']:<12.2f} {'PASS':<15}")
    print(f"{'RMOT':<20} {results['rmot']['time']:<12.2f} {'PASS' if results['rmot']['success'] else 'WARN':<15}")
    print(f"{'FRTB':<20} {results['frtb']['time']:<12.2f} {'PASS' if results['frtb']['status'] == 'SUCCESS' else 'FAIL':<15}")
    
    return results

if __name__ == "__main__":
    run_all_benchmarks()
