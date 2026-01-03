#!/usr/bin/env python3
"""
Multi-Asset RMOT Research Runner

Main entry point for research experiments and publication.

Usage:
    python3 run_experiment.py --mode batch --tickers SPY QQQ IWM
    python3 run_experiment.py --mode stream --tickers SPY QQQ --interval 60
    python3 run_experiment.py --mode benchmark
"""

import argparse
import numpy as np
import time
from datetime import datetime
from typing import List

# Local imports
from src.real_time_data import RealTimeDataStream, ResearchMonitor, ExperimentResult
from src.pipeline import multi_asset_rmot_pipeline
from src.data_structures import MultiAssetConfig


def run_batch_experiment(
    tickers: List[str],
    n_paths: int = 30000,
    n_steps: int = 50,
    verbose: bool = True
):
    """
    Run a single batch experiment with real market data.
    
    This is the standard mode for research publications.
    """
    print("\n" + "=" * 80)
    print("🔬 BATCH EXPERIMENT: Multi-Asset RMOT")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Assets: {', '.join(tickers)}")
    print(f"Paths: {n_paths:,}")
    
    # Fetch live data
    stream = RealTimeDataStream()
    config = stream.fetch_live_data(tickers)
    
    print(f"\n📊 Market Data Summary:")
    for asset in config.assets:
        print(f"   {asset.ticker}: ${asset.spot:.2f}, {len(asset.strikes)} strikes, T={asset.maturity*365:.0f}d")
    
    basket_spot = sum(w * a.spot for w, a in zip(config.basket_weights, config.assets))
    print(f"   Basket: ${basket_spot:.2f}")
    
    # Run pipeline
    print(f"\n{'='*80}")
    print("RUNNING MULTI-ASSET RMOT PIPELINE")
    print("=" * 80)
    
    start_time = time.time()
    result = multi_asset_rmot_pipeline(config, n_paths=n_paths, n_steps=n_steps, verbose=verbose)
    elapsed = time.time() - start_time
    
    # Print results
    print(f"\n{'='*80}")
    print("📈 RESULTS")
    print("=" * 80)
    
    print("\n1. MARGINAL CALIBRATION:")
    for i, params in enumerate(result['marginal_calibration'].params):
        print(f"   {config.assets[i].ticker}: H={params.H:.4f}, η={params.eta:.4f}, ρ={params.rho:.4f}, ξ₀={params.xi0:.4f}")
    
    print("\n2. CORRELATION MATRIX:")
    rho = result['correlation_estimation'].rho
    print(f"   {rho}")
    
    print("\n3. BASKET PRICING:")
    for price_result in result['basket_prices']:
        print(f"   K={price_result.strike:.2f}: Call=${price_result.price:.4f}")
    
    print("\n4. FRTB BOUNDS:")
    for K, bound in zip(config.basket_strikes, result['frtb_bounds']):
        print(f"   K={K:.2f}: Width=${bound.width:.4f}, P∈[${bound.P_low:.4f}, ${bound.P_up:.4f}]")
    
    print(f"\n5. PERFORMANCE:")
    print(f"   Total time: {elapsed:.2f}s")
    print(f"   Paths/sec: {n_paths/elapsed:,.0f}")
    
    print(f"\n{'='*80}")
    print("✅ EXPERIMENT COMPLETE")
    print("=" * 80)
    
    return result


def run_streaming_experiment(
    tickers: List[str],
    n_iterations: int = 10,
    update_interval: int = 60,
    output_file: str = None
):
    """
    Run streaming experiment with multiple iterations.
    
    Tracks results over time for stability analysis.
    """
    from src.pipeline import multi_asset_rmot_pipeline
    
    print("\n" + "=" * 80)
    print("🔬 STREAMING EXPERIMENT: Multi-Asset RMOT")
    print("=" * 80)
    print(f"Assets: {', '.join(tickers)}")
    print(f"Iterations: {n_iterations}")
    print(f"Interval: {update_interval}s")
    
    stream = RealTimeDataStream()
    monitor = ResearchMonitor(f"rmot_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    for iteration in range(n_iterations):
        print(f"\n{'─'*40}")
        print(f"ITERATION {iteration + 1}/{n_iterations}")
        print(f"{'─'*40}")
        
        try:
            # Fetch data
            config = stream.fetch_live_data(tickers)
            
            # Run pipeline
            start = time.time()
            result = multi_asset_rmot_pipeline(config, n_paths=20000, n_steps=50, verbose=False)
            elapsed = time.time() - start
            
            # Extract metrics
            basket_price = np.mean([p.price for p in result['basket_prices']])
            bound_width = np.mean([b.width for b in result['frtb_bounds']])
            
            rho_target = config.correlation_guess
            rho_est = result['correlation_estimation'].rho
            corr_error = np.max(np.abs(rho_target - rho_est))
            
            # Log
            exp_result = ExperimentResult(
                timestamp=datetime.now(),
                iteration=iteration,
                basket_price=basket_price,
                bound_width=bound_width,
                correlation_error=corr_error,
                elapsed_sec=elapsed,
                params={p.ticker: {'H': p.H, 'eta': p.eta} for p in result['marginal_calibration'].params}
            )
            monitor.log_result(exp_result)
            
        except Exception as e:
            print(f"⚠️  Iteration failed: {e}")
        
        if iteration < n_iterations - 1:
            time.sleep(update_interval)
    
    # Summary
    monitor.print_summary()
    
    # Save
    if output_file is None:
        output_file = f"/tmp/rmot_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    monitor.save_results(output_file)
    
    return monitor


def run_benchmark_suite():
    """Run the full benchmark suite."""
    import subprocess
    import sys
    
    print("\n" + "=" * 80)
    print("🔬 RUNNING FULL BENCHMARK SUITE")
    print("=" * 80)
    
    result = subprocess.run(
        [sys.executable, '-m', 'tests.benchmark_suite'],
        cwd='/Volumes/Hippocampus/Antigravity/RMOT/RMOT/MULTI',
        capture_output=False
    )
    
    return result.returncode == 0


def run_monitored_experiment(
    tickers: List[str],
    n_iterations: int = 5,
    update_interval: int = 60,
    slack_webhook: str = None
):
    """Run experiment with full monitoring and alerts."""
    from src.monitoring import run_monitored_experiment as _run_monitored
    return _run_monitored(tickers, n_iterations, update_interval, slack_webhook)


def main():
    parser = argparse.ArgumentParser(description='Multi-Asset RMOT Research Runner')
    parser.add_argument('--mode', choices=['batch', 'stream', 'benchmark', 'monitored'], default='batch',
                       help='Experiment mode')
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ'],
                       help='Stock tickers')
    parser.add_argument('--paths', type=int, default=30000,
                       help='Monte Carlo paths')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of iterations (streaming/monitored mode)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Update interval in seconds (streaming/monitored mode)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results')
    parser.add_argument('--slack', type=str, default=None,
                       help='Slack webhook URL for alerts')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce verbosity')
    
    args = parser.parse_args()
    
    if args.mode == 'batch':
        run_batch_experiment(
            tickers=args.tickers,
            n_paths=args.paths,
            verbose=not args.quiet
        )
    elif args.mode == 'stream':
        run_streaming_experiment(
            tickers=args.tickers,
            n_iterations=args.iterations,
            update_interval=args.interval,
            output_file=args.output
        )
    elif args.mode == 'monitored':
        run_monitored_experiment(
            tickers=args.tickers,
            n_iterations=args.iterations,
            update_interval=args.interval,
            slack_webhook=args.slack
        )
    elif args.mode == 'benchmark':
        run_benchmark_suite()


if __name__ == "__main__":
    main()
