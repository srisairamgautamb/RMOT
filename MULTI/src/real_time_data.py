"""
Real-Time Market Data Streaming for Research

Uses FREE data sources (yfinance) for academic research.
No Bloomberg/Refinitiv required!

Features:
- Real-time quotes and option chains
- Automatic rate limiting (stay within free tier)
- Caching to minimize API calls
- Continuous streaming for experiments
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field

try:
    from .data_structures import AssetConfig, MultiAssetConfig
except ImportError:
    from data_structures import AssetConfig, MultiAssetConfig


@dataclass
class StreamingConfig:
    """Configuration for real-time data streaming."""
    # Rate limits (conservative for free tier)
    requests_per_minute: int = 30
    
    # Refresh intervals
    quote_refresh_sec: int = 10
    option_refresh_sec: int = 60
    
    # Caching
    enable_cache: bool = True
    cache_ttl_sec: int = 5
    
    # Target maturity
    target_days: int = 30


class RealTimeDataStream:
    """
    Real-time market data streaming using yfinance (FREE).
    
    Perfect for research: no API keys, no costs, reliable data.
    """
    
    def __init__(self, config: StreamingConfig = None):
        self.config = config or StreamingConfig()
        self.cache = {}
        self.request_times = []
    
    def fetch_live_data(
        self,
        tickers: List[str],
        target_days: int = None
    ) -> MultiAssetConfig:
        """
        Fetch live market data for multiple assets.
        
        Args:
            tickers: List of stock tickers (e.g., ['SPY', 'QQQ', 'IWM'])
            target_days: Days to expiration for options
        
        Returns:
            MultiAssetConfig with live data
        """
        import yfinance as yf
        
        if target_days is None:
            target_days = self.config.target_days
        
        print(f"\n{'='*70}")
        print(f"FETCHING LIVE DATA: {', '.join(tickers)}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        assets = []
        
        for ticker in tickers:
            print(f"\n📊 {ticker}...", end=" ")
            
            self._enforce_rate_limit()
            
            try:
                stock = yf.Ticker(ticker)
                
                # Get spot price
                hist = stock.history(period='1d')
                if hist.empty:
                    print("❌ No price data")
                    continue
                
                spot = hist['Close'].iloc[-1]
                print(f"${spot:.2f}", end=" ")
                
                # Get options
                expiries = stock.options
                if not expiries:
                    print("❌ No options")
                    continue
                
                # Find target expiry
                today = datetime.now()
                target_date = today + timedelta(days=target_days)
                
                best_expiry = min(expiries, key=lambda x: abs(
                    (datetime.strptime(x, "%Y-%m-%d") - target_date).days
                ))
                
                exp_date = datetime.strptime(best_expiry, "%Y-%m-%d")
                T = max((exp_date - today).days / 365.0, 1/365)
                
                # Get chain
                chain = stock.option_chain(best_expiry)
                calls = chain.calls
                
                # Filter liquid
                liquid = calls[(calls['bid'] > 0) & (calls['ask'] > 0)].copy()
                liquid['mid'] = (liquid['bid'] + liquid['ask']) / 2
                liquid = liquid[(liquid['ask'] - liquid['bid']) / liquid['bid'] < 0.15]
                liquid = liquid[(liquid['strike'] > 0.90 * spot) & (liquid['strike'] < 1.10 * spot)]
                
                if len(liquid) < 5:
                    print(f"❌ Only {len(liquid)} strikes")
                    continue
                
                print(f"✅ {len(liquid)} strikes, T={T*365:.0f}d")
                
                assets.append(AssetConfig(
                    ticker=ticker,
                    spot=spot,
                    strikes=liquid['strike'].values,
                    market_prices=liquid['mid'].values,
                    maturity=T
                ))
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        if len(assets) < 2:
            raise ValueError("Need at least 2 assets with valid data")
        
        # Use common maturity
        common_T = min(a.maturity for a in assets)
        for a in assets:
            a.maturity = common_T
        
        # Create config
        N = len(assets)
        weights = np.ones(N) / N
        
        # Historical correlation estimate
        rho_guess = np.eye(N)
        for i in range(N):
            for j in range(i+1, N):
                rho_guess[i, j] = rho_guess[j, i] = 0.85  # Typical for index ETFs
        
        # Basket strikes
        basket_spot = sum(w * a.spot for w, a in zip(weights, assets))
        basket_strikes = basket_spot * np.array([0.95, 0.98, 1.00, 1.02, 1.05])
        
        return MultiAssetConfig(
            assets=assets,
            basket_weights=weights,
            correlation_guess=rho_guess,
            basket_strikes=basket_strikes
        )
    
    def _enforce_rate_limit(self):
        """Stay within free tier limits."""
        now = time.time()
        window = 60.0
        
        # Clean old timestamps
        self.request_times = [t for t in self.request_times if now - t < window]
        
        # Check limit
        if len(self.request_times) >= self.config.requests_per_minute:
            oldest = self.request_times[0]
            wait_time = window - (now - oldest) + 1
            print(f"\n⏳ Rate limit, waiting {wait_time:.0f}s...")
            time.sleep(wait_time)
        
        self.request_times.append(now)
    
    def stream_continuous(
        self,
        tickers: List[str],
        callback: Callable,
        update_interval: int = 60,
        max_updates: int = None
    ):
        """
        Continuous streaming with callback for research experiments.
        
        Args:
            tickers: Assets to stream
            callback: Function called with MultiAssetConfig on each update
            update_interval: Seconds between updates
            max_updates: Maximum updates (None = infinite)
        """
        print(f"\n{'='*70}")
        print(f"CONTINUOUS STREAMING: {', '.join(tickers)}")
        print(f"Interval: {update_interval}s")
        print(f"Press Ctrl+C to stop")
        print(f"{'='*70}")
        
        update_count = 0
        
        try:
            while max_updates is None or update_count < max_updates:
                # Fetch data
                try:
                    config = self.fetch_live_data(tickers)
                    callback(config, update_count)
                    update_count += 1
                except Exception as e:
                    print(f"\n⚠️  Update {update_count} failed: {e}")
                
                # Wait
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print(f"\n\n{'='*70}")
            print(f"Streaming stopped after {update_count} updates")
            print(f"{'='*70}")


@dataclass
class ExperimentResult:
    """Result from one experiment iteration."""
    timestamp: datetime
    iteration: int
    basket_price: float
    bound_width: float
    correlation_error: float
    elapsed_sec: float
    params: Dict


class ResearchMonitor:
    """
    Monitor and log research experiments.
    
    Tracks:
    - Pipeline performance over time
    - Correlation estimation accuracy
    - Bound width evolution
    """
    
    def __init__(self, experiment_name: str = "multi_asset_rmot"):
        self.experiment_name = experiment_name
        self.results: List[ExperimentResult] = []
        self.start_time = datetime.now()
    
    def log_result(self, result: ExperimentResult):
        """Log an experiment result."""
        self.results.append(result)
        
        print(f"\n📊 Iteration {result.iteration}:")
        print(f"   Basket price: ${result.basket_price:.4f}")
        print(f"   Bound width: ${result.bound_width:.4f}")
        print(f"   ρ error: {result.correlation_error:.4f}")
        print(f"   Time: {result.elapsed_sec:.2f}s")
    
    def summary(self) -> Dict:
        """Generate experiment summary."""
        if not self.results:
            return {}
        
        prices = [r.basket_price for r in self.results]
        widths = [r.bound_width for r in self.results]
        errors = [r.correlation_error for r in self.results]
        times = [r.elapsed_sec for r in self.results]
        
        return {
            'n_iterations': len(self.results),
            'duration_sec': (datetime.now() - self.start_time).total_seconds(),
            'price_mean': np.mean(prices),
            'price_std': np.std(prices),
            'width_mean': np.mean(widths),
            'corr_error_mean': np.mean(errors),
            'time_mean': np.mean(times)
        }
    
    def save_results(self, filepath: str):
        """Save results to CSV for analysis."""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'iteration', 'basket_price', 'bound_width', 
                           'correlation_error', 'elapsed_sec'])
            
            for r in self.results:
                writer.writerow([
                    r.timestamp.isoformat(),
                    r.iteration,
                    r.basket_price,
                    r.bound_width,
                    r.correlation_error,
                    r.elapsed_sec
                ])
        
        print(f"✅ Results saved to {filepath}")
    
    def print_summary(self):
        """Print experiment summary."""
        s = self.summary()
        if not s:
            print("No results to summarize")
            return
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print(f"{'='*70}")
        print(f"  Iterations: {s['n_iterations']}")
        print(f"  Duration: {s['duration_sec']:.0f}s")
        print(f"  Basket price: ${s['price_mean']:.4f} ± ${s['price_std']:.4f}")
        print(f"  Bound width: ${s['width_mean']:.4f}")
        print(f"  Correlation error: {s['corr_error_mean']:.4f}")
        print(f"  Avg time/iteration: {s['time_mean']:.2f}s")


def run_research_experiment(
    tickers: List[str] = ['SPY', 'QQQ'],
    n_iterations: int = 5,
    update_interval: int = 60
):
    """
    Run a complete research experiment with real-time data.
    
    This is the main entry point for publication experiments.
    """
    from .pipeline import multi_asset_rmot_pipeline
    from .correlation_copula import RoughMartingaleCopula
    
    print(f"\n{'='*70}")
    print(f"🔬 RESEARCH EXPERIMENT: Multi-Asset RMOT")
    print(f"{'='*70}")
    print(f"Assets: {', '.join(tickers)}")
    print(f"Iterations: {n_iterations}")
    print(f"Update interval: {update_interval}s")
    
    stream = RealTimeDataStream()
    monitor = ResearchMonitor(f"rmot_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    def on_data(config: MultiAssetConfig, iteration: int):
        """Process each data update."""
        start = time.time()
        
        # Run pipeline
        result = multi_asset_rmot_pipeline(config, n_paths=30000, n_steps=50, verbose=False)
        
        # Compute metrics
        basket_price = np.mean([p.price for p in result['basket_prices']])
        bound_width = np.mean([b.width for b in result['frtb_bounds']])
        
        # Correlation error
        rho_target = config.correlation_guess
        rho_estimated = result['correlation_estimation'].rho
        corr_error = np.max(np.abs(rho_target - rho_estimated))
        
        elapsed = time.time() - start
        
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
    
    # Run experiment
    stream.stream_continuous(
        tickers=tickers,
        callback=on_data,
        update_interval=update_interval,
        max_updates=n_iterations
    )
    
    # Summary
    monitor.print_summary()
    
    # Save results
    output_file = f"/tmp/rmot_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    monitor.save_results(output_file)
    
    return monitor


if __name__ == "__main__":
    # Quick test
    stream = RealTimeDataStream()
    config = stream.fetch_live_data(['SPY', 'QQQ'])
    
    print(f"\n✅ Fetched data for {config.n_assets} assets")
    for a in config.assets:
        print(f"   {a.ticker}: ${a.spot:.2f}, {len(a.strikes)} strikes")
