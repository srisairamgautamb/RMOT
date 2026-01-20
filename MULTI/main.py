"""
Multi-Asset RMOT Main Entry Point

Demonstrates the full pipeline with REAL market data from CBOE via yfinance.
Uses SPX and NDX options for a 2-asset basket.
"""

import numpy as np
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_structures import (
    RoughHestonParams, AssetConfig, MultiAssetConfig
)
from src.pipeline import multi_asset_rmot_pipeline


# =====================================================================
# REAL MARKET DATA DOWNLOAD
# =====================================================================

def download_options_data(ticker: str, target_days_out: int = 30):
    """
    Download real options data from CBOE via yfinance.
    
    Args:
        ticker: '^SPX', '^NDX', 'SPY', 'QQQ', etc.
        target_days_out: Target days to expiration
    
    Returns:
        AssetConfig with real market data
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required. Install: pip install yfinance")
    
    print(f"Downloading {ticker} options data...")
    
    # Try to get ticker data
    stock = yf.Ticker(ticker)
    
    # Get spot price
    hist = stock.history(period='5d')
    if hist.empty:
        raise ValueError(f"Could not fetch price history for {ticker}")
    spot = hist['Close'].iloc[-1]
    print(f"  {ticker} spot: ${spot:.2f}")
    
    # Get options expiries
    try:
        expiries = stock.options
    except Exception as e:
        raise ValueError(f"Could not fetch options chain for {ticker}: {e}")
    
    if not expiries:
        raise ValueError(f"No options expiries available for {ticker}")
    
    # Find expiry close to target
    from datetime import datetime, timedelta
    today = datetime.now()
    target_date = today + timedelta(days=target_days_out)
    
    best_expiry = None
    min_diff = float('inf')
    for exp_str in expiries:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        diff = abs((exp_date - target_date).days)
        if diff < min_diff:
            min_diff = diff
            best_expiry = exp_str
    
    print(f"  Selected expiry: {best_expiry} ({min_diff} days from target)")
    
    # Get options chain
    chain = stock.option_chain(best_expiry)
    calls = chain.calls
    
    # Filter liquid options
    liquid = calls[(calls['bid'] > 0) & (calls['ask'] > 0)].copy()
    liquid['mid'] = (liquid['bid'] + liquid['ask']) / 2
    
    # Filter by spread
    liquid = liquid[(liquid['ask'] - liquid['bid']) / liquid['bid'] < 0.20]
    
    # Filter by moneyness [0.90, 1.10]
    liquid = liquid[(liquid['strike'] > 0.90 * spot) & (liquid['strike'] < 1.10 * spot)]
    
    if len(liquid) < 5:
        raise ValueError(f"Not enough liquid strikes for {ticker}: {len(liquid)}")
    
    # Calculate maturity
    exp_date = datetime.strptime(best_expiry, "%Y-%m-%d")
    T = max((exp_date - today).days / 365.0, 1/365)
    
    strikes = liquid['strike'].values
    prices = liquid['mid'].values
    
    print(f"  Found {len(strikes)} liquid strikes")
    print(f"  Strike range: [{strikes.min():.1f}, {strikes.max():.1f}]")
    print(f"  Maturity: {T*365:.1f} days ({T:.4f} years)")
    
    return AssetConfig(
        ticker=ticker,
        spot=spot,
        strikes=strikes,
        market_prices=prices,
        maturity=T
    )


def create_real_market_config():
    """
    Create MultiAssetConfig with real SPX and NDX data.
    
    Returns:
        MultiAssetConfig with live market data
    """
    print("\n" + "=" * 70)
    print("DOWNLOADING REAL MARKET DATA")
    print("=" * 70)
    
    # Download SPX and NDX/QQQ options
    assets = []
    
    # Asset 1: SPX (or SPY as proxy)
    try:
        asset1 = download_options_data('^SPX', target_days_out=30)
    except Exception as e:
        print(f"SPX failed: {e}, trying SPY...")
        asset1 = download_options_data('SPY', target_days_out=30)
    assets.append(asset1)
    
    # Asset 2: NDX (or QQQ as proxy)
    try:
        asset2 = download_options_data('^NDX', target_days_out=30)
    except Exception as e:
        print(f"NDX failed: {e}, trying QQQ...")
        asset2 = download_options_data('QQQ', target_days_out=30)
    assets.append(asset2)
    
    # Use common maturity (minimum)
    common_T = min(a.maturity for a in assets)
    for a in assets:
        a.maturity = common_T
    
    # Basket weights (equal weighted)
    weights = np.array([0.5, 0.5])
    
    # Initial correlation guess (historical is ~0.85 for SPX/NDX)
    rho_guess = np.array([
        [1.0, 0.85],
        [0.85, 1.0]
    ])
    
    # Basket strikes (around weighted average spot)
    basket_spot = sum(w * a.spot for w, a in zip(weights, assets))
    basket_strikes = np.array([
        basket_spot * 0.95,
        basket_spot * 0.98,
        basket_spot,
        basket_spot * 1.02,
        basket_spot * 1.05
    ])
    
    print(f"\n  Basket spot: ${basket_spot:.2f}")
    print(f"  Basket strikes: {basket_strikes}")
    
    return MultiAssetConfig(
        assets=assets,
        basket_weights=weights,
        correlation_guess=rho_guess,
        basket_strikes=basket_strikes
    )


# =====================================================================
# MAIN
# =====================================================================

def main():
    """Main entry point for Multi-Asset RMOT with real market data."""
    print("\n" + "=" * 70)
    print("🚀 MULTI-ASSET RMOT WITH REAL MARKET DATA")
    print("=" * 70)
    
    try:
        # Create configuration with real data
        config = create_real_market_config()
        
        # Run pipeline
        result = multi_asset_rmot_pipeline(
            config,
            n_paths=30000,
            n_steps=50,
            verbose=True
        )
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 FINAL SUMMARY")
        print("=" * 70)
        
        print("\nMarginal Parameters:")
        for i, p in enumerate(result['marginal_calibration'].params):
            asset = config.assets[i]
            print(f"  {asset.ticker}: H={p.H:.4f}, η={p.eta:.4f}, ρ={p.rho:.4f}, ξ₀={p.xi0:.6f}")
        
        print("\nCorrelation Matrix:")
        print(result['correlation_estimation'].rho)
        
        print("\nBasket Option Prices:")
        for bp in result['basket_prices']:
            print(f"  K={bp.strike:10.2f}: Price=${bp.price:8.3f} ± ${bp.std_error:.3f}")
        
        print("\nFRTB Capital Charges:")
        total_capital = 0
        for fb in result['frtb_bounds']:
            print(f"  Width=${fb.width:8.3f}, Capital=${fb.capital_charge:8.3f}")
            total_capital += fb.capital_charge
        print(f"  TOTAL CAPITAL: ${total_capital:.2f}")
        
        print(f"\nTotal runtime: {result['elapsed_time']:.2f}s")
        print("\n✅ MULTI-ASSET RMOT COMPLETE")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
