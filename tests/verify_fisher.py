
import numpy as np
import sys
import os

# Add root to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.rough_heston import RoughHestonParams
from src.calibration.fisher_information import FisherInformationAnalyzer

def verify_fisher():
    print("="*70)
    print("VERIFYING FISHER INFORMATION COMPUTATION")
    print("="*70)
    
    params = RoughHestonParams(
        H=0.1, eta=1.9, rho=-0.7, xi0=0.04,
        kappa=1.5, theta=0.04, S0=100.0, r=0.0
    )
    
    analyzer = FisherInformationAnalyzer()
    
    strikes = np.linspace(80, 120, 50)
    T = 1/12
    
    try:
        results = analyzer.validate_identifiability(strikes, T, params=params, n_paths=5000)
        
        print("\nFisher Information Results:")
        print(f"Number of Strikes: {results['n_strikes']}")
        print(f"Effective Dimension: {results['d_eff']}")
        print(f"Recommendation: {results['recommendation']}")
        
        print("\nCramér-Rao Bounds:")
        cr = results['cramèr_rao_bounds']
        for k, v in cr.items():
            print(f"  {k}: {v:.6f}")
            
        print("\nEigenvalues:")
        print(results['eigenvalues'])
        
        # Validation checks
        if cr['std_H'] < 0.1:
            print("\n✅ H is identifiable (std < 0.1)")
        else:
            print("\n⚠️ H is weakly identifiable")
            
        if results['d_eff'] >= 3:
            print("✅ Effective dimension sufficient")
        else:
            print("⚠️ Low effective dimension")
            
    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")
        raise

if __name__ == "__main__":
    verify_fisher()
