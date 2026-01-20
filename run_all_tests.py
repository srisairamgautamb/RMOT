
import subprocess
import sys
import os
import time

def run_test(script_path, description):
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"FILE: {script_path}")
    print(f"{'='*80}")
    
    start_time = time.time()
    try:
        # Run using same python interpreter
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=False 
        )
        duration = time.time() - start_time
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        if result.returncode == 0:
            print(f"\n✅ PASS ({duration:.2f}s)")
            return True, duration
        else:
            print(f"\n❌ FAIL (Return Code: {result.returncode})")
            return False, duration
            
    except Exception as e:
        print(f"\n❌ EXECUTION ERROR: {e}")
        return False, 0.0

def main():
    print("STARTING FULL PROJECT TEST SUITE")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    tests = [
        # 1. Core Unit Tests
        ("tests/test_rough_heston.py", "Unit Test: Rough Heston Simulator"),
        ("tests/test_malliavin.py", "Unit Test: Malliavin Derivs (Basic)"),
        ("tests/test_rmot_solver.py", "Unit Test: RMOT Solver Mechanics"),
        
        # 2. Critical Audit Verification Scripts
        ("tests/verify_solver_correctness.py", "Audit Validation: Solver Correctness (Self-Consistent)"),
        ("tests/verify_fisher.py", "Audit Validation: Fisher Information & Malliavin (Rigorous)"),
        
        # 3. New Validation Modules (P0/P2)
        ("src/validation/validate_real_market.py", "Validation: Real Market Data (SPX)"),
        ("src/validation/convergence_test.py", "Validation: Convergence Analysis"),
    ]
    
    results = []
    
    for script, desc in tests:
        if not os.path.exists(script):
            print(f"⚠️ SKIPPING MISSING FILE: {script}")
            results.append((desc, "MISSING", 0.0))
            continue
            
        success, duration = run_test(script, desc)
        status = "PASS" if success else "FAIL"
        results.append((desc, status, duration))
        
    print("\n" + "="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    print(f"{'Test Description':<60} | {'Status':<8} | {'Time':<8}")
    print("-" * 80)
    
    all_passed = True
    for desc, status, duration in results:
        print(f"{desc:<60} | {status:<8} | {duration:.2f}s")
        if status != "PASS":
            all_passed = False
            
    print("-" * 80)
    if all_passed:
        print("\n✅ ALL TESTS PASSED. PROJECT IS VERIFIED.")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
