
import numpy as np
from numba import njit
from math import gamma

@njit
def test_gamma(x):
    return gamma(x)

if __name__ == "__main__":
    try:
        print(f"Gamma(1.5) = {test_gamma(1.5)}")
    except Exception as e:
        print(f"Error: {e}")
