"""
JIT liquidity optimization algorithms.

- Utility: shared scoring context + `optimize(method=...)` entry point.
- CombinatorialOptimizer: brute-force search scored by swap simulation.
- AnalyticalOptimizer: closed-form solution (Lemmas 5.1/5.2, ported from MATLAB).
"""

from optimization.utility import Utility
from optimization.combinatorial import CombinatorialOptimizer
from optimization.analytical import AnalyticalOptimizer

__all__ = ["Utility", "CombinatorialOptimizer", "AnalyticalOptimizer"]
