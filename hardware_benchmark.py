"""Compatibility import for native hardware benchmarks."""
import sys
from bdd100k_eval import hardware_benchmark as _implementation
sys.modules[__name__] = _implementation
