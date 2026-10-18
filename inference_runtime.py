"""Compatibility import for the shared CPU runtime."""
import sys
from bdd100k_eval import inference_runtime as _implementation
sys.modules[__name__] = _implementation
