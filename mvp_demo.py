"""Compatibility import for MVP sessions."""
import sys
from bdd100k_eval import mvp_demo as _implementation
sys.modules[__name__] = _implementation
