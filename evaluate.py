#!/usr/bin/env python3
"""Compatible command and import entry point; implementation lives in bdd100k_eval."""
import sys
from bdd100k_eval import evaluation as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
