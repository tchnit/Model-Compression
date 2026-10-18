#!/usr/bin/env python3
"""Compatible demo command and server API imports."""
import sys
import os
# Auto re-execute with .venv python if available
VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'bin', 'python')
if os.path.exists(VENV_PYTHON) and sys.executable != VENV_PYTHON and 'VIRTUAL_ENV' not in os.environ:
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

from bdd100k_eval import api as _implementation

if __name__ == "__main__":
    _implementation.run()
else:
    sys.modules[__name__] = _implementation
