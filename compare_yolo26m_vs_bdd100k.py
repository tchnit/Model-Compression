#!/usr/bin/env python3
"""Compatible command and import entry point; implementation lives in bdd100k_eval."""
import sys
import os
# Tự động nạp môi trường ảo .venv nếu chạy từ python hệ thống
VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
if os.path.exists(VENV_PYTHON) and sys.executable != VENV_PYTHON and "VIRTUAL_ENV" not in os.environ:
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

from bdd100k_eval.tools import compare_yolo26m_vs_bdd100k as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
