"""Compatibility and import boundaries for the flat package."""
from pathlib import Path
import subprocess
import sys
import unittest


class ProjectLayoutTests(unittest.TestCase):
    def test_cpu_runtime_compatibility_import_is_the_same_module(self):
        import inference_runtime
        from bdd100k_eval import inference_runtime as implementation
        self.assertIs(inference_runtime, implementation)

    def test_offline_tools_do_not_initialize_the_server(self):
        result = subprocess.run([sys.executable, '-c', '''
import sys, threading
import refresh_demo_data, verify_project
from bdd100k_eval import models
assert 'run_demo' not in sys.modules
assert 'bdd100k_eval.api' not in sys.modules
assert not models.LOADED_MODELS
assert threading.active_count() == 1
'''], cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
