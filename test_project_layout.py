"""Compatibility and import boundaries for the flat package."""
from pathlib import Path
import subprocess
import sys
import unittest


class ProjectLayoutTests(unittest.TestCase):
    def test_legacy_imports_share_canonical_modules_and_model_cache(self):
        import importlib
        mapping = {'evaluate': 'evaluation', 'run_demo': 'api', 'mvp_demo': 'mvp_demo',
                   'hardware_benchmark': 'hardware_benchmark',
                   **{name: 'tools.' + name for name in ('compare_benchmarks', 'compare_ptq_vs_qat',
                       'compare_yolo26m_vs_bdd100k', 'prepare_runtime_models', 'refresh_demo_data',
                       'verify_project', 'verify_hardware')}}
        for legacy, target in mapping.items():
            with self.subTest(module=legacy):
                self.assertIs(importlib.import_module(legacy), importlib.import_module('bdd100k_eval.' + target))
        import run_demo
        from bdd100k_eval import models
        self.assertIs(run_demo.MODEL_CONFIGS, models.MODEL_CONFIGS)
        self.assertIs(run_demo.LOADED_MODELS, models.LOADED_MODELS)
        self.assertIs(run_demo.WARMED_MODELS, models.WARMED_MODELS)

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
