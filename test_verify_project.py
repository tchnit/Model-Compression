from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from verify_project import inspect_artifact, check_rerun, check_requirements


class ArtifactInventoryTests(unittest.TestCase):
    def test_dependency_profile_accepts_cpu_cuda_wheel_suffix_and_rejects_wrong_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'requirements.txt'
            path.write_text('# CPU profile\n--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.14.1\n')
            for actual in ['2.14.1+cpu', '2.14.1+cu130']:
                with patch('verify_project.importlib.metadata.version', return_value=actual):
                    self.assertEqual(check_requirements(path), {'torch': actual})
            with patch('verify_project.importlib.metadata.version', return_value='2.13.0'):
                with self.assertRaisesRegex(ValueError, 'Environment mismatch'):
                    check_requirements(path)

    def test_missing_rerun_is_not_reported_as_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(check_rerun(Path(directory))['status'], 'not_completed')

    def test_stored_half_dtype_is_not_confused_with_runtime_float32(self):
        import torch
        from evaluate import DEFAULT_NAMES
        model = torch.nn.Linear(2, 10).half()
        model.names = DEFAULT_NAMES
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            torch.save({"model": model}, path)
            record = inspect_artifact(path)
        self.assertTrue(record['bdd100k_compatible'])
        self.assertEqual(record['stored_tensor_elements_by_dtype'], {'torch.float16': 30})
        self.assertFalse(record['training_provenance_verified'])

    def test_coco_model_is_not_marked_bdd100k_compatible(self):
        import torch
        model = torch.nn.Linear(2, 80)
        model.names = [f'class{i}' for i in range(80)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coco.pt"
            torch.save({"model": model}, path)
            record = inspect_artifact(path)
        self.assertFalse(record['bdd100k_compatible'])


if __name__ == '__main__':
    unittest.main()
