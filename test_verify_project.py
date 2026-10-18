from pathlib import Path
import tempfile
import unittest

from verify_project import inspect_artifact, check_rerun


class ArtifactInventoryTests(unittest.TestCase):
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
