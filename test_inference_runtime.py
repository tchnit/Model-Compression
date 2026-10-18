"""Integration check: verify the real tensors and runtime, not configuration labels."""
from pathlib import Path
import unittest

from inference_runtime import cpu_device, configure_cpu_backend


class CPURuntimeTests(unittest.TestCase):
    @unittest.skipUnless(Path('models/yolo26m-bdd100k.pt').is_file(), 'Requires local model artifacts')
    def test_pytorch_and_openvino_share_square_input_and_four_threads(self):
        import numpy as np
        import torch
        from ultralytics import YOLO
        for path in ['models/yolo26m-bdd100k.pt', 'models/int8_yolo26m_640_openvino_model']:
            with self.subTest(path=path):
                model = YOLO(path, task='detect')
                options = dict(imgsz=640, rect=False, device=cpu_device(path), verbose=False)
                image = np.zeros((720, 1280, 3), dtype=np.uint8)
                model(image, **options)
                if Path(path).is_dir():
                    import openvino as ov
                    core = ov.Core()
                    backend = model.predictor.model.backend
                    graph = core.read_model(next(Path(path).glob('*.xml')))
                    graph.reshape(list(backend.ov_compiled_model.input().shape))
                    backend.ov_compiled_model = core.compile_model(graph, 'CPU',
                        {'INFERENCE_NUM_THREADS': 4, 'NUM_STREAMS': 2})
                runtime = configure_cpu_backend(model, path, 4)
                model(image, **options)
                self.assertEqual(tuple(model.predictor.preprocess([image]).shape), (1, 3, 640, 640))
                self.assertEqual(torch.get_num_threads(), 4)
                if Path(path).suffix == '.pt':
                    self.assertEqual(runtime['parameter_dtype'], 'torch.float16')
                    self.assertTrue(model.predictor.model.fp16)
                    self.assertEqual(model.predictor.preprocess([image]).dtype, torch.float16)
                    self.assertTrue(torch.isfinite(model(image, **options)[0].boxes.data).all())
                    configure_cpu_backend(model, path, 4, 'FP32')
                    self.assertFalse(model.predictor.model.fp16)
                    self.assertEqual(model.predictor.preprocess([image]).dtype, torch.float32)
                if Path(path).is_dir():
                    self.assertEqual(runtime['ov_threads'], 4)
                    self.assertEqual(runtime['execution_devices'], ['CPU'])
                    self.assertEqual(runtime['ov_streams'], 1)
