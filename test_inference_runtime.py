"""Integration check: verify the real tensors and runtime, not configuration labels."""
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from inference_runtime import cpu_device, configure_cpu_backend


class CPURuntimeTests(unittest.TestCase):
    def clamped_backend(self, threads=2, streams=1, devices=None):
        properties = {'INFERENCE_NUM_THREADS': threads, 'NUM_STREAMS': streams,
                      'EXECUTION_DEVICES': devices or ['CPU']}
        compiled = SimpleNamespace(get_property=properties.__getitem__,
                                   input=lambda: SimpleNamespace(shape=[1, 3, 640, 640]))
        return SimpleNamespace(backend=SimpleNamespace(ov_compiled_model=compiled)), compiled

    def test_demo_reports_clamped_threads_but_benchmark_stays_strict(self):
        model, compiled = self.clamped_backend()
        core = Mock()
        core.compile_model.return_value = compiled
        with patch('openvino.Core', return_value=core):
            runtime = configure_cpu_backend(model, 'model.xml', 4, strict_threads=False)
            self.assertEqual(runtime['requested_threads'], 4)
            self.assertEqual(runtime['ov_threads'], 2)
            self.assertEqual(runtime['ov_streams'], 1)
            self.assertFalse(runtime['thread_protocol_matches'])
            self.assertIn('thực tế 2 luồng', runtime['thread_warning'])
            self.assertEqual(core.compile_model.call_args.kwargs['config']['INFERENCE_NUM_THREADS'], 4)
            with self.assertRaisesRegex(RuntimeError, 'thực tế 2 luồng'):
                configure_cpu_backend(model, 'model.xml', 4)

    def test_demo_still_rejects_wrong_device_streams_or_excess_threads(self):
        for props in [dict(devices=['GPU']), dict(streams=2), dict(threads=0), dict(threads=8)]:
            with self.subTest(props=props):
                model, compiled = self.clamped_backend(**props)
                core = Mock()
                core.compile_model.return_value = compiled
                with patch('openvino.Core', return_value=core):
                    with self.assertRaisesRegex(RuntimeError, 'không hợp lệ'):
                        configure_cpu_backend(model, 'model.xml', 4, strict_threads=False)

    @unittest.skipUnless(hasattr(os, 'sched_setaffinity'), 'Requires Linux CPU affinity')
    def test_real_mvp_on_two_cpus_and_strict_benchmark_rejection(self):
        script = '''
import os
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
import io, json, time
from pathlib import Path
import run_demo
from inference_runtime import configure_cpu_backend
service = run_demo.MVP
data = next(Path('demo_data/images').glob('*.jpg')).read_bytes()
for iteration in range(2):
    task = service.submit(io.BytesIO(data), len(data), {'filename':['test.jpg'], 'classes':['0,1,2,3,4,5,6,7,8,9']})
    job = service.get(task['id'])
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state = service.snapshot(job)
        if state['status'] not in ('starting', 'running'):
            break
        time.sleep(.05)
    assert state['status'] == 'completed', state
    report = json.loads((job['folder'] / 'report.json').read_text())
    runtime = report['runtime']
    assert runtime['execution_devices'] == ['CPU'], runtime
    assert runtime['requested_threads'] == 2 and runtime['ov_threads'] == 2, runtime
    assert runtime['thread_protocol_matches'] is True and 'thread_warning' not in runtime, runtime
    assert report['processed'] == 1 and report['inference_mean_ms'] > 0, report
    if iteration:
        assert report['timings']['initialization_ms'] == 0, report
    print(json.dumps({'runtime': runtime, 'processed': report['processed'], 'timings': report['timings']}), flush=True)
model = run_demo.LOADED_MODELS['yolo26s_ptq_int8']
try:
    configure_cpu_backend(model, service.model_path, 4)
except RuntimeError as error:
    assert 'thực tế' in str(error)
else:
    raise AssertionError('Benchmark incorrectly accepted a clamped runtime')
service.close()
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).parent,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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
