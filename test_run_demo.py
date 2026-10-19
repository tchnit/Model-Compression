import base64
import builtins
import http.client
import json
import os
import re
from io import BytesIO
from pathlib import Path
import socketserver
import threading
import unittest
import tempfile
from unittest.mock import patch

from PIL import Image

import run_demo


class RuntimeStartupTests(unittest.TestCase):
    def test_unsupported_python_stops_before_binding_port(self):
        with patch.object(run_demo.sys, 'version_info', (3, 10)), \
                patch.object(run_demo.socketserver, 'ThreadingTCPServer') as server:
            with self.assertRaisesRegex(SystemExit, 'Python >= 3.12'):
                run_demo.run()
        server.assert_not_called()

    def test_missing_runtime_stops_before_binding_port(self):
        with patch.object(run_demo.importlib, 'import_module', side_effect=ModuleNotFoundError('numpy')), \
                patch.object(run_demo.socketserver, 'ThreadingTCPServer') as server:
            with self.assertRaisesRegex(SystemExit, 'pip install -r requirements.txt'):
                run_demo.run()
        server.assert_not_called()

    def test_missing_libgl_reports_system_package_instead_of_pip(self):
        for error_type in (ImportError, OSError):
            with self.subTest(error_type=error_type):
                def import_without_libgl(module):
                    if module == 'cv2':
                        raise error_type('libGL.so.1: cannot open shared object file: No such file or directory')

                with patch.object(run_demo.importlib, 'import_module', side_effect=import_without_libgl), \
                        patch.object(run_demo.socketserver, 'ThreadingTCPServer') as server:
                    with self.assertRaises(SystemExit) as raised:
                        run_demo.run()
                self.assertIn('apt-get update && apt-get install -y libgl1', str(raised.exception))
                self.assertNotIn('pip install', str(raised.exception))
                server.assert_not_called()


class DemoServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), run_demo.StudioRequestHandler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        payload = response.read()
        conn.close()
        return response.status, payload

    def test_only_demo_assets_are_public(self):
        self.assertEqual(self.request("GET", "/demo.html")[0], 200)
        self.assertEqual(self.request("HEAD", "/demo_data/detections.json")[0], 200)
        self.assertEqual(self.request("HEAD", "/benchmark_comparison.csv")[0], 200)
        self.assertEqual(self.request("GET", "/PROGRESS_REPORT.md")[0], 404)
        self.assertEqual(self.request("GET", "/bdd100k_eval/config.py")[0], 404)
        self.assertEqual(self.request("GET", "/demo_data/../PROGRESS_REPORT.md")[0], 404)
        self.assertEqual(self.request("HEAD", "/models/yolo26m-bdd100k.pt")[0], 404)
        with tempfile.TemporaryDirectory() as folder:
            exports = Path(folder) / 'runtime_exports'; exports.mkdir()
            for model in ['yolo26s','yolo26m']:
                (exports / f'{model}_int8.onnx').write_bytes(b'public artifact fixture')
            with patch.object(run_demo,'DIRECTORY',folder):
                self.assertEqual(self.request('HEAD','/runtime_exports/yolo26s_int8.onnx')[0],200)
                self.assertEqual(self.request('HEAD','/runtime_exports/yolo26m_int8.onnx')[0],200)
        self.assertEqual(self.request('GET','/verification/browser_int8_calibration.json')[0],404)

    def test_dashboard_policy_and_static_paths_allow_extracted_assets(self):
        status, body = self.request('GET', '/model-compression-yolo26-dashboard.html')
        self.assertEqual(status, 200)
        policy = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', body.decode())[1]
        self.assertIn("script-src 'self'", policy)
        self.assertIn("style-src 'self' 'unsafe-inline'", policy)
        self.assertIn("connect-src 'none'", policy)
        for asset in ('dashboard.js', 'dashboard.css', 'studio.js', 'studio.css', 'overlays.js'):
            with self.subTest(asset=asset):
                self.assertEqual(self.request('GET', '/demo_data/' + asset)[0], 200)

    def test_wasm_shared_memory_headers(self):
        for method, path in [('GET','/demo.html'), ('HEAD','/demo_data/hardware_benchmark.js'),
                             ('HEAD','/demo_data/vendor/onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs')]:
            with self.subTest(method=method,path=path):
                conn = http.client.HTTPConnection('127.0.0.1',self.server.server_address[1],timeout=5)
                conn.request(method,path)
                response = conn.getresponse()
                self.assertEqual(response.status,200)
                self.assertEqual(response.getheader('Cross-Origin-Opener-Policy'),'same-origin')
                self.assertEqual(response.getheader('Cross-Origin-Embedder-Policy'),'require-corp')
                response.read()
                conn.close()

    def test_live_comparison_pins_every_model_to_the_same_cpu_runtime(self):
        available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
        self.assertEqual(run_demo.CPU_THREADS, min(2, available))
        self.assertTrue(run_demo.MODEL_CONFIGS)
        self.assertEqual(len(run_demo.MODEL_CONFIGS), 10)
        self.assertEqual({cfg["device"] for cfg in run_demo.MODEL_CONFIGS.values()}, {"cpu"})
        self.assertEqual(
            Path(run_demo.MODEL_CONFIGS["yolo26s_int8"]["path"]).name,
            "int8_yolo26s_qat_640_openvino_model",
        )
        self.assertEqual(Path(run_demo.MODEL_CONFIGS["yolo26s_fp32"]["path"]).name, "yolo26s-bdd100k.pt")
        self.assertEqual(Path(run_demo.MODEL_CONFIGS["yolo26s_ptq_int8"]["path"]).name, "int8_yolo26s_640_openvino_model")
        self.assertEqual(Path(run_demo.MODEL_CONFIGS["yolo26m_qat_int8"]["path"]).name, "int8_yolo26m_qat_640_openvino_model")
        self.assertEqual(
            {key: run_demo.MODEL_CONFIGS[key].get('precision') for key in
             ('yolo26m_fp32', 'yolo26s_fp32', 'yolo26s_fp32_pack')},
            {'yolo26m_fp32': 'FP32', 'yolo26s_fp32': 'FP32', 'yolo26s_fp32_pack': 'FP32'},
        )
        self.assertNotIn("yolo26s_ptq_480", run_demo.MODEL_CONFIGS)
        self.assertNotIn("yolo26m_ptq_480", run_demo.MODEL_CONFIGS)

    def test_built_in_scene_fallbacks_use_the_same_cpu_runtime(self):
        data = json.loads(Path("demo_data/detections.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["scenes"]), 5)
        for scene in data["scenes"]:
            self.assertTrue(scene["data_uri"].startswith("data:image/jpeg;base64,"))
            self.assertTrue((Path("demo_data/images") / scene["file"]).is_file())
        self.assertEqual(set(data["models"]), set(run_demo.MODEL_CONFIGS))
        self.assertEqual(set(data["detections"]), set(run_demo.MODEL_CONFIGS))
        bundle = Path("demo_data/demo_bundle.js").read_text(encoding="utf-8")
        self.assertEqual(json.loads(bundle.removeprefix("window.DEMO_DATA = ").rstrip(";\n")), data)
        dashboard = Path('model-compression-yolo26-dashboard.html').read_text()
        summary = re.search(r'<script id="benchmarkSummaryData" type="application/json">(.*?)</script>', dashboard, re.S)[1]
        self.assertEqual(json.loads(summary), data['benchmarks'])
        self.assertEqual(data['scenes'][1]['attributes']['timeofday'], 'night')
        for model_key in run_demo.MODEL_CONFIGS:
            records = data["detections"][model_key]
            self.assertEqual(len(records), 5)
            self.assertTrue(all(row["runtime"] == "CPU ×4 · 640×640" for row in records.values()))
            if run_demo.MODEL_CONFIGS[model_key]['path'].endswith('.pt'):
                precision = run_demo.MODEL_CONFIGS[model_key].get('precision', 'FP16')
                dtype = 'torch.float32' if precision == 'FP32' else 'torch.float16'
                self.assertTrue(all(row['runtime_settings']['parameter_dtype'] == dtype for row in records.values()))
        for model_key in ('yolo26m_fp32', 'yolo26s_fp32', 'yolo26s_fp32_pack'):
            self.assertIn('FP16', data['models'][model_key]['name'])

    def test_fp16_display_baseline_runs_with_fp32_tensors(self):
        image = BytesIO()
        Image.new('RGB', (1, 1)).save(image, 'JPEG')
        encoded = base64.b64encode(image.getvalue()).decode()

        class Result:
            boxes = []
            names = {}

        class Model:
            def __call__(self, *args, **kwargs):
                return [Result()]

        config = {'path': 'models/yolo26m-bdd100k.pt', 'task': None, 'device': 'cpu', 'precision': 'FP32'}
        def runtime_for(_model, _path, threads, _precision, *, strict_threads):
            self.assertFalse(strict_threads)
            return {'device': 'CPU', 'torch_threads': threads, 'precision': 'FP32',
                    'parameter_dtype': 'torch.float32', 'input_dtype': 'torch.float32'}

        with patch.dict(run_demo.MODEL_CONFIGS, {'yolo26m_fp32': config}, clear=True), \
                patch.dict(run_demo.LOADED_MODELS, {}, clear=True), \
                patch.object(run_demo, 'get_model', return_value=Model()), \
                patch.object(run_demo, 'configure_cpu_backend', side_effect=runtime_for) as configure, \
                patch.object(run_demo, 'cpu_device', return_value='cpu'):
            run_demo.WARMED_MODELS.clear()
            run_demo.MODEL_RUNTIMES.clear()
            status, body = self.request('POST', '/api/predict',
                                        json.dumps({'image': encoded, 'models': ['yolo26m_fp32'], 'threads': 1}),
                                        {'Content-Type': 'application/json'})
            status_2, body_2 = self.request('POST', '/api/predict',
                                            json.dumps({'image': encoded, 'models': ['yolo26m_fp32'], 'threads': 2}),
                                            {'Content-Type': 'application/json'})
        self.assertEqual(status, 200)
        self.assertEqual(status_2, 200)
        self.assertEqual([item.args[2] for item in configure.call_args_list], [1, 2])
        result = json.loads(body)['detections']['yolo26m_fp32']
        result_2 = json.loads(body_2)['detections']['yolo26m_fp32']
        self.assertEqual(result['runtime_settings']['precision'], 'FP32')
        self.assertEqual(result['runtime_settings']['parameter_dtype'], 'torch.float32')
        self.assertEqual(result['runtime_settings']['torch_threads'], 1)
        self.assertEqual(result_2['runtime_settings']['torch_threads'], 2)

    def test_predict_rejects_oversized_and_unknown_model_requests(self):
        status, _ = self.request(
            "POST",
            "/api/predict",
            b"{}",
            {"Content-Type": "application/json", "Content-Length": str(run_demo.MAX_REQUEST_BYTES + 1)},
        )
        self.assertEqual(status, 413)

        image = BytesIO()
        Image.new("RGB", (1, 1)).save(image, "JPEG")
        payload = json.dumps({
            "image": base64.b64encode(image.getvalue()).decode(),
            "models": ["not-a-model"],
        }).encode()
        status, body = self.request("POST", "/api/predict", payload, {"Content-Type": "application/json"})
        self.assertEqual(status, 400)
        self.assertIn("không hợp lệ", json.loads(body)["message"])

    def test_malformed_uploads_are_client_errors(self):
        for image in [None, 12, [], {}, 'AA==']:
            with self.subTest(image=image):
                status, _ = self.request('POST', '/api/predict', json.dumps({'image': image}),
                                         {'Content-Type': 'application/json'})
                self.assertEqual(status, 400)

    def test_missing_model_artifacts_are_reported_without_loading_models(self):
        image = BytesIO()
        Image.new('RGB', (1, 1)).save(image, 'JPEG')
        encoded = base64.b64encode(image.getvalue()).decode()
        with tempfile.TemporaryDirectory() as folder:
            configs = {
                key: {**run_demo.MODEL_CONFIGS[key], 'path': str(Path(folder) / 'models' / artifact)}
                for key, artifact in [('yolo26m_fp32', 'yolo26m-bdd100k.pt'),
                                      ('yolo26m_ptq_int8', 'int8_yolo26m_640_openvino_model')]
            }
            with patch.dict(run_demo.MODEL_CONFIGS, configs, clear=True), \
                    patch.dict(run_demo.LOADED_MODELS, {}, clear=True), patch.object(run_demo, 'DIRECTORY', folder):
                _, body = self.request('GET', '/api/status')
                inventory = json.loads(body)
                self.assertEqual(inventory['cpu_threads'], run_demo.CPU_THREADS)
                self.assertEqual(inventory['models_available'], [])
                self.assertEqual(set(inventory['models_missing']), set(configs))
                for key, cfg in configs.items():
                    with self.subTest(model=key):
                        status, body = self.request('POST', '/api/predict',
                                                   json.dumps({'image': encoded, 'models': [key]}),
                                                   {'Content-Type': 'application/json'})
                        error = json.loads(body)
                        self.assertEqual(status, 503)
                        self.assertEqual(error['code'], 'model_artifact_missing')
                        self.assertIn('models/' + Path(cfg['path']).name, error['message'])
                        self.assertIn('git pull', error['message'])
                        self.assertEqual(run_demo.LOADED_MODELS, {})
                self.assertTrue(run_demo.INFERENCE_SLOT.acquire(blocking=False))
                run_demo.INFERENCE_SLOT.release()

    def test_missing_dependencies_return_json_instead_of_dropping_connection(self):
        original_import = builtins.__import__
        for missing in ['numpy', 'PIL']:
            with self.subTest(missing=missing):
                def import_without_dependency(name, *args, **kwargs):
                    if name == missing:
                        raise ModuleNotFoundError(f"No module named '{missing}'")
                    return original_import(name, *args, **kwargs)

                with patch('builtins.__import__', side_effect=import_without_dependency), patch('sys.stderr'):
                    status, body = self.request('POST', '/api/predict', '{}',
                                                {'Content-Type': 'application/json'})
                self.assertEqual(status, 500)
                self.assertEqual(json.loads(body)['status'], 'error')

    def test_decompression_bombs_remain_client_errors(self):
        with patch.object(Image, 'open', side_effect=Image.DecompressionBombError('too large')):
            status, _ = self.request('POST', '/api/predict', json.dumps({'image': 'AA=='}),
                                     {'Content-Type': 'application/json'})
        self.assertEqual(status, 400)

    def test_malformed_models_and_confidence_are_client_errors(self):
        image = BytesIO()
        Image.new('RGB', (1, 1)).save(image, 'JPEG')
        encoded = base64.b64encode(image.getvalue()).decode()
        for fields in [{'models': [{}]}, {'models': [None]}, {'conf': None}, {'conf': []},
                       {'threads': 0}, {'threads': 3}, {'threads': '2'}, {'threads': True}]:
            with self.subTest(fields=fields):
                payload = json.dumps({'image': encoded, **fields})
                status, _ = self.request('POST', '/api/predict', payload, {'Content-Type': 'application/json'})
                self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
