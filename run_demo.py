#!/usr/bin/env python3
"""
YOLO26 x BDD100K Model Compression Studio & Live Inference Server.
Provides:
  - Static file serving (demo.html, dashboard, assets)
  - Live AI Inference API on user-uploaded images: POST /api/predict
  - Status API: GET /api/status
"""
import os
import sys

# Auto re-execute with .venv python if available
VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'bin', 'python')
if os.path.exists(VENV_PYTHON) and sys.executable != VENV_PYTHON and 'VIRTUAL_ENV' not in os.environ:
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

import http.server
import importlib
import socketserver
import json
import base64
import threading
import time
from urllib.parse import unquote, urlsplit
from inference_runtime import cpu_device, configure_cpu_backend
import hardware_benchmark as hardware_bench
from mvp_demo import MVPService, MODEL_KEY as MVP_MODEL_KEY

HOST = "0.0.0.0"
PORT = 8765
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CPU_THREADS = 4
MAX_REQUEST_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
PUBLIC_FILES = {
    "/demo.html",
    "/model-compression-yolo26-dashboard.html",
    "/benchmark_comparison.md",
    "/benchmark_comparison.csv",
}
INFERENCE_SLOT = threading.BoundedSemaphore(1)

# The OpenVINO exports run on CPU.  Pin the PyTorch reference to the same
# device, otherwise Ultralytics will silently select CUDA for `.pt` models and
# the side-by-side latency comparison becomes meaningless.
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))

# Model definitions
MODEL_CONFIGS = {
    'yolo26m_fp32': {
        'path': os.path.join(DIRECTORY, 'models', 'yolo26m-bdd100k.pt'),
        'task': None,
        'device': 'cpu',
    },
    'yolo26m_ptq_int8': {
        'path': os.path.join(DIRECTORY, 'models', 'int8_yolo26m_640_openvino_model'),
        'task': 'detect',
        'device': 'cpu',
    },
    'yolo26m_qat_int8': {
        'path': os.path.join(DIRECTORY, 'models', 'int8_yolo26m_qat_640_openvino_model'),
        'task': 'detect',
        'device': 'cpu',
    },
    'yolo26s_int8': {
        'path': os.path.join(DIRECTORY, 'models', 'int8_yolo26s_qat_640_openvino_model'),
        'task': 'detect',
        'device': 'cpu',
    },
    'yolo26s_fp32': {
        'path': os.path.join(DIRECTORY, 'models', 'yolo26s-bdd100k.pt'),
        'task': None,
        'device': 'cpu',
    },
    'yolo26s_ptq_int8': {
        'path': os.path.join(DIRECTORY, 'models', 'int8_yolo26s_640_openvino_model'),
        'task': 'detect',
        'device': 'cpu',
    },
    'yolo26s_fp32_pack': {
        'path': os.path.join(DIRECTORY, 'models', 'fp32.pt'),
        'task': None,
        'device': 'cpu',
    },
    'yolo26s_fp16_pt': {
        'path': os.path.join(DIRECTORY, 'models', 'fp16.pt'),
        'task': None,
        'device': 'cpu',
    },
    'yolo26s_fp16_ov': {
        'path': os.path.join(DIRECTORY, 'models', 'fp16_openvino_model'),
        'task': 'detect',
        'device': 'cpu',
    },
    'yolo26s_export_int8_640': {
        'path': os.path.join(DIRECTORY, 'models', 'int8_yolo26s-bdd100k_640_openvino_model'),
        'task': 'detect',
        'device': 'cpu',
    }
}

LOADED_MODELS = {}
WARMED_MODELS = set()
MODEL_RUNTIMES = {}


def configure_cpu_threads():
    """Restore the PyTorch CPU budget after Ultralytics initializes a predictor."""
    import torch
    torch.set_num_threads(CPU_THREADS)

def get_model(model_key):
    global LOADED_MODELS
    if model_key not in LOADED_MODELS:
        cfg = MODEL_CONFIGS.get(model_key)
        if not cfg:
            raise ValueError(f'Model không hợp lệ: {model_key}')
        if not os.path.exists(cfg['path']):
            artifact = os.path.relpath(cfg['path'], DIRECTORY)
            raise FileNotFoundError(
                f'Thiếu model {model_key}: {artifact}. '
                'Hãy chạy git pull để cập nhật models/ hoặc sao chép model từ máy gốc '
                '(OpenVINO cần cả thư mục chứa XML, BIN và metadata.yaml).'
            )
        from ultralytics import YOLO
        print(f"[*] Nạp mô hình {model_key} từ {cfg['path']}...")
        if cfg['task']:
            model = YOLO(cfg['path'], task=cfg['task'])
        else:
            model = YOLO(cfg['path'])
        LOADED_MODELS[model_key] = model
    return LOADED_MODELS[model_key]


MVP = MVPService(get_model, INFERENCE_SLOT, MODEL_CONFIGS[MVP_MODEL_KEY]['path'], CPU_THREADS)

class StudioRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def is_public_file(self):
        path = unquote(urlsplit(self.path).path)
        if ".." in path.split("/"):
            return False
        # Only browser ONNX exports are downloadable. Weights and engines stay private.
        onnx_files = {f'/runtime_exports/{model}_{precision}.onnx'
                      for model in ['yolo26m', 'yolo26s'] for precision in ['fp32', 'fp16', 'int8']}
        return path in PUBLIC_FILES or path.startswith("/demo_data/") or path in onnx_files

    def do_GET(self):
        if MVP.handle(self, 'GET'):
            return
        # Redirect / to /demo.html
        if urlsplit(self.path).path in ('/', '/index.html'):
            self.path = '/demo.html'
            return super().do_GET()

        if urlsplit(self.path).path == '/api/benchmark/capabilities':
            try:
                self.send_json(200, hardware_bench.capabilities())
            except Exception:
                import traceback
                traceback.print_exc()
                self.send_json(500, {'message': 'Không đọc được cấu hình phần cứng backend'})
            return

        if urlsplit(self.path).path == '/api/status':
            self.send_json(200, {
                'status': 'online',
                'models_available': [key for key, cfg in MODEL_CONFIGS.items() if os.path.exists(cfg['path'])],
                'models_missing': {key: os.path.relpath(cfg['path'], DIRECTORY)
                                   for key, cfg in MODEL_CONFIGS.items() if not os.path.exists(cfg['path'])},
                'models_loaded': list(LOADED_MODELS.keys())
            })
            return

        if self.is_public_file():
            return super().do_GET()
        self.send_error(404, "Not found")

    def do_HEAD(self):
        if self.is_public_file():
            return super().do_HEAD()
        self.send_error(404, "Not found")

    def do_POST(self):
        if MVP.handle(self, 'POST'):
            return
        if urlsplit(self.path).path == '/api/benchmark/run':
            self.run_hardware_benchmark()
            return
        if urlsplit(self.path).path == '/api/predict':
            try:
                import numpy as np

                content_len = int(self.headers.get('Content-Length', '0'))
                if not 0 < content_len <= MAX_REQUEST_BYTES:
                    self.send_json(413, {'status': 'error', 'message': 'Ảnh tải lên quá lớn'})
                    return
                post_body = self.rfile.read(content_len)
                payload = json.loads(post_body.decode('utf-8'))
                if not isinstance(payload, dict):
                    raise ValueError('Payload phải là JSON object')

                img, _ = hardware_bench.decode_image(payload.get('image', ''), MAX_IMAGE_PIXELS)

                h, w = img.shape[:2]
                target_models = payload.get('models', list(MODEL_CONFIGS.keys()))
                if not isinstance(target_models, list) or not target_models or any(not isinstance(key, str) or key not in MODEL_CONFIGS for key in target_models):
                    raise ValueError('Danh sách model không hợp lệ')
                try:
                    conf_thresh = float(payload.get('conf', 0.10))
                except (TypeError, ValueError) as error:
                    raise ValueError('conf phải thuộc [0, 1]') from error
                if not 0 <= conf_thresh <= 1:
                    raise ValueError('conf phải thuộc [0, 1]')
                if not INFERENCE_SLOT.acquire(blocking=False):
                    self.send_json(503, {'status': 'error', 'message': 'Server đang xử lý ảnh khác'})
                    return

                detections_res = {}
                try:
                    for m_key in target_models:
                        model = get_model(m_key)

                        configure_cpu_threads()
                        imgsz = MODEL_CONFIGS[m_key].get('imgsz', 640)
                        if m_key not in WARMED_MODELS:
                            model(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), imgsz=imgsz, conf=0.10,
                                  verbose=False, rect=False, device=cpu_device(MODEL_CONFIGS[m_key]['path']))
                            MODEL_RUNTIMES[m_key] = configure_cpu_backend(model, MODEL_CONFIGS[m_key]['path'], CPU_THREADS)
                            model(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), imgsz=imgsz, conf=0.10,
                                  verbose=False, rect=False, device=cpu_device(MODEL_CONFIGS[m_key]['path']))
                            WARMED_MODELS.add(m_key)
                            configure_cpu_threads()
                        t0 = time.perf_counter()
                        results = model(
                            img,
                            imgsz=imgsz,
                            conf=conf_thresh,
                            verbose=False,
                            device=cpu_device(MODEL_CONFIGS[m_key]['path']),
                            rect=False,
                        )[0]
                        infer_time = (time.perf_counter() - t0) * 1000

                        boxes_list = []
                        for b in results.boxes:
                            xyxy = b.xyxy[0].tolist()
                            c_val = float(b.conf[0])
                            cls_id = int(b.cls[0])
                            cls_name = results.names.get(cls_id, str(cls_id))

                            norm_box = [
                                round(xyxy[0] / w, 4),
                                round(xyxy[1] / h, 4),
                                round(xyxy[2] / w, 4),
                                round(xyxy[3] / h, 4)
                            ]
                            boxes_list.append({
                                'class_id': cls_id,
                                'class_name': cls_name,
                                'conf': round(c_val, 3),
                                'box_norm': norm_box,
                                'box_px': [round(x, 1) for x in xyxy]
                            })

                        detections_res[m_key] = {
                            'inference_time_ms': round(infer_time, 1),
                            'runtime': f"CPU ×{CPU_THREADS} · {imgsz}×{imgsz}",
                            'runtime_settings': MODEL_RUNTIMES[m_key],
                            'detections_count': len(boxes_list),
                            'boxes': boxes_list
                        }
                finally:
                    INFERENCE_SLOT.release()

                self.send_json(200, {
                    'status': 'success',
                    'width': w,
                    'height': h,
                    'detections': detections_res
                })

            # decode_image converts Pillow decoding errors into ValueError.
            except (ValueError, base64.binascii.Error) as e:
                self.send_json(400, {'status': 'error', 'message': str(e)})
            except FileNotFoundError as error:
                print(f'[!] {error}', file=sys.stderr)
                self.send_json(503, {'status': 'error', 'code': 'model_artifact_missing', 'message': str(error)})
            except Exception:
                import traceback
                traceback.print_exc()
                self.send_json(500, {'status': 'error', 'message': 'Lỗi suy luận nội bộ'})
            return

        self.send_error(404, "Endpoint not found")

    def do_DELETE(self):
        if not MVP.handle(self, 'DELETE'):
            self.send_error(404, 'Endpoint not found')

    def run_hardware_benchmark(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_REQUEST_BYTES:
                self.send_json(413, {'message': 'Ảnh tải lên quá lớn'})
                return
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('Payload phải là JSON object')
            choice = hardware_bench.validate(payload)
            image, digest = hardware_bench.decode_image(payload.get('image', ''), MAX_IMAGE_PIXELS)
        except (ValueError, TypeError, base64.binascii.Error) as error:
            self.send_json(400, {'message': str(error)})
            return
        if not INFERENCE_SLOT.acquire(blocking=False):
            self.send_json(503, {'message': 'Backend đang chạy suy luận hoặc benchmark khác; thử lại sau.'})
            return
        try:
            self.send_json(200, hardware_bench.run_benchmark(choice, image, digest))
        except Exception as error:
            import traceback
            traceback.print_exc()
            # Runtime failure is visible; do not replace a GPU run with CPU measurements.
            self.send_json(422, {'message': f'Runtime không chạy được cấu hình đã chọn: {error}'})
        finally:
            INFERENCE_SLOT.release()

    def log_message(self, format, *args):
        # Keep terminal output clean
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

def check_runtime():
    if sys.version_info < (3, 12):
        raise SystemExit(
            f'Cần Python >= 3.12 để cài requirements.txt; đang chạy {sys.version.split()[0]} '
            f'({sys.executable}). Hãy tạo và kích hoạt venv bằng Python 3.12.'
        )
    for module in ('numpy', 'PIL.Image', 'cv2', 'torch', 'ultralytics', 'openvino'):
        try:
            importlib.import_module(module)
        except (ImportError, OSError) as error:
            if 'libGL.so.1' in str(error):
                raise SystemExit(
                    f'Không tải được {module}: {error}\n'
                    'Thiếu thư viện hệ thống OpenGL. Trên Ubuntu/Debian, chạy:\n'
                    'apt-get update && apt-get install -y libgl1\n'
                    'Sau đó chạy lại python run_demo.py trong venv đang kích hoạt.'
                ) from error
            raise SystemExit(
                f'Không tải được {module}: {error}\nPython đang chạy: {sys.executable}\n'
                f'Cài thư viện trong môi trường này: {sys.executable} -m pip install -r requirements.txt'
            ) from error


def run():
    os.chdir(DIRECTORY)
    check_runtime()
    for key, cfg in MODEL_CONFIGS.items():
        if not os.path.exists(cfg['path']):
            print(f"[!] Thiếu model {key}: {cfg['path']}. Chạy git pull để cập nhật models/.",
                  file=sys.stderr)

    print("[*] Đang khởi động AI Model Compression Studio...")
    print(f"[*] Các mô hình sẽ warm-up khi được chọn trên CPU ({CPU_THREADS} threads).")

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((HOST, PORT), StudioRequestHandler) as httpd:
        url_demo = f"http://localhost:{PORT}/demo.html"
        url_dash = f"http://localhost:{PORT}/model-compression-yolo26-dashboard.html"
        print("=" * 72)
        print("  YOLO26 x BDD100K MODEL COMPRESSION STUDIO & INFERENCE SERVER")
        print("=" * 72)
        print(f"  🚀 MVP Demo Studio (Trực quan & Upload):  {url_demo}")
        print(f"  📊 Bảng Quản Trị & Nghiệm Thu 14 Ngày:    {url_dash}")
        print(f"  ⚡ Live Prediction API Endpoint:           http://localhost:{PORT}/api/predict")
        print(f"  🌐 Lắng nghe: {HOST}:{PORT} · Máy khác dùng http://<IP-server>:{PORT}/demo.html")
        print("=" * 72)
        print("  Nhấn Ctrl+C để dừng server.")
        print()
        httpd.serve_forever()

if __name__ == "__main__":
    run()
