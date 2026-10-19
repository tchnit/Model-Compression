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

import http.server
import importlib
import socketserver
import json
import base64
import math
import threading
import time
from urllib.parse import unquote, urlsplit, parse_qs
import re
import shutil
import tempfile
from bdd100k_eval.inference_runtime import cpu_device, configure_cpu_backend
from bdd100k_eval import hardware_benchmark as hardware_bench
from bdd100k_eval.mvp_demo import MVPService, MODEL_KEY as MVP_MODEL_KEY
from bdd100k_eval.models import get_model, configure_cpu_threads, LOADED_MODELS, WARMED_MODELS, MODEL_RUNTIMES
from bdd100k_eval.config import (HOST, PORT, DIRECTORY, CPU_THREADS, MAX_REQUEST_BYTES,
                                  MAX_IMAGE_PIXELS, PUBLIC_FILES, MODEL_CONFIGS, BROWSER_ONNX_MODELS)

INFERENCE_SLOT = threading.BoundedSemaphore(1)
HUGGINGFACE_INFERENCE_SLOT = threading.BoundedSemaphore(1)
HUGGINGFACE_MODEL_KEY = 'yolo26s_export_int8_640'
HUGGINGFACE_MODEL_KEYS = {
    HUGGINGFACE_MODEL_KEY: HUGGINGFACE_MODEL_KEY,
    'yolo26s_ptq_int8': 'yolo26s_ptq_int8',
    'yolo26m_ptq_int8': 'yolo26m_ptq_int8',
    'yolo26s_baseline_fp32': 'yolo26s_baseline_fp32',
    'yolo26m_baseline_fp32': 'yolo26m_baseline_fp32',
    'yolo26s_fp32': 'yolo26s_fp32',
    'yolo26m_fp32': 'yolo26m_fp32',
}
HUGGINGFACE_SPACE = 'kiioss/m38-edge-labeling'
HUGGINGFACE_SPACE_URL = 'https://kiioss-m38-edge-labeling.hf.space'
HUGGINGFACE_CPU = 'Intel(R) Xeon(R) Platinum 8559C'
_HUGGINGFACE_CLIENT = None


def get_huggingface_client():
    global _HUGGINGFACE_CLIENT
    if _HUGGINGFACE_CLIENT is None:
        import certifi
        import ssl
        from gradio_client import Client
        tls = ssl.create_default_context()
        tls.load_verify_locations(cafile=certifi.where())
        custom_ca = os.environ.get('HUGGINGFACE_CA_BUNDLE')
        if custom_ca:
            custom_ca = os.path.abspath(os.path.expanduser(custom_ca))
            if not os.path.isfile(custom_ca):
                raise RuntimeError(f'HUGGINGFACE_CA_BUNDLE không tồn tại: {custom_ca}')
            tls.load_verify_locations(cafile=custom_ca)
            tls.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
        try:
            _HUGGINGFACE_CLIENT = Client(
                HUGGINGFACE_SPACE_URL, verbose=False, ssl_verify=tls,
                httpx_kwargs={'timeout': 60.0},
            )
        except Exception as error:
            if 'CERTIFICATE_VERIFY_FAILED' in str(error):
                raise RuntimeError(
                    'Máy chủ chưa tin cậy chứng chỉ HTTPS. Cài CA nội bộ hoặc đặt '
                    'HUGGINGFACE_CA_BUNDLE tới tệp CA rồi khởi động lại server.'
                ) from error
            raise
    return _HUGGINGFACE_CLIENT


def predict_on_huggingface(image, width, height, conf, model_key):
    """Run the selected model on the remote Space and normalize its response."""
    import cv2
    from gradio_client import handle_file

    encoded, buffer = cv2.imencode('.png', image)
    if not encoded:
        raise RuntimeError('Không thể chuẩn bị ảnh cho Hugging Face')
    with tempfile.NamedTemporaryFile(suffix='.png') as image_file:
        image_file.write(buffer.tobytes())
        image_file.flush()
        started = time.perf_counter()
        result = get_huggingface_client().predict(
            image_input=handle_file(image_file.name),
            model_key=HUGGINGFACE_MODEL_KEYS[model_key],
            conf_threshold=conf,
            iou_threshold=0.45,
            api_name='/xeon_bridge_model',
        )
        request_time_ms = (time.perf_counter() - started) * 1000

    if not isinstance(result, dict) or result.get('success') is not True or not isinstance(result.get('detections'), list):
        raise RuntimeError('Hugging Face trả về dữ liệu không hợp lệ')
    boxes = []
    for detection in result['detections']:
        try:
            xyxy = [float(value) for value in detection['bbox_xyxy']]
            if len(xyxy) != 4:
                raise ValueError
            class_id = int(detection['class_id'])
            class_name = str(detection['class_name'])
            confidence = float(detection['confidence'])
            if not all(math.isfinite(value) for value in xyxy) or not 0 <= confidence <= 1:
                raise ValueError
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError('Hugging Face trả về detection không hợp lệ') from error
        boxes.append({
            'class_id': class_id,
            'class_name': class_name,
            'conf': round(confidence, 3),
            'box_norm': [round(xyxy[0] / width, 4), round(xyxy[1] / height, 4),
                         round(xyxy[2] / width, 4), round(xyxy[3] / height, 4)],
            'box_px': [round(value, 1) for value in xyxy],
        })
    server_wall_ms = float(result.get('total_wall_ms', result.get('real_inference_ms', 0)))
    server_cpu = str(result.get('server_cpu') or HUGGINGFACE_CPU)
    return {
        'inference_time_ms': round(server_wall_ms, 1),
        'runtime': f'Hugging Face · {server_cpu} · API {request_time_ms:.1f} ms',
        'runtime_settings': {
            'provider': 'huggingface',
            'server_cpu': server_cpu,
            'server_inference_ms': float(result.get('real_inference_ms', server_wall_ms)),
            'server_wall_ms': server_wall_ms,
            'request_time_ms': round(request_time_ms, 1),
        },
        'detections_count': len(boxes),
        'boxes': boxes,
    }

# The OpenVINO exports run on CPU.  Pin the PyTorch reference to the same
# device, otherwise Ultralytics will silently select CUDA for `.pt` models and
# the side-by-side latency comparison becomes meaningless.
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))

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
        onnx_files.update(f'/onnx_models/{model["file"]}' for model in BROWSER_ONNX_MODELS.values())
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
            cpu_name = hardware_bench.cpu_name()
            self.send_json(200, {
                'status': 'online',
                'models_available': [key for key, cfg in MODEL_CONFIGS.items() if os.path.exists(cfg['path'])],
                'models_missing': {key: os.path.relpath(cfg['path'], DIRECTORY)
                                   for key, cfg in MODEL_CONFIGS.items() if not os.path.exists(cfg['path'])},
                'models_loaded': list(LOADED_MODELS.keys()),
                'cpu_name': cpu_name,
                'cpu_threads': CPU_THREADS,
                'int8_cpu_threads': hardware_bench.int8_cpu_thread_limit(cpu_name, CPU_THREADS),
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
                processor = payload.get('processor', 'host')
                if processor not in ('host', 'huggingface'):
                    raise ValueError('CPU suy luận không hợp lệ')
                try:
                    conf_thresh = float(payload.get('conf', 0.10))
                except (TypeError, ValueError) as error:
                    raise ValueError('conf phải thuộc [0, 1]') from error
                if not 0 <= conf_thresh <= 1:
                    raise ValueError('conf phải thuộc [0, 1]')
                if processor == 'huggingface':
                    if len(target_models) != 1 or target_models[0] not in HUGGINGFACE_MODEL_KEYS:
                        raise ValueError('Model đã chọn chưa có trên CPU Hugging Face')
                    if not HUGGINGFACE_INFERENCE_SLOT.acquire(blocking=False):
                        self.send_json(503, {'status': 'error', 'message': 'CPU Hugging Face đang xử lý ảnh khác'})
                        return
                    try:
                        detection = predict_on_huggingface(img, w, h, conf_thresh, target_models[0])
                    except Exception as error:
                        print(f'[!] Hugging Face inference failed: {error}', file=sys.stderr)
                        self.send_json(502, {'status': 'error', 'message': f'Không gọi được CPU Hugging Face: {error}'})
                        return
                    finally:
                        HUGGINGFACE_INFERENCE_SLOT.release()
                    self.send_json(200, {
                        'status': 'success', 'width': w, 'height': h,
                        'detections': {target_models[0]: detection},
                    })
                    return
                cpu_name = hardware_bench.cpu_name()
                max_threads = min(hardware_bench.int8_cpu_thread_limit(cpu_name, CPU_THREADS)
                                  if 'int8' in key else CPU_THREADS for key in target_models)
                threads = payload.get('threads', max_threads)
                if type(threads) is not int or not 1 <= threads <= max_threads:
                    raise ValueError(f'threads phải là số nguyên thuộc [1, {max_threads}] cho model đã chọn')
                if not INFERENCE_SLOT.acquire(blocking=False):
                    self.send_json(503, {'status': 'error', 'message': 'Server đang xử lý ảnh khác'})
                    return

                detections_res = {}
                try:
                    for m_key in target_models:
                        model = get_model(m_key)

                        configure_cpu_threads(threads)
                        imgsz = MODEL_CONFIGS[m_key].get('imgsz', 640)
                        previous = MODEL_RUNTIMES.get(m_key, {})
                        previous_threads = previous.get('requested_threads', previous.get('torch_threads'))
                        needs_warmup = m_key not in WARMED_MODELS or previous_threads != threads
                        if m_key not in WARMED_MODELS:
                            model(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), imgsz=imgsz, conf=0.10,
                                  verbose=False, rect=False, device=cpu_device(MODEL_CONFIGS[m_key]['path']))
                        MODEL_RUNTIMES[m_key] = configure_cpu_backend(model, MODEL_CONFIGS[m_key]['path'], threads,
                                                                    MODEL_CONFIGS[m_key].get('precision', 'FP16'),
                                                                    strict_threads=False)
                        if needs_warmup:
                            model(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), imgsz=imgsz, conf=0.10,
                                  verbose=False, rect=False, device=cpu_device(MODEL_CONFIGS[m_key]['path']))
                            WARMED_MODELS.add(m_key)
                            configure_cpu_threads(threads)
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
                            'runtime': f"CPU ×{MODEL_RUNTIMES[m_key].get('ov_threads', MODEL_RUNTIMES[m_key]['torch_threads'])} · {imgsz}×{imgsz}",
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


def handle_mvp(service, handler, method):
    parsed = urlsplit(handler.path)
    if not parsed.path.startswith('/api/mvp/'):
        return False
    service.cleanup()
    try:
        parts = parsed.path.strip('/').split('/')
        if method == 'GET' and parts == ['api', 'mvp', 'config']:
            handler.send_json(200, service.config())
        elif method == 'POST' and parts == ['api', 'mvp', 'jobs']:
            length = int(handler.headers.get('Content-Length', '0'))
            handler.connection.settimeout(30)
            handler.send_json(202, service.submit(handler.rfile, length, parse_qs(parsed.query)))
        elif method == 'POST' and parts == ['api', 'mvp', 'evaluate']:
            length = int(handler.headers.get('Content-Length', '0'))
            if not 0 < length <= 12 * 1024**2:
                raise ValueError('Ground truth tối đa 12 MiB')
            handler.send_json(200, service.evaluate(json.loads(handler.rfile.read(length))))
        elif len(parts) >= 4 and parts[:3] == ['api', 'mvp', 'jobs']:
            job = service.get(parts[3])
            if method == 'GET' and len(parts) == 4:
                handler.send_json(200, service.snapshot(job))
            elif method == 'POST' and parts[4:] == ['cancel']:
                job['cancel'].set()
                handler.send_json(200, {'message': 'Đã yêu cầu dừng'})
            elif method == 'DELETE' and len(parts) == 4:
                with service.lock:
                    job['cancel'].set()
                    if job['status'] in ('starting', 'running'):
                        job['delete'] = True
                    else:
                        shutil.rmtree(job['folder'], ignore_errors=True)
                        service.jobs.pop(job['id'], None)
                handler.send_json(200, {'message': 'Đã xóa dữ liệu phiên'})
            elif method == 'GET' and len(parts) == 6 and parts[4] == 'files' and parts[5] in job['files']:
                service.serve_file(handler, job['folder'] / parts[5])
            else:
                raise KeyError('Tài nguyên không tồn tại hoặc chưa hoàn tất')
        else:
            raise KeyError('Endpoint không tồn tại')
    except FileNotFoundError as error:
        handler.send_json(503, {'message': str(error)})
    except KeyError:
        handler.send_json(404, {'message': 'Phiên/tài nguyên không tồn tại hoặc đã hết hạn'})
    except (ValueError, TypeError, OverflowError) as error:
        handler.send_json(400, {'message': str(error)})
    except RuntimeError as error:
        handler.send_json(503, {'message': str(error)})
    except Exception:
        import traceback
        traceback.print_exc()
        handler.send_json(500, {'message': 'Lỗi nội bộ MVP; xem log server để biết nguyên nhân.'})
    return True


def serve_file(handler, path):
    import mimetypes
    size = path.stat().st_size
    start, end = 0, size - 1
    value = handler.headers.get('Range')
    if value:
        match = re.fullmatch(r'bytes=(\d+)-(\d*)', value)
        if not match:
            handler.send_json(416, {'message': 'Range không hợp lệ'})
            return
        start = int(match[1])
        end = min(int(match[2]) if match[2] else end, end)
        if start > end:
            handler.send_json(416, {'message': 'Range ngoài tệp'})
            return
    handler.send_response(206 if value else 200)
    handler.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
    handler.send_header('Content-Length', str(end - start + 1))
    handler.send_header('Accept-Ranges', 'bytes')
    if value:
        handler.send_header('Content-Range', f'bytes {start}-{end}/{size}')
    handler.end_headers()
    with path.open('rb') as source:
        source.seek(start)
        remaining = end - start + 1
        while remaining:
            block = source.read(min(65536, remaining))
            if not block:
                break
            handler.wfile.write(block)
            remaining -= len(block)


from .server import check_runtime, run
