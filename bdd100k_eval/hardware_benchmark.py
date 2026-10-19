"""Opt-in, single-image hardware microbenchmarks; independent of accuracy reports."""
import base64
import hashlib
import io
import json
import math
import platform
import time
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata
from pathlib import Path

from bdd100k_eval.config import ROOT
from bdd100k_eval.artifacts import sha256
from bdd100k_eval.image_ops import NAMES, decode_image, preprocess, postprocess
EXPORTS = ROOT / 'runtime_exports'
THREADS = 4
MODELS = {'yolo26m': 'YOLO26m · Baseline / PTQ', 'yolo26s': 'YOLO26s · Baseline / PTQ',
          'yolo26m_qat': 'YOLO26m · QAT INT8', 'yolo26s_qat': 'YOLO26s · QAT INT8'}


def cpu_name(cpuinfo=Path('/proc/cpuinfo')):
    try:
        for line in cpuinfo.read_text(errors='ignore').splitlines():
            if line.lower().startswith(('model name', 'hardware')) and ':' in line:
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or 'CPU'


@lru_cache(maxsize=1)
def hardware():
    import torch
    import openvino as ov
    versions = {}
    for package in ['torch', 'openvino', 'onnxruntime-gpu', 'tensorrt-cu13']:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    cuda = torch.cuda.is_available()
    core = ov.Core()
    devices = {}
    for device in core.available_devices:
        devices[device] = {'name': core.get_property(device, 'FULL_DEVICE_NAME'),
                           'capabilities': list(core.get_property(device, 'OPTIMIZATION_CAPABILITIES'))}
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        versions['onnxruntime'] = ort.__version__
    except ImportError:
        providers = []
    try:
        import tensorrt as trt
        versions['tensorrt'] = trt.__version__
    except (ImportError, OSError):
        versions['tensorrt'] = None
    return dict(cpu=cpu_name(), logical_cpus=__import__('os').cpu_count(),
                cuda=cuda, gpu=torch.cuda.get_device_name(0) if cuda else None,
                compute_capability=list(torch.cuda.get_device_capability(0)) if cuda else None,
                cuda_version=torch.version.cuda, openvino_devices=devices, providers=providers,
                versions=versions, os=platform.platform())


def manifest():
    path = EXPORTS / 'manifest.json'
    return json.loads(path.read_text()) if path.exists() else {}


def artifact(model, engine, precision):
    family = model.removesuffix('_qat')
    if engine == 'pytorch':
        return ROOT / 'models' / f'{family}-bdd100k.pt'
    if engine == 'openvino' and precision == 'INT8':
        suffix = '_qat' if model.endswith('_qat') else ''
        folder = ROOT / 'models' / f'int8_{family}{suffix}_640_openvino_model'
        return next(folder.glob('*.xml'), folder / 'model.xml')
    kind = 'onnx' if engine in ['onnxruntime', 'webgpu', 'wasm'] else engine
    ext = {'onnx': 'onnx', 'openvino': 'xml', 'tensorrt': 'engine'}[kind]
    return EXPORTS / f'{family}_{precision.lower()}.{ext}'


def capabilities():
    facts, entries = hardware(), manifest()
    configs = []
    for model in MODELS:
        for engine, devices in [('pytorch', ['cpu', 'cuda']), ('onnxruntime', ['cpu', 'cuda']),
                                ('tensorrt', ['cuda']), ('openvino', list(facts['openvino_devices'])),
                                ('webgpu', ['browser-gpu']), ('wasm', ['browser-cpu'])]:
            for device in devices:
                for precision in ['FP32', 'FP16', 'INT8']:
                    path = artifact(model, engine, precision)
                    reason = None
                    if model.endswith('_qat') and not (engine == 'openvino' and precision == 'INT8'):
                        reason = 'Artifact QAT hiện chỉ có OpenVINO INT8.'
                    elif precision == 'INT8' and engine != 'openvino':
                        reason = ('ONNX INT8 hiện dùng ở chế độ trình duyệt; native chưa bật.'
                                  if engine == 'onnxruntime' and path.is_file() else
                                  'Chưa có artifact INT8 đã calibration cho runtime này.')
                    elif precision == 'FP16' and device in ['cpu', 'browser-cpu'] and engine != 'pytorch':
                        reason = 'Chế độ CPU này chỉ cung cấp FP32; không ép FP16 rồi gắn nhãn tăng tốc.'
                    elif device == 'cuda' and not facts['cuda']:
                        reason = 'Backend không có CUDA GPU khả dụng.'
                    elif engine == 'onnxruntime' and (not facts['providers'] or
                            (device == 'cuda' and 'CUDAExecutionProvider' not in facts['providers'])):
                        reason = 'Chưa cài ONNX Runtime với execution provider được chọn.'
                    elif engine == 'tensorrt' and not facts['versions']['tensorrt']:
                        reason = 'Chưa cài TensorRT (NVIDIA CUDA).'
                    elif engine == 'openvino' and precision not in facts['openvino_devices'][device]['capabilities']:
                        reason = f'OpenVINO {device} không công bố hỗ trợ {precision} compute.'
                    elif not path.is_file():
                        reason = ('Chưa cung cấp export OpenVINO FP16; hiện có FP32 và INT8.'
                                  if engine == 'openvino' and precision == 'FP16' else
                                  'Chưa export artifact. Chạy .venv/bin/python prepare_runtime_models.py')
                    if not reason and path.is_relative_to(EXPORTS):
                        kind = 'onnx' if engine in ['onnxruntime', 'webgpu', 'wasm'] else engine
                        entry = entries.get(f'{model}_{precision}_{kind}', {})
                        if not entry:
                            reason = 'Artifact chưa có manifest nguồn gốc; cần export lại.'
                        elif engine == 'tensorrt' and (entry.get('gpu') != facts['gpu'] or
                                entry.get('tensorrt_version') != facts['versions']['tensorrt']):
                            reason = 'Engine TensorRT cần build lại cho GPU / phiên bản hiện tại.'
                    config = dict(model=model, engine=engine, device=device, precision=precision,
                                  available=reason is None, reason=reason, artifact=str(path.relative_to(ROOT)))
                    if engine in ['webgpu', 'wasm'] and not reason:
                        entry = entries[f'{model}_{precision}_onnx']
                        config.update(url=f'/runtime_exports/{path.name}', sha256=entry['sha256'],
                                      source_sha256=entry['source_sha256'])
                    configs.append(config)
    return dict(hardware=facts, models=MODELS, configs=configs, input_shape=[1, 3, 640, 640],
                scope='Backend = máy chạy Python; WebGPU / Browser CPU = máy mở trình duyệt.')


def validate(payload):
    choice = {key: payload.get(key) for key in ['model', 'engine', 'device', 'precision']}
    config = next((c for c in capabilities()['configs'] if all(c[k] == v for k, v in choice.items())), None)
    if config is None or config['engine'] in ['webgpu', 'wasm']:
        raise ValueError('Cấu hình benchmark backend không hợp lệ')
    if not config['available']:
        raise ValueError(config['reason'])
    for key, low, high, default in [('warmup', 0, 10, 3), ('repeats', 1, 50, 10)]:
        value = payload.get(key, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'{key} phải là số nguyên trong [{low}, {high}]')
        choice[key] = value
    conf = payload.get('conf', .25)
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(conf) or not 0 <= conf <= 1:
        raise ValueError('conf phải thuộc [0, 1]')
    choice['conf'] = conf
    return choice


def load_runner(choice):
    """Return one explicitly selected backend; never retry on a different device."""
    import torch
    engine, device, precision = (choice[k] for k in ['engine', 'device', 'precision'])
    path = artifact(choice['model'], engine, precision)
    runtime = dict(artifact=str(path.relative_to(ROOT)), sha256=sha256(path), input_dtype='float16' if precision == 'FP16' else 'float32')
    if path.is_relative_to(EXPORTS):
        kind = 'onnx' if engine == 'onnxruntime' else engine
        entry = manifest()[f'{choice["model"]}_{precision}_{kind}']
        if entry['sha256'] != runtime['sha256']:
            raise RuntimeError('Artifact hash differs from export manifest; export again')
        source = ROOT / 'models' / f'{choice["model"]}-bdd100k.pt'
        if sha256(source) != entry['source_sha256']:
            raise RuntimeError('Source checkpoint changed; export again')
        if engine == 'openvino' and sha256(path.with_suffix('.bin')) != entry['bin_sha256']:
            raise RuntimeError('OpenVINO weights hash differs from export manifest')
        runtime['source_sha256'] = entry['source_sha256']
    torch.set_num_threads(THREADS)
    if engine in ['pytorch', 'tensorrt']:
        from ultralytics import YOLO
        target = torch.device('cuda:0' if device == 'cuda' else 'cpu')
        if engine == 'pytorch':
            model = YOLO(path).model
            model.end2end = True  # Match nms=False ONNX/OpenVINO/TensorRT exports (one-to-one head).
            model = model.fuse(verbose=False).eval().to(target)
            model = model.half() if precision == 'FP16' else model.float()
            runtime['parameter_dtype'] = str(next(model.parameters()).dtype)
        else:
            from ultralytics.nn.backends.tensorrt import TensorRTBackend
            model = TensorRTBackend(path, device=target)
            if model.fp16 != (precision == 'FP16'):
                raise RuntimeError('TensorRT input dtype does not match selected precision')
            runtime['build'] = manifest()[f'{choice["model"]}_{precision}_tensorrt']
        def infer(array):
            with torch.inference_mode():
                output = model(torch.from_numpy(array).to(target))
                if isinstance(output, (tuple, list)):
                    output = output[0]
                return output.float().cpu().numpy()
        runtime.update(device=str(target), device_name=hardware()['gpu'] if device == 'cuda' else hardware()['cpu'])
    elif engine == 'onnxruntime':
        import onnxruntime as ort
        ort.preload_dlls()
        provider = 'CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'
        options = ort.SessionOptions()
        options.intra_op_num_threads = THREADS
        options.inter_op_num_threads = 1
        providers = [(provider, {'use_tf32': '0'})] if device == 'cuda' else [provider]
        session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        session.disable_fallback()
        if session.get_providers()[0] != provider:
            raise RuntimeError(f'Requested {provider}, got {session.get_providers()}; fallback refused')
        name = session.get_inputs()[0].name
        infer = lambda array: session.run(None, {name: array})[0]
        runtime.update(providers=session.get_providers(), provider_options=session.get_provider_options(),
                       device=device, device_name=hardware()['gpu'] if device == 'cuda' else hardware()['cpu'])
    else:
        import openvino as ov
        core = ov.Core()
        settings = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1'}
        if device == 'CPU':
            settings['INFERENCE_NUM_THREADS'] = THREADS
        if precision == 'FP32':
            settings['INFERENCE_PRECISION_HINT'] = 'f32'
        graph = core.read_model(path)
        if precision == 'INT8' and not any(op.get_type_name() == 'FakeQuantize' for op in graph.get_ops()):
            raise RuntimeError('INT8 graph has no FakeQuantize nodes')
        compiled = core.compile_model(graph, device, settings)
        actual = list(compiled.get_property('EXECUTION_DEVICES'))
        if not all(d.split('.')[0] == device.split('.')[0] for d in actual):
            raise RuntimeError(f'OpenVINO device mismatch: {actual}')
        request = compiled.create_infer_request()
        infer = lambda array: request.infer({0: array})[compiled.output(0)].copy()
        runtime.update(device=device, execution_devices=actual,
                       device_name=hardware()['openvino_devices'][device]['name'],
                       precision_policy='INT8 graph; mixed INT8 / floating-point kernels' if precision == 'INT8' else 'FP32 hint',
                       weights_sha256=sha256(path.with_suffix('.bin')))
        precisions = {}
        for op in compiled.get_runtime_model().get_ops():
            info = op.get_rt_info()
            if 'runtimePrecision' in info:
                label = str(info['runtimePrecision'].value)
                precisions[label] = precisions.get(label, 0) + 1
        runtime['runtime_precision_nodes'] = precisions
    return infer, runtime


def run_benchmark(choice, image, image_sha256):
    import numpy as np
    import torch
    previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        started = time.perf_counter()
        infer, runtime = load_runner(choice)
        load_ms = (time.perf_counter() - started) * 1000
        def once():
            tensor, transform = preprocess(image, choice['precision'] == 'FP16')
            return postprocess(infer(tensor), transform, choice['conf'])
        for _ in range(choice['warmup']):
            once()
        samples = []
        for _ in range(choice['repeats']):
            if choice['device'] == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            boxes = once()  # CPU materialization synchronizes every backend, including GPU providers.
            if choice['device'] == 'cuda':
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        mean = float(np.mean(samples))
        return dict(status='success', timestamp=datetime.now(timezone.utc).isoformat(), location='backend',
                    **choice, runtime=runtime, hardware=hardware(), image_sha256=image_sha256,
                    protocol=dict(input_shape=[1, 3, 640, 640], batch=1, cpu_threads=THREADS,
                                  scope='preprocess + input transfer + inference + output transfer + postprocess',
                                  preprocessing='OpenCV INTER_LINEAR letterbox / RGB / NCHW / divide255',
                                  tf32=False, nms=False, accuracy='single-image microbenchmark, not mAP'),
                    load_ms=load_ms, samples_ms=samples, mean_ms=mean, p50_ms=float(np.percentile(samples, 50)),
                    p95_ms=float(np.percentile(samples, 95)), min_ms=min(samples), max_ms=max(samples),
                    fps=1000 / mean, detections_count=len(boxes), boxes=boxes)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32
