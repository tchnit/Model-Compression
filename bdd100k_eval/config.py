"""Project-owned paths and existing deployment defaults."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HOST = "0.0.0.0"

PORT = 8765

DIRECTORY = str(ROOT)


def available_cpu_threads():
    """Return the CPUs this process can actually schedule on."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


CPU_THREADS = available_cpu_threads()

MAX_REQUEST_BYTES = 12 * 1024 * 1024

MAX_IMAGE_PIXELS = 20_000_000

PUBLIC_FILES = {
    "/demo.html",
    "/model-compression-yolo26-dashboard.html",
    "/benchmark_comparison.md",
    "/benchmark_comparison.csv",
}

BROWSER_ONNX_MODELS = {
    'int8_qat_yolo26m': {'file': 'int8_qat_yolo26m.onnx', 'label': 'YOLO26m QAT INT8', 'output_format': 'end2end'},
    'int8_qat_yolo26s': {'file': 'int8_qat_yolo26s.onnx', 'label': 'YOLO26s QAT INT8', 'output_format': 'end2end'},
    'int8_qat_yolo26s_slim': {'file': 'int8_qat_yolo26s_slim.onnx', 'label': 'YOLO26s QAT INT8 Slim', 'output_format': 'end2end'},
    'int8_yolo26s_slim': {'file': 'int8_yolo26s_slim.onnx', 'label': 'YOLO26s INT8 Slim', 'output_format': 'raw_yolo'},
    'int8_yolom': {'file': 'int8_yolom.onnx', 'label': 'YOLO26m INT8', 'output_format': 'raw_yolo'},
    'int8_yolos': {'file': 'int8_yolos.onnx', 'label': 'YOLO26s INT8', 'output_format': 'raw_yolo'},
}

MODEL_CONFIGS = {
    'yolo26m_fp32': {
        'path': os.path.join(DIRECTORY, 'models', 'yolo26m-bdd100k.pt'),
        'task': None,
        'device': 'cpu',
        'precision': 'FP32',
    },
    'yolo26m_baseline_fp32': {
        'path': os.path.join(DIRECTORY, 'models', 'yolo26m_fp32_openvino_model'),
        'task': 'detect',
        'device': 'cpu',
        'precision': 'FP32',
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
        'precision': 'FP32',
    },
    'yolo26s_baseline_fp32': {
        'path': os.path.join(DIRECTORY, 'models', 'yolo26s_fp32_openvino_model'),
        'task': 'detect',
        'device': 'cpu',
        'precision': 'FP32',
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
        'precision': 'FP32',
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
