"""Project-owned paths and existing deployment defaults."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HOST = "0.0.0.0"

PORT = 8765

DIRECTORY = str(ROOT)

CPU_THREADS = 4

MAX_REQUEST_BYTES = 12 * 1024 * 1024

MAX_IMAGE_PIXELS = 20_000_000

PUBLIC_FILES = {
    "/demo.html",
    "/model-compression-yolo26-dashboard.html",
    "/benchmark_comparison.md",
    "/benchmark_comparison.csv",
}

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
