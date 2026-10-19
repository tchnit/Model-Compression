"""Existing lazy model cache, shared by the API and offline tools."""
import os
from .config import CPU_THREADS, MODEL_CONFIGS, DIRECTORY


LOADED_MODELS = {}


WARMED_MODELS = set()


MODEL_RUNTIMES = {}


def configure_cpu_threads(threads=CPU_THREADS):
    """Restore the PyTorch CPU budget after Ultralytics initializes a predictor."""
    import torch
    torch.set_num_threads(threads)


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
