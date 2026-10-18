"""Refresh demo predictions and a shared CPU latency benchmark (50 images × 3)."""
import gc
import argparse
import importlib.metadata
import os
import platform
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from bdd100k_eval.config import ROOT

import cv2
import numpy as np

from bdd100k_eval import models as server
from bdd100k_eval.artifacts import artifact_info, sha256
from bdd100k_eval.dataset import read_frames
from bdd100k_eval.evaluation import distribution
from bdd100k_eval.inference_runtime import configure_cpu_backend, cpu_device


def sync_benchmark_assets(eval_dirs=None):
    from bdd100k_eval.tools.compare_benchmarks import find_eval_dirs, parse_eval_dir, deduplicate_records, apply_reference_speedup
    records = [r for directory in find_eval_dirs(eval_dirs) for r in parse_eval_dir(directory)
               if r["artifact_exists"] and r["imgsz"] == 640]
    records = apply_reference_speedup(deduplicate_records(records))
    data_path = ROOT / "demo_data/detections.json"
    data = json.loads(data_path.read_text())
    by_name = {r['model_name']: r for r in records}
    for model_key, model in data["models"].items():
        for key, value in model.items():
            if isinstance(value, str):
                model[key] = value.replace("MB", "MiB")
        record = by_name.get(Path(model['path']).name)
        if record:
            model.update(map50_95=record['map50_95_pct'], map50=record['map50_pct'],
                         avg_latency_ms=round(record['lat_mean_ms'], 1), fps=record['serial_fps'])
        if model_key in {'yolo26m_fp32', 'yolo26s_fp32'}:
            model.update(name=model['name'].replace('FP32', 'FP16'), badge='CPU FP16 · trọng số FP16')
        elif model_key == 'yolo26s_fp32_pack':
            model.update(name='YOLO26s FP16 (từ fp32.pt)', badge='Lưu float32 · CPU FP16')
        elif model_key == 'yolo26s_fp16_pt':
            model['badge'] = 'Lưu FP16 · CPU FP16'
    data["benchmarks"] = [{
        "family": r["family"], "model": r["canonical_name"],
        "role": "baseline" if r["role"] == "baseline" else "qat" if r["role"] == "QAT" else "candidate",
        "imgsz": "640x640", "size": r["size_mib"], "map95": r["map50_95_pct"],
        "drop": r["map_drop_pp"], "map50": r["map50_pct"], "prec": r["precision_pct"],
        "rec": r["recall_pct"], "f1": r["f1_pct"], "lat": r["lat_mean_ms"],
        "fps": r["serial_fps"], "speedup": r["speedup"], "measurement": r["measurement"],
        "accuracy_source": r["experiment"] + "/report.json",
        "accuracy_pending": r["accuracy_pending"], "accuracy_precision": r["accuracy_precision"],
        "latency_precision": r["latency_precision"]
    } for r in records]
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    data_path.write_text(payload)
    data_path.with_name("demo_bundle.js").write_text("window.DEMO_DATA = " + payload + ";\n")
    dashboard = data_path.parent.parent / 'model-compression-yolo26-dashboard.html'
    summary = json.dumps(data['benchmarks'], ensure_ascii=False).replace('</', '<\\/')
    html, count = re.subn(r'(<script id="benchmarkSummaryData" type="application/json">).*?(</script>)',
                          lambda match: match[1] + summary + match[2], dashboard.read_text(), count=1, flags=re.S)
    if count != 1:
        raise ValueError('Dashboard is missing the benchmark data block')
    dashboard.write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dirs', nargs='+', help='Chỉ lấy accuracy từ các lượt đánh giá này')
    args = parser.parse_args()
    root = ROOT
    os.chdir(root)
    data_path = root / "demo_data/detections.json"
    data = json.loads(data_path.read_text())
    frames = {f['name']: f for f in read_frames(root / "data/bdd100k/labels/bdd100k_labels_images_val.json")}
    for scene in data['scenes']:
        scene['attributes'] = frames.get(scene['file'], {}).get('attributes') or {}
    del frames
    source = Path(args.eval_dirs[0]) if args.eval_dirs else root / "evaluation_int8_yolo26m_640"
    report = json.loads((source / "report.json").read_text())
    if args.eval_dirs:
        for directory in args.eval_dirs:
            candidate = json.loads((Path(directory) / 'report.json').read_text())
            if candidate['protocol']['dataset_stats']['images'] != 10000:
                raise ValueError('Chỉ công bố accuracy đầy đủ 10.000 ảnh; không đưa smoke vào bảng chính')
    names = report["protocol"]["benchmark_images"][:50]
    images = [root / "data/bdd100k/images/val" / name for name in names]
    versions = {}
    for package in ("ultralytics", "torch", "openvino", "numpy", "opencv-python", "Pillow"):
        versions[package] = importlib.metadata.version(package)
    protocol = {"platform": platform.platform(), "logical_cpus": os.cpu_count(), "created_at": datetime.now(timezone.utc).isoformat(), "imgsz": 640,
                "rect": False, "conf": 0.001, "iou": 0.7, "max_det": 300,
                "threads": 4, "device": "CPU", "exclusive_host": False, "warmup": 5, "repeats": 3,
                "pytorch_precision": "FP16", "export_precision": "artifact-defined",
                "images": [{"name": p.name, "sha256": sha256(p)} for p in images],
                "versions": versions,
                "timing": "predict wall; excludes image decode, model setup and warmup"}
    measurements = {}
    measured_keys = ("yolo26m_fp32", "yolo26m_ptq_int8", "yolo26m_qat_int8",
                     "yolo26s_fp32", "yolo26s_ptq_int8", "yolo26s_int8")
    model_keys = [*measured_keys, *(key for key in server.MODEL_CONFIGS if key not in measured_keys)]
    for key in model_keys:
        config = server.MODEL_CONFIGS[key]
        path = config["path"]
        model = server.get_model(key)
        options = dict(imgsz=640, rect=False, device=cpu_device(path), verbose=False,
                       conf=0.001, iou=0.7, max_det=300)
        model(np.zeros((640, 640, 3), dtype=np.uint8), **options)
        runtime = configure_cpu_backend(model, path, 4)
        for _ in range(5):
            model(cv2.imread(str(images[0])), **options)
        if key in measured_keys:
            samples = []
            for repeat in range(3):
                for index, image in enumerate(images, 1):
                    pixels = cv2.imread(str(image))
                    if pixels is None:
                        raise ValueError(f"Cannot decode {image}")
                    started = time.perf_counter()
                    model(pixels, **options)
                    samples.append({"image": image.name, "repeat": repeat,
                                    "ms": (time.perf_counter() - started) * 1000})
                    if index % 10 == 0:
                        print(f"PROGRESS {key}: repeat {repeat + 1}/3, image {index}/{len(images)}", flush=True)
            measurements[Path(path).name] = {"artifact": artifact_info(path), "runtime": runtime,
                                           "latency": distribution([s["ms"] for s in samples]),
                                           "samples": samples}
            latency = measurements[Path(path).name]["latency"]["mean"]
            data["models"][key].update(avg_latency_ms=round(latency, 1), fps=round(1000 / latency, 1))
            print(f"BENCHMARK {key}: {latency:.2f} ms, {runtime}", flush=True)
        else:
            data["models"][key].update(avg_latency_ms=None, fps=None)
        data["models"][key]["size_mb"] = round(artifact_info(path)["size_mib"], 2)
        for scene in data["scenes"]:
            pixels = cv2.imread(str(root / "demo_data/images" / scene["file"]))
            started = time.perf_counter()
            result = model(pixels, **{**options, "conf": 0.10})[0]
            elapsed = (time.perf_counter() - started) * 1000
            h, w = pixels.shape[:2]
            boxes = []
            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                cid = int(box.cls[0])
                boxes.append({"class_id": cid, "class_name": result.names[cid],
                              "conf": round(float(box.conf[0]), 3),
                              "box_px": [round(v, 1) for v in coords],
                              "box_norm": [round(v / (w if i % 2 == 0 else h), 6) for i, v in enumerate(coords)]})
            data["detections"][key][scene["id"]] = {
                "scene_id": scene["id"], "image_file": scene["file"], "width": w, "height": h,
                "inference_time_ms": round(elapsed, 1), "runtime": "CPU ×4 · 640×640",
                "runtime_settings": runtime, "detections_count": len(boxes), "boxes": boxes}
        del server.LOADED_MODELS[key], model
        gc.collect()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    data_path.write_text(payload)
    (root / "demo_data/demo_bundle.js").write_text("window.DEMO_DATA = " + payload + ";\n")
    (root / "demo_data/latency_benchmark.json").write_text(json.dumps(
        {"protocol": protocol, "models": measurements}, ensure_ascii=False, indent=2))
    sync_benchmark_assets(args.eval_dirs)


if __name__ == "__main__":
    main()
