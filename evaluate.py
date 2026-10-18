#!/usr/bin/env python3
"""CPU comparison of BDD100K detection checkpoints / Ultralytics exports.
See README.txt. Evaluation is Ultralytics YOLO mAP, not official BDD ignore rules.
"""
from __future__ import annotations
from datetime import date, datetime
from inference_runtime import cpu_device, configure_cpu_backend
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
from bdd100k_eval.artifacts import sha256, artifact_info
from bdd100k_eval.reporting import json_default, write_json

DEFAULT_NAMES = ["pedestrian", "rider", "car", "truck", "bus", "train",
                 "motorcycle", "bicycle", "traffic light", "traffic sign"]
ALIASES = {"person": "pedestrian", "bike": "bicycle", "motor": "motorcycle"}


def canonical(name):
    name = str(name).strip().lower().replace("_", " ")
    return ALIASES.get(name, name)


def read_frames(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("frames"), list):
        data = data["frames"]
    if not isinstance(data, list):
        raise ValueError("Nhãn phải là list frame BDD100K/Scalabel hoặc object có frames; không phải COCO JSON.")
    return data


def prepare_dataset(args, out):
    from PIL import Image
    import yaml

    images_dir = Path(args.images).expanduser().resolve()
    if not images_dir.is_dir():
        raise ValueError(f"Không tìm thấy thư mục ảnh: {images_dir}")
    frames = read_frames(args.labels)
    by_name = {}
    for frame in frames:
        name = frame.get("name")
        if not name or Path(name).name != name:
            raise ValueError("Frame name phải là tên ảnh, không chứa đường dẫn.")
        if name in by_name:
            raise ValueError(f"Nhãn trùng ảnh: {name}")
        by_name[name] = frame
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise ValueError("Không có ảnh.")
    missing = [p.name for p in images if p.name not in by_name]
    if missing:
        raise ValueError(f"{len(missing)} ảnh thiếu entry annotation; ví dụ: {missing[:5]}")
    if len({p.stem for p in images}) != len(images):
        raise ValueError("Tên stem ảnh trùng, sẽ ghi đè nhãn YOLO.")
    if args.limit:
        images = sorted(random.Random(args.seed).sample(images, min(args.limit, len(images))))
    names = [canonical(n) for n in args.names.split(",")]
    if len(names) != 10 or set(names) != set(DEFAULT_NAMES):
        raise ValueError("--names phải chứa đúng 10 lớp BDD100K, theo thứ tự lớp checkpoint.")
    mapping = {name: i for i, name in enumerate(names)}
    dataset = out / "dataset"
    image_out, label_out = dataset / "images/val", dataset / "labels/val"
    image_out.mkdir(parents=True)
    label_out.mkdir(parents=True)
    stats = {"images": len(images), "empty_images": 0, "boxes": 0, "unknown_box_categories": {},
             "ignore_or_crowd_boxes": 0, "clipped_boxes": 0, "degenerate_boxes": 0,
             "objects_per_class": {n: 0 for n in names}}
    manifest = []
    max_objects = 0
    for image in images:
        with Image.open(image) as im:
            width, height = im.size
        frame = by_name[image.name]
        if "labels" not in frame or frame["labels"] is None:
            raise ValueError(f"{image.name}: thiếu labels; ảnh không có vật thể cần labels: [].")
        rows = []
        for obj in frame["labels"]:
            box = obj.get("box2d")
            if box is None:
                continue  # lanes, drivable-area, and other non-detection labels
            category = canonical(obj.get("category", ""))
            if category not in mapping:
                unknown = stats["unknown_box_categories"]
                unknown[category] = unknown.get(category, 0) + 1
                continue
            attrs = obj.get("attributes") or {}
            ignored = any(attrs.get(k) is True or attrs.get(k) == 1
                          for k in ("ignore", "ignored", "crowd", "iscrowd"))
            if ignored:
                stats["ignore_or_crowd_boxes"] += 1
                if args.ignore_policy == "error":
                    raise ValueError("Có ignore/crowd: YOLO txt không biểu diễn được. Xem --ignore-policy.")
                if args.ignore_policy == "drop":
                    continue
            raw = [float(box[k]) for k in ("x1", "y1", "x2", "y2")]
            if not all(math.isfinite(v) for v in raw):
                raise ValueError(f"Box không hữu hạn: {image.name}")
            x1, y1, x2, y2 = raw
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Box đảo hoặc rỗng: {image.name}, {raw}")
            clipped = [max(0, min(width, x1)), max(0, min(height, y1)),
                       max(0, min(width, x2)), max(0, min(height, y2))]
            stats["clipped_boxes"] += int(clipped != raw)
            x1, y1, x2, y2 = clipped
            if x2 <= x1 or y2 <= y1:
                stats["degenerate_boxes"] += 1
                continue
            rows.append(f"{mapping[category]} {(x1+x2)/(2*width):.9f} {(y1+y2)/(2*height):.9f} "
                        f"{(x2-x1)/width:.9f} {(y2-y1)/height:.9f}")
            stats["objects_per_class"][category] += 1
        label = label_out / (image.stem + ".txt")
        label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        target = image_out / image.name
        try:
            target.symlink_to(image)
        except OSError:
            shutil.copy2(image, target)
        stats["empty_images"] += int(not rows)
        stats["boxes"] += len(rows)
        max_objects = max(max_objects, len(rows))
        manifest.append({"name": image.name, "width": width, "height": height,
                         "image_bytes": image.stat().st_size, "image_sha256": sha256(image),
                         "label_sha256": sha256(label)})
    if stats["unknown_box_categories"]:
        raise ValueError(f"Có box category ngoài 10 lớp; cần kiểm tra nguồn nhãn: {stats['unknown_box_categories']}")
    if not stats["boxes"]:
        raise ValueError("Không có ground-truth box.")
    if max_objects > args.max_det:
        raise ValueError(f"Có ảnh chứa {max_objects} objects > max_det={args.max_det}; tăng --max-det và kiểm tra graph export.")
    yaml_path = dataset / "bdd100k.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": str(dataset), "train": "images/val", "val": "images/val",
                                          "names": names}, allow_unicode=True), encoding="utf-8")
    write_json(out / "dataset_manifest.json", manifest)
    write_json(out / "dataset_stats.json", stats)
    return yaml_path, names, [str(image_out / p.name) for p in images], stats


def distribution(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max())}


class RSSMonitor:
    """Whole worker process RSS (includes libraries); polling may miss brief peaks."""
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.wait(0.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def inference_options(head, is_pt, precision="FP16"):
    from ultralytics.cfg import DEFAULT_CFG_DICT
    # Older YOLO26 releases exposed end2end; current releases expose nms.
    if "end2end" in DEFAULT_CFG_DICT:
        opts = {"end2end": head == "nms-free"}
    elif "nms" in DEFAULT_CFG_DICT:
        opts = {"nms": head != "nms-free"}
    else:
        raise RuntimeError("Phiên bản Ultralytics không có điều khiển head; hãy cập nhật.")
    if is_pt:
        opts.update({"quantize": 16 if precision == "FP16" else 32}
                    if "quantize" in DEFAULT_CFG_DICT else {"half": precision == "FP16"})
    return opts


def verify_names(model, names):
    raw = model.names
    actual = [raw[i] for i in range(len(raw))] if isinstance(raw, dict) else list(raw)
    if [canonical(n) for n in actual] != names:
        raise ValueError(f"Lớp model không khớp dataset. Model: {actual}; dataset: {names}. "
                         "Cần checkpoint BDD100K 10 lớp; dùng --names để chỉ định đúng thứ tự.")


def verify_runtime(backend, cfg):
    from ultralytics.cfg import DEFAULT_CFG_DICT
    metadata = getattr(backend, "metadata", {}) or {}
    if Path(cfg["model"]).suffix.lower() == ".pt":
        actual_head = "nms-free" if getattr(backend, "end2end", False) else "nms"
        import torch
        expected = torch.float16 if cfg.get("precision", "FP16") == "FP16" else torch.float32
        if next(backend.model.parameters()).dtype != expected or bool(backend.fp16) != (expected == torch.float16):
            raise ValueError("Precision PyTorch thực tế khác protocol đã yêu cầu")
    else:
        export_args = metadata.get("args", {})
        if export_args.get("nms") is True:
            actual_head = "nms"
        elif "end2end" in export_args:
            actual_head = "nms-free" if export_args["end2end"] else "nms"
        elif "end2end" in DEFAULT_CFG_DICT and "end2end" in metadata:
            actual_head = "nms-free" if metadata["end2end"] else "nms"
        elif "end2end" not in DEFAULT_CFG_DICT and "nms" in export_args:
            actual_head = "nms-free" if export_args["nms"] is False else "nms"
        else:
            raise ValueError("Export thiếu metadata xác định head. Re-export bằng cùng phiên bản Ultralytics, "
                             "chỉ định nms/head rõ ràng; shape output không đủ phân biệt hai head.")
        if int(metadata.get("batch", 1)) != 1:
            raise ValueError("Đánh giá latency batch=1 yêu cầu artifact export batch=1.")
        shape = metadata.get("imgsz")
        if shape and not metadata.get("dynamic", False):
            shape = [shape, shape] if isinstance(shape, int) else list(shape)
            if shape != [cfg["imgsz"], cfg["imgsz"]]:
                raise ValueError(f"imgsz export {shape} khác --imgsz={cfg['imgsz']}; re-export hoặc đổi cấu hình.")
    if actual_head != cfg["head"]:
        raise ValueError(f"Head thực tế {actual_head} khác --head={cfg['head']}; cần export cùng head baseline.")
    return {"actual_head": actual_head, "export_metadata": metadata}


def worker(config_path):
    cfg = json.loads(Path(config_path).read_text())
    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    torch.set_num_threads(cfg["threads"])
    cv2.setNumThreads(1)
    precision = cfg.get("precision", "FP16")
    options = inference_options(cfg["head"], Path(cfg["model"]).suffix.lower() == ".pt", precision)
    common = dict(imgsz=cfg["imgsz"], device=cpu_device(cfg["model"]), conf=cfg["conf"], iou=cfg["iou"],
                  max_det=cfg["max_det"], rect=False, augment=False, agnostic_nms=False, **options)
    validation_runtime = {}

    class CPUValidator(DetectionValidator):
        def init_metrics(self, backend):
            # val() builds a fresh AutoBackend, separate from the predictor probe.
            validation_runtime.update(configure_cpu_backend(backend, cfg["model"], cfg["threads"], precision))
            if Path(cfg["model"]).suffix.lower() == ".pt":
                if hasattr(self.args, "quantize"):
                    self.args.quantize = 16 if precision == "FP16" else 32
                else:
                    self.args.half = precision == "FP16"
            verify_runtime(backend, cfg)
            super().init_metrics(backend)

    with RSSMonitor() as memory:
        model = YOLO(cfg["model"], task="detect")
        verify_names(model, cfg["names"])
        probe = cv2.imread(cfg["benchmark_images"][0])
        if probe is None:
            raise ValueError("Không đọc được ảnh kiểm tra runtime.")
        model.predict(source=probe, verbose=False, save=False, **common)
        configure_cpu_backend(model, cfg["model"], cfg["threads"], precision)
        verify_runtime(model.predictor.model, cfg)
        del probe
        metrics = model.val(validator=CPUValidator, data=cfg["data"], split="val", batch=1, workers=0,
                            plots=cfg["plots"], save_json=False, project=cfg["output"],
                            name="validation", exist_ok=False, verbose=True, **common)
        box = metrics.box
        accuracy = {"map50_95": float(box.map), "map50": float(box.map50), "map75": float(box.map75),
                    "precision_macro": float(box.mp), "recall_macro": float(box.mr),
                    "f1_macro": float(box.f1.mean())}
        per_class = []
        for row, class_id in enumerate(box.ap_class_index):
            cid = int(class_id)
            per_class.append({"class_id": cid, "class": cfg["names"][cid],
                              "precision": float(box.p[row]), "recall": float(box.r[row]),
                              "f1": float(box.f1[row]), "ap50": float(box.ap50[row]),
                              "ap50_95": float(box.ap[row])})
        # Fresh model: validation may fuse/change internal model state.
        del metrics, model, box
        gc.collect()
        model = YOLO(cfg["model"], task="detect")
        predict_opts = dict(common, verbose=False, save=False, stream=False, batch=1)
        images = cfg["benchmark_images"]
        model.predict(source=cv2.imread(images[0]), **predict_opts)
        cpu_runtime = configure_cpu_backend(model, cfg["model"], cfg["threads"], precision)
        for i in range(cfg["warmup"]):
            im = cv2.imread(images[i % len(images)])
            if im is None:
                raise ValueError(f"Không đọc được ảnh: {images[i % len(images)]}")
            model.predict(source=im, **predict_opts)
        # Ultralytics changes this on its first predict call; restore the requested
        # value after warmup so the measured samples match the recorded protocol.
        torch.set_num_threads(cfg["threads"])
        if torch.get_num_threads() != cfg["threads"]:
            raise RuntimeError("Không thể áp dụng số luồng PyTorch đã yêu cầu.")
        verified_runtime = verify_runtime(model.predictor.model, cfg)
        samples = []
        for repeat in range(cfg["repeats"]):
            for path in images:
                t0 = time.perf_counter_ns()
                im = cv2.imread(path)
                t1 = time.perf_counter_ns()
                if im is None:
                    raise ValueError(f"Không đọc được ảnh: {path}")
                results = model.predict(source=im, **predict_opts)
                count = len(results[0].boxes)
                t2 = time.perf_counter_ns()
                speed = results[0].speed
                samples.append({"image": Path(path).name, "repeat": repeat, "detections": count,
                                "decode_ms": (t1-t0)/1e6, "predict_wall_ms": (t2-t1)/1e6,
                                "decode_plus_predict_ms": (t2-t0)/1e6,
                                "preprocess_ms": float(speed["preprocess"]),
                                "inference_ms": float(speed["inference"]),
                                "postprocess_ms": float(speed["postprocess"])})
        backend = model.predictor.model
        runtime = {"backend_class": type(backend).__name__, "torch_threads": torch.get_num_threads(),
                   "fp16_flag": bool(getattr(backend, "fp16", False)),
                   "head_options": options}
        runtime.update(verified_runtime)
        runtime.update(cpu_runtime)
        runtime["validation"] = validation_runtime
        session = getattr(backend, "session", None)
        if session is not None and hasattr(session, "get_providers"):
            runtime["onnx_providers"] = session.get_providers()
            so = session.get_session_options()
            runtime["ort_intra_op_threads"] = so.intra_op_num_threads
            runtime["ort_inter_op_threads"] = so.inter_op_num_threads
        # cpu_runtime records the executed backend, not YOLO's separate source copy.
        latency = {key: distribution([s[key] for s in samples]) for key in
                   ("predict_wall_ms", "decode_plus_predict_ms", "decode_ms", "preprocess_ms",
                    "inference_ms", "postprocess_ms")}
    result = {"label": cfg["label"], "accuracy": accuracy, "per_class": per_class,
              "latency": latency, "serial_fps": 1000 / latency["predict_wall_ms"]["mean"],
              "rss_peak_sampled_mib": memory.peak / 2**20, "runtime": runtime,
              "latency_samples": len(samples), "latency_unique_images": len(images)}
    write_json(Path(cfg["output"]) / "metrics.json", result)
    with (Path(cfg["output"]) / "latency_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)


def comparison(baseline, candidate):
    b, c = baseline["accuracy"], candidate["accuracy"]
    bsize, csize = baseline["artifact"]["size_mib"], candidate["artifact"]["size_mib"]
    latency_b = baseline["latency"]["predict_wall_ms"]["mean"]
    latency_c = candidate["latency"]["predict_wall_ms"]["mean"]
    return {"map_drop_pp": 100 * (b["map50_95"] - c["map50_95"]),
            "map_drop_relative_pct": 100 * (b["map50_95"] - c["map50_95"]) / b["map50_95"]
            if b["map50_95"] > 0 else None,
            "map50_drop_pp": 100 * (b["map50"] - c["map50"]),
            "speedup": latency_b / latency_c,
            "latency_reduction_pct": 100 * (1 - latency_c / latency_b),
            "size_compression_ratio": bsize / csize,
            "size_reduction_pct": 100 * (1 - csize / bsize)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--images", help="BDD100K images/100k/val")
    p.add_argument("--labels", help="BDD100K detection val JSON")
    p.add_argument("--baseline", help="Checkpoint gốc 10 lớp BDD100K, thường FP32 .pt")
    p.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--names", default=",".join(DEFAULT_NAMES))
    p.add_argument("--output", default="bdd100k_evaluation")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--precision", choices=["FP16", "FP32"], default="FP16", help="Precision PyTorch; không ép dtype graph INT8")
    p.add_argument("--head", choices=["nms", "nms-free"], default="nms-free")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU; không phải ngưỡng tính mAP")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--limit", type=int, default=0, help="0: toàn bộ ảnh; >0: subset ngẫu nhiên cố định")
    p.add_argument("--latency-images", type=int, default=1000, help="0: toàn bộ ảnh đã chọn")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--ignore-policy", choices=["include", "drop", "error"], default="include")
    args = p.parse_args()
    if args.worker:
        worker(args.worker)
        return
    for name in ("images", "labels", "baseline"):
        if not getattr(args, name):
            p.error(f"Thiếu --{name}")
    args.labels = str(Path(args.labels).expanduser().resolve())
    if any(getattr(args, n) < 1 for n in ("imgsz", "threads", "max_det", "warmup", "repeats")):
        p.error("imgsz, threads, max-det, warmup, repeats phải > 0")
    if args.limit < 0 or args.latency_images < 0 or not 0 <= args.conf <= 1 or not 0 < args.iou <= 1:
        p.error("Giới hạn >=0, conf thuộc [0,1], iou thuộc (0,1]")
    models = [("baseline", args.baseline)]
    for value in args.candidate:
        if "=" not in value:
            p.error("--candidate phải có dạng NAME=PATH")
        label, path = value.split("=", 1)
        if not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in label):
            p.error("NAME chỉ gồm chữ, số, _ và -")
        models.append((label, path))
    if len({label for label, _ in models}) != len(models):
        p.error("NAME model trùng")
    models = [(label, str(Path(path).expanduser().resolve())) for label, path in models]
    for label, path in models:
        if not Path(path).exists():
            p.error(f"Không tìm thấy model {label}: {path}")
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)  # prevents silently reusing old labels / cache / results
    data, names, images, stats = prepare_dataset(args, out)
    bench = sorted(random.Random(args.seed + 1).sample(images, min(args.latency_images or len(images), len(images))))
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(args.threads)
    env["CUDA_VISIBLE_DEVICES"] = ""
    versions = {}
    for package in ("ultralytics", "torch", "onnxruntime", "openvino", "numpy", "opencv-python", "Pillow", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {"protocol": {"metric": "Ultralytics detection mAP (not official BDD ignore/crowd evaluation)",
                           "args": vars(args), "names": names, "dataset_stats": stats,
                           "manifest_sha256": sha256(out / "dataset_manifest.json"),
                           "source_labels_sha256": sha256(args.labels),
                           "benchmark_images": [Path(s).name for s in bench],
                           "accuracy_confidence_note": "P/R/F1 at Ultralytics validation max-F1 operating point; not fixed-conf P/R",
                           "latency_note": "Batch=1 predict wall: preprocessing + inference + postprocessing + Python API, excluding disk/decode; warmup excluded",
                           "cpu_thread_note": "PyTorch/OpenVINO CPU threads enforced for latency; ORT settings reported separately",
                           "memory_note": "sampled peak worker RSS includes validation, runtime, model, and allocator retained memory",
                           "platform": platform.platform(), "processor": platform.processor(),
                           "logical_cpus": os.cpu_count(), "versions": versions}, "models": []}
    for label, path in models:
        run = out / label
        run.mkdir()
        cfg = {**vars(args), "model": path, "label": label, "output": str(run), "data": str(data),
               "names": names, "benchmark_images": bench}
        config_path = run / "config.json"
        write_json(config_path, cfg)
        print(f"\n=== {label}: {path} ===", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(config_path)], env=env, check=True)
        result = json.loads((run / "metrics.json").read_text())
        result["artifact"] = artifact_info(path)
        result["comparison"] = comparison(report["models"][0], result) if report["models"] else None
        report["models"].append(result)
        write_json(out / "report.json", report)  # checkpoint after each model
    rows = []
    for r in report["models"]:
        a, lat = r["accuracy"], r["latency"]["predict_wall_ms"]
        rows.append({"model": r["label"], "mAP50_95_pct": 100*a["map50_95"], "mAP50_pct": 100*a["map50"],
                     "mAP75_pct": 100*a["map75"], "precision_pct": 100*a["precision_macro"],
                     "recall_pct": 100*a["recall_macro"], "f1_pct": 100*a["f1_macro"],
                     "latency_mean_ms": lat["mean"], "latency_median_ms": lat["median"], "latency_p95_ms": lat["p95"],
                     "inference_mean_ms": r["latency"]["inference_ms"]["mean"], "serial_FPS": r["serial_fps"],
                     "size_MiB": r["artifact"]["size_mib"], "RSS_peak_MiB": r["rss_peak_sampled_mib"],
                     **(r["comparison"] or {"map_drop_pp": 0, "map_drop_relative_pct": 0, "map50_drop_pp": 0,
                                             "speedup": 1, "latency_reduction_pct": 0,
                                             "size_compression_ratio": 1, "size_reduction_pct": 0})})
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nModel          mAP50-95  drop(pp)  mean(ms)  p95(ms)   FPS    speedup")
    for r in rows:
        print(f"{r['model']:<14} {r['mAP50_95_pct']:8.3f} {r['map_drop_pp']:9.3f} "
              f"{r['latency_mean_ms']:9.3f} {r['latency_p95_ms']:8.3f} {r['serial_FPS']:7.2f} {r['speedup']:8.3f}")
    print(f"\nKết quả: {out / 'summary.csv'} và {out / 'report.json'}")


if __name__ == "__main__":
    main()
