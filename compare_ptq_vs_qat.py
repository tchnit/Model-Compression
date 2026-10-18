#!/usr/bin/env python3
"""Script chạy lại và so sánh đối đầu PTQ INT8 vs QAT INT8 của YOLO26m trên BDD100K."""

from __future__ import annotations
from inference_runtime import cpu_device, configure_cpu_backend
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional
from bdd100k_eval.reporting import format_table, export_csv


DEFAULT_BASELINE = "models/yolo26m-bdd100k.pt"
DEFAULT_PTQ = "models/int8_yolo26m_640_openvino_model"
DEFAULT_QAT = "models/int8_yolo26m_qat_640_openvino_model"
DEFAULT_YOLO26S = "models/int8_yolo26s_640_openvino_model"
DEFAULT_YOLO26S_BASELINE = "models/yolo26s-bdd100k.pt"
DEFAULT_IMAGES = "data/bdd100k/images/val"
DEFAULT_LABELS = "data/bdd100k/labels/bdd100k_labels_images_val.json"


def parse_report_and_compare(report_path: Path) -> Dict[str, Any]:
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    models_map = {}
    for m in data.get("models", []):
        label = m.get("label", "").lower()
        artifact = Path(m.get("artifact", {}).get("path", "")).name.lower()
        role = "baseline" if label == "baseline" else "qat" if "qat" in artifact or "qat" in label else "ptq" if "ptq" in label or label == "int8" else None
        if role:
            if role in models_map:
                raise ValueError(f"Báo cáo có nhiều model cho vai trò {role}")
            models_map[role] = m
    missing = {"baseline", "ptq", "qat"} - models_map.keys()
    if missing:
        raise ValueError("Báo cáo thiếu model: " + ", ".join(sorted(missing)))
    for role, model in models_map.items():
        artifact = Path(model.get("artifact", {}).get("path", "")).name.lower()
        if "yolo26s" in artifact:
            raise ValueError("Báo cáo YOLO26m không được dùng baseline/candidate YOLO26s")
        required = [("accuracy", key) for key in ("map50_95", "map50", "map75", "precision_macro", "recall_macro", "f1_macro")]
        if any(model.get(group, {}).get(key) is None for group, key in required):
            raise ValueError(f"{role}: thiếu chỉ số accuracy")
        if any(not isinstance(model[group][key], (int, float)) or not math.isfinite(model[group][key])
               or not 0 <= model[group][key] <= 1 for group, key in required):
            raise ValueError(f"{role}: accuracy phải hữu hạn và thuộc [0, 1]")
        for group, keys in (("predict_wall_ms", ("mean", "median", "p95")), ("inference_ms", ("mean",))):
            if any(model.get("latency", {}).get(group, {}).get(key) is None for key in keys):
                raise ValueError(f"{role}: thiếu chỉ số latency")
        if model.get("artifact", {}).get("size_mib") is None or model.get("serial_fps") is None:
            raise ValueError(f"{role}: thiếu dung lượng/FPS")
        positive = [model['artifact']['size_mib'], model['serial_fps'],
                    *[model['latency']['predict_wall_ms'][key] for key in ('mean', 'median', 'p95')]]
        positive.append(model['latency']['inference_ms']['mean'])
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError(f"{role}: dung lượng/FPS/latency phải hữu hạn và lớn hơn 0")
    base, ptq, qat = (models_map[key] for key in ("baseline", "ptq", "qat"))

    def get_acc(m, key):
        return m.get("accuracy", {}).get(key, 0.0) * 100

    def get_lat(m, key):
        return m.get("latency", {}).get("predict_wall_ms", {}).get(key, 0.0)

    def get_inf(m):
        return m.get("latency", {}).get("inference_ms", {}).get("mean", 0.0)

    def get_size(m):
        return m.get("artifact", {}).get("size_mib", 0.0)

    def get_fps(m):
        return m.get("serial_fps", 0.0)

    headers = [
        "Chỉ số (Metric)",
        "Baseline",
        "PTQ INT8",
        "QAT INT8",
        "PTQ vs Baseline",
        "QAT vs Baseline",
        "QAT vs PTQ (Δ)"
    ]
    align_right = [False, True, True, True, True, True, True]

    rows = []

    # Accuracy metrics
    for name, key in [
        ("mAP50-95 (%)", "map50_95"),
        ("mAP50 (%)", "map50"),
        ("mAP75 (%)", "map75"),
        ("Precision (%)", "precision_macro"),
        ("Recall (%)", "recall_macro"),
        ("F1 Score (%)", "f1_macro"),
    ]:
        v_base = get_acc(base, key)
        v_ptq = get_acc(ptq, key)
        v_qat = get_acc(qat, key)
        d_ptq = v_ptq - v_base
        d_qat = v_qat - v_base
        d_diff = v_qat - v_ptq
        rows.append([
            name,
            f"{v_base:.2f}%",
            f"{v_ptq:.2f}%",
            f"{v_qat:.2f}%",
            f"{d_ptq:+.2f} pp",
            f"{d_qat:+.2f} pp",
            f"{d_diff:+.2f} pp",
        ])

    # Latency metrics
    for name, fn in [
        ("Latency Mean (ms)", lambda m: get_lat(m, "mean")),
        ("Latency Median (ms)", lambda m: get_lat(m, "median")),
        ("Latency P95 (ms)", lambda m: get_lat(m, "p95")),
        ("Inference Mean (ms)", get_inf),
    ]:
        l_base = fn(base)
        l_ptq = fn(ptq)
        l_qat = fn(qat)
        sp_ptq = (l_base / l_ptq) if l_ptq > 0 else 1.0
        sp_qat = (l_base / l_qat) if l_qat > 0 else 1.0
        diff_ms = l_qat - l_ptq
        rows.append([
            name,
            f"{l_base:.1f}",
            f"{l_ptq:.1f}",
            f"{l_qat:.1f}",
            f"{sp_ptq:.2f}x nhanh hơn",
            f"{sp_qat:.2f}x nhanh hơn",
            f"{diff_ms:+.1f} ms",
        ])

    # Throughput & size
    fps_base = get_fps(base)
    fps_ptq = get_fps(ptq)
    fps_qat = get_fps(qat)
    rows.append([
        "Serial FPS",
        f"{fps_base:.1f}",
        f"{fps_ptq:.1f}",
        f"{fps_qat:.1f}",
        f"{fps_ptq - fps_base:+.1f} FPS",
        f"{fps_qat - fps_base:+.1f} FPS",
        f"{fps_qat - fps_ptq:+.1f} FPS",
    ])

    sz_base = get_size(base)
    sz_ptq = get_size(ptq)
    sz_qat = get_size(qat)
    rows.append([
        "Kích thước (MiB)",
        f"{sz_base:.1f}",
        f"{sz_ptq:.1f}",
        f"{sz_qat:.1f}",
        f"{sz_base / sz_ptq:.1f}x nhỏ hơn" if sz_ptq else "-",
        f"{sz_base / sz_qat:.1f}x nhỏ hơn" if sz_qat else "-",
        f"{sz_qat - sz_ptq:+.1f} MiB",
    ])

    return {"headers": headers, "rows": rows, "align_right": align_right}


def export_markdown(headers: List[str], rows: List[List[str]], out_path: Path, title: str) -> None:
    content = f"# {title}\n\n" + format_table(headers, rows) + "\n"
    out_path.write_text(content, encoding="utf-8")


def run_latency_only_benchmark(
    baseline_path: str,
    ptq_path: str,
    qat_path: str,
    yolo26s_path: Optional[str] = DEFAULT_YOLO26S,
    yolo26s_baseline: Optional[str] = DEFAULT_YOLO26S_BASELINE,
    threads: int = 4,
    repeats: int = 50,
    warmup: int = 10,
) -> None:
    """Đo so sánh latency nhanh batch=1 cho Baseline, PTQ, QAT và YOLO26s mà không cần chạy mAP validation."""
    import numpy as np
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)

    models_info = [
        ("YOLO26m FP16 (Base)", baseline_path, "baseline"),
        ("YOLO26m PTQ INT8", ptq_path, "yolo26m"),
        ("YOLO26m QAT INT8", qat_path, "yolo26m"),
    ]
    if yolo26s_baseline and Path(yolo26s_baseline).exists():
        models_info.append(("YOLO26s FP16 (Base)", yolo26s_baseline, "baseline_s"))
    if yolo26s_path and Path(yolo26s_path).exists():
        models_info.append(("YOLO26s INT8 (OpenVINO)", yolo26s_path, "yolo26s"))

    print(f"\n[INFO] Chạy benchmark latency (threads={threads}, warmup={warmup}, repeats={repeats})...\n")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    results = []

    for label, path, fam in models_info:
        print(f"--> Đang đo {label:<25} ({path})...", end="", flush=True)
        m = YOLO(path, task="detect")
        m.predict(dummy, imgsz=640, device=cpu_device(path), rect=False, verbose=False)
        configure_cpu_backend(m, path, threads)
        for _ in range(warmup):
            m.predict(dummy, imgsz=640, device=cpu_device(path), rect=False, verbose=False)
        torch.set_num_threads(threads)

        times = []
        inf_times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            preds = m.predict(dummy, imgsz=640, device=cpu_device(path), rect=False, verbose=False)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
            inf_times.append(float(preds[0].speed.get("inference", 0.0)))

        t_arr = np.array(times)
        lat_mean = float(t_arr.mean())
        lat_med = float(np.median(t_arr))
        lat_p95 = float(np.percentile(t_arr, 95))
        inf_mean = float(np.mean(inf_times))
        fps = 1000.0 / lat_mean if lat_mean > 0 else 0.0

        p = Path(path)
        sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024) if p.is_dir() else p.stat().st_size / (1024 * 1024)

        results.append({
            "label": label,
            "fam": fam,
            "size": sz,
            "mean": lat_mean,
            "median": lat_med,
            "p95": lat_p95,
            "inf": inf_mean,
            "fps": fps,
        })
        print(f" Xong! Latency: {lat_mean:.1f} ms | FPS: {fps:.1f}")

    base_m_lat = next((r["mean"] for r in results if r["fam"] == "baseline"), results[0]["mean"])
    base_s_lat = next((r["mean"] for r in results if r["fam"] == "baseline_s"), base_m_lat)

    headers = ["Model", "Size (MiB)", "Mean (ms)", "Median (ms)", "p95 (ms)", "Inf (ms)", "FPS", "Speedup vs Base"]
    rows = []
    for r in results:
        ref_lat = base_s_lat if "26s" in r["label"] else base_m_lat
        speedup = ref_lat / r["mean"] if r["mean"] > 0 else 1.0
        rows.append([
            r["label"],
            f"{r['size']:.1f}",
            f"{r['mean']:.1f}",
            f"{r['median']:.1f}",
            f"{r['p95']:.1f}",
            f"{r['inf']:.1f}",
            f"{r['fps']:.1f}",
            f"{speedup:.2f}x"
        ])

    print("\n" + "=" * 95)
    print(" BẢNG SO SÁNH LATENCY & THROUGHPUT (Batch=1, 640x640, CPU)")
    print("=" * 95)
    print(format_table(headers, rows, align_right=[False] + [True] * (len(headers) - 1)))
    print("=" * 95 + "\n")


def self_test():
    dummy_report = {
        "models": [
            {
                "label": "baseline",
                "accuracy": {"map50_95": 0.3306, "map50": 0.573, "map75": 0.321, "precision_macro": 0.74, "recall_macro": 0.50, "f1_macro": 0.56},
                "latency": {"predict_wall_ms": {"mean": 227.0, "median": 223.0, "p95": 249.0}, "inference_ms": {"mean": 224.0}},
                "serial_fps": 4.4,
                "artifact": {"size_mib": 42.0}
            },
            {
                "label": "PTQ_INT8",
                "accuracy": {"map50_95": 0.3276, "map50": 0.571, "map75": 0.317, "precision_macro": 0.73, "recall_macro": 0.51, "f1_macro": 0.56},
                "latency": {"predict_wall_ms": {"mean": 76.4, "median": 70.2, "p95": 108.0}, "inference_ms": {"mean": 70.5}},
                "serial_fps": 13.1,
                "artifact": {"size_mib": 20.5}
            },
            {
                "label": "QAT_INT8",
                "accuracy": {"map50_95": 0.3145, "map50": 0.555, "map75": 0.298, "precision_macro": 0.63, "recall_macro": 0.52, "f1_macro": 0.56},
                "latency": {"predict_wall_ms": {"mean": 82.0, "median": 72.0, "p95": 130.0}, "inference_ms": {"mean": 77.0}},
                "serial_fps": 12.2,
                "artifact": {"size_mib": 20.5}
            }
        ]
    }
    with TemporaryDirectory() as directory:
        p = Path(directory) / "report.json"
        p.write_text(json.dumps(dummy_report), encoding="utf-8")
        res = parse_report_and_compare(p)
        assert len(res["rows"]) >= 6, "Thiếu dữ liệu so sánh trong self-test"
    print("Self-test thành công!")


def main():
    parser = argparse.ArgumentParser(description="Chạy lại và so sánh đối đầu PTQ INT8 vs QAT INT8 của YOLO26m trên BDD100K.")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="Đường dẫn checkpoint baseline (.pt)")
    parser.add_argument("--ptq", default=DEFAULT_PTQ, help="Đường dẫn OpenVINO model PTQ INT8")
    parser.add_argument("--qat", default=DEFAULT_QAT, help="Đường dẫn OpenVINO model QAT INT8")
    parser.add_argument("--yolo26s", default=DEFAULT_YOLO26S, help="Đường dẫn model int8_yolo26s_640_openvino_model")
    parser.add_argument("--yolo26s-baseline", default=DEFAULT_YOLO26S_BASELINE, help="Đường dẫn baseline yolo26s-bdd100k.pt")
    parser.add_argument("--images", default=DEFAULT_IMAGES, help="Thư mục ảnh val BDD100K")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="File nhãn JSON BDD100K")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước input ảnh (mặc định: 640)")
    parser.add_argument("--threads", type=int, default=4, help="Số luồng CPU (mặc định: 4)")
    parser.add_argument("--limit", type=int, default=0, help="0: toàn bộ 10,000 ảnh val; >0: số lượng subset ảnh")
    parser.add_argument("--smoke", action="store_true", help="Chạy nhanh smoke test trên 50 ảnh")
    parser.add_argument("--latency-only", action="store_true", help="Chỉ đo latency/FPS nhanh trên CPU mà không tính mAP")
    parser.add_argument("--latency-images", type=int, default=1000, help="Số ảnh đo latency (mặc định: 1000)")
    parser.add_argument("--warmup", type=int, default=30, help="Số lần warmup latency (mặc định: 30)")
    parser.add_argument("--repeats", type=int, default=3, help="Số lần lặp benchmark latency (mặc định: 3)")
    parser.add_argument("--output", default="evaluation_ptq_vs_qat", help="Thư mục lưu kết quả đánh giá")
    parser.add_argument("--report-only", action="store_true", help="Chỉ đọc và hiển thị báo cáo từ thư mục output đã chạy")
    parser.add_argument("--test", action="store_true", help="Chạy kiểm tra logic script")

    args = parser.parse_args()

    if args.test:
        self_test()
        return

    if not args.report_only:
        for name in ("baseline", "ptq", "qat", "yolo26s", "yolo26s_baseline"):
            if not Path(getattr(args, name)).exists():
                parser.error(f"Không tìm thấy model --{name}: {getattr(args, name)}")

    # Chế độ đo latency nhanh
    if args.latency_only:
        run_latency_only_benchmark(
            baseline_path=args.baseline,
            ptq_path=args.ptq,
            qat_path=args.qat,
            yolo26s_path=args.yolo26s,
            yolo26s_baseline=args.yolo26s_baseline,
            threads=args.threads,
            repeats=args.repeats,
            warmup=args.warmup
        )
        return

    out_dir = Path(args.output).expanduser().resolve()

    # Nếu chưa chạy hoặc không chọn report-only, thực thi evaluate.py
    if not args.report_only:
        if out_dir.exists():
            # Tạo thư mục output mới nếu thư mục đã có sẵn để tránh lỗi evaluate.py
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_dir = out_dir.parent / f"{out_dir.name}_{timestamp}"
            print(f"[INFO] Thư mục output cũ đã tồn tại, chuyển sang thư mục mới: {out_dir}")

        limit = 50 if args.smoke else args.limit
        lat_images = 20 if args.smoke else args.latency_images
        warmup = 5 if args.smoke else args.warmup
        repeats = 1 if args.smoke else args.repeats

        cmd = [
            sys.executable,
            "evaluate.py",
            "--images", args.images,
            "--labels", args.labels,
            "--baseline", args.baseline,
            "--candidate", f"PTQ_INT8={args.ptq}",
            "--candidate", f"QAT_INT8={args.qat}",
            "--imgsz", str(args.imgsz),
            "--threads", str(args.threads),
            "--head", "nms-free",
            "--limit", str(limit),
            "--latency-images", str(lat_images),
            "--warmup", str(warmup),
            "--repeats", str(repeats),
            "--output", str(out_dir)
        ]

        print(f"\n[INFO] Bắt đầu chạy đánh giá đối đầu qua evaluate.py...")
        print(f"Lệnh: {' '.join(cmd)}\n")
        subprocess.run(cmd, check=True)

    report_file = out_dir / "report.json"
    if not report_file.exists():
        print(f"[LỖI] Không tìm thấy file {report_file}")
        return

    comparison = parse_report_and_compare(report_file)
    title = f"SO SÁNH ĐỐI ĐẦU YOLO26m: BASELINE vs PTQ INT8 vs QAT INT8"

    print("\n" + "=" * 115)
    print(f" {title}")
    print("=" * 115)
    print(format_table(comparison["headers"], comparison["rows"], align_right=comparison["align_right"]))
    print("=" * 115 + "\n")

    csv_path = out_dir / "comparison_ptq_vs_qat.csv"
    md_path = out_dir / "comparison_ptq_vs_qat.md"
    export_csv(comparison["headers"], comparison["rows"], csv_path)
    export_markdown(comparison["headers"], comparison["rows"], md_path, title=title)
    print(f"[EXPORT] Đã lưu báo cáo đối đầu vào:")
    print(f"  - CSV:      {csv_path}")
    print(f"  - Markdown: {md_path}")


if __name__ == "__main__":
    main()
