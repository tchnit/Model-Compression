#!/usr/bin/env python3
"""Script so sánh benchmark (mAP, FPS, latency, compression) và đo latency các model YOLO26 trên BDD100K."""

from __future__ import annotations
from bdd100k_eval.inference_runtime import cpu_device, configure_cpu_backend
import hashlib
from bdd100k_eval.artifacts import artifact_info, get_dir_size_mib
from bdd100k_eval.reporting import format_table, export_csv
import argparse
import csv
import glob
import json
import os
from pathlib import Path
from bdd100k_eval.config import ROOT
import time
from typing import Any, Dict, List, Optional


def find_eval_dirs(custom_dirs: Optional[List[str]] = None) -> List[str]:
    if custom_dirs:
        return [d for d in custom_dirs if os.path.isdir(d)]
    candidates = sorted(glob.glob("evaluation_*"))
    if os.path.isdir("TESTT") and "TESTT" not in candidates:
        candidates.append("TESTT")
    return [d for d in candidates if os.path.exists(os.path.join(d, "report.json")) or os.path.exists(os.path.join(d, "summary.csv"))]


def parse_eval_dir(eval_dir: str) -> List[Dict[str, Any]]:
    rep_path = os.path.join(eval_dir, "report.json")
    report_data = {}
    if os.path.exists(rep_path):
        with open(rep_path, encoding="utf-8") as f:
            report_data = json.load(f)

    imgsz = report_data.get("protocol", {}).get("args", {}).get("imgsz", 640)
    models_data: List[Dict[str, Any]] = list(report_data.get("models", []))
    known_labels = {m.get("label") for m in models_data}

    # Quét thêm các subfolder có metrics.json chưa được lưu vào report.json (ví dụ QAT INT8)
    for entry in sorted(os.listdir(eval_dir)):
        sub_dir = os.path.join(eval_dir, entry)
        if not os.path.isdir(sub_dir) or entry in known_labels:
            continue
        m_file = os.path.join(sub_dir, "metrics.json")
        cfg_file = os.path.join(sub_dir, "config.json")
        if os.path.exists(m_file):
            with open(m_file, encoding="utf-8") as f:
                sub_metrics = json.load(f)
            model_path = ""
            if os.path.exists(cfg_file):
                with open(cfg_file, encoding="utf-8") as f:
                    sub_cfg = json.load(f)
                    model_path = sub_cfg.get("model", "")
            sub_metrics["label"] = entry
            sub_metrics["artifact"] = {"path": model_path}
            models_data.append(sub_metrics)

    if not models_data:
        return []

    protocol = report_data.get("protocol", {})
    args = protocol.get("args", {})
    accuracy_protocol = {key: args.get(key) for key in ("imgsz", "conf", "iou", "max_det", "head", "ignore_policy")}
    accuracy_protocol["precision"] = args.get("precision", "FP32")
    accuracy_protocol.update(manifest=protocol.get("manifest_sha256"),
                             ultralytics=protocol.get("versions", {}).get("ultralytics"))
    latency_file = ROOT / "demo_data/latency_benchmark.json"
    refreshed = json.loads(latency_file.read_text()) if latency_file.exists() else {}

    # Baseline đầu tiên làm mốc so sánh
    base = models_data[0]
    base_map = base.get("accuracy", {}).get("map50_95", 0.0) * 100
    records = []
    for m in models_data:
        lbl = m.get("label", "unknown")
        acc = m.get("accuracy", {})
        lat = m.get("latency", {}).get("predict_wall_ms", {})
        m_path_raw = m.get("artifact", {}).get("path", "")
        m_path = Path(m_path_raw) if m_path_raw else Path(lbl)
        is_pt = m_path.suffix.lower() == ".pt"
        accuracy_precision = args.get("precision", "FP32") if is_pt else "INT8"
        accuracy_available = not is_pt or (accuracy_precision == "FP16" and
            m.get("runtime", {}).get("validation", {}).get("parameter_dtype") == "torch.float16")
        
        artifact_exists = bool(m_path_raw) and m_path.exists()
        if artifact_exists and m.get("artifact", {}).get("files"):
            artifact_exists = artifact_info(m_path)["files"] == m["artifact"]["files"]
        sz = get_dir_size_mib(m_path) if artifact_exists else 0.0
        measured = refreshed.get("models", {}).get(m_path.name)
        snapshot_protocol = refreshed.get("protocol", {})
        snapshot_runtime = measured.get("runtime", {}) if measured else {}
        verified = bool(artifact_exists and measured and measured["artifact"]["files"] == m.get("artifact", {}).get("files")
                        and snapshot_protocol.get("imgsz") == imgsz and snapshot_protocol.get("rect") is False
                        and snapshot_protocol.get("threads") == snapshot_runtime.get("torch_threads")
                        and snapshot_runtime.get("device") == snapshot_protocol.get("device") == "CPU"
                        and ("ov_threads" not in snapshot_runtime
                             or snapshot_runtime["ov_threads"] == snapshot_protocol["threads"]))
        if is_pt:
            verified = verified and snapshot_runtime.get("precision") == "FP16" and snapshot_runtime.get("parameter_dtype") == "torch.float16"
        latency_protocol = {"source": eval_dir, "runtime": m.get("runtime", {}),
                            "requested_threads": args.get("threads")}
        if verified:
            lat = measured["latency"]
            latency_protocol = refreshed["protocol"]
        protocol_key = hashlib.sha256(json.dumps(latency_protocol, sort_keys=True).encode()).hexdigest()
        accuracy_key = hashlib.sha256(json.dumps(accuracy_protocol, sort_keys=True).encode()).hexdigest()

        map50_95 = acc.get("map50_95", 0.0) * 100 if accuracy_available else None
        map50 = acc.get("map50", 0.0) * 100 if accuracy_available else None
        precision = acc.get("precision_macro", 0.0) * 100 if accuracy_available else None
        recall = acc.get("recall_macro", 0.0) * 100 if accuracy_available else None
        f1 = acc.get("f1_macro", 0.0) * 100 if accuracy_available else None
        lat_mean = lat.get("mean", 0.0)
        lat_p95 = lat.get("p95", 0.0)
        fps = 1000.0 / lat_mean if lat_mean > 0 else 0.0

        drop_pp = (0.0 if lbl == "baseline" else base_map - map50_95) if accuracy_available else None

        records.append({
            "experiment": eval_dir,
            "accuracy_protocol": accuracy_key,
            "latency_protocol": protocol_key,
            "latency_verified": verified,
            "measurement": (f"CPU {snapshot_protocol['threads']} threads; {len(snapshot_protocol['images'])} images × "
                            f"{snapshot_protocol['repeats']}; {imgsz}×{imgsz}; baseline PyTorch "
                            f"{snapshot_protocol.get('pytorch_precision', 'FP32')}") if verified else f"Historical/unverified: {eval_dir}",
            "role": "QAT" if lbl != "baseline" and "qat" in m_path.name.lower() else lbl,
            "model_name": m_path.name if m_path.name else lbl,
            "artifact_path": str(m_path),
            "artifact_exists": artifact_exists,
            "accuracy_precision": accuracy_precision,
            "accuracy_pending": not accuracy_available,
            "latency_precision": snapshot_runtime.get("precision", "INT8") if verified else "unverified",
            "imgsz": imgsz,
            "size_mib": round(sz, 2),
            "map50_95_raw_pct": map50_95,
            "map50_95_pct": round(map50_95, 2) if map50_95 is not None else None,
            "map50_pct": round(map50, 2) if map50 is not None else None,
            "precision_pct": round(precision, 2) if precision is not None else None,
            "recall_pct": round(recall, 2) if recall is not None else None,
            "f1_pct": round(f1, 2) if f1 is not None else None,
            "map_drop_pp": round(drop_pp, 2) if drop_pp is not None else None,
            "lat_mean_ms": round(lat_mean, 2),
            "lat_p95_ms": round(lat_p95, 2),
            "serial_fps": round(fps, 1),
            # Filled after all experiments have been aggregated, using the
            # global YOLO26m-640 reference selected below.
            "speedup": None
        })
    return records


def infer_family(name: str) -> str:
    n = name.lower()
    if "yolo26s" in n:
        return "YOLO26s"
    if "yolo26m" in n or "qat" in n:
        return "YOLO26m"
    return "YOLO26"


def deduplicate_records(all_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep distinct protocols; never select a run because it happens to be faster."""
    def canonical_name(r: Dict[str, Any]) -> str:
        name = r["model_name"]
        # int8_openvino_model và int8_640_openvino_model là cùng 1 model 28.01% mAP
        if name == "int8_openvino_model" and abs(r["map50_95_pct"] - 28.01) < 0.1:
            return "int8_640_openvino_model"
        return name

    seen: Dict[tuple[str, int], Dict[str, Any]] = {}
    for r in all_records:
        c_name = canonical_name(r)
        r["canonical_name"] = c_name
        r["family"] = infer_family(c_name)
        key = (c_name, r["imgsz"], r.get("accuracy_protocol"), r.get("latency_protocol"))

        if key not in seen:
            seen[key] = r

    deduped = list(seen.values())

    # Sắp xếp: theo họ model, baseline trước, sau đó theo độ phân giải và FPS
    def sort_key(x):
        role_order = 0 if x["role"] == "baseline" else 1
        return (x["family"], role_order, -x["imgsz"], -x["serial_fps"])

    deduped.sort(key=sort_key)
    return deduped


SPEEDUP_REFERENCE_MODEL = "yolo26m-bdd100k.pt"
SPEEDUP_REFERENCE_IMGSZ = 640


def apply_reference_speedup(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Use YOLO26m FP16 at 640x640 as the single speedup reference.

    Accuracy drop remains relative to the baseline at the same model family and
    resolution, because a resolution change also changes the task difficulty.
    """
    reference = next((
        r for r in records
        if r["canonical_name"] == SPEEDUP_REFERENCE_MODEL
        and r["role"] == "baseline"
        and r["imgsz"] == SPEEDUP_REFERENCE_IMGSZ
    ), None)
    if reference is None or reference["lat_mean_ms"] <= 0:
        raise ValueError(
            "Không tìm thấy baseline yolo26m-bdd100k.pt tại 640x640 để tính speedup."
        )

    baselines = {
        (r["family"], r["imgsz"], r.get("accuracy_protocol")): r
        for r in records
        if r["role"] == "baseline"
    }
    for r in records:
        baseline = baselines.get((r["family"], r["imgsz"], r.get("accuracy_protocol")))
        if r["role"] == "baseline":
            r["map_drop_pp"] = 0.0 if r["map50_95_pct"] is not None else None
        elif baseline is None:
            r["map_drop_pp"] = None
        else:
            baseline_map = baseline.get("map50_95_raw_pct", baseline["map50_95_pct"])
            candidate_map = r.get("map50_95_raw_pct", r["map50_95_pct"])
            r["map_drop_pp"] = round(baseline_map - candidate_map, 2) if baseline_map is not None and candidate_map is not None else None
        r["speedup"] = (
            round(reference["lat_mean_ms"] / r["lat_mean_ms"], 2)
            if r["lat_mean_ms"] > 0 and reference.get("latency_verified") and r.get("latency_verified")
            and reference.get("latency_protocol") == r.get("latency_protocol") else None
        )
    return records


def percent(value):
    return f"{value:.2f}%" if value is not None else "N/A (chưa đo FP16)"


def export_markdown(headers: List[str], rows: List[List[str]], out_path: str, title: str = "Benchmark Comparison") -> None:
    align = [False] * len(headers)
    for i, h in enumerate(headers):
        if any(keyword in h.lower() for keyword in ["ms", "fps", "pct", "%", "pp", "size", "speedup", "map"]):
            align[i] = True

    lines = [f"# {title}\n", format_table(headers, rows, align_right=align)]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_live_latency_benchmark(
    model_paths: Optional[List[str]] = None,
    repeats: int = 30,
    warmup: int = 5,
    threads: int = 4,
    device: str = "cpu"
) -> List[Dict[str, Any]]:
    """Đo latency thực tế trực tiếp các models trên CPU."""
    import numpy as np
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)

    if not model_paths:
        model_paths = sorted(glob.glob("models/*"))
    
    results = []
    print(f"\n[INFO] Bắt đầu đo latency live ({len(model_paths)} models, threads={threads}, warmup={warmup}, repeats={repeats})...\n")

    for path_str in model_paths:
        path = Path(path_str)
        if not path.exists():
            continue
        # Bỏ qua metadata/bin lẻ trong openvino folder nếu quét nhầm
        if path.is_file() and path.suffix not in {".pt", ".onnx", ".xml"}:
            continue

        model_name = path.name
        imgsz = 480 if "480" in model_name else 640
        size_mib = get_dir_size_mib(path)

        print(f"--> Đang đo {model_name:<36} (imgsz={imgsz})...", end="", flush=True)
        try:
            m = YOLO(str(path), task="detect")
            dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

            m.predict(dummy, imgsz=imgsz, device=cpu_device(path), rect=False, verbose=False)
            configure_cpu_backend(m, path, threads)
            # Warmup
            for _ in range(warmup):
                m.predict(dummy, imgsz=imgsz, device=cpu_device(path), rect=False, verbose=False)
            torch.set_num_threads(threads)

            times_ms = []
            pre_ms, inf_ms, post_ms = [], [], []

            for _ in range(repeats):
                t0 = time.perf_counter()
                preds = m.predict(dummy, imgsz=imgsz, device=cpu_device(path), rect=False, verbose=False)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

                speed = preds[0].speed
                pre_ms.append(float(speed.get("preprocess", 0.0)))
                inf_ms.append(float(speed.get("inference", 0.0)))
                post_ms.append(float(speed.get("postprocess", 0.0)))

            t_arr = np.array(times_ms)
            lat_mean = float(t_arr.mean())
            lat_med = float(np.median(t_arr))
            lat_p95 = float(np.percentile(t_arr, 95))
            fps = 1000.0 / lat_mean if lat_mean > 0 else 0.0

            results.append({
                "model_name": model_name,
                "format": "OpenVINO" if path.is_dir() or path.suffix == ".xml" else ("PyTorch" if path.suffix == ".pt" else "ONNX"),
                "imgsz": imgsz,
                "size_mib": round(size_mib, 2),
                "lat_mean_ms": round(lat_mean, 2),
                "lat_median_ms": round(lat_med, 2),
                "lat_p95_ms": round(lat_p95, 2),
                "inf_ms": round(float(np.mean(inf_ms)), 2),
                "fps": round(fps, 1)
            })
            print(f" Xong! Latency: {lat_mean:.2f} ms | FPS: {fps:.1f}")
        except Exception as e:
            print(f" LỖI: {e}")

    # Sắp xếp theo FPS giảm dần (nhanh nhất đứng đầu)
    results.sort(key=lambda x: x["fps"], reverse=True)
    return results


def self_test():
    """Kiểm tra parser, logic so sánh và format bảng."""
    dummy_headers = ["Model", "mAP50-95", "Latency (ms)", "FPS"]
    dummy_rows = [["baseline", "33.06", "100.5", "9.95"], ["int8", "32.76", "45.2", "22.12"]]
    table_str = format_table(dummy_headers, dummy_rows)
    assert "baseline" in table_str, "Test table format failed"
    assert "22.12" in table_str, "Test table format failed"

    eval_dirs = find_eval_dirs()
    assert len(eval_dirs) > 0, "Không tìm thấy thư mục evaluation nào trong repo"
    records = parse_eval_dir(eval_dirs[0])
    assert len(records) > 0, "Không parse được dữ liệu evaluation"
    assert "map50_95_pct" in records[0], "Thiếu trường accuracy"

    all_test_recs = []
    for ed in eval_dirs:
        all_test_recs.extend(parse_eval_dir(ed))
    deduped = deduplicate_records(all_test_recs)
    assert 0 < len(deduped) <= len(all_test_recs), "Deduplication test failed"
    print("Self-test hoàn thành thành công!")


def main():
    parser = argparse.ArgumentParser(description="So sánh benchmark mAP và latency các YOLO26 models trên BDD100K.")
    parser.add_argument("--eval-dirs", nargs="*", default=None, help="Chỉ định các thư mục evaluation cần đọc (mặc định: tự quét)")
    parser.add_argument("--all-runs", action="store_true", help="Hiển thị tất cả các lần chạy (bao gồm cả model trùng lặp)")
    parser.add_argument("--latency", "--benchmark", action="store_true", help="Chạy đo latency trực tiếp trên các model trong models/")
    parser.add_argument("--models", nargs="*", default=None, help="Danh sách model cụ thể khi chạy live latency")
    parser.add_argument("--repeats", type=int, default=30, help="Số lần lặp đo latency (mặc định: 30)")
    parser.add_argument("--warmup", type=int, default=5, help="Số lần warmup (mặc định: 5)")
    parser.add_argument("--threads", type=int, default=4, help="Số luồng CPU cho benchmark live (mặc định: 4)")
    parser.add_argument("--csv", type=str, default=None, help="Đường dẫn file CSV xuất ra")
    parser.add_argument("--markdown", type=str, default=None, help="Đường dẫn file Markdown xuất ra")
    parser.add_argument("--include-480", action="store_true", help="Bao gồm các mô hình độ phân giải 480x480 (mặc định: bỏ)")
    parser.add_argument("--test", action="store_true", help="Chạy test tự kiểm tra script")

    args = parser.parse_args()

    if args.test:
        self_test()
        return

    # 1. So sánh các kết quả benchmark đã chạy
    eval_dirs = find_eval_dirs(args.eval_dirs)
    all_records = []
    for ed in eval_dirs:
        recs = parse_eval_dir(ed)
        all_records.extend(recs)

    missing_artifacts = [r for r in all_records if not r["artifact_exists"]]
    if missing_artifacts:
        print(f"[WARN] Bỏ qua {len(missing_artifacts)} kết quả vì artifact gốc không còn trong workspace.")
    all_records = [r for r in all_records if r["artifact_exists"]]
    if not args.include_480:
        all_records = [r for r in all_records if r["imgsz"] != 480]

    if all_records:
        if args.all_runs:
            references = [
                r for r in all_records
                if r["model_name"] == SPEEDUP_REFERENCE_MODEL
                and r["role"] == "baseline"
                and r["imgsz"] == SPEEDUP_REFERENCE_IMGSZ
            ]
            if not references:
                raise ValueError("Không tìm thấy baseline YOLO26m 640 để tính speedup.")
            for r in all_records:
                r["canonical_name"] = r["model_name"]
                r["family"] = infer_family(r["model_name"])
            apply_reference_speedup(all_records)
            title = "BẢNG SO SÁNH BENCHMARK TẤT CẢ CÁC LẦN CHẠY (BAO GỒM MODEL TRÙNG LẶP)"
            headers = [
                "Experiment", "Role", "Model", "Size(MiB)", "mAP50-95", "mAP50", "Drop(pp)", "Mean(ms)", "p95(ms)", "FPS", "Speedup vs YOLO26m 640", "Measurement"
            ]
            align_right = [False, False, False, True, True, True, True, True, True, True, True]
            rows = [
                [
                    r["experiment"],
                    r["role"],
                    r["model_name"],
                    f"{r['size_mib']:.1f}",
                    percent(r['map50_95_pct']),
                    percent(r['map50_pct']),
                    f"{r['map_drop_pp']:+.2f}" if r["map_drop_pp"] is not None else "N/A",
                    f"{r['lat_mean_ms']:.1f}",
                    f"{r['lat_p95_ms']:.1f}",
                    f"{r['serial_fps']:.1f}",
                    f"{r['speedup']:.2f}x" if r["speedup"] is not None else "N/A",
                    r["measurement"]
                ]
                for r in all_records
            ]
        else:
            deduped_records = apply_reference_speedup(deduplicate_records(all_records))
            title = "BẢNG SO SÁNH BENCHMARK CÁC MODEL CÒN ARTIFACT CÓ THỂ TÁI LẬP"
            headers = [
                "Family", "Role", "Model", "Imgsz", "Size(MiB)", "mAP50-95", "Drop(pp)", "mAP50", "Precision", "Recall", "F1", "Latency(ms)", "FPS", "Speedup vs YOLO26m 640", "Measurement"
            ]
            align_right = [False, False, False, False, True, True, True, True, True, True, True, True, True, True]
            rows = [
                [
                    r.get("family", ""),
                    r["role"],
                    r.get("canonical_name", r["model_name"]),
                    f"{r['imgsz']}x{r['imgsz']}",
                    f"{r['size_mib']:.1f}",
                    percent(r['map50_95_pct']),
                    f"{r['map_drop_pp']:+.2f}" if r["map_drop_pp"] is not None else "N/A",
                    percent(r['map50_pct']),
                    percent(r['precision_pct']),
                    percent(r['recall_pct']),
                    percent(r['f1_pct']),
                    f"{r['lat_mean_ms']:.1f}",
                    f"{r['serial_fps']:.1f}",
                    f"{r['speedup']:.2f}x" if r["speedup"] is not None else "N/A",
                    r["measurement"]
                ]
                for r in deduped_records
            ]

        align_right.append(False)
        print("\n" + "=" * 115)
        print(f" {title}")
        print("=" * 115)
        print(format_table(headers, rows, align_right=align_right))
        print("=" * 115 + "\n")

        if args.csv:
            export_csv(headers, rows, args.csv)
            print(f"[EXPORT] Đã lưu bảng benchmark vào CSV: {args.csv}")
        if args.markdown:
            export_markdown(headers, rows, args.markdown, title=title)
            print(f"[EXPORT] Đã lưu bảng benchmark vào Markdown: {args.markdown}")
    else:
        print("[WARN] Không tìm thấy dữ liệu benchmark nào từ các thư mục evaluation.")

    # 2. Đo latency live nếu có cờ --latency
    if args.latency:
        live_records = run_live_latency_benchmark(
            model_paths=args.models,
            repeats=args.repeats,
            warmup=args.warmup,
            threads=args.threads
        )
        if live_records:
            print("\n" + "=" * 95)
            print(f" KẾT QUẢ ĐO LATENCY THỰC TẾ (Batch=1, CPU Threads={args.threads}, Repeats={args.repeats})")
            print("=" * 95)
            lat_headers = ["Model", "Format", "Imgsz", "Size(MiB)", "Mean(ms)", "Median(ms)", "p95(ms)", "Inf(ms)", "FPS"]
            lat_align = [False, False, False, True, True, True, True, True, True]
            lat_rows = [
                [
                    r["model_name"][:32],
                    r["format"],
                    f"{r['imgsz']}x{r['imgsz']}",
                    f"{r['size_mib']:.1f}",
                    f"{r['lat_mean_ms']:.1f}",
                    f"{r['lat_median_ms']:.1f}",
                    f"{r['lat_p95_ms']:.1f}",
                    f"{r['inf_ms']:.1f}",
                    f"{r['fps']:.1f}"
                ]
                for r in live_records
            ]
            print(format_table(lat_headers, lat_rows, align_right=lat_align))
            print("=" * 95 + "\n")

            if args.csv:
                lat_csv_path = args.csv.replace(".csv", "_live_latency.csv") if ".csv" in args.csv else f"{args.csv}_live_latency.csv"
                export_csv(lat_headers, lat_rows, lat_csv_path)
                print(f"[EXPORT] Đã lưu bảng latency live vào CSV: {lat_csv_path}")


if __name__ == "__main__":
    main()
