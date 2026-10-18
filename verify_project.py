#!/usr/bin/env python3
"""Record local model/data evidence without inventing training provenance."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys

import yaml

from evaluate import artifact_info, canonical, DEFAULT_NAMES, read_frames, sha256, write_json
import json


def check_rerun(directory, original_report=None):
    report_path = directory / "report.json"
    if not report_path.exists():
        return {"status": "not_completed", "path": str(directory)}
    report = json.loads(report_path.read_text())
    manifest = json.loads((directory / "dataset_manifest.json").read_text())
    if sha256(directory / "dataset_manifest.json") != report['protocol']['manifest_sha256']:
        raise ValueError(f"Changed evaluation manifest: {directory}")
    if not manifest or any(not image.get('image_sha256') for image in manifest):
        raise ValueError(f"Rerun is missing image content hashes: {directory}")
    if not report['models']:
        raise ValueError(f"Rerun has no completed models: {directory}")
    for model in report['models']:
        if artifact_info(model['artifact']['path'])['files'] != model['artifact']['files']:
            raise ValueError(f"Changed model since evaluation: {model['label']}")
        for phase in (model['runtime'], model['runtime'].get('validation', {})):
            if phase.get('torch_threads') != 4 or phase.get('device') != 'CPU':
                raise ValueError(f"Rerun did not verify CPU 4-thread runtime: {directory}")
            if 'ov_threads' in phase and (phase['ov_threads'] != 4 or phase.get('ov_streams') != 1):
                raise ValueError(f"Rerun did not verify OpenVINO single-stream runtime: {directory}")
    result = {"status": "completed_in_existing_environment", "report": str(report_path),
              "report_sha256": sha256(report_path), "images": len(manifest),
              "models": [m['label'] for m in report['models']],
              "artifacts": [Path(m['artifact']['path']).name for m in report['models']],
              "independent_clean_environment": False}
    if original_report:
        original = json.loads(original_report.read_text())
        old_manifest = json.loads((original_report.parent / "dataset_manifest.json").read_text())
        without_content_hash = [{k: v for k, v in image.items() if k != 'image_sha256'} for image in manifest]
        if without_content_hash != old_manifest or report['protocol']['source_labels_sha256'] != original['protocol']['source_labels_sha256']:
            raise ValueError("Rerun uses different images or annotations from the historical report")
        measured = report['models'][0]
        old = next(m for m in original['models'] if m['artifact']['files'] == measured['artifact']['files'])
        delta = (measured['accuracy']['map50_95'] - old['accuracy']['map50_95']) * 100
        result.update(original_report=str(original_report), map50_95=measured['accuracy']['map50_95'] * 100,
                      difference_pp=delta, agrees_within_0_05_pp=abs(delta) <= 0.05,
                      historical_image_contents_verified=False)
    return result


def inspect_artifact(path):
    path = Path(path)
    result = {"artifact": artifact_info(path)}
    if path.suffix == ".pt":
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = checkpoint.get("ema") or checkpoint["model"]
        names = model.names
        result.update(
            optimizer_present=checkpoint.get("optimizer") is not None,
            scale=getattr(model, "yaml", {}).get("scale"),
            head="nms-free" if getattr(model, "end2end", False) else "nms",
            tensors=[{"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape),
                      "elements": tensor.numel(), "bytes": tensor.numel() * tensor.element_size()}
                     for name, tensor in model.state_dict().items()])
        counts = Counter()
        for tensor in result["tensors"]:
            counts[tensor["dtype"]] += tensor["elements"]
        result["stored_tensor_elements_by_dtype"] = dict(counts)
        result["runtime_note"] = "Current demo/evaluator explicitly use PyTorch CPU FP16; stored dtype is measured before any runtime cast."
    else:
        import openvino as ov
        xmls = sorted(path.glob("*.xml")) if path.is_dir() else [path]
        if len(xmls) != 1:
            raise ValueError(f"Expected one OpenVINO XML: {path}")
        graph = ov.Core().read_model(xmls[0])
        metadata = yaml.safe_load((xmls[0].parent / "metadata.yaml").read_text())
        names = metadata["names"]
        constants = Counter()
        for node in graph.get_ops():
            if node.get_type_name() == "Constant":
                constants[node.get_element_type().get_type_name()] += node.get_byte_size()
        result.update(metadata=metadata, graph_ops=dict(Counter(n.get_type_name() for n in graph.get_ops())),
                      constant_bytes_by_dtype=dict(constants),
                      inputs=[str(p.get_partial_shape()) for p in graph.inputs],
                      outputs=[str(p.get_partial_shape()) for p in graph.outputs],
                      precision_note="Constant dtypes and FakeQuantize nodes describe the IR; not all operations necessarily execute in INT8.")
        # A fresh interpreter loads only the IR, OpenVINO and NumPy, with no PyTorch checkpoint.
        probe = subprocess.run([sys.executable, "-I", "-c", """
import json, sys, time
import numpy as np
import openvino as ov
started = time.perf_counter()
core = ov.Core()
compiled = core.compile_model(core.read_model(sys.argv[1]), 'CPU',
    {'PERFORMANCE_HINT':'LATENCY', 'INFERENCE_NUM_THREADS':4, 'NUM_STREAMS':1})
ready = time.perf_counter()
outputs = compiled(np.zeros(tuple(compiled.input().shape), dtype=np.float32))
finished = time.perf_counter()
if 'torch' in sys.modules or not outputs or not all(np.isfinite(value).all() for value in outputs.values()):
    raise RuntimeError('Standalone OpenVINO probe failed')
threads = compiled.get_property('INFERENCE_NUM_THREADS')
streams = int(compiled.get_property('NUM_STREAMS'))
devices = compiled.get_property('EXECUTION_DEVICES')
if threads != 4 or streams != 1 or devices != ['CPU']:
    raise RuntimeError('Standalone probe has a different CPU protocol')
print(json.dumps({'torch_imported':False, 'ov_threads':threads, 'ov_streams':streams,
    'execution_devices':devices, 'output_shapes':[list(value.shape) for value in outputs.values()],
    'read_compile_ms':(ready-started)*1000, 'first_inference_ms':(finished-ready)*1000,
    'load_and_first_inference_ms':(finished-started)*1000,
    'timing_note':'Fresh process; filesystem cache and shared-host load uncontrolled; excludes library imports.'}))
""", str(xmls[0].resolve())], check=True, capture_output=True, text=True)
        result["standalone_openvino"] = json.loads(probe.stdout)
    names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    result["class_names"] = names
    result["bdd100k_compatible"] = [canonical(n) for n in names] == DEFAULT_NAMES
    result["training_provenance_verified"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="verification", help="Generated evidence directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    from run_demo import MODEL_CONFIGS

    models = {}
    for key, config in MODEL_CONFIGS.items():
        record = inspect_artifact(config["path"])
        if not record["bdd100k_compatible"]:
            raise ValueError(f"Demo model has an incompatible class map: {key}")
        models[key] = record
        print(f"Verified {key}: {record['artifact']['size_mib']:.3f} MiB", flush=True)
    write_json(output / "model_inventory.json", {"created_at": datetime.now(timezone.utc), "models": models,
        "excluded": {"models/yolo26m.pt": "COCO 80 classes; excluded from the BDD100K demo"},
        "provenance_limits": ["Original model cards/revisions and training logs are unavailable locally.",
                              "Calibration manifests and QAT training logs are unavailable locally."]})

    labels = root / "data/bdd100k/labels/bdd100k_labels_images_val.json"
    frames = read_frames(labels)
    by_name = {frame["name"]: frame for frame in frames}
    if len(by_name) != len(frames):
        raise ValueError("Duplicate annotation names")
    from PIL import Image
    images = sorted(p for p in (root / "data/bdd100k/images/val").iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    entries, conditions, content_hashes, classes, content_groups = [], Counter(), Counter(), Counter(), {}
    for path in images:
        frame = by_name[path.name]
        if not isinstance(frame.get("labels"), list):
            raise ValueError(f"Missing annotation labels: {path.name}")
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        digest = sha256(path)
        content_hashes[digest] += 1
        content_groups.setdefault(digest, []).append(path.name)
        classes.update(canonical(obj.get('category', '')) for obj in frame['labels'] if obj.get('box2d') is not None)
        attrs = frame.get("attributes") or {}
        conditions[str((attrs.get("timeofday"), attrs.get("weather"), attrs.get("scene")))] += 1
        entries.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest,
                        "width": width, "height": height})
    write_json(output / "image_manifest.json", entries)
    write_json(output / "data_inventory.json", {
        "images": len(entries), "annotation_frames": len(frames), "labels_sha256": sha256(labels),
        "manifest_sha256": sha256(output / "image_manifest.json"),
        "duplicate_image_content_groups": sum(count > 1 for count in content_hashes.values()),
        "duplicate_images": [{"sha256": digest, "names": names} for digest, names in content_groups.items() if len(names) > 1],
        "objects_per_class": dict(classes),
        "annotations_without_image": sorted(set(by_name) - {p.name for p in images}),
        "conditions": dict(conditions), "split": "Existing validation; previously used for evaluation",
        "held_out_final_verified": False, "calibration_disjointness_verified": False})

    packages = {}
    for line in (root / "requirements.txt").read_text().splitlines():
        if "==" not in line:
            continue
        name, expected = line.strip().split("==", 1)
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"Environment mismatch: {name}: {actual} != {expected}")
        packages[name] = actual
    pip_check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=True)
    source_paths = sorted(root.glob("*.py")) + sorted(root.glob("test_*.js")) + [
        root / name for name in ("requirements.txt", "demo.html", "model-compression-yolo26-dashboard.html",
                                "demo_data/detections.json", "demo_data/demo_bundle.js",
                                "demo_data/latency_benchmark.json")]
    source_paths += [root / name for name in (
        'requirements-hardware.txt', 'demo_data/hardware_benchmark.js',
        'demo_data/hardware_benchmark.css', 'demo_data/browser_models.json', 'demo_data/vendor/onnxruntime-web/vendor.json',
        'runtime_exports/manifest.json') if (root / name).is_file()]
    sources = {str(p.relative_to(root)): sha256(p) for p in source_paths}
    write_json(output / "environment.lock.json", {"python": sys.version, "platform": platform.platform(),
        "logical_cpus": os.cpu_count(), "packages": packages, "pip_check": pip_check.stdout.strip(),
        "source_sha256": sources, "clean_environment_reinstall_verified": False})
    protocol = {"version": 1, "status": "Observed protocol; acceptance thresholds not approved by the team",
        "created_at": datetime.now(timezone.utc).isoformat(), "imgsz": [640, 640], "rect": False,
        "batch": 1, "device": "CPU", "threads": 4, "openvino_streams": 1, "head": "nms-free",
        "conf": 0.001, "iou": 0.7, "max_det": 300, "ignore_policy": "include", "seed": 42,
        "metric": "Ultralytics detection mAP; not official BDD ignore/crowd evaluation",
        "class_names": DEFAULT_NAMES, "labels_sha256": sha256(labels),
        "image_manifest_sha256": sha256(output / "image_manifest.json"),
        "model_inventory_sha256": sha256(output / "model_inventory.json"),
        "environment_lock_sha256": sha256(output / "environment.lock.json"),
        "demo_conf": 0.10, "latency": {"images": 50, "warmup": 5, "repeats": 3,
            "scope": "predict wall; decode/setup/warmup excluded", "exclusive_host": False},
        "baseline_storage": "FP16 checkpoint; current CPU inference explicitly uses FP16. Historical FP32 reports remain separate.",
        "limits": ["No independent development/final split or calibration manifest verified.",
                   "A rerun in the existing environment is not independent clean-environment reproduction."]}
    (output / "protocol.yaml").write_text(yaml.safe_dump(protocol, allow_unicode=True, sort_keys=False))
    write_json(output / "reproduction.json", {
        "smoke": check_rerun(output / "smoke_ptq_qat"),
        "interrupted_full_ptq": check_rerun(output / "full_ptq"),
        "full_ptq": check_rerun(output / "full_ptq_retry", root / "evaluation_int8_yolo26m_640/report.json")})
    print(f"Evidence written to {output}; {len(entries)} images verified.")


if __name__ == "__main__":
    main()
