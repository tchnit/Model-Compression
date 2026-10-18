#!/usr/bin/env python3
"""Explicit runtime exports; INT8 calibration is opt-in and excludes validation images."""
import argparse
import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path

from bdd100k_eval.config import ROOT
from bdd100k_eval.artifacts import sha256 as digest
OUT = ROOT / 'runtime_exports'


def write_browser_manifest(manifest):
    """Publish only ONNX provenance so browser inference needs no native API."""
    artifacts = {}
    for name in ['yolo26m', 'yolo26s']:
        for precision in ['FP32', 'FP16', 'INT8']:
            entry = manifest.get(f'{name}_{precision}_onnx')
            if entry and (ROOT / entry['path']).is_file():
                artifacts[f'{name}_{precision}'] = {
                    'artifact': entry['path'], 'url': entry['path'],
                    'sha256': entry['sha256'], 'source_sha256': entry['source_sha256'],
                    'input_dtype': 'float16' if precision == 'FP16' else 'float32',
                    'precision_policy': entry.get('precision_policy', f'{precision} graph; compute precision depends on runtime kernels'),
                    **{k:entry[k] for k in ['calibration_sha256','calibration_images','quantized_convs'] if k in entry}}
    path = ROOT / 'demo_data' / 'browser_models.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(schema_version=1, models={
        'yolo26m': 'YOLO26m · BDD100K', 'yolo26s': 'YOLO26s · BDD100K'},
        artifacts=artifacts), ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def select_calibration_images(list_path, count, validation_manifest):
    if type(count) is not int or not 1 <= count <= 1000:
        raise ValueError('Calibration count must be 1..1000')
    list_path = Path(list_path).resolve()
    paths = list(dict.fromkeys(line.strip() for line in list_path.read_text().splitlines() if line.strip()))
    if len(paths) < count:
        raise ValueError(f'Need {count} independent training images, found {len(paths)}')
    validation = json.loads(Path(validation_manifest).read_text())
    val_names = {item['name'] for item in validation}
    val_hashes = {item['sha256'] for item in validation}
    images, hashes = [], set()
    for value in random.Random(42).sample(paths, count):
        path = Path(value)
        path = (list_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if 'train' not in path.parts:
            raise ValueError(f'Calibration must use the train split: {path}')
        content_hash = digest(path)
        if path.name in val_names or content_hash in val_hashes:
            raise ValueError(f'Calibration overlaps validation: {path.name}')
        if content_hash in hashes:
            raise ValueError(f'Duplicate calibration image content: {path.name}')
        hashes.add(content_hash)
        images.append(dict(path=str(path), sha256=content_hash))
    return images


def prepare(models, engines, calibration_list=None, calibration_count=128):
    import torch
    import onnx
    from ultralytics import YOLO
    OUT.mkdir(exist_ok=True)
    manifest_path = OUT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    def record(key, path, source, **extra):
        manifest[key] = dict(path=str(path.relative_to(ROOT)), sha256=digest(path),
                             source_sha256=digest(source), shape=[1, 3, 640, 640], **extra)
        temp = manifest_path.with_suffix('.tmp')
        temp.write_text(json.dumps(manifest, indent=2) + '\n')
        temp.replace(manifest_path)
        if key.endswith('_onnx'):
            write_browser_manifest(manifest)

    trt_logger = None
    for name in models:
        source = ROOT / 'models' / f'{name}-bdd100k.pt'
        for precision in ['FP32', 'FP16']:
            if precision == 'FP16' and not torch.cuda.is_available():
                print(f'Skip {name} FP16: this exporter requires CUDA; FP32 remains available.', flush=True)
                continue
            onnx_path = OUT / f'{name}_{precision.lower()}.onnx'
            key = f'{name}_{precision}_onnx'
            entry = manifest.get(key, {})
            if not (onnx_path.exists() and entry.get('sha256') == digest(onnx_path)
                    and entry.get('source_sha256') == digest(source)):
                with tempfile.TemporaryDirectory(dir=OUT) as work:
                    copied = Path(work) / source.name
                    shutil.copy2(source, copied)
                    exported = YOLO(copied).export(format='onnx', imgsz=640, batch=1,
                        dynamic=False, simplify=False, opset=17, nms=False,
                        quantize=16 if precision == 'FP16' else 32,
                        device='0' if precision == 'FP16' else 'cpu')
                    shutil.move(exported, onnx_path)
                graph = onnx.load(onnx_path)
                expected = onnx.TensorProto.FLOAT16 if precision == 'FP16' else onnx.TensorProto.FLOAT
                assert graph.graph.input[0].type.tensor_type.elem_type == expected
                onnx.checker.check_model(graph)
                record(key, onnx_path, source, precision=precision, onnx_version=onnx.__version__)

            if 'openvino' in engines and precision == 'FP32':
                import openvino as ov
                xml = OUT / f'{name}_fp32.xml'
                ov.save_model(ov.convert_model(onnx_path), xml, compress_to_fp16=False)
                record(f'{name}_FP32_openvino', xml, source, precision='FP32',
                       bin_sha256=digest(xml.with_suffix('.bin')), openvino_version=ov.__version__)

            if 'tensorrt' in engines and torch.cuda.is_available():
                import tensorrt as trt
                engine_path = OUT / f'{name}_{precision.lower()}.engine'
                trt_key = f'{name}_{precision}_tensorrt'
                gpu = torch.cuda.get_device_name(0)
                prior = manifest.get(trt_key, {})
                if (engine_path.exists() and prior.get('sha256') == digest(engine_path)
                        and prior.get('source_sha256') == digest(source)
                        and prior.get('gpu') == gpu and prior.get('tensorrt_version') == trt.__version__):
                    continue
                if trt_logger is None:
                    trt_logger = trt.Logger(trt.Logger.WARNING)
                logger = trt_logger
                builder = trt.Builder(logger)
                network = builder.create_network(0)
                parser = trt.OnnxParser(network, logger)
                if not parser.parse(onnx_path.read_bytes()):
                    raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
                config = builder.create_builder_config()
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
                config.clear_flag(trt.BuilderFlag.TF32)
                if precision == 'FP16':
                    config.set_flag(trt.BuilderFlag.FP16)
                engine = builder.build_serialized_network(network, config)
                if engine is None:
                    raise RuntimeError('TensorRT engine build failed')
                engine_path.write_bytes(bytes(engine))
                record(trt_key, engine_path, source, precision=precision, gpu=gpu,
                       compute_capability=list(torch.cuda.get_device_capability(0)),
                       tensorrt_version=trt.__version__, tf32=False,
                       precision_policy='FP16 permitted; unsupported layers FP32' if precision == 'FP16' else 'FP32; TF32 disabled')
                print(f'Ready: {engine_path}', flush=True)
    if calibration_list:
        import cv2
        import numpy as np
        import onnxruntime as ort
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
        from bdd100k_eval.image_ops import preprocess
        images = select_calibration_images(calibration_list, calibration_count,
                                          ROOT / 'verification/image_manifest.json')
        evidence = ROOT / 'verification/browser_int8_calibration.json'
        evidence.write_text(json.dumps(dict(seed=42, split='train', images=images,
            validation_manifest_sha256=digest(ROOT / 'verification/image_manifest.json'),
            validation_overlap_by_name_or_content=False), indent=2) + '\n')

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.images = iter(images)

            def get_next(self):
                row = next(self.images, None)
                if row is None:
                    return None
                image = cv2.imread(row['path'])
                if image is None:
                    raise ValueError(f'Invalid calibration image: {row["path"]}')
                return {'images': preprocess(image)[0]}

        for name in models:
            source = ROOT / 'models' / f'{name}-bdd100k.pt'
            fp32 = OUT / f'{name}_fp32.onnx'
            output = OUT / f'{name}_int8.onnx'
            # Keep the decode/scores floating point; calibrate only learned convolution layers.
            with tempfile.TemporaryDirectory(dir=OUT) as work:
                staged = Path(work) / output.name
                quantize_static(fp32, staged, Reader(), quant_format=QuantFormat.QDQ,
                    op_types_to_quantize=['Conv'], activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True,
                    calibration_providers=['CPUExecutionProvider'])
                graph = onnx.load(staged)
                onnx.checker.check_model(graph)
                assert any(t.data_type == onnx.TensorProto.INT8 for t in graph.graph.initializer)
                producers = {value:node for node in graph.graph.node for value in node.output}
                weights = {tensor.name:tensor.data_type for tensor in graph.graph.initializer}
                quantized_convs = 0
                for node in graph.graph.node:
                    if node.op_type != 'Conv':
                        continue
                    weight = producers.get(node.input[1])
                    if weight is not None and weight.op_type == 'DequantizeLinear' and weights.get(weight.input[0]) == onnx.TensorProto.INT8:
                        quantized_convs += 1
                assert quantized_convs > 0, 'No Conv with INT8 weights'
                onnx.helper.set_model_props(graph, {**{p.key:p.value for p in graph.metadata_props},
                    'graph_precision':'INT8', 'quantization':'PTQ_QDQ',
                    'calibration_sha256':digest(evidence), 'calibration_images':str(len(images))})
                onnx.save(graph, staged)
                options = ort.SessionOptions(); options.intra_op_num_threads = 4
                session = ort.InferenceSession(str(staged), sess_options=options, providers=['CPUExecutionProvider'])
                result = session.run(None, Reader().get_next())[0]
                if result.shape != (1,300,6) or not np.isfinite(result).all():
                    raise RuntimeError('INT8 validation failed: shape or nonfinite output')
                del session
                shutil.move(staged, output)
            record(f'{name}_INT8_onnx', output, source, precision='INT8', onnx_version=onnx.__version__,
                onnxruntime_version=ort.__version__, calibration_images=len(images),
                calibration_sha256=digest(evidence),
                quantized_convs=quantized_convs,
                precision_policy='PTQ QDQ: uint8 activations / int8 per-channel Conv weights; other operations floating point')
            print(f'Ready INT8: {output}', flush=True)
    write_browser_manifest(manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=['yolo26m', 'yolo26s'], default=['yolo26s', 'yolo26m'])
    parser.add_argument('--engines', nargs='+', choices=['onnx', 'openvino', 'tensorrt'], default=['onnx', 'openvino', 'tensorrt'])
    parser.add_argument('--int8-calibration-list', type=Path, help='Explicit list of train images; checked against validation names and hashes')
    parser.add_argument('--calibration-count', type=int, default=128)
    args = parser.parse_args()
    prepare(args.models, args.engines, args.int8_calibration_list, args.calibration_count)
