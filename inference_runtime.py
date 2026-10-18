"""CPU settings shared by the demo and reproducible latency measurements."""
from functools import partial
from pathlib import Path


def cpu_device(path):
    path = Path(path)
    return "intel:cpu" if path.is_dir() or path.suffix == ".xml" else "cpu"


def configure_cpu_backend(model, path, threads, precision="FP16", *, strict_threads=True):
    """Configure CPU inference; benchmarks require exact threads, live demos may accept fewer."""
    if precision not in {"FP16", "FP32"}:
        raise ValueError("Precision PyTorch phải là FP16 hoặc FP32")
    import torch
    torch.set_num_threads(threads)
    backend = model.predictor.model.backend if hasattr(model, "predictor") else model.backend
    runtime = {"device": "CPU", "torch_threads": torch.get_num_threads()}
    if Path(path).suffix.lower() == ".pt":
        # AutoBackend disables FP16 on CPU even when quantize=16 was requested.
        network = backend.model
        network.half() if precision == "FP16" else network.float()
        backend.fp16 = precision == "FP16"
        dtype = next(network.parameters()).dtype
        if dtype != (torch.float16 if backend.fp16 else torch.float32):
            raise RuntimeError(f"PyTorch không áp dụng {precision}")
        runtime.update(precision=precision, parameter_dtype=str(dtype), input_dtype=str(dtype))
    if hasattr(backend, "ov_compiled_model"):
        import openvino as ov
        compiled = backend.ov_compiled_model
        if (compiled.get_property("INFERENCE_NUM_THREADS") != threads
                or compiled.get_property("EXECUTION_DEVICES") != ["CPU"]
                or int(compiled.get_property("NUM_STREAMS")) != 1):
            core = ov.Core()
            xml = Path(path)
            if xml.is_dir():
                xml = next(xml.glob("*.xml"))
            graph = core.read_model(xml)
            graph.reshape(list(compiled.input().shape))
            # OpenVINO uses its own scheduler; torch/OMP settings do not pin it.
            backend.compile_model = partial(core.compile_model, device_name="CPU", config={
                "PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": threads,
                "NUM_STREAMS": 1,
            })
            backend.ov_compiled_model = backend.compile_model(graph)
        compiled = backend.ov_compiled_model
        runtime.update(requested_threads=threads,
                       ov_threads=int(compiled.get_property("INFERENCE_NUM_THREADS")),
                       execution_devices=compiled.get_property("EXECUTION_DEVICES"),
                       ov_streams=int(compiled.get_property("NUM_STREAMS")))
        runtime['thread_protocol_matches'] = runtime['ov_threads'] == threads
        if (runtime["execution_devices"] != ["CPU"] or runtime["ov_streams"] != 1
                or not 1 <= runtime["ov_threads"] <= threads):
            raise RuntimeError(f"Cấu hình OpenVINO không hợp lệ: {runtime}")
        if not runtime['thread_protocol_matches']:
            message = (f"Yêu cầu {threads} luồng; OpenVINO thực tế {runtime['ov_threads']} luồng, "
                       f"{runtime['ov_streams']} stream, thiết bị {runtime['execution_devices']}. "
                       'CPU khả dụng hoặc chính sách lập lịch có thể giới hạn số luồng. '
                       'Kết quả này không cùng điều kiện với benchmark yêu cầu đủ số luồng.')
            if strict_threads:
                raise RuntimeError(message)
            runtime['thread_warning'] = message
    return runtime
