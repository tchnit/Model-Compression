#!/usr/bin/env bash
# Six models, sequentially; each family keeps its own accuracy baseline.
set -euo pipefail
cd -- "$(dirname -- "$0")"
benchmark_dir="verification/rebenchmark_fp16_$(date +%Y%m%d_%H%M%S)"
for family in m s; do
  .venv/bin/python evaluate.py \
    --images data/bdd100k/images/val \
    --labels data/bdd100k/labels/bdd100k_labels_images_val.json \
    --baseline "models/yolo26${family}-bdd100k.pt" \
    --candidate "PTQ_INT8=models/int8_yolo26${family}_640_openvino_model" \
    --candidate "QAT_INT8=models/int8_yolo26${family}_qat_640_openvino_model" \
    --precision FP16 --imgsz 640 --threads 4 --head nms-free \
    --conf 0.001 --iou 0.7 --max-det 300 --limit 0 \
    --latency-images 50 --warmup 5 --repeats 3 --seed 42 \
    --ignore-policy include --plots --output "${benchmark_dir}/yolo26${family}"
done
.venv/bin/python refresh_demo_data.py \
  --eval-dirs "${benchmark_dir}/yolo26m" "${benchmark_dir}/yolo26s"
.venv/bin/python compare_benchmarks.py \
  --eval-dirs "${benchmark_dir}/yolo26m" "${benchmark_dir}/yolo26s" \
  --csv benchmark_comparison.csv --markdown benchmark_comparison.md
