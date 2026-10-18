import csv
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from compare_benchmarks import apply_reference_speedup, deduplicate_records, infer_family, parse_eval_dir


def record(name, role, imgsz, map50_95, latency):
    return {
        "experiment": "fixture",
        "latency_verified": True,
        "latency_protocol": "fixture",
        "role": role,
        "model_name": name,
        "imgsz": imgsz,
        "size_mib": 1,
        "map50_95_pct": map50_95,
        "map50_pct": map50_95,
        "precision_pct": 0,
        "recall_pct": 0,
        "f1_pct": 0,
        "map_drop_pp": 0,
        "lat_mean_ms": latency,
        "lat_p95_ms": latency,
        "serial_fps": 1000 / latency,
        "speedup": 1,
    }


class BenchmarkAggregationTests(unittest.TestCase):
    def test_640_snapshot_is_not_reused_for_480_or_mismatched_threads(self):
        from evaluate import artifact_info
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'yolo26m-bdd100k.pt'
            model.write_bytes(b'fixture')
            artifact = artifact_info(model)
            run = root / 'run'
            run.mkdir()
            report = {'protocol': {'args': {'imgsz': 480, 'precision': 'FP16'}}, 'models': [
                {'label': 'baseline', 'artifact': artifact, 'accuracy': {'map50_95': .3}}]}
            (run / 'report.json').write_text(json.dumps(report))
            (root / 'demo_data').mkdir()
            snapshot = {'protocol': {'imgsz': 640, 'rect': False, 'threads': 4, 'device': 'CPU',
                                    'images': [], 'repeats': 3}, 'models': {model.name: {
                'artifact': artifact, 'runtime': {'torch_threads': 4, 'device': 'CPU', 'precision': 'FP16', 'parameter_dtype': 'torch.float16'},
                'latency': {'mean': 10, 'p95': 12}}}}
            path = root / 'demo_data/latency_benchmark.json'
            path.write_text(json.dumps(snapshot))
            with patch('compare_benchmarks.ROOT', root):
                self.assertFalse(parse_eval_dir(str(run))[0]['latency_verified'])
                report['protocol']['args']['imgsz'] = 640
                (run / 'report.json').write_text(json.dumps(report))
                self.assertTrue(parse_eval_dir(str(run))[0]['latency_verified'])
                self.assertTrue(parse_eval_dir(str(run))[0]['accuracy_pending'])
                report['models'][0]['runtime'] = {'validation': {'parameter_dtype': 'torch.float16'}}
                (run / 'report.json').write_text(json.dumps(report))
                self.assertEqual(parse_eval_dir(str(run))[0]['map50_95_pct'], 30)
                snapshot['models'][model.name]['runtime']['precision'] = 'FP32'
                path.write_text(json.dumps(snapshot))
                self.assertFalse(parse_eval_dir(str(run))[0]['latency_verified'])
                snapshot['models'][model.name]['runtime']['precision'] = 'FP16'
                report['protocol']['args'].pop('precision')
                (run / 'report.json').write_text(json.dumps(report))
                self.assertIsNone(parse_eval_dir(str(run))[0]['map50_95_pct'])
                snapshot['models'][model.name]['runtime']['torch_threads'] = 8
                path.write_text(json.dumps(snapshot))
                self.assertFalse(parse_eval_dir(str(run))[0]['latency_verified'])

    def test_export_includes_yolo26s_640_without_duplicate_yolo26m(self):
        project = Path(__file__).resolve().parent
        with TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark.csv"
            subprocess.run(
                [sys.executable, "compare_benchmarks.py", "--csv", str(output)],
                cwd=project,
                check=True,
                capture_output=True,
                text=True,
            )
            with output.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(r["Model"] == "int8_yolo26m_640_openvino_model" for r in rows), 1)
        ptq_s = next(r for r in rows if r["Model"] == "int8_yolo26s_640_openvino_model")
        self.assertEqual(ptq_s["Imgsz"], "640x640")
        self.assertEqual(ptq_s["mAP50-95"], "28.91%")
        self.assertRegex(ptq_s["Speedup vs YOLO26m 640"], r"^\d+\.\d{2}x$")
        qat_s = next(r for r in rows if r["Model"] == "int8_yolo26s_qat_640_openvino_model")
        self.assertEqual(qat_s["Role"], "QAT")
        self.assertEqual(qat_s["mAP50-95"], "27.92%")
        self.assertRegex(qat_s["Speedup vs YOLO26m 640"], r"^\d+\.\d{2}x$")
        self.assertFalse(any(r["Imgsz"] == "480x480" for r in rows))
        self.assertTrue(any(r["Model"] == "int8_yolo26m_qat_640_openvino_model" for r in rows))
        baseline = next(r for r in rows if r['Model'] == 'yolo26m-bdd100k.pt')
        self.assertEqual(baseline['mAP50-95'], 'N/A (chưa đo FP16)')
        self.assertEqual(ptq_s['Drop(pp)'], 'N/A')

    def test_speedup_uses_yolo26m_640_as_the_single_reference(self):
        records = [
            record("yolo26m-bdd100k.pt", "baseline", 640, 33.06, 226.99),
            record("yolo26m-bdd100k.pt", "baseline", 480, 28.54, 139.76),
            record("int8_qat_openvino_model", "INT8", 640, 31.45, 82.05),
            record("int8_yolo26m_480_openvino_model", "INT8", 480, 28.32, 58.13),
        ]

        rows = apply_reference_speedup(deduplicate_records(records))
        by_key = {(row["canonical_name"], row["imgsz"]): row for row in rows}

        self.assertAlmostEqual(by_key[("yolo26m-bdd100k.pt", 640)]["speedup"], 1.00)
        self.assertAlmostEqual(by_key[("yolo26m-bdd100k.pt", 480)]["speedup"], 1.62)
        self.assertAlmostEqual(by_key[("int8_qat_openvino_model", 640)]["speedup"], 2.77)
        self.assertAlmostEqual(by_key[("int8_yolo26m_480_openvino_model", 480)]["speedup"], 3.90)
        self.assertEqual(sum(row["role"] == "baseline" for row in rows), 2)

    def test_incompatible_protocols_are_not_merged_or_compared(self):
        base = record("yolo26m-bdd100k.pt", "baseline", 640, 33, 100)
        candidate = record("int8_yolo26m_640_openvino_model", "INT8", 640, 32, 20)
        other = {**candidate, "latency_protocol": "other", "serial_fps": 100}
        rows = apply_reference_speedup(deduplicate_records([base, candidate, other]))
        self.assertEqual(len(rows), 3)
        self.assertIsNone(next(r for r in rows if r["latency_protocol"] == "other")["speedup"])
        unverified = {**candidate, "latency_verified": False}
        rows = apply_reference_speedup(deduplicate_records([base, unverified]))
        self.assertIsNone(rows[1]["speedup"])
        first = {**candidate, "lat_mean_ms": 25, "serial_fps": 40}
        self.assertEqual(deduplicate_records([first, candidate])[0]["lat_mean_ms"], 25)

    def test_yolo26s_qat_stays_in_the_yolo26s_family(self):
        self.assertEqual(infer_family("int8_yolo26s_qat_640_openvino_model"), "YOLO26s")


if __name__ == "__main__":
    unittest.main()
