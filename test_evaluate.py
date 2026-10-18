"""Offline tests of label conversion and comparison; no model inference."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
import evaluate as ev


class EvaluationTests(unittest.TestCase):
    def test_conversion_empty_frame_alias_and_clipping(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            images = root / 'images'
            images.mkdir()
            for name in ['a.jpg', 'b.jpg']:
                Image.new('RGB', (100, 50)).save(images / name)
            frames = [{'name': 'a.jpg', 'labels': [
                {'category': 'person', 'box2d': {'x1': -10, 'y1': 0, 'x2': 50, 'y2': 50}},
                {'category': 'lane', 'poly2d': []}]}, {'name': 'b.jpg', 'labels': []}]
            labels = root / 'labels.json'
            ev.write_json(labels, {'frames': frames})
            args = argparse.Namespace(images=str(images), labels=str(labels), names=','.join(ev.DEFAULT_NAMES),
                                      limit=0, seed=42, max_det=300, ignore_policy='include')
            out = root / 'out'
            out.mkdir()
            yaml, names, selected, stats = ev.prepare_dataset(args, out)
            row = (out / 'dataset/labels/val/a.txt').read_text().split()
            self.assertEqual(row[0], '0')
            self.assertEqual([float(v) for v in row[1:]], [0.25, 0.5, 0.5, 1.0])
            self.assertEqual((out / 'dataset/labels/val/b.txt').read_text(), '')
            self.assertEqual(stats['empty_images'], 1)
            self.assertEqual(stats['clipped_boxes'], 1)
            self.assertEqual(len(selected), 2)
            self.assertTrue(yaml.is_file())
            manifest = json.loads((out / 'dataset_manifest.json').read_text())
            self.assertEqual(manifest[0]['image_sha256'], ev.sha256(images / 'a.jpg'))
            # Missing annotation entry must not become a false negative-only image.
            ev.write_json(labels, frames[:1])
            with self.assertRaisesRegex(ValueError, 'thiếu entry'):
                ev.prepare_dataset(args, root / 'missing')

    def test_drop_units_zero_baseline_and_improvement(self):
        def result(ap, latency, size):
            return {'accuracy': {'map50_95': ap, 'map50': ap},
                    'latency': {'predict_wall_ms': {'mean': latency}},
                    'artifact': {'size_mib': size}}
        c = ev.comparison(result(.420, 100, 40), result(.407, 25, 10))
        self.assertAlmostEqual(c['map_drop_pp'], 1.3)
        self.assertAlmostEqual(c['map_drop_relative_pct'], 3.0952380952380953)
        self.assertEqual(c['speedup'], 4)
        self.assertEqual(c['size_compression_ratio'], 4)
        self.assertIsNone(ev.comparison(result(0, 100, 40), result(.1, 25, 10))['map_drop_relative_pct'])
        self.assertLess(ev.comparison(result(.4, 100, 40), result(.5, 25, 10))['map_drop_pp'], 0)

    def test_latency_percentile(self):
        stats = ev.distribution([10, 20, 30, 40, 50])
        self.assertEqual(stats['mean'], 30)
        self.assertEqual(stats['median'], 30)
        self.assertAlmostEqual(stats['p95'], 48)

    def test_coco_names_rejected(self):
        model = argparse.Namespace(names={i: f'coco{i}' for i in range(80)})
        with self.assertRaisesRegex(ValueError, 'Lớp model không khớp'):
            ev.verify_names(model, ev.DEFAULT_NAMES)


if __name__ == '__main__':
    unittest.main()
