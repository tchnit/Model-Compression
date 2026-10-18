import io
import json
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from mvp_demo import MVPService, LIMITS

MODEL = Path(__file__).parent / 'models/int8_yolo26s_640_openvino_model'


class MVPTests(unittest.TestCase):
    def setUp(self):
        self.slot = threading.BoundedSemaphore(1)
        self.service = MVPService(lambda key: None, self.slot, MODEL)
        self.query = {'filename': ['image.jpg'], 'conf': ['0.25'], 'classes': ['0,2']}

    def tearDown(self):
        self.service.close()

    def wait(self, job):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            result = self.service.snapshot(job)
            if result['status'] not in ('starting', 'running'):
                return result
            time.sleep(.01)
        self.fail('Worker did not terminate')

    def test_upload_validation_precedes_model_loading(self):
        for patch_query, length in [({'filename': ['../x.jpg']}, 1), ({'filename': ['x.exe']}, 1),
                                    ({'conf': ['nan']}, 1), ({'conf': ['1.1']}, 1),
                                    ({'classes': ['99']}, 1), ({'classes': ['']}, 1), ({}, LIMITS['image_bytes'] + 1)]:
            with self.subTest(query=patch_query):
                with self.assertRaises(ValueError):
                    self.service.submit(io.BytesIO(b'x'), length, {**self.query, **patch_query})
        self.assertEqual(self.service.jobs, {})

    def test_busy_and_interrupted_uploads_leave_no_temporary_files(self):
        self.slot.acquire()
        with self.assertRaisesRegex(RuntimeError, 'tác vụ khác'):
            self.service.submit(io.BytesIO(b'x'), 1, self.query)
        self.slot.release()
        with self.assertRaisesRegex(ValueError, 'bị ngắt'):
            self.service.submit(io.BytesIO(b'x'), 5, self.query)
        self.assertEqual(list(Path(self.service.temp.name).iterdir()), [])
        self.assertTrue(self.slot.acquire(blocking=False))
        self.slot.release()

    def test_corrupt_video_produces_downloadable_failure_report(self):
        data = b'not a video'
        task = self.service.submit(io.BytesIO(data), len(data), {**self.query, 'filename': ['broken.webm']})
        job = self.service.get(task['id'])
        result = self.wait(job)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['processed'], 0)
        report = json.loads((job['folder'] / 'report.json').read_text())
        self.assertIn('report.json', result['files'])
        self.assertIsNone(report['inference_mean_ms'])
        self.assertIsNone(report['processing_fps'])
        self.assertTrue(self.slot.acquire(blocking=False))
        self.slot.release()

    def test_cancel_during_initialization_and_expiry_cleanup(self):
        import cv2
        import numpy as np
        entered, release = threading.Event(), threading.Event()
        def load(key):
            entered.set()
            release.wait(5)
            return None
        self.service.get_model = load
        data = cv2.imencode('.jpg', np.zeros((10, 10, 3), np.uint8))[1].tobytes()
        task = self.service.submit(io.BytesIO(data), len(data), self.query)
        self.assertTrue(entered.wait(5))
        job = self.service.get(task['id'])
        job['cancel'].set()
        release.set()
        result = self.wait(job)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['processed'], 0)
        job['touched'] = time.monotonic() - LIMITS['ttl_seconds'] - 1
        folder = job['folder']
        self.service.cleanup()
        self.assertFalse(folder.exists())
        self.assertNotIn(task['id'], self.service.jobs)

    def test_report_write_failure_releases_inference_slot(self):
        with patch.object(self.service, 'write_report', side_effect=OSError('disk full')):
            task = self.service.submit(io.BytesIO(b'bad'), 3, {**self.query, 'filename': ['bad.webm']})
            job = self.service.get(task['id'])
            result = self.wait(job)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('ổ đĩa', result['message'])
        self.assertTrue(self.slot.acquire(blocking=False))
        self.slot.release()

    def evaluation_job(self, predictions):
        job = dict(id='a' * 32, filename='image.jpg', kind='image', status='completed', width=100, height=100,
                   touched=time.monotonic(), settings={'conf': .001, 'classes': [0]},
                   frames=[{'detections': predictions}])
        self.service.jobs[job['id']] = job
        return job

    def test_ap_matching_penalizes_duplicates_and_handles_zero_predictions(self):
        good = dict(class_id=0, class_name='person', confidence=.9, xyxy=[10, 10, 40, 40])
        job = self.evaluation_job([good])
        gt = [{'name': 'image.jpg', 'labels': [{'category': 'pedestrian', 'box2d': dict(x1=10, y1=10, x2=40, y2=40)}]}]
        gt[0]['labels'].append({'category': 'lane', 'poly2d': []})
        payload = dict(jobs=[job['id']], ground_truth=gt)
        perfect = self.service.evaluate(payload)
        self.assertGreater(perfect['map50_95'], .99)
        self.assertEqual(perfect['skipped_non_detection_labels'], 1)
        job['frames'][0]['detections'] = [dict(good, xyxy=[60, 60, 90, 90], confidence=.99), good]
        bad = self.service.evaluate(payload)
        self.assertLess(bad['map50'], perfect['map50'])
        job['frames'][0]['detections'] = []
        empty = self.service.evaluate(payload)
        self.assertEqual(empty['map50'], 0)
        self.assertEqual(empty['recall'], 0)
        self.service.jobs.clear()

    def test_ground_truth_requires_exact_image_names_and_valid_boxes(self):
        job = self.evaluation_job([])
        for gt in [[], [{'name': 'image.jpg', 'labels': [{'category': 'person'}]}],
                   [{'name': 'image.jpg', 'labels': [{'category': 'person', 'box2d': dict(x1=10, y1=10, x2=500, y2=40)}]}]]:
            with self.subTest(gt=gt):
                with self.assertRaises(ValueError):
                    self.service.evaluate(dict(jobs=[job['id']], ground_truth=gt))
        self.service.jobs.clear()

    def test_media_ranges_and_invalid_ranges(self):
        path = Path(self.service.temp.name) / 'annotated.webm'
        path.write_bytes(b'0123456789')
        class Handler:
            headers = {'Range': 'bytes=2-5'}
            wfile = io.BytesIO()
            def send_response(self, status): self.status = status
            def send_header(self, key, value): pass
            def end_headers(self): pass
            def send_json(self, status, payload): self.status = status
        handler = Handler()
        self.service.serve_file(handler, path)
        self.assertEqual(handler.status, 206)
        self.assertEqual(handler.wfile.getvalue(), b'2345')
        handler.headers = {'Range': 'bytes=999-'}
        self.service.serve_file(handler, path)
        self.assertEqual(handler.status, 416)

    def test_real_image_and_stream_style_video_reuse_openvino(self):
        import cv2
        import run_demo
        self.service.get_model = run_demo.get_model
        source = next((Path(__file__).parent / 'demo_data/images').glob('*.jpg'))
        image_data = source.read_bytes()
        task = self.service.submit(io.BytesIO(image_data), len(image_data), self.query)
        image_job = self.service.get(task['id'])
        self.assertEqual(self.wait(image_job)['status'], 'completed')
        model = run_demo.LOADED_MODELS['yolo26s_ptq_int8']
        path = Path(self.service.temp.name) / 'test.webm'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'VP80'), 6, (640, 360))
        self.assertTrue(writer.isOpened())
        frame = cv2.resize(cv2.imread(str(source)), (640, 360))
        for _ in range(4):
            writer.write(frame)
        writer.release()
        video_data = path.read_bytes()
        capture_factory = cv2.VideoCapture
        class StreamCapture:
            def __init__(self, *args): self.capture = capture_factory(*args)
            def get(self, prop): return 0 if prop == cv2.CAP_PROP_FRAME_COUNT else self.capture.get(prop)
            def __getattr__(self, key): return getattr(self.capture, key)
        with patch.object(cv2, 'VideoCapture', StreamCapture):
            task = self.service.submit(io.BytesIO(video_data), len(video_data), {**self.query, 'filename': ['test.webm']})
            video_job = self.service.get(task['id'])
            self.assertEqual(self.wait(video_job)['status'], 'completed')
        report = json.loads((video_job['folder'] / 'report.json').read_text())
        self.assertEqual(report['processed'], 4)
        self.assertEqual(report['total'], 4)
        self.assertEqual(report['runtime']['execution_devices'], ['CPU'])
        self.assertEqual(report['runtime']['ov_threads'], 4)
        self.assertEqual(report['timings']['initialization_ms'], 0)
        self.assertGreater(report['inference_mean_ms'], 0)
        self.assertGreater(report['processing_fps'], 0)
        self.assertIs(run_demo.LOADED_MODELS['yolo26s_ptq_int8'], model)
        capture = cv2.VideoCapture(str(video_job['folder'] / 'annotated.webm'))
        self.assertTrue(capture.read()[0])
        capture.release()


if __name__ == '__main__':
    unittest.main()
