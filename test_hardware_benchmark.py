import unittest
from unittest.mock import patch
import numpy as np
import hardware_benchmark as h


class HardwareBenchmarkTests(unittest.TestCase):
    def test_pytorch_cpu_fp16_is_available_without_changing_fp32_mode(self):
        facts = dict(cuda=False, versions={'tensorrt': None}, providers=[],
                     openvino_devices={'CPU': {'capabilities': ['FP32', 'INT8']}})
        with patch.object(h, 'hardware', return_value=facts), patch.object(h, 'manifest', return_value={}):
            configs = h.capabilities()['configs']
        choices = [c for c in configs if c['model'] == 'yolo26s' and c['engine'] == 'pytorch' and c['device'] == 'cpu']
        for precision in ['FP16', 'FP32']:
            self.assertTrue(next(c for c in choices if c['precision'] == precision)['available'])

    def test_calibration_is_deterministic_and_rejects_validation_or_duplicates(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from prepare_runtime_models import select_calibration_images
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); train = root / 'train'; train.mkdir()
            paths = [train / f'{i}.jpg' for i in range(3)]
            for i,path in enumerate(paths):
                path.write_bytes(f'train-{i}'.encode())
            listing = root / 'calibration.txt'; listing.write_text('\n'.join(map(str,paths)))
            val = root / 'validation.json'
            val.write_text(json.dumps([dict(name='held-out.jpg',sha256=hashlib.sha256(b'validation').hexdigest())]))
            self.assertEqual(select_calibration_images(listing,2,val),select_calibration_images(listing,2,val))
            with self.assertRaises(ValueError):
                select_calibration_images(listing,0,val)
            with self.assertRaises(ValueError):
                select_calibration_images(listing,4,val)
            paths[0].write_bytes(b'validation')
            with self.assertRaisesRegex(ValueError,'overlaps validation'):
                select_calibration_images(listing,3,val)
            paths[0].write_bytes(paths[1].read_bytes())
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                select_calibration_images(listing,3,val)
            listing.write_text(str(val))
            with self.assertRaisesRegex(ValueError,'train split'):
                select_calibration_images(listing,1,val)

    def test_browser_manifest_publishes_only_onnx_models(self):
        import json
        import tempfile
        from pathlib import Path
        import prepare_runtime_models as export
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'demo_data').mkdir()
            (root / 'runtime_exports').mkdir()
            path = 'runtime_exports/yolo26s_fp32.onnx'
            (root / path).write_bytes(b'fixture')
            entries = {'yolo26s_FP32_onnx': dict(path=path,sha256='artifact-hash',source_sha256='source-hash'),
                       'yolo26s_INT8_onnx': dict(path=path,sha256='quantized-hash',source_sha256='source-hash',
                            calibration_images=128,calibration_sha256='calibration-hash',quantized_convs=102),
                       'yolo26s_FP32_tensorrt': dict(path='private.engine',gpu='private backend info')}
            with patch.object(export,'ROOT',root):
                export.write_browser_manifest(entries)
            result = json.loads((root / 'demo_data/browser_models.json').read_text())
            self.assertEqual(set(result['artifacts']),{'yolo26s_FP32','yolo26s_INT8'})
            self.assertEqual(result['artifacts']['yolo26s_FP32']['url'],path)
            self.assertEqual(result['artifacts']['yolo26s_INT8']['input_dtype'],'float32')
            self.assertEqual(result['artifacts']['yolo26s_INT8']['calibration_images'],128)
            self.assertNotIn('gpu',json.dumps(result))

    def test_letterbox_and_output_coordinates(self):
        image = np.zeros((720,1280,3),dtype=np.uint8)
        image[:,:,2] = 255
        tensor, transform = h.preprocess(image)
        self.assertEqual(tensor.shape,(1,3,640,640))
        self.assertEqual(tensor.dtype,np.float32)
        self.assertEqual(float(tensor[0,0,140,0]),1.)  # BGR -> RGB
        self.assertAlmostEqual(float(tensor[0,0,0,0]),114/255,places=6)
        output = np.zeros((1,300,6),dtype=np.float32)
        output[0,0] = [50,190,100,240,.8,2]
        output[0,1] = [0,140,640,500,.2,0]
        boxes = h.postprocess(output,transform,.25)
        self.assertEqual(len(boxes),1)
        self.assertEqual(boxes[0]['box_px'],[100,100,200,200])
        self.assertEqual(boxes[0]['class_name'],'car')
        half,_ = h.preprocess(image,True)
        self.assertEqual(half.dtype,np.float16)

    def test_rejects_unexpected_graph_and_invalid_class(self):
        with self.assertRaises(RuntimeError):
            h.postprocess(np.zeros((1,84,8400)),(1,0,0,640,640),.25)
        output = np.zeros((1,300,6)); output[0,0] = [0,0,1,1,.8,80]
        with self.assertRaises(RuntimeError):
            h.postprocess(output,(1,0,0,640,640),.25)

    def test_configuration_validation_has_no_fallback(self):
        base = dict(model='yolo26s',engine='pytorch',device='cpu',precision='FP32')
        configs = [{**base,'available':True,'reason':None},
                   {**base,'device':'cuda','available':False,'reason':'No GPU'}]
        with patch.object(h,'capabilities',return_value={'configs':configs}):
            self.assertEqual(h.validate(base)['repeats'],10)
            for fields in [{'device':'cuda'}, {'precision':'INT8'}, {'repeats':True},
                           {'repeats':0}, {'warmup':11}, {'conf':float('nan')}, {'conf':None}]:
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    h.validate({**base,**fields})

    def test_busy_benchmark_returns_503_and_does_not_run(self):
        import run_demo
        from test_run_demo import DemoServerTests
        # Reuse HTTP harness, mock heavyweight native runtime entirely.
        DemoServerTests.setUpClass()
        try:
            import json
            with patch.object(h,'validate',return_value={}), patch.object(h,'decode_image',return_value=(None,'hash')), patch.object(h,'run_benchmark') as runner:
                run_demo.INFERENCE_SLOT.acquire()
                try:
                    status,_ = DemoServerTests().request('POST','/api/benchmark/run',json.dumps({}))
                    self.assertEqual(status,503)
                    runner.assert_not_called()
                finally:
                    run_demo.INFERENCE_SLOT.release()
            self.assertEqual(DemoServerTests().request('HEAD','/runtime_exports/manifest.json')[0],404)
            self.assertEqual(DemoServerTests().request('HEAD','/runtime_exports/yolo26s_fp32.engine')[0],404)
            with patch.object(h,'validate',return_value={}), patch.object(h,'decode_image',return_value=(None,'hash')), patch.object(h,'run_benchmark',side_effect=RuntimeError('GPU failed')) as runner:
                status,body = DemoServerTests().request('POST','/api/benchmark/run',json.dumps({}))
                self.assertEqual(status,422)
                self.assertIn('GPU failed',json.loads(body)['message'])
                self.assertEqual(runner.call_count,1)  # No retry as CPU after GPU failure.
                self.assertTrue(run_demo.INFERENCE_SLOT.acquire(blocking=False))
                run_demo.INFERENCE_SLOT.release()
        finally:
            DemoServerTests.tearDownClass()


if __name__ == '__main__':
    unittest.main()
