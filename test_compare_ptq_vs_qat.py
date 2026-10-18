import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import compare_ptq_vs_qat as compare

class PTQComparisonTests(unittest.TestCase):
    def report(self, models):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_text(json.dumps({'models': models}))
            return compare.parse_report_and_compare(path)

    def fixture(self):
        models = []
        for label, name, ap in [('baseline', 'yolo26m-bdd100k.pt', .33),
                                ('PTQ_INT8', 'int8_yolo26m_640_openvino_model', .32),
                                ('INT8', 'int8_yolo26m_qat_640_openvino_model', .31)]:
            models.append({'label':label, 'artifact':{'path':name,'size_mib':20},
                'accuracy':dict.fromkeys(['map50_95','map50','map75','precision_macro','recall_macro','f1_macro'],ap),
                'latency':{'predict_wall_ms':{'mean':10,'median':9,'p95':12},'inference_ms':{'mean':8}},'serial_fps':100})
        return models

    def test_default_models_exist(self):
        for path in [compare.DEFAULT_BASELINE,compare.DEFAULT_PTQ,compare.DEFAULT_QAT,compare.DEFAULT_YOLO26S]:
            self.assertTrue(Path(path).exists(),path)

    def test_artifact_identifies_qat_and_missing_models_raise(self):
        models=self.fixture()
        self.assertEqual(self.report(models)['rows'][0][1:4],['33.00%','32.00%','31.00%'])
        with self.assertRaisesRegex(ValueError,'thiếu model'):
            self.report(models[:2])
        models[0]['artifact']['path']='yolo26s-bdd100k.pt'
        with self.assertRaisesRegex(ValueError,'YOLO26s'):
            self.report(models)

    def test_missing_measurements_are_not_zero(self):
        models=self.fixture()
        del models[1]['accuracy']['map50_95']
        with self.assertRaisesRegex(ValueError,'accuracy'):
            self.report(models)

    def test_nonfinite_accuracy_and_nonpositive_latency_are_rejected(self):
        for value in [float('nan'), float('inf'), -1, 2]:
            models=self.fixture()
            models[1]['accuracy']['map50_95']=value
            with self.assertRaisesRegex(ValueError,'accuracy'):
                self.report(models)
        for value in [0, -1, float('nan'), 'fast']:
            models=self.fixture()
            models[2]['latency']['predict_wall_ms']['mean']=value
            with self.assertRaisesRegex(ValueError,'latency'):
                self.report(models)
