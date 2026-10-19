'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {mvpBuildBDD100KJson} = require('./demo_data/mvp.js');
const {readApiJson} = require('./demo_data/overlays.js');

const source = fs.readFileSync(`${__dirname}/demo_data/mvp.js`, 'utf8');
assert.match(source, /const DEFAULT_MODEL_KEY = 'yolo26s_ptq_int8'/);
assert.match(source, /YOLO26s Baseline \(FP32\)/);
assert.match(source, /YOLO26m Baseline \(FP32\)/);
assert.match(source, /SỐ CORE CPU/);
assert.match(source, /Intel\(R\) Xeon\(R\) Platinum 8559C/);
assert.doesNotMatch(source, /mvpCpuInfo|· Hugging Face/);
assert.doesNotMatch(source, /yolo26s_export_int8_640|INT8 export 640/);
assert.doesNotMatch(source, /mvpFrameAtTime|mvpFiles|mvpGroundTruth|mvpCamera/);

const converted = mvpBuildBDD100KJson([
  {class_name: 'person', conf: .9, box_norm: [.1, .2, .3, .6]},
  {class_name: 'bike', conf: .8, box_px: [10, 20, 30, 40]}
], {file: 'sample.jpg', attributes: {weather: 'rainy'}}, 100, 100);
assert.equal(converted.name, 'sample.jpg');
assert.equal(converted.attributes.weather, 'rainy');
assert.deepEqual(converted.labels.map(item => item.category), ['pedestrian', 'bicycle']);
assert.deepEqual(converted.labels[0].box2d, {x1: 10, y1: 20, x2: 30, y2: 60});

const elements = new Map();
const handlers = new Map();
function element(id = '') {
  if (!elements.has(id)) {
    const classes = new Set();
    elements.set(id, {id, style: {}, value: '', checked: true, disabled: false, innerHTML: '', textContent: '', naturalWidth: 1280, naturalHeight: 720, src: '',
      classList: {add: value => classes.add(value), remove: value => classes.delete(value), toggle(value, enabled) { enabled ? classes.add(value) : classes.delete(value); }, contains: value => classes.has(value)},
      addEventListener(type, callback) { handlers.set(`${id}:${type}`, callback); }, setAttribute() {}, removeAttribute() {},
      append(...children) { this.children = children; }, replaceChildren(...children) { this.children = children; }, scrollIntoView() {}}
    );
  }
  return elements.get(id);
}

const requests = [];
let ready;
const model = {name: 'YOLO26s PTQ INT8 640 (OpenVINO)', badge: 'INT8', size_mb: 9.99, map50: 51.44, path: 'model.xml'};
const models = {
  yolo26s_ptq_int8: model,
  yolo26s_fp32: {...model, name: 'YOLO26s Baseline (FP16)', path: 'yolo26s.pt'},
  yolo26m_fp32: {...model, name: 'YOLO26m Baseline (FP16)', path: 'yolo26m.pt'}
};
const context = vm.createContext({
  window: {DEMO_DATA: {scenes: [{id: 'scene_01', title: 'Scene', file: 'scene.jpg', data_uri: 'data:image/jpeg;base64,AA', width: 1280, height: 720}], models}, addEventListener() {}},
  document: {body: element('body'), getElementById: element, createElement: tag => {
    const created = element(`created-${tag}-${elements.size}`); created.tagName = tag.toUpperCase(); return created;
  }, querySelectorAll: () => [], addEventListener(type, callback) { if (type === 'DOMContentLoaded') ready = callback; }},
  location: {hash: '#mvp'}, history: {replaceState() {}}, requestAnimationFrame: callback => callback(),
  CLASS_COLORS: {car: '#fff'}, generateSvgElements: () => '', readApiJson, navigator: {clipboard: {writeText: async () => {}}},
  Blob, URL: {createObjectURL: () => 'blob:test', revokeObjectURL() {}}, setTimeout() {}, console,
  fetch: async (url, options) => {
    if (url === 'api/status') return {ok: true, json: async () => ({status: 'online', cpu_name: 'Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz', cpu_threads: 8, int8_cpu_threads: 3, huggingface_cpu_threads: 8})};
    requests.push(JSON.parse(options.body));
    const key = requests.at(-1).models[0];
    return {ok: true, json: async () => ({status: 'success', width: 1280, height: 720, detections: {[key]: {inference_time_ms: 50, runtime: 'CPU ×3', boxes: []}}})};
  }
});
vm.runInContext(source, context);

(async () => {
  await assert.rejects(
    readApiJson({status: 502, url: 'https://proxy.invalid/api/predict', headers: {get: () => 'text/html'}, json: async () => { throw new SyntaxError('Unexpected token <'); }}),
    /HTTP 502 \(text\/html\).*thay vì JSON.*proxy\.invalid/
  );
  await ready();
  assert.equal(elements.get('mvpModelSelect').value, 'yolo26s_ptq_int8');
  const sceneInfo = elements.get('mvpSceneList').children[0].children[1];
  assert.equal(sceneInfo.tagName, 'DIV');
  assert.deepEqual(sceneInfo.children.map(child => child.tagName), ['DIV', 'DIV']);
  assert.match(elements.get('mvpCoreSelect').innerHTML, /value="3">CPU ×3/);
  assert.doesNotMatch(elements.get('mvpCoreSelect').innerHTML, /value="4"/);
  assert.equal(elements.get('mvpProcessorHost').textContent, 'Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz');
  assert.equal(requests.length, 1);
  assert.deepEqual(requests[0].models, ['yolo26s_ptq_int8']);
  assert.equal(requests[0].threads, 3);
  assert.equal(requests[0].processor, 'host');
  assert.equal(elements.get('mvpFps').textContent, '20.0 FPS');
  assert.equal(elements.get('mvpMap50').textContent, '51.44%');
  await handlers.get('mvpRunDetection:click')();
  assert.equal(requests.length, 2, 'Nhận diện reruns the selected model');
  handlers.get('mvpModelSelect:change')({target: {value: 'yolo26s_baseline_fp32'}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(requests.at(-1).models[0], 'yolo26s_baseline_fp32');
  assert.equal(elements.get('mvpPanelModel').textContent, 'YOLO26s Baseline (FP32)');
  assert.equal(elements.get('mvpModelInfo').textContent, 'OpenVINO FP32 · 36.56 MiB');
  handlers.get('mvpProcessorSelect:change')({target: {value: 'huggingface'}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(requests.at(-1).models[0], 'yolo26s_baseline_fp32');
  assert.equal(requests.at(-1).processor, 'huggingface');
  assert.equal(requests.at(-1).threads, 3);
  assert.match(elements.get('mvpCoreSelect').innerHTML, /value="8">CPU ×8/);
  assert.equal(elements.get('mvpCoreSelect').disabled, false);
  handlers.get('mvpCoreSelect:change')({target: {value: '1'}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(requests.at(-1).processor, 'huggingface');
  assert.equal(requests.at(-1).threads, 1);
  assert.equal(elements.get('mvpProcessorHost').textContent, 'Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz');
  console.log('MVP single-model Studio flow, CPU selection, live FPS and BDD100K export PASS');
})().catch(error => { console.error(error); process.exitCode = 1; });
