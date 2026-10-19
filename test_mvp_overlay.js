'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {mvpBuildBDD100KJson} = require('./demo_data/mvp.js');

const source = fs.readFileSync(`${__dirname}/demo_data/mvp.js`, 'utf8');
assert.match(source, /const MODEL_KEY = 'yolo26s_ptq_int8'/);
assert.match(source, /models: \[MODEL_KEY\]/);
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
const context = vm.createContext({
  window: {DEMO_DATA: {scenes: [{id: 'scene_01', title: 'Scene', file: 'scene.jpg', data_uri: 'data:image/jpeg;base64,AA', width: 1280, height: 720}], models: {yolo26s_ptq_int8: model}}, addEventListener() {}},
  document: {body: element('body'), getElementById: element, createElement: tag => element(`created-${tag}-${elements.size}`), querySelectorAll: () => [], addEventListener(type, callback) { if (type === 'DOMContentLoaded') ready = callback; }},
  location: {hash: '#mvp'}, history: {replaceState() {}}, requestAnimationFrame: callback => callback(),
  CLASS_COLORS: {car: '#fff'}, generateSvgElements: () => '', navigator: {clipboard: {writeText: async () => {}}},
  Blob, URL: {createObjectURL: () => 'blob:test', revokeObjectURL() {}}, setTimeout() {}, console,
  fetch: async (url, options) => {
    if (url === '/api/status') return {ok: true, json: async () => ({status: 'online', cpu_threads: 8})};
    requests.push(JSON.parse(options.body));
    return {ok: true, json: async () => ({status: 'success', width: 1280, height: 720, detections: {yolo26s_ptq_int8: {inference_time_ms: 50, runtime: 'CPU ×3', boxes: []}}})};
  }
});
vm.runInContext(source, context);

(async () => {
  await ready();
  assert.equal(elements.get('mvpModelSelect').value, 'yolo26s_ptq_int8');
  assert.match(elements.get('mvpCoreSelect').innerHTML, /value="8">CPU ×8/);
  assert.equal(requests.length, 1);
  assert.deepEqual(requests[0].models, ['yolo26s_ptq_int8']);
  assert.equal(requests[0].threads, 3);
  assert.equal(elements.get('mvpFps').textContent, '20.0 FPS');
  assert.equal(elements.get('mvpMap50').textContent, '51.44%');
  await handlers.get('mvpRunDetection:click')();
  assert.equal(requests.length, 2, 'Nhận diện reruns the selected model');
  console.log('MVP single-model Studio flow, CPU selection, live FPS and BDD100K export PASS');
})().catch(error => { console.error(error); process.exitCode = 1; });
