const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const scripts = ['overlays.js', 'studio.js'].map(name => fs.readFileSync(`${__dirname}/demo_data/${name}`, 'utf8'));
const html = fs.readFileSync(`${__dirname}/demo.html`, 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));

const expectedModels = ['yolo26s_ptq_int8', 'yolo26s_fp32', 'yolo26m_fp32', 'yolo26m_ptq_int8'];
for (const side of ['A', 'B']) {
  const select = html.match(new RegExp(`<select class="model-select" id="modelSelect${side}">([\\s\\S]*?)<\\/select>`))[1];
  assert.deepEqual([...select.matchAll(/<option value="([^"]+)"/g)].map(match => match[1]).filter(Boolean), expectedModels);
}
assert.doesNotMatch(html, /applyPreset\('qat'\)/);
assert.doesNotMatch(html, /Tỷ Lệ Nén Trọng Số|Tăng Tốc Suy Luận \(CPU\)|Bảo Toàn mAP50–95|Precision Trên Tập Validation/);
assert.match(html, /telemetry-label">Chỉ số mô hình A<\/span>/);
assert.match(html, /telemetry-label">Chỉ số mô hình B<\/span>/);

async function studio() {
  const elements = new Map();
  const requests = [];
  const element = id => {
    if (!elements.has(id)) {
      const classes = new Set();
      elements.set(id, {
        style: {}, value: '', textContent: '', innerHTML: '',
        naturalWidth: 100, naturalHeight: 80, complete: true,
        offsetWidth: 100, offsetHeight: 80,
        setAttribute() {}, addEventListener() {},
        append(...children) { this.children = [...(this.children || []), ...children]; },
        replaceChildren(...children) { this.children = children; },
        classList: {
          add: value => classes.add(value), remove: value => classes.delete(value),
          contains: value => classes.has(value),
          toggle(value, enabled) { enabled ? classes.add(value) : classes.delete(value); }
        }
      });
    }
    return elements.get(id);
  };
  const modelKeys = ['a', 'b', 'c', ...expectedModels];
  const data = {
    scenes: ['scene_01', 'scene_02'].map(id => ({
      id, title: id, file: `${id}.jpg`, desc: id,
      width: 100, height: 80, data_uri: `data:image/png;base64,${id}`
    })),
    models: Object.fromEntries(modelKeys.map(key => [key, {
      name: key, badge: key, path: key, size_mb: 10, map50: 55.25,
      avg_latency_ms: key === 'yolo26s_fp32' ? 40 : 20, fps: 50
    }])),
    benchmarks: JSON.parse(fs.readFileSync(`${__dirname}/demo_data/detections.json`)).benchmarks || [],
    detections: {}
  };
  const context = vm.createContext({
    window: { DEMO_DATA: data, addEventListener() {} },
    document: { createElement: tag => ({ tag, style: {}, addEventListener() {}, append(...children) { this.children = children; } }), getElementById: element, addEventListener() {}, querySelectorAll: () => [] },
    console: { warn() {}, log() {} },
    setTimeout() {}, requestAnimationFrame: callback => callback(),
    FileReader: class {
      readAsDataURL(file) { this.onload({ target: { result: file.data_uri } }); }
    },
    Image: class {
      naturalWidth = 300;
      naturalHeight = 200;
      set src(value) { queueMicrotask(() => this.onload()); }
    },
    fetch(url, options) {
      if (url === '/api/status') return Promise.resolve({ ok: true, json: async () => ({ status: 'online' }) });
      assert.equal(url, '/api/predict');
      return new Promise((resolve, reject) => {
        requests.push({ payload: JSON.parse(options.body), resolve, reject });
      });
    }
  });
  for (const script of scripts) {
    vm.runInContext(script, context);
  }
  await context.initApp();
  return {
    context, requests, data, element,
    read: expression => vm.runInContext(expression, context),
    finish(index, error) {
      const request = requests[index];
      assert.ok(request, `Missing request ${index}`);
      request.resolve({
        ok: !error, status: error ? 503 : 200,
        json: async () => error ? { status: 'error', message: error } : {
          status: 'success', width: 100, height: 80,
          detections: Object.fromEntries(request.payload.models.map(key => [key, {
            inference_time_ms: 1, runtime: 'CPU ×4', detections_count: 0, boxes: []
          }]))
        }
      });
    }
  };
}

(async () => {
  // Opening the page and uploading an image do not start an unselected model.
  {
    const s = await studio();
    assert.equal(s.read('currentModelA'), '');
    assert.equal(s.read('currentModelB'), '');
    assert.equal(s.requests.length, 0);
    assert.equal(s.element('downloadJsonBtn').disabled, true);
    s.context.updateSceneView();
    assert.equal(s.requests.length, 0);
    await s.context.handleCustomImageUpload({ target: { files: [{
      name: 'upload.png', data_uri: 'data:image/png;base64,uploaded'
    }] } });
    await tick();
    const scene = s.data.scenes[0];
    assert.equal(scene.file, 'upload.png');
    assert.equal(scene.width, 300);
    assert.equal(scene.height, 200);
    assert.equal(s.read('currentSceneId'), scene.id);
    assert.equal(s.requests.length, 0);
    const selected = s.context.selectModel('A', 'a');
    await tick();
    assert.deepEqual(s.requests[0].payload.models, ['a']);
    assert.equal(s.requests[0].payload.image, scene.data_uri);
    s.finish(0);
    await selected;
    assert.ok(scene.live_models.has('a'));
    assert.equal(s.data.detections.a[scene.id].detections_count, 0);
  }

  // Concurrent selections are serialized and the same scene/model is reused.
  {
    const s = await studio();
    const a = s.context.selectModel('A', 'a');
    await tick();
    const b = s.context.selectModel('B', 'b');
    const same = s.context.selectModel('A', 'b');
    await tick();
    assert.equal(s.requests.length, 1);
    assert.equal(s.element('downloadJsonBtn').disabled, true);
    assert.deepEqual(s.requests[0].payload.models, ['a']);
    s.finish(0);
    await tick();
    assert.equal(s.requests.length, 2);
    assert.deepEqual(s.requests[1].payload.models, ['b']);
    s.finish(1);
    await Promise.all([a, b, same]);
    assert.equal(s.element('downloadJsonBtn').disabled, false);
    await s.context.selectModel('A', 'a');
    await s.context.selectScene('scene_01');
    s.context.toggleClass('car', false);
    s.context.updateSceneView();
    assert.equal(s.requests.length, 2);
    assert.equal(s.element('aiLoadingOverlay').classList.contains('active'), false);
  }

  // A queued model that is no longer selected must never start.
  {
    const s = await studio();
    const a = s.context.selectModel('A', 'a');
    await tick();
    const b = s.context.selectModel('B', 'b');
    const c = s.context.selectModel('B', 'c');
    s.finish(0);
    await tick();
    assert.equal(s.requests.length, 2);
    assert.deepEqual(s.requests[1].payload.models, ['c']);
    s.finish(1);
    await Promise.all([a, b, c]);
    assert.ok(!s.data.scenes[0].live_models.has('b'));
  }

  // A response for an old scene is cached there; queued old-scene work is skipped.
  {
    const s = await studio();
    const a = s.context.selectModel('A', 'a');
    await tick();
    const b = s.context.selectModel('B', 'b');
    const nextScene = s.context.selectScene('scene_02');
    s.finish(0);
    await tick();
    assert.equal(s.read('currentSceneId'), 'scene_02');
    assert.equal(s.requests.length, 2);
    assert.equal(s.requests[1].payload.image, s.data.scenes[1].data_uri);
    assert.deepEqual(s.requests[1].payload.models, ['a']);
    assert.ok(s.data.detections.a.scene_01);
    assert.ok(!s.data.detections.b?.scene_01);
    assert.equal(s.element('aiLoadingOverlay').classList.contains('active'), true);
    s.finish(1);
    await tick();
    assert.equal(s.requests.length, 3);
    assert.deepEqual(s.requests[2].payload.models, ['b']);
    s.finish(2);
    await Promise.all([a, b, nextScene]);
    assert.ok(s.data.detections.b.scene_02);
    assert.equal(s.element('aiLoadingOverlay').classList.contains('active'), false);
  }

  // A failed request stays retryable and does not poison the serial queue.
  {
    const s = await studio();
    const failed = s.context.selectModel('A', 'a');
    await tick();
    s.finish(0, 'Busy');
    await failed;
    assert.ok(!s.data.scenes[0].live_models?.has('a'));
    const retry = s.context.selectModel('A', 'a');
    await tick();
    assert.equal(s.requests.length, 2);
    s.finish(1);
    await retry;
    assert.ok(s.data.scenes[0].live_models.has('a'));
  }

  // A preset is an explicit choice of two models, and uses the same queue.
  {
    const s = await studio();
    const preset = s.context.applyPreset('ptq');
    await tick();
    assert.equal(s.requests.length, 1);
    assert.deepEqual(s.requests[0].payload.models, ['yolo26m_fp32']);
    s.finish(0);
    await tick();
    assert.equal(s.requests.length, 2);
    assert.deepEqual(s.requests[1].payload.models, ['yolo26m_ptq_int8']);
    s.finish(1);
    await preset;
    await tick();
    assert.equal(s.read('currentModelA'), 'yolo26m_fp32');
    assert.equal(s.read('currentModelB'), 'yolo26m_ptq_int8');
  }

  {
    const s = await studio();
    const preset = s.context.applyPreset('ptq-s');
    await tick();
    assert.deepEqual(s.requests[0].payload.models, ['yolo26s_fp32']);
    s.finish(0);
    await tick();
    assert.deepEqual(s.requests[1].payload.models, ['yolo26s_ptq_int8']);
    s.finish(1);
    await preset;
    assert.equal(s.read('currentModelA'), 'yolo26s_fp32');
    assert.equal(s.read('currentModelB'), 'yolo26s_ptq_int8');
    assert.equal(s.element('speedupSourceLabel').textContent, 'yolo26s_ptq_int8 tăng tốc so với yolo26s_fp32');
    assert.equal(s.element('speedupDisplay').textContent, '2.00x');
    assert.match(s.element('telemetryMetricsA').textContent, /mAP@50: 55\.25% · FPS: 50\.0/);
    assert.match(s.element('telemetryMetricsB').textContent, /mAP@50: 55\.25% · FPS: 50\.0/);
  }

  console.log('Model selection: lazy start, upload, serialization, cache, stale work, retry and presets passed.');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
