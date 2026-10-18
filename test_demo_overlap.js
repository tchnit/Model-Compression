const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const scripts = ['overlays.js', 'studio.js'].map(name => fs.readFileSync(`${__dirname}/demo_data/${name}`, 'utf8'));
const elements = new Map();
const context = vm.createContext({
  window: { DEMO_DATA: JSON.parse(fs.readFileSync(`${__dirname}/demo_data/detections.json`)), addEventListener() {} },
  document: {
    addEventListener() {},
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, { setAttribute() {} });
      return elements.get(id);
    }
  }
});
for (const script of scripts) {
  vm.runInContext(script, context);
}
const agreement = context.calculateDetectionAgreement;
const box = (name, coords = [0, 0, 10, 10]) => ({ class_name: name, conf: 0.9, box_px: coords });
const car = box('car');
assert.equal(agreement([car], [car], 100, 100).retention, 100);
assert.equal(agreement([car], [box('bus')], 100, 100).matches, 0);
assert.equal(agreement([car], [box('car', [20, 20, 30, 30])], 100, 100).matches, 0);
assert.equal(agreement([car], [car, car], 100, 100).matches, 1);
assert.equal(agreement([car, car], [car], 100, 100).retention, 50);
assert.equal(agreement([], [], 100, 100).retention, null);
assert.equal(agreement([car], [], 100, 100).retention, 0);
assert.equal(agreement([box('car', [0, 0, 0, 0])], [car], 100, 100).matches, 0);
assert.equal(agreement([car], [{ class_name: 'car', box_norm: [0, 0, 0.1, 0.1] }], 100, 100).meanIou, 1);
assert.equal(agreement([car], [box('car', [0, 0, 5, 10])], 100, 100).meanIou, 0.5);
assert.equal(agreement([car], [box('car', [0, 0, 4.9, 10])], 100, 100).matches, 0);

// Exercise the shared render path used by model, scene, confidence and class changes.
vm.runInContext(`
  demoData = {
    scenes: [{ id: 'one' }, { id: 'two' }],
    models: {},
    detections: {
      a: { one: { boxes: [${JSON.stringify(car)}] }, two: { boxes: [${JSON.stringify(car)}] } },
      b: { one: { boxes: [${JSON.stringify(car)}] }, two: { boxes: [] } }
    }
  };
  currentModelA = 'a'; currentModelB = 'b'; currentSceneId = 'one';
  renderBoundingBoxes();
`, context);
const display = () => elements.get('overlapDisplay').textContent;
assert.equal(display(), '100.0%');
vm.runInContext("currentSceneId = 'two'; renderBoundingBoxes();", context);
assert.equal(display(), '0.0%');
vm.runInContext("currentModelB = 'a'; renderBoundingBoxes();", context);
assert.equal(display(), '100.0%');
vm.runInContext('currentConf = 0.95; renderBoundingBoxes();', context);
assert.equal(display(), '—');
vm.runInContext("currentConf = 0.25; toggleClass('car', false);", context);
assert.equal(display(), '—');
vm.runInContext("toggleClass('car', true);", context);
assert.equal(display(), '100.0%');
vm.runInContext("currentModelB = 'missing'; renderBoundingBoxes();", context);
assert.equal(display(), '—');
assert.match(elements.get('overlapDetails').textContent, /Chưa có dữ liệu/);
console.log('Detection overlap: matching, empty/missing data and render updates passed.');

context.filterTable('YOLO26s');
assert.equal(elements.get('benchmarkSpeedupReference').textContent, 'vs YOLO26s');
const speedups = () => [...elements.get('benchmarkTableBody').innerHTML.matchAll(/font-weight:750">([\d.]+)x/g)].map(m => m[1]);
const measured = JSON.parse(fs.readFileSync(`${__dirname}/demo_data/detections.json`)).benchmarks;
const sBase = measured.find(d => d.family === 'YOLO26s' && d.role === 'baseline');
assert.deepEqual(speedups(), measured.filter(d => d.family === 'YOLO26s').map(d => (sBase.lat / d.lat).toFixed(2)));
assert.match(elements.get('benchmarkSpeedupReference').title, /yolo26s-bdd100k.pt.*ms/);
const sSpeedups = speedups();
const original = vm.runInContext('JSON.stringify(BENCHMARK_DATA)', context);
for (const filter of ['YOLO26m', 'all', 'INT8']) {
  context.filterTable(filter);
  assert.equal(elements.get('benchmarkSpeedupReference').textContent, 'vs YOLO26m 640');
  assert.ok(speedups().length > 0);
}
context.filterTable('YOLO26s');
assert.deepEqual(speedups(), sSpeedups);
assert.equal(vm.runInContext('JSON.stringify(BENCHMARK_DATA)', context), original);
console.log('Benchmark speedup: family reference, filter switching and unchanged source data passed.');

for (const filter of ['all', 'YOLO26s', 'INT8', 'YOLO26m']) {
  context.filterTable(filter);
  const winners = elements.get('benchmarkTableBody').innerHTML.match(/<tr class="highlight-row">[\s\S]*?<\/tr>/g) || [];
  assert.equal(winners.length, filter === 'YOLO26m' ? 0 : 1);
  if (winners.length) {
    assert.match(winners[0], /<strong>int8_yolo26s_640_openvino_model<\/strong>/);
    assert.match(winners[0], /PTQ Winner/);
  }
}
console.log('Winner: only YOLO26s PTQ is highlighted across benchmark filters.');

vm.runInContext('for (const row of BENCHMARK_DATA) row.benchmark_temporary = true; filterTable("all"); updateBenchmarkSummary();', context);
assert.match(elements.get('benchmarkAccuracyNote').textContent, /TẠM.*baseline hiển thị là FP16/);
assert.match(elements.get('benchmarkTableBody').innerHTML, /PyTorch FP16/);
assert.match(elements.get('retentionKpiDetail').textContent, /\(FP16\)/);
assert.doesNotMatch(elements.get('benchmarkTableBody').innerHTML, /FP32/);
vm.runInContext('for (const row of BENCHMARK_DATA) { row.benchmark_temporary = false; if (row.role === "baseline") { row.accuracy_pending = true; row.map95 = null; } } filterTable("all"); updateBenchmarkSummary();', context);
assert.match(elements.get('benchmarkAccuracyNote').textContent, /đang chờ/);
assert.match(elements.get('benchmarkTableBody').innerHTML, /Chưa đo FP16/);
assert.doesNotMatch(elements.get('benchmarkTableBody').innerHTML, /NaN|Infinity/);
vm.runInContext('for (const row of BENCHMARK_DATA) { row.accuracy_pending = false; if (row.map95 == null) row.map95 = 30; } updateBenchmarkSummary();', context);
assert.match(elements.get('benchmarkAccuracyNote').textContent, /FP16 đầy đủ/);
