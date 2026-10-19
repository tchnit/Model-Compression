const assert = require('node:assert/strict');
const {floatToHalf, halfToFloat, geometry, decode, statistics, isFallback, browserCapabilities, webgpuReason, previewMatches, wasmThreadLimit} = require('./demo_data/hardware_benchmark.js');
assert.equal(wasmThreadLimit(8,true,true),8);
assert.equal(wasmThreadLimit(8,false,true),1);
assert.equal(wasmThreadLimit(8,true,false),1);
for (const cpus of [undefined,0,-1,2.5,NaN]) assert.equal(wasmThreadLimit(cpus,true,true),1);
const config = {model:'yolo26s',engine:'wasm',device:'browser-cpu',precision:'INT8'};
const preview = {image:'image-a',result:{...config,conf:.25,boxes:[]}};
assert.equal(previewMatches(preview,config,'image-a',.25),true); // A successful zero-detection result is still current.
assert.equal(previewMatches(null,config,'image-a',.25),false);
assert.equal(previewMatches(preview,config,'image-b',.25),false);
assert.equal(previewMatches(preview,config,'image-a',.5),false);
assert.equal(previewMatches(preview,config,'image-a',NaN),false);
const threadedPreview = {...preview,result:{...preview.result,protocol:{cpu_threads:4}}};
assert.equal(previewMatches(threadedPreview,config,'image-a',.25,4),true);
assert.equal(previewMatches(threadedPreview,config,'image-a',.25,1),false);
for (const key of Object.keys(config)) assert.equal(previewMatches(preview,{...config,[key]:'changed'},'image-a',.25),false);
const {generateSvgElements: draw} = require('./demo_data/overlays.js');
const annotation = draw([{class_name:'car',conf:.8,box_px:[1250,20,1270,80]}],1280,720,2);
assert.match(annotation,/class="bbox-rect" x="1250" y="20" width="20" height="60"/);
assert.match(annotation,/<rect x="1166" y="0" width="114" height="32"/);
assert.match(annotation,/font-size:22px/);
assert.match(annotation,/car 80%/);
assert.equal(draw([],720,1280), '');
const manifest = require('./demo_data/browser_models.json');
const browser = browserCapabilities(manifest);
assert.deepEqual([...new Set(browser.configs.map(c => c.engine))],['wasm','webgpu']);
assert.equal(Object.keys(browser.models).length,8);
for (const model of ['yolo26m','yolo26s']) {
  assert.ok(browser.configs.find(c => c.model === model && c.engine === 'wasm' && c.precision === 'FP32').available);
  assert.ok(browser.configs.find(c => c.model === model && c.engine === 'wasm' && c.precision === 'FP16').available);
  assert.ok(browser.configs.find(c => c.model === model && c.engine === 'webgpu' && c.precision === 'FP16').available);
  assert.ok(browser.configs.filter(c => c.model === model && c.precision === 'INT8').every(c => c.available));
  assert.equal(manifest.artifacts[`${model}_INT8`].input_dtype,'float32');
  assert.equal(manifest.artifacts[`${model}_INT8`].calibration_images,128);
}
for (const model of Object.keys(browser.models).filter(model => !['yolo26m','yolo26s'].includes(model))) {
  assert.ok(browser.configs.filter(c => c.model === model && c.precision === 'INT8').every(c => c.available));
  assert.ok(browser.configs.filter(c => c.model === model && c.precision !== 'INT8').every(c => !c.available));
  assert.match(manifest.artifacts[`${model}_INT8`].url, /^onnx_models\/.+\.onnx$/);
}
assert.equal(browserCapabilities({...manifest,artifacts:{}}).configs.some(c => c.available),false);
assert.throws(() => browserCapabilities({schema_version:9}));
const gpu = {info:{isFallbackAdapter:false},features:new Set(['shader-f16'])};
assert.equal(webgpuReason({engine:'wasm'},null,false,false),null);
assert.match(webgpuReason({engine:'webgpu'},null,false,false),/HTTPS/);
assert.match(webgpuReason({engine:'webgpu'},null,true,false),/chưa cung cấp/);
assert.match(webgpuReason({engine:'webgpu'},null,true,true),/adapter/);
assert.match(webgpuReason({engine:'webgpu'}, {info:{isFallbackAdapter:true}},true,true),/software fallback/);
assert.match(webgpuReason({engine:'webgpu',precision:'FP16'},{...gpu,features:new Set()},true,true),/shader-f16/);
assert.equal(webgpuReason({engine:'webgpu',precision:'FP16'},gpu,true,true),null);
assert.equal(webgpuReason({engine:'webgpu',precision:'INT8'},{...gpu,features:new Set()},true,true),null);
assert.equal(isFallback({info:{isFallbackAdapter:true}}),true);
assert.equal(isFallback({isFallbackAdapter:true}),true);
assert.equal(isFallback({info:{isFallbackAdapter:false},isFallbackAdapter:true}),false);
for (const value of [0,-0,1,-1,.5,114/255,65504,2**-24,Infinity,-Infinity]) {
  const restored = halfToFloat(floatToHalf(value));
  if (Number.isFinite(value)) assert.ok(Math.abs(restored-value) <= Math.max(1e-7,Math.abs(value)*.001));
  else assert.equal(restored,value);
}
assert.ok(Number.isNaN(halfToFloat(floatToHalf(NaN))));
assert.deepEqual(geometry(1280,720),{ratio:.5,nw:640,nh:360,left:0,top:140,w:1280,h:720});
const raw = new Float32Array(1800); raw.set([50,190,100,240,.8,2]);
const boxes = decode(raw,[1,300,6],geometry(1280,720),.25,false);
assert.equal(boxes.length,1); assert.deepEqual(boxes[0].box_px,[100,100,200,200]);
assert.equal(boxes[0].class_name,'car');
assert.equal(decode(Uint16Array.from(raw,floatToHalf),[1,300,6],geometry(1280,720),.25,true).length,1);
const rawHead = new Float32Array(14 * 8400);
function prediction(index, score, classId) {
  [100,100,40,20].forEach((value, field) => { rawHead[field * 8400 + index] = value; });
  rawHead[(4 + classId) * 8400 + index] = score;
}
prediction(0,.9,2); prediction(1,.8,2); prediction(2,.7,4);
const decodedHead = decode(rawHead,[1,14,8400],geometry(640,640),.25,false);
assert.equal(decodedHead.length,2, 'Class-aware NMS removes the weaker duplicate only');
assert.deepEqual(decodedHead[0].box_px,[80,90,120,110]);
assert.deepEqual(decodedHead.map(box => box.class_name),['car','bus']);
assert.throws(() => decode(raw,[1,84,8400],geometry(1280,720),.25,false));
const stats = statistics([1,2,3,4,5]); assert.equal(stats.p50_ms,3); assert.equal(stats.p95_ms,4.8); assert.equal(stats.fps,1000/3);
console.log('Hardware browser preprocessing, precision, decoding and statistics passed.');
