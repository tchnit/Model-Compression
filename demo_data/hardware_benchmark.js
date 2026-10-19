/* Hardware microbenchmarks. No inference runs from selection/change listeners. */
(function () {
  'use strict';
  const CLASS_NAMES = ['person','rider','car','truck','bus','train','motor','bike','traffic light','traffic sign'];
  const floatView = new Float32Array(1), intView = new Uint32Array(floatView.buffer);
  function floatToHalf(value) {
    floatView[0] = value;
    const x = intView[0], sign = (x >>> 16) & 0x8000;
    let exp = ((x >>> 23) & 255) - 127 + 15, mant = x & 0x7fffff;
    if (exp >= 31) return sign | (mant && ((x >>> 23) & 255) === 255 ? 0x7e00 : 0x7c00);
    if (exp <= 0) {
      if (exp < -10) return sign;
      mant |= 0x800000;
      const shift = 14 - exp;
      return sign | ((mant + (1 << (shift - 1)) - 1 + ((mant >>> shift) & 1)) >>> shift);
    }
    mant += 0xfff + ((mant >>> 13) & 1);
    if (mant & 0x800000) { mant = 0; exp++; }
    return sign | (exp << 10) | (mant >>> 13);
  }
  function halfToFloat(x) {
    const sign = (x & 0x8000) ? -1 : 1, exp = (x >>> 10) & 31, mant = x & 1023;
    return exp === 31 ? (mant ? NaN : sign * Infinity) : sign * (exp ? (1 + mant / 1024) * 2 ** (exp - 15) : mant * 2 ** -24);
  }
  function geometry(w, h) {
    const ratio = Math.min(640 / w, 640 / h);
    // Python round is ties-to-even; source image sizes can hit exact half pixels.
    const roundEven = x => x % 1 === .5 ? 2 * Math.round(x / 2) : Math.round(x);
    const nw = roundEven(w * ratio), nh = roundEven(h * ratio);
    return {ratio, nw, nh, left:Math.max(0,Math.round((640 - nw) / 2 - .1)), top:Math.max(0,Math.round((640 - nh) / 2 - .1)), w, h};
  }
  function overlap(a, b) {
    const intersection = Math.max(0, Math.min(a.box_px[2], b.box_px[2]) - Math.max(a.box_px[0], b.box_px[0]))
      * Math.max(0, Math.min(a.box_px[3], b.box_px[3]) - Math.max(a.box_px[1], b.box_px[1]));
    const area = box => Math.max(0, box.box_px[2] - box.box_px[0]) * Math.max(0, box.box_px[3] - box.box_px[1]);
    const union = area(a) + area(b) - intersection;
    return union > 0 ? intersection / union : 0;
  }
  function decode(data, dims, t, conf, fp16) {
    const val = i => fp16 ? halfToFloat(data[i]) : data[i];
    if (dims.join(',') === '1,300,6') {
      const boxes = [];
      for (let i = 0; i < 1800; i += 6) {
        const score = val(i + 4), cls = val(i + 5);
        if (!Number.isFinite(score) || score < conf || score <= 0) continue;
        if (!Number.isInteger(cls) || cls < 0 || cls >= 10) throw new Error('Class ID không thuộc BDD100K.');
        const box = [0,1,2,3].map(j => Math.max(0, Math.min(j % 2 ? t.h : t.w,
          (val(i + j) - (j % 2 ? t.top : t.left)) / t.ratio)));
        if (box.some(x => !Number.isFinite(x))) continue;
        boxes.push({class_id:cls, class_name:CLASS_NAMES[cls], conf:score, box_px:box});
      }
      return boxes;
    }
    if (dims.join(',') !== '1,14,8400') throw new Error('Model không trả về YOLO26 [1,300,6] hoặc [1,14,8400].');
    const candidates = [];
    for (let i = 0; i < 8400; i++) {
      let cls = 0, score = val(4 * 8400 + i);
      for (let classId = 1; classId < 10; classId++) {
        const classScore = val((4 + classId) * 8400 + i);
        if (classScore > score) { score = classScore; cls = classId; }
      }
      if (!Number.isFinite(score) || score < conf || score <= 0) continue;
      const cx = val(i), cy = val(8400 + i), width = val(2 * 8400 + i), height = val(3 * 8400 + i);
      const letterbox = [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2];
      const box = letterbox.map((value, index) => Math.max(0, Math.min(index % 2 ? t.h : t.w,
        (value - (index % 2 ? t.top : t.left)) / t.ratio)));
      if (box.some(value => !Number.isFinite(value)) || box[2] <= box[0] || box[3] <= box[1]) continue;
      candidates.push({class_id:cls, class_name:CLASS_NAMES[cls], conf:score, box_px:box});
    }
    candidates.sort((a, b) => b.conf - a.conf);
    const boxes = [];
    for (const candidate of candidates) {
      if (!boxes.some(box => box.class_id === candidate.class_id && overlap(box, candidate) > 0.7)) boxes.push(candidate);
      if (boxes.length === 300) break;
    }
    return boxes;
  }
  function statistics(samples) {
    const sorted = [...samples].sort((a,b) => a-b), mean = samples.reduce((a,b) => a+b,0) / samples.length;
    const percentile = p => { const i = (sorted.length-1)*p, low = Math.floor(i); return sorted[low]+(sorted[Math.ceil(i)]-sorted[low])*(i-low); };
    return {mean_ms:mean, p50_ms:percentile(.5), p95_ms:percentile(.95), min_ms:sorted[0], max_ms:sorted.at(-1), fps:1000/mean};
  }
  const isFallback = adapter => adapter?.info?.isFallbackAdapter ?? adapter?.isFallbackAdapter ?? null;
  const isBrowserEngine = engine => engine === 'wasm' || engine === 'webgpu';
  const runtimeName = engine => ({wasm:'ONNX Runtime Web · CPU (WASM)',webgpu:'ONNX Runtime Web · GPU (WebGPU)'})[engine] || engine;
  function browserCapabilities(manifest) {
    const configs = [];
    if (manifest.schema_version !== 1 || !manifest.models || !manifest.artifacts) throw new Error('Danh sách ONNX trình duyệt không hợp lệ.');
    for (const model of Object.keys(manifest.models)) for (const engine of ['wasm','webgpu']) for (const precision of ['FP32','FP16','INT8']) {
      const artifact = manifest.artifacts[`${model}_${precision}`];
      const reason = !artifact ? `Chưa có ONNX ${precision} cho cấu hình này.` : null;
      configs.push({model,engine,device:engine === 'wasm' ? 'browser-cpu' : 'browser-gpu',precision,
        ...artifact,available:!reason,reason});
    }
    return {models:manifest.models,configs};
  }
  function webgpuReason(config, adapter, secureContext, hasWebGPU) {
    if (config?.engine !== 'webgpu') return null;
    if (!secureContext) return 'WebGPU cần HTTPS hoặc localhost. Trang hiện tại không có secure context.';
    if (!hasWebGPU) return 'Trình duyệt này chưa cung cấp WebGPU. Mở trang bằng Chrome/Edge có hỗ trợ WebGPU.';
    if (!adapter) return 'Không lấy được WebGPU adapter. Kiểm tra GPU, driver và tăng tốc phần cứng của trình duyệt; CPU (WASM) vẫn dùng được.';
    if (isFallback(adapter)) return 'Adapter WebGPU là software fallback; không gắn nhãn GPU phần cứng.';
    if (config.precision === 'FP16' && !adapter.features.has('shader-f16')) return 'GPU trình duyệt không hỗ trợ shader-f16; chọn FP32.';
    return null;
  }
  function wasmThreadLimit(logicalCPUs, isolated, sharedMemory) {
    return isolated && sharedMemory && Number.isInteger(logicalCPUs) && logicalCPUs > 0 ? logicalCPUs : 1;
  }
  function previewMatches(preview, choice, image, conf, threads = 1) {
    return !!preview && preview.image === image && preview.result.conf === conf
      && (choice.engine !== 'wasm' || (preview.result.protocol?.cpu_threads ?? 1) === threads)
      && ['model','engine','device','precision'].every(key => preview.result[key] === choice[key]);
  }
  async function hash(buffer, subtle = globalThis.crypto?.subtle) {
    // Plain HTTP LAN pages do not expose SubtleCrypto; the server verifies tracked artifacts in tests.
    return subtle ? Array.from(new Uint8Array(await subtle.digest('SHA-256',buffer)),b => b.toString(16).padStart(2,'0')).join('') : null;
  }
  async function browserBenchmark(ort, c, imageURI, protocol, adapter, status, baseURL) {
    ort.env.wasm.wasmPaths = new URL('demo_data/vendor/onnxruntime-web/', baseURL).href;
    ort.env.wasm.numThreads = c.engine === 'wasm' ? protocol.cpu_threads : 1;
    const inWorker = typeof document === 'undefined';
    let image;
    if (inWorker) image = await createImageBitmap(await (await fetch(imageURI)).blob());
    else { image = new Image(); image.src = imageURI; await image.decode(); }
    const width = image.naturalWidth || image.width, height = image.naturalHeight || image.height;
    if (width * height > 20000000) throw new Error('Ảnh có quá nhiều pixel.');
    const transform = geometry(width, height);
    const canvas = inWorker ? new OffscreenCanvas(640,640) : document.createElement('canvas'); canvas.width = canvas.height = 640;
    const ctx = canvas.getContext('2d', {willReadFrequently:true});
    const fp16 = c.precision === 'FP16', setupStart = performance.now();
    const modelResponse = await fetch(new URL(c.url,baseURL));
    if (!modelResponse.ok) throw new Error('Không tải được ONNX artifact.');
    const modelBytes = await modelResponse.arrayBuffer();
    const artifactHash = await hash(modelBytes);
    if (artifactHash && artifactHash !== c.sha256) throw new Error('SHA256 artifact không trùng manifest; quét lại cấu hình.');
    // Bind the measured device to the adapter shown above, including on hybrid-GPU laptops.
    const gpuDevice = c.engine === 'webgpu' ? await adapter.requestDevice({
      requiredFeatures:[...adapter.features], requiredLimits:{
        maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize:adapter.limits.maxBufferSize,
        maxComputeWorkgroupStorageSize:adapter.limits.maxComputeWorkgroupStorageSize}
    }) : null;
    let session;
    try {
      if (gpuDevice) ort.env.webgpu.adapter = adapter;
      session = await ort.InferenceSession.create(modelBytes, {executionProviders:[gpuDevice ? {name:'webgpu',device:gpuDevice} : 'wasm'], graphOptimizationLevel:'all'});
    } catch (error) { gpuDevice?.destroy(); throw error; }
    const load_ms = performance.now() - setupStart;
    const inputName = session.inputNames[0], outputName = session.outputNames[0];
    let outputType;
    async function once() {
      ctx.fillStyle = 'rgb(114,114,114)'; ctx.fillRect(0,0,640,640);
      ctx.drawImage(image,transform.left,transform.top,transform.nw,transform.nh);
      const rgba = ctx.getImageData(0,0,640,640).data;
      const data = fp16 ? new Uint16Array(3*640*640) : new Float32Array(3*640*640);
      for (let i=0; i<640*640; i++) for (let ch=0; ch<3; ch++) {
        const value = rgba[i*4+ch]/255; data[ch*640*640+i] = fp16 ? floatToHalf(value) : value;
      }
      const input = new ort.Tensor(fp16 ? 'float16' : 'float32', data, [1,3,640,640]);
      let outputs;
      try {
        outputs = await session.run({[inputName]:input});
        const output = outputs[outputName];
        outputType = output.type;
        const values = await output.getData(); // Materialize on CPU before stopping the timer.
        return decode(values,output.dims,transform,protocol.conf,output.type === 'float16');
      } finally { input.dispose(); if (outputs) Object.values(outputs).forEach(t => t.dispose()); }
    }
    try {
      for (let i=0;i<protocol.warmup;i++) { status(`Warmup trình duyệt ${i+1}/${protocol.warmup}…`); await once(); }
      const samples_ms = []; let boxes;
      for (let i=0;i<protocol.repeats;i++) {
        status(`Đang đo trình duyệt ${i+1}/${protocol.repeats}…`);
        const start = performance.now(); boxes = await once(); samples_ms.push(performance.now()-start);
        if (!inWorker) await new Promise(resolve => requestAnimationFrame(resolve));
      }
      const rawImage = await (await fetch(imageURI)).arrayBuffer();
      return {status:'success', timestamp:new Date().toISOString(), location:'browser', model:c.model,engine:c.engine,device:c.device,precision:c.precision, runtime_name:runtimeName(c.engine),
        warmup:protocol.warmup,repeats:protocol.repeats,conf:protocol.conf,
        ...statistics(samples_ms), samples_ms, load_ms, boxes, detections_count:boxes.length,
        image_sha256:await hash(rawImage), runtime:{artifact:c.artifact,sha256:artifactHash || c.sha256,sha256_verified:!!artifactHash,source_sha256:c.source_sha256,version:ort.env.versions,
          graph_precision:c.precision,input_dtype:fp16 ? 'float16' : 'float32',output_dtype:outputType,output_format:c.output_format || 'end2end',
          precision_policy:c.precision_policy,calibration_sha256:c.calibration_sha256,
          calibration_images:c.calibration_images,quantized_convs:c.quantized_convs,
          device_name:c.engine === 'webgpu' ? (adapter?.info?.description || adapter?.info?.device || 'WebGPU adapter') : navigator.platform,
          provider:c.engine === 'webgpu' ? 'webgpu' : 'wasm', adapter:c.engine === 'webgpu' ? {vendor:adapter.info?.vendor,architecture:adapter.info?.architecture,device:adapter.info?.device,description:adapter.info?.description,isFallbackAdapter:isFallback(adapter),shader_f16:adapter.features.has('shader-f16')} : null},
        hardware:{userAgent:navigator.userAgent,logical_cpus:navigator.hardwareConcurrency},
        protocol:{input_shape:[1,3,640,640],batch:1,cpu_threads:ort.env.wasm.numThreads,cpu_threads_requested:protocol.cpu_threads,
          cross_origin_isolated:globalThis.crossOriginIsolated === true,
          scope:'preprocess + input transfer + inference + output transfer + postprocess',
          preprocessing:`${inWorker ? 'OffscreenCanvas' : 'Canvas'} drawImage / RGB / NCHW / divide255`,
          nms:c.output_format === 'raw_yolo' ? {class_aware:true,iou:0.7,max_detections:300} : false,
          provider_note:'WebGPU EP may use CPU kernels for unsupported nodes; not a kernel-only GPU timing.',
          accuracy:'single-image microbenchmark, not mAP'}};
    } finally { try { await session.release(); } finally { gpuDevice?.destroy(); image.close?.(); } }
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {floatToHalf, halfToFloat, geometry, decode, statistics, isFallback, browserCapabilities, webgpuReason, previewMatches, wasmThreadLimit, hash};
  if (typeof document === 'undefined') {
    if (typeof importScripts === 'function') self.onmessage = async ({data}) => {
      try {
        importScripts(new URL('demo_data/vendor/onnxruntime-web/ort.webgpu.min.js',data.baseURL).href);
        const result = await browserBenchmark(self.ort,data.config,data.image,data.protocol,null,
          status => self.postMessage({status}),data.baseURL);
        self.postMessage({result});
      } catch (error) { self.postMessage({error:error.message}); }
    };
    return;
  }
  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    const el = id => document.getElementById('hw' + id);
    if (!el('Run')) return;
    function updateNavigation() {
      document.querySelectorAll('.nav-link').forEach(link => link.classList.toggle('active',
        link.getAttribute('href') === (location.hash || '#studio')));
    }
    window.addEventListener('hashchange',updateNavigation);
    updateNavigation();
    let caps = null, adapter = null, uploaded = null, busy = false, ortPromise = null, refreshId = 0, preview = null;
    const records = [], scenes = window.DEMO_DATA?.scenes || [];
    function option(value, label) { const o = document.createElement('option'); o.value = value; o.textContent = label; return o; }
    let sharedMemory = false;
    try { sharedMemory = typeof SharedArrayBuffer !== 'undefined' && new WebAssembly.Memory({initial:1,maximum:1,shared:true}).buffer instanceof SharedArrayBuffer; } catch { /* Single-thread WASM remains available. */ }
    const maxThreads = wasmThreadLimit(navigator.hardwareConcurrency,window.crossOriginIsolated,sharedMemory);
    el('Threads').replaceChildren(...Array.from({length:maxThreads},(_,i) => option(String(i+1),`${i+1} luồng`)));
    scenes.forEach(s => el('Image').append(option(s.id, s.title || s.label || s.name || s.file)));
    function currentImage() { return el('Image').value === 'upload' ? uploaded : scenes.find(s => s.id === el('Image').value)?.data_uri; }
    function choice() { return {model:el('Model').value, engine:el('Engine').value, device:el('Device').value, precision:el('Precision').value}; }
    function selected() { const c = choice(); return caps?.configs.find(row => Object.keys(c).every(k => row[k] === c[k])); }
    function browserReason(c) {
      return webgpuReason(c,adapter,window.isSecureContext,!!navigator.gpu);
    }
    function renderPreview() {
      const image = el('Preview'), overlay = el('PreviewBoxes'), uri = currentImage();
      overlay.replaceChildren();
      if (image.getAttribute('src') !== (uri || '')) image.src = uri || '';
      el('PreviewCaption').textContent = 'Ảnh đầu vào · Bấm Chạy benchmark để xem box phát hiện.';
      const conf = el('Conf').value === '' ? NaN : Number(el('Conf').value);
      if (!previewMatches(preview,choice(),uri,conf,Number(el('Threads').value)) || !image.complete || !image.naturalWidth) return;
      const result = preview.result;
      overlay.setAttribute('viewBox',`0 0 ${image.naturalWidth} ${image.naturalHeight}`);
      const boxes = result.boxes.map(box => ({...box,class_name:CLASS_NAMES[box.class_id] || 'unknown'}));
      overlay.innerHTML = generateSvgElements(boxes,image.naturalWidth,image.naturalHeight,
        image.naturalWidth / (image.clientWidth || image.naturalWidth));
      el('PreviewCaption').textContent = `${result.model} · ${result.runtime_name || result.engine} · ${result.precision} · CPU ${result.protocol.cpu_threads} luồng · ${boxes.length} hộp · conf ≥ ${result.conf}`;
    }
    function update() {
      const c = selected(), reason = !c ? 'Không có thiết bị cho runtime này.' : c.reason || browserReason(c);
      const wasm = el('Engine').value === 'wasm';
      el('Threads').disabled = busy || !wasm || maxThreads === 1;
      el('ThreadInfo').textContent = !wasm ? 'Số luồng này chỉ áp dụng cho CPU trình duyệt (WASM). WebGPU tự quản lý GPU; các runtime backend giữ cấu hình CPU 4 luồng.'
        : maxThreads > 1 ? `WASM dùng ${el('Threads').value} luồng; có thể chọn 1–${maxThreads} theo CPU trình duyệt. Chọn số luồng rồi bấm Chạy benchmark.`
        : 'WASM dùng 1 luồng. Đa luồng cần cross-origin isolation và WebAssembly shared memory; nếu dùng server khác, hãy mở trang qua run_demo.py.';
      const browserCPU = el('Device').querySelector('option[value="browser-cpu"]');
      if (browserCPU) browserCPU.textContent = `CPU trình duyệt · WASM ${el('Threads').value} luồng`;
      el('Run').disabled = busy || !c?.available || !!reason || !currentImage();
      renderPreview();
      const precisionState = ['FP32','FP16','INT8'].map(precision => {
        const config = caps?.configs.find(row => row.model === el('Model').value && row.engine === el('Engine').value && row.device === el('Device').value && row.precision === precision);
        return `${precision}: ${config?.available && !browserReason(config) ? 'sẵn sàng' : 'chưa hỗ trợ'}`;
      }).join(' · ');
      el('PrecisionInfo').textContent = precisionState + (c?.engine === 'wasm' && c.precision === 'FP16'
        ? '. FP16 là precision của graph; kernel WASM có thể tính một số phép toán bằng FP32.'
        : c?.precision === 'INT8' ? '. INT8 lượng tử hóa Conv; các lớp khác vẫn có thể dùng số thực.' : '');
      if (!busy) el('Status').textContent = reason || `${c.artifact} · ${c.device} · ${c.precision}. Sẵn sàng; bấm Chạy benchmark.`;
    }
    function updateDevices() {
      const old = el('Device').value;
      el('Device').replaceChildren();
      const devices = [...new Set((caps?.configs || []).filter(c => c.engine === el('Engine').value).map(c => c.device))];
      const labels = {cpu:'CPU backend · 4 luồng', cuda:'NVIDIA CUDA · GPU backend', 'browser-gpu':'GPU trình duyệt · WebGPU', 'browser-cpu':`CPU trình duyệt · WASM ${el('Threads').value} luồng`};
      devices.forEach(d => el('Device').append(option(d, labels[d] || `OpenVINO ${d}`)));
      if (devices.includes(old)) el('Device').value = old;
      update();
    }
    async function refresh() {
      const ticket = ++refreshId, browser = isBrowserEngine(el('Engine').value);
      el('Refresh').disabled = true;
      el('Run').disabled = true;
      el('Status').textContent = browser ? 'Đang kiểm tra CPU/WebGPU của trình duyệt…' : 'Đang đọc phần cứng backend…';
      caps = null;
      renderPreview();
      try {
        const response = await fetch(browser ? 'demo_data/browser_models.json' : '/api/benchmark/capabilities');
        const data = await response.json();
        if (!response.ok) throw new Error(data.message || 'Không đọc được danh sách model.');
        let nextAdapter = null;
        if (browser) try { nextAdapter = await navigator.gpu?.requestAdapter({powerPreference:'high-performance'}); } catch { /* Availability is reported below. */ }
        if (ticket !== refreshId) return;
        caps = browser ? browserCapabilities(data) : data;
        adapter = nextAdapter;
        const old = Object.hasOwn(caps.models,el('Model').value) ? el('Model').value : 'yolo26s';
        el('Model').replaceChildren(...Object.entries(caps.models).map(([id,label]) => option(id,label)));
        el('Model').value = old;
        if (browser) {
          const gpuReason = webgpuReason({engine:'webgpu',precision:'FP32'},adapter,window.isSecureContext,!!navigator.gpu);
          const platform = navigator.userAgentData?.platform || navigator.platform || 'không rõ nền tảng';
          el('Hardware').textContent = `CPU client (${platform}): ${navigator.hardwareConcurrency || '?'} luồng logic · WASM hỗ trợ chọn tối đa ${maxThreads} luồng. ${gpuReason || `WebGPU sẵn sàng: ${adapter.info?.description || adapter.info?.device || adapter.info?.vendor || 'GPU trình duyệt'}${adapter.features.has('shader-f16') ? ' · hỗ trợ FP16' : ' · FP32'}.`}`;
        } else {
          const h = caps.hardware;
          el('Hardware').textContent = `Máy backend Python: ${h.openvino_devices.CPU?.name || h.cpu} · ${h.logical_cpus} CPU logic · CUDA: ${h.gpu || 'không khả dụng'}.`;
        }
        updateDevices();
      } catch (error) { if (ticket === refreshId) { el('Status').textContent = error.message; el('Run').disabled = true; } }
      finally { if (ticket === refreshId) el('Refresh').disabled = false; }
    }
    function loadOrt() {
      if (!ortPromise) ortPromise = new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = 'demo_data/vendor/onnxruntime-web/ort.webgpu.min.js';
        script.onload = () => resolve(window.ort);
        script.onerror = () => { script.remove(); ortPromise = null; reject(new Error('Không tải được ONNX Runtime Web.')); };
        document.head.append(script);
      });
      return ortPromise;
    }
    function renderResults() {
      el('Results').replaceChildren();
      records.slice().reverse().forEach(r => {
        const row = document.createElement('tr');
        const cells = [`${r.model} · ${r.image_label}`,`${r.runtime_name || r.engine} · ${r.location} / ${r.device}`,r.precision,r.protocol.cpu_threads,
          `${r.p50_ms.toFixed(2)} ms`,`${r.p95_ms.toFixed(2)} ms`,r.fps.toFixed(1),r.detections_count,`${r.repeats} (+${r.warmup})`];
        cells.forEach(value => { const td = document.createElement('td'); td.textContent = value; row.append(td); });
        el('Results').append(row);
      });
      el('Export').disabled = records.length === 0;
    }
    el('Run').addEventListener('click', async () => {
      if (busy || el('Run').disabled) return;
      const c = selected(), image = currentImage(), imageLabel = el('Image').selectedOptions[0].textContent;
      const protocol = {warmup:Number(el('Warmup').value),repeats:Number(el('Repeats').value),conf:Number(el('Conf').value)};
      if (!Number.isInteger(protocol.warmup) || protocol.warmup<0 || protocol.warmup>10 || !Number.isInteger(protocol.repeats) || protocol.repeats<1 || protocol.repeats>50 || !Number.isFinite(protocol.conf) || protocol.conf<0 || protocol.conf>1 || ['Warmup','Repeats','Conf'].some(id => el(id).value === '')) {
        el('Status').textContent = 'Kiểm tra warmup 0–10, lượt đo 1–50 và confidence 0–1.'; return;
      }
      const threads = Number(el('Threads').value);
      if (c.engine === 'wasm' && (!Number.isInteger(threads) || threads < 1 || threads > maxThreads)) {
        el('Status').textContent = `Chọn số luồng từ 1 đến ${maxThreads}.`; return;
      }
      busy = true; el('Run').disabled = true; el('Refresh').disabled = true;
      preview = null; renderPreview();
      document.querySelectorAll('#hardware select, #hardware input').forEach(node => { node.disabled = true; });
      el('Status').textContent = 'Đang tải runtime / model và đo… (lần đầu có thể lâu)';
      try {
        let result;
        if (c.engine === 'wasm') result = await new Promise((resolve,reject) => {
          // WASM thread pools cannot be resized after initialization.
          const worker = new Worker(new URL('demo_data/hardware_benchmark.js',location.href));
          worker.onmessage = ({data}) => {
            if (data.status) { el('Status').textContent = data.status; return; }
            worker.terminate();
            if (data.error) reject(new Error(data.error)); else resolve(data.result);
          };
          worker.onerror = event => { worker.terminate(); reject(new Error(event.message || 'Không khởi tạo được worker WASM.')); };
          worker.postMessage({config:c,image,protocol:{...protocol,cpu_threads:Number(el('Threads').value)},baseURL:location.href});
        });
        else if (c.engine === 'webgpu') result = await browserBenchmark(await loadOrt(),c,image,{...protocol,cpu_threads:1},adapter,
          status => { el('Status').textContent = status; },location.href);
        else {
          const response = await fetch('/api/benchmark/run', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...choice(),...protocol,image})});
          result = await response.json(); if (!response.ok) throw new Error(result.message || 'Benchmark thất bại.');
        }
        result.image_label = imageLabel;
        preview = {image,result}; renderPreview();
        records.push(result); renderResults();
        el('Details').textContent = JSON.stringify(result,null,2);
        el('Status').textContent = `Hoàn tất: ${result.detections_count} hộp · CPU ${result.protocol.cpu_threads} luồng · p50 ${result.p50_ms.toFixed(2)} ms · p95 ${result.p95_ms.toFixed(2)} ms · tải/khởi tạo ${result.load_ms.toFixed(0)} ms (ngoài thời gian đo).`
          + (c.engine === 'wasm' && result.protocol.cpu_threads !== threads ? ` Runtime chỉ dùng được ${result.protocol.cpu_threads}/${threads} luồng đã chọn.` : '');
      } catch (error) { el('Status').textContent = `Không ghi kết quả: ${error.message}`; }
      finally {
        busy = false; document.querySelectorAll('#hardware select, #hardware input').forEach(node => { node.disabled = false; });
        el('Refresh').disabled = false;
        const status = el('Status').textContent;
        update(); el('Status').textContent = status;
      }
    });
    el('Export').addEventListener('click', () => {
      const url = URL.createObjectURL(new Blob([JSON.stringify({schema_version:1,records},null,2)],{type:'application/json'}));
      const a = document.createElement('a'); a.href=url; a.download='hardware-benchmark.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
    });
    el('Upload').addEventListener('change', async e => {
      const file = e.target.files[0]; if (!file) return;
      if (file.size>8*1024*1024) { el('Status').textContent='Ảnh tối đa 8 MiB.'; return; }
      try {
        uploaded = await new Promise((resolve,reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror=reject; reader.readAsDataURL(file); });
        const image = new Image(); image.src=uploaded; await image.decode();
        if (image.naturalWidth*image.naturalHeight>20000000) throw new Error('Ảnh có quá nhiều pixel.');
        el('Image').querySelector('option[value="upload"]')?.remove();
        el('Image').append(option('upload',file.name)); el('Image').value='upload'; update();
      } catch (error) {
        uploaded=null; el('Image').querySelector('option[value="upload"]')?.remove();
        update(); el('Status').textContent=error.message || 'Ảnh không hợp lệ.';
      }
    });
    el('Model').addEventListener('change', () => {
      if (isBrowserEngine(el('Engine').value) && !selected()?.available
          && caps?.configs.some(config => config.model === el('Model').value && config.engine === el('Engine').value
            && config.device === el('Device').value && config.precision === 'INT8' && config.available)) {
        el('Precision').value = 'INT8';
      }
      if (el('Model').value.endsWith('_qat')) { el('Engine').value='openvino'; el('Precision').value='INT8'; updateDevices(); } else update();
    });
    el('Engine').addEventListener('change',refresh);
    ['Device','Precision','Image','Threads'].forEach(id => el(id).addEventListener('change',update));
    el('Conf').addEventListener('input',renderPreview);
    el('Preview').addEventListener('load',renderPreview);
    window.addEventListener('resize',renderPreview);
    el('Refresh').addEventListener('click',refresh);
    await refresh();
  }
})();
