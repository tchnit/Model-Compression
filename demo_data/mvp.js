function mvpBuildBDD100KJson(boxes, scene, width, height) {
  return {
    name: scene.file,
    attributes: scene.attributes || {},
    labels: boxes.map((box, index) => ({
      id: index,
      category: ({person: 'pedestrian', motor: 'motorcycle', bike: 'bicycle'})[box.class_name] || box.class_name,
      manualShape: false,
      manualAttributes: false,
      box2d: {
        x1: Math.round(box.box_px ? box.box_px[0] : box.box_norm[0] * width),
        y1: Math.round(box.box_px ? box.box_px[1] : box.box_norm[1] * height),
        x2: Math.round(box.box_px ? box.box_px[2] : box.box_norm[2] * width),
        y2: Math.round(box.box_px ? box.box_px[3] : box.box_norm[3] * height)
      },
      confidence: box.conf
    }))
  };
}

if (typeof module !== 'undefined') module.exports = {mvpBuildBDD100KJson};

(() => {
  'use strict';
  if (typeof document === 'undefined') return;

  const DEFAULT_MODEL_KEY = 'yolo26s_ptq_int8';
  const MODEL_NAMES = {
    yolo26s_ptq_int8: 'YOLO26s PTQ INT8 640 (OpenVINO)',
    yolo26s_baseline_fp32: 'YOLO26s Baseline (FP32)',
    yolo26m_baseline_fp32: 'YOLO26m Baseline (FP32)'
  };
  const MODEL_SOURCES = {yolo26s_baseline_fp32: 'yolo26s_fp32', yolo26m_baseline_fp32: 'yolo26m_fp32'};
  const MODEL_SIZES = {yolo26s_baseline_fp32: 36.56, yolo26m_baseline_fp32: 78.13};
  const root = document.getElementById('mvp');
  root.innerHTML = `
    <div class="studio-container mvp-single">
      <aside class="studio-sidebar">
        <div class="control-block">
          <div class="control-title"><span>1. Tình Huống Giao Thông</span><span id="mvpSceneBadge" class="mvp-accent"></span></div>
          <div class="scene-list" id="mvpSceneList"></div>
          <label class="upload-box">
            <input type="file" id="mvpImageUploader" accept="image/jpeg,image/png,image/webp" hidden>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
            <div><span>Tải lên ảnh của bạn</span></div>
          </label>
        </div>

        <div class="control-block">
          <div class="control-title">2. Mô Hình Nhận Diện</div>
          <div class="model-select-group"><div class="select-wrapper">
            <span class="select-label"><span>MÔ HÌNH</span><span id="mvpModelInfo">9.99 MiB</span></span>
            <select class="model-select" id="mvpModelSelect" aria-label="Mô hình nhận diện">
              <option value="yolo26s_ptq_int8" selected>YOLO26s PTQ INT8 640 (OpenVINO)</option>
              <option value="yolo26s_baseline_fp32">YOLO26s Baseline (FP32)</option>
              <option value="yolo26m_baseline_fp32">YOLO26m Baseline (FP32)</option>
            </select>
            <span class="select-label"><span>CPU SUY LUẬN</span></span>
            <select class="model-select" id="mvpProcessorSelect">
              <option value="host" id="mvpProcessorHost" selected>Đang kiểm tra CPU…</option>
              <option value="huggingface">Intel(R) Xeon(R) Platinum 8559C</option>
            </select>
            <span class="select-label"><span>SỐ CORE CPU</span></span>
            <select class="model-select" id="mvpCoreSelect">
              <option value="1">CPU ×1</option><option value="2">CPU ×2</option><option value="3" selected>CPU ×3</option>
            </select>
          </div></div>
          <button class="btn btn-primary run-detection-btn" id="mvpRunDetection" type="button">Nhận diện</button>
        </div>

        <div class="control-block">
          <div class="control-title"><span>3. Ngưỡng Tự Tin (Conf)</span><span id="mvpConfValue">0.25</span></div>
          <div class="range-container"><input type="range" id="mvpConf" min="0.10" max="0.80" step="0.05" value="0.25">
            <div class="range-row"><small>0.10 (Bao quát)</small><small>0.80 (Chặt chẽ)</small></div>
          </div>
        </div>

        <div class="control-block">
          <div class="control-title"><span>4. Lọc Lớp Đối Tượng</span><button class="btn btn-sm" id="mvpToggleClasses" type="button">Tất cả</button></div>
          <div class="class-filter-grid" id="mvpClassFilters"></div>
        </div>
      </aside>

      <div class="studio-stage">
        <div class="stage-toolbar"><div><h2 id="mvpSceneTitle"></h2><span id="mvpSceneDesc"></span></div></div>
        <div class="canvas-viewport">
          <div class="ai-loading-overlay" id="mvpLoading"><div class="scanner-laser"></div><div class="ai-spinner"></div>
            <strong>Đang chạy mô hình AI trực tiếp...</strong><small id="mvpLoadingText">YOLO26s PTQ INT8 · OpenVINO</small>
          </div>
          <div class="split-view-container">
            <div class="canvas-panel">
              <div class="panel-model-badge"><span class="mvp-model-dot"></span><span id="mvpPanelModel"></span></div>
              <div class="image-stage-wrap"><img class="source-image" id="mvpImage" alt="Ảnh nhận diện"><svg class="detection-overlay" id="mvpOverlay"></svg></div>
              <div class="panel-stat-pill">Phát hiện: <strong id="mvpBoxCount">—</strong> hộp · <span id="mvpLatency">Chưa suy luận</span></div>
            </div>
          </div>
        </div>

        <div class="stage-telemetry mvp-telemetry">
          <div class="telemetry-card"><span class="telemetry-label">Tốc độ mô hình</span><div class="telemetry-primary" id="mvpFps">—</div><div class="telemetry-secondary" id="mvpSpeed">Chưa đo live</div></div>
          <div class="telemetry-card"><span class="telemetry-label">mAP@50</span><div class="telemetry-primary" id="mvpMap50">51.44%</div><div class="telemetry-secondary">BDD100K validation</div></div>
          <div class="telemetry-card"><span class="telemetry-label">Mô hình đang chạy</span><div class="telemetry-secondary" id="mvpRuntime">YOLO26s PTQ INT8 · OpenVINO · CPU ×3</div></div>
        </div>

        <div class="export-card">
          <div><h3>Tác Vụ Tiền Gán Nhãn Tự Động (Auto Pre-Labeling)</h3>
            <p>Kiểm tra nhãn trước khi sử dụng.</p>
            <div class="mvp-export-actions"><button class="btn btn-outline-primary" id="mvpCopy" disabled>Sao chép JSON BDD100K</button>
              <button class="btn" id="mvpDownloadJson" disabled>Tải file .json</button><button class="btn" id="mvpDownloadYolo" disabled>Tải file YOLO .txt</button></div>
          </div>
          <div><div class="mvp-preview-title"><span>XEM TRƯỚC SCHEMA BDD100K EXPORT</span><span id="mvpExportCount">Chưa có kết quả</span></div><div class="json-preview" id="mvpJsonPreview"></div></div>
        </div>
      </div>
    </div>`;

  const byId = id => document.getElementById(id);
  const data = window.DEMO_DATA;
  if (!data?.scenes?.length || Object.keys(MODEL_NAMES).some(key => !data.models?.[MODEL_SOURCES[key] || key])) return;
  const scenes = data.scenes.map(scene => ({...scene, mvpDetections: {}}));
  let currentSceneId = scenes[0].id;
  let currentModelKey = DEFAULT_MODEL_KEY;
  let currentThreads = 3;
  let currentProcessor = 'host';
  let currentConf = 0.25;
  let serverOnline = false;
  let serverCpuThreads = null;
  let serverInt8CpuThreads = null;
  let huggingFaceCpuThreads = 3;
  let inferenceQueue = Promise.resolve();
  const pending = new Set();
  const enabledClasses = new Set(Object.keys(CLASS_COLORS));

  function route() {
    if (!location.hash) history.replaceState(null, '', '#mvp');
    const active = location.hash === '#mvp';
    document.body.classList.toggle('mvp-active', active);
    document.querySelectorAll('.nav-links a').forEach(link => {
      link.classList.toggle('active', link.hash === location.hash);
      if (link.hash === location.hash) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    if (!active) requestAnimationFrame(() => document.getElementById(location.hash.slice(1))?.scrollIntoView({block: 'start'}));
  }

  function toast(message) {
    const box = byId('toastBox');
    box.textContent = message;
    box.classList.add('show');
    setTimeout(() => box.classList.remove('show'), 3000);
  }

  function scene() { return scenes.find(item => item.id === currentSceneId) || scenes[0]; }
  function signature() { return currentProcessor === 'host' ? `${currentModelKey}:${currentThreads}` : `${currentModelKey}:huggingface:${currentThreads}`; }
  function record() { return scene().mvpDetections[signature()]; }
  function filteredBoxes() { return (record()?.boxes || []).filter(box => box.conf >= currentConf && enabledClasses.has(box.class_name)); }

  function populateCoreSelect() {
    const limit = currentProcessor === 'huggingface' ? huggingFaceCpuThreads
      : currentModelKey.includes('int8') ? serverInt8CpuThreads : serverCpuThreads;
    byId('mvpCoreSelect').innerHTML = Array.from({length: limit}, (_, index) => `<option value="${index + 1}">CPU ×${index + 1}</option>`).join('');
    byId('mvpCoreSelect').value = String(currentThreads);
  }

  async function checkServer() {
    try {
      const response = await fetch('api/status', {headers: {'Accept':'application/json'}});
      const status = await readApiJson(response);
      if (!response.ok || status.status !== 'online') throw new Error('offline');
      serverOnline = true;
      serverCpuThreads = Number.isInteger(status.cpu_threads) && status.cpu_threads > 0 ? status.cpu_threads : null;
      serverInt8CpuThreads = Number.isInteger(status.int8_cpu_threads) && status.int8_cpu_threads > 0 ? status.int8_cpu_threads : serverCpuThreads;
      huggingFaceCpuThreads = Number.isInteger(status.huggingface_cpu_threads) && status.huggingface_cpu_threads > 0
        ? status.huggingface_cpu_threads : 3;
      if (serverCpuThreads) {
        currentThreads = Math.min(3, currentModelKey.includes('int8') ? serverInt8CpuThreads : serverCpuThreads);
        populateCoreSelect();
      }
      byId('mvpProcessorHost').textContent = status.cpu_name || 'CPU không rõ tên';
      render();
      return inferScene(scene());
    } catch (_) {
      serverOnline = false;
      byId('mvpProcessorHost').textContent = 'CPU không kết nối';
      render();
      return undefined;
    }
  }

  function renderSceneList() {
    byId('mvpSceneList').replaceChildren(...scenes.map(item => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `scene-item ${item.id === currentSceneId ? 'active' : ''}`;
      button.addEventListener('click', () => selectScene(item.id));
      const image = document.createElement('img');
      image.className = 'scene-thumb'; image.src = item.data_uri || item.image_url; image.alt = item.title;
      const info = document.createElement('div'); info.className = 'scene-info';
      const name = document.createElement('div'); name.className = 'scene-name'; name.textContent = item.title;
      const tag = document.createElement('div'); tag.className = 'scene-tag'; tag.textContent = item.tag || 'Ảnh tải lên';
      info.append(name, tag); button.append(image, info); return button;
    }));
    byId('mvpSceneBadge').textContent = `${scenes.length} ảnh`;
  }

  function render() {
    const current = scene();
    const result = record();
    const boxes = filteredBoxes();
    const model = data.models[MODEL_SOURCES[currentModelKey] || currentModelKey];
    const modelName = MODEL_NAMES[currentModelKey];
    const image = byId('mvpImage');
    const source = current.data_uri || current.image_url;
    if (image.src !== source) image.src = source;
    image.onload = () => {
      byId('mvpOverlay').setAttribute('viewBox', `0 0 ${image.naturalWidth} ${image.naturalHeight}`);
      render();
    };
    const width = image.naturalWidth || current.width || 1280;
    const height = image.naturalHeight || current.height || 720;
    byId('mvpOverlay').setAttribute('viewBox', `0 0 ${width} ${height}`);
    byId('mvpOverlay').innerHTML = generateSvgElements(boxes, width, height);
    byId('mvpSceneTitle').textContent = current.title;
    byId('mvpSceneDesc').textContent = current.desc || '';
    const sizeMb = MODEL_SIZES[currentModelKey] || model.size_mb;
    byId('mvpModelInfo').textContent = `${sizeMb.toFixed(2)} MiB`;
    byId('mvpPanelModel').textContent = modelName;
    const runtime = (result?.runtime || `${modelName} · CPU ×${currentThreads}`).replace(' · CPU ×', '\nCPU ×');
    byId('mvpRuntime').textContent = runtime;
    byId('mvpBoxCount').textContent = result ? boxes.length : '—';
    byId('mvpLatency').textContent = result ? `${result.inference_time_ms.toFixed(1)} ms\n${runtime}` : serverOnline ? 'Chưa suy luận' : 'AI Server offline';
    const fps = result?.inference_time_ms > 0 ? 1000 / result.inference_time_ms : null;
    byId('mvpFps').textContent = fps ? `${fps.toFixed(1)} FPS` : '—';
    byId('mvpSpeed').textContent = result ? `${result.inference_time_ms.toFixed(1)} ms / ảnh` : 'Chưa đo live';
    byId('mvpMap50').textContent = Number.isFinite(model.map50) ? `${model.map50.toFixed(2)}%` : 'Chưa đo';
    const busy = pending.has(`${current.id}:${signature()}`);
    byId('mvpLoading').classList.toggle('active', busy);
    byId('mvpRunDetection').disabled = busy || !serverOnline;
    byId('mvpCoreSelect').disabled = busy;
    byId('mvpProcessorSelect').disabled = busy;
    byId('mvpLoadingText').textContent = `${modelName} · CPU ×${currentThreads}`;
    renderExport(boxes, current, width, height, !!result);
  }

  function renderExport(boxes, current, width, height, hasResult) {
    for (const id of ['mvpCopy', 'mvpDownloadJson', 'mvpDownloadYolo']) byId(id).disabled = !hasResult;
    byId('mvpExportCount').textContent = hasResult ? `${boxes.length} đối tượng` : 'Chưa có kết quả';
    byId('mvpJsonPreview').textContent = hasResult ? JSON.stringify(mvpBuildBDD100KJson(boxes, current, width, height), null, 2) : 'Bấm Nhận diện để chạy mô hình và xuất nhãn.';
  }

  function inferScene(target, force = false) {
    if (!serverOnline) { toast('AI Server đang offline.'); return Promise.resolve(); }
    const modelKey = currentModelKey;
    const processor = currentProcessor;
    const requestSignature = processor === 'host' ? `${modelKey}:${currentThreads}` : `${modelKey}:huggingface:${currentThreads}`;
    const requestKey = `${target.id}:${requestSignature}`;
    if (!force && target.mvpDetections[requestSignature]) { render(); return Promise.resolve(); }
    if (pending.has(requestKey)) return inferenceQueue;
    pending.add(requestKey); render();
    const threads = currentThreads;
    const request = inferenceQueue.then(async () => {
      try {
        const response = await fetch('api/predict', {
          method: 'POST', headers: {'Accept':'application/json','Content-Type':'application/json'},
          body: JSON.stringify({image: target.data_uri, conf: 0.10, models: [modelKey], threads, processor})
        });
        const payload = await readApiJson(response);
        if (!response.ok || payload.status !== 'success') throw new Error(payload.message || `HTTP ${response.status}`);
        const detection = payload.detections[modelKey];
        if (!detection) throw new Error('Server không trả về kết quả model đã chọn');
        target.mvpDetections[requestSignature] = {...detection, width: payload.width, height: payload.height};
      } catch (error) {
        toast(`Không thể chạy nhận diện: ${error.message}`);
      } finally {
        pending.delete(requestKey);
        if (target.id === currentSceneId) render();
      }
    });
    inferenceQueue = request.catch(() => {});
    return request;
  }

  function selectScene(id) {
    currentSceneId = id;
    renderSceneList(); render();
    return inferScene(scene());
  }

  function rerun() {
    delete scene().mvpDetections[signature()];
    render();
    return inferScene(scene(), true);
  }

  function download(content, type, filename) {
    const link = document.createElement('a');
    const url = URL.createObjectURL(new Blob([content], {type}));
    link.href = url; link.download = filename; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  function exportData(format) {
    const current = scene();
    const image = byId('mvpImage');
    const boxes = filteredBoxes();
    const width = image.naturalWidth || current.width || 1280;
    const height = image.naturalHeight || current.height || 720;
    if (format === 'json') {
      const content = JSON.stringify(mvpBuildBDD100KJson(boxes, current, width, height), null, 2);
      download(content, 'application/json', `prelabel_${current.file.replace(/\.[^.]+$/, '')}_${currentModelKey}.json`);
    } else {
      const lines = boxes.map(box => {
        const coords = box.box_norm || [box.box_px[0] / width, box.box_px[1] / height, box.box_px[2] / width, box.box_px[3] / height];
        const [x1, y1, x2, y2] = coords;
        return `${box.class_id} ${((x1 + x2) / 2).toFixed(6)} ${((y1 + y2) / 2).toFixed(6)} ${(x2 - x1).toFixed(6)} ${(y2 - y1).toFixed(6)}`;
      });
      download(lines.join('\n'), 'text/plain', current.file.replace(/\.[^.]+$/, '') + '.txt');
    }
  }

  function upload(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onerror = () => toast('Không thể đọc ảnh đã chọn.');
    reader.onload = loaded => {
      const image = new Image();
      image.onerror = () => toast('Không thể giải mã ảnh đã chọn.');
      image.onload = () => {
        const uploaded = {id: `mvp_custom_${Date.now()}`, title: file.name.slice(0, 30), file: file.name,
          desc: `Ảnh người dùng (${image.naturalWidth}×${image.naturalHeight})`, tag: 'Ảnh tải lên',
          data_uri: loaded.target.result, width: image.naturalWidth, height: image.naturalHeight, mvpDetections: {}};
        scenes.unshift(uploaded); selectScene(uploaded.id);
      };
      image.src = loaded.target.result;
    };
    reader.readAsDataURL(file);
  }

  async function init() {
    byId('mvpModelSelect').value = currentModelKey;
    byId('mvpModelSelect').addEventListener('change', event => {
      currentModelKey = event.target.value;
      const limit = currentProcessor === 'huggingface' ? huggingFaceCpuThreads
        : currentModelKey.includes('int8') ? serverInt8CpuThreads : serverCpuThreads;
      if (limit) { currentThreads = Math.min(currentThreads, limit); populateCoreSelect(); }
      render(); inferScene(scene());
    });
    byId('mvpProcessorSelect').addEventListener('change', event => {
      currentProcessor = event.target.value;
      const limit = currentProcessor === 'huggingface' ? huggingFaceCpuThreads
        : currentModelKey.includes('int8') ? serverInt8CpuThreads : serverCpuThreads;
      if (limit) { currentThreads = Math.min(currentThreads, limit); populateCoreSelect(); }
      render(); inferScene(scene());
    });
    byId('mvpClassFilters').innerHTML = Object.keys(CLASS_COLORS).map(name => `<label class="class-check"><input type="checkbox" value="${name}" checked><span class="class-color-dot" style="background:${CLASS_COLORS[name]}"></span><span>${name}</span></label>`).join('');
    byId('mvpClassFilters').addEventListener('change', event => {
      if (!event.target.value) return;
      event.target.checked ? enabledClasses.add(event.target.value) : enabledClasses.delete(event.target.value);
      render();
    });
    byId('mvpToggleClasses').addEventListener('click', () => {
      const enable = enabledClasses.size !== Object.keys(CLASS_COLORS).length;
      enabledClasses.clear();
      document.querySelectorAll('#mvpClassFilters input').forEach(input => { input.checked = enable; if (enable) enabledClasses.add(input.value); });
      render();
    });
    byId('mvpCoreSelect').addEventListener('change', event => {
      currentThreads = Number(event.target.value); render(); inferScene(scene());
    });
    byId('mvpConf').addEventListener('input', event => {
      currentConf = Number(event.target.value); byId('mvpConfValue').textContent = currentConf.toFixed(2); render();
    });
    byId('mvpRunDetection').addEventListener('click', rerun);
    byId('mvpImageUploader').addEventListener('change', upload);
    byId('mvpCopy').addEventListener('click', () => navigator.clipboard.writeText(byId('mvpJsonPreview').textContent).then(() => toast('Đã sao chép nhãn BDD100K JSON.')).catch(() => toast('Không thể sao chép vào clipboard.')));
    byId('mvpDownloadJson').addEventListener('click', () => exportData('json'));
    byId('mvpDownloadYolo').addEventListener('click', () => exportData('yolo'));
    window.addEventListener('hashchange', route); route(); renderSceneList(); render();
    return checkServer();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
