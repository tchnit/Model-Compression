
// Benchmark Data
const BENCHMARK_DATA = window.DEMO_DATA?.benchmarks || [];

// App State
let demoData = window.DEMO_DATA || null;
let currentSceneId = 'scene_01';
let currentModelA = '';
let currentModelB = '';
let currentThreadsA = 3;
let currentThreadsB = 3;
let currentConf = 0.25;
let currentViewMode = 'split'; // split, slider, toggle
let selectedToggleModel = 'A';
let enabledClasses = new Set(Object.keys(CLASS_COLORS));
let sliderPercent = 50;
let isDraggingSlider = false;
let isServerOnline = false;
let serverCpuThreads = null;
let serverInt8CpuThreads = null;
const liveSceneRequests = new Map();
let inferenceQueue = Promise.resolve();

// DOM helper
const $ = id => document.getElementById(id);

function inferenceSelection(side) {
  const modelKey = side === 'A' ? currentModelA : currentModelB;
  const threads = side === 'A' ? currentThreadsA : currentThreadsB;
  return { modelKey, threads, signature: `${modelKey}:${threads}` };
}

function modelThreadLimit(modelKey) {
  return modelKey.includes('int8') ? (serverInt8CpuThreads || serverCpuThreads || 3) : (serverCpuThreads || 3);
}

function populateCoreSelect(id, modelKey, selected) {
  const select = $(id);
  select.innerHTML = Array.from({ length: modelThreadLimit(modelKey) }, (_, index) =>
    `<option value="${index + 1}">CPU ×${index + 1}</option>`).join('');
  select.value = String(selected);
}

function showToast(msg) {
  const t = $('toastBox');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3000);
}

// Check Backend AI Server Status
async function checkServerStatus() {
  try {
    const res = await fetch('/api/status');
    if (res.ok) {
      const data = await res.json();
      if (data.status === 'online') {
        isServerOnline = true;
        serverCpuThreads = Number.isInteger(data.cpu_threads) && data.cpu_threads > 0 ? data.cpu_threads : null;
        serverInt8CpuThreads = Number.isInteger(data.int8_cpu_threads) && data.int8_cpu_threads > 0 ? data.int8_cpu_threads : serverCpuThreads;
        if (serverCpuThreads) {
          currentThreadsA = Math.min(currentThreadsA, modelThreadLimit(currentModelA));
          currentThreadsB = Math.min(currentThreadsB, modelThreadLimit(currentModelB));
          populateCoreSelect('coreSelectA', currentModelA, currentThreadsA);
          populateCoreSelect('coreSelectB', currentModelB, currentThreadsB);
        }
        const int8Limit = serverInt8CpuThreads < serverCpuThreads ? ` · INT8 tối đa ${serverInt8CpuThreads} core` : '';
        $('studioCpuInfo').textContent = `CPU host server: ${data.cpu_name || 'không rõ'} · tối đa ${serverCpuThreads || '?'} core${int8Limit}`;
        $('serverStatusDot').classList.add('online');
        $('serverStatusText').textContent = serverCpuThreads ? `AI Server Online (CPU host ×${serverCpuThreads})` : 'AI Server Online (Live)';
        $('serverStatusChip').style.background = '#ecfdf5';
        $('serverStatusChip').style.borderColor = '#10b981';
        return;
      }
    }
  } catch(e) {
    // server not reachable
  }
  isServerOnline = false;
  serverCpuThreads = null;
  serverInt8CpuThreads = null;
  $('studioCpuInfo').textContent = 'CPU host server: không kết nối';
  $('serverStatusDot').classList.remove('online');
  $('serverStatusText').textContent = 'Offline (Chạy python3 run_demo.py)';
}

function updateInferenceStatus() {
  const pending = ['A', 'B'].map(inferenceSelection).filter((item, index, items) =>
    item.modelKey && items.findIndex(other => other.signature === item.signature) === index
      && liveSceneRequests.has(`${currentSceneId}:${item.signature}`));
  $('aiLoadingOverlay').classList.toggle('active', pending.length > 0);
  $('aiLoadingSubtext').textContent = `Đang chuẩn bị/suy luận ${pending.map(item => `${demoData.models[item.modelKey].name} (CPU ×${item.threads})`).join(', ')} tuần tự...`;
}

function inferScene(scene, sides) {
  if (!isServerOnline) return Promise.resolve();
  const selections = sides.map(inferenceSelection).filter((item, index, items) =>
    demoData.models[item.modelKey]
      && items.findIndex(other => other.signature === item.signature) === index
      && !scene.live_detections?.[item.signature]);
  return Promise.all(selections.map(({ modelKey, threads, signature }) => {
    const requestKey = `${scene.id}:${signature}`;
    if (liveSceneRequests.has(requestKey)) return liveSceneRequests.get(requestKey);

    // The server accepts one inference at a time; discard obsolete queued selections.
    const request = inferenceQueue.then(async () => {
      try {
        const stillSelected = ['A', 'B'].some(side => inferenceSelection(side).signature === signature);
        if (currentSceneId !== scene.id || !stillSelected) return;
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: scene.data_uri, conf: 0.10, models: [modelKey], threads })
        });
        const result = await res.json();
        if (!res.ok || result.status !== 'success') throw new Error(result.message || `HTTP ${res.status}`);
        const detection = result.detections[modelKey];
        if (!detection) throw new Error('Server không trả về kết quả model đã chọn');
        if (!demoData.detections[modelKey]) demoData.detections[modelKey] = {};
        const record = {
          scene_id: scene.id,
          image_file: scene.file,
          width: result.width,
          height: result.height,
          ...detection
        };
        demoData.detections[modelKey][scene.id] = record;
        if (!scene.live_detections) scene.live_detections = {};
        scene.live_detections[signature] = record;
      } catch (err) {
        console.warn('Live inference failed:', err);
        if (currentSceneId === scene.id && [currentModelA, currentModelB].includes(modelKey)) {
          showToast(`Không thể chạy ${demoData.models[modelKey].name}: ${err.message}. Chọn lại model để thử lại.`);
        }
      } finally {
        liveSceneRequests.delete(requestKey);
        if (currentSceneId === scene.id) renderBoundingBoxes();
        updateInferenceStatus();
      }
    });
    inferenceQueue = request;
    liveSceneRequests.set(requestKey, request);
    updateInferenceStatus();
    return request;
  }));
}

function selectModel(side, modelKey) {
  if (side === 'A') {
    currentModelA = modelKey;
    currentThreadsA = Math.min(currentThreadsA, modelThreadLimit(modelKey));
    populateCoreSelect('coreSelectA', modelKey, currentThreadsA);
  } else {
    currentModelB = modelKey;
    currentThreadsB = Math.min(currentThreadsB, modelThreadLimit(modelKey));
    populateCoreSelect('coreSelectB', modelKey, currentThreadsB);
  }
  updateSceneView();
  const scene = demoData.scenes.find(s => s.id === currentSceneId);
  return inferScene(scene, [side]);
}

function selectThreads(side, value) {
  const threads = Number(value);
  const modelKey = side === 'A' ? currentModelA : currentModelB;
  if (!Number.isInteger(threads) || threads < 1 || threads > modelThreadLimit(modelKey)) return Promise.resolve();
  if (side === 'A') currentThreadsA = threads;
  else currentThreadsB = threads;
  updateSceneView();
  return inferScene(demoData.scenes.find(s => s.id === currentSceneId), [side]);
}

function runDetection() {
  if (!isServerOnline) {
    showToast('AI Server đang offline.');
    return Promise.resolve();
  }
  const sides = ['A', 'B'].filter(side => inferenceSelection(side).modelKey);
  if (!sides.length) {
    showToast('Hãy chọn ít nhất một mô hình.');
    return Promise.resolve();
  }
  const scene = demoData.scenes.find(item => item.id === currentSceneId);
  if (!scene.live_detections) scene.live_detections = {};
  sides.forEach(side => delete scene.live_detections[inferenceSelection(side).signature]);
  const request = inferScene(scene, sides);
  renderBoundingBoxes();
  return request;
}

// Initialize Application
async function initApp() {
  if (!demoData) {
    console.warn("DEMO_DATA bundle not found");
    return;
  }

  // Render Scene List
  renderSceneList();

  // Render Class Filters
  const filterEl = $('classFilterContainer');
  filterEl.innerHTML = Object.keys(CLASS_COLORS).map(cls => `
    <label class="class-check">
      <input type="checkbox" value="${cls}" checked onchange="toggleClass('${cls}', this.checked)">
      <span class="class-color-dot" style="background:${CLASS_COLORS[cls]}"></span>
      <span>${cls}</span>
    </label>
  `).join('');

  // Setup Event Listeners
  $('modelSelectA').addEventListener('change', e => {
    void selectModel('A', e.target.value);
  });
  $('modelSelectB').addEventListener('change', e => {
    void selectModel('B', e.target.value);
  });
  $('coreSelectA').addEventListener('change', e => {
    void selectThreads('A', e.target.value);
  });
  $('coreSelectB').addEventListener('change', e => {
    void selectThreads('B', e.target.value);
  });
  $('confSlider').addEventListener('input', e => {
    currentConf = parseFloat(e.target.value);
    $('confValueText').textContent = currentConf.toFixed(2);
    renderBoundingBoxes();
  });

  // Setup slider dragging
  setupSliderInteraction();

  // Custom Image Upload listener
  $('imageUploader').addEventListener('change', handleCustomImageUpload);

  // Render Benchmark Table
  renderBenchmarkTable(BENCHMARK_DATA);
  updateBenchmarkSummary();

  updateSceneView();
  await checkServerStatus();
}

function renderSceneList() {
  const listEl = $('sceneListContainer');
  listEl.replaceChildren(...demoData.scenes.map(scene => {
    const item = document.createElement('div');
    item.className = `scene-item ${scene.id === currentSceneId ? 'active' : ''}`;
    item.addEventListener('click', () => selectScene(scene.id));
    const img = document.createElement('img');
    img.className = 'scene-thumb';
    img.src = scene.data_uri || scene.image_url;
    img.alt = scene.title;
    const info = document.createElement('div');
    info.className = 'scene-info';
    for (const [className, text] of [['scene-name', scene.title], ['scene-tag', scene.tag]]) {
      const label = document.createElement('div');
      label.className = className;
      label.textContent = text;
      info.append(label);
    }
    item.append(img, info);
    return item;
  }));
  $('sceneBadge').textContent = `${demoData.scenes.length} Scenes`;
}

function selectScene(sceneId) {
  currentSceneId = sceneId;
  renderSceneList();
  updateSceneView();
  return inferScene(demoData.scenes.find(s => s.id === sceneId), ['A', 'B']);
}

function applyPreset(preset) {
  if (preset === 'ptq') {
    currentModelA = 'yolo26m_fp32';
    currentModelB = 'yolo26m_ptq_int8';
  } else if (preset === 'ptq-s') {
    currentModelA = 'yolo26s_fp32';
    currentModelB = 'yolo26s_ptq_int8';
  } else if (preset === 'edge') {
    currentModelA = 'yolo26m_ptq_int8';
    currentModelB = 'yolo26s_ptq_int8';
  }
  $('modelSelectA').value = currentModelA;
  $('modelSelectB').value = currentModelB;
  updateSceneView();
  void inferScene(demoData.scenes.find(s => s.id === currentSceneId), ['A', 'B']);
  showToast(`Đã áp dụng cấu hình so sánh: ${preset.toUpperCase()}`);
}

function toggleClass(cls, checked) {
  if (checked) enabledClasses.add(cls);
  else enabledClasses.delete(cls);
  renderBoundingBoxes();
}

function toggleAllClasses() {
  const allInputs = document.querySelectorAll('#classFilterContainer input');
  const allChecked = enabledClasses.size === Object.keys(CLASS_COLORS).length;
  allInputs.forEach(i => {
    i.checked = !allChecked;
    if (!allChecked) enabledClasses.add(i.value);
    else enabledClasses.delete(i.value);
  });
  renderBoundingBoxes();
}

function setViewMode(mode) {
  currentViewMode = mode;
  ['Split', 'Slider', 'Toggle'].forEach(m => {
    $(`modeBtn${m}`).classList.toggle('active', m.toLowerCase() === mode);
  });

  if (mode === 'split') {
    $('splitViewContainer').style.display = 'grid';
    $('sliderViewWrapper').style.display = 'none';
    $('panelA').style.display = 'flex';
    $('panelB').style.display = 'flex';
  } else if (mode === 'slider') {
    $('splitViewContainer').style.display = 'none';
    $('sliderViewWrapper').style.display = 'flex';
    requestAnimationFrame(syncSliderWidth);
  } else if (mode === 'toggle') {
    $('splitViewContainer').style.display = 'grid';
    $('sliderViewWrapper').style.display = 'none';
    selectedToggleModel = selectedToggleModel === 'A' ? 'B' : 'A';
    $('panelA').style.display = selectedToggleModel === 'A' ? 'flex' : 'none';
    $('panelB').style.display = selectedToggleModel === 'B' ? 'flex' : 'none';
    showToast(`Đang hiển thị mô hình: ${selectedToggleModel}`);
  }
}

function updateSceneView() {
  const scene = demoData.scenes.find(s => s.id === currentSceneId) || demoData.scenes[0];
  const imgSrc = scene.data_uri || scene.image_url;

  // Update scene text
  $('currentSceneTitle').textContent = scene.title;
  $('currentSceneDesc').textContent = scene.desc;

  // Set image sources and wait for load to ensure exact aspect ratio
  const setupImg = (imgEl, callback) => {
    imgEl.onload = () => {
      if (callback) callback(imgEl.naturalWidth, imgEl.naturalHeight);
    };
    imgEl.src = imgSrc;
    if (imgEl.complete && imgEl.naturalWidth) {
      if (callback) callback(imgEl.naturalWidth, imgEl.naturalHeight);
    }
  };

  setupImg($('imgA'), (w, h) => {
    $('overlayA').setAttribute('viewBox', `0 0 ${w} ${h}`);
    renderBoundingBoxes();
  });
  setupImg($('imgB'), (w, h) => {
    $('overlayB').setAttribute('viewBox', `0 0 ${w} ${h}`);
  });
  setupImg($('sliderImgA'), (w, h) => {
    $('sliderOverlayA').setAttribute('viewBox', `0 0 ${w} ${h}`);
    syncSliderWidth();
  });
  setupImg($('sliderImgB'), (w, h) => {
    $('sliderOverlayB').setAttribute('viewBox', `0 0 ${w} ${h}`);
  });

  // Update model labels
  const infoA = demoData.models[currentModelA] || { name: 'Chọn mô hình A để chạy' };
  const infoB = demoData.models[currentModelB] || { name: 'Chọn mô hình B để chạy' };

  $('labelAInfo').textContent = currentModelA ? `${infoA.badge} · ${infoA.size_mb.toFixed(2)} MiB` : 'Chưa chọn';
  $('labelBInfo').textContent = currentModelB ? `${infoB.badge} · ${infoB.size_mb.toFixed(2)} MiB` : 'Chưa chọn';
  $('modelPathA').textContent = infoA.path || '';
  $('modelPathB').textContent = infoB.path || '';

  $('panelTitleA').textContent = infoA.name;
  $('panelTitleB').textContent = infoB.name;
  $('sliderBadgeA').textContent = `Mô hình A: ${infoA.name}`;
  $('sliderBadgeB').textContent = `Mô hình B: ${infoB.name}`;

  // Telemetry updates
  $('speedupSourceLabel').textContent = currentModelA && currentModelB
    ? `${infoB.name} tăng tốc so với ${infoA.name}`
    : 'Mô hình B tăng tốc so với mô hình A';
  const mapA = infoA.map50;
  const mapB = infoB.map50;
  const hasMapPair = currentModelA && currentModelB && Number.isFinite(mapA) && Number.isFinite(mapB);
  const mapDelta = hasMapPair ? mapB - mapA : null;
  $('map50Delta').textContent = mapDelta == null ? '—' : `${mapDelta > 0 ? '+' : ''}${mapDelta.toFixed(2)} pp`;
  $('telemetryMap50A').textContent = currentModelA && Number.isFinite(mapA) ? `${mapA.toFixed(2)}%` : 'Chưa đo A';
  $('telemetryMap50B').textContent = currentModelB && Number.isFinite(mapB) ? `${mapB.toFixed(2)}%` : 'Chưa đo B';

  const sizeA = infoA.size_mb;
  const sizeB = infoB.size_mb;
  const red = ((1 - sizeB / sizeA) * 100).toFixed(1);
  $('sizeDiffDisplay').textContent = currentModelA && currentModelB ? `${red > 0 ? '-' : '+'}${Math.abs(red)}%` : '—';
  $('telemetrySizeA').textContent = currentModelA ? `${sizeA.toFixed(2)} MiB` : 'Chưa chọn A';
  $('telemetrySizeB').textContent = currentModelB ? `${sizeB.toFixed(2)} MiB` : 'Chưa chọn B';

  renderBoundingBoxes();
  updateInferenceStatus();
}

function renderBoundingBoxes() {
  const scene = demoData.scenes.find(s => s.id === currentSceneId) || demoData.scenes[0];
  const w = $('imgA').naturalWidth || scene.width || 1280;
  const h = $('imgA').naturalHeight || scene.height || 720;

  // Ensure SVGs have exact natural dimensions as viewBox
  ['overlayA', 'overlayB', 'sliderOverlayA', 'sliderOverlayB'].forEach(id => {
    const el = $(id);
    if (el) el.setAttribute('viewBox', `0 0 ${w} ${h}`);
  });

  const selectionA = inferenceSelection('A');
  const selectionB = inferenceSelection('B');
  const liveRecordA = scene.live_detections?.[selectionA.signature];
  const liveRecordB = scene.live_detections?.[selectionB.signature];
  const liveSignatures = Object.keys(scene.live_detections || {});
  const pendingA = liveSceneRequests.has(`${scene.id}:${selectionA.signature}`);
  const pendingB = liveSceneRequests.has(`${scene.id}:${selectionB.signature}`);
  const recordA = !pendingA && (liveRecordA || (!liveSignatures.some(key => key.startsWith(`${currentModelA}:`))
    ? demoData.detections[currentModelA]?.[scene.id] : null));
  const recordB = !pendingB && (liveRecordB || (!liveSignatures.some(key => key.startsWith(`${currentModelB}:`))
    ? demoData.detections[currentModelB]?.[scene.id] : null));
  const detA = recordA?.boxes || [];
  const detB = recordB?.boxes || [];
  const timeA = recordA?.inference_time_ms;
  const timeB = recordB?.inference_time_ms;
  const liveTimeA = liveRecordA?.inference_time_ms;
  const liveTimeB = liveRecordB?.inference_time_ms;
  const fpsA = Number.isFinite(liveTimeA) && liveTimeA > 0 ? 1000 / liveTimeA : null;
  const fpsB = Number.isFinite(liveTimeB) && liveTimeB > 0 ? 1000 / liveTimeB : null;
  $('speedupDisplay').textContent = fpsA != null && fpsB != null ? `${(liveTimeA / liveTimeB).toFixed(2)}x` : 'Chưa đo';
  $('telemetryTimeA').textContent = fpsA != null ? `${liveTimeA.toFixed(1)} ms` : 'Chưa đo live';
  $('telemetryTimeB').textContent = fpsB != null ? `${liveTimeB.toFixed(1)} ms (${fpsB.toFixed(1)} FPS)` : 'Chưa đo live';
  $('telemetryMetricsA').textContent = fpsA != null ? `Mô hình A · ${demoData.models[currentModelA].name} · ${liveTimeA.toFixed(1)} ms · ${fpsA.toFixed(1)} FPS` : 'Mô hình A · Chưa đo live';
  $('telemetryMetricsB').textContent = fpsB != null ? `Mô hình B · ${demoData.models[currentModelB].name} · ${liveTimeB.toFixed(1)} ms · ${fpsB.toFixed(1)} FPS` : 'Mô hình B · Chưa đo live';

  const filteredA = detA.filter(b => b.conf >= currentConf && enabledClasses.has(b.class_name));
  const filteredB = detB.filter(b => b.conf >= currentConf && enabledClasses.has(b.class_name));

  const agreement = calculateDetectionAgreement(filteredA, filteredB, w, h);
  $('overlapDisplay').textContent = recordA && recordB && agreement.retention !== null
    ? `${agreement.retention.toFixed(1)}%` : '—';
  $('overlapDetails').textContent = !recordA || !recordB
    ? 'Chưa có dữ liệu cho cả hai model trên ảnh này'
    : !filteredA.length ? 'Không có hộp A ở ngưỡng và lớp đang chọn'
    : `B giữ ${agreement.matches}/${filteredA.length} hộp A · B thêm ${filteredB.length - agreement.matches} hộp chưa khớp · IoU TB: ${agreement.meanIou === null ? '—' : (agreement.meanIou * 100).toFixed(1) + '%'}`;

  $('boxCountA').textContent = recordA ? filteredA.length : '—';
  $('timeA').textContent = recordA ? `${timeA}ms${recordA.runtime ? ` · ${recordA.runtime}` : ''}` : currentModelA ? 'Chưa suy luận' : 'Chọn mô hình A';

  $('boxCountB').textContent = recordB ? filteredB.length : '—';
  $('timeB').textContent = recordB ? `${timeB}ms${recordB.runtime ? ` · ${recordB.runtime}` : ''}` : currentModelB ? 'Chưa suy luận' : 'Chọn mô hình B';

  // Render SVG elements
  $('overlayA').innerHTML = generateSvgElements(filteredA, w, h);
  $('overlayB').innerHTML = generateSvgElements(filteredB, w, h);
  $('sliderOverlayA').innerHTML = generateSvgElements(filteredB, w, h);
  $('sliderOverlayB').innerHTML = generateSvgElements(filteredA, w, h);

  // Update export preview
  updateExportJson(filteredB, scene, w, h);
}

function calculateDetectionAgreement(boxesA, boxesB, w, h) {
  const pixels = box => box.box_px || box.box_norm.map((v, i) => v * (i % 2 ? h : w));
  const pairs = [];
  boxesA.forEach((a, ai) => {
    const [ax1, ay1, ax2, ay2] = pixels(a);
    boxesB.forEach((b, bi) => {
      if (a.class_name !== b.class_name) return;
      const [bx1, by1, bx2, by2] = pixels(b);
      const intersection = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1))
        * Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
      const union = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1)
        + Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1) - intersection;
      const iou = union > 0 ? intersection / union : 0;
      if (iou >= 0.5) pairs.push({ ai, bi, iou });
    });
  });
  pairs.sort((a, b) => b.iou - a.iou);
  const usedA = new Set();
  const usedB = new Set();
  let totalIou = 0;
  for (const pair of pairs) {
    if (usedA.has(pair.ai) || usedB.has(pair.bi)) continue;
    usedA.add(pair.ai);
    usedB.add(pair.bi);
    totalIou += pair.iou;
  }
  return {
    matches: usedA.size,
    retention: boxesA.length ? usedA.size / boxesA.length * 100 : null,
    meanIou: usedA.size ? totalIou / usedA.size : null
  };
}

function buildBDD100KJson(boxes, scene, w, h) {
  return {
    name: scene.file,
    attributes: scene.attributes || {},
    labels: boxes.map((b, idx) => ({
      id: idx,
      category: ({person:'pedestrian',motor:'motorcycle',bike:'bicycle'})[b.class_name] || b.class_name,
      manualShape: false,
      manualAttributes: false,
      box2d: {
        x1: b.box_px ? Math.round(b.box_px[0]) : Math.round(b.box_norm[0] * w),
        y1: b.box_px ? Math.round(b.box_px[1]) : Math.round(b.box_norm[1] * h),
        x2: b.box_px ? Math.round(b.box_px[2]) : Math.round(b.box_norm[2] * w),
        y2: b.box_px ? Math.round(b.box_px[3]) : Math.round(b.box_norm[3] * h)
      },
      confidence: b.conf
    }))
  };
}

function updateExportJson(boxes, scene, w, h) {
  const hasResult = !!demoData.detections[currentModelB]?.[scene.id];
  ['copyLabelsBtn', 'downloadJsonBtn', 'downloadYoloBtn'].forEach(id => { $(id).disabled = !hasResult; });
  if (!hasResult) {
    $('exportBoxesCount').textContent = 'Chưa có kết quả Model B';
    $('jsonPreviewBox').textContent = 'Chọn mô hình B và chờ kết quả để xuất nhãn.';
    return;
  }
  $('exportBoxesCount').textContent = `${boxes.length} đối tượng (Model B)`;
  
  const bddFormat = buildBDD100KJson(boxes, scene, w, h);

  const jsonStr = JSON.stringify(bddFormat, null, 2);
  $('jsonPreviewBox').textContent = jsonStr.slice(0, 1500) + (jsonStr.length > 1500 ? '\n... [Đã rút gọn]' : '');
}

function copyBDD100KJson() {
  const scene = demoData.scenes.find(s => s.id === currentSceneId) || demoData.scenes[0];
  const w = $('imgA').naturalWidth || scene.width || 1280;
  const h = $('imgA').naturalHeight || scene.height || 720;
  const detB = demoData.detections[currentModelB]?.[scene.id]?.boxes || [];
  const filteredB = detB.filter(b => b.conf >= currentConf && enabledClasses.has(b.class_name));

  const bddFormat = buildBDD100KJson(filteredB, scene, w, h);

  navigator.clipboard.writeText(JSON.stringify(bddFormat, null, 2)).then(() => {
    showToast("Đã sao chép nhãn BDD100K JSON vào Clipboard!");
  });
}

function downloadBDD100KJson() {
  const scene = demoData.scenes.find(s => s.id === currentSceneId) || demoData.scenes[0];
  const w = $('imgA').naturalWidth || scene.width || 1280;
  const h = $('imgA').naturalHeight || scene.height || 720;
  const detB = demoData.detections[currentModelB]?.[scene.id]?.boxes || [];
  const filteredB = detB.filter(b => b.conf >= currentConf && enabledClasses.has(b.class_name));

  const bddFormat = buildBDD100KJson(filteredB, scene, w, h);

  const blob = new Blob([JSON.stringify(bddFormat, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `prelabel_${scene.file.replace(/\.[^.]+$/, '')}_${currentModelB}.json`;
  a.click();
  showToast("Đã tải xuống file BDD100K JSON!");
}

function downloadYoloTxt() {
  const scene = demoData.scenes.find(s => s.id === currentSceneId) || demoData.scenes[0];
  const detB = demoData.detections[currentModelB]?.[scene.id]?.boxes || [];
  const filteredB = detB.filter(b => b.conf >= currentConf && enabledClasses.has(b.class_name));

  const lines = filteredB.map(b => {
    const [x1, y1, x2, y2] = b.box_norm;
    const xc = (x1 + x2) / 2;
    const yc = (y1 + y2) / 2;
    const w = x2 - x1;
    const h = y2 - y1;
    return `${b.class_id} ${xc.toFixed(6)} ${yc.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)}`;
  });

  const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = scene.file.replace(/\.[^.]+$/, '') + '.txt';
  a.click();
  showToast("Đã tải xuống file YOLO TXT!");
}

// Slider View drag handling. Both image layers and their SVG overlays use the
// measured base-image rectangle, so the divider cannot drift after resize.
function setupSliderInteraction() {
  const container = $('sliderStage');
  const divider = $('sliderDivider');

  function moveTo(clientX) {
    if (!isDraggingSlider) return;
    const rect = container.getBoundingClientRect();
    let pct = ((clientX - rect.left) / rect.width) * 100;
    pct = Math.max(2, Math.min(98, pct));
    sliderPercent = pct;
    updateSliderPosition(pct);
  }

  divider.addEventListener('pointerdown', e => {
    isDraggingSlider = true;
    divider.setPointerCapture(e.pointerId);
    moveTo(e.clientX);
    e.preventDefault();
  });
  window.addEventListener('pointermove', e => moveTo(e.clientX));
  window.addEventListener('pointerup', () => isDraggingSlider = false);
  window.addEventListener('pointercancel', () => isDraggingSlider = false);
}

function syncSliderWidth() {
  const stage = $('sliderStage');
  if (stage && stage.offsetWidth > 0 && stage.offsetHeight > 0) {
    $('sliderInner').style.width = stage.offsetWidth + 'px';
    $('sliderInner').style.height = stage.offsetHeight + 'px';
    updateSliderPosition(sliderPercent);
  }
}

function updateSliderPosition(pct) {
  const stage = $('sliderStage');
  if (!stage) return;
  const stageW = stage.offsetWidth;
  const stageH = stage.offsetHeight;
  if (stageW > 0 && stageH > 0) {
    $('sliderInner').style.width = stageW + 'px';
    $('sliderInner').style.height = stageH + 'px';
  }
  $('sliderClip').style.width = pct + '%';
  $('sliderDivider').style.left = pct + '%';
}

// Read the image locally; only selected models are sent for inference.
function handleCustomImageUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onerror = () => showToast('Không thể đọc ảnh đã chọn.');
  reader.onload = evt => {
    const img = new Image();
    img.onerror = () => showToast('Không thể giải mã ảnh đã chọn.');
    img.onload = () => {
      const scene = {
        id: 'custom_' + Date.now(),
        title: file.name.slice(0, 18),
        file: file.name,
        desc: `Ảnh người dùng (${img.naturalWidth}×${img.naturalHeight})`,
        tag: 'Ảnh tải lên',
        data_uri: evt.target.result,
        width: img.naturalWidth,
        height: img.naturalHeight
      };
      demoData.scenes.unshift(scene);
      void selectScene(scene.id);
    };
    img.src = evt.target.result;
  };
  reader.readAsDataURL(file);
}

function updateBenchmarkSummary() {
  const base = BENCHMARK_DATA.find(d => d.model === 'yolo26m-bdd100k.pt');
  const ptq = BENCHMARK_DATA.find(d => d.model === 'int8_yolo26m_640_openvino_model');
  const qat = BENCHMARK_DATA.find(d => d.model === 'int8_yolo26m_qat_640_openvino_model');
  if (!base || !ptq || !qat) return;
  const temporary = BENCHMARK_DATA.some(d => d.benchmark_temporary);
  const referenceLabel = 'FP16';
  if ($('benchmarkAccuracyNote')) $('benchmarkAccuracyNote').textContent = temporary
    ? 'TẠM: dùng snapshot benchmark lịch sử cho cả 6 model; baseline hiển thị là FP16.'
    : BENCHMARK_DATA.some(d => d.accuracy_pending)
    ? 'Accuracy baseline FP16 và Δ pp đang chờ đánh giá lại đủ validation.'
    : 'Accuracy baseline và Δ pp dùng kết quả đánh giá FP16 đầy đủ.';
  for (const [id, row] of [['ptqAccuracySummary', ptq], ['qatAccuracySummary', qat]]) $(id).textContent = `mAP@50–95 INT8: ${row.map95.toFixed(2)}%; ` + (row.drop == null ? 'chờ accuracy baseline FP16 để tính Δ pp.' : `Δ ${row.drop.toFixed(2)} pp so với baseline ${referenceLabel}.`);
  const latencyLabel = temporary ? 'Độ trễ benchmark FP16 (snapshot tạm)' : 'Độ trễ đo lại';
  $('ptqLatencySummary').textContent = `${latencyLabel}: ${ptq.lat.toFixed(1)} ms (${ptq.fps.toFixed(1)} FPS), CPU 4 luồng.`;
  $('qatLatencySummary').textContent = `${latencyLabel}: ${qat.lat.toFixed(1)} ms (${qat.fps.toFixed(1)} FPS), CPU 4 luồng. Chất lượng xem mAP ở bảng trên.`;
}

// Benchmark Table Rendering
function renderBenchmarkTable(data, referenceFamily = 'YOLO26m') {
  const reference = BENCHMARK_DATA.find(d => d.family === referenceFamily && d.role === 'baseline' && d.imgsz === '640x640');
  $('benchmarkSpeedupReference').textContent = referenceFamily === 'YOLO26s' ? 'vs YOLO26s' : 'vs YOLO26m 640';
  if (!reference) return;
  $('benchmarkSpeedupReference').title = `${reference.model} · ${reference.imgsz} · ${reference.lat} ms`;
  const tbody = $('benchmarkTableBody');
  tbody.innerHTML = data.map(d => {
    const isHighlight = d.model === 'int8_yolo26s_640_openvino_model';
    const isBaseline = d.role === 'baseline';
    const badgeClass = isHighlight ? 'candidate' : isBaseline ? 'baseline' : d.role === 'qat' ? 'qat' : 'nano';
    const badgeText = isHighlight ? 'PTQ Winner ⭐' : isBaseline ? 'Baseline' : d.role === 'qat' ? 'QAT' : 'INT8';

    return `
      <tr class="${isHighlight ? 'highlight-row' : ''}">
        <td>
          <strong>${d.model}</strong>
          ${isBaseline ? '<br><small>PyTorch FP16</small>' : ''}
          ${isHighlight ? '<br><small style="color:var(--green-dark);font-weight:700">Winner được chọn</small>' : ''}
        </td>
        <td><span class="badge-pill ${badgeClass}">${badgeText}</span></td>
        <td><span style="font-family:monospace">${d.imgsz}</span></td>
        <td>
          <div class="mini-bar-wrap">
            <strong>${d.size.toFixed(1)}</strong>
            <div class="mini-bar"><div class="mini-bar-fill" style="width:${(d.size/42.0*100)}%"></div></div>
          </div>
        </td>
        <td><strong>${d.map95 == null ? 'Chưa đo FP16' : d.map95.toFixed(2) + '%'}</strong></td>
        <td style="color:${d.drop > 1.0 ? 'var(--red)' : d.drop > 0 ? 'var(--ink)' : 'var(--green-dark)'}">
          ${d.drop == null ? 'N/A' : d.drop === 0 ? '0.00 (mốc)' : `${d.drop > 0 ? '+' : ''}${d.drop.toFixed(2)}`}
        </td>
        <td>${d.map50 == null ? 'Chưa đo FP16' : d.map50.toFixed(2) + '%'}</td>
        <td>${d.prec == null ? 'Chưa đo FP16' : d.prec.toFixed(2) + '%'}</td>
        <td>${d.rec == null ? 'Chưa đo FP16' : d.rec.toFixed(2) + '%'}</td>
        <td>${d.f1 == null ? 'Chưa đo FP16' : d.f1.toFixed(2) + '%'}</td>
        <td>
          <span style="font-family:monospace">${d.lat.toFixed(1)}</span>
        </td>
        <td><strong>${d.fps.toFixed(1)}</strong></td>
        <td>
          <span style="color:var(--accent);font-weight:750">${d.speedup == null ? 'Chưa đo đồng nhất' : (referenceFamily === 'YOLO26s' ? reference.lat / d.lat : d.speedup).toFixed(2) + 'x'}</span>
        </td>
      </tr>
    `;
  }).join('');
}

function filterTable(filter) {
  if (filter === 'all') {
    renderBenchmarkTable(BENCHMARK_DATA);
  } else if (filter === 'INT8') {
    renderBenchmarkTable(BENCHMARK_DATA.filter(d => d.role !== 'baseline'));
  } else {
    renderBenchmarkTable(BENCHMARK_DATA.filter(d => d.family === filter), filter);
  }
}

// Lightbox for Charts
function openLightbox(src, title) {
  $('modalImg').src = src;
  $('modalTitle').textContent = title;
  $('lightboxModal').classList.add('active');
}
function closeLightbox() {
  $('lightboxModal').classList.remove('active');
}

// Window resize sync for slider
window.addEventListener('resize', syncSliderWidth);

// Start
document.addEventListener('DOMContentLoaded', initApp);
