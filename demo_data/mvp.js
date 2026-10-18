/* MVP sessions always call the Python/OpenVINO backend; no stored detections. */
function mvpFrameAtTime(frames, time, fps) {
  if (!frames?.length || !Number.isFinite(time) || !(fps > 0)) return null;
  const timestamp = frame => frame.source_time_seconds ?? frame.time_seconds;
  if (time + 1e-7 < timestamp(frames[0]) || time + 1e-7 >= timestamp(frames.at(-1)) + 1 / fps) return null;
  let low = 0, high = frames.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (timestamp(frames[middle]) <= time + 1e-7) low = middle + 1; else high = middle;
  }
  return frames[low - 1] || null;
}
if (typeof module !== 'undefined') module.exports = {mvpFrameAtTime};
(() => {
  'use strict';
  if (typeof document === 'undefined') return;
  const root = document.getElementById('mvp');
  root.innerHTML = `
    <div class="mvp-hero"><div><div class="mvp-eyebrow">MVP · Phát hiện đối tượng giao thông</div>
      <h1>Đưa dữ liệu vào. Nhìn mô hình hoạt động.</h1>
      <p>Trải nghiệm mô hình nén trên dữ liệu của bạn: chọn dữ liệu → chạy nhận diện → xem kết quả → tải kết quả. Mọi nhận diện và số đo tại đây đều được chạy thật trên server.</p>
      <span class="mvp-badge">YOLO26s · PTQ INT8</span><span class="mvp-badge">OpenVINO · 640 × 640</span><span class="mvp-badge" id="mvpDevice">CPU · chưa khởi tạo</span>
      <p id="mvpModelName" style="margin-top:8px;font-size:11px">int8_yolo26s_640_openvino_model</p></div>
      <button class="btn btn-outline-primary" id="mvpPresent" type="button">Trình chiếu</button></div>
    <div class="mvp-layout"><aside class="mvp-card mvp-controls">
      <h2>1. Chọn dữ liệu</h2>
      <label class="mvp-drop" id="mvpDrop"><strong>Kéo thả ảnh hoặc video</strong>JPG · PNG · WEBP · MP4 · WebM
        <input id="mvpFiles" type="file" multiple accept=".jpg,.jpeg,.png,.webp,.mp4,.webm" aria-label="Chọn ảnh hoặc video"></label>
      <p>Ảnh ≤12 MiB, ≤20 triệu pixel. Video ≤64 MiB, ≤60 giây, ≤600 khung hình, ≤1920×1080. Tối đa 30 tệp/phiên.</p>
      <div class="mvp-row"><button id="mvpCamera" class="btn btn-outline-primary" type="button">Dùng webcam (ghi ≤10 giây)</button></div>
      <p id="mvpCameraNote">Webcam cần HTTPS hoặc localhost và quyền camera. Bản ghi sẽ được gửi đến server để nhận diện.</p>
      <div class="mvp-samples mvp-row" id="mvpSamples" aria-label="Ảnh mẫu chạy thật"></div>
      <div id="mvpFileList" class="mvp-file-list" aria-label="Dữ liệu đã chọn"></div>
      <label>Chế độ<select id="mvpMode"><option value="demo">Demo · ảnh / video / webcam</option><option value="evaluation">Đánh giá có nhãn · chỉ ảnh</option></select></label>
      <label id="mvpGroundTruthLabel" hidden>Nhãn chuẩn BDD100K JSON (≤12 MiB)<input type="file" id="mvpGroundTruth" accept=".json,application/json"></label>
      <p id="mvpGtHelp" hidden>List frame {name, labels: [{category, box2d: {x1,y1,x2,y2}}]} hoặc {frames: [...]}. Tên phải khớp ảnh. Không hỗ trợ COCO, ignore/crowd hay nhãn video. Đánh giá chạy conf=0.001 để tính AP.<br><a href="demo_data/mvp_sample_labels.json" download="mvp_sample_labels.json">Tải nhãn chuẩn cho các ảnh mẫu</a></p>
      <label>Ngưỡng confidence <output id="mvpConfValue">0.25</output><input id="mvpConf" type="range" min="0.001" max="1" step="0.001" value="0.25"></label>
      <fieldset id="mvpClasses" class="mvp-classes"><legend>Lớp cần nhận diện</legend></fieldset>
      <div class="mvp-row" id="mvpDisplay"><label><input id="mvpBoxes" type="checkbox" checked> Hộp</label><label><input id="mvpLabels" type="checkbox" checked> Tên lớp</label><label><input id="mvpScores" type="checkbox" checked> Confidence</label></div>
      <p>Rê chuột hoặc bấm hộp để xem chi tiết; bật/tắt hộp, nhãn ngay trên màn hình. Đổi ngưỡng/lớp hoặc cập nhật tệp xuất: bấm Chạy nhận diện. Pipeline không dùng NMS.</p>
      <button id="mvpRun" class="btn btn-primary" type="button" disabled>2. Chạy nhận diện</button>
      <div class="mvp-row"><button id="mvpStop" class="btn btn-outline-primary" type="button" disabled>Dừng tác vụ</button><button id="mvpClear" class="btn btn-outline-primary" type="button">Xóa dữ liệu</button></div>
    </aside><div class="mvp-workspace">
      <div class="mvp-status"><div id="mvpStatus" role="status" aria-live="polite">Đang kiểm tra model và server…</div><progress id="mvpProgress" value="0" max="100" aria-label="Tiến độ xử lý"></progress></div>
      <div class="mvp-metrics"><div class="mvp-metric"><span>Suy luận trung bình</span><strong id="mvpInfer">—</strong><small>OpenVINO, không warm-up</small></div>
        <div class="mvp-metric"><span>Toàn bộ trên server</span><strong id="mvpTotal">—</strong><small>Sau upload, gồm khởi tạo</small></div>
        <div class="mvp-metric"><span>FPS xử lý video</span><strong id="mvpFps">—</strong><small>Gồm vẽ và mã hóa, không FPS phát</small></div>
        <div class="mvp-metric"><span>Đã xử lý</span><strong id="mvpProcessed">0</strong><small id="mvpProcessedUnit">ảnh / khung hình</small></div></div>
      <figure class="mvp-viewer"><figcaption>Màn hình nhận diện <span id="mvpInputSize">OpenVINO INT8</span></figcaption>
        <div class="mvp-stage" id="mvpStage"><div id="mvpMedia"><div class="mvp-placeholder">Chọn dữ liệu của bạn hoặc một ảnh mẫu để bắt đầu.</div></div>
          <svg id="mvpOverlay" class="mvp-overlay" aria-label="Các đối tượng phát hiện" preserveAspectRatio="xMidYMid meet"></svg></div>
        <div class="mvp-playback" id="mvpPlayback" hidden><button type="button" id="mvpPlay">Phát</button><input type="range" id="mvpSeek" min="0" max="0" step="0.001" value="0" aria-label="Tua video"><output id="mvpTime">0.0 / 0.0 s</output></div>
        <p class="mvp-selection" id="mvpSelection" role="status">Bấm Chạy nhận diện để xem các hộp tương tác.</p>
      </figure>
      <div class="mvp-card"><h2>3. Xem kết quả</h2><div id="mvpAccuracy" class="mvp-accuracy">Chưa có nhãn chuẩn để đánh giá độ chính xác</div>
        <p id="mvpTiming" style="margin-top:10px">Confidence là độ tin cậy của từng dự đoán, không phải độ chính xác. Thời gian khởi tạo/warm-up được tách riêng.</p>
        <div id="mvpCounts" class="mvp-counts"></div><p id="mvpFrameNote">Số phát hiện trên ảnh; chưa có tracking để đếm đối tượng duy nhất.</p>
        <div class="mvp-table-wrap"><table class="mvp-table"><thead><tr><th>Khung</th><th>Lớp</th><th>Confidence</th><th>x1, y1, x2, y2 (pixel)</th></tr></thead><tbody id="mvpDetections"><tr><td colspan="4">Chưa có kết quả.</td></tr></tbody></table></div>
      </div>
      <div class="mvp-card mvp-secondary"><h2>4. Tải kết quả</h2><div class="mvp-row mvp-downloads" id="mvpDownloads"></div>
        <div class="mvp-row" style="margin-top:12px"><button class="btn btn-outline-primary" id="mvpExportJSON" disabled>JSON toàn phiên</button><button class="btn btn-outline-primary" id="mvpExportCSV" disabled>CSV toàn phiên</button><button class="btn btn-outline-primary" id="mvpReport" disabled>Báo cáo phiên</button></div>
        <p style="margin-top:10px">Video xuất WebM VP8, không âm thanh. Mỗi khung hình được xử lý tuần tự, không nội suy kết quả. Tệp tạm tự hết hạn sau 30 phút không truy cập; Xóa dữ liệu sẽ xóa ngay khi tác vụ dừng.</p>
      </div>
    </div></div>`;
  const $ = id => document.getElementById(id);
  let config, items = [], selected = -1, running = false, stopped = false, generation = 0, activeJob, upload;
  let accuracy = null, recorder = null, cameraStream = null, cameraTimer;
  let player = null, videoCallback = null, currentFrame = null, selectedBox = -1;
  const objectURLs = new Set();
  function status(text) { $('mvpStatus').textContent = text; }
  function route() {
    if (!location.hash) history.replaceState(null, '', '#mvp');
    const active = location.hash === '#mvp';
    document.body.classList.toggle('mvp-active', active);
    if (!active) document.body.classList.remove('mvp-present');
    document.querySelectorAll('.nav-links a').forEach(a => {
      a.classList.toggle('active', a.hash === location.hash);
      if (a.hash === location.hash) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    if (!active) requestAnimationFrame(() => document.getElementById(location.hash.slice(1))?.scrollIntoView({block: 'start'}));
  }
  window.addEventListener('hashchange', route); route();
  function url(blob) { const value = URL.createObjectURL(blob); objectURLs.add(value); return value; }
  const endpoint = id => `/api/mvp/jobs/${id}`;
  async function api(path, options) {
    const response = await fetch(path, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || `HTTP ${response.status}`);
    return data;
  }
  function buttons() {
    root.classList.toggle('mvp-processing', running);
    $('mvpRun').disabled = running || recorder?.state === 'recording' || !config?.available || !items.length;
    $('mvpStop').disabled = !running;
    for (const id of ['mvpFiles', 'mvpMode', 'mvpConf', 'mvpGroundTruth', 'mvpClasses', 'mvpBoxes', 'mvpLabels', 'mvpScores', 'mvpCamera']) $(id).disabled = running || (id === 'mvpConf' && $('mvpMode').value === 'evaluation');
    const hasResults = items.some(item => item.report);
    for (const id of ['mvpExportJSON', 'mvpExportCSV', 'mvpReport']) $(id).disabled = running || !hasResults;
  }
  function list() {
    $('mvpFileList').replaceChildren(...items.map((item, i) => {
      const button = document.createElement('button'); button.type = 'button';
      button.classList.toggle('selected', i === selected);
      button.textContent = `${i + 1}. ${item.file.name} · ${item.job?.status || 'chưa chạy'}`;
      button.onclick = () => { selected = i; preview(); list(); };
      return button;
    }));
  }
  function addFiles(files) {
    if (running) return;
    try {
      for (const file of files) {
        if (items.length >= 30) throw new Error('Tối đa 30 tệp; hãy xóa phiên cũ.');
        const extension = file.name.split('.').pop().toLowerCase();
        if (!['jpg', 'jpeg', 'png', 'webp', 'mp4', 'webm'].includes(extension)) throw new Error(`Không hỗ trợ: ${file.name}`);
        const video = ['mp4', 'webm'].includes(extension);
        if (!file.size || file.size > (video ? 64 : 12) * 1024 ** 2) throw new Error(`${file.name}: vượt giới hạn ${video ? 64 : 12} MiB hoặc tệp rỗng.`);
        items.push({file, video, source: url(file)});
      }
      selected = items.length - 1; accuracy = null; list(); preview(); buttons();
      status(`Đã chọn ${items.length} tệp. Bấm Chạy nhận diện để xử lý.`);
    } catch (error) { status(error.message); list(); buttons(); }
  }
  function clearPlayer() {
    if (videoCallback !== null && player?.cancelVideoFrameCallback) player.cancelVideoFrameCallback(videoCallback);
    videoCallback = null;
    if (player) {
      for (const name of ['onload', 'onerror', 'onloadedmetadata', 'ontimeupdate', 'onseeked', 'onpause', 'onplay', 'onended']) player[name] = null;
      if (player.tagName === 'VIDEO') player.pause();
    }
    player = null; currentFrame = null; selectedBox = -1;
    $('mvpMedia').replaceChildren(); $('mvpOverlay').replaceChildren();
    $('mvpPlayback').hidden = true;
    $('mvpSelection').textContent = 'Bấm Chạy nhận diện để xem các hộp tương tác.';
  }
  function media(item) {
    clearPlayer();
    const element = document.createElement(item.video ? 'video' : 'img');
    element.src = item.source;
    if (item.video) { element.preload = 'metadata'; element.muted = true; element.playsInline = true; }
    else element.alt = 'Ảnh với các hộp nhận diện tương tác';
    element.onerror = () => status('Trình duyệt không đọc được dữ liệu/codec này. Với video, hãy thử MP4 H.264 hoặc WebM VP8.');
    $('mvpMedia').append(element); player = element;
    if (item.video) {
      $('mvpPlayback').hidden = false;
      const update = time => {
        const frame = element.ended && item.job?.status === 'completed' ? item.frames?.at(-1) : mvpFrameAtTime(item.frames, time, item.report?.source_fps);
        if (frame !== currentFrame) { currentFrame = frame; selectedBox = -1; renderOverlay(); }
        $('mvpSeek').value = time;
        $('mvpTime').textContent = `${time.toFixed(1)} / ${Number.isFinite(element.duration) ? element.duration.toFixed(1) : '?'} s`;
      };
      element.onloadedmetadata = () => {
        $('mvpSeek').max = Number.isFinite(element.duration) ? element.duration : (item.report?.total || 0) / (item.report?.source_fps || 1);
        $('mvpSeek').disabled = !$('mvpSeek').max || +$('mvpSeek').max <= 0;
        update(element.currentTime); renderOverlay();
      };
      element.ontimeupdate = () => { update(element.currentTime); table(currentFrame?.detections || [], currentFrame?.index || 0); };
      element.onseeked = element.onpause = () => { update(element.currentTime); renderOverlay(); table(currentFrame?.detections || [], currentFrame?.index || 0); $('mvpPlay').textContent = element.paused ? 'Phát' : 'Tạm dừng'; };
      element.onplay = () => { $('mvpPlay').textContent = 'Tạm dừng'; };
      element.onended = () => { $('mvpPlay').textContent = 'Phát lại'; };
      $('mvpPlay').textContent = 'Phát';
      if (element.requestVideoFrameCallback) {
        const onFrame = (_, metadata) => {
          if (player !== element) return;
          update(metadata.mediaTime);
          videoCallback = element.requestVideoFrameCallback(onFrame);
        };
        videoCallback = element.requestVideoFrameCallback(onFrame);
      }
    } else element.onload = renderOverlay;
    return element;
  }
  function preview() {
    const item = items[selected]; if (!item) return;
    media(item);
    const job = item.job, report = item.report;
    $('mvpInputSize').textContent = report?.width ? `${report.width} × ${report.height}` : '';
    $('mvpDownloads').replaceChildren();
    $('mvpInfer').textContent = report?.inference_mean_ms != null ? `${report.inference_mean_ms.toFixed(1)} ms` : '—';
    $('mvpTotal').textContent = report ? `${(report.timings.server_total_ms / 1000).toFixed(2)} s` : '—';
    $('mvpFps').textContent = item.video && report?.processing_fps ? report.processing_fps.toFixed(1) : '—';
    $('mvpProcessed').textContent = report?.processed ?? job?.processed ?? 0;
    $('mvpProcessedUnit').textContent = item.video ? 'khung hình · tệp đang xem' : 'ảnh · tệp đang xem';
    $('mvpDevice').textContent = report?.runtime ? `${report.runtime.execution_devices.join(', ')} · ${report.runtime.ov_threads} luồng` : 'CPU · chưa khởi tạo';
    $('mvpTiming').textContent = report ? `Kết quả ở conf=${report.settings.conf}. Khởi tạo/warm-up: ${report.timings.initialization_ms.toFixed(1)} ms (0 nếu tái sử dụng). Pipeline trung bình: ${report.processed ? (report.timings.pipeline_ms / report.processed).toFixed(1) : '—'} ms. ${report.timing_scope} Thiết bị: ${report.runtime?.device_name || 'chưa khởi tạo'}. ${report.runtime?.thread_warning || ''}` : 'Confidence không phải độ chính xác. Chạy nhận diện để đo hiệu năng.';
    $('mvpAccuracy').textContent = accuracy ? `Đánh giá ${accuracy.images} ảnh · Precision ${(accuracy.precision * 100).toFixed(2)}% · Recall ${(accuracy.recall * 100).toFixed(2)}% · mAP50 ${(accuracy.map50 * 100).toFixed(2)}% · mAP50–95 ${(accuracy.map50_95 * 100).toFixed(2)}%. ${accuracy.protocol}` : 'Chưa có nhãn chuẩn để đánh giá độ chính xác';
    if (!report || !job.files.includes(item.video ? 'annotated.webm' : 'annotated.jpg')) {
      $('mvpSelection').textContent = job?.message || 'Chưa chạy nhận diện trên tệp này.';
      table([], 0); return;
    }
    currentFrame = item.frames[0] || null; renderOverlay();
    table(item.frames[0]?.detections || [], 0);
    for (const filename of job.files) {
      const a = document.createElement('a'); a.href = `${endpoint(job.id)}/files/${filename}`; a.download = `${item.file.name}-${filename}`;
      a.textContent = filename.startsWith('annotated.') ? 'Tải ảnh/video nhận diện' : filename; $('mvpDownloads').append(a);
    }
  }
  function renderOverlay() {
    const item = items[selected], overlay = $('mvpOverlay');
    if (!item?.report || !currentFrame || !overlay.clientWidth || !overlay.clientHeight) {
      overlay.replaceChildren();
      if (item?.report && item.video) $('mvpSelection').textContent = 'Khung này chưa được xử lý; không có hộp nhận diện.';
      return;
    }
    const {width, height} = item.report;
    overlay.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const colors = Object.fromEntries(Object.entries(config.names).map(([id, name]) => [name, config.colors[+id % config.colors.length]]));
    const boxes = currentFrame.detections.map(box => ({class_name: box.class_name, conf: box.confidence, box_px: box.xyxy}));
    const display = {boxes: $('mvpBoxes').checked, labels: $('mvpLabels').checked, scores: $('mvpScores').checked, colors, interactive: true};
    overlay.innerHTML = display.boxes || display.labels || display.scores ? generateSvgElements(boxes, width, height,
      Math.max(width / overlay.clientWidth, height / overlay.clientHeight), display) : '';
    highlightSelection();
  }
  function highlightSelection() {
    const boxes = currentFrame?.detections || [];
    $('mvpOverlay').querySelectorAll('[data-box-index]').forEach(group => {
      const active = +group.dataset.boxIndex === selectedBox;
      group.classList.toggle('selected', active); group.setAttribute('aria-pressed', String(active));
    });
    $('mvpDetections').querySelectorAll('tr').forEach((row, i) => row.classList.toggle('selected', i === selectedBox));
    const box = boxes[selectedBox];
    if (!currentFrame) return;
    $('mvpSelection').textContent = box ? `${box.class_name} · ${(box.confidence * 100).toFixed(1)}% · [${box.xyxy.map(x => x.toFixed(1)).join(', ')}] px` :
      `${boxes.length} phát hiện · Rê chuột để xem; bấm hộp hoặc dòng trong bảng để chọn${items[selected]?.video ? '. Video tạm dừng khi chọn hộp.' : '.'}`;
  }
  function selectBox(index, frameIndex = currentFrame?.index) {
    const item = items[selected];
    const frame = item?.frames?.find(value => value.index === frameIndex);
    if (player?.tagName === 'VIDEO') player.pause();
    if (frame && frame !== currentFrame) {
      currentFrame = frame; selectedBox = -1;
      if (player?.tagName === 'VIDEO') player.currentTime = frame.source_time_seconds ?? frame.time_seconds;
      renderOverlay();
    }
    selectedBox = selectedBox === index ? -1 : index; highlightSelection();
  }
  function table(boxes, index) {
    const counts = {};
    for (const box of boxes) counts[box.class_name] = (counts[box.class_name] || 0) + 1;
    $('mvpCounts').replaceChildren(...Object.entries(counts).map(([name, count]) => {
      const chip = document.createElement('span'); chip.textContent = `${name}: ${count}`; return chip;
    }));
    $('mvpFrameNote').textContent = items[selected]?.video ? `Khung ${index + 1}: ${boxes.length} phát hiện. Đây là số phát hiện trong khung đang xem; chưa có tracking hoặc số đối tượng duy nhất.` : `${boxes.length} phát hiện trên ảnh. Không phát hiện đối tượng cũng là một kết quả hợp lệ.`;
    if (!currentFrame) $('mvpFrameNote').textContent = 'Chưa có kết quả nhận diện cho ảnh/khung hình đang xem.';
    const rows = boxes.map((box, boxIndex) => {
      const row = document.createElement('tr');
      row.tabIndex = 0;
      row.onclick = () => selectBox(boxIndex, index);
      row.onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectBox(boxIndex, index); } };
      for (const value of [index + 1, box.class_name, `${(box.confidence * 100).toFixed(1)}%`, box.xyxy.map(x => x.toFixed(1)).join(', ')]) {
        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      }
      return row;
    });
    if (!rows.length) { const row = document.createElement('tr'), cell = document.createElement('td'); cell.colSpan = 4; cell.textContent = currentFrame ? 'Không có đối tượng ở ngưỡng và lớp đã chọn.' : 'Chưa có kết quả.'; row.append(cell); rows.push(row); }
    $('mvpDetections').replaceChildren(...rows);
    highlightSelection();
  }
  function sendFile(item, params, currentGeneration) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest(); upload = xhr;
      xhr.open('POST', `/api/mvp/jobs?${params}`); xhr.timeout = 120000;
      xhr.upload.onprogress = e => { if (e.lengthComputable && generation === currentGeneration) status(`Đang tải ${item.file.name}: ${Math.round(100 * e.loaded / e.total)}%`); };
      xhr.onload = () => { try { const data = JSON.parse(xhr.responseText); if (xhr.status !== 202) throw new Error(data.message || 'Không tải được tệp'); resolve(data); } catch (e) { reject(e); } };
      xhr.onerror = () => reject(new Error('Mất kết nối server khi tải tệp.'));
      xhr.ontimeout = () => reject(new Error('Tải tệp quá thời gian; thử tệp nhỏ hơn.'));
      xhr.onabort = () => reject(new Error('Đã hủy tải lên.'));
      xhr.send(item.file);
    });
  }
  async function run() {
    if (running || !items.length) return;
    const currentGeneration = generation;
    const evaluation = $('mvpMode').value === 'evaluation';
    let groundTruth;
    try {
      if (evaluation) {
        if (items.some(item => item.video)) throw new Error('Đánh giá có nhãn chỉ hỗ trợ ảnh. Xóa video hoặc chuyển sang Demo.');
        const labels = $('mvpGroundTruth').files[0];
        if (!labels || labels.size > 12 * 1024 ** 2) throw new Error('Hãy chọn ground truth JSON ≤12 MiB.');
        groundTruth = JSON.parse(await labels.text());
      }
      const classes = [...$('mvpClasses').querySelectorAll('input:checked')].map(el => el.value);
      if (!classes.length) throw new Error('Hãy chọn ít nhất một lớp.');
      running = true; stopped = false; accuracy = null; buttons();
      for (let i = 0; i < items.length && !stopped && generation === currentGeneration; i++) {
        const item = items[i]; selected = i;
        if (item.job) await api(endpoint(item.job.id), {method: 'DELETE'}).catch(() => {});
        item.job = null; item.report = null; item.frames = []; preview(); list();
        const params = new URLSearchParams({filename: item.file.name, conf: evaluation ? '.001' : $('mvpConf').value, classes: classes.join(','),
          boxes: +$('mvpBoxes').checked, labels: +$('mvpLabels').checked, scores: +$('mvpScores').checked});
        item.job = await sendFile(item, params, currentGeneration); upload = null; activeJob = item.job.id;
        if (stopped || generation !== currentGeneration) await api(`${endpoint(item.job.id)}/cancel`, {method: 'POST'});
        list(); preview();
        while (['starting', 'running'].includes(item.job.status)) {
          await new Promise(resolve => setTimeout(resolve, 350));
          item.job = await api(endpoint(item.job.id));
          if (generation !== currentGeneration) return;
          $('mvpProgress').value = 100 * (i + item.job.progress / 100) / items.length;
          $('mvpProcessed').textContent = item.job.processed;
          status(`Tệp ${i + 1}/${items.length} · ${item.file.name}\n${item.job.message} ${item.job.processed}/${item.job.total || '?'} khung/ảnh.`);
        }
        if (item.job.files.includes('report.json')) {
          item.report = await api(`${endpoint(item.job.id)}/files/report.json`);
          item.frames = item.job.files.includes('detections.json') ? await api(`${endpoint(item.job.id)}/files/detections.json`) : [];
        }
        activeJob = null; list(); preview();
        if (item.job.status === 'failed') throw new Error(`${item.file.name}: ${item.job.message}`);
        if (item.job.status === 'cancelled') stopped = true;
      }
      if (generation === currentGeneration && !stopped) $('mvp').scrollIntoView({behavior: 'smooth', block: 'start'});
      if (evaluation && !stopped && generation === currentGeneration) {
        status('Đang ghép nhãn chuẩn và tính precision, recall, mAP…');
        accuracy = await api('/api/mvp/evaluate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({jobs: items.map(item => item.job.id), ground_truth: groundTruth})});
        preview();
      }
      if (generation === currentGeneration) status(stopped ? 'Đã dừng. Có thể xem/tải phần đã xử lý hoặc chạy lại.' : `Hoàn tất ${items.length} tệp. Chọn từng tệp để xem và tải kết quả.`);
    } catch (error) { if (generation === currentGeneration) status(error.message); }
    finally { if (generation === currentGeneration) { running = false; activeJob = null; upload = null; buttons(); list(); } }
  }
  async function stop() {
    stopped = true; upload?.abort();
    if (activeJob) await api(`${endpoint(activeJob)}/cancel`, {method: 'POST'}).catch(() => {});
    status('Đang dừng sau khung hình hiện tại…');
  }
  function stopCamera() {
    clearTimeout(cameraTimer);
    if (recorder?.state === 'recording') recorder.stop();
    cameraStream?.getTracks().forEach(track => track.stop()); cameraStream = null;
  }
  async function clear() {
    generation++; stopped = true; upload?.abort();
    stopCamera();
    const old = items; items = []; selected = -1; accuracy = null;
    await Promise.allSettled(old.filter(item => item.job).map(item => api(endpoint(item.job.id), {method: 'DELETE'})));
    objectURLs.forEach(value => URL.revokeObjectURL(value)); objectURLs.clear();
    running = false; activeJob = upload = null;
    $('mvpFiles').value = ''; $('mvpGroundTruth').value = ''; $('mvpProgress').value = 0;
    clearPlayer(); $('mvpDownloads').replaceChildren();
    for (const id of ['mvpInfer', 'mvpTotal', 'mvpFps']) $(id).textContent = '—';
    $('mvpProcessed').textContent = '0'; $('mvpInputSize').textContent = '';
    $('mvpAccuracy').textContent = 'Chưa có nhãn chuẩn để đánh giá độ chính xác'; table([], 0);
    list(); buttons(); status('Đã xóa dữ liệu. Sẵn sàng cho phiên demo mới.');
  }
  function download(data, type, name) {
    const link = document.createElement('a'); const value = URL.createObjectURL(new Blob([data], {type}));
    link.href = value; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(value), 1000);
  }
  $('mvpFiles').onchange = event => addFiles(event.target.files);
  for (const event of ['dragenter', 'dragover']) $('mvpDrop').addEventListener(event, e => { e.preventDefault(); $('mvpDrop').classList.add('drag'); });
  $('mvpDrop').ondragleave = () => $('mvpDrop').classList.remove('drag');
  $('mvpDrop').ondrop = event => { event.preventDefault(); $('mvpDrop').classList.remove('drag'); addFiles(event.dataTransfer.files); };
  $('mvpRun').onclick = run; $('mvpStop').onclick = stop; $('mvpClear').onclick = clear;
  $('mvpMode').onchange = () => {
    const evaluation = $('mvpMode').value === 'evaluation';
    $('mvpGroundTruthLabel').hidden = $('mvpGtHelp').hidden = !evaluation;
    $('mvpConf').value = evaluation ? '.001' : '.25'; $('mvpConfValue').textContent = $('mvpConf').value; accuracy = null; buttons();
  };
  for (const id of ['mvpConf', 'mvpClasses']) $(id).addEventListener('input', () => {
    $('mvpConfValue').textContent = (+$('mvpConf').value).toFixed(3); accuracy = null;
    $('mvpAccuracy').textContent = $('mvpMode').value === 'evaluation' ? 'Cấu hình đã đổi; chạy lại để đánh giá theo nhãn chuẩn.' : 'Chưa có nhãn chuẩn để đánh giá độ chính xác';
    if (items.some(item => item.report)) status('Cấu hình đã đổi. Bấm Chạy nhận diện để cập nhật hình, số liệu và tệp xuất.');
  });
  for (const id of ['mvpBoxes', 'mvpLabels', 'mvpScores']) $(id).addEventListener('input', () => {
    renderOverlay();
    if (items.some(item => item.report)) status('Đã cập nhật hiển thị. Tệp tải xuống giữ cấu hình lúc chạy; bấm Chạy nhận diện nếu muốn cập nhật tệp xuất.');
  });
  $('mvpOverlay').onclick = event => {
    const group = event.target.closest('.bbox-group');
    if (group) selectBox(+group.dataset.boxIndex);
  };
  $('mvpOverlay').onkeydown = event => {
    const group = event.target.closest('.bbox-group');
    if (group && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); selectBox(+group.dataset.boxIndex); }
  };
  $('mvpPlay').onclick = () => {
    if (player?.tagName !== 'VIDEO') return;
    if (player.paused) player.play().catch(error => status(`Không phát được video: ${error.message}`)); else player.pause();
  };
  $('mvpSeek').oninput = () => { if (player?.tagName === 'VIDEO') { player.pause(); player.currentTime = +$('mvpSeek').value; } };
  new ResizeObserver(renderOverlay).observe($('mvpStage'));
  $('mvpPresent').onclick = () => { const present = document.body.classList.toggle('mvp-present'); $('mvpPresent').textContent = present ? 'Thoát trình chiếu (Esc)' : 'Trình chiếu'; };
  document.addEventListener('keydown', e => { if (e.key === 'Escape') { document.body.classList.remove('mvp-present'); $('mvpPresent').textContent = 'Trình chiếu'; } });
  $('mvpExportJSON').onclick = () => download(JSON.stringify(items.filter(x => x.report).map(x => ({filename: x.file.name, frames: x.frames})), null, 2), 'application/json', 'mvp-detections.json');
  $('mvpExportCSV').onclick = () => {
    const cell = value => { let text = String(value); if (/^[=+\-@\t\r]/.test(text)) text = "'" + text; return '"' + text.replaceAll('"', '""') + '"'; };
    const rows = [['filename', 'frame', 'time_seconds', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2']];
    for (const item of items.filter(x => x.report)) for (const frame of item.frames) for (const box of frame.detections) rows.push([item.file.name, frame.index, frame.time_seconds, box.class_name, box.confidence, ...box.xyxy]);
    download('\uFEFF' + rows.map(row => row.map(cell).join(',')).join('\r\n'), 'text/csv;charset=utf-8', 'mvp-detections.csv');
  };
  $('mvpReport').onclick = () => download(JSON.stringify({created_at: new Date().toISOString(), model: config.model, names: config.names, sessions: items.filter(x => x.report).map(x => x.report), evaluation: accuracy, accuracy_message: accuracy ? 'Có đánh giá theo nhãn đã cung cấp' : 'Chưa có nhãn chuẩn để đánh giá độ chính xác'}, null, 2), 'application/json', 'mvp-session-report.json');
  $('mvpCamera').onclick = async () => {
    if (recorder?.state === 'recording') { stopCamera(); return; }
    try {
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) throw new Error('Webcam cần HTTPS hoặc localhost và trình duyệt hỗ trợ MediaRecorder.');
      const mime = ['video/webm;codecs=vp8', 'video/webm;codecs=vp9', 'video/webm'].find(x => MediaRecorder.isTypeSupported(x));
      if (!mime) throw new Error('Trình duyệt chưa hỗ trợ ghi WebM. Hãy tải video từ thiết bị.');
      cameraStream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640, max: 1280}, height: {ideal: 480, max: 720}, frameRate: {ideal: 15, max: 30}}, audio: false});
      clearPlayer();
      const preview = document.createElement('video'); preview.autoplay = preview.muted = preview.playsInline = true; preview.srcObject = cameraStream; $('mvpMedia').replaceChildren(preview); player = preview;
      recorder = new MediaRecorder(cameraStream, {mimeType: mime}); const chunks = [], capturedGeneration = generation;
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = () => {
        if (capturedGeneration === generation) addFiles([new File(chunks, `webcam-${Date.now()}.webm`, {type: 'video/webm'})]);
        $('mvpCamera').textContent = 'Dùng webcam (ghi ≤10 giây)'; recorder = null; buttons();
      };
      recorder.start(500); buttons(); $('mvpCamera').textContent = 'Dừng ghi & dùng video'; status('Đang ghi webcam, tối đa 10 giây. Sau đó bấm Chạy nhận diện.');
      cameraTimer = setTimeout(stopCamera, 10000);
    } catch (error) { stopCamera(); status(`Không dùng được webcam: ${error.message}`); }
  };
  window.addEventListener('pagehide', () => { generation++; stopCamera(); for (const item of items) if (item.job) fetch(endpoint(item.job.id), {method: 'DELETE', keepalive: true}).catch(() => {}); });
  api('/api/mvp/config').then(data => {
    config = data;
    for (const [id, name] of Object.entries(config.names)) {
      const label = document.createElement('label'), input = document.createElement('input'); input.type = 'checkbox'; input.value = id; input.checked = true;
      label.append(input, document.createTextNode(name)); $('mvpClasses').append(label);
    }
    const scenes = window.DEMO_DATA?.scenes || [];
    scenes.slice(0, 3).forEach((scene, i) => {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = ['Mẫu ban ngày', 'Mẫu ban đêm', 'Mẫu giao thông'][i];
      button.onclick = async () => { try { const response = await fetch(`demo_data/images/${encodeURIComponent(scene.file)}`); if (!response.ok) throw new Error('Không tải được ảnh mẫu'); addFiles([new File([await response.blob()], scene.file, {type: 'image/jpeg'})]); } catch (e) { status(e.message); } };
      $('mvpSamples').append(button);
    });
    status(config.available ? 'Model sẵn sàng. Chọn dữ liệu rồi bấm Chạy nhận diện; model chỉ được nạp khi chạy lần đầu.' : 'Thiếu model MVP. Chạy git pull trên server để lấy XML, BIN và metadata.yaml.'); buttons();
  }).catch(error => status(`Không kết nối được MVP backend: ${error.message}. Hãy chạy python run_demo.py.`));
})();
