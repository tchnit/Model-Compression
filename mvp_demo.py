"""Bounded, cancellable MVP sessions using the existing cached OpenVINO model."""
import atexit
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from inference_runtime import configure_cpu_backend, cpu_device

MODEL_KEY = 'yolo26s_ptq_int8'
MODEL_NAME = 'int8_yolo26s_640_openvino_model'
LIMITS = dict(upload_bytes=64 * 1024**2, image_bytes=12 * 1024**2, image_pixels=20_000_000,
              video_pixels=1920 * 1080, video_seconds=60, video_frames=600, jobs=30, ttl_seconds=1800)
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.mp4', '.webm'}
COLORS = ['#38bdf8', '#fb7185', '#a3e635', '#fbbf24', '#c084fc', '#2dd4bf', '#f97316', '#e879f9', '#60a5fa', '#facc15']


class MVPService:
    def __init__(self, get_model, slot, model_path, threads=4):
        self.get_model, self.slot = get_model, slot
        self.model_path, self.threads = Path(model_path), threads
        self.temp = tempfile.TemporaryDirectory(prefix='yolo26-mvp-')
        self.jobs, self.lock = {}, threading.RLock()
        self.runtime = None
        self.closed = threading.Event()
        threading.Thread(target=self.expire, daemon=True).start()
        atexit.register(self.close)

    def expire(self):
        while not self.closed.wait(60):
            self.cleanup()

    def close(self):
        self.closed.set()
        with self.lock:
            for job in self.jobs.values():
                job['cancel'].set()
        self.temp.cleanup()

    def config(self):
        import yaml
        metadata = self.model_path / 'metadata.yaml'
        names = {}
        if metadata.is_file():
            data = yaml.safe_load(metadata.read_text())
            names = data.get('names', {})
            if isinstance(names, list):
                names = dict(enumerate(names))
            if data.get('imgsz') != [640, 640] or data.get('end2end') is not True or data.get('args', {}).get('nms'):
                raise ValueError('MVP cần graph YOLO26 640×640 end-to-end, không NMS.')
        available = metadata.is_file() and any(self.model_path.glob('*.xml')) and all(
            p.with_suffix('.bin').is_file() for p in self.model_path.glob('*.xml'))
        return dict(model=MODEL_NAME, model_key=MODEL_KEY, available=bool(available), precision='INT8',
                    input_size=[640, 640], names=names, colors=COLORS, device_requested='CPU',
                    runtime=self.runtime, limits=LIMITS, formats=['JPG', 'JPEG', 'PNG', 'WEBP', 'MP4', 'WebM'],
                    video_export='WebM VP8, không âm thanh', tracking=False)

    def cleanup(self):
        with self.lock:
            for key, job in list(self.jobs.items()):
                if job['status'] not in ('running', 'starting') and time.monotonic() - job['touched'] > LIMITS['ttl_seconds']:
                    shutil.rmtree(job['folder'], ignore_errors=True)
                    del self.jobs[key]

    def get(self, key):
        if not re.fullmatch('[0-9a-f]{32}', key):
            raise KeyError('Phiên không tồn tại')
        with self.lock:
            job = self.jobs[key]
            job['touched'] = time.monotonic()
            return job

    def snapshot(self, job):
        with self.lock:
            return {k: v for k, v in job.items() if k not in ('folder', 'cancel', 'frames', 'touched', 'delete')}

    def submit(self, stream, length, query):
        filename = query.get('filename', [''])[0]
        if not filename or Path(filename).name != filename or '\\' in filename or len(filename) > 200:
            raise ValueError('Tên tệp không hợp lệ')
        suffix = Path(filename).suffix.lower()
        if suffix not in EXTENSIONS:
            raise ValueError('Chỉ hỗ trợ JPG, JPEG, PNG, WEBP, MP4 và WebM')
        video = suffix in ('.mp4', '.webm')
        max_bytes = LIMITS['upload_bytes'] if video else LIMITS['image_bytes']
        if not 0 < length <= max_bytes:
            raise ValueError(f'Tệp phải có dung lượng từ 1 byte đến {max_bytes // 1024**2} MiB')
        conf = float(query.get('conf', ['0.25'])[0])
        if not math.isfinite(conf) or not 0.001 <= conf <= 1:
            raise ValueError('Confidence phải thuộc [0.001, 1]')
        config = self.config()
        if not config['available']:
            raise FileNotFoundError(f'Thiếu {MODEL_NAME}: chạy git pull để tải đầy đủ XML, BIN và metadata.yaml.')
        classes = [int(x) for x in query.get('classes', [''])[0].split(',') if x != '']
        if not classes or any(str(c) not in {str(k) for k in config['names']} for c in classes):
            raise ValueError('Hãy chọn ít nhất một lớp hợp lệ')
        self.cleanup()
        with self.lock:
            if len(self.jobs) >= LIMITS['jobs']:
                raise ValueError('Đã đạt giới hạn 30 tệp; xóa phiên cũ để tiếp tục.')
            disk_bytes = sum(p.stat().st_size for p in Path(self.temp.name).rglob('*') if p.is_file())
            if disk_bytes + length > 512 * 1024**2:
                raise ValueError('Phiên đã đạt 512 MiB; xóa dữ liệu cũ để tiếp tục.')
        if not self.slot.acquire(blocking=False):
            raise RuntimeError('Server đang chạy tác vụ khác; vui lòng thử lại sau.')
        key = uuid.uuid4().hex
        folder = Path(self.temp.name) / key
        try:
            folder.mkdir()
            digest = hashlib.sha256()
            with (folder / ('input' + suffix)).open('wb') as output:
                remaining = length
                while remaining:
                    block = stream.read(min(65536, remaining))
                    if not block:
                        raise ValueError('Tệp tải lên bị ngắt; hãy tải lại')
                    output.write(block)
                    digest.update(block)
                    remaining -= len(block)
            job = dict(id=key, filename=filename, kind='video' if video else 'image', status='starting',
                       input_bytes=length, input_sha256=digest.hexdigest(),
                       processed=0, total=1, progress=0, message='Đang nạp / warm-up mô hình…',
                       created_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                       settings=dict(conf=conf, classes=classes,
                                     boxes=query.get('boxes', ['1'])[0] == '1',
                                     labels=query.get('labels', ['1'])[0] == '1',
                                     scores=query.get('scores', ['1'])[0] == '1'),
                       folder=folder, cancel=threading.Event(), touched=time.monotonic(), delete=False, frames=[],
                       timings=dict(initialization_ms=0, inference_ms=0, pipeline_ms=0, processing_ms=0), files=[])
            with self.lock:
                self.jobs[key] = job
            threading.Thread(target=self.process, args=(job, suffix), daemon=True).start()
            return self.snapshot(job)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            self.slot.release()
            raise

    def infer(self, model, image, settings):
        start = time.perf_counter()
        result = model(image, imgsz=640, conf=settings['conf'], classes=settings['classes'],
                       rect=False, max_det=300, verbose=False, device=cpu_device(self.model_path))[0]
        pipeline_ms = (time.perf_counter() - start) * 1000
        boxes = []
        for row in result.boxes.data.cpu().tolist():
            x1, y1, x2, y2, confidence, class_id = row
            boxes.append(dict(class_id=int(class_id), class_name=result.names[int(class_id)],
                              confidence=confidence, xyxy=[x1, y1, x2, y2]))
        return result, boxes, float(result.speed['inference']), pipeline_ms

    def annotate(self, image, boxes, settings):
        import cv2
        annotated = image.copy()
        for box in boxes:
            color = COLORS[box['class_id'] % len(COLORS)].lstrip('#')
            bgr = tuple(int(color[i:i + 2], 16) for i in (4, 2, 0))
            x1, y1, x2, y2 = [round(x) for x in box['xyxy']]
            if settings['boxes']:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)
            parts = ([box['class_name']] if settings['labels'] else []) + ([f"{box['confidence']:.0%}"] if settings['scores'] else [])
            if parts:
                text = ' '.join(parts)
                (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .6, 2)
                y = max(height + 8, y1)
                cv2.rectangle(annotated, (x1, y - height - 8), (x1 + width + 6, y), bgr, -1)
                cv2.putText(annotated, text, (x1 + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .6, (15, 23, 42), 2)
        return annotated

    def process(self, job, suffix):
        import cv2
        import numpy as np
        from hardware_benchmark import decode_image
        import base64
        capture = writer = None
        started = time.perf_counter()
        terminal = 'failed'
        try:
            path = job['folder'] / ('input' + suffix)
            if job['kind'] == 'image':
                from PIL import Image
                with Image.open(path) as uploaded:
                    if uploaded.format not in ('JPEG', 'PNG', 'WEBP'):
                        raise ValueError('Nội dung ảnh phải là JPEG, PNG hoặc WEBP.')
                    if getattr(uploaded, 'is_animated', False):
                        raise ValueError('Không hỗ trợ ảnh động. Hãy dùng ảnh tĩnh JPEG, PNG, WEBP hoặc video MP4/WebM.')
                image, _ = decode_image(base64.b64encode(path.read_bytes()).decode(), LIMITS['image_pixels'])
                fps, total = 0, 1
            else:
                with path.open('rb') as source:
                    signature = source.read(16)
                if (suffix == '.webm' and signature[:4] != b'\x1aE\xdf\xa3'
                        or suffix == '.mp4' and signature[4:8] != b'ftyp'):
                    raise ValueError('Nội dung tệp không phải container MP4 hoặc WebM hợp lệ.')
                capture = cv2.VideoCapture(str(path))
                if not capture.isOpened():
                    raise ValueError('Không đọc được video/codec. Hãy dùng MP4 H.264 hoặc WebM VP8/VP9.')
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                if not math.isfinite(fps) or not .1 <= fps <= 120:
                    raise ValueError('Video không có FPS hợp lệ; hãy chuyển sang MP4 H.264 hoặc WebM FPS cố định.')
                if total > LIMITS['video_frames'] or total / fps > LIMITS['video_seconds']:
                    raise ValueError('Video tối đa 60 giây và 600 khung hình. Hãy cắt video ngắn hơn.')
                dimensions = [capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)]
                if dimensions[0] * dimensions[1] > LIMITS['video_pixels'] or max(dimensions) > 1920:
                    raise ValueError('Video tối đa 1920×1080 pixel mỗi khung hình.')
                ok, image = capture.read()
                if not ok:
                    raise ValueError('Không giải mã được khung hình đầu tiên.')
                if image.shape[0] * image.shape[1] > LIMITS['video_pixels'] or max(image.shape[:2]) > 1920:
                    raise ValueError('Video tối đa 1920×1080 pixel mỗi khung hình.')
            height, width = image.shape[:2]
            with self.lock:
                job.update(width=width, height=height, source_fps=fps, total=total)
            if job['cancel'].is_set():
                terminal = 'cancelled'
                return
            init_start = time.perf_counter()
            import torch
            torch.set_num_threads(self.threads)
            model = self.get_model(MODEL_KEY)
            if job['cancel'].is_set():
                terminal = 'cancelled'
                return
            if self.runtime is None:
                model(np.zeros((640, 640, 3), np.uint8), imgsz=640, conf=.25, rect=False,
                      verbose=False, device=cpu_device(self.model_path))
                runtime = configure_cpu_backend(model, self.model_path, self.threads, strict_threads=False)
                if runtime.get('execution_devices') != ['CPU']:
                    raise RuntimeError('Chưa xác minh được thiết bị thực thi OpenVINO CPU.')
                model(np.zeros((640, 640, 3), np.uint8), imgsz=640, conf=.25, rect=False,
                      verbose=False, device=cpu_device(self.model_path))
                import openvino as ov
                runtime.update(device_name=ov.Core().get_property('CPU', 'FULL_DEVICE_NAME'), openvino_version=ov.__version__)
                for artifact in [*self.model_path.glob('*.xml'), *self.model_path.glob('*.bin')]:
                    with artifact.open('rb') as source:
                        runtime[artifact.suffix[1:] + '_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
                self.runtime = runtime
                job['timings']['initialization_ms'] = (time.perf_counter() - init_start) * 1000
            processing_start = time.perf_counter()
            with self.lock:
                job.update(status='running', runtime=self.runtime, message='Đang nhận diện thật bằng OpenVINO…')
            if job['kind'] == 'video':
                # VP8/WebM is browser-playable and available in the OpenCV FFmpeg build.
                writer = cv2.VideoWriter(str(job['folder'] / 'annotated.webm'), cv2.VideoWriter_fourcc(*'VP80'), fps, (width, height))
                if not writer.isOpened():
                    raise ValueError('OpenCV trên máy này không có bộ mã hóa WebM VP8; chưa thể xuất video.')
            while image is not None and not job['cancel'].is_set():
                if (job['processed'] >= LIMITS['video_frames'] or time.perf_counter() - processing_start > 180
                        or capture and job['processed'] / fps >= LIMITS['video_seconds']):
                    raise ValueError('Đã đạt giới hạn 600 khung hình hoặc 180 giây xử lý. Hãy dùng video ngắn hơn.')
                if image.shape[:2] != (height, width):
                    raise ValueError('Video thay đổi kích thước khung hình; hãy chuyển sang video kích thước cố định.')
                if writer and (job['folder'] / 'annotated.webm').stat().st_size > 128 * 1024**2:
                    raise ValueError('Video xuất vượt 128 MiB; hãy dùng video ngắn hoặc nhỏ hơn.')
                _, boxes, infer_ms, pipeline_ms = self.infer(model, image, job['settings'])
                annotated = self.annotate(image, boxes, job['settings'])
                if writer:
                    writer.write(annotated)
                else:
                    if not cv2.imwrite(str(job['folder'] / 'annotated.jpg'), annotated):
                        raise OSError('Không ghi được ảnh kết quả')
                frame = dict(index=job['processed'], time_seconds=job['processed'] / fps if fps else 0,
                             detections=boxes, inference_ms=infer_ms, pipeline_ms=pipeline_ms)
                if capture:
                    timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                    previous = job['frames'][-1]['source_time_seconds'] if job['frames'] else -1
                    frame['source_time_seconds'] = (timestamp if math.isfinite(timestamp) and timestamp >= 0 and timestamp > previous
                                                    else previous + 1 / fps if job['frames'] else 0)
                with self.lock:
                    job['frames'].append(frame)
                    job['processed'] += 1
                    job['progress'] = min(99, 100 * job['processed'] / (total or LIMITS['video_frames']))
                    job['timings']['inference_ms'] += infer_ms
                    job['timings']['pipeline_ms'] += pipeline_ms
                if not capture:
                    break
                ok, image = capture.read()
                if not ok:
                    image = None
            if capture and not job['cancel'].is_set() and job['processed'] < total - 1:
                raise ValueError('Video bị ngắt hoặc không giải mã được đầy đủ khung hình.')
            if writer:
                writer.release()
                writer = None
            job['timings']['processing_ms'] = (time.perf_counter() - processing_start) * 1000
            if not total:
                job['total'] = job['processed']
            terminal = 'cancelled' if job['cancel'].is_set() else 'completed'
            job['message'] = 'Đã dừng; kết quả chỉ gồm phần đã xử lý.' if job['cancel'].is_set() else 'Hoàn tất nhận diện.'
            job['progress'] = 100 if terminal == 'completed' else job['progress']
            self.export(job)
        except Exception as error:
            import traceback
            traceback.print_exc()
            job.update(message=str(error), files=[])
        finally:
            if capture:
                capture.release()
            if writer:
                writer.release()
            job['timings']['server_total_ms'] = (time.perf_counter() - started) * 1000
            if terminal == 'cancelled' and not job['files']:
                job['message'] = 'Đã hủy trước khi suy luận.'
            try:
                with self.lock:
                    job.update(status=terminal, touched=time.monotonic())
                    if job['delete']:
                        shutil.rmtree(job['folder'], ignore_errors=True)
                        self.jobs.pop(job['id'], None)
                    else:
                        if 'report.json' not in job['files']:
                            job['files'].append('report.json')
                        try:
                            self.write_report(job)
                        except OSError:
                            job.update(status='failed', message='Không ghi được báo cáo; kiểm tra dung lượng ổ đĩa.', files=[])
                        if job['kind'] == 'video':
                            job['frames'].clear()
            finally:
                self.slot.release()

    def export(self, job):
        folder = job['folder']
        (folder / 'detections.json').write_text(json.dumps(job['frames'], ensure_ascii=False), encoding='utf-8')
        with (folder / 'detections.csv').open('w', newline='', encoding='utf-8-sig') as output:
            writer = csv.writer(output)
            writer.writerow(['frame', 'time_seconds', 'class_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])
            for frame in job['frames']:
                for box in frame['detections']:
                    writer.writerow([frame['index'], frame['time_seconds'], box['class_id'], box['class_name'], box['confidence'], *box['xyxy']])
        job['files'] = ['detections.json', 'detections.csv', 'report.json']
        if job['processed']:
            job['files'].insert(0, 'annotated.webm' if job['kind'] == 'video' else 'annotated.jpg')

    def write_report(self, job):
        report = self.snapshot(job)
        n = job['processed']
        counts = {}
        for frame in job['frames']:
            for box in frame['detections']:
                counts[box['class_name']] = counts.get(box['class_name'], 0) + 1
        timings = job['timings']
        report.update(model=MODEL_NAME, precision='INT8', input_size=[640, 640], tracking=False,
                      accuracy_message='Chưa có nhãn chuẩn để đánh giá độ chính xác',
                      counts=counts, inference_mean_ms=timings['inference_ms'] / n if n else None,
                      counts_scope='Lượt phát hiện cộng dồn qua khung hình; không phải số đối tượng duy nhất.',
                      video_timebase='Video xuất ở FPS cố định theo metadata đầu vào; không giữ âm thanh.',
                      processing_fps=1000 * n / timings['processing_ms'] if timings['processing_ms'] else None,
                      timing_scope='Thời gian xử lý gồm tiền/hậu xử lý, suy luận, vẽ, mã hóa/ghi và đọc khung tiếp theo; '
                                   'loại upload, giải mã khung đầu và khởi tạo/warm-up. Toàn bộ trên server gồm các bước sau upload.',
                      unique_objects=None, audio_preserved=False)
        (job['folder'] / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    def evaluate(self, payload):
        """BDD100K frame JSON; Ultralytics matching and AP on completed image sessions."""
        import numpy as np
        import torch
        from types import SimpleNamespace
        from evaluate import canonical
        from ultralytics.engine.validator import BaseValidator
        from ultralytics.utils.metrics import ap_per_class, box_iou
        if not isinstance(payload, dict):
            raise ValueError('Payload đánh giá phải là JSON object')
        keys = payload.get('jobs')
        if not isinstance(keys, list) or not 1 <= len(keys) <= LIMITS['jobs'] or len(set(keys)) != len(keys):
            raise ValueError('Danh sách phiên đánh giá không hợp lệ')
        jobs = [self.get(key) for key in keys]
        if any(j['kind'] != 'image' or j['status'] != 'completed' or j['settings']['conf'] != .001 for j in jobs):
            raise ValueError('Đánh giá chỉ dùng ảnh đã hoàn tất trong chế độ có nhãn (conf=0.001).')
        if any(j['settings']['classes'] != jobs[0]['settings']['classes'] for j in jobs):
            raise ValueError('Các ảnh phải dùng cùng bộ lọc lớp')
        names = {int(k): v for k, v in self.config()['names'].items()}
        mapping = {canonical(v): k for k, v in names.items()}
        frames = payload.get('ground_truth')
        if isinstance(frames, dict):
            frames = frames.get('frames')
        if not isinstance(frames, list) or len(frames) > 10000:
            raise ValueError('Ground truth phải là list BDD100K frame hoặc {"frames": [...]} (tối đa 10.000 ảnh).')
        by_name = {}
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(frame.get('name'), str) or frame['name'] in by_name:
                raise ValueError('Ground truth phải có tên ảnh duy nhất')
            by_name[frame['name']] = frame
        stats = []
        skipped_non_detection = 0
        if len({j['filename'] for j in jobs}) != len(jobs):
            raise ValueError('Tên ảnh trong phiên bị trùng; không thể ghép ground truth')
        for job in jobs:
            frame = by_name.get(job['filename'])
            if frame is None or not isinstance(frame.get('labels'), list):
                raise ValueError(f"Thiếu ground truth/labels cho {job['filename']}")
            if len(frame['labels']) > 1000:
                raise ValueError('Mỗi ảnh tối đa 1.000 đối tượng ground truth trong phiên MVP.')
            targets = []
            for label in frame['labels']:
                if not isinstance(label, dict):
                    raise ValueError('Mỗi nhãn phải là JSON object')
                if label.get('category') in ('lane', 'drivable area') and label.get('box2d') is None:
                    skipped_non_detection += 1
                    continue
                if canonical(label.get('category', '')) not in mapping:
                    raise ValueError('Ground truth chứa lớp không thuộc 10 lớp BDD100K')
                attrs = label.get('attributes') or {}
                if not isinstance(attrs, dict):
                    raise ValueError('attributes phải là JSON object')
                if any(attrs.get(k) for k in ('ignore', 'ignored', 'crowd', 'iscrowd')) or any(label.get(k) for k in ('ignore', 'ignored', 'crowd', 'iscrowd')):
                    raise ValueError('Phiên MVP chưa hỗ trợ ignore/crowd; hãy dùng evaluator chuyên biệt.')
                cid = mapping[canonical(label['category'])]
                box = label.get('box2d', {})
                if not isinstance(box, dict) or any(k not in box for k in ('x1', 'y1', 'x2', 'y2')):
                    raise ValueError('Nhãn cần box2d có x1, y1, x2, y2')
                xyxy = [float(box[k]) for k in ('x1', 'y1', 'x2', 'y2')]
                if not all(math.isfinite(v) for v in xyxy) or not (0 <= xyxy[0] < xyxy[2] <= job['width'] and 0 <= xyxy[1] < xyxy[3] <= job['height']):
                    raise ValueError('Tọa độ ground truth không hợp lệ hoặc nằm ngoài ảnh')
                if cid in job['settings']['classes']:
                    targets.append([cid, *xyxy])
            target = torch.tensor(targets, dtype=torch.float32).reshape(-1, 5)
            predictions = job['frames'][0]['detections']
            pred = torch.tensor([[*b['xyxy'], b['confidence'], b['class_id']] for b in predictions], dtype=torch.float32).reshape(-1, 6)
            iou = box_iou(target[:, 1:], pred[:, :4])
            correct = BaseValidator.match_predictions(SimpleNamespace(iouv=torch.linspace(.5, .95, 10)), pred[:, 5], target[:, 0], iou)
            stats.append((correct.numpy(), pred[:, 4].numpy(), pred[:, 5].numpy(), target[:, 0].numpy()))
        tp, conf, pred_cls, target_cls = [np.concatenate([s[i] for s in stats], axis=0) for i in range(4)]
        if not len(target_cls):
            raise ValueError('Không có đối tượng ground truth trong các lớp đã chọn; mAP không xác định.')
        result = ap_per_class(tp, conf, pred_cls, target_cls, names=names)
        return dict(images=len(jobs), ground_truth_objects=len(target_cls), precision=float(result[2].mean()),
                    skipped_non_detection_labels=skipped_non_detection,
                    recall=float(result[3].mean()), map50=float(result[5][:, 0].mean()), map50_95=float(result[5].mean()),
                    per_class=[dict(class_name=names[int(cid)], map50=float(result[5][i, 0]), map50_95=float(result[5][i].mean())) for i, cid in enumerate(result[6])],
                    protocol='Ultralytics AP, IoU 0.50:0.95; P/R macro tại điểm F1 tối ưu; conf=0.001, max_det=300, '
                             'chỉ lớp có GT; không phải evaluator BDD100K chính thức.', jobs=keys)

    def handle(self, handler, method):
        parsed = urlsplit(handler.path)
        if not parsed.path.startswith('/api/mvp/'):
            return False
        self.cleanup()
        try:
            parts = parsed.path.strip('/').split('/')
            if method == 'GET' and parts == ['api', 'mvp', 'config']:
                handler.send_json(200, self.config())
            elif method == 'POST' and parts == ['api', 'mvp', 'jobs']:
                length = int(handler.headers.get('Content-Length', '0'))
                handler.connection.settimeout(30)
                handler.send_json(202, self.submit(handler.rfile, length, parse_qs(parsed.query)))
            elif method == 'POST' and parts == ['api', 'mvp', 'evaluate']:
                length = int(handler.headers.get('Content-Length', '0'))
                if not 0 < length <= 12 * 1024**2:
                    raise ValueError('Ground truth tối đa 12 MiB')
                handler.send_json(200, self.evaluate(json.loads(handler.rfile.read(length))))
            elif len(parts) >= 4 and parts[:3] == ['api', 'mvp', 'jobs']:
                job = self.get(parts[3])
                if method == 'GET' and len(parts) == 4:
                    handler.send_json(200, self.snapshot(job))
                elif method == 'POST' and parts[4:] == ['cancel']:
                    job['cancel'].set()
                    handler.send_json(200, {'message': 'Đã yêu cầu dừng'})
                elif method == 'DELETE' and len(parts) == 4:
                    with self.lock:
                        job['cancel'].set()
                        if job['status'] in ('starting', 'running'):
                            job['delete'] = True
                        else:
                            shutil.rmtree(job['folder'], ignore_errors=True)
                            self.jobs.pop(job['id'], None)
                    handler.send_json(200, {'message': 'Đã xóa dữ liệu phiên'})
                elif method == 'GET' and len(parts) == 6 and parts[4] == 'files' and parts[5] in job['files']:
                    self.serve_file(handler, job['folder'] / parts[5])
                else:
                    raise KeyError('Tài nguyên không tồn tại hoặc chưa hoàn tất')
            else:
                raise KeyError('Endpoint không tồn tại')
        except FileNotFoundError as error:
            handler.send_json(503, {'message': str(error)})
        except KeyError:
            handler.send_json(404, {'message': 'Phiên/tài nguyên không tồn tại hoặc đã hết hạn'})
        except (ValueError, TypeError, OverflowError) as error:
            handler.send_json(400, {'message': str(error)})
        except RuntimeError as error:
            handler.send_json(503, {'message': str(error)})
        except Exception:
            import traceback
            traceback.print_exc()
            handler.send_json(500, {'message': 'Lỗi nội bộ MVP; xem log server để biết nguyên nhân.'})
        return True

    @staticmethod
    def serve_file(handler, path):
        import mimetypes
        size = path.stat().st_size
        start, end = 0, size - 1
        value = handler.headers.get('Range')
        if value:
            match = re.fullmatch(r'bytes=(\d+)-(\d*)', value)
            if not match:
                handler.send_json(416, {'message': 'Range không hợp lệ'})
                return
            start = int(match[1])
            end = min(int(match[2]) if match[2] else end, end)
            if start > end:
                handler.send_json(416, {'message': 'Range ngoài tệp'})
                return
        handler.send_response(206 if value else 200)
        handler.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        handler.send_header('Content-Length', str(end - start + 1))
        handler.send_header('Accept-Ranges', 'bytes')
        if value:
            handler.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        handler.end_headers()
        with path.open('rb') as source:
            source.seek(start)
            remaining = end - start + 1
            while remaining:
                block = source.read(min(65536, remaining))
                if not block:
                    break
                handler.wfile.write(block)
                remaining -= len(block)
