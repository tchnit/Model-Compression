# Báo cáo hiện trạng và kiểm toán dự án YOLO26 × BDD100K

Cập nhật ngày 18/10/2026 sau rà soát mã nguồn, runtime và bằng chứng.
**Trạng thái: demo hoạt động; chưa đủ bằng chứng nghiệm thu toàn bộ dự án.**

## Chuyển baseline sang FP16 (18/10/2026)

Theo yêu cầu người dùng, demo, evaluator và benchmark latency hiện yêu cầu
PyTorch CPU FP16 thực: trọng số và input float16, CPU 4 luồng, 640×640.
Helper cấu hình backend sau init vì Ultralytics tự bỏ FP16 trên CPU;
validator cũng được đặt input float16 và kiểm tra dtype thực.
Các kết quả FP32 bên dưới là lịch sử, không đại diện cho baseline FP16 mới.
Accuracy baseline và Δ pp chờ chạy lại đủ 10.000 ảnh; không dùng mAP smoke.
Dùng `bash rebenchmark_all.sh` để chạy lần lượt m FP16/PTQ/QAT rồi s FP16/PTQ/QAT,
đánh giá đầy đủ và xuất lại CSV/Markdown/demo từ đúng hai report mới.
Sáu model vẫn có đầy đủ các cột chỉ số; ô chưa có kết quả FP16 để Chưa đo.
Các artifact Hardware FP32 thực và khóa model/API cũ giữ nguyên để tương thích.
Bản lưu trước thay đổi ở `verification/before_fp16_20261018_152947/`;
trạng thái trình duyệt cũ được lưu tại localStorage hậu tố `:before-fp16`.
Ngưỡng nghiệm thu không tự thay đổi theo precision; xác nhận protocol cũ bị hủy.

## Benchmark Hardware mới (18/10/2026)

Đã bổ sung mục `demo.html#hardware`, độc lập với bảng CPU/accuracy bên dưới.
PyTorch, ONNX Runtime, TensorRT và OpenVINO dùng phần cứng máy backend;
ONNX Runtime Web WASM/WebGPU dùng máy mở trình duyệt. Chỉ bấm Chạy mới đo.
Phát hiện thiết bị, kiểm tra artifact/precision, chặn cấu hình không hỗ trợ;
không tự thay GPU bằng CPU. Warmup, p50/p95, FPS, setup/load, từng mẫu đo,
SHA256 ảnh/model, version/provider/device và JSON export được lưu theo phép đo.

Export riêng qua `prepare_runtime_models.py`, manifest trong `runtime_exports/`;
không sửa checkpoint gốc hoặc calibration trên tập validation. INT8 hiện chỉ
dùng các graph OpenVINO PTQ/QAT có sẵn. TensorRT FP16 là policy mixed precision.
Các artifact export mới chưa có đánh giá mAP 10.000 ảnh riêng; không dùng
chúng để thay số accuracy lịch sử. WebGPU cần browser/adapter thực hỗ trợ.
Xem hướng dẫn, giới hạn đo và API ở cuối `README.txt`.

## 1. Kết luận có bằng chứng

- Có 10 artifact BDD100K trong danh sách demo, không có cấu hình 480.
- Sáu cấu hình benchmark có hash artifact khớp báo cáo accuracy gốc và cùng manifest 10.000 ảnh.
- YOLO26m baseline: mAP50–95 33,06%; PTQ INT8: 32,76%, giảm khoảng 0,30 điểm phần trăm.
- Dung lượng thực: 42,022485 MiB → 20,488908 MiB, giảm **51,24299%**.
- Kiểm kê trực tiếp xác nhận checkpoint nguồn lưu FP16, còn PyTorch suy luận FP32. **51,24299% là giảm so với checkpoint lưu FP16**, không phải mốc lưu FP32 chuẩn hóa. Ngưỡng ≥60% so với FP32 chưa đủ mốc để kết luận; đã bỏ dung lượng 42,02 MiB khỏi cả ô gói lưu FP32 và nguồn sạch chuẩn hóa. Không có bằng chứng cho mốc 52,4 MiB cũ.
- Baseline/QAT accuracy giữ từ log gốc. PTQ đã chạy lại đủ 10.000 ảnh trong `verification/full_ptq_retry/`: mAP50–95 **32.76336147%**, chênh **0.00000000 điểm phần trăm** so với log gốc. Hash model/annotation và danh sách ảnh/nhãn khớp; đây là chạy lại trong môi trường hiện có.
- Latency đã đo lại trên cùng 50 ảnh × 3 lượt, warmup 5, batch 1, CPU 4 luồng, rect=False, input 640×640, conf=0.001, iou=0.7, max_det=300.
- Thời gian đo gồm preprocess + inference + postprocess, không gồm đọc ảnh, compile/load model hoặc warmup. Đây là snapshot trên máy dùng chung; tải nền/nhiệt độ có thể ảnh hưởng.

## 2. Nguồn số liệu và giới hạn

| Nội dung | Nguồn |
|---|---|
| YOLO26m baseline/PTQ accuracy | `evaluation_int8_yolo26m_640/report.json` |
| YOLO26m QAT accuracy | `evaluation_int8_qat_yolo26m_640/INT8/metrics.json` |
| YOLO26s baseline/PTQ accuracy | `evaluation_int8_yolo26s_640/report.json` |
| YOLO26s QAT accuracy | `evaluation_int8_qat_yolo26s_640/INT8/metrics.json` |
| Latency mới, hash ảnh/model, runtime thực, từng mẫu đo | `demo_data/latency_benchmark.json` |
| Bảng tổng hợp | `benchmark_comparison.csv`, `benchmark_comparison.md` |
| Kết quả ảnh demo | `demo_data/detections.json`, `demo_data/demo_bundle.js` |

Các log cũ giữ nguyên để truy vết. Baseline trong report QAT YOLO26m là YOLO26s, nên không dùng phép so sánh baseline/candidate của report đó để kết luận YOLO26m. Bảng tổng hợp ghép accuracy với baseline cùng họ/kích thước/protocol, kiểm tra hash và dùng latency mới cùng protocol. Không chọn lần chạy nhanh nhất để loại trùng.

mAP là Ultralytics detection mAP sau chuyển nhãn BDD100K, không phải official BDD evaluator với đầy đủ ignore/crowd rules. Precision/Recall tại điểm max-F1 không chứng minh tiết kiệm thời gian gán nhãn ở ngưỡng demo. Chưa đo mức tiết kiệm thao tác của người gán nhãn.

## 3. Các mục chưa xác minh

- Accuracy FP16 YOLO26m riêng: artifact `models/fp16.pt`, `models/fp16_openvino_model` cũ thuộc YOLO26s. Mới có export FP16 YOLO26m trong `runtime_exports/` cho benchmark hardware, chưa đánh giá mAP riêng. Không có căn cứ cho mAP 33% / latency 140 ms cũ; đã bỏ số đo này khỏi dashboard.
- Gói B0 YOLO26m lưu FP32 chuẩn hóa độc lập. Dtype checkpoint nguồn đã xác minh là FP16; cast lên FP32 không khôi phục độ chính xác đã mất.
- Quy trình calibration/huấn luyện QAT, manifest calibration và sự độc lập với tập final.
- Chia development/final, nghiệm thu protocol/ngưỡng và log tái lập độc lập từ môi trường sạch.
- Đánh giá riêng theo ngày/đêm/thời tiết/kích thước vật thể, thời gian nạp model.

Dashboard đưa các nhiệm vụ chưa đối chiếu về “Chờ kiểm tra” hoặc “Chưa bắt đầu”; không tự gán 28/28 task và 4/4 tiêu chí PASS. Tệp model và log lịch sử không bị xóa.

## 4. Chín vấn đề đã xử lý

1. Sửa dung lượng và kết luận nghiệm thu theo artifact thật; dùng MiB thống nhất.
2. Demo dùng input vuông 640×640 cho PyTorch/OpenVINO; ép và đọc lại OpenVINO CPU 4 luồng. Module chung: `inference_runtime.py`.
3. Aggregator giữ các protocol khác nhau, không chọn FPS cao nhất; không tính speedup từ latency chưa xác minh/khác protocol. Snapshot mới được đối chiếu hash trước khi dùng.
4. Gỡ các số đo và bằng chứng không tồn tại; ghi rõ phần chưa xác minh.
5. Xuất nhãn YOLO 5 cột, bỏ confidence khỏi nhãn, dùng tên `.txt` cùng stem ảnh.
6. Render tên file bằng DOM/textContent, không chèn tên file vào innerHTML.
7. Tách trạng thái trống khỏi dữ liệu đã đối chiếu; reset, reload và import trạng thái 0 task done không còn bị thay bằng completed state.
8. Sửa đường dẫn mặc định PTQ/QAT/YOLO26s, kiểm tra file trước khi chạy CLI.
9. Nhận diện QAT từ artifact/nhãn; báo lỗi khi thiếu model/chỉ số thay vì tạo kết quả 0%.

Các chức năng chọn A/B mới suy luận, hàng đợi/cache theo ảnh-model, slider và overlap/retention được giữ. Export chỉ bật khi model B có kết quả.

## 5. Chạy lại

```bash
.venv/bin/python run_demo.py
# http://127.0.0.1:8765/demo.html
# http://127.0.0.1:8765/model-compression-yolo26-dashboard.html

# Đo lại latency và dự đoán ảnh demo khi máy rảnh:
.venv/bin/python refresh_demo_data.py
.venv/bin/python compare_benchmarks.py --csv benchmark_comparison.csv --markdown benchmark_comparison.md

# Đánh giá đối đầu PTQ/QAT bằng model hiện có (smoke 50 ảnh):
.venv/bin/python compare_ptq_vs_qat.py --smoke --output evaluation_ptq_qat_smoke
```

Mốc speedup CSV là YOLO26m baseline 640×640 của phép đo mới. Bộ lọc YOLO26s trên demo lấy baseline YOLO26s cùng phép đo. Không dùng 227 ms cố định nữa.

## 6. Kiểm tra hồi quy

```bash
.venv/bin/python -m unittest discover -v
node test_demo_overlap.js
node test_demo_selection.js
node test_demo_exports.js
node test_dashboard_state.js
```

Đã đạt 23 kiểm thử Python và bốn bộ kiểm thử JavaScript. Đã chạy smoke evaluator thực trên hai ảnh với baseline và PTQ; số liệu smoke không đưa vào bảng benchmark.

Có kiểm tra runtime thực cho shape 640×640, PyTorch 4 luồng và OpenVINO CPU 4 luồng. Các kiểm tra còn lại bao phủ chuyển nhãn, protocol benchmark, parser PTQ/QAT, API, chọn model, upload, hàng đợi/cache, overlap, export và trạng thái dashboard.

Dashboard tự sao lưu dữ liệu cũ trước migration ở localStorage key `compression-lab-yolo26-bdd100k-v1:before-audit-v2`; thay các giá trị mặc định cũ nhận diện được, giữ ghi chú/bằng chứng riêng. Task hoàn tất nhưng có phụ thuộc chưa xác minh được chuyển về chờ kiểm tra, kèm ghi chú lý do. Bản sao này không thay cho xuất JSON để bàn giao.

## 7. Bằng chứng và sửa tiếp ngày 18/10/2026

- `verify_project.py` kiểm kê 10 model: SHA-256, lớp, head/dtype checkpoint, graph/Constant/FakeQuantize OpenVINO. Kiểm kê không tự chứng minh cách tạo model hoặc huấn luyện QAT.
- Sáu IR OpenVINO đã chạy probe trong Python riêng với OpenVINO/NumPy, không import PyTorch và không nạp checkpoint `.pt`. Xác nhận CPU 4 luồng, 1 stream, output hữu hạn. Đây là kiểm tra nạp/shape bằng zero-input; không thay accuracy. Thời gian read/compile/first inference ghi rõ cache/tải máy chưa kiểm soát.
- Giải mã và hash toàn bộ 10.000 ảnh, đối chiếu annotation, ghi phân bố điều kiện/lớp. Phát hiện 2 nhóm ảnh trùng nội dung; liệt kê trong `verification/data_inventory.json`, giữ nguyên tập benchmark đã dùng.
- `verification/protocol.yaml` mô tả cấu hình quan sát được, kèm hash ảnh, nhãn, inventory model và môi trường. Đây không phải xác nhận trước thực nghiệm của nhóm.
- `verification/environment.lock.json` đối chiếu dependency với requirements, chạy pip check và ghi hash mã nguồn/asset. Chưa cài lại từ môi trường sạch.
- Chạy thật baseline/PTQ/QAT trên cùng 50 ảnh: `verification/smoke_ptq_qat/report.json`. Không đưa mAP subset vào bảng accuracy 10.000 ảnh.
- Sửa CPU runtime cho backend mới do `val()` tạo ra, ghi riêng `runtime.validation`; kiểm tra thêm OpenVINO NUM_STREAMS=1.
- Các manifest evaluator mới thêm hash nội dung ảnh. Log cũ chỉ có hash nhãn được giữ nguyên; không tuyên bố đã chứng minh nội dung ảnh tại thời điểm chạy lịch sử.
- Snapshot latency 640 không còn bị dùng cho record 480; snapshot khác số luồng cũng không được xác nhận là cùng protocol.
- Upload sai kiểu Base64/model/conf hoặc dữ liệu không phải ảnh trả 400. Parser báo cáo từ chối NaN, infinity và số đo thời gian/dung lượng không dương.
- Preview/sao chép/tải JSON dùng chung schema; metadata lấy đúng annotation ảnh mẫu, upload để trống khi chưa biết. Chuẩn hóa person/bike/motor sang pedestrian/bicycle/motorcycle. Sửa stem tên JSON cho PNG/JPEG.
- Dashboard lấy tóm tắt YOLO26s QAT từ cùng dữ liệu benchmark; bỏ số cứng 35,3 ms / 6,42× cũ. Kế hoạch weight-only/Safetensors được ghi là kế hoạch ban đầu, khác pipeline PyTorch/OpenVINO hiện có.
- Migration v5 sao lưu bản hiện tại tại `compression-lab-yolo26-bdd100k-v1:before-audit-v5`, giữ bản v2 nếu đã tồn tại; sửa các mặc định cũ nhận diện được, giữ ghi chú riêng. Reset vẫn tạo trạng thái trống.

Lệnh kiểm kê: `.venv/bin/python verify_project.py`. Kết quả tái chạy/hash/manifest được tổng hợp trong `verification/reproduction.json`.

Các mục chưa thể nghiệm thu: nguồn/revision và log tạo artifact, calibration/QAT, development/final độc lập, gói FP32 chuẩn hóa/đối chứng FP16 riêng, tái lập trong môi trường sạch và xác nhận ngưỡng của nhóm. Không tạo lại một split validation đã sử dụng rồi gọi đó là held-out final.

Lượt đánh giá toàn bộ đầu tiên bị SIGTERM (exit 143) ở 9.490/10.000 ảnh; chưa có metrics cuối. Giữ log/status trong `verification/full_ptq/`; lượt chạy lại tách biệt ở `verification/full_ptq_retry/`, vẫn cùng protocol CPU 4 luồng. Không dùng lượt bị ngắt làm bằng chứng accuracy.

Lượt PTQ chạy lại đã hoàn tất đủ 10.000 ảnh và 150 mẫu latency. `verification/reproduction.json` đối chiếu report/hash/manifest với log lịch sử; không ghép latency của lượt một model này với baseline của snapshot khác để tính speedup.
API live đã kiểm tra bằng ảnh thật: `verification/live_api_probe.json`. Log kiểm thử: `verification/tests.log`, `verification/frontend-tests.log`; log smoke: `verification/smoke_ptq_qat/run.log`.


## Kiểm chứng Benchmark Hardware

- Hoàn tất bốn ONNX (FP32/FP16 × YOLO26m/s), hai OpenVINO FP32 và bốn TensorRT
  engine FP32/FP16; manifest ghi hash model nguồn, file export và GPU/version build.
- Máy thực: Intel i7-11800H; NVIDIA RTX 3070 Laptop GPU; Torch CUDA 13.0,
  ONNX Runtime GPU 1.30.0, TensorRT 10.16.1.11, OpenVINO 2026.4.1.
- `verification/hardware_smoke.json`: **28 cấu hình native thành công, 0 lỗi**,
  bao gồm PyTorch CPU/CUDA, ORT CPU/CUDA, TensorRT CUDA, OpenVINO CPU/GPU,
  FP32/FP16 và các graph PTQ/QAT INT8 phù hợp. Warmup1, đo3, chỉ xác nhận chức
  năng trên một ảnh; máy dùng chung và đang build engine, không dùng làm xếp hạng.
- `verification/hardware_ui_export.json`: JSON tải từ giao diện chứa ba phép đo
  thật: PyTorch CUDA FP16, TensorRT CUDA FP16 và Browser CPU WASM FP32. Cùng ảnh,
  đều trả 13 hộp. Đã đối chiếu số mẫu, SHA256, p50/p95 và scope.
- Trình duyệt kiểm tra **không có WebGPU adapter**. Đã xác minh thông báo và khóa
  nút chạy; chưa xác minh inference WebGPU trên GPU trình duyệt thực. Test JS có
  chuyển FP16, letterbox, decode box, percentile và phát hiện software fallback
  theo API mới (`adapter.info.isFallbackAdapter`) và API cũ.
- Đổi model/runtime/precision không tự chạy; cấu hình thiếu INT8 khóa nút chạy.
  Backend trả503 khi bận, runtime lỗi không retry bằng CPU, trả422 và nhả semaphore.
- `verification/tests-hardware.log`: **27 test Python PASS**.
  `verification/frontend-hardware-tests.log`: **5 bộ test JS PASS**.
- ONNX Runtime Web 1.30.0 được lưu cục bộ cùng LICENSE, ThirdPartyNotices và hash;
  không phụ thuộc CDN khi chạy. Nguồn: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
  và https://www.w3.org/TR/webgpu/.

Bảng benchmark CPU/accuracy lịch sử không bị cập nhật bằng các microbenchmark mới.
Không chạy lại mAP 10.000 ảnh cho export mới và không tuyên bố mọi tổ hợp
runtime × precision đều hỗ trợ. Cần artifact/calibration phù hợp để mở INT8
trên ONNX Runtime/TensorRT/WebGPU/PyTorch; OpenVINO FP16 compute hiện không
được phần cứng này công bố hỗ trợ.

## Ưu tiên ONNX Runtime Web và WebGPU

Theo yêu cầu tiếp theo, phần #hardware đổi thành “ONNX Runtime Web & WebGPU”,
mặc định CPU (WASM), tiếp theo GPU (WebGPU). Runtime native được giữ trong
nhóm Nâng cao. Cả hai chế độ browser dùng ONNX Runtime Web vendored 1.30.0.

- Browser đọc `demo_data/browser_models.json` thay vì API capabilities Python.
  CLI export tự xuất manifest công khai chỉ gồm model ONNX, URL và hash nguồn.
  Chỉ YOLO26m/s FP32/FP16; chưa có ONNX INT8 calibration hay QAT cho browser.
- Hiển thị phần cứng máy mở trang, phân biệt secure context, API WebGPU,
  adapter, software fallback và shader-f16. Thay đổi runtime chỉ tải cấu hình,
  không tự suy luận. Kết quả ghi rõ ONNX Runtime Web CPU/WASM hoặc GPU/WebGPU.
- Kiểm thử bằng web server tĩnh ở localhost:8765, không có API AI. ONNX Runtime
  Web CPU FP32 chạy 2 lượt, trả 13 hộp. Log xác nhận chỉ GET asset/model,
  GET /api/status trả404 (chip trạng thái A/B), không có native capability API
  hoặc POST ảnh/benchmark. Bằng chứng: `verification/browser_static.log`,
  `verification/browser_static_result.json`. Server tạm đã dừng.
- Bản demo chính cũng chạy browser CPU thành công; bằng chứng
  `verification/browser_only_result.json`. WebGPU bị khóa với lý do không lấy
  được adapter; không tạo số đo GPU từ phép đo CPU.
- `verification/browser-runtime-python-tests.log`: 28 test Python PASS;
  `verification/browser-runtime-js-tests.log`: 5 bộ test JavaScript PASS,
  gồm capability browser độc lập backend và các nhánh WebGPU không hỗ trợ.

Chưa kiểm thử inference WebGPU trên adapter phần cứng thực trong trình duyệt
này. Không đo lại mAP hoặc 28 cấu hình native vì thay đổi chỉ ở luồng chọn /
khởi tạo browser và xuất manifest; trọng số ONNX giữ nguyên hash.


## Bổ sung FP16 và INT8 cho trình duyệt

Đã bỏ giới hạn FP32-only không cần thiết trên WASM: ONNX FP16 hiện có chạy được
trong ONNX Runtime Web CPU. UI hiện trạng thái riêng FP32/FP16/INT8, không
đồng nhất precision của graph với precision của mọi kernel phần cứng.

Đã tạo hai ONNX PTQ QDQ mới từ cùng FP32 end-to-end export: YOLO26s và YOLO26m.
Calibration chọn 128 ảnh train từ danh sách có sẵn trong project nén model,
seed42; kiểm tra tên lẫn SHA256 so với manifest 10.000 ảnh validation và từ chối
nội dung trùng trong calibration. Không dùng validation làm calibration.
Bằng chứng: `verification/browser_int8_calibration.json` (không public qua HTTP).
INT8 Conv thực: YOLO26s 102 lớp, YOLO26m 112 lớp; activations uint8, weights int8
per-channel, các phép toán còn lại giữ floating point. Input/output vẫn FP32.
Dung lượng mới khoảng 9,78 / 20,30 MiB. Không sửa artifact OpenVINO gốc, checkpoint
hay accuracy/latency lịch sử. Chưa có mAP riêng cho hai ONNX INT8 mới.

Manifest browser có đủ sáu artifact: FP32/FP16/INT8 × YOLO26s/m, kèm input dtype,
precision policy, hash calibration và số Conv lượng tử hóa. Server chỉ mở thêm
hai URL ONNX INT8; không public calibration manifest, weights native hoặc engine.
JSON kết quả ghi graph_precision, input_dtype, output_dtype và nguồn calibration.

Lượt export đầu dừng do `CalibMaxIntermediateOutputs=1` trong ONNX Runtime 1.30
xóa toàn bộ dữ liệu trước khi tổng hợp range. Đã bỏ tùy chọn đó, dùng MinMax
mặc định, chạy lại đủ 128 ảnh mỗi model và kiểm tra graph / output hữu hạn.
Log đầu: `verification/browser_int8_export_attempt1.log`; lượt hoàn tất:
`verification/browser_int8_export.log`. Không publish artifact từ lượt lỗi.


Kiểm thử precision mới: `verification/precision-python-tests.log` có 29 test
Python PASS; `verification/precision-api-tests.log` có 12 test API/calibration
PASS; `verification/precision-js-tests.log` có 5 bộ JavaScript PASS. HTTP HEAD
cả sáu ONNX trả200, calibration manifest trả404. Bộ test chặn ảnh validation,
trùng nội dung và sai split, kiểm tra WASM FP16/INT8 và WebGPU shader-f16.

Kết quả kiểm thử browser được ghi riêng tại `verification/browser_precision_results.json`.
Đây là smoke một ảnh với warmup1, đo2, không phải đánh giá accuracy hoặc bảng
xếp hạng tốc độ. FP16 CPU có thể promote một số kernel sang FP32; INT8 giữ
các phép toán ngoài Conv bằng số thực. WebGPU vẫn chưa có adapter trong trình
duyệt kiểm tra, nên không tuyên bố đã đo GPU WebGPU.


## Preview benchmark có box phát hiện

Preview được phóng lớn, dùng renderer SVG chung với A/B để hiển thị box, tên
lớp và confidence theo kết quả vừa chạy. ViewBox dùng kích thước ảnh gốc; chữ
co theo kích thước hiển thị. Đổi ảnh/model/runtime/device/precision/conf sẽ ẩn
annotation không còn phù hợp; bắt đầu chạy lại xóa box cũ, kể cả khi lượt mới
lỗi. Vẽ preview sau đo nên không cộng vào latency.

Năm bộ test JavaScript PASS: `verification/preview-js-tests.log`. Kiểm thử
ONNX Runtime Web WASM INT8 thực: 11 box và 11 nhãn, ảnh/overlay có cùng tọa độ
và kích thước, viewBox 1280×720. Chuyển ảnh/model/precision/conf không tự chạy
thêm inference; box chỉ hiện khi kết quả khớp lựa chọn. Bằng chứng:
`verification/preview_ui_probe.json`. Chưa thử WebGPU vì trình duyệt không
có adapter. Không thay đổi model hoặc đo lại mAP.


## Tùy chọn số luồng CPU WASM

Thêm selector 1 đến số CPU logic do trình duyệt báo (mặc định 1), chỉ mở khi
chạy WASM và có cross-origin isolation + shared WebAssembly memory. Nếu thiếu
điều kiện, chỉ cho 1 luồng và hiện lý do. Server gửi COOP same-origin và COEP
require-corp để bật shared memory; không thay cấu hình CPU backend 4 luồng.
Mỗi lượt WASM khởi tạo worker riêng vì thread pool ORT không đổi sau init.
Dùng lại cùng mã suy luận với WebGPU, tiền xử lý worker bằng OffscreenCanvas.
Worker/khởi tạo model không nằm trong latency. Bảng kết quả, caption và JSON
ghi cpu_threads; protocol thêm cpu_threads_requested và cross_origin_isolated.
Đổi số luồng ẩn preview cũ, không tự chạy benchmark.

Kiểm tra: 5 bộ JavaScript PASS (`verification/threads-js-tests.log`), 7 test
API PASS bao gồm header trang/worker/module (`verification/threads-api-tests.log`).
Browser thật: INT8 đổi liên tiếp 4 → 1 → 2 luồng không reload, mỗi lượt 13 box,
protocol ghi đúng số luồng đã chọn. FP16 và FP32 cũng chạy được với 2 luồng;
WebGPU khóa selector. Lượt cuối INT8 4 luồng warmup1/repeats3: p50 416,74 ms;
đây chỉ là smoke một ảnh, không phải benchmark accuracy hay kết luận tốc độ.
Bằng chứng `verification/browser_threads_results.json`. Không export lại model
hoặc chạy lại mAP; chưa kiểm thử inference WebGPU do không có adapter.

## Lựa chọn Winner trên dashboard

Theo yêu cầu người dùng, `int8_yolo26s_640_openvino_model` là Winner của bảng benchmark. Chỉ dòng này mang badge Winner và highlight, kể cả khi lọc YOLO26s hoặc INT8; bộ lọc YOLO26m không gán Winner thay thế. Badge thông tin model A/B được đồng bộ và YOLO26m PTQ trở thành đối chứng. Đây là lựa chọn model hiển thị; không thay đổi accuracy, latency, speedup hay các mốc nghiệm thu.

## Kết quả chuyển FP16 đã kiểm tra

Đã đo đủ 6 model tuần tự, mỗi model 50 ảnh ×3 (150 mẫu), CPU 4 luồng, 640×640.
Mean latency: m FP16 4195,45 ms; m PTQ 134,08 ms; m QAT 158,09 ms;
s FP16 1311,01 ms; s PTQ 63,20 ms; s QAT 49,55 ms.
FP16 trên CPU máy này chậm hơn FP32 lịch sử; speedup không tách tác động backend/precision.
Snapshot, CSV/Markdown, JSON/bundle và dashboard đã đồng bộ. Winner vẫn là s PTQ.
Runtime xác nhận parameter_dtype/input_dtype torch.float16 cho hai baseline.
Đã chạy smoke 2 ảnh ×6 model; riêng baseline đã kiểm tra lại report một ảnh mỗi họ
để xác nhận cả validation lẫn predictor ghi dtype backend thực, không ghi dtype bản
checkpoint riêng của YOLO (hai đối tượng khác nhau trong AutoBackend hiện tại).
Các report dtype mới: verification/fp16_dtype_m_20261018/report.json và
verification/fp16_dtype_s_retry_20261018/report.json. Smoke không đưa vào accuracy bảng chính.

31 kiểm thử Python PASS, 5 bộ JS PASS, 10 kiểm tra mục tiêu cuối PASS.
Browser thật: chọn YOLO26m Baseline (FP16) trả 13 hộp; bảng giữ sáu dòng, bộ lọc
YOLO26s dùng đúng mốc cùng họ; dashboard B0 và default Hardware là FP16.
Chi tiết: verification/fp16_validation_summary.json và các log được tham chiếu.
Chưa chạy lại accuracy FP16 đủ 10.000 ảnh. Các ô baseline mAP/P/R/F1 và Δ pp
hiển thị Chưa đo/N/A; chạy bash rebenchmark_all.sh để đánh giá đủ và cập nhật.

## Gán tạm benchmark FP32 lịch sử

Theo yêu cầu người dùng, bảng/demo/KPI và CSV/Markdown tạm dùng trọn lượt benchmark trước FP16 từ verification/before_fp16_20261018_152947/. Sáu model cùng protocol cũ; không ghép latency FP32 baseline với INT8 của lượt mới. Mọi dòng mang benchmark_temporary và nguồn; UI ghi rõ số FP32 cũ, chưa phải kết quả FP16. Suy luận demo vẫn FP16. Snapshot FP16 thật trong demo_data/latency_benchmark.json và log/report được giữ nguyên. Chạy lại rebenchmark_all.sh sẽ thay dữ liệu tạm bằng số đo mới.

## Cổng demo tránh dịch vụ đang chạy

Đổi cổng mặc định 8000 → 8765, ngoài danh sách cổng người dùng yêu cầu tránh. URL khởi động, README và lệnh mở demo/dashboard đã cập nhật. API và asset dùng URL cùng origin. Đã chạy 7 kiểm thử server và kiểm tra HTTP 200 thực cho demo/dashboard/api/status trên 8765; không chạy lại suy luận hoặc benchmark.
