ĐÁNH GIÁ YOLO26 TRÊN BDD100K — CPU
=================================

1. Mục đích
So sánh checkpoint gốc với các model nén KHÔNG huấn luyện lại.
Đầu vào: ảnh validation, nhãn detection BDD100K JSON, baseline và các artifact.
Không tự tải dataset, tải weights, huấn luyện hoặc nén model.
Chạy chỉ baseline vẫn cho accuracy/latency, nhưng không có mAP drop cho model nén.

2. Cài đặt trên máy khác (Python 3.12)
Các bản NumPy đang ghim yêu cầu Python >=3.12; bộ này được kiểm tra với Python 3.12 trên Linux x86_64.
git clone/pull lấy source (bao gồm bdd100k_eval/), models/ và toàn bộ demo_data/; sao chép thêm runtime_exports/ nếu dùng Hardware.
Đánh giá accuracy còn cần data/bdd100k/; không sao chép thư mục .venv từ máy cũ.

Máy CPU (demo, PyTorch/OpenVINO/ONNX Runtime, export ONNX và calibration ONNX):
python3.12 -m venv .venv
source .venv/bin/activate
# Windows: py -3.12 -m venv .venv rồi .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python run_demo.py
# Máy khác truy cập http://<IP-server>:8765/demo.html

requirements.txt ưu tiên wheel PyTorch CPU từ https://download.pytorch.org/whl/cpu.
Máy Ubuntu/Debian thiếu libGL.so.1 hoặc libgthread: cài libgl1 và libglib2.0-0 bằng apt.
Riêng lỗi "libGL.so.1: cannot open shared object file", chạy bằng root:
apt-get update && apt-get install -y libgl1
Sau đó chạy lại python run_demo.py trong venv; pip không cài được thư viện hệ thống này.
Nếu log báo thiếu numpy/PIL hoặc Python 3.10, dừng server và tạo môi trường mới:
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python run_demo.py
Server kiểm tra Python và thư viện runtime trước khi mở cổng; chưa tải model ở bước này.

Nếu API báo model_artifact_missing hoặc "Model không sẵn sàng", thiếu artifact trên máy đích.
models/ được lưu trên Git: chạy git pull để tải các checkpoint và artifact OpenVINO.
Nếu sao chép thủ công, giữ nguyên tên và cấu trúc trong thư mục dự án trên VPS:
- models/yolo26m-bdd100k.pt
- models/int8_yolo26m_640_openvino_model/ (đầy đủ model.xml, model.bin, metadata.yaml)
- Các model khác muốn sử dụng: sao chép checkpoint hoặc toàn bộ thư mục OpenVINO tương ứng.
Không thay yolo26m-bdd100k.pt bằng yolo26m.pt (model COCO khác bộ lớp).
GET /api/status trả models_available và models_missing để kiểm tra file đã có.

Máy NVIDIA/CUDA 13 (profile riêng, tạo môi trường mới, thay cho profile CPU):
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-hardware.txt
python -m pip check
python run_demo.py
Không cài đồng thời onnxruntime và onnxruntime-gpu trong cùng môi trường.
GPU cần driver/CUDA/cuDNN phù hợp; TensorRT engine cần build lại trên máy đích.
Kiểm kê môi trường GPU dùng: python verify_project.py --requirements requirements-hardware.txt

Hai file requirements ghim cùng phiên bản thư viện chung, chỉ khác các engine CPU/NVIDIA.
report.json vẫn ghi phiên bản thực tế và số luồng runtime của mỗi lần chạy.
Một số phiên bản YOLO26 dùng end2end=True/False; phiên bản mới dùng nms=False/True.
Script phát hiện key cấu hình để chọn tương ứng và kiểm tra head runtime.

3. Dữ liệu và checkpoint
Ví dụ cấu trúc BDD100K bản JSON cũ:
/data/bdd100k/images/100k/val/*.jpg
/data/bdd100k/labels/bdd100k_labels_images_val.json
Các bản mới có thể đặt nhãn ở labels/det_20/det_val.json: truyền đúng đường dẫn.
Hỗ trợ list frame JSON kiểu BDD100K/Scalabel hoặc {"frames": [...]}.
Mỗi frame cần name và labels, mỗi detection có category và box2d {x1,y1,x2,y2}.
Không hỗ trợ COCO JSON, JSONL hoặc một JSON riêng cho mỗi ảnh trong phiên bản này.
Không cần tập train để chạy đánh giá. YAML sinh ra có train trỏ val chỉ để đáp
ứng schema validator; TUYỆT ĐỐI không dùng YAML này để training/calibration.

Model cần là Ultralytics YOLO26 detection đã có 10 lớp BDD100K đúng thứ tự:
pedestrian,rider,car,truck,bus,train,motorcycle,bicycle,traffic light,traffic sign
Chấp nhận các tên tương đương: person/pedestrian, bike/bicycle, motor/motorcycle.
Nếu checkpoint dùng thứ tự khác, truyền --names "class0,class1,...,class9" theo
chính thứ tự model.names. Chương trình kiểm tra và dừng khi mapping không khớp.
Checkpoint YOLO26 COCO 80 lớp KHÔNG đáp ứng điều kiện này. Chỉ có checkpoint COCO
thì không thể đánh giá đủ 10 lớp theo quy trình này mà không đổi phạm vi bài toán.
Artifact .pt phải load được bằng YOLO(); không nhận state_dict độc lập hay model
quantized PyTorch tùy biến không tương thích Ultralytics.
ONNX/OpenVINO cần giữ metadata Ultralytics về names, head và input shape.

4. Chạy nhanh để kiểm tra luồng (không dùng kết quả này để nghiệm thu)
python evaluate.py --images /data/bdd100k/images/100k/val --labels /data/bdd100k/labels/bdd100k_labels_images_val.json --baseline /models/best.pt --candidate INT8=/models/best_int8_openvino_model --limit 100 --latency-images 30 --warmup 10 --repeats 1 --output smoke_run

5. Chạy đánh giá accuracy toàn bộ validation và benchmark CPU
python evaluate.py --images /data/bdd100k/images/100k/val --labels /data/bdd100k/labels/bdd100k_labels_images_val.json --baseline /models/best.pt --candidate ONNX_FP32=/models/best_fp32.onnx --candidate INT8=/models/best_int8_openvino_model --imgsz 640 --threads 4 --head nms-free --latency-images 1000 --warmup 30 --repeats 3 --plots --output evaluation_full

--limit 0 (mặc định): dùng toàn bộ ảnh ở --images. Hãy trỏ đúng validation;
script không thể tự chứng minh đây là split val nếu người dùng đưa tập khác.
--limit >0: subset ảnh ngẫu nhiên có seed, dùng giống nhau cho mọi model.
--latency-images 1000: chọn tối đa 1000 ảnh từ tập đang đánh giá, seed cố định.
--latency-images 0: đo latency trên toàn bộ ảnh đã chọn.
--repeats 3: lặp 3 lần bộ ảnh latency. Benchmark dùng batch=1 và chạy tuần tự.
--output phải là thư mục mới để tránh trộn nhãn/cache/kết quả cũ.
Ảnh được symlink, nếu OS không hỗ trợ thì sao chép (tốn dung lượng tương ứng).

6. Chọn head và export
--head nms-free: one-to-one, không IoU NMS, lựa chọn mặc định của script.
--head nms: one-to-many có NMS.
Baseline và model nén phải dùng cùng head. Graph đã export không đổi head chỉ
bằng tùy chọn predict. Script kiểm tra metadata và báo lỗi nếu không xác định
được hoặc không khớp; không tự đoán head từ shape (1,300,6).
Với tài liệu Ultralytics hiện hành, export nms=False cho one-to-one:
from ultralytics import YOLO
YOLO('/models/best.pt').export(format='onnx', imgsz=640, batch=1, dynamic=False, nms=False)
Nếu dùng release cũ, theo tài liệu của đúng release để export end2end tương ứng.
INT8 cần artifact đã quantize và calibration thích hợp; script không thay đổi
precision graph. PyTorch .pt trên CPU mặc định được đánh giá FP16 (trọng số và input float16); --precision FP32 chỉ dùng khi cần đối chứng riêng. FP16/INT8 export phụ
thuộc graph/runtime, fp16_flag không chứng minh toàn bộ graph dùng FP16/INT8.
Tên candidate là nhãn do bạn đặt, không phải kết luận về precision thực tế.
Export phải batch=1, static input khớp --imgsz. max_det/conf/NMS của graph có thể
đã cố định khi export: dùng cùng cấu hình khi export và khi đánh giá. Export nhúng
NMS ở conf=0.25 sẽ loại sớm detection: không thích hợp cho mAP conf=0.001.
Không bật DNN/OpenCV thay cho ONNX Runtime. Không chạy TensorRT GPU trong script CPU này.

7. Các chỉ số được xuất
- mAP50-95, mAP50, mAP75; Precision, Recall, F1 macro; AP/P/R/F1 theo lớp có GT.
- P/R/F1: operating point tối ưu F1 của Ultralytics ở IoU 0.5; không phải P/R
  cố định ở conf=0.001. f1_macro là trung bình F1 từng lớp, không tính từ macro P/R.
- Latency predict_wall_ms: preprocessing + inference + postprocessing + API
  Python, batch=1, ảnh numpy, không gồm đọc/giải mã ảnh hoặc ghi file.
- decode_plus_predict_ms: thêm đọc/giải mã ảnh; OS cache có thể làm đọc nhanh hơn.
- inference_ms riêng và preprocess/postprocess do Ultralytics trả về.
- Mỗi loại latency có mean, median, p90, p95, p99, std, min, max; đơn vị ms/ảnh.
- serial_FPS = 1000 / latency predict mean. Đây là tốc độ suy luận tuần tự;
  không phải throughput batch lớn hoặc tốc độ toàn pipeline video/camera.
- mAP drop (pp) = 100 * (mAP50-95_baseline - mAP50-95_candidate), mAP ở [0,1].
- mAP drop tương đối (%) = 100 * (baseline - candidate) / baseline.
  Baseline bằng 0: tương đối = null. Drop âm nghĩa là candidate tăng accuracy.
- mAP50 drop cũng được lưu riêng. Ví dụ 0.420 -> 0.407: drop = 1.3 điểm phần
  trăm, giảm tương đối ≈ 3.095%; hai đại lượng này không được gọi thay nhau.
- evaluate.py: speedup = latency baseline của lần chạy / latency candidate.
- Bảng tổng hợp: mốc yolo26m-bdd100k.pt 640×640; chỉ tính khi latency cùng protocol đã xác minh.
  Bộ lọc YOLO26s trên demo dùng baseline YOLO26s của cùng phép đo.
- Kích thước MiB, tỉ lệ nén và giảm dung lượng. Directory export tính mọi file
  trong thư mục; dùng thư mục chỉ chứa artifact, tránh lẫn log/cache.
- Peak RSS lấy mẫu mỗi 10ms của worker riêng: gồm runtime, model, validation,
  bộ nhớ allocator giữ lại; có thể bỏ lỡ peak ngắn. Không phải RAM thuần weights.

8. File kết quả
summary.csv: bảng so sánh, mở bằng Excel.
report.json: tất cả chỉ số, protocol, cấu hình, phiên bản, hash artifact/nhãn.
dataset_manifest.json: tên ảnh/kích thước/hash nhãn và SHA-256 nội dung ảnh (các lượt chạy mới).
dataset_stats.json: thống kê nhãn, số objects từng lớp và box bị clip/ignore.
<model>/metrics.json: accuracy, per_class, latency, thông tin runtime.
<model>/latency_samples.csv: latency từng lần chạy, tên ảnh, repeat.
<model>/validation/: biểu đồ/confusion matrix/PR curve nếu bật --plots.
dataset/: ảnh symlink và nhãn YOLO sinh từ JSON.
report được checkpoint sau mỗi model; nếu worker lỗi, chương trình dừng, không
giả vờ hoàn tất bảng so sánh. Khắc phục rồi chạy với thư mục output mới.

9. Giới hạn và cách dùng trong báo cáo
Đây là mAP detector của Ultralytics trên nhãn BDD100K đã chuyển YOLO, KHÔNG phải
điểm benchmark chính thức BDD100K với đầy đủ ignore/crowd matching rules.
--ignore-policy include (mặc định) coi ignore/crowd như box GT thông thường;
drop bỏ GT này nhưng không loại prediction nằm trong ignore region, nên vẫn
KHÔNG tương đương official evaluation; error dừng khi gặp các cờ này.
Occluded/truncated không bị loại. Polygon/lane labels không thuộc box detection.
Muốn công bố điểm official BDD100K cần chạy thêm evaluator chính thức của BDD.
Các lớp không có GT trong subset không được gán AP suy đoán; xem per_class và
dataset_stats trước khi diễn giải macro mAP. Subset nhỏ có thể thiếu train/rider.

So sánh cùng CPU, imgsz, head, confidence, max_det, số luồng, tập ảnh và seed.
Mặc định conf=0.001 để không cắt sớm PR curve. iou=0.7 là ngưỡng NMS, độc lập
các IoU 0.50:0.05:0.95 dùng tính mAP. rect=False cho input vuông cố định ở cả
val và predict, tránh đo kích thước input khác giữa PyTorch và export.
--threads điều khiển PyTorch và OpenVINO CPU trong phép đo latency.
inference_runtime.py ép OpenVINO chạy CPU, NUM_STREAMS=1 và xác nhận số luồng thực.
ONNX vẫn ghi thread options ORT khi truy cập được; 0 là auto.
Nếu nghiên cứu tác động riêng của quantization, nên so FP32 và INT8 cùng backend
(ví dụ OpenVINO FP32 với OpenVINO INT8), cùng export settings. So PyTorch FP32
với OpenVINO INT8 đo cả thay đổi runtime và precision. Có thể đặt OpenVINO FP32
làm --baseline để tính drop và speedup trong evaluate.py theo backend phù hợp hơn.
Không suy luận model nén tốt chỉ từ giảm dung lượng: xem đồng thời mAP drop,
latency p95/mean, AP từng lớp, đặc biệt đối tượng nhỏ và nhóm ít mẫu.
Muốn AP small/medium/large hoặc average recall chuẩn COCO cần evaluator bổ sung;
script không gán nhầm các số này từ metrics.box.map.

10. Trạng thái kiểm thử của gói bàn giao
Đã kiểm tra fixture, API, hash artifact, runtime thực PyTorch/OpenVINO và demo.
Chạy các kiểm tra hồi quy:
.venv/bin/python -m unittest discover -v
node test_demo_overlap.js
node test_demo_selection.js
node test_demo_exports.js
node test_dashboard_state.js
node test_hardware_browser.js
node test_mvp_overlay.js

11. Tái tạo demo và benchmark latency
Bảng so sánh 6 cấu hình trong demo ghi CPU đo: 11th Gen Intel(R) Core(TM) i7-11800H.
.venv/bin/python refresh_demo_data.py
.venv/bin/python compare_benchmarks.py --csv benchmark_comparison.csv --markdown benchmark_comparison.md
Lệnh đầu đo 6 cấu hình trên cùng 50 ảnh × 3 lượt, warmup=5, CPU 4 luồng, baseline PyTorch FP32,
rect=False 640×640, conf=0.001; cập nhật dự đoán ảnh mẫu cho 10 model ở conf=0.10.
Lệnh refresh không chạy lại mAP. Accuracy baseline FP16 để trống và Δ pp chưa tính nếu chỉ có report FP32 cũ; các cột INT8 giữ kết quả accuracy đã xác minh.
Chạy khi máy rảnh; đây là snapshot latency trên máy dùng chung, không phải
cam kết hiệu năng trên thiết bị khác. demo_data/latency_benchmark.json lưu
hash ảnh/model, runtime thực, phân phối latency và từng mẫu đo.

12. Trạng thái nghiệm thu
Mức giảm artifact YOLO26m thực tế là 51,24% (42,022485 → 20,488908 MiB).
Checkpoint này lưu trọng số FP16; các baseline có khóa `*_fp32` trong demo và benchmark hiện suy luận CPU FP32. Nhãn Studio vẫn hiển thị FP16 theo yêu cầu giao diện; `runtime_settings` ghi FP32 thực tế. Chưa có gói lưu FP32
chuẩn hóa để kiểm tra ngưỡng giảm 60%; không dùng 42,02 MiB làm mốc lưu FP32. Không có bằng chứng
FP16 YOLO26m, calibration hoặc tái lập độc lập thì để chưa xác minh.
Dashboard giữ bản lưu cũ trước migration tại localStorage key hậu tố
:before-audit-v5 (bản v2 cũ vẫn giữ nếu có). Đặt lại chỉ xóa trạng thái dashboard, không xóa model.


Tham khảo API hiện hành:
https://docs.ultralytics.com/modes/val/
https://docs.ultralytics.com/guides/end2end-detection/
https://docs.ultralytics.com/reference/nn/autobackend/

13. Kiểm kê và xác minh lại bằng chứng
Kích hoạt môi trường đã cài ở mục 2, rồi chạy:
python verify_project.py
# Profile NVIDIA: python verify_project.py --requirements requirements-hardware.txt
Lệnh kiểm tra 10 model: tensor dtype gốc, class map, kích thước, SHA-256;
6 IR OpenVINO được nạp trong tiến trình riêng chỉ dùng OpenVINO/NumPy,
không import PyTorch, CPU 4 luồng và một stream. Probe zero-input kiểm tra
nạp/shape/giá trị hữu hạn, không thay cho đánh giá accuracy. Thời gian probe
không bao gồm import thư viện và không bảo đảm cache ổ đĩa lạnh.
Toàn bộ 10.000 ảnh được giải mã, hash và đối chiếu annotation; ảnh trùng
nội dung được liệt kê, giữ nguyên benchmark lịch sử. Không tự chia lại
validation đã sử dụng rồi gọi đó là final chưa từng thấy.
Mặc định output trong verification/: model_inventory.json, image_manifest.json,
data_inventory.json, environment.lock.json, protocol.yaml, reproduction.json.
Để giữ nguyên bằng chứng cũ, truyền thư mục mới, ví dụ:
python verify_project.py --output verification/check_20261018
environment.lock.json hash cả implementation trong bdd100k_eval/, các entry point
ở root, JS/CSS của demo/dashboard và manifest runtime; không hash lại vendor internals.
Protocol này ghi cấu hình quan sát được; không thay xác nhận ngưỡng của nhóm.

Chạy lại phép so sánh 50 ảnh (không dùng mAP subset làm kết quả 10.000 ảnh):
.venv/bin/python compare_ptq_vs_qat.py --smoke --output verification/smoke_ptq_qat

Chạy lại PTQ trên toàn bộ validation, giữ log lịch sử:
.venv/bin/python evaluate.py --images data/bdd100k/images/val --labels data/bdd100k/labels/bdd100k_labels_images_val.json --baseline models/int8_yolo26m_640_openvino_model --latency-images 50 --warmup 5 --repeats 3 --threads 4 --output verification/full_ptq_retry
Ở lượt kiểm tra một artifact này, nhãn baseline trong report chỉ là vai trò
CLI; artifact thực tế là PTQ INT8. Không diễn giải nó là baseline FP32.
Chạy verify_project.py sau khi hoàn tất để đối chiếu report/hash/manifest.
Output evaluator phải là thư mục mới; dùng --output khác khi đã có kết quả.

Ảnh demo giữ metadata BDD100K đúng nguồn; ảnh upload không được tự gán
ban ngày/trời quang. Preview, sao chép và tải JSON cùng dùng một schema,
chuẩn hóa person/bike/motor sang pedestrian/bicycle/motorcycle.

BENCHMARK HARDWARE / RUNTIME (18-10-2026)
========================================
Mở http://127.0.0.1:8765/demo.html#hardware sau khi chạy run_demo.py.
Server lắng nghe 0.0.0.0:8765; từ máy khác dùng http://<IP-server>:8765/demo.html.
Chọn model, runtime, thiết bị, precision, ảnh, warmup/số lượt rồi bấm Chạy benchmark.
Đổi lựa chọn không tự chạy. Có thể xuất toàn bộ số đo trong phiên thành JSON.

- Backend: PyTorch CPU FP32 / CUDA FP32, FP16; ONNX Runtime CPU FP32 /
  CUDA FP32, FP16; TensorRT CUDA FP32, FP16; OpenVINO CPU/GPU theo khả năng
  thiết bị, với graph FP32 và các artifact PTQ/QAT INT8 sẵn có.
- Trình duyệt: ONNX Runtime Web WASM CPU FP32/FP16/INT8 (chọn số luồng); WebGPU có các graph tương ứng
  khi có adapter phần cứng và shader-f16 cho FP16. Cần secure context
  (HTTPS hoặc localhost). Trình duyệt không có adapter thì không cho chạy GPU.
- CUDA/backend là máy chạy Python, không phải máy mở trình duyệt ở xa.
  Muốn chạy thư viện native trên máy người dùng, họ chạy backend localhost.
- Cấu hình thiếu runtime, artifact hoặc phần cứng hiển thị lý do và khóa nút chạy.
  Không thay GPU bằng CPU khi runtime lỗi. ORT CUDA/WebGPU vẫn có thể chạy một
  số node phụ trên CPU; JSON ghi providers, không tuyên bố mọi node chạy GPU.
- INT8 native hiện dùng OpenVINO; browser có ONNX INT8 PTQ QDQ mới, calibration
  128 ảnh train. Không tự cast FP32 thành INT8, không lấy validation làm calibration. FP16 TensorRT là policy cho
  phép FP16; một số lớp vẫn giữ FP32. OpenVINO FP16 storage không chứng minh
  FP16 compute; chỉ bật FP16 khi thiết bị công bố hỗ trợ và có export phù hợp.

Chọn profile NVIDIA/CUDA 13 và kích hoạt .venv-gpu theo mục 2, rồi chạy:
python prepare_runtime_models.py
# Hoặc chỉ export ONNX/OpenVINO:
python prepare_runtime_models.py --engines onnx openvino
# Giới hạn họ model:
python prepare_runtime_models.py --models yolo26s

Export ghi riêng runtime_exports/, không sửa models/ hay số liệu accuracy cũ.
TensorRT build tốn vài phút/model/precision, engine phụ thuộc GPU và phiên bản;
chạy lại lệnh export trên máy đích. Manifest ghi SHA256/source/GPU/version/shape.
Khi thay checkpoint, phải export lại. Máy không có CUDA vẫn export FP32, bỏ qua
FP16 và TensorRT. Không cần internet khi dùng runtime web vendored và model đã
export. Assets ONNX Runtime Web 1.30.0, MIT: demo_data/vendor/onnxruntime-web/.
Chỉ ONNX có URL tải; checkpoint .pt, .engine và manifest không public qua HTTP.

Protocol mới là microbenchmark MỘT ảnh: RGB/NCHW, letterbox 640x640, batch1,
head one-to-one end-to-end, không NMS; confidence mặc định .25, warmup3, đo10.
Đồng bộ/đọc output về CPU trước dừng giờ. Đo tiền xử lý + truyền CPU/GPU +
inference + hậu xử lý; không gồm giải mã/tải ảnh, HTTP, download/load/compile,
warmup. p50/p95 nội suy tuyến tính; FPS = 1000/mean_ms. OpenCV và Canvas có
nội suy ảnh khác nhau. Đây không phải mAP, throughput batch hoặc phép đo kernel.
TF32 bị tắt cho PyTorch/ORT CUDA và TensorRT FP32. OpenVINO JSON có histogram
runtime_precision_nodes để kiểm tra graph INT8 dùng các kernel mixed precision.
Bảng Hardware riêng, không ghi đè benchmark CPU 50 ảnh x3 / accuracy 10.000 ảnh.
Chỉ so sánh khi cùng ảnh/hash, scope, precision/head, phần cứng và tải nền.

API:
GET /api/benchmark/capabilities
POST /api/benchmark/run
{ "model":"yolo26s", "engine":"pytorch", "device":"cuda", "precision":"FP16",
  "warmup":3, "repeats":10, "conf":0.25, "image":"data:image/jpeg;base64,..." }
Giới hạn ảnh 12MiB request /20 triệu pixel; warmup0-10, lượt đo1-50.
Backend serialize cùng inference A/B, trả503 khi bận. Không có engine build
hoặc tải model tự động qua API. WebGPU/WASM chạy ở JS, không gọi native run API.

Tài liệu runtime chính thức:
https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html

ƯU TIÊN ONNX RUNTIME WEB / WEBGPU
================================
Mục #hardware mặc định chạy ONNX Runtime Web · CPU (WASM), có lựa chọn
ONNX Runtime Web · GPU (WebGPU). Cả hai chạy suy luận trên máy mở trang,
không POST ảnh tới /api/benchmark/run và không cần API capabilities Python.
Runtime native được giữ trong nhóm Nâng cao.

Chọn Số luồng CPU (WASM) từ 1 đến số CPU logic trình duyệt báo (mặc định 1).
Đa luồng cần WebAssembly shared memory và cross-origin isolation. run_demo.py
đã gửi COOP: same-origin và COEP: require-corp; hosting khác cần cấu hình hai
header này, nếu thiếu UI chỉ cho 1 luồng. Tùy chọn bị khóa ở WebGPU/backend.
Mỗi lượt WASM dùng worker mới để đổi số luồng có hiệu lực mà không tải lại trang;
worker và model khởi tạo ngoài thời gian đo. Tiền xử lý dùng OffscreenCanvas.
Bảng kết quả và JSON ghi protocol.cpu_threads (runtime đã khởi tạo), cùng
cpu_threads_requested. Đổi số luồng không chạy inference và ẩn preview cũ.

Danh sách browser: demo_data/browser_models.json, tự cập nhật khi export ONNX.
CPU/WASM: FP32, FP16 và INT8. WebGPU: các graph FP32/FP16/INT8 khi có adapter;
FP16 trên GPU cần shader-f16. FP16 là precision của graph, không đảm bảo mọi
kernel WASM tính FP16. ONNX INT8 mới là PTQ QDQ (Conv U8/S8, lớp còn lại float),
calibration 128 ảnh train; graph OpenVINO INT8 không dùng trực tiếp trong browser.
WebGPU yêu cầu secure context và adapter GPU hợp lệ. Giao diện phân biệt thiếu
API WebGPU, thiếu adapter, software fallback và thiếu shader-f16; không tự đổi
WebGPU sang WASM rồi ghi thành kết quả GPU.

Để chạy trên hosting tĩnh, phục vụ demo.html, thư mục demo_data/ (bao gồm runtime
web vendored, browser_models.json và ảnh), cùng sáu file .onnx trong
runtime_exports/. Giữ nguyên cấu trúc thư mục và MIME .wasm=application/wasm,
.mjs=text/javascript. Dùng HTTPS hoặc localhost. Không cần triển khai .pt,
.engine, model OpenVINO hoặc backend AI để dùng hai chế độ trình duyệt.
Phần A/B native và nhóm Nâng cao vẫn cần backend khi chọn suy luận native.


Tạo ONNX INT8 trình duyệt từ các ONNX FP32 cùng head end-to-end:
.venv/bin/python prepare_runtime_models.py --engines onnx --int8-calibration-list /path/train_images.txt --calibration-count 128
Mỗi dòng danh sách là một đường dẫn ảnh train; đường dẫn phải thuộc thư mục
train. CLI đối chiếu tên và SHA256 với verification/image_manifest.json, từ
chối ảnh trùng validation và nội dung trùng trong calibration. Seed42, chọn
128 ảnh; manifest riêng: verification/browser_int8_calibration.json. Không
sửa checkpoint gốc, graph OpenVINO cũ hay số liệu accuracy lịch sử.

Các file mới: runtime_exports/yolo26s_int8.onnx, yolo26m_int8.onnx.
Input/output INT8 ONNX vẫn FP32 theo QDQ; không cast pixel trực tiếp sang INT8.
JSON ghi graph_precision, input_dtype, output_dtype, precision_policy,
calibration hash/count và số Conv lượng tử hóa. INT8 graph có thể chạy kernel
mixed precision theo backend, không bảo đảm mọi node dùng INT8 hoặc tăng tốc.
Chưa có đánh giá mAP riêng cho hai ONNX INT8 mới; không gán mAP OpenVINO cũ
cho chúng. Chế độ FP16/WASM đã được kiểm thử thực tế, bỏ khóa CPU chỉ FP32 cũ.


Preview benchmark hiển thị ảnh lớn với box, tên lớp và confidence của kết quả
vừa chạy, dùng chung renderer SVG với phần A/B. Tọa độ là box_px theo ảnh gốc;
viewBox tự đổi theo kích thước thật, chữ giữ kích thước dễ đọc khi co ảnh.
Đổi ảnh/model/runtime/device/precision/conf sẽ ẩn box không còn phù hợp. Chạy
lại xóa box cũ trước khi đo; nếu lỗi không để lại annotation của lượt trước.
Kết quả 0 detection hiển thị 0 hộp. Vẽ preview sau phép đo, không cộng vào latency.

Chuyển sang FP16 (2026-10-18)
Demo cho chọn riêng từ 1 đến toàn bộ CPU khả dụng cho mỗi model, mặc định 3; evaluator và benchmark giữ số luồng theo protocol.
AutoBackend bỏ FP16 CPU mặc định: helper cấu hình dtype sau khởi tạo, trước warmup/đo;
validator cũng dùng input float16. Kiểm tra dtype thực trong runtime report.
FP16 CPU có thể chậm hơn FP32 trên máy không có tăng tốc FP16 native.
Các chế độ FP32 thật trong Hardware và file models/fp32.pt vẫn giữ đúng dtype.

Chạy lại đầy đủ sáu model THEO THỨ TỰ, 10.000 ảnh/model, sau đó cập nhật bảng/demo:
bash rebenchmark_all.sh
Script dừng khi có lỗi; kết quả ở verification/rebenchmark_fp16_<timestamp>/.
Sáu model: m FP16 → m PTQ → m QAT → s FP16 → s PTQ → s QAT.
Latency cùng 50 ảnh ×3, warmup5, 640×640, conf0.001, IoU0.7, max_det300, seed42.
Giữ toàn bộ các cột benchmark; Δ pp theo baseline FP16 cùng họ; speedup CSV theo
YOLO26m FP16 640, bộ lọc YOLO26s theo baseline YOLO26s FP16.
Nhãn Studio hiển thị FP16; precision thực của snapshot vẫn được giữ trong metadata kỹ thuật.

Khôi phục suy luận FP32 cho nhãn Studio FP16 (2026-10-18)
Các khóa yolo26m_fp32, yolo26s_fp32 và yolo26s_fp32_pack chạy tensor FP32 trên CPU.
Tên model trên Studio vẫn là FP16; đây chỉ là nhãn hiển thị. API/report giữ precision và
parameter_dtype FP32 để không làm mất dấu cấu hình thực tế. Các model FP16 thật, INT8,
Hardware và tùy chọn --precision của evaluator không thay đổi.

MVP – Demo & Đánh giá (2026-10-18)
Trang mặc định: http://<IP-server>:8765/demo.html#mvp
Khởi chạy: kích hoạt venv rồi python run_demo.py (0.0.0.0:8765).
Không cần cài thêm dependency ngoài requirements.txt hiện có.
Model chính thức: models/int8_yolo26s_640_openvino_model/ (tên thật là _model, không phải _models).
MVP dùng get_model() cache chung, mỗi bên chọn từ CPU ×1 đến mức affinity/cgroup cho phép, mặc định ×3, 1 stream, 640×640; A chạy xong mới đến B và không dùng kết quả mẫu lưu sẵn.
Nút Nhận diện xóa kết quả live của cấu hình A/B hiện tại rồi chạy lại tuần tự với số CPU đang chọn.
Trên VPS/container, OpenVINO có thể giới hạn số luồng thấp hơn theo CPU khả dụng/chính sách lập lịch.
Demo dùng số luồng thực và hiển thị cảnh báo; báo cáo ghi requested_threads, ov_threads, thread_protocol_matches.
Benchmark/evaluate/refresh_demo_data vẫn yêu cầu đúng số luồng protocol; sai cấu hình sẽ dừng kèm số thực tế.
Tên 10 lớp lấy từ metadata.yaml/model.names; cấu hình end-to-end không dùng IoU/NMS.

Luồng: chọn/kéo thả nhiều ảnh hoặc video → Chạy nhận diện → chọn từng kết quả → tải.
Có ảnh mẫu có sẵn; mỗi lần chạy ảnh mẫu vẫn suy luận thật. Tab khác tiếp tục hoạt động.
Ảnh tĩnh JPG/JPEG/PNG/WEBP: tối đa 12 MiB, 20 triệu pixel. Không hỗ trợ ảnh động WebP/PNG.
Video MP4/WebM: tối đa 64 MiB, 60 giây, 600 frame, 1920×1080; tối đa 180 giây xử lý.
Codec cần OpenCV giải mã được; phát gốc trên trình duyệt nên ưu tiên MP4 H.264/WebM VP8.
Video xuất WebM VP8 ở FPS metadata đầu vào, không âm thanh; video FPS biến thiên có thể khác nhịp gốc.
Webcam ghi tối đa 10 giây rồi gửi server để nhận diện; cần HTTPS hoặc localhost, MediaRecorder và quyền camera.
Webcam qua http://<IP-VPS> thường không khả dụng: tải video hoặc đặt ứng dụng sau HTTPS.
Chế độ Trình chiếu phóng lớn hình/chỉ số, ẩn điều khiển phụ; Esc để thoát.
MVP chỉ có một màn hình ảnh/video với lớp box SVG tương tác, dùng chung renderer với Studio.
Rê chuột xem tooltip; bấm box/dòng bảng hoặc Enter/Space để chọn, video tạm dừng khi chọn.
Bật/tắt hộp, tên lớp, confidence đổi hiển thị ngay; dữ liệu phát hiện và chỉ số không thay đổi.
Đổi confidence/lớp cần Chạy lại. Tệp xuất giữ cấu hình lúc chạy; Chạy lại để cập nhật tệp xuất theo tùy chọn hiển thị.
Video có nút phát/tạm dừng và thanh tua riêng; box theo timestamp khung nguồn, không giữ box trên phần chưa xử lý.
Kiểm tra frontend: node test_mvp_overlay.js.

Suy luận: timer engine từ Ultralytics/OpenVINO. Pipeline: tiền xử lý + inference + hậu xử lý/API.
FPS xử lý: số frame / thời gian xử lý tuần tự, gồm vẽ/mã hóa/đọc frame tiếp; không phải FPS phát.
Toàn bộ trên server: từ sau upload đến kết thúc, gồm giải mã và khởi tạo/warm-up.
Khởi tạo/warm-up đo riêng, bằng 0 khi tái dùng runtime đã chuẩn bị; không cộng vào suy luận thường.
Báo cáo chứa hash đầu vào/XML/BIN, cấu hình, thiết bị CPU thực, phiên bản OpenVINO và phạm vi timer.
Số đếm video theo khung hình; counts trong báo cáo là tổng lượt phát hiện, không phải đối tượng duy nhất.
Chưa triển khai tracking. Không có ground truth thì không công bố precision, recall hoặc mAP.

Đánh giá có nhãn: chỉ ảnh, BDD100K JSON list frame hoặc {frames: [...]} ≤12 MiB; cung cấp subset theo ảnh demo.
Mỗi frame: name khớp chính xác tên tệp, labels chứa category và box2d x1,y1,x2,y2 theo pixel ảnh gốc.
Tối đa 1.000 GT/ảnh; phải có entry labels=[] cho ảnh không có đối tượng.
Không nhận COCO JSON, nhãn video, ignore/crowd hay tên lớp ngoài mapping BDD100K.
Chế độ này chạy conf=0.001, max_det=300; mAP theo Ultralytics IoU 0.50:0.95, P/R macro tại F1 tối ưu.
Chỉ lấy trung bình trên lớp có GT trong subset/bộ lọc lớp; đây không phải đánh giá BDD100K chính thức hoặc toàn validation.
Nhãn BDD100K gốc toàn bộ lớn hơn giới hạn upload: trích frame theo tên ảnh hoặc dùng evaluate.py cho tập đầy đủ.

Tài nguyên: một tác vụ inference/benchmark server tại một thời điểm; tối đa 30 tệp, 512 MiB dữ liệu hiện có,
video đầu ra tối đa 128 MiB. Giới hạn tính theo từng tiến trình server, chưa phải hệ thống nhiều tenant.
Dừng yêu cầu ngừng sau bước/khung đang chạy; phần đã xử lý được đánh dấu cancelled và có thể tải.
Xóa dữ liệu hủy việc đang chạy rồi dọn file; TTL 30 phút không truy cập, kiểm tra định kỳ mỗi phút.
Khi đóng trang, trình duyệt yêu cầu xóa phiên; TTL xử lý trường hợp mất kết nối/đóng tab bất thường.
API /api/mvp/config, POST /api/mvp/jobs (raw file + query cấu hình), GET/DELETE /api/mvp/jobs/<id>,
POST /api/mvp/jobs/<id>/cancel, GET /api/mvp/jobs/<id>/files/<file>, POST /api/mvp/evaluate.
File media hỗ trợ HTTP Range để phát/tua. Không lưu upload vào Git, không tự đổi sang model/runtime khác.
Nhãn lane/drivable area không có box2d được bỏ qua như pipeline hiện có; báo cáo ghi số nhãn bỏ qua.
Để hội đồng thử đánh giá nhanh: chọn ảnh mẫu, chọn Đánh giá có nhãn, tải liên kết nhãn mẫu rồi chọn file JSON đó.
demo_data/mvp_sample_labels.json trích box2d từ nhãn BDD100K sẵn có cho 5 ảnh demo; không phải prediction.

CẤU TRÚC SOURCE SAU REFACTOR (2026-10-18)
=======================================
Các lệnh Python ở root giữ nguyên tên, tham số và import cũ. Implementation nằm
trong package phẳng bdd100k_eval/, không cần cài package hay đổi working directory:

bdd100k_eval/
  config.py              đường dẫn root, model registry và giới hạn hiện có
  artifacts.py           SHA-256, kích thước và metadata artifact
  reporting.py           JSON, CSV và bảng báo cáo dùng chung
  dataset.py             mapping BDD100K và chuẩn bị dataset evaluation
  image_ops.py           giải mã, letterbox và hậu xử lý benchmark Hardware
  inference_runtime.py   cấu hình CPU/precision/head hiện có
  models.py              lazy loading và cache model dùng chung
  evaluation.py          evaluator, worker CLI và đánh giá phiên MVP
  hardware_benchmark.py  backend benchmark native hiện có
  mvp_demo.py            quản lý phiên và xử lý ảnh/video
  api.py                 HTTP handlers, route và phản hồi API
  server.py              kiểm tra runtime và khởi động HTTP server
  tools/                 implementation của 7 CLI phụ trợ

demo.html và model-compression-yolo26-dashboard.html vẫn là URL cũ.
demo_data/studio.js, studio.css, dashboard.js và dashboard.css chứa source
đã tách từ HTML. overlays.js là renderer SVG chung; tải trước studio.js,
hardware_benchmark.js và mvp.js. Hardware worker vẫn dùng URL cũ.
JSON benchmark nhúng trong dashboard giữ nguyên để refresh_demo_data cập nhật.

Không đổi đường dẫn models/, runtime_exports/, data/, evaluation_*/, verification/
và các symlink. Giữ các entry point root khi triển khai hoặc viết script cũ;
code mới có thể import trực tiếp từ bdd100k_eval. Không chỉ sao chép wrapper root
mà bỏ package hoặc JS/CSS mới. Tests hồi quy vẫn ở root.

Báo cáo thay đổi, bằng chứng baseline, kiểm tra tương đương và giới hạn còn lại:
REFACTOR_REPORT.md. Chưa tái chạy full validation 10.000 ảnh cho mỗi model trong
đợt refactor; kết quả lịch sử không bị ghi đè hoặc đổi nhãn.
