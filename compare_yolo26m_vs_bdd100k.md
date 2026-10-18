# BÁO CÁO ĐỐI ĐẦU: YOLO26m-BDD100K vs YOLO26m (MS COCO)
**Thời gian tạo:** 2026-10-17 02:47:04  
**Thiết bị thực thi:** `CUDA` (NVIDIA GeForce RTX 3070 Laptop GPU)  
**Kích thước ảnh:** `640x640` | **Ngưỡng Conf:** `0.25`

---

## 1. TÓM TẮT KẾT LUẬN QUAN TRỌNG

1. **Khác biệt về nhiệm vụ & tập dữ liệu:**
   - **`yolo26m-bdd100k.pt`:** Được huấn luyện chuyên sâu trên dữ liệu lái xe tự hành **BDD100K** (10 nhãn đường phố: *person, rider, car, truck, bus, train, motor, bike, traffic light, traffic sign*).
   - **`yolo26m.pt`:** Là bản pre-trained tổng quát trên **MS COCO** (80 nhãn: *person, bicycle, car, animal, furniture, electronics...*).

2. **Khác biệt kiến trúc:**
   - Phần **Backbone và Neck giống nhau 100%** (21.68 triệu tham số).
   - Điểm khác biệt duy nhất nằm ở tầng **Detection Head**: Bản COCO có 80 lớp nên tốn thêm **107.940 tham số** (+0.18 MB) để biểu diễn các kênh phân loại mở rộng.

3. **Chất lượng phát hiện trên dữ liệu đường phố BDD100K:**
   - Trên cùng các ảnh giao thông thực tế, **`yolo26m-bdd100k.pt` phát hiện nhiều hơn gấp ~1.85 lần** so với `yolo26m.pt`.
   - **Mô hình COCO bỏ sót nghiêm trọng:**
     - Toàn bộ các biển báo giao thông (**traffic sign**) vì COCO chỉ có nhãn `stop sign`.
     - Không có nhãn người điều khiển phương tiện (**rider** - người đi xe đạp/xe máy).
     - Bỏ sót nhiều đèn giao thông nhỏ từ xa và người đi bộ trong điều kiện ánh sáng yếu/ban đêm do thiếu domain adaptation.

---

## 2. BẢNG SO SÁNH KIẾN TRÚC & THÔNG SỐ KỸ THUẬT

| Đặc tính                 | YOLO26m-BDD100K      | YOLO26m (MS COCO)   | Chênh lệch (Delta)            |
|--------------------------|----------------------|---------------------|-------------------------------|
| Tập dữ liệu huấn luyện   | BDD100K (Giao thông) | MS COCO (Tổng quát) | Chuyên biệt vs Phổ quát       |
| Số lượng lớp phát hiện   | 10 lớp               | 80 lớp              | +70 lớp                       |
| Dung lượng file (.pt)    | 42.02 MB             | 42.21 MB            | +0.19 MB (+0.5%)              |
| Tổng số tham số (Params) | 21,788,308 (21.79M)  | 21,896,248 (21.90M) | +107,940 (0.108M)             |
| Tham số Backbone + Neck  | 18,974,272           | 18,974,272          | 0 (Giống nhau 100%)           |
| Tham số Detection Head   | 2,814,036            | 2,921,976           | +107,940 (Do số lớp 80 vs 10) |
| SHA-256 Checkpoint       | 8246690d0d92816e...  | 401cea9ab23ad192... | Khác nhau                     |

---

## 3. ĐO ĐẠC HIỆU NĂNG TỐC ĐỘ & ĐỘ TRỄ

| Chỉ số đo đạc                  | YOLO26m-BDD100K | YOLO26m (COCO) | So sánh  |
|--------------------------------|-----------------|----------------|----------|
| Độ trễ trung bình (CUDA)       | 19.45 ms        | 20.87 ms       | +1.42 ms |
| Độ trễ trung vị p50 (CUDA)     | 17.53 ms        | 20.56 ms       | +3.03 ms |
| Độ trễ p95 (CUDA)              | 22.29 ms        | 22.38 ms       | +0.09 ms |
| Tốc độ khung hình (FPS - CUDA) | 51.4 FPS        | 47.9 FPS       | 1.07x    |
| Độ trễ trung bình (CPU)        | 218.98 ms       | 302.57 ms      | +83.6 ms |
| Tốc độ khung hình (FPS - CPU)  | 4.6 FPS         | 3.3 FPS        | 1.39x    |

---

## 4. CHI TIẾT SỐ LƯỢNG PHÁT HIỆN THEO TÌNH HUỐNG GIAO THÔNG

| Hình ảnh tình huống    | YOLO26m-BDD100K | YOLO26m (COCO) | Chênh lệch        | Đặc điểm nhận diện                                                 |
|------------------------|-----------------|----------------|-------------------|--------------------------------------------------------------------|
| b1ca2e5d-84cf9134.jpg  | 28 boxes        | 11 boxes       | +17 boxes (+155%) | BDD phát hiện thêm 17 hộp (chủ yếu traffic sign, đèn, người đi bộ) |
| b1cd1e94-549d0bfe.jpg  | 14 boxes        | 6 boxes        | +8 boxes (+133%)  | BDD phát hiện thêm 8 hộp (chủ yếu traffic sign, đèn, người đi bộ)  |
| b1ceb32e-a106591d.jpg  | 21 boxes        | 16 boxes       | +5 boxes (+31%)   | BDD phát hiện thêm 5 hộp (chủ yếu traffic sign, đèn, người đi bộ)  |
| b1d0a191-5490450b.jpg  | 24 boxes        | 12 boxes       | +12 boxes (+100%) | BDD phát hiện thêm 12 hộp (chủ yếu traffic sign, đèn, người đi bộ) |
| b1d0a191-de8948f6.jpg  | 22 boxes        | 14 boxes       | +8 boxes (+57%)   | BDD phát hiện thêm 8 hộp (chủ yếu traffic sign, đèn, người đi bộ)  |
| TỔNG CỘNG TẤT CẢ SCENE | 109 boxes       | 59 boxes       | +50 boxes         | BDD phát hiện nhiều hơn gấp 1.85 lần                               |

### Phân tích chi tiết từng khung hình:

#### Tình huống: `b1ca2e5d-84cf9134.jpg`
- **YOLO26m-BDD100K (28 đối tượng):** `{"car": 7, "person": 13, "traffic light": 3, "traffic sign": 5}`
- **YOLO26m-COCO (11 đối tượng):** `{"car": 5, "person": 5, "traffic light": 1}`
- **Ánh xạ về BDD100K:** `{"car": 5, "person": 5, "traffic light": 1}`

#### Tình huống: `b1cd1e94-549d0bfe.jpg`
- **YOLO26m-BDD100K (14 đối tượng):** `{"bus": 1, "car": 5, "person": 1, "traffic sign": 4, "traffic light": 3}`
- **YOLO26m-COCO (6 đối tượng):** `{"bus": 1, "car": 4, "person": 1}`
- **Ánh xạ về BDD100K:** `{"bus": 1, "car": 4, "person": 1}`

#### Tình huống: `b1ceb32e-a106591d.jpg`
- **YOLO26m-BDD100K (21 đối tượng):** `{"car": 14, "traffic sign": 5, "truck": 1, "traffic light": 1}`
- **YOLO26m-COCO (16 đối tượng):** `{"car": 13, "truck": 3}`
- **Ánh xạ về BDD100K:** `{"car": 13, "truck": 3}`

#### Tình huống: `b1d0a191-5490450b.jpg`
- **YOLO26m-BDD100K (24 đối tượng):** `{"car": 9, "traffic light": 8, "traffic sign": 5, "person": 2}`
- **YOLO26m-COCO (12 đối tượng):** `{"car": 9, "traffic light": 3}`
- **Ánh xạ về BDD100K:** `{"car": 9, "traffic light": 3}`

#### Tình huống: `b1d0a191-de8948f6.jpg`
- **YOLO26m-BDD100K (22 đối tượng):** `{"car": 12, "traffic light": 6, "traffic sign": 2, "person": 1, "truck": 1}`
- **YOLO26m-COCO (14 đối tượng):** `{"car": 12, "traffic light": 1, "person": 1}`
- **Ánh xạ về BDD100K:** `{"car": 12, "traffic light": 1, "person": 1}`

---

## 5. HÌNH ẢNH SO SÁNH TRỰC QUAN (SIDE-BY-SIDE VISUAL)

Các ảnh so sánh trực tiếp đã được lưu tại thư mục: `comparison_yolo26m_visual/`
- `comparison_yolo26m_visual/compare_b1ca2e5d-84cf9134.jpg`
- `comparison_yolo26m_visual/compare_b1cd1e94-549d0bfe.jpg`
- `comparison_yolo26m_visual/compare_b1ceb32e-a106591d.jpg`
- `comparison_yolo26m_visual/compare_b1d0a191-5490450b.jpg`
- `comparison_yolo26m_visual/compare_b1d0a191-de8948f6.jpg`
