"""Runtime checks and startup for the existing HTTP server."""
import os
import sys
import importlib
import socketserver


def check_runtime():
    if sys.version_info < (3, 12):
        raise SystemExit(
            f'Cần Python >= 3.12 để cài requirements.txt; đang chạy {sys.version.split()[0]} '
            f'({sys.executable}). Hãy tạo và kích hoạt venv bằng Python 3.12.'
        )
    for module in ('numpy', 'PIL.Image', 'cv2', 'torch', 'ultralytics', 'openvino'):
        try:
            importlib.import_module(module)
        except (ImportError, OSError) as error:
            if 'libGL.so.1' in str(error):
                raise SystemExit(
                    f'Không tải được {module}: {error}\n'
                    'Thiếu thư viện hệ thống OpenGL. Trên Ubuntu/Debian, chạy:\n'
                    'apt-get update && apt-get install -y libgl1\n'
                    'Sau đó chạy lại python run_demo.py trong venv đang kích hoạt.'
                ) from error
            raise SystemExit(
                f'Không tải được {module}: {error}\nPython đang chạy: {sys.executable}\n'
                f'Cài thư viện trong môi trường này: {sys.executable} -m pip install -r requirements.txt'
            ) from error


def run():
    from . import api
    os.chdir(api.DIRECTORY)
    check_runtime()
    for key, cfg in api.MODEL_CONFIGS.items():
        if not os.path.exists(cfg['path']):
            print(f"[!] Thiếu model {key}: {cfg['path']}. Chạy git pull để cập nhật models/.",
                  file=sys.stderr)

    print("[*] Đang khởi động AI Model Compression Studio...")
    print(f"[*] Các mô hình sẽ warm-up khi được chọn trên CPU (yêu cầu tối đa {api.CPU_THREADS} luồng; báo số thực tế khi chạy).")

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((api.HOST, api.PORT), api.StudioRequestHandler) as httpd:
        url_demo = f"http://localhost:{api.PORT}/demo.html"
        url_dash = f"http://localhost:{api.PORT}/model-compression-yolo26-dashboard.html"
        print("=" * 72)
        print("  YOLO26 x BDD100K MODEL COMPRESSION STUDIO & INFERENCE SERVER")
        print("=" * 72)
        print(f"  🚀 MVP Demo Studio (Trực quan & Upload):  {url_demo}")
        print(f"  📊 Bảng Quản Trị & Nghiệm Thu 14 Ngày:    {url_dash}")
        print(f"  ⚡ Live Prediction API Endpoint:           http://localhost:{api.PORT}/api/predict")
        print(f"  🌐 Lắng nghe: {api.HOST}:{api.PORT} · Máy khác dùng http://<IP-server>:{api.PORT}/demo.html")
        print("=" * 72)
        print("  Nhấn Ctrl+C để dừng server.")
        print()
        httpd.serve_forever()
