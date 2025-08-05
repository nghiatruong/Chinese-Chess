import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import shutil

# Kiểm tra GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected, using CPU")

# Tạo cấu trúc dataset
dataset_dir = "xiangqi_dataset"
os.makedirs(f"{dataset_dir}/images/train", exist_ok=True)
os.makedirs(f"{dataset_dir}/images/val", exist_ok=True)
os.makedirs(f"{dataset_dir}/labels/train", exist_ok=True)
os.makedirs(f"{dataset_dir}/labels/val", exist_ok=True)


# Hàm phát hiện lưới bàn cờ 9x10
def detect_board_grid(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Tăng độ tương phản
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Phát hiện đường thẳng bằng Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=100, minLineLength=100, maxLineGap=10)

    # Phân loại đường ngang và dọc
    h_lines, v_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) > abs(y1 - y2):  # Đường ngang
            h_lines.append((y1 + y2) / 2)
        else:  # Đường dọc
            v_lines.append((x1 + x2) / 2)

    # Lấy 10 đường ngang và 9 đường dọc
    h_lines = sorted(h_lines)[:10]
    v_lines = sorted(v_lines)[:9]

    # Tạo lưới giao điểm 9x10
    grid = []
    board_map = {}
    for i, y in enumerate(h_lines):
        for j, x in enumerate(v_lines):
            grid.append((x, y))
            board_map[(x, y)] = f"{chr(97+j)}{9-i}"  # a0-i9
    return grid, board_map


# Hàm ánh xạ quân cờ vào lưới
def map_to_board(boxes, grid, board_map):
    board_state = {}
    for box in boxes:
        x, y, w, h = box.xywh[0]
        label = box.cls  # Nhãn quân cờ
        closest = min(grid, key=lambda p: ((p[0]-x)**2 + (p[1]-y)**2)**0.5)
        board_state[label] = board_map[closest]
    return board_state


# Huấn luyện YOLOv8
def train_model():
    model = YOLO("yolov8s.pt")  # Model base
    model.train(
        data="chinese-chess-detect-for-yolo-8/data.yaml",  # Dataset
        imgsz=640,               # Kích thước phù hợp với quân cờ
        epochs=150,              # Tăng số epoch để học tốt hơn
        batch=16,                # Tăng batch size cho GPU RTX 3060
        name="xiangqi_model",    # Tên thư mục output
        workers=4,               # Số lượng luồng load data
        patience=30,             # Tăng patience để tránh dừng sớm
        optimizer='AdamW',       # Dùng AdamW cho hiệu quả tốt hơn
        lr0=0.001,              # Learning rate khởi đầu ổn định
        lrf=0.0001,             # Learning rate cuối thấp hơn
        weight_decay=0.0005,     # Giữ regularization
        warmup_epochs=5,         # Tăng warmup để ổn định
        box=7.0,                 # Tăng box loss vì vị trí quan trọng
        cls=1.0,                 # Tăng class loss
        hsv_h=0.2,              # Giảm hue vì màu sắc quan trọng
        hsv_s=0.5,              # Giảm saturation
        hsv_v=0.3,              # Giảm value
        degrees=15.0,           # Tăng góc xoay
        translate=0.3,          # Tăng mức dịch chuyển
        scale=0.4,              # Tăng scale
        shear=0.2,              # Tăng shear
        perspective=0.0,        # Không dùng perspective
        flipud=0.0,             # Không lật dọc
        fliplr=0.3,             # Lật ngang vừa phải
        mosaic=0.7,             # Giảm mosaic
        mixup=0.1,              # Thêm mixup nhẹ
        amp=False,              # Tắt mixed precision do lỗi CUDA
        plots=True              # Lưu biểu đồ loss
    )
    shutil.copy("runs/detect/xiangqi_model/weights/best.pt", "xiangqi_yolov8.pt")
    return model

# Phát hiện quân cờ và vị trí


def detect_pieces(image_path):
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("Không thể đọc ảnh!")
        return
        
    # In thông tin về ảnh
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Pixel value range: [{np.min(image)}, {np.max(image)}]")
    
    # Kiểm tra và chuẩn hóa ảnh
    if len(image.shape) != 3:
        print("Warning: Image is not in RGB format")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] != 3:
        print("Warning: Unexpected number of channels")
    
    # Phát hiện lưới bàn cờ
    grid, board_map = detect_board_grid(image)

    # Tải model YOLOv8 đã huấn luyện
    print("Loading model...")
    model_path = "xiangqi_yolov8.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return
    try:
        model = YOLO(model_path)
        print("Model loaded successfully")
        # In thông tin về model
        print(f"Model info: {model.info()}")
        print(f"Model names (classes): {model.names}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
        
    try:
        # Resize ảnh về đúng kích thước training
        image_resized = cv2.resize(image, (640, 640))
        results = model(image_resized, conf=0.05)  # Giảm ngưỡng confidence xuống 0.05
        print(f"Raw detection results: {results[0]}")  # In chi tiết kết quả detection
        
        if len(results[0].boxes) == 0:
            # Thử detect với ảnh gốc
            print("Trying detection with original image...")
            results = model(image, conf=0.05)
            print(f"Raw detection results (original size): {results[0]}")
            
        if len(results[0].boxes) == 0:
            print("No objects detected. Try adjusting confidence threshold.")
    except Exception as e:
        print(f"Error during detection: {str(e)}")
        return
    print(f"Detection results: {len(results[0].boxes)} objects found")

    # Ánh xạ quân cờ
    board_state = map_to_board(results[0].boxes, grid, board_map)

    # In kết quả
    for piece, pos in board_state.items():
        print(f"{piece}: {pos}")

    # Vẽ nhãn lên ảnh
    for box in results[0].boxes:
        x, y, w, h = box.xywh[0]
        label = box.cls
        cv2.putText(
            image,
            f"{label}: {board_map[min(grid, key=lambda p: ((p[0]-x)**2 + (p[1]-y)**2)**0.5)]}",
            (int(x-w/2), int(y-h/2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    cv2.imwrite("output.jpg", image)


# Chạy huấn luyện và phát hiện
if __name__ == "__main__":
    # Bỏ comment để huấn luyện
    train_model()

    # Phát hiện trên ảnh mới
    # detect_pieces("testdata/test10.png")
    
