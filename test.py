"""
test.py
-------
Nhận dạng tiền Việt Nam theo thời gian thực qua webcam.

Yêu cầu:
    - File mô hình: vggmodel.h5 (sau khi chạy train.py)
    - File encoder: label_encoder.pickle (sau khi chạy train.py)

Chạy:
    python test.py

Điều khiển:
    - Nhấn 'Q' để thoát
"""

import cv2
import numpy as np
import pickle
from keras.models import load_model

# =============================================
# CẤU HÌNH
# =============================================
MODEL_PATH = "vggmodel.h5"
ENCODER_PATH = "label_encoder.pickle"
IMG_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.75    # Ngưỡng tin cậy tối thiểu để hiển thị kết quả

# Màu sắc hiển thị (BGR)
COLOR_SUCCESS = (0, 220, 80)    # Xanh lá – nhận dạng thành công
COLOR_WARNING = (0, 165, 255)   # Cam – độ tin cậy thấp
COLOR_INFO = (200, 200, 200)    # Xám – thông tin phụ

CURRENCY_LABELS = {
    "00000": "Khong cam tien",
    "01000": "1.000 VND",
    "02000": "2.000 VND",
    "05000": "5.000 VND",
    "10000": "10.000 VND",
    "20000": "20.000 VND",
    "50000": "50.000 VND",
    "100000": "100.000 VND",
    "200000": "200.000 VND",
    "500000": "500.000 VND",
}


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Tiền xử lý frame từ webcam để đưa vào mô hình."""
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def draw_overlay(frame: np.ndarray, label: str, confidence: float, low_conf: bool = False) -> np.ndarray:
    """Vẽ thông tin nhận dạng lên frame."""
    h, w = frame.shape[:2]

    # Nền mờ phía trên
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if low_conf:
        cv2.putText(frame, "Do tin cay thap...", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WARNING, 2)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_INFO, 1)
    else:
        display_label = CURRENCY_LABELS.get(label, label)
        cv2.putText(frame, display_label, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLOR_SUCCESS, 2)
        cv2.putText(frame, f"Confidence: {confidence:.1%}  |  Nhan: {label}", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1)

    # Viền khung trên
    cv2.line(frame, (0, 85), (w, 85), COLOR_SUCCESS if not low_conf else COLOR_WARNING, 2)
    return frame


def main():
    # Tải mô hình và encoder
    print("[INFO] Đang tải mô hình và encoder...")
    try:
        with open(ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
        class_names = label_encoder.classes_
        model = load_model(MODEL_PATH)
        print(f"[INFO] Đã tải mô hình: {MODEL_PATH}")
        print(f"[INFO] Các nhãn: {list(class_names)}")
    except FileNotFoundError as e:
        print(f"[LỖI] Không tìm thấy file: {e}")
        print("[GỢI Ý] Hãy chạy train.py trước để tạo mô hình!")
        return

    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[LỖI] Không thể mở webcam!")
        return

    print("[INFO] Bắt đầu nhận dạng. Nhấn 'Q' để thoát...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Tiền xử lý & dự đoán
        processed = preprocess_frame(frame)
        predictions = model.predict(processed, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        label = class_names[predicted_idx]

        # Hiển thị kết quả
        if confidence >= CONFIDENCE_THRESHOLD:
            frame = draw_overlay(frame, label, confidence, low_conf=False)
        else:
            frame = draw_overlay(frame, label, confidence, low_conf=True)

        cv2.imshow("Nhan Dang Tien Viet Nam  |  Nhan Q de thoat", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Đã dừng chương trình.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()