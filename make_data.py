"""
make_data.py
------------
Thu thập dữ liệu (ảnh) từ webcam để tạo tập dữ liệu huấn luyện.

Hướng dẫn sử dụng:
    1. Chỉnh biến `LABEL` thành mệnh giá bạn muốn chụp (ví dụ: "50000")
    2. Chạy script: python make_data.py
    3. Đưa tờ tiền vào trước camera
    4. Nhấn 'Q' để dừng
    
Dữ liệu được lưu vào thư mục: data/<LABEL>/
"""

import os
import cv2
import numpy as np

# =============================================
# CẤU HÌNH – Thay đổi theo mệnh giá bạn chụp
# =============================================
LABEL = "00000"        # Nhãn: "00000" = không cầm tiền, còn lại là mệnh giá (VD: "50000")
SKIP_FRAMES = 60       # Số frame bỏ qua ban đầu (để kịp đưa tiền lên)
MAX_FRAMES = 1500      # Số ảnh tối đa muốn thu thập
SCALE = 0.3            # Tỉ lệ thu nhỏ ảnh từ webcam


def main():
    save_dir = os.path.join("data", LABEL)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[LỖI] Không thể mở webcam!")
        return

    print(f"[INFO] Bắt đầu thu thập dữ liệu cho nhãn: '{LABEL}'")
    print(f"[INFO] Ảnh sẽ được lưu vào: {save_dir}")
    print(f"[INFO] Đang chờ {SKIP_FRAMES} frame... Chuẩn bị đưa tiền lên!")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        frame = cv2.resize(frame, dsize=None, fx=SCALE, fy=SCALE)

        # Hiển thị thông tin lên màn hình
        status = f"Chờ... ({frame_count}/{SKIP_FRAMES})" if frame_count < SKIP_FRAMES else f"Đang lưu: {saved_count}/{MAX_FRAMES}"
        color = (0, 0, 255) if frame_count < SKIP_FRAMES else (0, 255, 0)
        cv2.putText(frame, f"Nhan: {LABEL}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Thu thap du lieu - Nhan Q de thoat", frame)

        # Lưu ảnh sau khi đủ số frame bỏ qua
        if SKIP_FRAMES <= frame_count <= SKIP_FRAMES + MAX_FRAMES:
            filename = os.path.join(save_dir, f"{frame_count}.png")
            cv2.imwrite(filename, frame)
            saved_count += 1

        # Dừng khi đủ ảnh hoặc nhấn Q
        if saved_count >= MAX_FRAMES:
            print(f"[INFO] Đã thu thập đủ {MAX_FRAMES} ảnh!")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Dừng thu thập theo yêu cầu.")
            break

    print(f"[XONG] Tổng số ảnh đã lưu: {saved_count} → {save_dir}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()