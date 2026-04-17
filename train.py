"""
train.py
--------
Huấn luyện mô hình nhận dạng tiền Việt Nam bằng VGG16 Transfer Learning.

Pipeline:
    1. Đọc ảnh từ thư mục data/ và lưu vào file pix.data (nếu chưa có)
    2. Chia tập train/test (80/20)
    3. Xây dựng mô hình VGG16 với các lớp fully-connected tùy chỉnh
    4. Huấn luyện với data augmentation
    5. Lưu mô hình và biểu đồ kết quả

Chạy:
    python train.py
"""

import os
import pickle
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =============================================
# CẤU HÌNH
# =============================================
RAW_FOLDER = "data"            # Thư mục chứa ảnh phân loại theo nhãn
DATA_FILE = "pix.data"         # File lưu cache dữ liệu đã xử lý
ENCODER_FILE = "label_encoder.pickle"
MODEL_FILE = "vggmodel.h5"
IMG_SIZE = (128, 128)
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 75
TEST_SIZE = 0.2
RANDOM_STATE = 100


# =============================================
# 1. XỬ LÝ & LƯU DỮ LIỆU
# =============================================
def save_data(raw_folder: str = RAW_FOLDER) -> None:
    """Đọc toàn bộ ảnh từ thư mục raw_folder, resize và lưu vào file pickle."""
    print("=" * 50)
    print("[BƯỚC 1] Đang xử lý ảnh từ thư mục:", raw_folder)
    print("=" * 50)

    pixels, labels = [], []

    for folder in sorted(os.listdir(raw_folder)):
        if folder == ".DS_Store":
            continue
        folder_path = os.path.join(raw_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"  → Đang đọc nhãn: {folder}")
        for file in os.listdir(folder_path):
            if file == ".DS_Store":
                continue
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"    [CẢNH BÁO] Không đọc được ảnh: {img_path}")
                continue
            pixels.append(cv2.resize(img, dsize=IMG_SIZE))
            labels.append(folder)

    pixels = np.array(pixels)
    labels = np.array(labels)
    print(f"\n[INFO] Tổng số ảnh: {len(pixels)}, Số nhãn phân biệt: {len(set(labels))}")

    encoder = LabelBinarizer()
    labels_encoded = encoder.fit_transform(labels)
    print("[INFO] Các nhãn:", list(encoder.classes_))

    with open(DATA_FILE, "wb") as f:
        pickle.dump((pixels, labels_encoded), f)
    with open(ENCODER_FILE, "wb") as f:
        pickle.dump(encoder, f)

    print(f"[XONG] Đã lưu dữ liệu → {DATA_FILE} và {ENCODER_FILE}\n")


def load_data() -> tuple:
    """Tải dữ liệu đã lưu từ file pickle."""
    with open(DATA_FILE, "rb") as f:
        pixels, labels = pickle.load(f)
    print(f"[INFO] Đã tải dữ liệu: pixels={pixels.shape}, labels={labels.shape}")
    return pixels, labels


# =============================================
# 2. XÂY DỰNG MÔ HÌNH
# =============================================
def build_model(num_classes: int = NUM_CLASSES) -> Model:
    """
    Xây dựng mô hình VGG16 Transfer Learning.
    Các lớp conv của VGG16 được đóng băng (frozen).
    Thêm 2 lớp FC + Dropout phía sau.
    """
    base_model = VGG16(weights="imagenet", include_top=False)
    for layer in base_model.layers:
        layer.trainable = False  # Đóng băng toàn bộ VGG16

    inp = Input(shape=(*IMG_SIZE, 3), name="image_input")
    x = base_model(inp)
    x = Flatten(name="flatten")(x)
    x = Dense(4096, activation="relu", name="fc1")(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation="relu", name="fc2")(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    print("\n[INFO] Kiến trúc mô hình:")
    model.summary()
    return model


# =============================================
# 3. VẼ BIỂU ĐỒ KẾT QUẢ
# =============================================
def plot_training_history(history, output_path: str = "roc.png") -> None:
    """Vẽ và lưu biểu đồ Accuracy & Loss theo epoch."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Kết quả huấn luyện mô hình VGG16", fontsize=14, fontweight="bold")

    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    # Accuracy
    ax1.plot(epochs_range, history.history["accuracy"], label="Train")
    ax1.plot(epochs_range, history.history["val_accuracy"], label="Validation")
    ax1.set_title("Độ chính xác (Accuracy)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs_range, history.history["loss"], label="Train")
    ax2.plot(epochs_range, history.history["val_loss"], label="Validation")
    ax2.set_title("Hàm mất mát (Loss)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Đã lưu biểu đồ → {output_path}")


# =============================================
# 4. MAIN
# =============================================
def main():
    # Bước 1: Chuẩn bị dữ liệu
    if not os.path.exists(DATA_FILE):
        save_data()
    else:
        print(f"[INFO] Đã tìm thấy file cache '{DATA_FILE}', bỏ qua bước xử lý ảnh.")

    X, y = load_data()

    # Bước 2: Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    # Bước 3: Xây dựng mô hình
    model = build_model(num_classes=NUM_CLASSES)

    # Bước 4: Callbacks
    checkpoint_cb = ModelCheckpoint(
        filepath="weights-{epoch:02d}-{val_accuracy:.2f}.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        mode="max"
    )
    early_stop_cb = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Bước 5: Data Augmentation
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.2, 1.5],
        fill_mode="nearest"
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    # Bước 6: Huấn luyện
    print("\n" + "=" * 50)
    print(f"[BƯỚC 2] Bắt đầu huấn luyện ({EPOCHS} epochs, batch={BATCH_SIZE})")
    print("=" * 50)
    history = model.fit(
        train_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=val_gen.flow(X_test, y_test, batch_size=BATCH_SIZE),
        callbacks=[checkpoint_cb, early_stop_cb],
    )

    # Bước 7: Lưu mô hình và biểu đồ
    model.save(MODEL_FILE)
    print(f"\n[XONG] Mô hình đã được lưu → {MODEL_FILE}")
    plot_training_history(history)


if __name__ == "__main__":
    main()