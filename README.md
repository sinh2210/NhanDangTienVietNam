# 💵 Nhận Dạng Tiền Việt Nam

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-VGG16-D00000?style=for-the-badge&logo=keras&logoColor=white)

**Hệ thống nhận dạng mệnh giá tiền Việt Nam theo thời gian thực sử dụng Deep Learning**

</div>

---

## 📖 Giới thiệu

Dự án này xây dựng một mô hình **phân loại ảnh** có khả năng nhận dạng **10 mệnh giá tiền giấy Việt Nam** theo thời gian thực thông qua webcam. Mô hình được xây dựng dựa trên kiến trúc **VGG16** (Transfer Learning) và được huấn luyện với dữ liệu tự thu thập.

### Các mệnh giá được nhận dạng:
| Mệnh giá | Nhãn |
|-----------|------|
| Không cầm tiền | `00000` |
| 1.000 VNĐ | `01000` |
| 2.000 VNĐ | `02000` |
| 5.000 VNĐ | `05000` |
| 10.000 VNĐ | `10000` |
| 20.000 VNĐ | `20000` |
| 50.000 VNĐ | `50000` |
| 100.000 VNĐ | `100000` |
| 200.000 VNĐ | `200000` |
| 500.000 VNĐ | `500000` |

---

## 🏗️ Kiến trúc mô hình

```
Input (128x128x3)
     │
  VGG16 (frozen weights - ImageNet)
     │
  Flatten
     │
  Dense(4096, relu) → Dropout(0.5)
     │
  Dense(4096, relu) → Dropout(0.5)
     │
  Dense(10, softmax)
     │
  Output (10 classes)
```

---

## 📁 Cấu trúc dự án

```
NhanDangTienVietNam/
├── make_data.py          # Thu thập dữ liệu từ webcam
├── train.py              # Huấn luyện mô hình VGG16
├── test.py               # Nhận dạng thời gian thực qua webcam
├── requirements.txt      # Các thư viện cần cài đặt
└── README.md             # Tài liệu dự án
```

---

## ⚙️ Cài đặt

### Yêu cầu hệ thống
- Python 3.10
- Webcam (để thu thập dữ liệu và test)
- GPU (khuyên dùng để training nhanh hơn)

### Bước 1: Clone repository
```bash
git clone https://github.com/sinh2210/NhanDangTienVietNam.git
cd NhanDangTienVietNam
```

### Bước 2: Tạo môi trường ảo
```bash
python -m venv .venv310
# Windows:
.venv310\Scripts\activate
# Linux/Mac:
source .venv310/bin/activate
```

### Bước 3: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

---

## 🚀 Hướng dẫn sử dụng

### Bước 1: Thu thập dữ liệu
Mở `make_data.py` và chỉnh label tương ứng với mệnh giá bạn muốn chụp (ví dụ: `"50000"`):

```bash
python make_data.py
```

> ⚠️ Script sẽ tự động lưu ảnh sau **60 frame đầu** (tránh lúc chưa kịp cầm tiền lên). Nhấn `Q` để dừng.

Lặp lại bước này cho từng mệnh giá. Dữ liệu sẽ được lưu vào thư mục `data/`:
```
data/
├── 00000/   ← Không cầm tiền
├── 10000/
├── 50000/
└── ...
```

### Bước 2: Huấn luyện mô hình
```bash
python train.py
```

Sau khi train xong, các file sau sẽ được tạo ra:
- `vggmodel.h5` – Mô hình đã huấn luyện
- `label_encoder.pickle` – Encoder nhãn phân loại
- `weights-XX-X.XX.h5` – Checkpoint tốt nhất
- `roc.png` – Biểu đồ Accuracy và Loss

### Bước 3: Nhận dạng thời gian thực
```bash
python test.py
```

> 💡 Đưa tờ tiền trước webcam. Hệ thống sẽ hiển thị mệnh giá nếu độ tin cậy ≥ 75%.

---

## 📊 Thông số huấn luyện

| Thông số | Giá trị |
|----------|---------|
| Kích thước ảnh đầu vào | 128 × 128 |
| Số epoch | 75 |
| Batch size | 64 |
| Optimizer | Adam |
| Loss function | Categorical Crossentropy |
| Tỉ lệ dropout | 0.5 |
| Ngưỡng tin cậy (test) | 75% |

### Data Augmentation
- Xoay ảnh ngẫu nhiên: ±20°
- Zoom: 10%
- Dịch trái/phải, lên/xuống: 10%
- Lật ngang ngẫu nhiên
- Thay đổi độ sáng: [0.2, 1.5]

---

## 🔧 Yêu cầu thư viện

Xem file [`requirements.txt`](requirements.txt) để biết đầy đủ các thư viện cần thiết.

---

## 📝 Ghi chú

- Dữ liệu huấn luyện **không được đính kèm** trong repository (quá lớn). Bạn cần tự thu thập bằng `make_data.py`.
- Mô hình `vggmodel.h5` cũng **không được đính kèm**. Bạn cần tự train.
- Nên thu thập **ít nhất 500 ảnh/mệnh giá** để đạt độ chính xác cao.

---

## 👨‍💻 Tác giả

**Do Van Sinh** – [@sinh2210](https://github.com/sinh2210)

---

<div align="center">
  Made with ❤️ using VGG16 Transfer Learning
</div>
