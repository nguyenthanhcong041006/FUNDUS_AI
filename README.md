# FUNDUS AI — Retinal Fundus Image Analysis Prototype

> **Medical Disclaimer:** This system is a modular research and screening prototype, NOT a replacement for an ophthalmologist. Color fundus photography alone cannot confirm every disease or substitute comprehensive clinical examination (e.g. OCT, visual field, intraocular pressure).

---

## 1. Project Overview

**FUNDUS AI** is a modular deep learning and OpenCV pipeline for comprehensive retinal fundus image processing, anatomical structure segmentation, lesion detection, quantitative biomarker measurement, disease screening, and structured report generation.

### Pipeline Workflow:
```text
Fundus Image ──► Quality Assessment (Blur, Brightness, Contrast)
             ──► Retinal FOV Detection (Aperture Masking)
             ──► Contrast Enhancement (Green-channel CLAHE, LAB, HSV)
             ──► Anatomical Segmentation:
                 • Optic Disc & Optic Cup (VCDR, HCDR, Area Ratio)
                 • Blood Vessel Network (Density, Length, Width, Branch points)
                 • Fovea & Macula Localization (Disc-Fovea Distance)
             ──► Lesion Detection:
                 • Microaneurysms (MA) & Hemorrhages (HE)
                 • Hard Exudates (EX) & Soft Exudates / CWS
                 • Drusen & Neovascularization signs
             ──► Disease Screening & Grading:
                 • Diabetic Retinopathy (5-grade ICDR scale)
                 • Glaucoma Risk Indicators (VCDR evaluation)
                 • Hypertensive Retinopathy Signs (A/V ratio)
                 • Age-Related Macular Degeneration (AMD)
             ──► Multi-Layer Visualization & Reporting (JSON, CSV, Annotated PNG)
```

---

## 2. Directory Structure

```text
eyes/
├── datasets/                     # Folder để đặt các bộ dữ liệu
│   ├── raw/                      # Dữ liệu ảnh và nhãn gốc
│   │   ├── drive/                # Bộ dữ liệu DRIVE (Vessel segmentation)
│   │   ├── rite/                 # Bộ dữ liệu RITE (Artery/Vein classification)
│   │   ├── refuge/               # Bộ dữ liệu REFUGE (Optic Disc/Cup & Glaucoma)
│   │   ├── idrid/                # Bộ dữ liệu IDRiD (Lesions & DR grading)
│   │   ├── aptos/                # Bộ dữ liệu APTOS 2019 (DR severity grading)
│   │   └── odir/                 # Bộ dữ liệu ODIR-5K (Multi-disease classification)
│   ├── processed/                # Ảnh đã tiền xử lý / chuẩn hóa
│   ├── annotations/              # Masks và ground truths đã đồng bộ
│   ├── splits/                   # File phân chia train/val/test theo bệnh nhân
│   └── README.md                 # Hướng dẫn chi tiết cách tải và đặt dữ liệu
├── configs/                      # File cấu hình YAML
│   ├── preprocessing.yaml        # Ngưỡng chất lượng ảnh, FOV, CLAHE, chuẩn hóa
│   ├── segmentation.yaml         # Kiến trúc mạng, loss, siêu tham số phân đoạn
│   ├── classification.yaml       # Backbone phân loại, nhãn bệnh, loss
│   └── evaluation.yaml           # Metric đánh giá, màu sắc hiển thị overlay
├── src/                          # Toàn bộ mã nguồn module hóa
│   ├── preprocessing/            # Image loader, Quality Assessment, FOV, CLAHE
│   ├── anatomical/               # Optic Disc, Optic Cup, Vessels, Artery/Vein, Fovea
│   ├── lesions/                  # Microaneurysms, Hemorrhages, Exudates, CWS, Drusen
│   ├── models/                   # U-Net, Attention U-Net, EfficientNet, Fusion Model
│   ├── measurements/             # Đo đạc gai thị, mạch máu, tổn thương, phân bố không gian
│   ├── classification/           # Chấm điểm DR, Glaucoma, Tăng huyết áp, AMD
│   ├── visualization/            # Overlay đa lớp, Landmark crosshairs, Bảng tổng hợp
│   └── evaluation/               # Dice, IoU, Precision, Recall, MAE, Bland-Altman
├── checkpoints/                  # Lưu trữ trọng số mô hình đã huấn luyện (.pth)
│   ├── optic_disc/
│   ├── vessels/
│   ├── lesions/
│   └── classification/
├── outputs/                      # Kết quả phân tích xuất ra
│   ├── annotated_images/         # Ảnh vẽ mask và nhãn
│   ├── masks/                    # File mask nhị phân PNG từng cấu trúc/tổn thương
│   ├── reports/                  # Báo cáo JSON và CSV
│   └── logs/                     # Log thực thi
├── tests/                        # Unit tests kiểm thử hệ thống
├── main.py                       # CLI Runner chính chạy toàn bộ pipeline
├── requirements.txt              # Danh sách thư viện Python
├── .gitignore                    # Bỏ qua dữ liệu nặng và cache
└── FUNDUS_AI_Complete_Pipeline_Implementation_Plan.md
```

---

## 3. Cài đặt môi trường (Setup Guide)

Khuyến nghị sử dụng **Python 3.11** (hoặc 3.12/3.14):

```powershell
# Tạo môi trường ảo
py -3.11 -m venv .venv

# Kích hoạt môi trường ảo trên Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Nâng cấp pip và cài đặt thư viện
python -m pip install --upgrade pip
pip install -r requirements.txt

# Cài đặt PyTorch phù hợp với GPU NVIDIA của bạn (nếu có CUDA):
# Tham khảo: https://pytorch.org/get-started/locally/
# Ví dụ CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 4. Hướng dẫn chạy thử nghiệm (Milestone 1)

### Chạy chế độ Demo (Synthetic Fundus)
Bạn có thể chạy thử toàn bộ pipeline ngay lập tức mà chưa cần tải dataset. Hệ thống sẽ tự tạo ảnh đáy mắt mô phỏng, chạy qua tất cả các bước tiền xử lý, phân đoạn cấu trúc, phát hiện tổn thương, tính toán số đo và xuất báo cáo:

```powershell
python main.py --demo --output_dir outputs/sample_demo
```

### Chạy phân tích trên ảnh đáy mắt thật của bạn
```powershell
python main.py --image "duong_dan_anh_day_mat.jpg" --output_dir "outputs/sample_001"
```

### Kết quả xuất ra tại `outputs/sample_001/`:
* `original.png`: Ảnh gốc chuẩn hóa kích thước.
* `green_channel.png`: Kênh màu xanh lá (tương phản mạch máu cao nhất).
* `enhanced_green.png`: Kênh xanh lá áp dụng CLAHE và giảm nhiễu.
* `fov_mask.png`: Mặt nạ vùng võng mạc (loại bỏ viền đen).
* `annotated.png`: Ảnh tổng hợp phủ mask màu các lớp và định vị hoàng điểm (Fovea), gai thị.
* `report_panel.png`: Bảng ảnh 6 khung so sánh toàn diện các giai đoạn xử lý.
* `masks/`: Thư mục chứa từng mask riêng biệt: `optic_disc.png`, `optic_cup.png`, `vessels.png`, `hard_exudates.png`, `hemorrhages.png`, `microaneurysms.png`, `cotton_wool.png`.
* `quality_report.json`: Báo cáo đánh giá chất lượng kỹ thuật của ảnh (độ mờ, độ sáng, độ tương phản, FOV).
* `report.json`: Báo cáo chẩn đoán hoàn chỉnh dạng JSON theo cấu trúc Phase 14.
* `measurements.csv`: Bảng tổng hợp các chỉ số định lượng đo được.

---

## 5. Nơi để Dataset

Toàn bộ dataset bạn sẽ để vào thư mục `datasets/raw/`. Xem chi tiết tại [datasets/README.md](file:///c:/Users/THANH%20CONG/Documents/eyes/datasets/README.md):
* **DRIVE:** `datasets/raw/drive/` (Mạch máu)
* **RITE:** `datasets/raw/rite/` (Động mạch / Tĩnh mạch)
* **REFUGE:** `datasets/raw/refuge/` (Gai thị, Lõm gai, Hoàng điểm, Glaucoma)
* **IDRiD:** `datasets/raw/idrid/` (Tổn thương: MA, HE, EX, SE và chấm điểm DR)
* **APTOS 2019:** `datasets/raw/aptos/` (Phân loại mức độ DR)
* **ODIR-5K:** `datasets/raw/odir/` (Đa bệnh lý đáy mắt)

