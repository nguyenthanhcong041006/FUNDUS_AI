# Retinal Fundus Datasets Directory

Thư mục này được cấu trúc để lưu trữ và quản lý các bộ dữ liệu ảnh đáy mắt (fundus images) theo tài liệu kế hoạch `FUNDUS_AI_Complete_Pipeline_Implementation_Plan.md`.

---

## 1. Cấu trúc thư mục

```text
datasets/
├── raw/                      # Dữ liệu gốc tải về từ các nguồn chính thức
│   ├── drive/                # Bộ dữ liệu DRIVE (Vessel segmentation)
│   ├── rite/                 # Bộ dữ liệu RITE (Artery/Vein segmentation)
│   ├── refuge/               # Bộ dữ liệu REFUGE (Optic Disc/Cup, Fovea, Glaucoma)
│   ├── idrid/                # Bộ dữ liệu IDRiD (Lesions: MA, HE, EX, SE & DR Grading)
│   ├── aptos/                # Bộ dữ liệu APTOS 2019 (DR Severity Grading)
│   └── odir/                 # Bộ dữ liệu ODIR-5K (Multi-disease classification)
├── processed/                # Ảnh đã tiền xử lý (crop FOV, resize 512x512, CLAHE, v.v.)
├── annotations/              # Masks và nhãn được chuẩn hóa thành format chung
└── splits/                   # File danh sách train/val/test chia theo bệnh nhân (patient-level)
```

---

## 2. Hướng dẫn đặt dữ liệu vào từng thư mục

### 2.1 DRIVE (Digital Retinal Images for Vessel Extraction)

- **Mục đích:** Huấn luyện mô hình phân đoạn mạch máu võng mạc (Blood vessel segmentation).
- **Nguồn tải:** [DRIVE Grand Challenge](https://drive.grand-challenge.org/)
- **Cấu trúc gợi ý:**
  ```text
  datasets/raw/drive/
  ├── training/
  │   ├── images/         # 21_training.tif ... 40_training.tif
  │   ├── 1st_manual/     # 21_manual1.gif ... 40_manual1.gif (Ground truth)
  │   └── mask/           # 21_training_mask.gif ... (FOV mask)
  └── test/
      ├── images/         # 01_test.tif ... 20_test.tif
      ├── 1st_manual/     # 01_manual1.gif ... 20_manual1.gif
      └── mask/           # 01_test_mask.gif ...
  ```

### 2.2 RITE (Retinal Image Bank for Artery and Vein Segmentation)

- **Mục đích:** Phân loại động mạch (Artery) và tĩnh mạch (Vein) (A/V ratio).
- **Nguồn tải:** Dựa trên tập DRIVE mở rộng với nhãn động - tĩnh mạch.
- **Cấu trúc gợi ý:**
  ```text
  datasets/raw/rite/
  ├── training/
  │   ├── images/
  │   └── av/             # Masks màu: Đỏ (Artery), Xanh lam (Vein), Xanh lá (Overlap)
  └── test/
      ├── images/
      └── av/
  ```

### 2.3 REFUGE (Retinal Fundus Glaucoma Challenge)

- **Mục đích:** Phân đoạn Gai thị (Optic Disc - OD) và Lõm gai (Optic Cup - OC), định vị hoàng điểm (Fovea), phân loại Glaucoma.
- **Nguồn tải:** [REFUGE Challenge](https://refuge.grand-challenge.org/)
- **Cấu trúc gợi ý:**
  ```text
  datasets/raw/refuge/
  ├── train/
  │   ├── images/
  │   └── Disc_Cup_Masks/
  ├── val/
  │   ├── images/
  │   └── Disc_Cup_Masks/
  ├── test/
  └── labels.csv          # Cột: Image_ID, Glaucoma_Label, Fovea_X, Fovea_Y
  ```

### 2.4 IDRiD (Indian Diabetic Retinopathy Image Dataset)

- **Mục đích:** Phân đoạn vi tổn thương đáy mắt (Microaneurysms, Hemorrhages, Hard Exudates, Soft Exudates) và chấm điểm bệnh võng mạc tiểu đường (DR grading).
- **Nguồn tải:** [IDRiD Challenge](https://idrid.grand-challenge.org/)
- **Cấu trúc gợi ý:**
  ```text
  datasets/raw/idrid/
  ├── A_Segmentation/
  │   ├── 1_Microaneurysms/
  │   ├── 2_Haemorrhages/
  │   ├── 3_Hard_Exudates/
  │   ├── 4_Soft_Exudates/
  │   └── 5_Optic_Disc/
  └── B_Disease_Grading/
      ├── 1_Original_Images/
      └── 2_Groundtruths/
  ```

### 2.5 APTOS 2019 Blindness Detection

- **Mục đích:** Phân loại mức độ nghiêm trọng bệnh võng mạc đái tháo đường (DR severity grades 0-4).
- **Nguồn tải:** [Kaggle APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection)
- **Cấu trúc gợi ý:**
  ```text
  datasets/raw/aptos/
  ├── train_images/       # [id_code].png
  ├── test_images/
  └── train.csv           # id_code, diagnosis (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative)
  ```

### 2.6 ODIR-5K (Ocular Disease Intelligent Recognition)

- **Mục đích:** Nhận diện đa bệnh lý đáy mắt (Multi-label disease classification).
- **Nguồn tải:** Grand Challenge hoặc Kaggle ODIR-5K.
- **Cấu trúc gợi ý:**
  ```text
  datasets/raw/odir/
  ├── ODIR-5K_Training_Images/
  ├── ODIR-5K_Testing_Images/
  └── full_df.csv         # ID, Left-Fundus, Right-Fundus, N, D, G, C, A, H, M, O
  ```

---

## 3. Lưu ý quan trọng khi chuẩn bị dữ liệu (Từ mục 6.4 trong Kế hoạch)

1. **Chia dữ liệu theo bệnh nhân (Patient-level split):** Tuyệt đối không để ảnh mắt trái và mắt phải của cùng một bệnh nhân nằm ở 2 tập khác nhau (train và test) nhằm tránh data leakage.
2. **Không tự động coi pixel không gán nhãn là âm tính:** Trong các tập phân đoạn tổn thương (như IDRiD), chỉ những ảnh có mask mới dùng để tính độ nhạy/độ đặc hiệu ở cấp độ mask.
3. **Kích thước ảnh:** Thư mục `processed/` sẽ chứa các ảnh chuẩn hóa (khuyên dùng kích thước cơ sở `512x512` pixel hoặc crop FOV bảo toàn tỉ lệ).
