# Báo cáo huấn luyện & đánh giá FUNDUS AI (chạy tự động 07/11/2026)

Báo cáo sơ bộ về những gì đã chạy, kết quả đo được và các điểm cần lưu ý. Mọi số liệu lấy từ file log/CSV trong `outputs/`, không ước lượng.

## 1. Đã chạy những gì

Script `run_all.ps1` (chạy nền, Python global `C:\Program Files\Python310\python.exe`) chờ APTOS (PID 3308) xong rồi chạy tuần tự. Tất cả bước đều thoát mã 0.

| Bước | Thời gian | Log |
|---|---|---|
| M1 train U-Net trên REFUGE2, 30 epoch, batch 4, 512px | 02:48 → 03:35 | `outputs/logs/autorun/m1_refuge2_train.*` |
| M6 train DenseNet-121 trên ODIR, 30 epoch | 03:35 → 06:43 | `m6_odir_train.*` |
| Đánh giá M1 trên REFUGE2 test (400 ảnh) | 1 phút | `m1_refuge2_test.out.log` |
| Đánh giá M1 trên Drishti-GS1 (nội bộ) + PAPILA (ngoài) | 1 phút | `m1_drishti_papila.out.log` |
| Đánh giá M2 trên DRIVE test | 6 giây | `m2_vessels_eval.out.log` |
| pytest | 16 passed | `pytest.out.log` |
| Pipeline end-to-end, 2 ảnh mỗi dataset (16 ảnh) | 36 giây | `outputs/full_run/batch_summary.csv` |

M2 (DRIVE, 50 epoch), M3 (RITE, 50 epoch), M4 (IDRiD, 50 epoch) và M5 (APTOS, 30 epoch) đã train trước đó; kết quả lấy từ `outputs/plots/*/*_training_history.csv`.

Sao lưu trước khi ghi đè: `checkpoints/optic_disc_cup/best_od_oc_drishti_backup.pth` (model Drishti cũ) và `checkpoints/classification/best_dr_classifier_final_backup.pth`.

## 2. Kết quả theo mô hình

| Module | Dataset | Mô hình | Kết quả chính (checkpoint đã lưu) |
|---|---|---|---|
| M1 Đĩa/hõm gai | REFUGE2 | U-Net | **Test (400 ảnh): Dice trung bình 0.8148** (đĩa 0.8707, hõm 0.7589), sai số vCDR (MAE) 0.0818 |
| M2 Mạch máu | DRIVE | U-Net | Test (20 ảnh): Dice 0.8079, IoU 0.6785, Sensitivity 0.8092, Specificity 0.9828 |
| M3 Động/tĩnh mạch | RITE | U-Net 3 lớp | Dice trung bình 0.6239 (động mạch 0.5701, tĩnh mạch 0.6777), epoch 50 |
| M4 Tổn thương | IDRiD | Attention U-Net | Dice trung bình 0.2640 (tốt nhất epoch 46) |
| M5 Chấm điểm DR | APTOS | EfficientNet-B0 | Val: QWK 0.9124, accuracy 84.29% (checkpoint epoch 25) |
| M6 Đa bệnh | ODIR-5K | DenseNet-121 | Val: Macro F1 0.5222, val loss 0.2802 (checkpoint epoch 8) |

Ghi chú về cách đọc:
- Cột "od_dice"/"cup_dice" trong `rite_training_history.csv` thực chất là Dice động mạch/tĩnh mạch (đặt tên theo hàm vẽ biểu đồ dùng chung).
- M4: số 0.264 là Dice trung bình 4 loại tổn thương; chưa có số liệu riêng từng loại.
- M6: Macro F1 của epoch cuối là 0.5327, cao hơn epoch được lưu (0.5222) nhưng val loss của epoch cuối là 0.4988, tăng so với 0.2802 ở epoch 8 (dấu hiệu overfit). Checkpoint được chọn theo val loss.

## 3. Kiểm tra chéo M1 (kết quả đáng chú ý nhất)

| Model | Drishti test (51 ảnh) — Dice đĩa / hõm / MAE vCDR | PAPILA ngoài (488 ảnh) — Dice đĩa / hõm | AUC glaucoma từ vCDR (PAPILA) |
|---|---|---|---|
| **Train REFUGE2** (hiện đang là `best_od_oc_model.pth`) | 0.3139 / 0.1791 / 0.3674 | 0.0090 / 0.0101 | 0.4805 |
| **Train Drishti** (bản sao lưu) | 0.9348 / 0.8253 / 0.0894 | 0.6063 / 0.3039 | 0.4643 |

- Model REFUGE2 đạt tốt trên chính REFUGE2 (0.81) nhưng **gần như không dùng được trên Drishti và PAPILA** (Dice đĩa trên PAPILA 0.009). Chưa xác định nguyên nhân; khả năng cao là khác biệt miền dữ liệu (máy chụp, kích thước/vị trí đĩa trong ảnh, màu), nhưng chưa kiểm chứng.
- Model Drishti tốt trên Drishti nhưng giảm mạnh trên PAPILA (hõm 0.30).
- AUC sàng lọc glaucoma từ vCDR trên PAPILA ở cả hai model đều ≈ 0.46-0.48, tức không tốt hơn đoán ngẫu nhiên. Sensitivity thấp (0.00 và 0.14). Không nên dùng vCDR của các model này để sàng lọc glaucoma trên PAPILA.
- Kết quả đánh giá model Drishti nằm ở `outputs/drishti_baseline/`; model REFUGE2 ở `outputs/optic_disc_cup_*_evaluation.csv`.
- **Lưu ý cho pipeline `main.py`:** các lớp OpticDisc/OpticCup nạp `checkpoints/optic_disc_cup/best_od_oc_model.pth`, hiện là model REFUGE2. Với ảnh ngoài REFUGE, model Drishti cũ có thể cho vùng đĩa tốt hơn. Cần quyết định dùng bản nào.

## 4. Điểm cần lưu ý về độ tin cậy

1. **Chọn checkpoint trên tập test** ở M1-Drishti, M2 (DRIVE), M3 (RITE), M4 (IDRiD): script chọn epoch tốt nhất theo chính tập test nên các số này lạc quan. Riêng M1-REFUGE2 chọn theo `refuge2_val` và báo cáo trên `refuge2_test` riêng biệt, nên số 0.8148 sạch hơn.
2. **Dao động lớn khi train M1-REFUGE2:** Dice val giữa các epoch dao động mạnh (ví dụ epoch 25: 0.8029, epoch 26: 0.4731, epoch 29: 0.5990), gợi ý learning rate 1e-3 chưa ổn định. Chưa thử giảm lr hoặc dùng scheduler.
3. **Tập "val/test" của REFUGE2** lấy từ REFUGE1-val và REFUGE1-test trong thư mục `Train/` (có mask hõm), không dùng thư mục `Validation/` và `Test/` chính thức của REFUGE2.
4. **Tập kiểm tra rất nhỏ** với DRIVE (20 ảnh), RITE (20 ảnh), IDRiD (27 ảnh), Drishti (51 ảnh): số liệu có độ biến thiên cao.
5. **Pipeline end-to-end** chỉ chạy 16 ảnh mẫu để kiểm tra luồng xử lý chạy thông; nhãn đầu ra (DR, glaucoma, khuyến nghị chuyển viện) trong `batch_summary.csv` chưa được đối chiếu với nhãn thật. Đây là prototype nghiên cứu, không phải công cụ chẩn đoán.
6. M6: Macro F1 ≈ 0.52 với ngưỡng 0.5 cho thấy mô hình phát hiện kém các bệnh hiếm; chưa thử tinh chỉnh ngưỡng hoặc cân bằng lớp.
7. Cảnh báo `RuntimeWarning` của `runpy` ở bước đánh giá DRIVE vô hại.

## 5. Vị trí file kết quả

- Biểu đồ, ảnh dự đoán mẫu, lịch sử huấn luyện: `outputs/plots/<tên bài toán>/`
- Đánh giá M1: `outputs/optic_disc_cup_internal_evaluation.csv`, `outputs/optic_disc_cup_external_evaluation.csv`
- Đánh giá M2: `outputs/vessel_evaluation_results.csv`
- Pipeline: `outputs/full_run/` (mỗi ảnh một thư mục, tổng hợp ở `batch_summary.csv`)
- Log từng bước: `outputs/logs/autorun/`
- Checkpoint: `checkpoints/`

## 6. Việc gợi ý tiếp theo

1. Quyết định checkpoint OD/OC dùng cho pipeline (REFUGE2 hay Drishti), hoặc train gộp nhiều dataset để tăng khả năng tổng quát hoá.
2. Giảm learning rate / thêm scheduler cho M1 và chọn checkpoint theo tập val tách riêng cho M2-M4.
3. Bổ sung `evaluate_optic_disc_cup.py` hỗ trợ REFUGE2 và đánh giá glaucoma (AUC) theo nhãn REFUGE.
4. Cải thiện M4 (Dice 0.26) và M6 (F1 0.52).
