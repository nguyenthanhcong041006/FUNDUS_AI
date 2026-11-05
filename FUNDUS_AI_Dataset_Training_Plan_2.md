# FUNDUS AI — Kế hoạch huấn luyện theo Dataset, Nhiệm vụ và Model

> **Mục tiêu:** Xác định dataset nào được sử dụng cho từng nhiệm vụ, model phù hợp và vai trò của từng dataset trong hệ thống FUNDUS AI.
>
> **Phạm vi:** Prototype nghiên cứu và hỗ trợ sàng lọc bệnh võng mạc, không thay thế chẩn đoán của bác sĩ.

## 1. Tổng quan

Không gộp tất cả dataset thành một tập để train một model duy nhất. Các dataset có loại annotation khác nhau: pixel-level masks, nhãn phân loại cấp ảnh hoặc nhãn đa bệnh.

Vì vậy, FUNDUS AI sẽ sử dụng các model chuyên biệt theo từng nhiệm vụ rồi tích hợp vào một pipeline thống nhất.

| Module | Nhiệm vụ | Dataset chính | Model đề xuất |
|---|---|---|---|
| M1 | Phân đoạn optic disc/cup và hỗ trợ phân tích glaucoma | Drishti-GS1; PAPILA dùng External Validation | U-Net / Attention U-Net |
| M2 | Phân đoạn mạch máu | DRIVE; STARE dùng External Validation | U-Net |
| M3 | Phân biệt artery/vein | RITE | U-Net đa lớp |
| M4 | Phân đoạn tổn thương DR | IDRiD | U-Net / Attention U-Net |
| M5 | Phân loại mức độ DR | APTOS 2019 + IDRiD | EfficientNet-B0 |
| M6 | Phân loại đa bệnh | ODIR-5K | DenseNet121 / EfficientNet-B0 |
| M7 | Đánh giá chất lượng ảnh | Cần tự xây dựng/gán nhãn dữ liệu chất lượng | EfficientNet-B0 nhỏ hoặc OpenCV rules |

**Lưu ý:** Glaucoma classification chỉ được triển khai khi có image-level glaucoma labels phù hợp. Không mặc định sử dụng segmentation masks hoặc CDR để thay thế nhãn chẩn đoán.

## 2. Vai trò và cách sử dụng từng dataset

### 2.1 Drishti-GS1 — Optic Disc/Cup Segmentation

- **Vai trò:** Dataset chính cho module optic nerve, thay thế G1020 trong kế hoạch hiện tại.
- **Task A:** Optic Disc (OD) segmentation.
- **Task B:** Optic Cup (OC) segmentation.
- **Task C:** Tính vertical Cup-to-Disc Ratio (vCDR) từ segmentation masks.
- **Quy mô được công bố:** 101 ảnh, gồm 50 ảnh training và 51 ảnh test. Ground truth của test set có thể yêu cầu đăng ký để truy cập.
- **Annotation:** OD SoftMap, Cup SoftMap, AvgBoundary, Disk Center và CDR values.
- **Model:** U-Net làm baseline; Attention U-Net có thể dùng để so sánh.
- **Input:** Ảnh fundus RGB.
- **Output:** Hai segmentation maps, tương ứng OD và OC.
- **Loss:** BCE + Dice Loss; có thể thử nghiệm soft-label loss khi sử dụng SoftMap.
- **Metrics:** Dice, IoU, sensitivity, specificity và boundary distance riêng cho OD và OC.

**Cấu trúc annotation đã xác minh trên sample `drishtiGS_002`:**

| File | Vai trò |
|---|---|
| `drishtiGS_002.png` | Ảnh fundus gốc |
| `cupsegSoftmap.png` | Soft annotation của Optic Cup |
| `ODsegSoftmap.png` | Soft annotation của Optic Disc |
| `CupAvgBoundary.txt` | Tọa độ Cup AvgBoundary |
| `ODAvgBoundary.txt` | Tọa độ OD AvgBoundary |
| `diskCenter.txt` | Tọa độ tâm Optic Disc |
| `cdrValues.txt` | CDR values từ 4 chuyên gia |

CDR của sample `drishtiGS_002`:

- Expert 1: 0.76
- Expert 2: 0.77
- Expert 3: 0.79
- Expert 4: 0.75
- Mean CDR: **0.7675**

**Cách sử dụng annotation:**

1. Sử dụng ảnh fundus làm input.
2. Sử dụng OD và Cup SoftMap làm segmentation targets.
3. Giữ nguyên SoftMap gốc và tạo binary masks riêng nếu cần.
4. Xác định threshold chuyển SoftMap thành binary mask bằng thực nghiệm và ghi rõ trong configuration.
5. Sử dụng AvgBoundary để kiểm tra và đánh giá đường biên.
6. Sử dụng Disk Center cho các phân tích vị trí đĩa thị.
7. Sử dụng CDR values để đánh giá sai số CDR được tính từ segmentation masks.

**Lưu ý quan trọng:**

- Không mặc định SoftMap là binary mask.
- Không tự động threshold ở giá trị 128 nếu chưa xác minh cách mã hóa pixel.
- Không sử dụng `fillPoly` trực tiếp với AvgBoundary cho đến khi xác minh thứ tự điểm và quy ước tọa độ.
- Cần kiểm tra toàn bộ annotation của 50 ảnh training trước khi xây dựng DataLoader.
- CDR là chỉ số hình thái hỗ trợ nghiên cứu, không tự nó xác nhận glaucoma.
- Drishti-GS1 không được mặc định xem là dataset glaucoma classification nếu chưa xác minh nhãn image-level tương ứng.

### 2.2 PAPILA — External Validation cho Optic Disc/Cup và Glaucoma

- **Vai trò:** Dataset độc lập để đánh giá khả năng tổng quát hóa của model optic nerve.
- **Quy mô được công bố:** 488 ảnh từ 244 bệnh nhân, gồm dữ liệu hai mắt.
- **Task A:** OD/OC segmentation nếu annotation tương ứng có trong bản dataset sử dụng.
- **Task B:** Glaucoma classification dựa trên nhãn được cung cấp.
- **Model:** Sử dụng model đã train và chọn trên Drishti-GS1.
- **Metrics segmentation:** Dice, IoU, sensitivity, specificity.
- **Metrics classification:** ROC-AUC, sensitivity, specificity, macro-F1 và confusion matrix.

**Quy tắc External Validation:**

- Không dùng PAPILA để train, fine-tune, chọn checkpoint hoặc điều chỉnh threshold trước lần external evaluation chính thức.
- Nếu sau này dùng PAPILA cho phát triển model, phải có một external test độc lập khác.
- Nếu chia dữ liệu PAPILA cho nghiên cứu nội bộ, phải chia theo patient ID, không tách hai mắt của cùng bệnh nhân sang các split khác nhau.
- Kiểm tra tính tương thích của annotation, kích thước ảnh, FOV và quy ước CDR trước khi đánh giá.
- Báo cáo kết quả PAPILA riêng, không gộp trực tiếp với Drishti-GS1.

**Lưu ý:** PAPILA là dataset bổ sung, không thay thế hoàn toàn Drishti-GS1 trong giai đoạn xây dựng baseline.

### 2.3 DRIVE — Retinal Vessel Segmentation

- **Vai trò:** Dataset chính để train model phân đoạn hệ thống mạch máu võng mạc.
- **Task:** Semantic segmentation từng pixel.
- **Input:** Ảnh fundus màu.
- **Output:** Mask nhị phân (vessel/background).
- **Model:** U-Net, có thể sử dụng encoder ResNet34.
- **Loss:** Binary Cross-Entropy + Dice Loss.
- **Metrics:** Dice, IoU, sensitivity/recall, specificity.
- **Sử dụng:** Train trên official training set; tách validation từ training set nếu cần; giữ official test để đánh giá cuối.
- **Ứng dụng:** Tạo vessel map, làm nền cho phân tích mật độ và độ ngoằn ngoèo mạch máu, hỗ trợ module artery/vein.

**Không dùng DRIVE để:** phân loại DR hoặc phân đoạn đầy đủ các tổn thương võng mạc.

### 2.4 STARE — Vessel Segmentation / External Validation

- **Vai trò:** Dataset bổ sung để kiểm tra model mạch máu có tổng quát hóa sang nguồn ảnh khác DRIVE hay không.
- **Task:** Vessel segmentation trên các ảnh có annotation phù hợp.
- **Model:** Sử dụng U-Net đã train trên DRIVE để external evaluation; chưa cần train model riêng.
- **Metrics:** Dice, IoU, sensitivity, specificity.
- **Khuyến nghị:** Giữ STARE làm external validation ở giai đoạn đầu. Chỉ bổ sung vào training sau khi đã kiểm tra và chuẩn hóa annotation, cách mã hóa mask và vùng FOV.
- **Lưu ý:** Annotation giữa DRIVE và STARE có thể khác nhau. Báo cáo kết quả riêng theo dataset, không gộp metric thiếu giải thích.

### 2.5 RITE — Artery/Vein Classification

- **Vai trò:** Cung cấp nhãn phân biệt động mạch và tĩnh mạch.
- **Task:** Phân loại pixel/đoạn mạch thành artery, vein và background, tùy format annotation.
- **Input:** Ảnh fundus; có thể dùng vessel mask từ M2 để giới hạn vùng dự đoán.
- **Output:** Mask nhiều lớp artery/vein/background.
- **Model:** U-Net đa lớp (3 classes); lựa chọn khác là patch classifier ResNet18/ResNet34.
- **Loss:** Weighted Cross-Entropy kết hợp Dice để xử lý mất cân bằng lớp.
- **Metrics:** Macro-F1, precision/recall từng lớp, artery-vein accuracy trên vùng mạch có nhãn.
- **Ứng dụng:** Phân tích riêng động mạch và tĩnh mạch, hỗ trợ các phép đo A/V.
- **Lưu ý:** Kiểm tra format annotation, mã hóa lớp và split trước khi viết Dataset loader. RITE không phải dataset phân loại đa bệnh.

### 2.6 IDRiD — DR Lesion Segmentation và DR/DME Classification

- **Vai trò:** Dataset trọng tâm cho các tổn thương liên quan diabetic retinopathy (DR).
- **Task A — Segmentation:** Phân đoạn các loại tổn thương có annotation như microaneurysms (MA), haemorrhages (HE), hard exudates (EX), soft exudates (SE).
- **Task B — Classification:** Phân loại cấp ảnh theo DR severity và/hoặc DME theo nhãn có sẵn.
- **Segmentation model:** U-Net hoặc Attention U-Net, encoder ResNet34/EfficientNet-B0.
- **Classification model:** EfficientNet-B0, train riêng với nhãn cấp ảnh.
- **Segmentation loss:** BCE + Dice; có thể thử Focal/Tversky do tổn thương nhỏ và mất cân bằng.
- **Metrics:** Dice, IoU, sensitivity theo từng tổn thương; classification dùng QWK, macro-F1, balanced accuracy.
- **Ứng dụng:** Tạo lesion maps, thống kê tổn thương khi mask và hậu xử lý đủ tin cậy, hỗ trợ DR grading.

**Cảnh báo quan trọng:** Không phải mọi ảnh IDRiD đều có lesion masks. Chỉ ảnh có mask tương ứng mới được dùng cho segmentation. Ảnh chỉ có nhãn cấp ảnh có thể dùng cho classification nhưng không được tự gán mask rỗng.

### 2.7 APTOS 2019 — DR Severity Grading

- **Vai trò:** Dataset phân loại mức độ DR từ toàn ảnh.
- **Task:** 5-class image classification:
  - 0: No DR
  - 1: Mild
  - 2: Moderate
  - 3: Severe
  - 4: Proliferative DR
- **Input:** Ảnh fundus.
- **Output:** Một trong năm mức độ DR.
- **Model:** EfficientNet-B0; có thể so sánh ResNet50.
- **Loss:** Cross-Entropy có class weights; có thể nghiên cứu ordinal-aware loss.
- **Metrics:** Quadratic Weighted Kappa (QWK), macro-F1, balanced accuracy, confusion matrix.
- **Không dùng APTOS để:** train pixel-level lesion segmentation vì dataset không cung cấp lesion masks tương ứng.

**Kết hợp với IDRiD:** Chỉ khi mapping nhãn hợp lệ; kiểm tra khác biệt nguồn ảnh và phân phối nhãn; báo cáo kết quả riêng theo dataset. Không trộn mù quáng hai dataset.

### 2.8 ODIR-5K — Multi-label / Multi-disease Classification

- **Vai trò:** Dataset cho phân loại nhiều bệnh/đặc điểm từ ảnh fundus.
- **Task:** Multi-label image classification.
- **Input:** Ảnh fundus, có thể kèm thông tin hai mắt/bệnh nhân.
- **Output:** Vector nhiều nhãn theo schema chính thức ODIR-5K.
- **Model:** DenseNet121 hoặc EfficientNet-B0 với sigmoid output cho từng nhãn.
- **Loss:** Binary Cross-Entropy; cân nhắc class weights hoặc focal loss.
- **Metrics:** Macro/micro-F1, per-class AUC, sensitivity, specificity, PR-AUC.
- **Ứng dụng:** Bổ sung module sàng lọc đa bệnh theo đúng nhãn dataset.
- **Lưu ý:** Đây không phải DR 5-grade classification và không cung cấp lesion masks pixel-level. Nếu có hai ảnh mắt của cùng bệnh nhân, phải chia train/validation/test theo patient ID để tránh leakage.

## 3. Bảng quyết định nhanh

| Dataset | Task | Model | Vai trò |
|---|---|---|---|
| Drishti-GS1 | OD/OC segmentation | U-Net / Attention U-Net | Dataset chính |
| Drishti-GS1 | CDR evaluation | Mask-based measurement | CDR reference |
| PAPILA | OD/OC + glaucoma evaluation | Model đã train trên Drishti-GS1 | External Validation / bổ sung |
| DRIVE | Vessel segmentation | U-Net binary | Dataset chính |
| STARE | Vessel segmentation | Model đã train trên DRIVE | External Validation |
| RITE | Artery/vein segmentation | U-Net 3-class | Phân biệt động mạch/tĩnh mạch |
| IDRiD | Lesion segmentation | U-Net / Attention U-Net | Phân đoạn MA, HE, EX, SE có annotation |
| IDRiD | DR/DME classification | EfficientNet-B0 | Phân loại nhãn cấp ảnh |
| APTOS 2019 | 5-grade DR classification | EfficientNet-B0 | Dataset DR grading |
| ODIR-5K | Multi-label classification | DenseNet121 / EfficientNet-B0 | Phân loại đa bệnh |

## 4. Thứ tự triển khai/huấn luyện

### Phase 0 — Chuẩn bị dữ liệu

1. Kiểm kê dataset đã có, dataset chưa tải và dataset chưa xác minh annotation.
2. Đọc hướng dẫn chính thức, license và điều khoản sử dụng.
3. Kiểm kê số ảnh, định dạng, độ phân giải, nhãn và mask.
4. Kiểm tra ảnh trùng, ảnh lỗi và ảnh hai mắt cùng bệnh nhân.
5. Tạo manifest CSV: `image_path`, `patient_id` (nếu có), `dataset`, `split`, `label_path`, `task`.
6. Chuẩn hóa mã hóa mask và tên lớp theo từng task.
7. Không tạo mask rỗng cho ảnh không có annotation.
8. Lưu preprocessing config riêng cho từng dataset.

### Phase 1 — Optic Disc/Cup Segmentation

**Bước 1: Drishti-GS1 → U-Net**

- Kiểm tra toàn bộ cấu trúc dataset và annotation.
- Xác minh OD/Cup SoftMap, AvgBoundary, Disk Center và CDR.
- Tạo script dataset validation và báo cáo các annotation bị thiếu.
- Xây dựng preprocessing và DataLoader.
- Train U-Net baseline cho OD/OC segmentation.
- So sánh binary-label và soft-label supervision nếu phù hợp.
- Đánh giá Dice/IoU riêng cho OD và OC.
- Tính vCDR từ predicted masks và so sánh với CDR reference.
- Giữ official test set để đánh giá cuối nếu có ground truth và quyền truy cập phù hợp.

**Bước 2: PAPILA → External Validation**

- Sử dụng model đã train và chọn trên Drishti-GS1.
- Không fine-tune trước lần external evaluation chính thức.
- Đánh giá OD/OC segmentation và glaucoma classification nếu annotation tương ứng có sẵn.
- Báo cáo kết quả riêng cho PAPILA.
- Kiểm tra domain shift và khác biệt annotation.

### Phase 2 — Vessel Segmentation và Artery/Vein

**Bước 3: DRIVE → U-Net**

- Train vessel segmentation.
- Chọn model bằng validation từ training set.
- Đánh giá cuối trên official test.
- Sau đó đánh giá external trên STARE.

**Bước 4: RITE → U-Net đa lớp**

- Train artery/vein/background.
- Có thể dùng vessel mask từ DRIVE để hỗ trợ giới hạn vùng mạch.
- Báo cáo metric riêng cho artery và vein.

### Phase 3 — DR Lesion và DR Grading

**Bước 5: IDRiD → U-Net / Attention U-Net**

- Chỉ dùng ảnh có lesion masks để train segmentation.
- Train multi-class hoặc multi-label tùy format annotation thực tế.
- Đánh giá riêng từng loại tổn thương.

**Bước 6: APTOS + IDRiD → EfficientNet-B0**

- Train DR grading nếu nhãn đã được harmonize.
- Có thể dùng sampling có kiểm soát hoặc domain-aware training.
- Báo cáo kết quả riêng theo dataset để nhận diện domain shift.
- Không sử dụng test set để điều chỉnh hyperparameter.

### Phase 4 — Multi-disease Classification

**Bước 7: ODIR-5K → DenseNet121/EfficientNet-B0**

- Train multi-label classifier.
- Dùng sigmoid cho từng nhãn.
- Chọn threshold từng lớp trên validation.
- Đánh giá riêng từng bệnh/nhãn.

### Phase 5 — Image Quality Assessment

**Bước 8: Xây dựng module đánh giá chất lượng ảnh**

- Xác định các tiêu chí ảnh không đạt chất lượng: blur, overexposure, underexposure, poor illumination và vùng FOV không phù hợp.
- Xây dựng bộ dữ liệu có nhãn quality.
- Bắt đầu với OpenCV-based rules làm baseline.
- Có thể thử EfficientNet-B0 nhỏ khi đã có dữ liệu gán nhãn.
- Đánh giá khả năng từ chối ảnh không phù hợp trước khi đưa vào các module khác.

### Phase 6 — Tích hợp FUNDUS AI

1. Kiểm tra chất lượng ảnh.
2. Optic disc/cup segmentation (Drishti-GS1).
3. Vessel segmentation (DRIVE).
4. Artery/vein classification (RITE).
5. Lesion segmentation (IDRiD).
6. DR grading (APTOS và/hoặc IDRiD).
7. Multi-label disease classification (ODIR-5K).
8. Tổng hợp kết quả, confidence, giới hạn và ảnh overlay.

Không nên chạy hoặc đưa ra kết luận từ module nếu ảnh đầu vào không đạt chất lượng hoặc module không hỗ trợ loại ảnh đó.

## 5. Nguyên tắc train/validation/test

1. Nếu dataset có nhiều ảnh cùng bệnh nhân, chia theo patient ID, không chia ngẫu nhiên từng ảnh.
2. Giữ official test/challenge split nếu dataset cung cấp.
3. Validation dùng chọn model, threshold và hyperparameter; test chỉ đánh giá cuối.
4. External validation dùng nguồn khác, ví dụ train DRIVE rồi đánh giá STARE hoặc train Drishti-GS1 rồi đánh giá PAPILA.
5. Không fine-tune trên external test rồi tiếp tục gọi đó là external test độc lập.
6. Với dataset nhỏ, cân nhắc cross-validation ở cấp bệnh nhân.
7. Lưu seed, danh sách file từng split và phiên bản dataset.
8. Tạo manifest riêng cho từng task vì một ảnh có thể có classification label nhưng không có segmentation mask.
9. Báo cáo metric riêng theo dataset và protocol; không gộp kết quả nếu annotation hoặc cách đánh giá không tương thích.

## 6. Baseline model nên bắt đầu

| Thứ tự | Task | Baseline | Có thể nâng cấp |
|---|---|---|---|
| 1 | Optic disc/cup | U-Net | Attention U-Net |
| 2 | Vessel segmentation | U-Net + ResNet34 encoder | Attention U-Net |
| 3 | Artery/vein | U-Net 3-class | Patch classifier / kiến trúc chuyên biệt |
| 4 | Lesion segmentation | U-Net | Attention U-Net + Focal/Tversky |
| 5 | DR grading | EfficientNet-B0 | EfficientNet-B2 / ordinal model |
| 6 | Multi-disease | DenseNet121 | EfficientNet-B0/B2 |
| 7 | Image quality | OpenCV rules | EfficientNet-B0 nhỏ |

Nên bắt đầu bằng baseline dễ debug, chưa cần Transformer hoặc kiến trúc quá phức tạp.

## 7. Cấu trúc thư mục đề xuất

```text
FUNDUS_AI/
├── datasets/
│   ├── Drishti-GS1/
│   ├── PAPILA/
│   ├── DRIVE/
│   ├── STARE/
│   ├── RITE/
│   ├── IDRiD/
│   ├── APTOS_2019/
│   └── ODIR_5K/
├── manifests/
│   ├── optic_disc_cup/
│   ├── vessel/
│   ├── artery_vein/
│   ├── dr_lesions/
│   ├── dr_grading/
│   ├── multi_disease/
│   └── image_quality/
├── configs/
├── src/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── preprocessing/
│   └── inference/
├── checkpoints/
└── outputs/
    ├── masks/
    ├── overlays/
    ├── metrics/
    └── reports/
```

## 8. Phần cứng và cách train

Với laptop CPU/GPU tích hợp:

- Dùng laptop để kiểm tra dữ liệu, preprocessing, tạo manifest và chạy inference nhẹ.
- Dùng Google Colab hoặc Kaggle Notebook có GPU để train.
- Resize ảnh theo từng task; tránh resize quá nhỏ làm mất tổn thương li ti.
- Bắt đầu batch size nhỏ (ví dụ 4–8 cho segmentation, tùy VRAM).
- Chạy thử vài batch để xác nhận image-mask alignment và loss trước khi train toàn bộ.
- Lưu checkpoint định kỳ và log loss/metric.
- Không lưu nhiều bản dataset đã resize nếu dung lượng SSD hạn chế; có thể xử lý và lưu trên cloud storage.

## 9. Checklist trước khi train

- [ ] Đã xác định task và loại annotation.
- [ ] Đã đọc hướng dẫn chính thức và license/terms of use.
- [ ] Đã kiểm tra ảnh-mask khớp kích thước/orientation.
- [ ] Đã kiểm tra patient-level leakage.
- [ ] Đã giữ official test split.
- [ ] Đã tạo manifest riêng theo task.
- [ ] Không gán mask rỗng cho ảnh thiếu annotation.
- [ ] Đã chạy thử một batch qua model và loss.
- [ ] Đã lưu config, seed và phiên bản dataset.
- [ ] Đã chọn metric phù hợp với class imbalance.
- [ ] Đã xác minh SoftMap encoding và threshold nếu sử dụng Drishti-GS1.
- [ ] Đã xác minh quy ước tọa độ nếu sử dụng AvgBoundary và Disk Center.
- [ ] Đã xác định rõ external validation set và không sử dụng nó để tuning.

## 10. Kết luận

- **Drishti-GS1:** Dataset chính để train OD/OC segmentation và đánh giá CDR.
- **PAPILA:** External Validation cho OD/OC segmentation và dataset bổ sung cho glaucoma classification theo nhãn có sẵn.
- **DRIVE:** U-Net để phân đoạn mạch máu.
- **STARE:** External Validation cho vessel segmentation.
- **RITE:** U-Net đa lớp để phân biệt artery/vein.
- **IDRiD:** U-Net/Attention U-Net để phân đoạn tổn thương DR; EfficientNet-B0 riêng cho DR/DME image-level classification.
- **APTOS 2019:** EfficientNet-B0 để phân loại 5 mức độ DR.
- **ODIR-5K:** DenseNet121/EfficientNet-B0 cho multi-label disease classification.
- **Image Quality Assessment:** Xây dựng dữ liệu gán nhãn riêng; bắt đầu bằng OpenCV rules.

**Chiến lược khuyến nghị:**

Bắt đầu với **Drishti-GS1 → U-Net → OD/OC segmentation → CDR evaluation**. Sau khi hoàn thiện pipeline và có baseline đáng tin cậy, dùng PAPILA để External Validation.

Tiếp theo triển khai **DRIVE → U-Net → vessel segmentation**, sau đó STARE external evaluation. Tiếp tục với RITE, IDRiD, APTOS và ODIR-5K.