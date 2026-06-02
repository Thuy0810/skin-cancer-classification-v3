# Ghi chú học và báo cáo: HAM10000 Skin Cancer Classification

## 1. Tổng quan bài toán

Đề tài xây dựng mô hình học sâu để phân loại ảnh tổn thương da trong bộ dữ liệu HAM10000 thành 7 lớp bệnh. Dữ liệu thực tế đã được prepare trong project có 10015 ảnh, được chia thành train, validation và test theo tỉ lệ xấp xỉ 70/15/15.

Các lớp được dùng trong code:

| ID | Nhãn | Ý nghĩa thường dùng trong HAM10000 |
|---:|---|---|
| 0 | `akiec` | Actinic keratoses / intraepithelial carcinoma |
| 1 | `bcc` | Basal cell carcinoma |
| 2 | `bkl` | Benign keratosis-like lesions |
| 3 | `df` | Dermatofibroma |
| 4 | `mel` | Melanoma |
| 5 | `nv` | Melanocytic nevi |
| 6 | `vasc` | Vascular lesions |

Vị trí trong code:

| Nội dung | File code |
|---|---|
| Danh sách 7 nhãn | `configs/train_config.yaml:13-15` |
| Encode nhãn từ chuỗi sang số `0..6` | `src/skin_cancer/data/preparation.py:42-50` |
| Dataset trả về `(image_tensor, label_tensor)` | `src/skin_cancer/data/dataset.py:25-40` |

## 2. Chia dữ liệu train/validation/test

Project dùng chia dữ liệu có stratify theo nhãn, nghĩa là tỉ lệ từng lớp được giữ tương đối giống nhau ở train, validation và test.

Tỉ lệ cấu hình:

| Split | Tỉ lệ | Số ảnh thực tế |
|---|---:|---:|
| Train | 70% | 7009 |
| Validation | 15% | 1503 |
| Test | 15% | 1503 |
| Tổng | 100% | 10015 |

Phân bố lớp sau khi chia:

| Nhãn | Train | Validation | Test |
|---|---:|---:|---:|
| `nv` | 4693 | 1006 | 1006 |
| `mel` | 779 | 167 | 167 |
| `bkl` | 769 | 165 | 165 |
| `bcc` | 360 | 77 | 77 |
| `akiec` | 229 | 49 | 49 |
| `vasc` | 99 | 21 | 22 |
| `df` | 80 | 18 | 17 |

Cách code thực hiện:

| Nội dung | File code |
|---|---|
| `test_size: 0.15`, `val_size: 0.15` | `configs/train_config.yaml:17-20` |
| Tách test trước bằng `train_test_split(..., stratify=df["label"])` | `src/skin_cancer/data/preparation.py:53-64` |
| Tính validation theo phần còn lại: `val_size / (1 - test_size)` | `src/skin_cancer/data/preparation.py:66` |
| Tách train/validation có stratify | `src/skin_cancer/data/preparation.py:67-73` |
| Lưu `train.parquet`, `val.parquet`, `test.parquet` | `src/skin_cancer/data/preparation.py:101-103` |
| Lưu báo cáo split | `src/skin_cancer/data/preparation.py:105-115`, `reports/metrics/split_report.json` |

Điểm nên nói khi báo cáo: vì dữ liệu mất cân bằng rất mạnh, lớp `nv` chiếm nhiều nhất, trong khi `df` và `vasc` rất ít. Do đó project dùng Focal Loss và có hỗ trợ WeightedRandomSampler để giảm ảnh hưởng của mất cân bằng lớp.

## 3. Tiền xử lý và augmentation ảnh

Ảnh được đọc bằng OpenCV, chuyển từ BGR sang RGB, resize về kích thước cấu hình, normalize theo ImageNet mean/std rồi chuyển thành tensor.

Train transform có augmentation:

- Resize về `image_size x image_size`.
- Horizontal flip, vertical flip.
- Rotate, shift-scale-rotate.
- Random brightness/contrast.
- Normalize theo ImageNet.
- `ToTensorV2`.

Validation/test/inference transform không augmentation, chỉ resize, normalize và chuyển tensor.

Vị trí trong code:

| Nội dung | File code |
|---|---|
| Đọc ảnh, BGR -> RGB | `src/skin_cancer/data/dataset.py:30-33` |
| Train augmentation | `src/skin_cancer/data/transforms.py:10-35` |
| Validation/test transform | `src/skin_cancer/data/transforms.py:38-45` |
| Dataloader dùng train/valid transform | `src/skin_cancer/training/train.py:103-117` |

## 4. Xử lý mất cân bằng dữ liệu

### 4.1. Focal Loss

Focal Loss được dùng để tập trung học các mẫu khó và giảm ảnh hưởng của các mẫu dễ thuộc lớp chiếm đa số. Với bài toán phân loại nhiều lớp, code tính Cross Entropy trước, sau đó nhân thêm hệ số focal:

```text
FL = (1 - p_t)^gamma * CE
```

Trong đó `p_t` là xác suất mô hình gán cho lớp đúng. Khi mẫu đã được dự đoán dễ và `p_t` cao, hệ số `(1 - p_t)^gamma` nhỏ, loss của mẫu đó giảm. Khi mẫu khó hoặc bị dự đoán sai, loss vẫn lớn hơn để mô hình tập trung học.

Trong project, Focal Loss có thể kết hợp `alpha: balanced`. Khi bật `alpha: balanced`, code tính class weight theo công thức:

```text
weight_c = N / (C * count_c)
```

Lớp càng ít ảnh thì weight càng cao.

Vị trí trong code:

| Nội dung | File code |
|---|---|
| Class `FocalLoss` | `src/skin_cancer/modeling/losses.py:8-37` |
| Công thức `CE`, `pt`, `((1 - pt) ** gamma) * CE` | `src/skin_cancer/modeling/losses.py:28-31` |
| Tính class weight cân bằng | `src/skin_cancer/modeling/losses.py:40-47` |
| Chọn `cross_entropy` hoặc `focal_loss` theo config | `src/skin_cancer/modeling/losses.py:50-66` |
| Training gọi `build_criterion(...)` | `src/skin_cancer/training/train.py:237-244` |
| Config mặc định dùng `focal_loss`, `gamma: 2.0`, `alpha: balanced` | `configs/train_config.yaml:47-50` |

### 4.2. WeightedRandomSampler

WeightedRandomSampler là cơ chế lấy mẫu lại ở train loader. Mỗi ảnh được gán trọng số nghịch đảo với số lượng ảnh của lớp đó:

```text
sample_weight = 1 / count_of_class
```

Vì vậy ảnh thuộc lớp hiếm có xác suất được lấy mẫu cao hơn. Sampler chỉ dùng cho train loader, không dùng cho validation/test.

Vị trí trong code:

| Nội dung | File code |
|---|---|
| Import `WeightedRandomSampler` | `src/skin_cancer/training/train.py:9` |
| Tính `class_counts` và `weights` | `src/skin_cancer/training/train.py:93-96` |
| Bật sampler khi `use_weighted_sampler=True` | `src/skin_cancer/training/train.py:119` |
| Train loader dùng `sampler=sampler`, nếu không có sampler thì mới `shuffle=True` | `src/skin_cancer/training/train.py:120-127` |
| CLI có cờ `--use-weighted-sampler` và `--no-weighted-sampler` | `src/skin_cancer/training/train.py:288-315` |
| Script train mặc định bật sampler | `scripts/run_train.sh:3` |
| Airflow experiment DenseNet121 hiện đang tắt sampler | `dags/skin_cancer_pipeline.py:18-23`, `dags/skin_cancer_pipeline.py:72-82` |

Điểm nên nói khi báo cáo: Focal Loss xử lý mất cân bằng ở mức hàm mất mát, còn WeightedRandomSampler xử lý ở mức cách lấy batch. Hai kỹ thuật này có thể dùng riêng hoặc kết hợp tùy thí nghiệm.

## 5. Mô hình học sâu CNN

Project build model bằng thư viện `timm`. Lớp phân loại cuối được `timm.create_model(..., num_classes=7)` tạo lại theo số lớp trong config.

Vị trí chung trong code:

| Nội dung | File code |
|---|---|
| Build model bằng `timm.create_model` | `src/skin_cancer/modeling/model.py:14-21` |
| Training gọi `build_model(...)` | `src/skin_cancer/training/train.py:230-235` |
| Load checkpoint để inference/evaluate | `src/skin_cancer/modeling/model.py:37-47` |

### 5.1. EfficientNet-B0

EfficientNet là họ mô hình CNN được thiết kế dựa trên ý tưởng compound scaling, tức là mở rộng đồng thời chiều sâu, chiều rộng và độ phân giải đầu vào theo một hệ số cân bằng. EfficientNet-B0 là phiên bản cơ sở, có kích thước tương đối nhỏ, tốc độ huấn luyện nhanh và phù hợp để làm mô hình baseline.

Trong project, EfficientNet-B0 nhận ảnh đầu vào `224 x 224`. Lớp phân loại cuối có 7 output tương ứng với 7 lớp bệnh. Mô hình được huấn luyện bằng Focal Loss hoặc Cross Entropy tùy config, optimizer AdamW và scheduler cosine.

Vị trí trong code/config:

| Nội dung | File code |
|---|---|
| Model name `efficientnet_b0` | `configs/train_config.yaml:25-28`, `configs/eff_B0/b0_bs16_gamma1.yaml:25-28` |
| Input size `224` | `configs/train_config.yaml:17-20`, `configs/eff_B0/b0_bs16_gamma1.yaml:17-20` |
| Optimizer AdamW | `configs/train_config.yaml:39-42`, `src/skin_cancer/training/train.py:62-69` |
| Scheduler cosine | `configs/train_config.yaml:44-45`, `src/skin_cancer/training/train.py:80-90` |
| Focal Loss config | `configs/train_config.yaml:47-50` |

### 5.2. DenseNet121

DenseNet là kiến trúc CNN có kết nối dày đặc giữa các lớp. Mỗi lớp trong một dense block nhận feature maps từ tất cả các lớp trước đó và truyền feature maps của nó cho các lớp sau. Cách kết nối này giúp tăng khả năng tái sử dụng đặc trưng, giảm vấn đề vanishing gradient và hỗ trợ huấn luyện các mạng sâu hơn.

Trong project, DenseNet121 nhận ảnh đầu vào `300 x 300`. Lớp phân loại cuối có 7 output. DenseNet121 được dùng làm mô hình so sánh với EfficientNet-B0.

Vị trí trong code/config:

| Nội dung | File code |
|---|---|
| Model name `densenet121` | `configs/densenet121/densenet121_bs4_gamma2.yaml:25-28` |
| Input size `300` | `configs/densenet121/densenet121_bs4_gamma2.yaml:17-20` |
| Batch size/epoch thí nghiệm | `configs/densenet121/densenet121_bs4_gamma2.yaml:30-37` |
| Focal Loss + `alpha: balanced` | `configs/densenet121/densenet121_bs4_gamma2.yaml:47-50` |
| Optimizer AdamW | `configs/densenet121/densenet121_bs4_gamma2.yaml:39-42`, `src/skin_cancer/training/train.py:62-69` |
| Scheduler cosine | `configs/densenet121/densenet121_bs4_gamma2.yaml:44-45`, `src/skin_cancer/training/train.py:80-90` |

### 5.3. Bảng so sánh hai mô hình

| Tiêu chí | EfficientNet-B0 | DenseNet121 |
|---|---|---|
| Kiến trúc | CNN với compound scaling | CNN với dense connections |
| Kích thước đầu vào trong thí nghiệm | `224 x 224` | `300 x 300` |
| Ưu điểm | Nhẹ, nhanh, hiệu quả, phù hợp baseline | Tái sử dụng đặc trưng tốt, phù hợp ảnh y tế |
| Hạn chế | Có thể kém hơn model lớn ở đặc trưng phức tạp | Train chậm hơn, tốn bộ nhớ hơn |
| Vai trò trong đề tài | Baseline chính | Mô hình so sánh |
| Config chính | `configs/eff_B0/b0_bs16_gamma1.yaml` | `configs/densenet121/densenet121_bs4_gamma2.yaml` |

## 6. Quy trình huấn luyện

Luồng training chính:

1. Load config.
2. Load `train.parquet` và `val.parquet`.
3. Tạo train/validation transform.
4. Tạo dataset và dataloader.
5. Build model theo `cfg.model.name`.
6. Build loss theo `cfg.loss.name`.
7. Build optimizer AdamW.
8. Build scheduler cosine.
9. Train từng epoch, validate từng epoch.
10. Lưu checkpoint tốt nhất theo `monitor_metric`, hiện là `macro_f1`.

Vị trí trong code:

| Nội dung | File code |
|---|---|
| Entry point training | `src/skin_cancer/training/train.py:198-285` |
| Chuẩn bị dataloader | `src/skin_cancer/training/train.py:99-135` |
| Build model/loss/optimizer/scheduler | `src/skin_cancer/training/train.py:230-246` |
| Vòng lặp train/valid theo epoch | `src/skin_cancer/training/trainer.py:113-177` |
| Tính softmax, prediction, metric mỗi epoch | `src/skin_cancer/training/trainer.py:52-85` |
| Lưu best checkpoint | `src/skin_cancer/training/trainer.py:163-177` |
| Lưu validation metrics và confusion matrix | `src/skin_cancer/training/trainer.py:179-196` |

Command thường dùng:

```bash
python -m skin_cancer.data.preparation --config configs/train_config.yaml
python -m skin_cancer.training.train --config configs/train_config.yaml --use-weighted-sampler
python -m skin_cancer.evaluation.evaluate --config configs/train_config.yaml
```

## 7. Đánh giá mô hình

Project đánh giá bằng các metric phân loại nhiều lớp:

- Accuracy.
- Macro precision, macro recall, macro F1.
- Weighted precision, weighted recall, weighted F1.
- Classification report theo từng lớp.
- Confusion matrix thường và normalized.
- Macro AUC OVR nếu tính được.

Vị trí trong code:

| Nội dung | File code |
|---|---|
| Tính accuracy, precision, recall, F1, AUC | `src/skin_cancer/evaluation/metrics.py:17-74` |
| Lưu confusion matrix | `src/skin_cancer/evaluation/metrics.py:77-140` |
| Evaluate trên test split | `src/skin_cancer/evaluation/evaluate.py:61-82` |
| Lưu `test_metrics.json` | `src/skin_cancer/evaluation/evaluate.py:71-82` |

Kết quả hiện có trong repo:

| Mô hình/run | Split | Accuracy | Macro F1 | Macro AUC OVR | File |
|---|---|---:|---:|---:|---|
| EfficientNet-B0 `b0_bs16_gamma1_lr5e5_no_sampler_wd5e4` | Test | 0.8743 | 0.8012 | 0.9801 | `reports/b0_bs16_gamma1_lr5e5_no_sampler_wd5e4/metrics/test_metrics.json` |
| DenseNet121 `densenet121_bs4_gamma2_no_sampler_wd5e4` | Validation | 0.7718 | 0.7807 | N/A | `reports/densenet121_bs4_gamma2_no_sampler_wd5e4/metrics/validation_metrics.json` |

Lưu ý khi báo cáo: bảng trên không nên so trực tiếp như cùng một test set nếu DenseNet121 chưa có `test_metrics.json`; hiện DenseNet121 trong folder này mới có validation metrics.

## 8. Inference dự đoán ảnh mới

Khi dự đoán một ảnh, project load checkpoint, build lại model, dùng valid transform, chạy forward, softmax và lấy top-k nhãn có xác suất cao nhất.

Vị trí trong code:

| Nội dung | File code |
|---|---|
| Load ảnh inference, resize/normalize | `src/skin_cancer/inference/predict.py:14-20` |
| Load config, checkpoint và model | `src/skin_cancer/inference/predict.py:30-38` |
| Forward, softmax, top-k | `src/skin_cancer/inference/predict.py:40-50` |
| CLI predict một ảnh | `src/skin_cancer/inference/predict.py:53-67` |
| Script predict 7 ảnh mẫu và xuất grid/csv | `scripts/predict_image.py:25-33`, `scripts/predict_image.py:131-216` |

Command ví dụ:

```bash
python -m skin_cancer.inference.predict --config configs/train_config.yaml --image path/to/image.jpg
python scripts/predict_image.py
```

## 9. Đoạn thuyết minh ngắn để nói khi báo cáo

Đề tài sử dụng bộ dữ liệu HAM10000 gồm 10015 ảnh tổn thương da thuộc 7 lớp. Dữ liệu được chia thành train, validation và test theo tỉ lệ 70/15/15, đồng thời dùng stratified split để giữ phân bố nhãn giữa các tập. Vì dữ liệu mất cân bằng mạnh, đặc biệt lớp `nv` nhiều hơn rất nhiều so với `df` và `vasc`, project sử dụng Focal Loss để tập trung vào mẫu khó và hỗ trợ WeightedRandomSampler để tăng xác suất lấy mẫu các lớp ít dữ liệu trong quá trình train.

Hai mô hình CNN được sử dụng là EfficientNet-B0 và DenseNet121. EfficientNet-B0 là baseline chính vì nhẹ, nhanh và hiệu quả với ảnh đầu vào 224 x 224. DenseNet121 được dùng làm mô hình so sánh với ảnh đầu vào 300 x 300, có ưu điểm tái sử dụng đặc trưng nhờ dense connections. Cả hai mô hình đều thay lớp phân loại cuối thành 7 output, huấn luyện bằng AdamW và scheduler cosine, theo dõi macro F1 để lưu checkpoint tốt nhất.

## 10. Lưu ý khi chốt báo cáo từ code

Khi báo cáo kết quả, nên dựa vào giá trị trong config và lệnh chạy thực tế, không chỉ dựa vào tên file hoặc tên folder.

- `configs/densenet121/densenet121_bs4_gamma2.yaml:47-50` hiện đặt `gamma: 1.0`, dù tên file có chữ `gamma2`.
- `dags/skin_cancer_pipeline.py:18-23` hiện đặt DenseNet121 `use_sampler: False`; khi chạy qua DAG này, sampler không được bật.
- `scripts/run_train.sh:3` chạy config mặc định và bật `--use-weighted-sampler`.
- Nếu cần báo cáo một thí nghiệm cụ thể là "có sampler" hay "gamma = 2", hãy kiểm tra lại đúng config/lệnh chạy trước khi lấy số liệu.

## 11. Tài liệu tham khảo gợi ý

- [2] Tan, M. and Le, Q. V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML, 2019.
- [3] Huang, G., Liu, Z., Van Der Maaten, L. and Weinberger, K. Q. Densely Connected Convolutional Networks. CVPR, 2017.
