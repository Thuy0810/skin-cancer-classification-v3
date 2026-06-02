# Source Overview - HAM10000 Skin Cancer Classification

File này là bản đồ đọc nhanh source code của dự án. Nếu muốn hiểu hoặc sửa một chức năng, bắt đầu từ các mục tương ứng bên dưới.

## 1. Dự án làm gì

Dự án huấn luyện mô hình phân loại tổn thương da HAM10000 thành 7 lớp:

- `akiec`
- `bcc`
- `bkl`
- `df`
- `mel`
- `nv`
- `vasc`

Stack chính:

- PyTorch + `timm` để build model như `efficientnet_b0`, `efficientnet_b3`, `densenet121`.
- Albumentations để resize/augment/normalize ảnh.
- Focal Loss hoặc Cross Entropy để train dữ liệu mất cân bằng.
- MLflow để log experiment.
- FastAPI để serve endpoint predict.
- Airflow và Ray Tune là phần MLOps tùy chọn.

Luồng chuẩn:

```text
data/raw
-> validate raw data
-> clean metadata + split train/val/test parquet
-> Dataset/DataLoader + augmentation
-> build model + loss + optimizer + scheduler
-> train + save best checkpoint
-> evaluate + confusion matrix
-> predict single image / FastAPI / Grad-CAM
```

## 2. File cấu hình và lệnh chạy

### `configs/`

- `configs/train_config.yaml`: cấu hình mặc định cho toàn pipeline.
- `configs/eff_B0/*.yaml`: các cấu hình thử nghiệm EfficientNet-B0.
- `configs/densenet121/*.yaml`: các cấu hình thử nghiệm DenseNet121.

Các nhóm cấu hình quan trọng:

- `paths`: vị trí raw data, parquet, models, reports, checkpoint.
- `labels`: thứ tự nhãn và nhóm nhãn ác tính.
- `data`: image size, split ratio, dataloader workers, extension ảnh.
- `model`: tên model `timm`, pretrained, số class.
- `training`: epoch, batch size, mixed precision, metric theo dõi.
- `optimizer`, `scheduler`, `loss`: cách tối ưu và hàm loss.
- `mlflow`: tracking URI, experiment name, run name.
- `augmentation`: xác suất flip/rotate/brightness/shift-scale.
- `inference`: `top_k` cho predict.

### Entrypoint nhanh

- `Makefile`: gom lệnh thường dùng.
- `README.md`: hướng dẫn setup và chạy ở mức người dùng.
- `pyproject.toml`: package metadata, pytest path, ruff config.

Lệnh chính:

```bash
make validate
make prepare
make train
make evaluate
make predict IMAGE=path/to/image.jpg
make gradcam IMAGE=path/to/image.jpg
make serve
make tune
make test
```

## 3. Package chính: `src/skin_cancer/`

Đây là source chính của project. Khi cần sửa logic ML, gần như luôn bắt đầu ở đây.

### `src/skin_cancer/core/`

Các helper nền dùng chung.

- `config.py`
  - `load_config()`: đọc YAML thành `SimpleNamespace`, để code dùng kiểu `cfg.paths.raw_dir`.
  - `save_config()`: ghi config ra YAML.
  - `namespace_to_dict()` và `dict_to_namespace()`: chuyển qua lại giữa dict và namespace.
  - `config_arg_parser()`: parser chung có sẵn `--config`.

- `utils.py`
  - `set_seed()`: set seed cho Python, NumPy, PyTorch.
  - `get_device()`: chọn `cuda` nếu có GPU, nếu không dùng CPU.
  - `ensure_dir()`: tạo thư mục.
  - `save_json()` / `load_json()`: đọc ghi report JSON.
  - `save_checkpoint()` / `load_checkpoint()`: lưu và load model checkpoint.

### `src/skin_cancer/data/`

Phần này xử lý raw HAM10000 thành data dùng được cho training.

- `validation.py`
  - `build_image_index()`: scan `data/raw` để map `image_id -> path`.
  - `validate_metadata_columns()`: kiểm tra các cột bắt buộc trong CSV.
  - `validate_image_paths()`: kiểm tra metadata có file ảnh tương ứng hay không.
  - `validate_corrupted_images()`: dùng OpenCV thử đọc ảnh để phát hiện ảnh hỏng.
  - `validate_dataset()`: entrypoint chính cho `make validate`.
  - Output: `reports/metrics/data_validation_report.json`.

- `preparation.py`
  - `clean_metadata()`: chuẩn hóa `image_id`, `dx`, tuổi, giới tính, vị trí tổn thương.
  - `add_image_paths()`: thêm cột `image_path` từ raw image index.
  - `encode_labels()`: map nhãn string sang id theo `configs/*.yaml`.
  - `stratified_split()`: chia train/val/test có stratify theo label.
  - `prepare_data()`: entrypoint chính cho `make prepare`.
  - Output:
    - `data/interim/metadata_clean.parquet`
    - `data/processed/train.parquet`
    - `data/processed/val.parquet`
    - `data/processed/test.parquet`
    - `reports/metrics/split_report.json`

- `dataset.py`
  - `SkinLesionDataset`: PyTorch dataset đọc `image_path`, convert BGR sang RGB, apply transform, trả về `(image_tensor, label_tensor)`.
  - `load_split_dataframe()`: đọc một split parquet từ `data/processed`.

- `transforms.py`
  - `get_train_transforms()`: resize + augmentation + normalize ImageNet + tensor.
  - `get_valid_transforms()`: resize + normalize ImageNet + tensor, dùng cho validation/test/inference.

## 4. Model và loss

### `src/skin_cancer/modeling/`

- `model.py`
  - `build_model()`: gọi `timm.create_model(model_name, pretrained, num_classes)`.
  - `get_last_conv_layer()`: tìm Conv2d cuối cùng để dùng Grad-CAM.
  - `load_model_from_checkpoint()`: build model rồi load `model_state_dict`.

- `losses.py`
  - `FocalLoss`: focal loss multi-class.
  - `compute_class_weights()`: tính weight cân bằng theo công thức `N / (C * count_c)`.
  - `build_criterion()`: chọn `cross_entropy` hoặc `focal_loss`, có thể dùng `alpha: balanced`.

## 5. Training

### `src/skin_cancer/training/train.py`

Đây là entrypoint training cấp cao. CLI, Airflow và Ray Tune đều đi qua `run_training()`.

Các phần chính:

- `build_auto_run_name()`: tạo tên MLflow run tự động dựa theo số run đã có, model, loss, sampler.
- `normalize_stage_run_name()`: thêm prefix stage nếu cần.
- `build_optimizer()`: hỗ trợ `adamw` và `sgd`.
- `build_scheduler()`: hỗ trợ `cosine` hoặc `none`.
- `build_weighted_sampler()`: oversample class hiếm bằng `WeightedRandomSampler`.
- `_prepare_dataloaders()`: đọc parquet, tạo transform, dataset, dataloader.
- `_start_mlflow_run()`: bật MLflow, set experiment, log params.
- `run_training()`: orchestrate toàn bộ training:
  - load config nếu input là dict.
  - set seed và device.
  - tạo dataloader.
  - build model/loss/optimizer/scheduler.
  - gọi `train_model()` trong `trainer.py`.
  - lưu `reports/metrics/train_history.json`.
  - log metric và artifact lên MLflow nếu bật.
- `main()`: CLI cho `python -m skin_cancer.training.train`.

Lưu ý khi đọc source:

- `early_stopping_patience` đang được truyền vào `train_model()` nhưng trong `trainer.py` hiện chưa có logic dừng sớm thật sự.
- Trong `_start_mlflow_run()` có một đoạn code sau `return mlflow`; đoạn đó không chạy được và có vẻ là code cũ còn sót lại.

### `src/skin_cancer/training/trainer.py`

Đây là vòng lặp train/valid theo epoch.

- `EpochOutput`: dataclass chứa loss, metrics, y_true, y_pred, y_prob.
- `run_one_epoch()`:
  - Nếu có optimizer thì train, nếu không thì validate.
  - Hỗ trợ mixed precision khi device là CUDA.
  - Tính softmax, prediction, gom batch output.
  - Gọi `compute_classification_metrics()`.
- `train_model()`:
  - Lặp qua epoch.
  - Train một epoch, validate một epoch.
  - Step scheduler.
  - In metric ra terminal.
  - Log metric vào MLflow nếu có.
  - Lưu checkpoint khi metric theo dõi tốt hơn.
  - Lưu validation metrics và confusion matrix của best validation output.

### `src/skin_cancer/training/run_experiments.py`

Chạy nhiều experiment tuần tự từ một base config.

- `build_experiment_grid()`: định nghĩa các cấu hình B0/B3 batch size/lr.
- `apply_experiment_config()`: clone config, đổi model/image size/batch size/lr, đổi output folder.
- `main()`: chạy từng experiment bằng `run_training()`.

## 6. Evaluation và metrics

### `src/skin_cancer/evaluation/metrics.py`

Module tính metric và vẽ confusion matrix.

- `compute_classification_metrics()`:
  - accuracy
  - macro/weighted precision, recall, f1
  - classification report
  - confusion matrix thường và normalized
  - macro AUC OVR nếu có xác suất đủ hợp lệ
  - metric riêng theo class như `recall_mel`, `f1_mel`
- `save_confusion_matrix_artifacts()`: lưu cả matrix thường và normalized.
- `plot_confusion_matrix()`: vẽ PNG bằng matplotlib.

### `src/skin_cancer/evaluation/evaluate.py`

Entry point cho `make evaluate`.

- Load checkpoint từ `cfg.paths.best_model_path` hoặc `--checkpoint`.
- Load test split từ `data/processed/test.parquet`.
- Predict toàn test loader.
- Lưu:
  - `reports/metrics/test_metrics.json`
  - `reports/figures/confusion_matrix.png`
  - `reports/figures/confusion_matrix_normalized.png`
- Nếu MLflow bật, tạo run stage `evaluate_*` và log metric.

## 7. Inference, API và Grad-CAM

### `src/skin_cancer/inference/predict.py`

Predict một ảnh đơn.

- `load_image_tensor()`: đọc ảnh bằng OpenCV, convert RGB, apply valid transform.
- `predict_image()`:
  - Load config và checkpoint.
  - Load model.
  - Chạy softmax.
  - Trả về top-k prediction gồm `class_id`, `class_name`, `confidence`.
- `main()`: CLI cho `make predict`.

### `serving/`

FastAPI layer rất mỏng, dùng lại helper inference.

- `serving/app.py`
  - Env `MODEL_CONFIG`, mặc định `configs/train_config.yaml`.
  - Env `MODEL_CHECKPOINT`, mặc định `models/best_model.pth`.
  - `GET /health`: trả `{"status": "ok"}`.
  - `POST /predict`: nhận file upload, lưu file tạm, gọi `predict_image()`, xóa file tạm.

- `serving/schemas.py`
  - `PredictionItem`: shape của từng prediction.
  - `PredictionResponse`: list prediction trả về API.

### `src/skin_cancer/explainability/gradcam.py`

Sinh heatmap giải thích vùng ảnh model chú ý.

- `GradCAM`: đăng ký forward/backward hook trên Conv2d cuối.
- `overlay_cam()`: overlay heatmap lên ảnh gốc.
- `generate_gradcam()`:
  - Load checkpoint/model.
  - Tìm last conv layer.
  - Tạo CAM cho class dự đoán hoặc `--target-class`.
  - Lưu ảnh output, mặc định `reports/figures/gradcam.png`.

## 8. MLOps: Airflow, Ray, Docker, MLflow

### `dags/skin_cancer_pipeline.py`

DAG đang dùng cho docker-compose Airflow local.

- DAG id: `skin_cancer_training_pipeline`.
- `PROJECT_DIR = /opt/airflow/project`.
- Base command set `PYTHONPATH=/opt/airflow/project/src`.
- Luồng task:

```text
start
-> validate_data
-> prepare_data
-> train_<experiment>
-> finish
```

Hiện `EXPERIMENTS` đang bật một experiment:

- `densenet121_bs4_gamma2`
- config: `/opt/airflow/project/configs/densenet121/densenet121_bs4_gamma2.yaml`
- `use_sampler: False`

### `mlops/airflow/retrain_dag.py`

Template DAG retrain theo lịch tuần.

- DAG id: `ham10000_retrain_pipeline`.
- Luồng: `validate_data -> prepare_data -> train_model -> evaluate_model`.
- File này cần chỉnh `PROJECT_DIR` nếu copy sang Airflow ngoài docker-compose của repo.

### `mlops/ray/tune.py`

Ray Tune tùy chọn để search hyperparameter.

- `check_ray_installed()`: báo lỗi hướng dẫn cài nếu chưa có Ray.
- `trainable()`: build config trial rồi gọi lại `run_training()`.
- Search space gồm model, learning rate, batch size, epochs, focal gamma.
- MLflow bị tắt trong từng trial Ray.

### `docker-compose.yaml`

Dùng để chạy Airflow local.

- `airflow-webserver`: expose host port `8081 -> 8080`.
- `airflow-scheduler`: có khai báo GPU NVIDIA.
- Mount repo vào `/opt/airflow/project`.
- SQLite Airflow DB nằm ở `airflow.db` trong repo.

### `Dockerfile`

Image app/API cơ bản:

- Python 3.11 slim.
- Cài `requirements.txt`.
- Set `PYTHONPATH=/app/src`.
- CMD chạy `uvicorn serving.app:app --host 0.0.0.0 --port 8000`.

### `Dockerfile.airflow`

Image Airflow:

- Base `apache/airflow:2.10.5-python3.10`.
- Cài `requirements.txt`.

### `mlruns/`

MLflow local tracking store. Config hiện dùng `tracking_uri: "mlruns"` trong YAML, nên artifact/log được ghi vào thư mục này khi chạy local hoặc trong container với mount phù hợp.

## 9. Artifact và thư mục dữ liệu

- `data/raw/`: đặt HAM10000 raw metadata CSV và ảnh.
- `data/interim/`: metadata đã clean tạm thời.
- `data/processed/`: parquet train/val/test sau prepare.
- `models/`: checkpoint model, ví dụ `best_model.pth` hoặc checkpoint theo experiment.
- `reports/metrics/`: JSON metric/report.
- `reports/figures/`: confusion matrix, Grad-CAM, hình minh họa.
- `airflow_logs/`: log Airflow local.
- `airflow.db`: SQLite metadata DB của Airflow local.
- `mlruns/`: MLflow experiment store.

## 10. Test

### `tests/`

- `test_dataset.py`: kiểm tra dataset đọc ảnh và trả tensor đúng shape.
- `test_model.py`: kiểm tra model `efficientnet_b0` forward ra shape `(batch, 7)`.
- `test_transforms.py`: kiểm tra valid transform resize về `(3, 224, 224)`.

Chạy:

```bash
make test
```

## 11. Nên mở file nào khi muốn sửa từng việc

- Sửa đường dẫn data, model, report, hyperparameter: `configs/*.yaml`.
- Sửa validate raw CSV/ảnh: `src/skin_cancer/data/validation.py`.
- Sửa clean metadata hoặc split train/val/test: `src/skin_cancer/data/preparation.py`.
- Sửa cách đọc ảnh: `src/skin_cancer/data/dataset.py`.
- Sửa augmentation/normalize: `src/skin_cancer/data/transforms.py`.
- Đổi architecture model hoặc cách load checkpoint: `src/skin_cancer/modeling/model.py`.
- Sửa focal loss/class weight: `src/skin_cancer/modeling/losses.py`.
- Sửa optimizer/scheduler/sampler/MLflow setup: `src/skin_cancer/training/train.py`.
- Sửa vòng lặp epoch, checkpoint best model, validation artifact: `src/skin_cancer/training/trainer.py`.
- Sửa metric hoặc confusion matrix: `src/skin_cancer/evaluation/metrics.py`.
- Sửa evaluate test set: `src/skin_cancer/evaluation/evaluate.py`.
- Sửa predict một ảnh: `src/skin_cancer/inference/predict.py`.
- Sửa API upload/predict: `serving/app.py` và `serving/schemas.py`.
- Sửa Grad-CAM: `src/skin_cancer/explainability/gradcam.py`.
- Sửa Airflow local pipeline đang chạy trong docker-compose: `dags/skin_cancer_pipeline.py`.
- Sửa template retrain Airflow ngoài repo: `mlops/airflow/retrain_dag.py`.
- Sửa hyperparameter search: `mlops/ray/tune.py`.
- Sửa container API: `Dockerfile`.
- Sửa container Airflow: `Dockerfile.airflow` và `docker-compose.yaml`.

## 12. Ghi chú runtime hiện tại

- Airflow local của repo được cấu hình expose ở `http://localhost:8081`.
- FastAPI mặc định chạy ở `http://localhost:8000` nếu dùng `make serve`.
- MLflow config mặc định dùng local folder `mlruns`; nếu chạy trong Docker/Airflow cần để ý đường dẫn artifact phải hợp lệ trong container.
- `dags/skin_cancer_pipeline.py` là DAG phù hợp nhất với setup docker-compose hiện tại.
- `mlops/airflow/retrain_dag.py` giống template độc lập hơn, không phải file DAG chính được mount bởi `docker-compose.yaml` trừ khi bạn tự copy hoặc cấu hình thêm.
