# HAM10000 Skin Cancer Classification

Production-style computer vision project for **7-class skin lesion classification** on HAM10000.

Main stack:

- PyTorch + timm
- EfficientNet-B0 / EfficientNet-B3
- Focal Loss / Cross Entropy
- Class balancing with class weights and optional weighted sampler
- Parquet-based processed metadata
- MLflow experiment tracking
- Grad-CAM explainability
- FastAPI serving
- Optional Ray Tune and Airflow orchestration templates

> This project is for research and engineering practice. It is not a medical device and must not be used for diagnosis without clinical validation.

---

## Project structure

```text
skin-cancer-classification/
├── configs/
│   └── train_config.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/skin_cancer/
│   ├── core/
│   │   ├── config.py
│   │   └── utils.py
│   ├── data/
│   │   ├── validation.py
│   │   ├── preparation.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── modeling/
│   │   ├── model.py
│   │   └── losses.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── train.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate.py
│   ├── inference/
│   │   └── predict.py
│   └── explainability/
│       └── gradcam.py
├── mlops/
│   ├── ray/tune.py
│   └── airflow/retrain_dag.py
├── serving/
│   ├── app.py
│   └── schemas.py
├── tests/
├── models/
├── reports/
├── requirements.txt
├── requirements-serving.txt
├── requirements-mlops.txt
├── requirements-dev.txt
├── pyproject.toml
├── Dockerfile
├── Makefile
└── README.md
```

---

## Dataset layout

Put HAM10000 files here:

```text
data/raw/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
│   ├── ISIC_0024306.jpg
│   └── ...
└── HAM10000_images_part_2/
    ├── ISIC_0034318.jpg
    └── ...
```

The code searches images recursively under `data/raw`, so the exact image folder names can be changed as long as image filenames match `image_id` in the metadata CSV.

---

## Install core training environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then install:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` intentionally does **not** include Ray or Airflow. The core training pipeline does not need them.

---

## Validate, prepare, train, evaluate

```bash
make validate
make prepare
make train
make evaluate
```

Direct commands:

```bash
python -m skin_cancer.data.validation --config configs/train_config.yaml
python -m skin_cancer.data.preparation --config configs/train_config.yaml
python -m skin_cancer.training.train --config configs/train_config.yaml --use-weighted-sampler
python -m skin_cancer.evaluation.evaluate --config configs/train_config.yaml
```

Processed data will be saved as:

```text
data/processed/train.parquet
data/processed/val.parquet
data/processed/test.parquet
```

The best checkpoint will be saved to:

```text
models/best_model.pth
```

---

## Predict one image

```bash
python -m skin_cancer.inference.predict --config configs/train_config.yaml --image path/to/image.jpg
```

Or with Make:

```bash
make predict IMAGE=path/to/image.jpg
```

---

## Grad-CAM

```bash
python -m skin_cancer.explainability.gradcam --config configs/train_config.yaml --image path/to/image.jpg
```

Or:

```bash
make gradcam IMAGE=path/to/image.jpg
```

Outputs go to `reports/figures/`.

---

## Switch EfficientNet-B0 to EfficientNet-B3

Edit `configs/train_config.yaml`:

```yaml
model:
  name: "efficientnet_b3"

data:
  image_size: 300

training:
  batch_size: 16
```

B0 is lighter and better for a first baseline. B3 is stronger but slower and uses more VRAM.

---

## MLflow

MLflow is enabled in `configs/train_config.yaml`:

```yaml
mlflow:
  enabled: true
  tracking_uri: "mlruns"
  experiment_name: "ham10000-efficientnet"
```

Open the UI:

```bash
mlflow ui --backend-store-uri mlruns
```

Then open the local URL shown by MLflow in your browser.

---

## Optional Ray Tune

Ray is optional and only needed for hyperparameter tuning.

Install optional MLOps dependencies:

```bash
pip install -r requirements-mlops.txt
```

Run tuning:

```bash
make tune
```

Or:

```bash
python mlops/ray/tune.py --config configs/train_config.yaml --num-samples 8
```

If Ray fails to install on Windows, use Python 3.10 or 3.11 in a fresh environment, or use WSL2/Linux. Normal training does not require Ray.

---

## FastAPI serving

Install serving dependencies:

```bash
pip install -r requirements-serving.txt
```

Run API:

```bash
make serve
```

Prediction endpoint:

```text
POST /predict
```

---

## Development

Install dev tools:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

Run tests/lint:

```bash
make test
make lint
```

---

## Recommended workflow

```text
Raw HAM10000 Data
→ Data Validation
→ Preprocessing
→ Parquet Storage
→ Dataset/DataLoader
→ Augmentation
→ Balanced Training
→ EfficientNet-B0/B3
→ Focal Loss
→ MLflow Tracking
→ Evaluation
→ Grad-CAM
→ Prediction API
→ Optional Ray Tune / Airflow
```
