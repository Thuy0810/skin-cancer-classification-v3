# Architecture

This project is organized as a reusable Python package under `src/skin_cancer`.

## Layers

```text
core
  Configuration loading, utilities, checkpoints, JSON helpers.

data
  Dataset validation, metadata preparation, parquet split creation, PyTorch Dataset, augmentations.

modeling
  EfficientNet model factory and loss functions.

training
  Training loop, validation loop, checkpointing, reusable `run_training()` entrypoint.

evaluation
  Metrics, test-set evaluation, reports and confusion matrix output.

inference
  Single-image prediction utilities and CLI.

explainability
  Grad-CAM visualization.

serving
  FastAPI app for model inference.

mlops
  Optional Ray Tune and Airflow orchestration templates.
```

## Why Ray is optional

The normal training pipeline only needs PyTorch, timm, pandas, albumentations and MLflow. Ray is used only for hyperparameter tuning or distributed experiments, so it lives in `requirements-mlops.txt` instead of `requirements.txt`.

This keeps the core environment easy to install, especially on Windows where Ray support depends on Python version and available wheels.

## Training entrypoints

`src/skin_cancer/training/train.py` contains a reusable function:

```text
run_training(cfg, use_weighted_sampler=False, enable_mlflow=None)
```

This function is used by:

- the normal CLI command
- `mlops/ray/tune.py`
- orchestration systems such as Airflow

This is closer to a real-world project than placing all logic directly in a `main()` function.
