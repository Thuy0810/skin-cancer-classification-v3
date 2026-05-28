# Install troubleshooting

## `No matching distribution found for ray`

Ray is optional in this project. The core training pipeline does not require it.

Use the core install first:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Only install Ray when you need hyperparameter tuning:

```bash
pip install -r requirements-mlops.txt
```

If Ray fails on Windows, use one of these options:

1. Use WSL2 / Linux for Ray.
2. Use Python 3.10, 3.11, or 3.12 on 64-bit Windows.
3. Skip Ray and train normally with `python -m skin_cancer.training.train`.

To check your Python version:

```bash
python --version
python -c "import platform; print(platform.platform(), platform.machine())"
```
