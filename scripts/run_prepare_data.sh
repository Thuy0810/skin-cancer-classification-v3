#!/usr/bin/env bash
set -euo pipefail
python -m skin_cancer.data.validation --config configs/train_config.yaml
python -m skin_cancer.data.preparation --config configs/train_config.yaml
