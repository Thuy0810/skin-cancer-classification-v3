#!/usr/bin/env bash
set -euo pipefail
python -m skin_cancer.evaluation.evaluate --config configs/train_config.yaml
