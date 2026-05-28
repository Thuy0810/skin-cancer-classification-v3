#!/usr/bin/env bash
set -euo pipefail
python -m skin_cancer.training.train --config configs/train_config.yaml --use-weighted-sampler
