from __future__ import annotations

import argparse
import copy
from pathlib import Path

from skin_cancer.core.config import load_config
from skin_cancer.training.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple training experiments automatically."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Base config path.",
    )
    sampler_group = parser.add_mutually_exclusive_group()
    sampler_group.add_argument(
        "--use-weighted-sampler",
        dest="use_weighted_sampler",
        action="store_true",
        help="Use WeightedRandomSampler for all experiments.",
    )
    sampler_group.add_argument(
        "--no-weighted-sampler",
        dest="use_weighted_sampler",
        action="store_false",
        help="Disable WeightedRandomSampler for all experiments.",
    )
    parser.set_defaults(use_weighted_sampler=None)
    return parser.parse_args()


def build_experiment_grid() -> list[dict]:
    return [
        {
            "run_name": "train_001_b0_bs16_sampler",
            "model_name": "efficientnet_b0",
            "image_size": 224,
            "batch_size": 16,
            "learning_rate": 1e-4,
        },
        {
            "run_name": "train_002_b0_bs32_sampler",
            "model_name": "efficientnet_b0",
            "image_size": 224,
            "batch_size": 32,
            "learning_rate": 1e-4,
        },
        {
            "run_name": "train_003_b3_bs8_sampler",
            "model_name": "efficientnet_b3",
            "image_size": 300,
            "batch_size": 8,
            "learning_rate": 3e-5,
        },
        {
            "run_name": "train_004_b3_bs16_sampler",
            "model_name": "efficientnet_b3",
            "image_size": 300,
            "batch_size": 16,
            "learning_rate": 3e-5,
        },
    ]


def apply_experiment_config(base_cfg, exp: dict):
    cfg = copy.deepcopy(base_cfg)

    cfg.model.name = exp["model_name"]
    cfg.data.image_size = exp["image_size"]
    cfg.training.batch_size = exp["batch_size"]
    cfg.optimizer.learning_rate = exp["learning_rate"]

    cfg.mlflow.run_name = exp["run_name"]

    model_dir = Path(cfg.paths.model_dir) / exp["run_name"]
    report_dir = Path(cfg.paths.report_dir) / exp["run_name"]

    cfg.paths.model_dir = str(model_dir)
    cfg.paths.best_model_path = str(model_dir / "best_model.pth")
    cfg.paths.report_dir = str(report_dir)

    return cfg


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    experiments = build_experiment_grid()

    for exp in experiments:
        print("=" * 80)
        print(f"Running experiment: {exp['run_name']}")
        print(
            f"model={exp['model_name']} | "
            f"image_size={exp['image_size']} | "
            f"batch_size={exp['batch_size']} | "
            f"lr={exp['learning_rate']}"
        )
        print("=" * 80)

        cfg = apply_experiment_config(base_cfg, exp)

        result = run_training(
            cfg=cfg,
            use_weighted_sampler=args.use_weighted_sampler,
            enable_mlflow=True,
        )

        print(f"Finished: {exp['run_name']}")
        print(f"Best epoch: {result.get('best_epoch')}")
        print(f"Best metric: {result.get('best_metric')}")


if __name__ == "__main__":
    main()