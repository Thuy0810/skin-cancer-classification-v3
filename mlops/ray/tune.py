from __future__ import annotations

"""Optional Ray Tune entrypoint for hyperparameter search.

Ray is not required for the normal training pipeline. Install optional MLOps
dependencies before running this file:

    pip install -r requirements-mlops.txt
"""

import argparse
import copy
from pathlib import Path
import sys
from typing import Any


def check_ray_installed():
    try:
        import ray  # noqa: F401
        from ray import tune
    except ImportError as exc:
        message = (
            "Ray is not installed.\n\n"
            "Ray is optional in this project and is only needed for hyperparameter "
            "tuning or distributed experiments.\n\n"
            "Install it with:\n\n"
            "    pip install -r requirements-mlops.txt\n\n"
            "If you are on Windows and installation fails, create a Python 3.10 "
            "or 3.11 environment first, then install Ray again."
        )
        raise RuntimeError(message) from exc
    return tune


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ray Tune for HAM10000 training.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--num-samples", type=int, default=8)
    return parser.parse_args()


def trainable(trial_config: dict[str, Any]) -> None:
    tune = check_ray_installed()

    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from skin_cancer.core.config import load_config, namespace_to_dict
    from skin_cancer.training.train import run_training

    base_cfg = namespace_to_dict(load_config(trial_config["base_config_path"]))
    cfg = copy.deepcopy(base_cfg)

    trial_name = tune.get_context().get_trial_name()
    cfg["model"]["name"] = trial_config["model_name"]
    cfg["optimizer"]["learning_rate"] = trial_config["learning_rate"]
    cfg["training"]["batch_size"] = trial_config["batch_size"]
    cfg["training"]["epochs"] = trial_config["epochs"]
    cfg["loss"]["gamma"] = trial_config["focal_gamma"]
    cfg["paths"]["best_model_path"] = str(Path("models") / f"{trial_name}_best.pth")
    cfg["paths"]["report_dir"] = str(Path("reports") / "ray" / trial_name)
    cfg["mlflow"]["enabled"] = False

    if cfg["model"]["name"] == "efficientnet_b3":
        cfg["data"]["image_size"] = 300
        cfg["training"]["batch_size"] = min(int(cfg["training"]["batch_size"]), 16)
    else:
        cfg["data"]["image_size"] = 224

    result = run_training(cfg=cfg, use_weighted_sampler=True, enable_mlflow=False)
    tune.report(best_metric=float(result["best_metric"]), best_epoch=int(result["best_epoch"]))


def main() -> None:
    args = parse_args()
    tune = check_ray_installed()

    search_space = {
        "base_config_path": args.config,
        "model_name": tune.choice(["efficientnet_b0", "efficientnet_b3"]),
        "learning_rate": tune.loguniform(1e-5, 3e-4),
        "batch_size": tune.choice([8, 16, 32]),
        "epochs": tune.choice([5, 10]),
        "focal_gamma": tune.choice([1.0, 2.0, 3.0]),
    }

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="best_metric",
            mode="max",
            num_samples=args.num_samples,
        ),
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="best_metric", mode="max")

    print("Best config:")
    print(best_result.config)
    print("Best metrics:")
    print(best_result.metrics)


if __name__ == "__main__":
    main()
