from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from skin_cancer.core.config import (
    config_arg_parser,
    dict_to_namespace,
    load_config,
    namespace_to_dict,
)
from skin_cancer.core.utils import ensure_dir, get_device, save_json, set_seed
from skin_cancer.data.dataset import SkinLesionDataset, load_split_dataframe
from skin_cancer.data.transforms import get_train_transforms, get_valid_transforms
from skin_cancer.modeling.losses import build_criterion
from skin_cancer.modeling.model import build_model
from skin_cancer.training.trainer import train_model

def build_auto_run_name(
    cfg: SimpleNamespace | dict[str, Any],
    use_weighted_sampler: bool,
) -> str:
    import mlflow

    cfg_dict = cfg if isinstance(cfg, dict) else namespace_to_dict(cfg)

    experiment_name = cfg_dict.get("mlflow", {}).get(
        "experiment_name",
        "default",
    )

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is not None:
        existing_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            output_format="pandas",
        )
        run_index = len(existing_runs) + 1
    else:
        run_index = 1

    model_name = cfg_dict.get("model", {}).get("name", "model")
    loss_name = cfg_dict.get("loss", {}).get("name", "loss")

    sampler_name = "sampler" if use_weighted_sampler else "no_sampler"

    return f"train_{run_index:03d}_{model_name}_{loss_name}_{sampler_name}"


def normalize_stage_run_name(run_name: str, stage: str) -> str:
    if run_name.startswith((f"{stage}_", "train_", "evaluate_")):
        return run_name
    return f"{stage}_{run_name}"


def build_optimizer(model: torch.nn.Module, cfg: SimpleNamespace) -> torch.optim.Optimizer:
    name = cfg.optimizer.name.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.optimizer.learning_rate),
            weight_decay=float(cfg.optimizer.weight_decay),
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.optimizer.learning_rate),
            momentum=0.9,
            weight_decay=float(cfg.optimizer.weight_decay),
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: SimpleNamespace,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = cfg.scheduler.name.lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler.name}")


def build_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    class_counts = train_df["label"].value_counts().to_dict()
    weights = train_df["label"].map(lambda label: 1.0 / class_counts[int(label)]).to_numpy()
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _prepare_dataloaders(
    cfg: SimpleNamespace,
    use_weighted_sampler: bool,
) -> tuple[DataLoader, DataLoader, pd.DataFrame]:
    train_df = load_split_dataframe(cfg.paths.processed_dir, "train")
    val_df = load_split_dataframe(cfg.paths.processed_dir, "val")

    train_transform = get_train_transforms(
        image_size=int(cfg.data.image_size),
        horizontal_flip_p=float(cfg.augmentation.horizontal_flip_p),
        vertical_flip_p=float(cfg.augmentation.vertical_flip_p),
        rotate_limit=int(cfg.augmentation.rotate_limit),
        brightness_contrast_p=float(cfg.augmentation.brightness_contrast_p),
        shift_scale_rotate_p=float(cfg.augmentation.shift_scale_rotate_p),
    )
    valid_transform = get_valid_transforms(image_size=int(cfg.data.image_size))

    train_dataset = SkinLesionDataset(train_df, transform=train_transform)
    val_dataset = SkinLesionDataset(val_df, transform=valid_transform)

    sampler = build_weighted_sampler(train_df) if use_weighted_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
    )
    return train_loader, val_loader, train_df


def _start_mlflow_run(
    cfg: SimpleNamespace,
    use_weighted_sampler: bool,
) -> Any | None:
    if not bool(cfg.mlflow.enabled):
        return None

    import mlflow

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_name = getattr(cfg.mlflow, "run_name", None)

    if not run_name:
        run_name = build_auto_run_name(
            cfg=cfg,
            use_weighted_sampler=use_weighted_sampler,
        )

    run_name = normalize_stage_run_name(str(run_name), "train")

    mlflow.start_run(run_name=run_name)

    mlflow.log_params(
        {
            "run_name": run_name,
            "model": cfg.model.name,
            "image_size": cfg.data.image_size,
            "batch_size": cfg.training.batch_size,
            "epochs": cfg.training.epochs,
            "learning_rate": cfg.optimizer.learning_rate,
            "optimizer": cfg.optimizer.name,
            "loss": cfg.loss.name,
            "gamma": cfg.loss.gamma,
            "weighted_sampler": use_weighted_sampler,
        }
    )

    return mlflow

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    mlflow.start_run(run_name=f"{cfg.model.name}-{cfg.loss.name}")
    mlflow.log_params(
        {
            "model": cfg.model.name,
            "image_size": cfg.data.image_size,
            "batch_size": cfg.training.batch_size,
            "epochs": cfg.training.epochs,
            "learning_rate": cfg.optimizer.learning_rate,
            "optimizer": cfg.optimizer.name,
            "loss": cfg.loss.name,
            "gamma": cfg.loss.gamma,
            "weighted_sampler": use_weighted_sampler,
        }
    )
    return mlflow


def run_training(
    cfg: SimpleNamespace | dict[str, Any],
    use_weighted_sampler: bool = False,
    enable_mlflow: bool | None = None,
) -> dict[str, Any]:
    """Run the full training pipeline.

    This function is intentionally reusable by three entrypoints:
    - normal CLI training: python -m skin_cancer.training.train
    - Ray Tune: mlops/ray/tune.py
    - Airflow or other orchestration tools
    """
    if isinstance(cfg, dict):
        cfg = dict_to_namespace(cfg)

    cfg_dict = namespace_to_dict(cfg)
    if enable_mlflow is not None:
        cfg.mlflow.enabled = bool(enable_mlflow)
        cfg_dict["mlflow"]["enabled"] = bool(enable_mlflow)

    set_seed(int(cfg.seed))
    device = get_device(cfg.training.device)
    print(f"Using device: {device}")

    train_loader, val_loader, train_df = _prepare_dataloaders(
        cfg=cfg,
        use_weighted_sampler=use_weighted_sampler,
    )

    label_to_id = {label: idx for idx, label in enumerate(cfg.labels.label_names)}
    model = build_model(
        model_name=cfg.model.name,
        num_classes=int(cfg.model.num_classes),
        pretrained=bool(cfg.model.pretrained),
    ).to(device)

    criterion = build_criterion(
        loss_name=cfg.loss.name,
        num_classes=int(cfg.model.num_classes),
        train_labels=train_df["label"].tolist(),
        device=device,
        alpha=cfg.loss.alpha,
        gamma=float(cfg.loss.gamma),
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, int(cfg.training.epochs))

    ensure_dir(cfg.paths.model_dir)
    ensure_dir(Path(cfg.paths.report_dir) / "metrics")

    mlflow_module = _start_mlflow_run(cfg, use_weighted_sampler=use_weighted_sampler)
    history_path = Path(cfg.paths.report_dir) / "metrics" / "train_history.json"

    try:
        result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=int(cfg.training.epochs),
            label_names=list(cfg.labels.label_names),
            label_to_id=label_to_id,
            checkpoint_path=cfg.paths.best_model_path,
            config_dict=cfg_dict,
            monitor_metric=cfg.training.monitor_metric,
            mixed_precision=bool(cfg.training.mixed_precision),
            early_stopping_patience=int(cfg.training.early_stopping_patience),
            mlflow_module=mlflow_module,
        )
        save_json(result, history_path)

        if mlflow_module is not None:
            mlflow_module.log_metric("best_metric", float(result["best_metric"]))
            mlflow_module.log_metric("best_epoch", int(result["best_epoch"]))
            mlflow_module.log_artifact(str(cfg.paths.best_model_path))
            mlflow_module.log_artifact(str(history_path))
    finally:
        if mlflow_module is not None:
            mlflow_module.end_run()

    return result


def main() -> None:
    parser = config_arg_parser("Train EfficientNet on HAM10000.")
    parser.add_argument(
        "--use-weighted-sampler",
        action="store_true",
        help="Oversample minority classes during training.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow for this run, even if it is enabled in the config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_training(
        cfg=cfg,
        use_weighted_sampler=args.use_weighted_sampler,
        enable_mlflow=False if args.no_mlflow else None,
    )


if __name__ == "__main__":
    main()
