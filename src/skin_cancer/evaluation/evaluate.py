from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from skin_cancer.core.config import config_arg_parser, load_config
from skin_cancer.data.dataset import SkinLesionDataset, load_split_dataframe
from skin_cancer.data.transforms import get_valid_transforms
from skin_cancer.evaluation.metrics import compute_classification_metrics, save_confusion_matrix_artifacts
from skin_cancer.modeling.model import load_model_from_checkpoint
from skin_cancer.core.utils import get_device, load_checkpoint, save_json


def normalize_stage_run_name(run_name: str, stage: str) -> str:
    if run_name.startswith((f"{stage}_", "train_", "evaluate_")):
        return run_name
    return f"{stage}_{run_name}"


@torch.no_grad()
def predict_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_batches: list[np.ndarray] = []
    y_pred_batches: list[np.ndarray] = []
    y_prob_batches: list[np.ndarray] = []

    model.eval()
    for images, labels in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        y_true_batches.append(labels.numpy())
        y_pred_batches.append(predictions.cpu().numpy())
        y_prob_batches.append(probabilities.cpu().numpy())

    return np.concatenate(y_true_batches), np.concatenate(y_pred_batches), np.concatenate(y_prob_batches)


def main() -> None:
    parser = config_arg_parser("Evaluate best model on test set.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = args.checkpoint or cfg.paths.best_model_path
    device = get_device(cfg.training.device)

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = load_model_from_checkpoint(
        checkpoint=checkpoint,
        model_name=cfg.model.name,
        num_classes=int(cfg.model.num_classes),
        device=device,
    )

    test_df = load_split_dataframe(cfg.paths.processed_dir, "test")
    test_dataset = SkinLesionDataset(test_df, transform=get_valid_transforms(int(cfg.data.image_size)))
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
    )

    y_true, y_pred, y_prob = predict_loader(model, test_loader, device)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob, list(cfg.labels.label_names))

    report_dir = Path(cfg.paths.report_dir)
    save_json(metrics, report_dir / "metrics" / "test_metrics.json")
    save_confusion_matrix_artifacts(
        y_true,
        y_pred,
        list(cfg.labels.label_names),
        report_dir / "figures",
        base_name="confusion_matrix",
    )

    if bool(getattr(cfg.mlflow, "enabled", False)):
        import mlflow

        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

        checkpoint_name = Path(checkpoint_path).stem
        run_name = normalize_stage_run_name(f"{cfg.model.name}_{checkpoint_name}", "evaluate")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "stage": "evaluate",
                    "model": cfg.model.name,
                    "checkpoint": checkpoint_name,
                    "image_size": cfg.data.image_size,
                    "batch_size": cfg.training.batch_size,
                }
            )
            mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, (float, int))})

    print("Test metrics saved.")
    print({key: value for key, value in metrics.items() if isinstance(value, (float, int))})


if __name__ == "__main__":
    main()
