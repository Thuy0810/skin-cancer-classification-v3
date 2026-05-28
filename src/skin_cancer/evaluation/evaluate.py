from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from skin_cancer.core.config import config_arg_parser, load_config
from skin_cancer.data.dataset import SkinLesionDataset, load_split_dataframe
from skin_cancer.evaluation.metrics import compute_classification_metrics, plot_confusion_matrix
from skin_cancer.modeling.model import load_model_from_checkpoint
from skin_cancer.data.transforms import get_valid_transforms
from skin_cancer.core.utils import get_device, load_checkpoint, save_json


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
    plot_confusion_matrix(
        y_true,
        y_pred,
        list(cfg.labels.label_names),
        report_dir / "figures" / "confusion_matrix.png",
        normalize=False,
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        list(cfg.labels.label_names),
        report_dir / "figures" / "confusion_matrix_normalized.png",
        normalize=True,
    )

    print("Test metrics saved.")
    print({key: value for key, value in metrics.items() if isinstance(value, (float, int))})


if __name__ == "__main__":
    main()
