from __future__ import annotations

from pathlib import Path

import cv2
import torch

from skin_cancer.core.config import config_arg_parser, load_config
from skin_cancer.modeling.model import load_model_from_checkpoint
from skin_cancer.data.transforms import get_valid_transforms
from skin_cancer.core.utils import get_device, load_checkpoint


def load_image_tensor(image_path: str | Path, image_size: int) -> torch.Tensor:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = get_valid_transforms(image_size)(image=image)["image"]
    return transformed.unsqueeze(0)


@torch.no_grad()
def predict_image(
    image_path: str | Path,
    checkpoint_path: str | Path,
    config_path: str = "configs/train_config.yaml",
    top_k: int | None = None,
) -> dict[str, object]:
    cfg = load_config(config_path)
    device = get_device(cfg.training.device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = load_model_from_checkpoint(
        checkpoint=checkpoint,
        model_name=cfg.model.name,
        num_classes=int(cfg.model.num_classes),
        device=device,
    )

    image_tensor = load_image_tensor(image_path, int(cfg.data.image_size)).to(device)
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    k = top_k or int(cfg.inference.top_k)
    values, indices = torch.topk(probabilities, k=min(k, len(cfg.labels.label_names)))

    predictions = [
        {"class_id": int(idx), "class_name": cfg.labels.label_names[int(idx)], "confidence": float(value)}
        for value, idx in zip(values.cpu(), indices.cpu())
    ]
    return {"image_path": str(image_path), "predictions": predictions}


def main() -> None:
    parser = config_arg_parser("Predict one skin lesion image.")
    parser.add_argument("--image", type=str, required=True, help="Path to image.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of top predictions to print.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = args.checkpoint or cfg.paths.best_model_path
    result = predict_image(args.image, checkpoint_path, args.config, args.top_k)
    print(result)


if __name__ == "__main__":
    main()
