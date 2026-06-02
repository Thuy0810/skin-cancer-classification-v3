from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from skin_cancer.core.config import config_arg_parser, load_config
from skin_cancer.modeling.model import get_last_conv_layer, load_model_from_checkpoint
from skin_cancer.inference.predict import load_image_tensor
from skin_cancer.core.utils import get_device, load_checkpoint


class GradCAM:

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int, float]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(probabilities.argmax(dim=1).item())
        confidence = float(probabilities[0, target_class].item())

        score = logits[:, target_class].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class, confidence


def overlay_cam(original_image_path: str | Path, cam: np.ndarray, image_size: int, alpha: float = 0.45) -> np.ndarray:
    image = cv2.imread(str(original_image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {original_image_path}")
    image = cv2.resize(image, (image_size, image_size))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
    return overlay


def generate_gradcam(
    image_path: str | Path,
    checkpoint_path: str | Path,
    config_path: str,
    output_path: str | Path,
    target_class: int | None = None,
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

    layer_name, target_layer = get_last_conv_layer(model)
    input_tensor = load_image_tensor(image_path, int(cfg.data.image_size)).to(device)

    gradcam = GradCAM(model, target_layer)
    try:
        cam, predicted_class, confidence = gradcam(input_tensor, target_class=target_class)
    finally:
        gradcam.remove_hooks()

    overlay = overlay_cam(image_path, cam, int(cfg.data.image_size))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)

    return {
        "image_path": str(image_path),
        "output_path": str(output_path),
        "target_layer": layer_name,
        "class_id": predicted_class,
        "class_name": cfg.labels.label_names[predicted_class],
        "confidence": confidence,
    }


def main() -> None:
    parser = config_arg_parser("Generate Grad-CAM for one image.")
    parser.add_argument("--image", type=str, required=True, help="Path to image.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument("--output", type=str, default="reports/figures/gradcam.png", help="Output Grad-CAM image path.")
    parser.add_argument("--target-class", type=int, default=None, help="Optional target class id.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = args.checkpoint or cfg.paths.best_model_path
    result = generate_gradcam(args.image, checkpoint_path, args.config, args.output, args.target_class)
    print(result)


if __name__ == "__main__":
    main()
