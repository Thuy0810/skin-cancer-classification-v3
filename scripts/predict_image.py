from __future__ import annotations

import csv
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from skin_cancer.core.config import load_config
from skin_cancer.core.utils import get_device, load_checkpoint
from skin_cancer.data.transforms import get_valid_transforms
from skin_cancer.modeling.model import load_model_from_checkpoint


CONFIG_PATH = PROJECT_ROOT / "configs/densenet121/densenet121_bs4_gamma2.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "models/densenet121_bs4_gamma2_no_sampler_wd5e4/best_model.pth"
INPUT_DIR = PROJECT_ROOT / "reports/selected_test_images_7"
OUTPUT_DIR = PROJECT_ROOT / "reports/predict_image_output"
OUTPUT_IMAGE = OUTPUT_DIR / "prediction_grid.jpg"
OUTPUT_CSV = OUTPUT_DIR / "predictions.csv"

TOP_K = 3
MAX_IMAGES = 7
GRID_COLUMNS = 4
TILE_WIDTH = 420
TILE_HEIGHT = 440
IMAGE_HEIGHT = 245
GAP = 16
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clean_output_dir() -> None:
    output_dir = OUTPUT_DIR.resolve()
    project_root = PROJECT_ROOT.resolve()
    if output_dir == project_root or project_root not in output_dir.parents:
        raise ValueError(f"Refusing to clean unsafe output dir: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def load_selected_metadata() -> dict[str, dict[str, str]]:
    metadata_path = INPUT_DIR / "selected_images.csv"
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        return {row["image_id"]: row for row in rows if row.get("image_id")}


def load_input_images() -> list[Path]:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    images = [
        path
        for path in sorted(INPUT_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No images found in: {INPUT_DIR}")
    return images[:MAX_IMAGES]


def resize_to_box(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    img_h, img_w = image_bgr.shape[:2]
    scale = min(width / img_w, height / img_h)
    new_w = max(1, int(img_w * scale))
    new_h = max(1, int(img_h * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def put_text_lines(tile: np.ndarray, lines: list[str], x: int, y: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    for line in lines:
        cv2.putText(tile, line, (x, y), font, 0.58, (30, 30, 30), 2, cv2.LINE_AA)
        y += 27


def make_tile(image_bgr: np.ndarray, image_id: str, true_label: str, predictions: list[dict[str, object]]) -> np.ndarray:
    tile = np.full((TILE_HEIGHT, TILE_WIDTH, 3), 255, dtype=np.uint8)
    tile[:IMAGE_HEIGHT, :] = resize_to_box(image_bgr, TILE_WIDTH, IMAGE_HEIGHT)

    cv2.rectangle(tile, (0, 0), (TILE_WIDTH - 1, TILE_HEIGHT - 1), (210, 210, 210), 1)
    cv2.rectangle(tile, (0, IMAGE_HEIGHT), (TILE_WIDTH - 1, TILE_HEIGHT - 1), (255, 255, 255), -1)

    lines = [image_id]
    if true_label:
        lines.append(f"True: {true_label}")
    lines.extend(
        f"{rank}. {item['class_name']}: {float(item['confidence']) * 100:.2f}%"
        for rank, item in enumerate(predictions, start=1)
    )
    put_text_lines(tile, lines, 14, IMAGE_HEIGHT + 28)
    return tile


def make_grid(tiles: list[np.ndarray]) -> np.ndarray:
    rows = math.ceil(len(tiles) / GRID_COLUMNS)
    width = GRID_COLUMNS * TILE_WIDTH + (GRID_COLUMNS + 1) * GAP
    height = rows * TILE_HEIGHT + (rows + 1) * GAP
    grid = np.full((height, width, 3), 235, dtype=np.uint8)

    for index, tile in enumerate(tiles):
        row = index // GRID_COLUMNS
        col = index % GRID_COLUMNS
        x = GAP + col * (TILE_WIDTH + GAP)
        y = GAP + row * (TILE_HEIGHT + GAP)
        grid[y : y + TILE_HEIGHT, x : x + TILE_WIDTH] = tile

    return grid


@torch.no_grad()
def predict() -> None:
    clean_output_dir()

    cfg = load_config(CONFIG_PATH)
    device = get_device(cfg.training.device)
    checkpoint = load_checkpoint(CHECKPOINT_PATH, map_location=device)
    model = load_model_from_checkpoint(
        checkpoint=checkpoint,
        model_name=cfg.model.name,
        num_classes=int(cfg.model.num_classes),
        device=device,
    )
    transform = get_valid_transforms(int(cfg.data.image_size))
    metadata_by_id = load_selected_metadata()
    image_paths = load_input_images()

    tiles: list[np.ndarray] = []
    csv_rows: list[dict[str, object]] = []

    print(f"Input: {INPUT_DIR}")
    print(f"Using device: {device}")
    print(f"Predicting {len(image_paths)} image(s)...")

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)
        probabilities = torch.softmax(model(image_tensor), dim=1).squeeze(0)
        values, indices = torch.topk(probabilities, k=min(TOP_K, len(cfg.labels.label_names)))

        predictions = [
            {
                "class_id": int(class_id),
                "class_name": cfg.labels.label_names[int(class_id)],
                "confidence": float(confidence),
            }
            for confidence, class_id in zip(values.cpu(), indices.cpu())
        ]

        image_id = image_path.stem
        true_label = metadata_by_id.get(image_id, {}).get("true_label", "")
        tiles.append(make_tile(image_bgr, image_id, true_label, predictions))

        best = predictions[0]
        print(f"{image_id}: {best['class_name']} ({float(best['confidence']) * 100:.2f}%)")

        for rank, item in enumerate(predictions, start=1):
            csv_rows.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "true_label": true_label,
                    "rank": rank,
                    "class_id": item["class_id"],
                    "class_name": item["class_name"],
                    "confidence_percent": f"{float(item['confidence']) * 100:.2f}",
                }
            )

    if not tiles:
        raise RuntimeError("No readable images were predicted.")

    cv2.imwrite(str(OUTPUT_IMAGE), make_grid(tiles))
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "image_path",
                "true_label",
                "rank",
                "class_id",
                "class_name",
                "confidence_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Grid image: {OUTPUT_IMAGE}")
    print(f"CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    predict()
