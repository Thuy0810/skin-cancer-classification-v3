from __future__ import annotations

import csv
import glob
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from skin_cancer.core.config import load_config
from skin_cancer.core.utils import get_device, load_checkpoint
from skin_cancer.data.transforms import get_valid_transforms
from skin_cancer.modeling.model import load_model_from_checkpoint


# Sua cac dong ben duoi roi bam Run file nay.
CONFIG_PATH = PROJECT_ROOT / "configs/densenet121/densenet121_bs4_gamma2.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "models/densenet121_bs4_gamma2_no_sampler_wd5e4/best_model.pth"
OUTPUT_DIR = PROJECT_ROOT / "reports/predictions_test_only_7"
TOP_K = 3
RESET_OUTPUT_DIR = True
AUTO_PICK_VALID_IMAGES = True
AUTO_PICK_SPLIT = "test"
AUTO_PICK_COUNT = 7
AUTO_PICK_ONE_PER_CLASS = True

# Co the dien file anh, folder anh, hoac glob neu muon chon thu cong. Vi du:
# IMAGE_INPUTS = [
#     r"data/raw/HAM10000_images_part_1/ISIC_0024306.jpg",
#     r"D:\somewhere\image_01.jpg",
#     r"data/raw/HAM10000_images_part_1/*.jpg",
#     r"data/my_test_images",
# ]
IMAGE_INPUTS = [
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_image_inputs(inputs: list[str | Path]) -> list[Path]:
    image_paths: list[Path] = []
    seen: set[Path] = set()

    for item in inputs:
        raw_text = str(item).strip()
        if not raw_text:
            continue

        if any(char in raw_text for char in "*?[]"):
            pattern = str(resolve_path(raw_text))
            candidates = [Path(match) for match in glob.glob(pattern, recursive=True)]
        else:
            path = resolve_path(raw_text)
            if path.is_dir():
                candidates = [
                    candidate
                    for candidate in sorted(path.rglob("*"))
                    if candidate.suffix.lower() in IMAGE_EXTENSIONS
                ]
            else:
                candidates = [path]

        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            image_paths.append(candidate)

    return image_paths


def is_readable_image(image_path: Path) -> bool:
    if not image_path.exists():
        return False
    return cv2.imread(str(image_path)) is not None


def reset_output_dir() -> None:
    output_dir = OUTPUT_DIR.resolve()
    project_root = PROJECT_ROOT.resolve()
    if output_dir == project_root or project_root not in output_dir.parents:
        raise ValueError(f"Refusing to reset unsafe output dir: {output_dir}")
    if RESET_OUTPUT_DIR and output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "original_images").mkdir(parents=True, exist_ok=True)
    (output_dir / "annotated_images").mkdir(parents=True, exist_ok=True)


def load_train_identities(cfg) -> tuple[set[Path], set[str]]:
    train_path = resolve_path(cfg.paths.processed_dir) / "train.parquet"
    if not train_path.exists():
        return set(), set()

    train_df = pd.read_parquet(train_path)
    train_paths = {resolve_path(path).resolve() for path in train_df["image_path"].tolist()}
    train_image_ids = set(train_df["image_id"].astype(str).tolist()) if "image_id" in train_df.columns else set()
    return train_paths, train_image_ids


def auto_pick_valid_images(cfg) -> tuple[list[Path], dict[Path, dict[str, object]]]:
    split_path = resolve_path(cfg.paths.processed_dir) / f"{AUTO_PICK_SPLIT}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Processed split not found: {split_path}. Run prepare_data first.")

    df = pd.read_parquet(split_path)
    train_paths, train_image_ids = load_train_identities(cfg)
    if AUTO_PICK_ONE_PER_CLASS and "label" in df.columns:
        df = df.sort_values(["label", "image_id"] if "image_id" in df.columns else ["label"])

    selected: list[Path] = []
    metadata_by_path: dict[Path, dict[str, object]] = {}
    selected_labels: set[int] = set()

    for _, row in df.iterrows():
        image_path = resolve_path(row["image_path"]).resolve()
        image_id = str(row.get("image_id", image_path.stem))
        label = int(row["label"]) if "label" in row else None

        if AUTO_PICK_ONE_PER_CLASS and label is not None and label in selected_labels:
            continue
        if image_path in train_paths or image_id in train_image_ids:
            continue
        if not is_readable_image(image_path):
            continue

        selected.append(image_path)
        if label is not None:
            selected_labels.add(label)

        metadata_by_path[image_path] = {
            "split": AUTO_PICK_SPLIT,
            "true_label_id": label,
            "true_label": row.get("dx", ""),
            "image_id": image_id,
            "from_train_split": False,
        }

        if len(selected) >= AUTO_PICK_COUNT:
            break

    if len(selected) < AUTO_PICK_COUNT and AUTO_PICK_ONE_PER_CLASS:
        for _, row in df.iterrows():
            image_path = resolve_path(row["image_path"]).resolve()
            image_id = str(row.get("image_id", image_path.stem))
            if image_path in metadata_by_path:
                continue
            if image_path in train_paths or image_id in train_image_ids:
                continue
            if not is_readable_image(image_path):
                continue

            label = int(row["label"]) if "label" in row else None
            selected.append(image_path)
            metadata_by_path[image_path] = {
                "split": AUTO_PICK_SPLIT,
                "true_label_id": label,
                "true_label": row.get("dx", ""),
                "image_id": image_id,
                "from_train_split": False,
            }

            if len(selected) >= AUTO_PICK_COUNT:
                break

    return selected, metadata_by_path


def draw_predictions(image_bgr, predictions: list[dict[str, object]], metadata: dict[str, object] | None = None):
    lines = [
        f"{index}. {item['class_name']}: {float(item['confidence']) * 100:.2f}%"
        for index, item in enumerate(predictions, start=1)
    ]
    if metadata and metadata.get("true_label"):
        lines.insert(0, f"True: {metadata['true_label']}")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(image_bgr.shape[1] / 900.0, 0.9))
    thickness = max(1, int(round(font_scale * 2)))
    padding = 12
    line_height = int(28 * font_scale) + 8

    text_width = 0
    for line in lines:
        (width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_width = max(text_width, width)

    box_width = min(image_bgr.shape[1], text_width + padding * 2)
    box_height = min(image_bgr.shape[0], line_height * len(lines) + padding)

    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (box_width, box_height), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.82, image_bgr, 0.18, 0, image_bgr)

    y = padding + int(18 * font_scale)
    for line in lines:
        cv2.putText(
            image_bgr,
            line,
            (padding, y),
            font,
            font_scale,
            (20, 20, 20),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height

    return image_bgr


def unique_output_path(directory: Path, image_path: Path, suffix: str, used_names: set[str]) -> Path:
    base_name = f"{image_path.stem}_prediction{image_path.suffix.lower()}"
    if suffix:
        base_name = f"{image_path.stem}_{suffix}{image_path.suffix.lower()}"
    candidate = directory / base_name
    index = 2
    while candidate.name.lower() in used_names or candidate.exists():
        if suffix:
            candidate = directory / f"{image_path.stem}_{suffix}_{index}{image_path.suffix.lower()}"
        else:
            candidate = directory / f"{image_path.stem}_{index}{image_path.suffix.lower()}"
        index += 1
    used_names.add(candidate.name.lower())
    return candidate


def filter_out_train_images(cfg, image_paths: list[Path]) -> list[Path]:
    train_paths, train_image_ids = load_train_identities(cfg)
    filtered: list[Path] = []

    for image_path in image_paths:
        resolved = image_path.resolve()
        if resolved in train_paths or resolved.stem in train_image_ids:
            print(f"Skip train image: {image_path}")
            continue
        filtered.append(image_path)

    return filtered


@torch.no_grad()
def main() -> None:
    cfg = load_config(CONFIG_PATH)
    metadata_by_path: dict[Path, dict[str, object]] = {}
    image_paths = resolve_image_inputs(IMAGE_INPUTS)
    if image_paths:
        image_paths = filter_out_train_images(cfg, image_paths)
    if not image_paths and AUTO_PICK_VALID_IMAGES:
        image_paths, metadata_by_path = auto_pick_valid_images(cfg)

    if not image_paths:
        print("Chua co anh de predict. Hay them duong dan vao IMAGE_INPUTS trong scripts/predict_images.py")
        return

    reset_output_dir()
    original_dir = OUTPUT_DIR / "original_images"
    annotated_dir = OUTPUT_DIR / "annotated_images"

    device = get_device(cfg.training.device)
    checkpoint = load_checkpoint(CHECKPOINT_PATH, map_location=device)
    model = load_model_from_checkpoint(
        checkpoint=checkpoint,
        model_name=cfg.model.name,
        num_classes=int(cfg.model.num_classes),
        device=device,
    )
    transform = get_valid_transforms(int(cfg.data.image_size))
    top_k = int(TOP_K or cfg.inference.top_k)

    csv_path = OUTPUT_DIR / "predictions.csv"
    used_original_names: set[str] = set()
    used_annotated_names: set[str] = set()
    rows: list[dict[str, object]] = []

    print(f"Using device: {device}")
    print(f"Predicting {len(image_paths)} image(s)...")

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        values, indices = torch.topk(probabilities, k=min(top_k, len(cfg.labels.label_names)))

        predictions = [
            {
                "class_id": int(class_id),
                "class_name": cfg.labels.label_names[int(class_id)],
                "confidence": float(confidence),
            }
            for confidence, class_id in zip(values.cpu(), indices.cpu())
        ]

        metadata = metadata_by_path.get(image_path.resolve(), {})
        annotated = draw_predictions(image_bgr.copy(), predictions, metadata)
        original_copy_path = unique_output_path(original_dir, image_path, "original", used_original_names)
        annotated_path = unique_output_path(annotated_dir, image_path, "prediction", used_annotated_names)
        shutil.copy2(image_path, original_copy_path)
        cv2.imwrite(str(annotated_path), annotated)

        best = predictions[0]
        print(f"{image_path.name}: {best['class_name']} ({float(best['confidence']) * 100:.2f}%)")

        for rank, prediction in enumerate(predictions, start=1):
            rows.append(
                {
                    "image_path": str(image_path),
                    "original_image": str(original_copy_path),
                    "output_image": str(annotated_path),
                    "split": metadata.get("split", ""),
                    "image_id": metadata.get("image_id", image_path.stem),
                    "true_label_id": metadata.get("true_label_id", ""),
                    "true_label": metadata.get("true_label", ""),
                    "from_train_split": metadata.get("from_train_split", False),
                    "rank": rank,
                    "class_id": prediction["class_id"],
                    "class_name": prediction["class_name"],
                    "confidence": f"{float(prediction['confidence']):.8f}",
                    "confidence_percent": f"{float(prediction['confidence']) * 100:.2f}",
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "original_image",
                "output_image",
                "split",
                "image_id",
                "true_label_id",
                "true_label",
                "from_train_split",
                "rank",
                "class_id",
                "class_name",
                "confidence",
                "confidence_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Annotated images and CSV are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
