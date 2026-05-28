from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from skin_cancer.core.config import config_arg_parser, load_config
from skin_cancer.core.utils import save_json

REQUIRED_COLUMNS = {"image_id", "dx", "age", "sex", "localization"}


def build_image_index(raw_dir: str | Path, extensions: list[str]) -> dict[str, Path]:
    """Index images by stem, e.g. ISIC_0024306 -> path."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    normalized_extensions = {ext.lower() for ext in extensions}
    image_index: dict[str, Path] = {}
    for file_path in raw_path.rglob("*"):
        if file_path.suffix.lower() in normalized_extensions:
            image_index[file_path.stem] = file_path
    return image_index


def validate_metadata_columns(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing required metadata columns: {sorted(missing)}")
    return errors


def validate_image_paths(df: pd.DataFrame, image_index: dict[str, Path]) -> tuple[pd.Series, list[str]]:
    exists = df["image_id"].astype(str).isin(image_index.keys())
    errors: list[str] = []
    missing_count = int((~exists).sum())
    if missing_count > 0:
        sample = df.loc[~exists, "image_id"].astype(str).head(10).tolist()
        errors.append(f"Missing image files for {missing_count} rows. Sample: {sample}")
    return exists, errors


def validate_corrupted_images(paths: list[Path], max_images: int | None = None) -> list[str]:
    """Try reading images with OpenCV and return paths that fail."""
    corrupted: list[str] = []
    subset = paths if max_images is None else paths[:max_images]
    for path in tqdm(subset, desc="Checking images", leave=False):
        image = cv2.imread(str(path))
        if image is None:
            corrupted.append(str(path))
    return corrupted


def validate_dataset(config_path: str = "configs/train_config.yaml") -> dict[str, object]:
    cfg = load_config(config_path)
    metadata_path = Path(cfg.paths.metadata_file)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    image_index = build_image_index(cfg.paths.raw_dir, cfg.data.image_extensions)

    errors = validate_metadata_columns(df)
    image_exists, path_errors = validate_image_paths(df, image_index)
    errors.extend(path_errors)

    valid_paths = [image_index[image_id] for image_id in df.loc[image_exists, "image_id"].astype(str)]
    corrupted = validate_corrupted_images(valid_paths)
    if corrupted:
        errors.append(f"Found {len(corrupted)} corrupted/unreadable images.")

    class_counts = df["dx"].value_counts(dropna=False).to_dict() if "dx" in df.columns else {}
    missing_values = df.isna().sum().to_dict()

    report = {
        "num_rows": int(len(df)),
        "num_indexed_images": int(len(image_index)),
        "num_missing_image_rows": int((~image_exists).sum()),
        "num_corrupted_images": int(len(corrupted)),
        "corrupted_images_sample": corrupted[:20],
        "class_counts": class_counts,
        "missing_values": {key: int(value) for key, value in missing_values.items()},
        "errors": errors,
        "is_valid": len(errors) == 0,
    }

    output_path = Path(cfg.paths.report_dir) / "metrics" / "data_validation_report.json"
    save_json(report, output_path)
    return report


def main() -> None:
    parser = config_arg_parser("Validate HAM10000 raw data and metadata.")
    args = parser.parse_args()
    report = validate_dataset(args.config)
    if report["is_valid"]:
        print("Data validation passed.")
    else:
        print("Data validation found issues:")
        for error in report["errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    main()
