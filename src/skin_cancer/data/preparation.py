from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from skin_cancer.core.config import config_arg_parser, load_config
from skin_cancer.data.validation import build_image_index, validate_metadata_columns
from skin_cancer.core.utils import ensure_dir, save_json


def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Basic metadata cleaning for HAM10000."""
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["dx"] = df["dx"].astype(str).str.lower().str.strip()

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["age"] = df["age"].fillna(df["age"].median())

    for col in ["sex", "localization", "dx_type"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.lower().str.strip()

    return df


def add_image_paths(df: pd.DataFrame, raw_dir: str | Path, extensions: list[str]) -> pd.DataFrame:
    image_index = build_image_index(raw_dir, extensions)
    df = df.copy()
    df["image_path"] = df["image_id"].map(lambda image_id: str(image_index.get(str(image_id), "")))
    missing = df["image_path"].eq("")
    if missing.any():
        missing_ids = df.loc[missing, "image_id"].head(10).tolist()
        raise FileNotFoundError(
            f"Could not find image files for {int(missing.sum())} rows. Sample: {missing_ids}"
        )
    return df


def encode_labels(df: pd.DataFrame, label_names: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    unknown_labels = sorted(set(df["dx"].unique()) - set(label_to_id.keys()))
    if unknown_labels:
        raise ValueError(f"Unknown labels in metadata: {unknown_labels}")

    df = df.copy()
    df["label"] = df["dx"].map(label_to_id).astype(int)
    return df, label_to_id


def stratified_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test split while preserving class distribution."""
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    val_relative_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df["label"],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_data(config_path: str = "configs/train_config.yaml") -> dict[str, object]:
    cfg = load_config(config_path)
    metadata_path = Path(cfg.paths.metadata_file)
    processed_dir = ensure_dir(cfg.paths.processed_dir)
    interim_dir = ensure_dir(cfg.paths.interim_dir)

    df = pd.read_csv(metadata_path)
    errors = validate_metadata_columns(df)
    if errors:
        raise ValueError("; ".join(errors))

    df = clean_metadata(df)
    df = add_image_paths(df, cfg.paths.raw_dir, cfg.data.image_extensions)
    df, label_to_id = encode_labels(df, cfg.labels.label_names)

    clean_path = interim_dir / "metadata_clean.parquet"
    df.to_parquet(clean_path, index=False)

    train_df, val_df, test_df = stratified_split(
        df,
        test_size=float(cfg.data.test_size),
        val_size=float(cfg.data.val_size),
        random_state=int(cfg.seed),
    )

    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)

    split_report = {
        "num_total": int(len(df)),
        "num_train": int(len(train_df)),
        "num_val": int(len(val_df)),
        "num_test": int(len(test_df)),
        "label_to_id": label_to_id,
        "train_class_counts": train_df["dx"].value_counts().to_dict(),
        "val_class_counts": val_df["dx"].value_counts().to_dict(),
        "test_class_counts": test_df["dx"].value_counts().to_dict(),
    }
    save_json(split_report, Path(cfg.paths.report_dir) / "metrics" / "split_report.json")
    return split_report


def main() -> None:
    parser = config_arg_parser("Prepare HAM10000 data into train/val/test parquet files.")
    args = parser.parse_args()
    report = prepare_data(args.config)
    print("Data prepared successfully.")
    print(report)


if __name__ == "__main__":
    main()
