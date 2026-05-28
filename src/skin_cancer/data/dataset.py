from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class SkinLesionDataset(Dataset):
    """PyTorch Dataset for HAM10000 skin lesion images."""

    def __init__(self, dataframe: pd.DataFrame, transform: Any | None = None) -> None:
        required = {"image_path", "label"}
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f"Dataframe missing required columns: {sorted(missing)}")
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        image_path = Path(row["image_path"])
        label = int(row["label"])

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, torch.tensor(label, dtype=torch.long)


def load_split_dataframe(processed_dir: str | Path, split: str) -> pd.DataFrame:
    path = Path(processed_dir) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed split not found: {path}. Run prepare_data.py first.")
    return pd.read_parquet(path)
