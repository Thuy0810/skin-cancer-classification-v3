from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from skin_cancer.data.dataset import SkinLesionDataset
from skin_cancer.data.transforms import get_valid_transforms


def test_dataset_loads_image(tmp_path: Path):
    image_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))
    df = pd.DataFrame({"image_path": [str(image_path)], "label": [1]})
    dataset = SkinLesionDataset(df, transform=get_valid_transforms(64))
    image, label = dataset[0]
    assert tuple(image.shape) == (3, 64, 64)
    assert int(label) == 1
