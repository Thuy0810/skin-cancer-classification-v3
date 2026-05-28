import numpy as np

from skin_cancer.data.transforms import get_valid_transforms


def test_valid_transform_shape():
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    transformed = get_valid_transforms(224)(image=image)["image"]
    assert tuple(transformed.shape) == (3, 224, 224)
