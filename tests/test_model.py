import torch

from skin_cancer.modeling.model import build_model


def test_build_model_forward_shape():
    model = build_model("efficientnet_b0", num_classes=7, pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 7)
