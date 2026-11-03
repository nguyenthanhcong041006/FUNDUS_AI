"""Disease classification backbones (EfficientNet, ResNet)."""

from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class FundusClassifier(nn.Module):
        """Fundus disease classifier supporting multi-label & ordinal grading."""

        def __init__(
            self,
            backbone: str = "efficientnet_b0",
            num_classes: int = 5,
            pretrained: bool = True,
            dropout: float = 0.3
        ):
            super().__init__()
            self.backbone_name = backbone
            self.num_classes = num_classes

            if backbone.startswith("efficientnet"):
                if backbone == "efficientnet_b0":
                    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                    self.base = models.efficientnet_b0(weights=weights)
                else:
                    weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
                    self.base = models.efficientnet_b2(weights=weights)

                in_features = self.base.classifier[1].in_features
                self.base.classifier = nn.Sequential(
                    nn.Dropout(p=dropout, inplace=True),
                    nn.Linear(in_features, num_classes)
                )

            elif backbone.startswith("resnet"):
                if backbone == "resnet34":
                    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
                    self.base = models.resnet34(weights=weights)
                else:
                    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                    self.base = models.resnet50(weights=weights)

                in_features = self.base.fc.in_features
                self.base.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features, num_classes)
                )

            else:
                raise ValueError(f"Unsupported backbone: {backbone}")

        def forward(self, x):
            return self.base(x)

        def extract_features(self, x):
            """Extracts embedding vector before the classification head."""
            if hasattr(self.base, "features"):
                features = self.base.features(x)
                features = self.base.avgpool(features)
                return torch.flatten(features, 1)
            elif hasattr(self.base, "conv1"):
                x = self.base.conv1(x)
                x = self.base.bn1(x)
                x = self.base.relu(x)
                x = self.base.maxpool(x)
                x = self.base.layer1(x)
                x = self.base.layer2(x)
                x = self.base.layer3(x)
                x = self.base.layer4(x)
                x = self.base.avgpool(x)
                return torch.flatten(x, 1)
            return None

else:
    class FundusClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch and torchvision are required for FundusClassifier.")


def build_classifier(
    backbone: str = "efficientnet_b0",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.3
):
    """Factory helper to build a classification model."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed in the current environment.")
    return FundusClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )

