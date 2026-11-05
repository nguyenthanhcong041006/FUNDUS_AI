"""DenseNet-121 architecture for Multi-Label Disease Classification (ODIR-5K)."""

import torch
import torch.nn as nn
import torchvision.models as models


class FundusMultiLabelDenseNet(nn.Module):
    """DenseNet-121 backbone predicting 8 independent disease probabilities."""

    def __init__(self, num_classes: int = 8, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.base = models.densenet121(weights=weights)

        in_features = self.base.classifier.in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw unnormalized logits for BCEWithLogitsLoss."""
        return self.base(x)

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probabilities [0, 1] for each disease."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

