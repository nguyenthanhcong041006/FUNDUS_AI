"""Multimodal feature fusion model combining image embeddings with quantitative measurements."""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class FundusFusionModel(nn.Module):
        """Fuses deep CNN embeddings with engineered quantitative measurement features."""

        def __init__(
            self,
            cnn_feature_dim: int = 1280,
            measurement_dim: int = 16,
            hidden_dim: int = 128,
            num_classes: int = 5
        ):
            super().__init__()
            # Projection for measurement vector
            self.tabular_mlp = nn.Sequential(
                nn.Linear(measurement_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )

            # Combined fusion head
            fusion_dim = cnn_feature_dim + 64
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes)
            )

        def forward(self, cnn_features: torch.Tensor, measurement_vector: torch.Tensor):
            tab_feats = self.tabular_mlp(measurement_vector)
            fused = torch.cat([cnn_features, tab_feats], dim=1)
            logits = self.classifier(fused)
            return logits

else:
    class FundusFusionModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for FundusFusionModel.")

