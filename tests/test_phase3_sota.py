"""Unit tests for SOTA Phase 3 components: Ben Graham Equalization, Patch Dataset, Focal Tversky Loss, and Sliding Window Inference."""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.preprocessing.enhancement import ben_graham_equalize
from src.models.losses import MultiChannelFocalTverskyLoss
from src.evaluation.sliding_window import create_gaussian_weight_window, sliding_window_inference
from src.datasets.patch_idrid_dataset import PatchIDRiDLesionDataset


def test_ben_graham_equalize_synthetic():
    # Synthetic fundus with uniform tissue and dark border
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[30:170, 30:170] = 150  # Retinal tissue

    enhanced = ben_graham_equalize(img, sigma=10.0)
    assert enhanced.shape == img.shape
    assert enhanced.dtype == np.uint8
    # Background outside mask should remain 0
    assert enhanced[0, 0, 0] == 0
    # Central uniform retinal area should be equalized to neutral 128
    assert abs(int(enhanced[100, 100, 0]) - 128) <= 2


def test_focal_tversky_loss():
    loss_fn = MultiChannelFocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)
    logits = torch.randn(2, 4, 64, 64)
    targets = torch.randint(0, 2, (2, 4, 64, 64)).float()

    loss = loss_fn(logits, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


def test_gaussian_weight_window():
    w = create_gaussian_weight_window(patch_size=256, sigma_scale=0.25)
    assert w.shape == (1, 1, 256, 256)
    assert abs(w.max().item() - 1.0) < 1e-5
    # Corners should be significantly smaller than center
    assert w[0, 0, 0, 0].item() < 0.1
    assert w[0, 0, 128, 128].item() == 1.0


def test_sliding_window_inference_uniform():
    class ConstantModel(nn.Module):
        def forward(self, x):
            return torch.full((x.size(0), 4, x.size(2), x.size(3)), 1.5, device=x.device)

    model = ConstantModel()
    dummy_img = torch.rand(1, 3, 512, 512)
    probs = sliding_window_inference(model, dummy_img, patch_size=256, stride=128, batch_size=4)

    assert probs.shape == (1, 4, 512, 512)
    expected_prob = 1.0 / (1.0 + np.exp(-1.5))  # sigmoid(1.5) = 0.81757
    assert abs(probs.mean().item() - expected_prob) < 1e-3
    assert abs(probs.min().item() - expected_prob) < 1e-3
    assert abs(probs.max().item() - expected_prob) < 1e-3


def test_patch_idrid_dataset_sampling():
    manifest = "manifests/dr_lesions/idrid_lesions_train.csv"
    ds = PatchIDRiDLesionDataset(
        manifest_path=manifest,
        base_size=512,
        patch_size=128,
        patches_per_image=4,
        lesion_prob=0.70,
        cache_in_ram=True,
        limit=2
    )
    assert len(ds) == 8
    sample = ds[0]
    assert "image" in sample and "mask" in sample
    assert sample["image"].shape == (3, 128, 128)
    assert sample["mask"].shape == (4, 128, 128)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32
