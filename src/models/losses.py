"""Custom Loss functions for segmentation and ordinal classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Combined Dice and BCE Loss for binary and multi-label segmentation."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice

        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)


class FocalLoss(nn.Module):
    """Focal Loss to combat severe lesion class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce
        return focal_loss.mean()


class MultiClassDiceCrossEntropyLoss(nn.Module):
    """Combined CrossEntropy and Multi-Class Dice Loss for RITE Artery/Vein."""

    def __init__(self, num_classes: int = 3, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)

        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dice_total = 0.0
        # Exclude background (class 0) from dice calculation if desired, or average over all
        for c in range(1, self.num_classes):
            p = probs[:, c, :, :].contiguous().view(-1)
            t = targets_onehot[:, c, :, :].contiguous().view(-1)
            intersection = (p * t).sum()
            dice = (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_total += (1.0 - dice)

        dice_loss = dice_total / (self.num_classes - 1)
        return ce_loss + dice_loss


class MultiChannelDiceBCELoss(nn.Module):
    """Dice + BCE loss computed per channel and averaged (e.g. Channel 0: Disc, Channel 1: Cup)."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)

        num_channels = logits.size(1)
        dice_loss = 0.0
        for c in range(num_channels):
            p_c = probs[:, c].reshape(-1)
            t_c = targets[:, c].reshape(-1)
            inter = (p_c * t_c).sum()
            d_c = (2.0 * inter + self.smooth) / (p_c.sum() + t_c.sum() + self.smooth)
            dice_loss += (1.0 - d_c)

        dice_loss = dice_loss / max(num_channels, 1)
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)


class MultiChannelFocalDiceLoss(nn.Module):
    """Per-channel Focal Loss + Per-channel Soft Dice Loss for multi-class retinal lesions.

    Combines hard negative mining via Focal Loss (to heavily penalize false positive
    vessels and background regions) with independent per-channel Soft Dice loss (to ensure
    tiny microaneurysms and hemorrhages receive equitable gradient magnitude).
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        gamma: float = 2.0,
        alpha: float = 0.75,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_channels = logits.size(1)
        probs = torch.sigmoid(logits)

        channel_losses = []
        for c in range(num_channels):
            l_c = logits[:, c]
            p_c = probs[:, c]
            t_c = targets[:, c]

            # 1. Per-channel Focal Loss
            bce = F.binary_cross_entropy_with_logits(l_c, t_c, reduction="none")
            p_t = t_c * p_c + (1.0 - t_c) * (1.0 - p_c)
            alpha_t = t_c * self.alpha + (1.0 - t_c) * (1.0 - self.alpha)
            focal = (alpha_t * ((1.0 - p_t) ** self.gamma) * bce).mean()

            # 2. Per-channel Soft Dice Loss
            p_flat = p_c.reshape(-1)
            t_flat = t_c.reshape(-1)
            inter = (p_flat * t_flat).sum()
            dice = (2.0 * inter + self.smooth) / (p_flat.sum() + t_flat.sum() + self.smooth)
            dice_loss = 1.0 - dice

            channel_losses.append((self.focal_weight * focal) + (self.dice_weight * dice_loss))

        return sum(channel_losses) / max(num_channels, 1)


