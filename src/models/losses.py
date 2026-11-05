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


