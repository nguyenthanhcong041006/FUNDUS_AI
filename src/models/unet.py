"""U-Net architecture for retinal segmentation tasks."""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class DoubleConv(nn.Module):
        """[Conv2d -> Norm -> ReLU] * 2 with support for BatchNorm2d or GroupNorm."""
        def __init__(self, in_channels: int, out_channels: int, norm_type: str = "batch", num_groups: int = 8):
            super().__init__()
            if norm_type == "group":
                # Ensure num_groups divides out_channels evenly
                g = num_groups if out_channels % num_groups == 0 else (4 if out_channels % 4 == 0 else 1)
                norm1 = nn.GroupNorm(g, out_channels)
                norm2 = nn.GroupNorm(g, out_channels)
            else:
                norm1 = nn.BatchNorm2d(out_channels)
                norm2 = nn.BatchNorm2d(out_channels)

            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                norm1,
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                norm2,
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.double_conv(x)


    class Down(nn.Module):
        """Downscaling with MaxPool2d then DoubleConv"""
        def __init__(self, in_channels: int, out_channels: int, norm_type: str = "batch", num_groups: int = 8):
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            )

        def forward(self, x):
            return self.maxpool_conv(x)


    class Up(nn.Module):
        """Upscaling then double conv"""
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bilinear: bool = True,
            norm_type: str = "batch",
            num_groups: int = 8
        ):
            super().__init__()
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                self.conv = DoubleConv(in_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            else:
                self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels, out_channels, norm_type=norm_type, num_groups=num_groups)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            # Input shape: [N, C, H, W]
            diff_y = x2.size()[2] - x1.size()[2]
            diff_x = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)


    class OutConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            return self.conv(x)


    class UNet(nn.Module):
        """Classic U-Net for vessel, optic disc, and lesion segmentation.
        
        Supports norm_type='batch' or norm_type='group' (e.g. 8 or 16 groups) to avoid
        batch statistics instability during small-batch training (batch_size=2).
        """

        def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 1,
            bilinear: bool = True,
            norm_type: str = "batch",
            num_groups: int = 8
        ):
            super().__init__()
            self.in_channels = in_channels
            self.num_classes = num_classes
            self.bilinear = bilinear
            self.norm_type = norm_type
            self.num_groups = num_groups

            self.inc = DoubleConv(in_channels, 64, norm_type=norm_type, num_groups=num_groups)
            self.down1 = Down(64, 128, norm_type=norm_type, num_groups=num_groups)
            self.down2 = Down(128, 256, norm_type=norm_type, num_groups=num_groups)
            self.down3 = Down(256, 512, norm_type=norm_type, num_groups=num_groups)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor, norm_type=norm_type, num_groups=num_groups)
            self.up1 = Up(1024, 512 // factor, bilinear, norm_type=norm_type, num_groups=num_groups)
            self.up2 = Up(512, 256 // factor, bilinear, norm_type=norm_type, num_groups=num_groups)
            self.up3 = Up(256, 128 // factor, bilinear, norm_type=norm_type, num_groups=num_groups)
            self.up4 = Up(128, 64, bilinear, norm_type=norm_type, num_groups=num_groups)
            self.outc = OutConv(64, num_classes)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            return logits

else:
    class UNet:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate UNet. Please install torch.")


def build_unet(
    in_channels: int = 3,
    num_classes: int = 1,
    encoder_name: str = "custom",
    pretrained: bool = True,
    norm_type: str = "batch"
):
    """Factory function for U-Net models.

    Supports segmentation_models_pytorch if installed, otherwise uses native UNet.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed in the current environment.")

    if encoder_name != "custom":
        try:
            import segmentation_models_pytorch as smp
            weights = "imagenet" if pretrained else None
            return smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=num_classes,
            )
        except ImportError:
            pass

    return UNet(in_channels=in_channels, num_classes=num_classes, norm_type=norm_type)


def load_checkpoint_unet(
    checkpoint_path: str,
    in_channels: int = 3,
    num_classes: int = 2,
    map_location: str = "cpu"
) -> "UNet":
    """Safely loads a U-Net checkpoint, auto-detecting BatchNorm2d vs GroupNorm."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required.")

    state = torch.load(str(checkpoint_path), map_location=map_location)
    has_running_mean = any("running_mean" in k for k in state.keys())
    norm_type = "batch" if has_running_mean else "group"

    model = UNet(in_channels=in_channels, num_classes=num_classes, norm_type=norm_type)
    model.load_state_dict(state)
    model.eval()
    return model


