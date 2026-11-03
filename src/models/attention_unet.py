"""Attention U-Net architecture for retinal lesion and structure segmentation."""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)


    class AttentionGate(nn.Module):
        """Attention Gate to focus on salient structures/lesions."""
        def __init__(self, f_g: int, f_l: int, f_int: int):
            super().__init__()
            self.w_g = nn.Sequential(
                nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(f_int)
            )
            self.w_x = nn.Sequential(
                nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(f_int)
            )
            self.psi = nn.Sequential(
                nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            g1 = self.w_g(g)
            x1 = self.w_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi


    class AttentionUNet(nn.Module):
        """Attention U-Net implementation."""
        def __init__(self, in_channels: int = 3, num_classes: int = 4):
            super().__init__()
            filters = [64, 128, 256, 512, 1024]

            self.conv1 = ConvBlock(in_channels, filters[0])
            self.maxpool1 = nn.MaxPool2d(2)

            self.conv2 = ConvBlock(filters[0], filters[1])
            self.maxpool2 = nn.MaxPool2d(2)

            self.conv3 = ConvBlock(filters[1], filters[2])
            self.maxpool3 = nn.MaxPool2d(2)

            self.conv4 = ConvBlock(filters[2], filters[3])
            self.maxpool4 = nn.MaxPool2d(2)

            self.conv5 = ConvBlock(filters[3], filters[4])

            # Decoder
            self.up5 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
            self.att5 = AttentionGate(f_g=filters[3], f_l=filters[3], f_int=filters[2])
            self.up_conv5 = ConvBlock(filters[4], filters[3])

            self.up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
            self.att4 = AttentionGate(f_g=filters[2], f_l=filters[2], f_int=filters[1])
            self.up_conv4 = ConvBlock(filters[3], filters[2])

            self.up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
            self.att3 = AttentionGate(f_g=filters[1], f_l=filters[1], f_int=filters[0])
            self.up_conv3 = ConvBlock(filters[2], filters[1])

            self.up2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
            self.att2 = AttentionGate(f_g=filters[0], f_l=filters[0], f_int=filters[0] // 2)
            self.up_conv2 = ConvBlock(filters[1], filters[0])

            self.outc = nn.Conv2d(filters[0], num_classes, kernel_size=1)

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(self.maxpool1(x1))
            x3 = self.conv3(self.maxpool2(x2))
            x4 = self.conv4(self.maxpool3(x3))
            x5 = self.conv5(self.maxpool4(x4))

            d5 = self.up5(x5)
            x4_att = self.att5(g=d5, x=x4)
            d5 = torch.cat((x4_att, d5), dim=1)
            d5 = self.up_conv5(d5)

            d4 = self.up4(d5)
            x3_att = self.att4(g=d4, x=x3)
            d4 = torch.cat((x3_att, d4), dim=1)
            d4 = self.up_conv4(d4)

            d3 = self.up3(d4)
            x2_att = self.att3(g=d3, x=x2)
            d3 = torch.cat((x2_att, d3), dim=1)
            d3 = self.up_conv3(d3)

            d2 = self.up2(d3)
            x1_att = self.att2(g=d2, x=x1)
            d2 = torch.cat((x1_att, d2), dim=1)
            d2 = self.up_conv2(d2)

            return self.outc(d2)

else:
    class AttentionUNet:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for AttentionUNet. Please install torch.")

