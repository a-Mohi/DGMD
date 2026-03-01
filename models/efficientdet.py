import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0


# -------------------------------------------------
# Depthwise Separable Convolution (used in heads & BiFPN)
# -------------------------------------------------
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


# -------------------------------------------------
# BiFPN Block (Simplified Weighted Fusion)
# -------------------------------------------------
class BiFPN(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.w1 = nn.Parameter(torch.ones(2))
        self.w2 = nn.Parameter(torch.ones(2))

        self.conv3 = SeparableConvBlock(channels, channels)
        self.conv4 = SeparableConvBlock(channels, channels)
        self.conv5 = SeparableConvBlock(channels, channels)

    def forward(self, p3, p4, p5):

        # Normalize weights
        w1 = F.relu(self.w1)
        w1 = w1 / (w1.sum() + 1e-6)

        w2 = F.relu(self.w2)
        w2 = w2 / (w2.sum() + 1e-6)

        # Top-down
        p4_td = self.conv4(w1[0] * p4 + w1[1] * F.interpolate(p5, scale_factor=2))
        p3_td = self.conv3(w1[0] * p3 + w1[1] * F.interpolate(p4_td, scale_factor=2))

        # Bottom-up
        p4_out = self.conv4(w2[0] * p4_td + w2[1] * F.max_pool2d(p3_td, 2))
        p5_out = self.conv5(w2[0] * p5 + w2[1] * F.max_pool2d(p4_out, 2))

        return p3_td, p4_out, p5_out


# -------------------------------------------------
# Shared Head (classification or box)
# -------------------------------------------------
class DetectionHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=4):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(SeparableConvBlock(in_channels, in_channels))

        self.conv = nn.Sequential(*layers)
        self.output = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return self.output(x)


# -------------------------------------------------
# EfficientDet Model (D0-style)
# -------------------------------------------------
class EfficientDet(nn.Module):
    def __init__(self, num_classes, num_anchors=9, bifpn_channels=64, bifpn_layers=3):
        super().__init__()

        # Backbone
        backbone = efficientnet_b0(weights="IMAGENET1K_V1")
        self.backbone = backbone.features

        # Extract feature layers indices manually (for B0)
        self.p3_idx = 4
        self.p4_idx = 6
        self.p5_idx = 8

        # Reduce channels to BiFPN width
        self.reduce_p3 = nn.Conv2d(40, bifpn_channels, 1)
        self.reduce_p4 = nn.Conv2d(112, bifpn_channels, 1)
        self.reduce_p5 = nn.Conv2d(320, bifpn_channels, 1)

        # Stack BiFPN layers
        self.bifpn = nn.Sequential(
            *[BiFPN(bifpn_channels) for _ in range(bifpn_layers)]
        )

        # Heads
        self.cls_head = DetectionHead(
            bifpn_channels,
            num_anchors * num_classes
        )
        self.box_head = DetectionHead(
            bifpn_channels,
            num_anchors * 4
        )

        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x):

        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == self.p3_idx:
                p3 = x
            elif i == self.p4_idx:
                p4 = x
            elif i == self.p5_idx:
                p5 = x

        p3 = self.reduce_p3(p3)
        p4 = self.reduce_p4(p4)
        p5 = self.reduce_p5(p5)

        for bifpn_layer in self.bifpn:
            p3, p4, p5 = bifpn_layer(p3, p4, p5)

        cls_outputs = []
        box_outputs = []

        for feature in [p3, p4, p5]:
            cls_outputs.append(self.cls_head(feature))
            box_outputs.append(self.box_head(feature))

        return cls_outputs, box_outputs