import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ----------------------------
# Deconvolution Block
# ----------------------------
class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.deconv(x)


# ----------------------------
# Detection Head
# ----------------------------
class Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.head(x)


# ----------------------------
# CenterNet Model
# ----------------------------
class CenterNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Backbone (ResNet-18)
        backbone = resnet18(pretrained=True)

        # Remove fully connected layer
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # Decoder (upsample 3 times)
        self.deconv_layers = nn.Sequential(
            DeconvLayer(512, 256),
            DeconvLayer(256, 128),
            DeconvLayer(128, 64),
        )

        # Heads
        self.heatmap_head = Head(64, num_classes)
        self.size_head = Head(64, 2)      # width, height
        self.offset_head = Head(64, 2)    # x_offset, y_offset

        self._initialize_bias()

    def _initialize_bias(self):
        # Initialize heatmap bias to negative value
        # (helps training stability)
        self.heatmap_head.head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        features = self.backbone(x)
        features = self.deconv_layers(features)

        heatmap = self.heatmap_head(features)
        heatmap = torch.sigmoid(heatmap)

        size = self.size_head(features)
        offset = self.offset_head(features)

        return {
            "heatmap": heatmap,
            "size": size,
            "offset": offset
        }

