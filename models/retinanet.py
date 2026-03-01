import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


# ---------------------------------
# Feature Pyramid Network (FPN)
# ---------------------------------
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

        # Extra pyramid levels P6 and P7
        self.p6 = nn.Conv2d(in_channels_list[-1], out_channels,
                            kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Conv2d(out_channels, out_channels,
                            kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        c3, c4, c5 = inputs

        p5 = self.lateral_convs[2](c5)
        p4 = self.lateral_convs[1](c4) + F.interpolate(p5, scale_factor=2)
        p3 = self.lateral_convs[0](c3) + F.interpolate(p4, scale_factor=2)

        p3 = self.output_convs[0](p3)
        p4 = self.output_convs[1](p4)
        p5 = self.output_convs[2](p5)

        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))

        return [p3, p4, p5, p6, p7]


# ---------------------------------
# RetinaNet Subnet (shared structure)
# ---------------------------------
class RetinaHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=4):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)
        self.output = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return self.output(x)


# ---------------------------------
# RetinaNet Model
# ---------------------------------
class RetinaNet(nn.Module):
    def __init__(self, num_classes, num_anchors=9):
        super().__init__()

        backbone = resnet50(pretrained=True)

        # Extract C3, C4, C5
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,  # C3
            backbone.layer3,  # C4
            backbone.layer4,  # C5
        )

        self.fpn = FPN(
            in_channels_list=[512, 1024, 2048],
            out_channels=256
        )

        self.cls_head = RetinaHead(256, num_anchors * num_classes)
        self.box_head = RetinaHead(256, num_anchors * 4)

        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x):

        # Extract features
        x = self.backbone[0:6](x)  # up to C3
        c3 = x
        x = self.backbone[6](x)    # C4
        c4 = x
        x = self.backbone[7](x)    # C5
        c5 = x

        features = self.fpn([c3, c4, c5])

        cls_outputs = []
        box_outputs = []

        for feature in features:
            cls_outputs.append(self.cls_head(feature))
            box_outputs.append(self.box_head(feature))

        return cls_outputs, box_outputs