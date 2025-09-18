import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(c2)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class BottleneckBlock(nn.Module):
    """A bottleneck layer with optional shortcut and group convolution for efficient feature
    extraction."""

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut supporting channel
        expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBlock(c1, c_, 1, 1, 0)
        self.cv2 = ConvBlock(c_, c2, 3, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel
        dimensions match; input is a tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """Implements a CSP Bottleneck module with three convolutions for enhanced feature
    extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition,
        shortcut usage, group convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBlock(c1, c_, 1, 1)
        self.cv2 = ConvBlock(c1, c_, 1, 1)
        self.cv3 = ConvBlock(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(BottleneckBlock(c_, c_, shortcut, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a
        Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    """Implements a fast Spatial Pyramid Pooling (SPPF) layer for efficient feature extraction in
    CustomYOLOv5 models."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes SPPF layer with given channels and kernel size for model,
        combining convolution and max pooling.

        Equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvBlock(c1, c_, 1, 1)
        self.cv2 = ConvBlock(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Processes input through a series of convolutions and max pooling operations for
        feature extraction."""
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class CustomYOLOv5(nn.Module):
    def __init__(self, num_classes=10, num_anchors=1, anchors=None):
        super().__init__()
        # input = 384 x 640 x 3

        self.stride = 16
        anchors = [[8, 8]] if anchors is None else anchors
        self.anchors = torch.tensor(anchors)

        self.nc = num_classes
        self.na = num_anchors
        nx, ny = 40, 24  # grid design
        self.grid, self.anchor_grid = self._make_grid(nx, ny)

        # Iteration 1
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, 6, 2, 2),  # 192 x 320 x 32
            ConvBlock(32, 64, 3, 2, 1),  # 96 x 160 x 64
            C3(64, 64, 1),  # 96 x 160 x 64
            ConvBlock(64, 128, 3, 2, 1),  # 48 x 80 x 128
            C3(128, 128, 2),  # 48 x 80 x 128

            SPPF(128, 128),  # 48 x 80 x 128
            ConvBlock(128, 256, 3, 2, 1),  # 24 x 40 x 256
        )

        self.neck = nn.Sequential(
            C3(256, 256, 1),
            ConvBlock(256, self.na * (self.nc + 5), 1, 1, 0)
        )

    def head(self, x):
        bs, _, h, w = x.shape
        x = x.view(bs, self.na, (self.nc + 5), h, w).permute(0, 1, 3, 4, 2).contiguous()

        xy, wh, p, conf = x.sigmoid().split((2, 2, 1, self.nc), dim=4)
        xy = (xy * 2 + self.grid) * self.stride
        wh = (wh * 2) ** 2 * self.anchor_grid
        conf = F.softmax(conf, dim=4)
        y = torch.cat((xy, wh, p, conf), dim=4)
        return y

    def forward(self, x):
        # input image 384 x 640 x 3
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def _make_grid(self, nx=40, ny=24):
        """Generates a mesh grid for anchor boxes"""
        d = self.anchors.device
        t = self.anchors.dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xv, yv), 2).expand(
            shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors * self.stride).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
