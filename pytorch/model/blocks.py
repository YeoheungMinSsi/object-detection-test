import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """합성곱 → 배치 정규화 → SiLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """입력 특징을 더해 깊은 층에서도 기울기가 전달되도록 한다."""

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=1),
            ConvBlock(channels, channels, kernel_size=3),
        )

    def forward(self, x):
        return x + self.block(x)


class FusionBlock(nn.Module):
    """특징을 두 경로로 처리한 뒤 결합한다."""

    def __init__(self, in_channels, out_channels, repeats=1):
        super().__init__()

        hidden = out_channels // 2

        self.main_path = nn.Sequential(
            ConvBlock(in_channels, hidden, kernel_size=1),
            *[ResidualBlock(hidden) for _ in range(repeats)],
        )
        self.short_path = ConvBlock(
            in_channels, hidden, kernel_size=1
        )
        self.merge = ConvBlock(
            hidden * 2, out_channels, kernel_size=1
        )

    def forward(self, x):
        main = self.main_path(x)
        shortcut = self.short_path(x)
        return self.merge(torch.cat((main, shortcut), dim=1))

class SpatialPyramidPooling(nn.Module):
    """서로 다른 범위의 주변 정보를 결합한다."""

    def __init__(self, channels):
        super().__init__()

        hidden = channels // 2

        self.reduce = ConvBlock(channels, hidden, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.merge = ConvBlock(hidden * 4, channels, kernel_size=1)

    def forward(self, x):
        x = self.reduce(x)

        pooled1 = self.pool(x)
        pooled2 = self.pool(pooled1)
        pooled3 = self.pool(pooled2)

        return self.merge(
            torch.cat((x, pooled1, pooled2, pooled3), dim=1)
        )