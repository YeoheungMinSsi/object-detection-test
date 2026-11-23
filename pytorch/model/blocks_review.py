import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """합성곱 → 배치 정규화 → SiLU."""

    # 입력 특징 맵의 채널수, 출력 특징 맵의 채널수, 커널 크기, 스프라이드 크기
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        # block 초기화
        self.block = nn.Sequential(  # Sequential은 여러 층을 순서대로 실행해 주는 묶음 (레이어, 배치정규, 활성화 함수)
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,  # 패딩: 커널//2 = 1 (w, h의 크기가 변동되지 않음)
                bias=False,
            ),
            # 채널마다 배치와 공간 위치의 값을 사용해 정규화
            # 학습에서는 현재 입력의 통게로 정규화하고 이동 통계를 갱신, 평가에서는 저장된 이동 통계를 사용
            nn.BatchNorm2d(out_channels),  
            # 활성화 함수: 음수 값을 모두 잘라내는 대신 부드럽게 조절

            # 입력 텐서의 저장 공간을 재사용, BatchNorm의 출력을 활성화
            nn.SiLU(inplace=True),  
        )

    # 위 레이어 수행
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