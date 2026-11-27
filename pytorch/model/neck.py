import torch
from torch import nn
from torch.nn import functional as F

from .blocks import ConvBlock, FusionBlock

# Neck: Backbone이 만든 세 해상도의 특징을 섞어 Head에 전달
# 작은 특징 맵을 키우는 Top-Down, 다시 축소하는 Bottom-up 순으로 실행
class Neck(nn.Module):
    """여러 해상도의 특징을 양방향으로 결합한다."""

    def __init__(self, norm="bn"):
        super().__init__()

        # 512채널 -> 256채널 / 1x1합성곱으로 채널 수를 줄임(공간 크기는 유지)
        self.reduce5 = ConvBlock(512, 256, kernel_size=1, norm=norm) 
        self.fuse4 = FusionBlock(256 + 256, 256, norm=norm)   # 두 특징을 채널 방향으로 붙여서 가공 2특징 256 + 256 -> 256채널

        self.reduce4 = ConvBlock(256, 128, kernel_size=1, norm=norm) 
        self.fuse3 = FusionBlock(128 + 128, 128, norm=norm) 

        # 128채널 -> 256채널 / stride=2로 공간 크기를 n/2, 즉 1/2로 줄임 
        self.down3 = ConvBlock(128, 256, stride=2, norm=norm) 
        self.pan4 = FusionBlock(256 + 256, 256, norm=norm) 

        self.down4 = ConvBlock(256, 512, stride=2, norm=norm) 
        self.pan5 = FusionBlock(512 + 512, 512, norm=norm) 

    def forward(self, c3, c4, c5):
        # NOTE: Top-down(특징 맵 키우기)
        p5 = self.reduce5(c5)  # 채널을 줄임 / c5: 512채널, 16x16 -> p5: 256채널, 16x16

        # p5를 c4의 공간 크기를 맞춤(p5의 특징을 확대)
        up5 = F.interpolate(  # interpolate: 입력 텐서(이미지, 특징 맵 등)를 지정된 크기나 비율에 맞춰 확대 또는 축소
            p5,  # p5의 정보를 가지고옴(배치, 256, 16, 16)
            size=c4.shape[-2:],  # c4의 높이와 너비를 가지고 옴(배치, 256, 32, 32) /  -2: [너비, 높이]
            mode="nearest",  # mode-nearest: 가까운 값을 복제하는 방식으로 확대 / 학습할 가중치는 없음
        )

        # up5의 채널 256, c4의 채널 256을 더해 512채널을 만들고 가공하여 256채널을 만듦 -> p4(배치, 256, 32, 32)
        p4 = self.fuse4(torch.cat((up5, c4), dim=1))  # up5과 c4를 결합 / dim=1: 채널 방향으로 이어 붙이기  -> 채널이 커짐
        # 0: 배치 방향으로 맞추기  |  2: 너비 방향  |  3: 높이 방향

        # 특징을 줄인 p4와 c3의 높이, 너비를 가지고 와서 up4에 대입 / up4: (배치, 128, 64, 64)
        up4 = F.interpolate(
            self.reduce4(p4),  # p4의 채널을 줄임: 256 -> 128
            size=c3.shape[-2:],  # c3의 높이와 너비를 가지고 옴(x1, x2, 64, 64)
            mode="nearest",
        )

        # up4의 채널 128, c3의 채널 128을 더해 256채널 후 가공하여 128채널을 만듬 -> p3: (배치, 128, 64, 64)
        p3 = self.fuse3(torch.cat((up4, c3), dim=1))


        # NOTE: Bottom-up(특징맵 줄이기)
        # p3: (x1, 128, 64, 64) -> down3, stride=2 -> (x2, 256, 32, 32)  |  p4: (x2, 256, 32, 32)  ==> n4: (x, 256, 32, 32)
        n4 = self.pan4(torch.cat((self.down3(p3), p4), dim=1))
        # n4: (x1, 256, 32, 32) -> down4, stride=2 -> (x2, 512, 16, 16)  |  c5: (x2, 512, 16, 16)  ==> n4: (x, 512, 16, 16)
        n5 = self.pan5(torch.cat((self.down4(n4), c5), dim=1))



        return p3, n4, n5

        # NOTE: 출력 -> p3, n4, n5
        # p3: (x, 128, 64, 64)
        # p4: (x, 256, 32, 32)
        # p5: (x, 256, 16, 16)

        # n4: (x, 256, 32, 32)
        # n5: (x, 512, 16, 16)