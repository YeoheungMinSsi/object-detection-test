from torch import nn

from .blocks import ConvBlock, FusionBlock, SpatialPyramidPooling


class Backbone(nn.Module):
    """이미지에서 세 가지 해상도의 특징을 추출한다."""

    def __init__(self, norm="bn"):
        super().__init__()
        self.stem = ConvBlock(3, 32, stride=2, norm=norm)

        self.stage2 = nn.Sequential(
            # 32채널 -> 64채널,  256x256 ->(stride=2)-> 128x128
            ConvBlock(32, 64, stride=2, norm=norm),
            FusionBlock(64, 64, repeats=2, norm=norm),
        )

        self.stage3 = nn.Sequential(
            # 64채널 -> 128채널,  128x128 ->(stride=2)-> 64x64
            ConvBlock(64, 128, stride=2, norm=norm),
            # 잔차블록 2번 반복
            FusionBlock(128, 128, repeats=4, norm=norm),
        )

        self.stage4 = nn.Sequential(
            # 128채널 -> 256채널,  64x64 ->(stride=2)-> 32x32
            ConvBlock(128, 256, stride=2, norm=norm),
            FusionBlock(256, 256, repeats=3, norm=norm),
        )

        self.stage5 = nn.Sequential(
            # 256채널 -> 512채널,  32x32 ->(stride=2)-> 16x16 / 특징맵: 512채널과 16x16
            ConvBlock(256, 512, stride=2, norm=norm),
            FusionBlock(512, 512, repeats=1, norm=norm),
            # 같은 특징에 풀링을 연속 적용한 결과들을 합쳐 주변 정보를 보완 / 최종 공간 크기(16x16) 유지
            # SpatialPyramidPooling(SPP): 입력 이미지나 특징 맵의 크기가 어떻든 항상 고정된 크기의 벡터를 출력하는 풀링 기법
            SpatialPyramidPooling(512, norm=norm),
        )


    def forward(self, images):
        x = self.stem(images)  # 입력 이미지 -> 출력: 32채널, n/2 -> 512/2 = 256x256
        x = self.stage2(x)  # 입력 이미지(x) -> 출력: 64채널, n/2 ->  256/2 = 128x128

        c3 = self.stage3(x)  # 입력 이미지(x): 채널 64, 128x128 -> 출력(c3): 128채널, n/2 ->  128/2 = 64x64
        c4 = self.stage4(c3)  # 입력 이미지(c3): 채널 128, 64x64 -> 출력(c4): 256채널, n/2 ->  64/2 = 32x32
        c5 = self.stage5(c4)  # 입력 이미지(c4): 채널 256, 32x32 -> 출력(c5): 512채널, n/2 ->  32/2 = 16x16

        return c3, c4, c5