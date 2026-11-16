from torch import nn

from .blocks import ConvBlock, FusionBlock, SpatialPyramidPooling


class Backbone(nn.Module):
    """이미지에서 세 가지 해상도의 특징을 추출한다."""

    def __init__(self):
        super().__init__()

        # 이미지의 3채널을 32채널 특징으로 변환, stride로 인해 특징 맵이 입력 이미지에서 n/2(stride=2) 줄어듦
        # 입력 (배치, 3채널, 높이, 넓이) -> 출력 (배치, 32채널, 높이, 넓이)
        self.stem = ConvBlock(3, 32, stride=2)

        
        self.stage2 = nn.Sequential(
            # 32채널 -> 64채널,  256x256 ->(stride=2)-> 128x128
            ConvBlock(32, 64, stride=2),
            # 64채널과 특징맵 크기를 유지하면서 특징 처리 / repeats는 블록 내부의 ResidualBlock(잔차블록)을 한번 사용
            # ResidualBlock: 입력 데이터를 여러 층을 통과시킨 출력(F(x))에 원래 입력을 그대로 더하는 잔차 구조/ H(x) = F(x) + x
            FusionBlock(64, 64, repeats=1),
        )

        self.stage3 = nn.Sequential(
            # 64채널 -> 128채널,  128x128 ->(stride=2)-> 64x64
            ConvBlock(64, 128, stride=2),
            # 잔차블록 2번 반복 -> 두 번 쌓으면 공간 크기 유지, 더많은 연산으로 특징 조합, 더 넓은 주변 정보 반영 가능
            FusionBlock(128, 128, repeats=2),
        )

        self.stage4 = nn.Sequential(
            # 128채널 -> 256채널,  64x64 ->(stride=2)-> 32x32
            ConvBlock(128, 256, stride=2),
            FusionBlock(256, 256, repeats=2),
        )

        self.stage5 = nn.Sequential(
            # 256채널 -> 512채널,  32x32 ->(stride=2)-> 16x16 / 특징맵: 512채널과 16x16
            ConvBlock(256, 512, stride=2),
            FusionBlock(512, 512, repeats=1),
            # 같은 특징에 풀링을 연속 적용한 결과들을 합쳐 주변 정보를 보완 / 최종 공간 크기(16x16) 유지
            # SpatialPyramidPooling(SPP): 입력 이미지나 특징 맵의 크기가 어떻든 항상 고정된 크기의 벡터를 출력하는 풀링 기법
            SpatialPyramidPooling(512),
        )


    def forward(self, images):
        x = self.stem(images)  # 입력 이미지 -> 출력: 32채널, n/2 -> 512/2 = 256x256
        x = self.stage2(x)  # 입력 이미지(x) -> 출력: 64채널, n/2 ->  256/2 = 128x128

        c3 = self.stage3(x)  # 입력 이미지(x): 채널 64, 128x128 -> 출력(c3): 128채널, n/2 ->  128/2 = 64x64
        c4 = self.stage4(c3)  # 입력 이미지(c3): 채널 128, 64x64 -> 출력(c4): 256채널, n/2 ->  64/2 = 32x32
        c5 = self.stage5(c4)  # 입력 이미지(c4): 채널 256, 32x32 -> 출력(c5): 512채널, n/2 ->  32/2 = 16x16

        return c3, c4, c5

# fusionblock 초반 1번 -> 특징 맵이 너무 커서 블록을 늘리면 계산 비용이 너무 큼
# fusionblock 중반 2번 -> 위치 정보와 복합적인 특징을 함께 다뤄 처리 깊이를 확보
# fusionblock 마지막 1번 -> 채널이 많고 공간 정보를 결합하는 풀링 블록이 있어 1번으로 제한