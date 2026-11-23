import torch
from torch import nn
from torch.nn import functional as F
from .blocks import ConvBlock


class DetectionHead(nn.Module):
    """사람 존재 점수와 박스 좌표를 별도 경로에서 예측한다."""

    # in_chennels: Neck에서 들어오는 채널 수, hidden: Head 내부에서 사용할 채널 수(기본 128)
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()

        # neck에서 p3(x, 128, 64, 64), n4(x, 256, 32, 32), n5(x, 512, 16, 16)가 들어옴
        # stem: (neck에서 들어오는 채널 수, 128, 커널 사이즈 1) -> NOTE: 들어오는 모든 채널을 128로 만듦(ConvBlock)
        # ConvBlock() -> (self, in_channels, out_channels, kernel_size=3, stride=1)
        # neck -> stem -> p3(x, 128, 64, 64), n4(x, 128, 32, 32), n5(x, 128, 16, 16)
        self.stem = ConvBlock(
            in_channels, hidden_channels, kernel_size=1
        )

        # 객체/사람(objectness_branch)이 있는지 판단하기 위한 특징 가공
        self.objectness_branch = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels),  # 첫번째 필터로 특징 가공(3x3 필터 사용)->패딩이 1이라 특징맵이 줄어들지 않음
            ConvBlock(hidden_channels, hidden_channels),  # 다른 필터로 한 번 더 가공
        )

        # 박스(objectness_branch)의 위치, 크기를 예측하기 위한 특징 가공
        self.box_branch = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels),  # 첫번째 필터로 특징 가공(3x3 필터 사용)
            ConvBlock(hidden_channels, hidden_channels),  # 다른 필터로 한 번 더 가공
        )

        # Conv2d: 이미지 등 2차원 공간 데이터를 입력받아 합성곱 연산을 수행하는 레이어
        # 객체/사람: (x, 채널 번호, 너비, 높이), 채널 번호 위치에 1(채널이 단 하나)로 지정했으니 0부터 카운트하니 0이 됌
        # 이미지 채널의 행열의 위치값(logit)을 sigmoid를 통해서 사람 존재 점수 반환함
        self.objectness_output = nn.Conv2d(
            hidden_channels, 1, kernel_size=1
        )
        # 박스: (x, 4, 너비, 높이) -> 여기서 4개는 박스 경로, 위치마다 (tx, ty, tw, th) 네 값을 만듦
        # x, y, w, h를 저장할 각 채널 4개를 만듦 -> 이후 박스의 최종 값에서 나온 중심좌표 x, y에 w, h를 1/2를 해서 좌표에서 박스를 만듦
        self.box_output = nn.Conv2d(
            hidden_channels, 4, kernel_size=1
        )

        # 학습 초기에 모든 위치에서 사람을 예측하지 않도록 초기 존재 확률을 약 1%로 설정한다.
        prior_probability = 0.01

        # 학습 bias 초기 값 log(0.0101) = -4.xxx
        initial_bias = torch.log(
            torch.tensor(
                prior_probability / (1.0 - prior_probability)
            )
        ).item()

        # head를 처음 생성 시 사람 점수 출력층의 bias 초기값 설정 objectness_output.bias에 initial_bias 값을 넣음
        nn.init.constant_(self.objectness_output.bias, initial_bias)
        # head를 처음 생성 시 박스 위치 출력층의 bias 값을 모두 0으로 넣음
        nn.init.zeros_(self.box_output.bias)

    def forward(self, x):
        x = self.stem(x)

        # x에서 객체(사람)과 박스 경로 2가지 시작
        objectness = self.objectness_output(
            self.objectness_branch(x)
        )
        boxes = self.box_output(
            self.box_branch(x)
        )

        # sigmoid/softplus는 손실 함수 또는 좌표 복원에서 적용
        # 1채널, 4채널을 더해서 5채널[사람 존재 logit, tx, ty, tw, th]을 만듦 -> 최종 입력받은 값들의 채널수를 5로 만듦
        return torch.cat((objectness, boxes), dim=1)

        # p3: (x, 5, 64, 64)
        # n4: (x, 5, 32, 32)
        # n5: (x, 5, 16, 16) 로 최종 출력
        
        # Note: 각 채널
        # result[0, 0, 25, 10]  # 사람 존재 logit
        # result[0, 1, 25, 10]  # tx
        # result[0, 2, 25, 10]  # ty
        # result[0, 3, 25, 10]  # tw
        # result[0, 4, 25, 10]  # th