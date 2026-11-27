import torch
from torch import nn

from .backbone import Backbone
from .neck import Neck
from .head import DetectionHead


class PersonDetector(nn.Module):
    # 입력 이미지 대비 특징 맵이 줄어든 비율, 한성곱이 한번에 이동하는 칸의 수 (기준 512x512) / dict
    strides = {
        "small": 8,  # 특징 맵 (64x64)
        "medium": 16,  # 특징 맵 (32x32)
        "large": 32,  # 특징 맵 (16x16)
    }


    def __init__(self, norm="bn"):
        super().__init__()

        self.backbone = Backbone(norm=norm)  # 이미지 특징 추출
        self.neck = Neck(norm=norm)  # 서로 다른 해상도 특징 결합
        
        # 각 숫자는 입력 이미지 채널 수(이미지 크기 x) -> head에 들어가면 stem에서 각 채널 수가 128로 변경
        self.small_head = DetectionHead(128, norm=norm)
        self.medium_head = DetectionHead(256, norm=norm)
        self.large_head = DetectionHead(512, norm=norm)

    # forward를 먼저 실행
    def forward(self, images):
        # 입력이 4개(배치, 채널, 높이, 넓이) or 채널 rgb(3채널)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"입력은 (배치, 3, 높이, 너비)여야 합니다: "
                f"{tuple(images.shape)}"
            )

        # 높이/너비가 64이상이고 32 배수인지 확인 -> 반복 축소와 Neck의 특징 결합에서 공간 크기가 맞도록 제한
        height, width = images.shape[-2:]
        if min(height, width) < 64 or height % 32 or width % 32:
            raise ValueError(
                "입력 높이와 너비는 64 이상인 32의 배수여야 합니다."
            )

        # 백본(Backbone)에서 세 해상도의 특징을 받음
        # ex) 512x512 입력 -> c3: (배치, 128, 64, 64) c4: (배치, 256, 32, 32) c5: (배치, 512, 16, 16)
        c3, c4, c5 = self.backbone(images)
        # 세 특징을 서로 결합, 공간 크기와 최종 채널 수는 유지하면서 다른 해상도에서 얻은 정보 보완
        p3, n4, n5 = self.neck(c3, c4, c5)  

        # 각 출력: [존재 logit, tx, ty, tw, th]
        # sigmoid와 좌표 변환 전의 원시 값
        return {
            "small": self.small_head(p3),
            "medium": self.medium_head(n4),
            "large": self.large_head(n5),
        }

def create_person_detection(norm="bn"):
    return PersonDetector(norm=norm)

if __name__ == "__main__":
    # gpu가 있으면 사용하고 평가 모드로 바꿈
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_person_detection().to(device)
    model.eval()

    # 가상 이미지를 한장 넣어 출력 크기 확인
    with torch.inference_mode():
        images = torch.zeros((1, 3, 512, 512), device=device)
        outputs = model(images)

    print("사용 장치:", device)

    for name, output in outputs.items():
        print(name, tuple(output.shape))

    print(
        "파라미터 수:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )