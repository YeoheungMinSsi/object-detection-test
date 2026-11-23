from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# 기존 함수 중 경로/XML 읽기와 고정 분리만 재사용한다.
# 이전 30x30 정답 생성 함수와 손실 파일은 사용하지 않는다.
from prepare_dataset import load_dataset, split_dataset

# 이미지 읽어오는 함수
# 2단계로 읽어옴: 파일 -> 바이트 배열(np.fromfile) -> 이미지 배열(cv2.imdecode)
# fromfile: 파일 내용을 파이트로 읽음/   imdecode: JPEG 등의 압축 데이터를 실제 픽셀로 복원함
def read_image(path):  
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)  # shape: (높이, 너비, 3)으로 반환
    if image is None:
        raise RuntimeError(f'이미지를 읽을 수 없습니다: {path}')
    return image

# 모델 입력 형태로 변환함
def preprocess(image, image_size):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR->RGB 순으로 변경
    # resize: 이미지(rgb)의 높이,너비를 image_size로 크기 변환
    # astype(np.float32): 정수 픽셀을 소수로 변환/  255: 0~255 픽셀값을 0~1사이 소수 값으로 변환
    rgb = cv2.resize(rgb, (image_size, image_size)).astype(np.float32)/255  
    # torch.from_numpy: Numpy배열을 PyTorch 텐서로 변환
    # permute(2, 0, 1): HWC(높이, 너비, 채널)를 CHW 순서로 변경  /  contiguous: 변경된 축 순서에 맞게 메로리 배치를 정리
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    # ex) (720, 1280, 3) -> resize -> (512, 512, 3) -> 축 순서 변경 -> (3, 512, 512)

# 이미지 한 장과 정답 준비
class PersonDataset(Dataset):
    def __init__(self, samples, image_size=512, augment=False):
        # 이미지 경로와 박스 목록만 보관, 모든 이미지를 메모리에 올리지 않음
        self.samples = list(samples)
        self.image_size = image_size  # default값: 512
        self.augment = augment

    # 데이터(이미지) 개수를 반환
    def __len__(self):
        return len(self.samples)

    # 이미지를 읽어오는 함수
    def __getitem__(self, index):  # n번째 이미지를 가지고 오는 것을 설정
        sample = self.samples[index]
        image = read_image(sample['image'])

        # 이미지에 있는 박스를 텐서로 만듦
        # (사람 수, 4)의 박스 형태를 가짐 / -1은 데이터 개수에 맞춰 자동 계산, 
        # 4는 width, height 각 2개(박스 모서리 좌표) - (x_min, y_min, x_max, y_max)
        boxes = torch.tensor(sample['boxes'], dtype=torch.float32).reshape(-1, 4)

        # 데이터 증강(좌우 반전)
        if self.augment:  # augment가 True일때만 수행
            if random.random() < 0.5:  # 좌우 반전이 50% 확률로 적용
                image = cv2.flip(image, 1)  # 이미지가 좌우반전
                original = boxes.clone()  # original에 박스의 값(-1, 4)을 복사함
                boxes[:, 0] = 1-original[:, 2]  # boxes[:, 0](새 xmin) = 1-original[:, 2](1 - 기존 xmax)
                boxes[:, 2] = 1-original[:, 0]  # boxes[:, 2](새 xmax) = 1-original[:, 0](1 - 기존 xmin)
                # [;, 0]는 모든 사람의 xmin, [;, 2]는 모든 사람의 xmax를 뜻함
                # ex) 기존x 범위: 0.1 ~ 0.4 -> 반전 x 범위: 0.6~0.9

            # 데이터 증강(대비, 밝기)
            # 기존 값에서 대비기준(기존 값 x 0.8 ~ 1.2), 밝기기준(기존 값 x -20 ~ 20)
            contrast, brightness = random.uniform(0.8, 1.2), random.uniform(-20, 20) # contrast: 대비, brigthness: 밝기
            # 새 픽셀: 기존 픽셀 값 x 대비 계수 + 밝기 변화량 / 범위를 0 ~ 255로 제한
            image = np.clip(image.astype(np.float32)*contrast+brightness, 0, 255).astype(np.uint8)

        # 두 값을 반환
        return preprocess(image, self.image_size), {
            'boxes': boxes,  # (사람 수, 4(박스 모서리 값))
            'path': str(sample['image'])  # 이미지 경로
            }

# batch에는 이미지와 정답의 쌍이 들어옮 -> [(image1, target1), (image2, target2)]: 2개의 쌍
def collate_samples(batch):
    # 이미지끼리 정답끼리 분리해서 변수에 넣음
    # ex) (3, 512, 512) 이미지 2개 -> (2, 3, 512, 512)
    # target[0]['boxes'].shape: (2, 4) 사람 2명 / target[1]['boxes'].shape: (4, 4) 사람 4명
    images, targets = zip(*batch)  
    return torch.stack(images), list(targets)  # 이미지는 스택으로 배치수로 쌓고, 정답은 리스트화 해서 넣음


def move_targets(targets, device):
    return [
        # box 텐서는 장치로 이동(gpu or cpu), 정답(target) 딕셔너리는 유지
        {**target, 'boxes': target['boxes'].to(device)}  # device: cpu or gpu(cuda)
        for target in targets
        ]

# 평가에 사용, 학습할 때 사용했던 이미지 목록을 복원하는 함수(테스트 데이터를 가지고 와서 다시 평가)
def load_manifest_samples(checkpoint_path, split, dataset=None):
    import json
    folder = Path(checkpoint_path).resolve().parent  # 체크포이트가 저장된 폴더 찾고, conig/split.sjon 파일을 찾음
    config = json.loads((folder/'config.json').read_text(encoding='utf-8-sig'))  # config.json 파일을 읽고, 딕셔너리로 바꿈
    # 함수에 새 경로 전달하면 그 경로 사용, 전달하지 않으면 학습 당시 저장된 경로 사용
    root = Path(dataset) if dataset is not None else Path(config['dataset'])  
    samples = load_dataset(root)  # root: 데이터셋의 기준 폴더
    by_path = {str(s['image'].resolve()).casefold(): s for s in samples}  # 전체 경로 -> 해당 샘플의 대응표
    # 파일명 -> 해당 샘플 목록( ex) a.jpg:[sample_a]  /  목록으로 보관하는 이유: 같은 파일명을 가진 샘플이 여러 개일 가능성 확인
    by_name = {}
    for sample in samples:
        by_name.setdefault(sample['image'].name, []).append(sample)
    # 학습 당시 저장한 분리 기록을 읽음
    # 데이터를 새로 무작위 분리하지 않고 저장된 목록을 따름
    manifest = json.loads((folder/'split.json').read_text(encoding='utf-8-sig'))  # split:{train:[~], val:[~], test:[~]}
    selected = []
    for path in manifest[split]:
        # 저장된 경로에 해당하는 현재 샘플을 찾음
        sample = by_path.get(str(Path(path).resolve()).casefold())  # 찾지 못하면 파일명만 추출해서 다시 찾음
        if sample is None:
            matches = by_name.get(path.replace('\\', '/').rsplit('/', 1)[-1], [])
            if len(matches) != 1:
                raise ValueError(f'이미지 연결이 없거나 중복됩니다: {path}')
            sample = matches[0]
        selected.append(sample)
    if not selected:
        raise ValueError(f'{split} 그룹이 비어 있습니다.')
    return selected, config
