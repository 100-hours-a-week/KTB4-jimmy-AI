"""
과제 1. ResNet 모델을 불러와 새로운 이미지 데이터셋을 분류하세요.

핵심 아이디어
- ImageNet으로 미리 학습된 ResNet18을 불러온다 (사전학습 모델, → 08 사전학습 모델.md)
- 마지막 분류층(fc)만 우리 데이터셋의 클래스 수에 맞게 새로 갈아끼운다
- 이번 과제에서는 "전체 네트워크를 다시 학습"하는 fine-tuning 방식을 쓴다
  (과제 2의 VGG16 전이학습 - 일부 동결 - 과 비교 대상이 되도록 일부러 다르게 구성)

데이터셋
- torchvision이 기본 제공하는 CIFAR-10 (32x32 컬러 이미지, 10개 클래스)을
  "새로운 이미지 데이터셋"으로 사용한다. 처음 실행 시 자동 다운로드됨.
- 다른 이미지 데이터셋으로 바꾸려면 DATA_DIR에 ImageFolder 구조
  (클래스별 폴더)로 두고 datasets.ImageFolder(DATA_DIR, transform=...)를 쓰면 된다.

실행 방법
    pip install torch torchvision matplotlib
    python 1_resnet_classification.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ------------------------------------------------------------
# 0. 설정
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10          # CIFAR-10 클래스 수
BATCH_SIZE = 64
EPOCHS = 5                # 시간 여유 있으면 늘려도 됨
LR = 1e-4

# ------------------------------------------------------------
# 1. 데이터 준비
# ------------------------------------------------------------
# ResNet은 ImageNet(224x224, RGB)으로 학습됐기 때문에,
# 입력 이미지도 224x224로 맞추고 ImageNet의 평균/표준편차로 정규화해야
# 사전학습된 가중치가 의미를 갖는다.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),          # CIFAR-10(32x32) -> 224x224로 업스케일
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ------------------------------------------------------------
# 2. 사전학습 ResNet18 불러와서 출력층 교체
# ------------------------------------------------------------
# weights="IMAGENET1K_V1" : ImageNet 1000개 클래스로 학습된 가중치를 그대로 불러옴
model = models.resnet18(weights="IMAGENET1K_V1")

# ResNet18의 마지막 층은 fc: Linear(in_features=512, out_features=1000)
# -> 우리 데이터셋의 클래스 수(10개)에 맞게 새 Linear층으로 교체
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)

model = model.to(DEVICE)

# ------------------------------------------------------------
# 3. 손실함수 / 옵티마이저
# ------------------------------------------------------------
# 다중 클래스 분류 -> CrossEntropyLoss
# (내부적으로 softmax + log + NLL을 한번에 처리 -> 09 언어 모델.md의
#  "logit -> softmax -> 확률" 흐름과 동일한 구조)
criterion = nn.CrossEntropyLoss()

# 이번 과제는 "전체 fine-tuning" -> 모든 파라미터를 옵티마이저에 전달
optimizer = optim.Adam(model.parameters(), lr=LR)


# ------------------------------------------------------------
# 4. 학습/평가 함수
# ------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)            # logit (배치, NUM_CLASSES)
        loss = criterion(outputs, labels)
        loss.backward()                    # 역전파 (-> 04 ANN.md / 05 학습.md)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)      # logit이 가장 큰 클래스 선택 (greedy)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ------------------------------------------------------------
# 5. 학습 루프
# ------------------------------------------------------------
if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        print(
            f"[Epoch {epoch}/{EPOCHS}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    # 학습된 가중치 저장 (과제 3 비교에서 재사용 가능)
    torch.save(model.state_dict(), "resnet18_cifar10.pt")
    print("저장 완료: resnet18_cifar10.pt")
