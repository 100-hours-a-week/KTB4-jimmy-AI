"""
과제 2. 이미지 데이터셋과 사전 훈련된 VGG16 모델을 가져와 전이 학습을 수행하세요.

핵심 아이디어 (08 사전학습 모델.md - Transfer Learning)
- VGG16의 conv(특징 추출) 부분은 ImageNet에서 이미 "이미지의 일반적인 특징
  (선, 질감, 모양 등)"을 학습해 두었다 -> 이 부분은 그대로 동결(freeze)
- 우리 데이터셋의 클래스를 구분하는 건 마지막 분류기(classifier)만 새로
  학습시킨다 -> "특징 추출기는 재사용, 분류기만 새로 학습"이 전이학습의 핵심

과제 1(ResNet)과의 차이
- 과제 1: 전체 네트워크를 fine-tuning (모든 파라미터 업데이트)
- 과제 2: conv 특징 추출기는 동결(freeze), 새로 붙인 분류기만 학습
  -> 학습 파라미터 수가 훨씬 적어 더 빠르고, 데이터가 적을 때 과적합도 줄어든다

실행 방법
    pip install torch torchvision
    python 2_vgg16_transfer_learning.py
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
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3   # 새로 추가한 층만 학습하므로 ResNet 예제보다 큰 학습률 사용 가능

# ------------------------------------------------------------
# 1. 데이터 준비 (과제 1과 동일한 CIFAR-10 사용 -> 과제 3에서 직접 비교 가능)
# ------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ------------------------------------------------------------
# 2. 사전학습 VGG16 불러오기
# ------------------------------------------------------------
model = models.vgg16(weights="IMAGENET1K_V1")

# 2-1. 특징 추출기(features)는 전부 동결 -> 역전파 시 가중치 업데이트 안 됨
for param in model.features.parameters():
    param.requires_grad = False

# 2-2. classifier 구조 확인
#   VGG16.classifier = Sequential(
#     0: Linear(25088, 4096), 1: ReLU, 2: Dropout,
#     3: Linear(4096, 4096),  4: ReLU, 5: Dropout,
#     6: Linear(4096, 1000)   <- 이 마지막 층만 우리 클래스 수로 교체
#   )
in_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features, NUM_CLASSES)
# 새로 만든 Linear층은 기본적으로 requires_grad=True (학습 대상)

model = model.to(DEVICE)

# ------------------------------------------------------------
# 3. 손실함수 / 옵티마이저
# ------------------------------------------------------------
criterion = nn.CrossEntropyLoss()

# 동결되지 않은(=requires_grad=True) 파라미터만 옵티마이저에 전달
# -> 사실상 새로 갈아끼운 classifier[6]만 학습됨
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params, lr=LR)

print(f"전체 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
print(f"학습 대상 파라미터 수: {sum(p.numel() for p in trainable_params):,}")


# ------------------------------------------------------------
# 4. 학습/평가 함수 (1_resnet_classification.py와 동일한 구조)
# ------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
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

    torch.save(model.state_dict(), "vgg16_cifar10.pt")
    print("저장 완료: vgg16_cifar10.pt")
