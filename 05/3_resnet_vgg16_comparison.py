"""
과제 3. 동일한 데이터셋에서 ResNet과 VGG16을 각각 학습시켜 성능을 비교하세요.

구성
- 과제 1(ResNet18 fine-tuning)과 과제 2(VGG16 전이학습)를 "같은 데이터,
  같은 epoch 수, 같은 배치 크기"로 다시 실행하면서 epoch별
  train/test loss, accuracy를 기록한다.
- 학습이 끝나면 matplotlib으로 두 모델의 학습 곡선을 한 그래프에 겹쳐
  그리고, 최종 test accuracy를 막대그래프로 비교한다.

주의
- 실제 비교가 의미 있으려면 두 모델 모두 "같은 조건"이어야 하므로,
  여기서는 둘 다 동일한 옵티마이저 종류(Adam)와 동일한 epoch 수를 사용한다.
  (학습률은 모델 구조 특성상 다르게 둘 수 있다 - 과제 2 참고)

실행 방법
    pip install torch torchvision matplotlib
    python 3_resnet_vgg16_comparison.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 0. 설정
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 5

# 시간이 부족하면 SUBSET_SIZE를 줄여서 빠르게 돌려볼 수 있다 (None = 전체 사용)
SUBSET_SIZE = None  # 예: 5000 으로 두면 학습 데이터 5000장만 사용

# ------------------------------------------------------------
# 1. 데이터 준비 (1, 2번 과제와 동일)
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

if SUBSET_SIZE is not None:
    train_dataset = Subset(train_dataset, range(SUBSET_SIZE))
    test_dataset = Subset(test_dataset, range(SUBSET_SIZE // 5))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ------------------------------------------------------------
# 2. 모델 생성 함수
# ------------------------------------------------------------
def build_resnet18():
    """과제 1과 동일: 전체 fine-tuning"""
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model, model.parameters(), 1e-4


def build_vgg16():
    """과제 2와 동일: 특징 추출기 동결 + classifier 마지막 층만 학습"""
    model = models.vgg16(weights="IMAGENET1K_V1")
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return model, trainable_params, 1e-3


# ------------------------------------------------------------
# 3. 학습/평가 함수 (1, 2번과 동일한 구조)
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
        correct += (outputs.argmax(dim=1) == labels).sum().item()
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
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def run_experiment(name, build_fn):
    print(f"\n===== {name} 학습 시작 =====")
    model, params, lr = build_fn()
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=lr)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        print(
            f"[{name}][Epoch {epoch}/{EPOCHS}] "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"test_loss={te_loss:.4f} test_acc={te_acc:.4f}"
        )
    return history


# ------------------------------------------------------------
# 4. 두 모델 학습 + 결과 비교
# ------------------------------------------------------------
if __name__ == "__main__":
    resnet_history = run_experiment("ResNet18", build_resnet18)
    vgg_history = run_experiment("VGG16", build_vgg16)

    epochs_range = range(1, EPOCHS + 1)

    # 4-1. 학습 곡선 비교 (loss, accuracy)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs_range, resnet_history["test_loss"], marker="o", label="ResNet18")
    axes[0].plot(epochs_range, vgg_history["test_loss"], marker="o", label="VGG16")
    axes[0].set_title("Test Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs_range, resnet_history["test_acc"], marker="o", label="ResNet18")
    axes[1].plot(epochs_range, vgg_history["test_acc"], marker="o", label="VGG16")
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("resnet_vs_vgg16_curves.png")
    print("\n그래프 저장: resnet_vs_vgg16_curves.png")

    # 4-2. 최종 성능 막대그래프 + 표
    plt.figure(figsize=(5, 4))
    final_accs = [resnet_history["test_acc"][-1], vgg_history["test_acc"][-1]]
    plt.bar(["ResNet18\n(fine-tuning)", "VGG16\n(transfer learning)"], final_accs, color=["#4C72B0", "#DD8452"])
    plt.ylabel("Final Test Accuracy")
    plt.title("ResNet18 vs VGG16")
    plt.tight_layout()
    plt.savefig("resnet_vs_vgg16_final.png")
    print("그래프 저장: resnet_vs_vgg16_final.png")

    print("\n===== 최종 비교 =====")
    print(f"ResNet18 (전체 fine-tuning) test_acc = {resnet_history['test_acc'][-1]:.4f}")
    print(f"VGG16    (전이학습, 특징추출 동결) test_acc = {vgg_history['test_acc'][-1]:.4f}")
