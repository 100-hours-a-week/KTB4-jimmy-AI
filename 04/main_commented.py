# ================================================================
# 과제 1 — 가상 데이터셋 생성 & 학습/검증/테스트 분할
# ================================================================

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1000개 샘플, 특성 10개, 이진 분류(n_classes=2) 가상 데이터 생성
# random_state=42 → 실행할 때마다 같은 데이터가 나오도록 고정
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# 1차 분할: 전체 → 학습(60%) + 임시(40%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

# 2차 분할: 임시(40%) → 검증(20%) + 테스트(20%)
# test_size=0.5 → 임시의 절반씩
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# ================================================================
# 과제 2 — K-NN 분류
# ================================================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# n_neighbors=5 → 새 데이터 분류 시 가장 가까운 5개 이웃의 다수결로 결정
model = KNeighborsClassifier(n_neighbors=5)

# fit: 학습 데이터를 그대로 메모리에 저장 (K-NN은 별도 학습 과정 없음)
model.fit(X_train, y_train)

# predict: 테스트 데이터 각각에 대해 거리 계산 → 이웃 5개 찾기 → 다수결
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))  # 0.845


# ================================================================
# 과제 3 — 4가지 알고리즘 비교
# ================================================================

from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# 비교할 모델 딕셔너리로 묶어서 루프 한 번에 처리
models = {
    "Perceptron":     Perceptron(),             # 단층 퍼셉트론 — 선형 경계만
    "SVM":            SVC(),                    # 마진 최대화 — 커널로 비선형 가능
    "Random Forest":  RandomForestClassifier(), # 여러 트리 앙상블 → 다수결
    "Naive Bayes":    GaussianNB(),             # 특성 독립 가정 → 확률 곱으로 분류
}

for name, model in models.items():
    model.fit(X_train, y_train)          # 각 알고리즘 방식으로 학습
    y_pred = model.predict(X_test)
    print(f"{name}: {accuracy_score(y_test, y_pred):.3f}")


# ================================================================
# 과제 4 — 데이터 증강 (Data Augmentation)
# ================================================================

import numpy as np

# 학습 데이터에 가우시안 노이즈(평균 0, 표준편차 0.1) 추가 → 새 샘플 생성
# 실제 이미지에서는 밝기/회전 등으로 증강; 여기선 수치 노이즈로 대체
X_noise = X_train + np.random.normal(0, 0.1, X_train.shape)

# 원본 + 노이즈 버전을 세로로 합쳐서 학습 데이터 2배로 늘림
X_aug = np.vstack([X_train, X_noise])  # (600, 10) → (1200, 10)
y_aug = np.hstack([y_train, y_train])  # 정답 레이블도 동일하게 2배

for name, model in models.items():
    model.fit(X_aug, y_aug)
    y_pred = model.predict(X_test)
    print(f"double {name}: {accuracy_score(y_test, y_pred):.3f}")
# 결과가 원본과 비슷한 이유: 가상 데이터라 실제 노이즈가 없어 증강 효과가 미미


# ================================================================
# 과제 5 — 활성화 함수 시각화
# ================================================================

import matplotlib.pyplot as plt

# -5 ~ 5 사이를 100등분한 x값 배열
x = np.linspace(-5, 5, 100)

# 활성화 함수 직접 구현
def relu(x):    return np.maximum(0, x)           # 0 이하 → 0, 초과 → 그대로
def sigmoid(x): return 1 / (1 + np.exp(-x))      # 모든 값을 0~1로 압축
def tanh(x):    return np.tanh(x)                 # 모든 값을 -1~1로 압축

# 세 함수를 한 그래프에 겹쳐서 비교
plt.plot(x, relu(x),    label='ReLU')
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x),    label='Tanh')
plt.legend()
plt.title('Activation Functions')
plt.grid(True)
plt.show()


# ================================================================
# 과제 6 — MLP (PyTorch) — 비선형 데이터셋 분류
# ================================================================

import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# make_moons: 반달 두 개 모양의 비선형 데이터 — 직선으로 분리 불가
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# StandardScaler: 평균 0, 표준편차 1로 정규화 → 학습 안정성 향상
scaler = StandardScaler()
X = scaler.fit_transform(X)

# numpy 배열을 PyTorch 텐서로 변환 (모델이 텐서만 받음)
X_tensor = torch.FloatTensor(X)   # 실수형
y_tensor = torch.LongTensor(y)    # 정수형 (CrossEntropyLoss가 정수 레이블 요구)

# MLP 모델 정의
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32),   # 입력 2개(x좌표, y좌표) → 32개 뉴런
            nn.ReLU(),          # 비선형 활성화
            nn.Linear(32, 16),  # 32 → 16으로 압축
            nn.ReLU(),
            nn.Linear(16, 2)    # 최종 출력: 클래스 수(2)만큼
            # CrossEntropyLoss가 내부적으로 Softmax 처리하므로 여기선 생략
        )

    def forward(self, x):
        return self.network(x)  # 순전파: 입력이 층을 순서대로 통과

model = MLP()
loss_fn   = nn.CrossEntropyLoss()               # 다중 분류 손실함수
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 학습률 0.001

# 학습 루프
for epoch in range(100):
    optimizer.zero_grad()           # ① 이전 배치 gradient 초기화
    output = model(X_tensor)        # ② 순전파
    loss   = loss_fn(output, y_tensor)  # ③ 손실 계산
    loss.backward()                 # ④ 역전파 (각 가중치의 gradient 계산)
    optimizer.step()                # ⑤ 가중치 업데이트

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1} | loss: {loss.item():.4f}")

# 평가 (gradient 계산 불필요 → no_grad로 메모리 절약)
model.eval()
with torch.no_grad():
    pred = model(X_tensor).argmax(dim=1)         # 가장 높은 확률의 클래스 선택
    acc  = (pred == y_tensor).float().mean()
    print(f"Accuracy: {acc:.3f}")


# ================================================================
# 과제 7 — CNN (PyTorch) — MNIST 손글씨 분류
# ================================================================

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ToTensor: PIL 이미지(0~255)를 0~1 범위 텐서로 변환
transform = transforms.Compose([transforms.ToTensor()])

# MNIST: 28×28 흑백 손글씨 숫자 이미지, 10클래스(0~9)
# download=True → 없으면 자동 다운로드
train_data   = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_data    = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader: 배치 단위로 잘라서 공급 / shuffle=True → 매 epoch마다 순서 섞기
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature Extraction — Conv + Pooling으로 공간 특징 추출
        self.conv = nn.Sequential(
            # Conv2d(입력채널, 출력채널, kernel_size, padding)
            # padding=1 + kernel_size=3 → H×W 크기 유지
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 1채널(흑백) → 16채널 / 28×28 유지
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 28×28 → 14×14 (2×2 구역 최댓값만 유지)

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 16채널 → 32채널 / 14×14 유지
            nn.ReLU(),
            nn.MaxPool2d(2)                               # 14×14 → 7×7
        )
        # Classification — Flatten 후 FC층으로 분류
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),  # 32채널 × 7×7 = 1568개 → 128개로 압축
            nn.ReLU(),
            nn.Linear(128, 10)           # 128 → 클래스 10개 (0~9)
        )

    def forward(self, x):
        x = self.conv(x)              # Feature Extraction
        x = x.view(x.size(0), -1)    # Flatten: (batch, 32, 7, 7) → (batch, 1568)
        return self.fc(x)             # Classification

device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = CNN().to(device)
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    # 학습 모드 (Dropout 등 학습 전용 레이어 활성화)
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()                      # ① gradient 초기화
        loss = loss_fn(model(X_batch), y_batch)    # ② 순전파 + ③ 손실 계산
        loss.backward()                            # ④ 역전파
        optimizer.step()                           # ⑤ 가중치 업데이트

    # 평가 모드 (Dropout 비활성화 → 전체 뉴런 사용)
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred     = model(X_batch).argmax(dim=1)
            correct += (pred == y_batch).sum().item()

    print(f"Epoch {epoch+1} | Accuracy: {correct / len(test_data):.3f}")
