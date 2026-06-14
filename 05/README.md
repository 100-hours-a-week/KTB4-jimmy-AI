# 5주차 과제

## 구조

```
05/
├── 1_resnet_classification.py     # 1. ResNet18로 CIFAR-10 분류 (전체 fine-tuning)
├── 2_vgg16_transfer_learning.py   # 2. VGG16 전이학습 (특징추출기 동결)
├── 3_resnet_vgg16_comparison.py   # 3. ResNet vs VGG16 성능 비교 + 그래프
├── 4_hyperparameter_tuning.py     # 4. GridSearch vs RandomSearch 튜닝 비교
└── 5_chatbot/                     # 5. 챗봇 (다음 단어 생성 + 자기회귀 생성 + FastAPI)
    ├── data.py       # 토이 말뭉치, 토큰화, Vocab
    ├── model.py       # LSTM 언어모델
    ├── train.py       # 학습 스크립트 (chatbot_model.pt 생성)
    ├── generate.py    # 자기회귀 문장 생성
    └── app.py         # FastAPI 서빙 (/generate)
```

각 파일 상단 docstring에 핵심 아이디어와 실행 방법이 적혀 있다.

## 회고

이번 과제는 단순히 "코드를 짜는" 게 아니라 ResNet, VGG16, 전이학습, GridSearch/RandomSearch, LSTM 언어모델, 자기회귀 생성, FastAPI까지 — 개념 자체를 처음 배우면서 동시에 그걸로 뭔가를 만들어야 해서 부담이 컸다.

근데 막상 해보니, 개념을 모르는 것보다 더 힘들었던 건 "모르는 채로 일단 써보는" 거였다. 보통 하나씩 이해하고 다음으로 넘어가는 바텀업 방식에 익숙한데, 이번엔 전체 코드를 먼저 돌려보고 나중에 원리를 채워나가는 탑다운 방식으로 진행했다. `model.features.parameters()`를 동결한다는 게 정확히 뭘 의미하는지, `hidden state`가 왜 그렇게 전달되는지 다 이해하지 못한 채로 일단 코드는 돌아가는 상태로 만들어야 했다.

그런데 이게 정말 더 쉬운 방법인지는 아직 잘 모르겠다. 개념을 하나씩 쌓아가는 바텀업이 느리지만 확실한 느낌이었는데, 탑다운은 빠르게 결과물은 나오지만 "왜 되는지" 모르는 부분이 계속 쌓이는 느낌이라 오히려 더 불안하다. 둘 중 뭐가 맞는 방법인지는 앞으로 개념 공부를 더 하면서 판단해봐야 할 것 같다.
