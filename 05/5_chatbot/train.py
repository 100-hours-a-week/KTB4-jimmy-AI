"""
다음 단어 예측 모델 학습 스크립트

실행 방법
    pip install torch
    python train.py
-> 학습이 끝나면 chatbot_model.pt 에 모델 가중치 + vocab을 저장한다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from data import build_vocab, make_training_pairs, PAD
from model import LSTMLanguageModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200          # 토이 데이터셋이라 epoch을 많이 돌려야 패턴이 잡힌다
LR = 1e-2


def collate(pairs, pad_idx):
    """길이가 다른 시퀀스들을 가장 긴 길이에 맞춰 padding"""
    inputs = [torch.tensor(x, dtype=torch.long) for x, _ in pairs]
    targets = [torch.tensor(y, dtype=torch.long) for _, y in pairs]

    inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    targets = pad_sequence(targets, batch_first=True, padding_value=pad_idx)
    return inputs, targets


def main():
    vocab = build_vocab()
    pairs = make_training_pairs(vocab)
    pad_idx = vocab.stoi[PAD]

    inputs, targets = collate(pairs, pad_idx)
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

    model = LSTMLanguageModel(vocab_size=len(vocab), pad_idx=pad_idx).to(DEVICE)

    # padding 위치는 손실 계산에서 제외 (-> ignore_index)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()

        logits, _ = model(inputs)  # (batch, seq_len, vocab_size)

        # CrossEntropyLoss는 (N, C) vs (N,) 형태를 기대 -> 배치*시퀀스를 펼침
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"[Epoch {epoch}/{EPOCHS}] loss={loss.item():.4f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_itos": vocab.itos,
        },
        "chatbot_model.pt",
    )
    print("저장 완료: chatbot_model.pt")


if __name__ == "__main__":
    main()
