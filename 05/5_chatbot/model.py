"""
다음 단어 예측 모델 (LSTM 기반 언어 모델)

구조 (-> 09 언어 모델.md 1.2.2.1 "생성")
    입력 단어들 -> Embedding -> LSTM(hidden state h_t 누적) -> 출력층(W_out)
    -> logit(사전 크기) -> softmax -> 단어 확률분포

즉 NPLM/RNN에서 본 "(컨텍스트 표현) -> 출력층 -> logit -> softmax -> 선택"
구조를 그대로 LSTM으로 구현한 것이다.
"""

import torch
import torch.nn as nn


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, pad_idx: int = 0):
        super().__init__()
        # 1.1.2 Word Embedding: 단어 -> 밀집 벡터 (학습 대상)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # hidden state h_t = "지금까지 본 문맥을 압축한 표현" (-> 1.2 베이즈 관점의 prior)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # 출력층 W_out: h_t -> 사전 크기만큼의 logit
        #   z = W_out h_t + b   (-> 09 언어 모델.md 식 그대로)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len) 토큰 인덱스
        반환:
            logits: (batch, seq_len, vocab_size) - 각 시점의 다음 단어 logit
            hidden: LSTM의 (h, c) - 자기회귀 생성 시 다음 스텝에 이어서 전달
        """
        emb = self.embedding(x)                  # (batch, seq_len, embed_dim)
        output, hidden = self.lstm(emb, hidden)  # output: (batch, seq_len, hidden_dim)
        logits = self.fc_out(output)             # (batch, seq_len, vocab_size)
        return logits, hidden

    @torch.no_grad()
    def step(self, token_id: int, hidden=None, device="cpu"):
        """
        자기회귀 생성을 위한 한 스텝 forward.
        토큰 1개를 넣고, 다음 단어의 logit과 갱신된 hidden state를 반환한다.
        """
        x = torch.tensor([[token_id]], device=device)  # (1, 1)
        logits, hidden = self.forward(x, hidden)
        return logits[:, -1, :], hidden  # 마지막(=유일한) 시점의 logit, shape (1, vocab_size)
