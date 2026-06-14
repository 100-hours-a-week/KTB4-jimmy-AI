"""
자기회귀(autoregressive) 문장 생성

흐름 (-> 09 언어 모델.md 1.2.2.1, 10 RNN.md 3.1 Decoder)
    1. 사용자 입력(prompt)을 토큰화해 모델에 흘려보내 hidden state를 만든다
    2. 마지막 시점의 logit -> softmax -> 다음 단어 선택 (greedy 또는 샘플링)
    3. 그 단어를 다시 입력으로 넣어 다음 단어를 또 예측 -> 반복(자기회귀)
    4. <eos> 토큰이 나오거나 max_len에 도달하면 종료

실행 방법
    python train.py        # 먼저 학습 (chatbot_model.pt 생성)
    python generate.py "나는"
"""

import sys

import torch

from data import build_vocab, tokenize, BOS, EOS, PAD, UNK
from model import LSTMLanguageModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path="chatbot_model.pt"):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    vocab = build_vocab()
    assert vocab.itos == checkpoint["vocab_itos"], "vocab이 학습 시점과 다릅니다."

    model = LSTMLanguageModel(vocab_size=len(vocab), pad_idx=vocab.stoi[PAD]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocab


@torch.no_grad()
def generate_text(model, vocab, prompt: str, max_len: int = 15, mode: str = "greedy", temperature: float = 1.0):
    """
    prompt: 사용자가 입력한 문장 (예: "나는")
    mode  : "greedy" - 매 스텝 가장 높은 확률의 단어 선택
            "sample" - 확률분포에서 샘플링 (temperature로 다양성 조절)
    """
    tokens = [vocab.stoi[BOS]] + vocab.encode(tokenize(prompt))
    input_ids = torch.tensor([tokens], device=DEVICE)

    # 1) prompt 전체를 한 번에 흘려서 hidden state 만들기
    logits, hidden = model(input_ids)
    last_logit = logits[:, -1, :]  # 마지막 시점의 logit (1, vocab_size)

    generated_ids = []
    for _ in range(max_len):
        if mode == "greedy":
            # logit이 가장 큰 단어 선택 (-> argmax)
            next_id = int(last_logit.argmax(dim=-1).item())
        else:
            # softmax로 확률분포를 만들고 그 분포에서 샘플링
            probs = torch.softmax(last_logit / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        if next_id == vocab.stoi[EOS]:
            break

        generated_ids.append(next_id)

        # 2) 방금 생성한 단어를 다시 입력으로 넣어 다음 step 진행 (자기회귀)
        last_logit, hidden = model.step(next_id, hidden, device=DEVICE)

    words = vocab.decode(generated_ids)
    return (prompt + " " + " ".join(words)).strip()


if __name__ == "__main__":
    model, vocab = load_model()

    prompt = sys.argv[1] if len(sys.argv) > 1 else "나는"

    greedy_result = generate_text(model, vocab, prompt, mode="greedy")
    sample_result = generate_text(model, vocab, prompt, mode="sample", temperature=0.8)

    print(f"입력(prompt) : {prompt}")
    print(f"greedy 생성  : {greedy_result}")
    print(f"sample 생성  : {sample_result}")
