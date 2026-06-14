"""
챗봇용 토이 말뭉치 + Vocabulary 구성

실제 과제에서는 이 CORPUS를 더 큰 한국어/영어 대화 데이터셋으로 교체하면 된다.
구조(Vocab, 토큰화 방식)는 그대로 재사용 가능하다.

토큰화: 띄어쓰기 기준 (한국어 형태소 분석기를 쓰면 더 좋지만,
        의존성을 최소화하기 위해 공백 분리 + 문장부호 분리만 적용)
"""

import re
from collections import Counter

# ------------------------------------------------------------
# 1. 토이 말뭉치
# ------------------------------------------------------------
CORPUS = [
    "나는 밥을 먹었다",
    "나는 밥을 좋아한다",
    "나는 학교에 간다",
    "너는 학교에 간다",
    "그는 영화를 본다",
    "그녀는 음악을 듣는다",
    "나는 음악을 듣는다",
    "오늘 날씨가 좋다",
    "오늘 기분이 좋다",
    "내일 시험을 본다",
    "나는 코딩을 좋아한다",
    "그는 코딩을 잘한다",
    "나는 영화를 좋아한다",
    "우리는 같이 밥을 먹었다",
    "너는 코딩을 배운다",
    "나는 파이썬을 배운다",
    "그녀는 파이썬을 잘한다",
    "오늘 저녁에 밥을 먹는다",
    "나는 친구를 만난다",
    "그는 친구와 영화를 본다",
]

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD, UNK, BOS, EOS = SPECIAL_TOKENS


def tokenize(text: str):
    """공백 기준 토큰화. 필요하면 KoNLPy 형태소 분석기로 교체."""
    text = text.strip()
    return re.findall(r"[\w가-힣]+", text)


class Vocab:
    """단어 <-> 인덱스 매핑 (1.1.1 One-hot / 1.1.2 Word Embedding의 "사전" 역할)"""

    def __init__(self, sentences):
        counter = Counter()
        for sent in sentences:
            counter.update(tokenize(sent))

        # 특수 토큰을 앞에 고정 배치
        self.itos = list(SPECIAL_TOKENS) + sorted(counter.keys())
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(tok, self.stoi[UNK]) for tok in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


def build_vocab():
    return Vocab(CORPUS)


def make_training_pairs(vocab: Vocab):
    """
    각 문장을 <bos> ... <eos> 로 감싼 뒤, 한 칸씩 어긋난
    (input, target) 시퀀스 쌍을 만든다 (teacher forcing).

    예) "나는 밥을 먹었다"
        -> ids = [<bos>, 나는, 밥을, 먹었다, <eos>]
        -> input  = [<bos>, 나는, 밥을, 먹었다]
        -> target = [나는, 밥을, 먹었다, <eos>]

    즉 매 시점 t에서 "input[t]까지 봤을 때 target[t](=다음 단어)를
    맞히도록" 모든 시점을 한 번에 학습한다 (-> 10 RNN.md의
    "매 스텝 encode+decode" 구조와 동일).
    """
    pairs = []
    for sent in CORPUS:
        ids = [vocab.stoi[BOS]] + vocab.encode(tokenize(sent)) + [vocab.stoi[EOS]]
        input_seq = ids[:-1]
        target_seq = ids[1:]
        pairs.append((input_seq, target_seq))
    return pairs
