# 7주차 과제 — LangChain RAG
## 회고
6주차와 7주차 과제를 한번에 했다. 오히려 바닐라와 langchain을 한번에 하니깐 langchain이 어떤일을 하는지 정확히 알 수 있었고, 왜 쓰는지 알 것 같다.
다만 하다가 문서를 다 불러오려고 욕심내다가 무료 토큰을 다 써서 langchain으로 문서 인코딩해서 db화 하는것과 langsmith로 dataset 평가하기는 일부만 실행했다.
langchain으로 문서 인코딩하는 건 6주차 과제에서 만들었던 chroma_db를 복붙해 사용했다.
langsmith는 좀 더 고급 모델로 평가하고 싶었는데 다음엔 gemini로 돌려봐야겠다. 근데 일단 평가가 안되는게 아니라 쿼리에서 토큰을 다 쓴 거라서 하나씩 해봐야겠다.

langsmith가 .env에서 자동으로 인식되는게 신기했다.
내 모델이나 ollama랑 비교해보는것도 의미있겠다 싶었는데 우선은 langchain의 기능들을 더 공부해보고 싶다.
솔직히 비교하는건 별로 관심 없다. 비싼게 좋겠지 뭐

이번에 ai에게 한번에 코드 짜고 주석 달아줘 하는게 아니라 하나씩 기능 추가하면서 구현 어떻게 하는지 물어보니깐 좀 더 직관적으로 이해할 수 있었다. 레고블럭 하나씩 놓는것 처럼.
같은 맥락으로, 아직은 구조개선을 깊게 생각하지 않는게 좋겠다. 파일이 여러곳에 나뉘어져 있으니깐 그것들을 연결하기 위한 코드와 각각의 기능들이 헷갈려서 직관적으로 학습하기 어려웠다. 개인적으로는 한 문서에서 다 한 다음 마지막에 주석을 달면서 다시한번 보면서 '이거는 이거랑 나누면 좋겠는데?' 분리하면서 구조개선을 어떻게 해야 할 지 생각하게 되면서 구조개선도 공부되고, 짠 코드 자체도 한번 더 복습할 수 있었다.
## 개요

6주차 바닐라 RAG 파이프라인을 LangChain 기반으로 마이그레이션. FastAPI 래핑, LangSmith 트레이싱 + 데이터셋 평가까지 구현.

## 파일 구조

```
07/
├── docs/
│   └── feynman.txt       # 파인만 강의록 (06에서 복사)
├── chroma_db/            # ChromaDB 영구 저장소 (06에서 복사)
├── ingest.py             # 인덱싱: LangChain 기반 청킹 → 임베딩 → ChromaDB 저장
├── rag.py                # LCEL 체인: 검색 + 답변 생성 (ask 함수 제공)
├── main.py               # FastAPI 래핑: POST /query
├── evaluate.py           # LangSmith 데이터셋 생성 + 평가 실행
└── .env                  # GOOGLE_API_KEY, LANGCHAIN_API_KEY 등 (git 제외)
```

## 06 바닐라 vs 07 LangChain 비교

| 단계 | 06 바닐라 | 07 LangChain |
|---|---|---|
| 청킹 | `range()` 직접 구현 | `RecursiveCharacterTextSplitter` |
| 임베딩 | `client.models.embed_content()` + 배치/sleep 직접 구현 | `GoogleGenerativeAIEmbeddings` |
| VDB 저장 | `chromadb.PersistentClient` + `collection.add()` | `Chroma.from_texts()` 한 줄 |
| 검색+생성 | 임베딩 → 쿼리 → 프롬프트 조립 → LLM 호출 (직접 구현) | LCEL `retriever \| prompt \| llm \| parser` |
| API 키 | `os.getenv()` 직접 전달 | 환경변수 `GOOGLE_API_KEY` 자동 인식 |

## 진행 현황

- [x] 패키지 설치
- [x] `ingest.py` — LangChain 기반 청킹 + 임베딩 + ChromaDB 저장
- [x] `rag.py` — LCEL 체인 구성, `ask()` 함수
- [x] `main.py` — FastAPI `POST /query` 래핑
- [x] LangSmith 트레이싱 연동 (`.env` 환경변수만으로 자동 활성화)
- [x] `evaluate.py` — 데이터셋 10개 생성 + 평가 실행 (quota 소진으로 부분 완료)

## 남은 것

- [ ] Gemini quota 리셋 후 `ingest.py` 재실행 (현재는 06 chroma_db 복사본 사용 중)
- [ ] 평가 LLM을 타사 모델(Claude 등)로 교체 후 재평가

## 실행

```bash
# 인덱싱 (최초 1회 또는 문서 변경 시)
uv run ingest.py

# 서버
uv run uvicorn main:app --reload

# 평가
uv run evaluate.py
```

## 환경변수 (.env)

```
GOOGLE_API_KEY=...
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=feynman-rag
```

## 사용 라이브러리

- `langchain` + `langchain-text-splitters` — 청킹
- `langchain-google-genai` — Gemini 임베딩 + LLM
- `langchain-chroma` — ChromaDB 연동
- `langchain-core` — LCEL, 프롬프트 템플릿, 파서
- `langsmith` — 트레이싱 + 데이터셋 평가
- `fastapi` + `uvicorn` — REST API
- `python-dotenv` — API 키 관리
