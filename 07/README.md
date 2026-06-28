# 7주차 과제 — LangChain RAG (진행 중)

## 개요

6주차 바닐라 RAG 파이프라인을 LangChain 기반으로 마이그레이션.  
일일 임베딩 quota 한도(1000건/일) 소진으로 인덱싱 단계에서 중단, 이후 작업 예정.

## 파일 구조

```
07/
├── docs/
│   └── feynman.txt       # 파인만 강의록 (06에서 복사)
├── ingest.py             # 인덱싱: LangChain 기반 청킹 → 임베딩 → ChromaDB 저장
├── rag.py                # (미완성) 검색 + 답변 생성 체인
├── main.py               # (미완성) FastAPI 래핑
└── .env                  # GOOGLE_API_KEY (git 제외)
```

## 06 바닐라 vs 07 LangChain 비교

| 단계 | 06 바닐라 | 07 LangChain |
|---|---|---|
| 청킹 | `range()` 직접 구현 | `RecursiveCharacterTextSplitter` |
| 임베딩 | `client.models.embed_content()` + 배치/sleep 직접 구현 | `GoogleGenerativeAIEmbeddings` |
| VDB 저장 | `chromadb.PersistentClient` + `collection.add()` | `Chroma.from_texts()` 한 줄 |
| API 키 | `os.getenv()` 직접 전달 | 환경변수 `GOOGLE_API_KEY` 자동 인식 |

## 진행 현황

- [x] 패키지 설치 (`langchain`, `langchain-google-genai`, `langchain-chroma` 등)
- [x] `ingest.py` 작성 완료 (LangChain 기반)
- [ ] Gemini 일일 quota 소진 → 인덱싱 미완료 (내일 재시도)
- [ ] `rag.py` — LCEL 체인 구성
- [ ] `main.py` — FastAPI 래핑
- [ ] LangSmith 트레이싱 + 데이터셋 평가

## 다음 할 것

1. 06의 `chroma_db/` 복사해서 인덱싱 없이 쿼리 먼저 테스트
2. `rag.py` — LangChain LCEL로 검색 + 생성 체인 구성
3. `main.py` — FastAPI 래핑
4. LangSmith 연동

## 실행 (예정)

```bash
# 인덱싱
uv run ingest.py

# 서버
uv run uvicorn main:app --reload
```

## 사용 라이브러리

- `langchain` + `langchain-text-splitters` — 청킹
- `langchain-google-genai` — Gemini 임베딩 + LLM
- `langchain-chroma` — ChromaDB 연동
- `fastapi` + `uvicorn` — REST API
- `python-dotenv` — API 키 관리
