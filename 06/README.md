# 6주차 과제 — 바닐라 RAG + FastAPI

## 개요

Gemini API와 ChromaDB를 사용해 바닐라 RAG 파이프라인을 구축하고 FastAPI로 REST API로 배포.

## 파일 구조

```
06/
├── docs/
│   └── Feynman.txt       # 파인만 강의록 Vol.1~3 전체 텍스트
├── chroma_db/            # ChromaDB 영구 저장소 (자동 생성)
├── ingest.py             # 인덱싱: 문서 청킹 → 임베딩 → VDB 저장
├── main.py               # FastAPI 앱: 질문 받아 RAG 검색 → 답변 반환
└── .env                  # GEMINI_API_KEY (git 제외)
```

## RAG 파이프라인

### 인덱싱 (ingest.py)
1. `docs/Feynman.txt` 읽기
2. 500자 청크, 50자 overlap으로 분할 (`range(0, len(text), 450)`)
3. Gemini `gemini-embedding-001`로 임베딩 (배치 100개, 분당 100 제한으로 62초 대기)
4. ChromaDB `feynman` 컬렉션에 저장

### 쿼리 (main.py)
1. 사용자 질문 임베딩
2. ChromaDB에서 유사 청크 3개 검색
3. 프롬프트 구성: 검색된 문서 + 질문
4. Gemini `gemini-2.5-flash`로 답변 생성

## API

```
POST /query
{"prompt": "질문 내용"}
→ {"answer": "답변"}
```

## 실행

```bash
# 인덱싱 (최초 1회)
uv run ingest.py

# 서버 실행
uv run uvicorn main:app --reload
```

## 사용 라이브러리

- `google-genai` — Gemini 임베딩 + 텍스트 생성
- `chromadb` — 벡터 DB (로컬 영구 저장)
- `fastapi` + `uvicorn` — REST API 서버
- `python-dotenv` — API 키 관리
