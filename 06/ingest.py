import chromadb
from dotenv import load_dotenv
import os
from google import genai

# Chroma DB 가져오기
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="feynman")

#api key 가져오기
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

#gemini한테 api key 먹이기
client = genai.Client(api_key=api_key)

# 문서 청크 나누기 500(overlap=50)
with open("docs/feynman.txt", "r", encoding="utf-8") as f:
    text = f.read()
    chunks=[]
    for i in range(0, len(text), 450):
        chunk = text[i : i+500]
        chunks.append(chunk)

# 테스트 용이니깐 데이터 줄이기
chunks = chunks[:200]

# ChromaDB에 넣기 위해 배치 100으로 chunk 자르기
import time
for i in range(len(chunks)//100 + 1):
    batch = chunks[i*100 : (i+1)*100] #넘어가면 자동으로 잘라줌
    ## batch가 비어있으면 건너뛰기
    if not batch:
        break
    # gemini한테 임베딩 시키기
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=batch
    )

    #chroma DB에 넣기
        ##documents = ["청크0", "청크1", "청크2", ...]
        ##embeddings = [[0.007, ...], [0.013, ...], [0.002, ...], ...]
        ##ids        = ["0000",       "0001",       "0002",       ...]
    embeddings = [e.values for e in result.embeddings]
    ids=[str(i*100+j).zfill(4) for j in range(len(batch))]
    collection.add(
        documents=batch,
        embeddings=embeddings,
        ids=ids
    )
    print(f"배치 {i+1} 저장 완료")
    # 분당 gemini 요청 제한
    time.sleep(62)

print("finished!")
