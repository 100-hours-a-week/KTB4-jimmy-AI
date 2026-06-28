from dotenv import load_dotenv
import os
from google import genai
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
#chromadb 불러오기
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="feynman")

#api key 가져오기
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

#gemini한테 api key 먹이기
client = genai.Client(api_key=api_key)

# fastapi
app = FastAPI()

class Query(BaseModel):
    prompt: str

@app.post("/query")
def query(request: Query):
    #사용자 프롬프트
    prompt = request.prompt

    # 프롬프트 임베딩
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=prompt
    )

    # DB에 Query 먹이기
    results = collection.query(
        query_embeddings=[result.embeddings[0].values],
        n_results=3
    )

    # 가져온 RAG와 합쳐 프롬프트 구성
    context = f"""다음 문서를 참고해서 질문에 답해줘.

    문서:
    {"\n\n".join(results['documents'][0])}

    질문: {prompt}
    """

    #print(context)

    # 마지막 gemini에게 질의
    answer = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=context
    )
    #print(answer.text)
    return {"answer": answer.text}