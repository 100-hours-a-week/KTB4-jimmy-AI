#import chromadb
from dotenv import load_dotenv
import os
#from google import genai

#langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Chroma DB 가져오기.  langchain이 알아서

#api key 가져오기
load_dotenv()
#api_key = os.getenv("GOOGLE_API_KEY")이것도 필요없음
#gemini한테 api key 먹이기.  
    # langchain의 GoogleGenerativeAIEmbeddings가 환경변수로 GOOGLE_API_KEY 알아서 찾음


# 문서 청크 나누기 500(overlap=50)    langchain으로 간소화
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

with open("docs/feynman.txt", "r", encoding="utf-8") as f:
    text = f.read()
    chunks = splitter.split_text(text)

# 테스트 용이니깐 데이터 줄이기
chunks = chunks[:200]

# ChromaDB에 넣기 위해 배치 100으로 chunk 자르기    개고생한거 langchain으로 간소화
    ## gemini한테 임베딩 시키기
    ## chroma DB에 넣기
    ## 분당 gemini 요청 제한
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma.from_texts(chunks, embeddings, persist_directory="./chroma_db")
    

print("finished!")
