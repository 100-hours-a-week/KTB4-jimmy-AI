from dotenv import load_dotenv
#import os
#from google import genai
#import chromadb
#langchain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#api key 가져오기
load_dotenv()

#chromadb 불러오기
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="feynman"
)
def ask(question: str) -> str:
    # 사용자 프롬프트 받아서
    # 프롬프트 임베딩
    # DB에 Query 먹이기
    # 가져온 RAG와 합쳐 프롬프트 구성
    # 마지막 gemini에게 질의
    # 과정을 langchain에선 프롬프트 템플릿 만들고, 체인 설정하고 answer로 chain 실행 한번 돌려서 끝

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = ChatPromptTemplate.from_template("""
    다음 문서를 참고해서 질문에 답해줘.

    문서: {context}

    질문: {question}
    """)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    return answer
