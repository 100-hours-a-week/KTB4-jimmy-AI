from dotenv import load_dotenv
#from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END  

from typing_extensions import TypedDict
from langchain_core.documents import Document

from pydantic import BaseModel, Field
from typing import Literal

#api key 가져오기
load_dotenv()

#chromadb 불러오기
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="feynman"
)
#LangGraph State 구성
class State(TypedDict):
    question: str
    context: list[Document]
    answer: str
    fix_needed: bool
    what_to_fix: list[str] 

def retrieve(state: State) -> dict:
    # state에서 question 꺼내서
    q=state["question"]
    # vectorstore에서 검색하고
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(q)
    # context 반환
    return {"context": docs}

def generate(state: State) -> dict:

    if state.get("fix_needed", False):
        prompt_template = ChatPromptTemplate.from_template("""
        다음 문서를 참고해서 질문에 답해줘.

        문서: {context}

        질문: {question}
        
        답변: {answer}
                                                        
        이 부분이 틀렸어. 다시 수정해줘. 
                                                        
        고칠 부분: {what_to_fix}
        """)

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        chain = (prompt_template | llm | StrOutputParser())

        answer = chain.invoke({
            "context": state["context"], 
            "question": state["question"], 
            "answer": state["answer"],
            "what_to_fix": state["what_to_fix"]
            })

    else:
        prompt_template = ChatPromptTemplate.from_template("""
        다음 문서를 참고해서 질문에 답해줘.

        문서: {context}

        질문: {question}
        """)

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        chain = (prompt_template | llm | StrOutputParser())

        answer = chain.invoke({"context": state["context"], "question": state["question"]})

    
    return {"answer" : answer, "fix_needed" : False}


class verified(BaseModel):
    fix_needed: bool = Field(description="answer가 수정이 필요한지")
    what_to_fix: list[str] = Field(description="어떤 부분을 고쳐야 하는지")

def verify(state: State) ->dict:
    prompt_template = ChatPromptTemplate.from_template("""
    다음 문서를 참고해서 질문에 대한 답이 다음과 같이 나왔어. 이 내용이 맞는지 확인해줘.

    문서: {context}

    질문: {question}
    
    답변: {answer}
    """)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    structured_model = llm.with_structured_output(verified)

    chain = (prompt_template | structured_model)
    print(state["answer"])
    answer = chain.invoke({"context": state["context"], "question": state["question"], "answer": state["answer"]})
    print(answer)
    return {"fix_needed" : answer.fix_needed, "what_to_fix" : answer.what_to_fix}

def route_by_fix(state: State) -> Literal["generate", "end"]:
	if state["fix_needed"]:
		return "generate"

	elif not state["fix_needed"]:
		return "end"
      
# === 그래프 빌더 생성 === <-langchain의 chain과 동격
graph = StateGraph(State) # 상태 스키마를 기반으로 그래프 빌더 생성

# === 노드 등록 ===
graph.add_node("retrieve", retrieve) # 이름, 함수
graph.add_node("generate", generate)
graph.add_node("verify", verify)


# === 엣지 연결 ===
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate") 
graph.add_edge("generate", "verify") 
graph.add_conditional_edges(
	"verify",
	route_by_fix,
	{
	"generate": "generate",
	"end": END,
	},
)

# === 컴파일 ===
app = graph.compile()    # 빌더를 실행 가능한 그래프로 변환

# === 실행 ===
#final_answer = app.invoke({"question": "파인만이 설명한 대전하 뭐야?"})["answer"]
#print(final_answer)

# === 시각화용 그래프 구조 객체 가져오기 ===
graph_view = app.get_graph()

# === 형식 1: Mermaid 텍스트 출력 ===
mermaid_text = graph_view.draw_mermaid()
print(mermaid_text)