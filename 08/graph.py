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

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun

#bind tools
tools = [DuckDuckGoSearchRun(), WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()), ArxivQueryRun()]
tool_map = {tool.name: tool for tool in tools} #이름으로 검색할 수 있게


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
    fix_needed: bool #False
    what_to_fix: list[str]  
    needs_more_context: bool #False
    top_k: int #3
    try_count: int #0
    limit: int #4
    #arxiv_references: list[str]

def retrieve(state: State) -> dict:
    if state.get("needs_more_context", False)==False:
        # state에서 question 꺼내서
        q=state["question"]
        # vectorstore에서 검색하고
        retriever = vectorstore.as_retriever(search_kwargs={"k": state.get("top_k",3)})
        docs = retriever.invoke(q)
        # context 반환
        return {"context": docs, "needs_more_context": False ,"top_k": state.get("top_k",3)}
    
    elif state["needs_more_context"]==True:
                # state에서 question 꺼내서
        q=state["question"]
        # vectorstore에서 검색하고
        retriever = vectorstore.as_retriever(search_kwargs={"k": state["top_k"]+1})
        docs = retriever.invoke(q)
        # context 반환
        return {"context": docs, "needs_more_context": False ,"top_k": state["top_k"]+1}

def generate(state: State) -> dict:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # tool call
    llm_with_tools = llm.bind_tools(tools)  # tools 참조
    response = llm_with_tools.invoke(state["question"])

    tool_results={}
    for tool_call in response.tool_calls:
        tool_result = tool_map[tool_call["name"]].invoke(tool_call["args"])
        tool_results[tool_call["name"]] = tool_result
    tool_docs = [Document(page_content=result) for result in tool_results.values()]

    if state.get("fix_needed", False):
        prompt_template = ChatPromptTemplate.from_template("""
        다음 문서를 참고해서 질문에 답해줘.

        문서: {context}

        질문: {question}
        
        답변: {answer}
                                                        
        이 부분이 틀렸어. 다시 수정해줘. 
                                                        
        고칠 부분: {what_to_fix}
        """)

        chain = (prompt_template | llm | StrOutputParser())

        answer = chain.invoke({
            "context": state["context"]+tool_docs, 
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

        chain = (prompt_template | llm | StrOutputParser())

        answer = chain.invoke({"context": state["context"]+tool_docs,
                                "question": state["question"]
                                })

    
    return {"answer" : answer, "fix_needed" : False}


class verified(BaseModel):
    fix_needed: bool = Field(description="answer가 수정이 필요한지 여부")
    what_to_fix: list[str] = Field(description="고쳐야 하는 부분들")
    needs_more_context: bool = Field(description="수정할 때 추가 정보가 필요한지 여부")

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
    print(f"{state.get("try_count", 0)+1}번쨰 시도: \n {state["answer"]}\n")
    answer = chain.invoke({"context": state["context"], "question": state["question"], "answer": state["answer"]})
    print(answer)
    return {"fix_needed" : answer.fix_needed, 
            "what_to_fix" : answer.what_to_fix, 
            "try_count" : state.get("try_count", 0)+1,
            "needs_more_context" : answer.needs_more_context
            }


def route_by_fix(state: State) -> Literal["final_answer", "retrieve","generate"]:
    if not state["fix_needed"] or state["try_count"] >= state.get("limit",4):
        return "final_answer"
    
    elif state["fix_needed"] and state["needs_more_context"]:
        return "retrieve"
    
    elif state["fix_needed"] and not state["needs_more_context"]:
        return "generate"
    
def final_answer(state: State) ->dict:
    if state["fix_needed"]:
        return {"answer" : f"limit:{state["try_count"]} 내에 적합한 답변 도출 불가능 \n {state["answer"]} \n 발견된 문제점: {state["what_to_fix"]}"}
    else:
        return {"answer": state["answer"]}

      
# === 그래프 빌더 생성 === <-langchain의 chain과 동격
graph = StateGraph(State) # 상태 스키마를 기반으로 그래프 빌더 생성

# === 노드 등록 ===
graph.add_node("retrieve", retrieve) # 이름, 함수
graph.add_node("generate", generate)
graph.add_node("verify", verify)
graph.add_node("final_answer", final_answer)


# === 엣지 연결 ===
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate") 
graph.add_edge("generate", "verify") 
graph.add_conditional_edges(
	"verify",
	route_by_fix,
	{
	"generate": "generate",
	"final_answer": "final_answer",
    "retrieve": "retrieve"
	},
)
graph.add_edge("final_answer", END) 


# === 컴파일 ===
app = graph.compile()    # 빌더를 실행 가능한 그래프로 변환

# === 실행 ===
#end_answer = app.invoke({"question": "파인만이 설명한 강력이 뭐야?"})["answer"]
#print(end_answer)

# === 시각화용 그래프 구조 객체 가져오기 ===
graph_view = app.get_graph()

# === 형식 1: Mermaid 텍스트 출력 ===
mermaid_text = graph_view.draw_mermaid()
print(mermaid_text)