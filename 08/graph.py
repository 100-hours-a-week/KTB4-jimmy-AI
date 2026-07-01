from dotenv import load_dotenv
#from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START, END  

from typing_extensions import TypedDict
from langchain_core.documents import Document

from pydantic import BaseModel, Field
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchRun
#from langchain_community.tools import WikipediaQueryRun
#from langchain_community.utilities import WikipediaAPIWrapper
#from langchain_community.tools.arxiv.tool import ArxivQueryRun

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

#bind tools
tools = [DuckDuckGoSearchRun(), 
        #WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()), #api 오류-일시적인가?
        #ArxivQueryRun()
        ]
tool_map = {tool.name: tool for tool in tools} #이름으로 검색할 수 있게


#api key 가져오기
load_dotenv()

#모델 선택 기능을 위한 map 
model_map = {
    "gemini": ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
    "claude": ChatAnthropic(model="claude-haiku-4-5-20251001")
    }

#chromadb 불러오기
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001") # 이건 모델 선택 불가-이미 임베딩함
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
    what_to_fix: str 
    needs_more_context: bool #False
    top_k: int #3
    try_count: int #0
    limit: int #4
    #arxiv_references: list[str]
    model: str #"gemini" or "claude"

def retrieve(state: State) -> dict:
    if state.get("needs_more_context", False)==False:
        # state에서 question 꺼내서 vectorstore에서 검색하고
        retriever = vectorstore.as_retriever(search_kwargs={"k": state.get("top_k",3)})
        docs = retriever.invoke(state["question"])
        # context 반환
        return {"context": docs, "needs_more_context": False ,"top_k": state.get("top_k",3)}
    
    elif state["needs_more_context"]==True:
        # state에서 question 꺼내서 vectorstore에서 검색하고
        retriever = vectorstore.as_retriever(search_kwargs={"k": state["top_k"]+1})
        docs = retriever.invoke(state["question"])
        # context 반환
        return {"context": docs, "needs_more_context": False ,"top_k": state["top_k"]+1}

def generate(state: State) -> dict:
    print("---"+str(state.get("try_count", 0)+1)+"번째 시도---")
    llm = model_map[state["model"]]

    # tool call
    llm_with_tools = llm.bind_tools(tools)  # tools 참조
    messages = [
        SystemMessage(content=f"""
            다음 문서를 참고해서 답해줘. 문서에 없는 내용은 검색 tool을 사용해.
            문서: {state['context']}
            {f"고칠 부분: {state['what_to_fix']}" if state.get('fix_needed') else ''}
        """),
        HumanMessage(content=state["question"])
    ]

    tool_response = llm_with_tools.invoke(messages)

    max_tool_rounds = 3
    tool_rounds = 0
    while tool_response.tool_calls and tool_rounds < max_tool_rounds:
        tool_results = {}
        for tool_call in tool_response.tool_calls:
            tool_results[tool_call["name"]] = tool_map[tool_call["name"]].invoke(tool_call["args"])
        
        messages += [
            tool_response,
            *[ToolMessage(content=v, tool_call_id=tc["id"])
            for tc, v in zip(tool_response.tool_calls, tool_results.values())]
        ]
        tool_response = llm_with_tools.invoke(messages)

        tool_rounds += 1

    #tool_response.context는 str이거나, list[dict]이거나, text attribute를 가진 list[object]일 수 있음
    answer = tool_response.content if isinstance(tool_response.content, str) else "".join(
        block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
        for block in tool_response.content
    )    
    print(answer)
    
    return {"answer" : answer, "fix_needed" : False}


class verified(BaseModel):
    fix_needed: bool = Field(description="answer가 수정이 필요한지 여부")
    what_to_fix: str = Field(description="고쳐야 하는 부분들")
    needs_more_context: bool = Field(description="수정할 때 추가 정보가 필요한지 여부")

def verify(state: State) ->dict:
    print("---verify 단계 시작---")

    llm = model_map[state["model"]]

    messages = [
    SystemMessage(content=f"""
        다음 문서와 네가 알고 있는 지식을 종합해서 답이 맞는지 확인해줘.
        문서에 근거가 없더라도 네 지식으로 판단해도 돼.
        문서: {state["context"]}
    """),
    HumanMessage(content=state["question"]),
    AIMessage(content=state["answer"])
    ]

    structured_model = llm.with_structured_output(verified)
    answer = structured_model.invoke(messages)

    print("수정 필요한가: "+str(state["fix_needed"]))
    print("고칠점: "+str(state.get("what_to_fix","")))


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
    print("-----최종답변-----")
    if state["fix_needed"]:
        answer_f=f"limit:{state["try_count"]} 내에 적합한 답변 도출 불가능 \n {state["answer"]} \n 발견된 문제점: {state["what_to_fix"]}"
        print("최종답변: "+answer_f)
        return {"answer" : answer_f}

    else:
        print("최종답변: "+state["answer"])
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
#graph_view = app.get_graph()

# === 형식 1: Mermaid 텍스트 출력 ===
#mermaid_text = graph_view.draw_mermaid()
#print(mermaid_text)