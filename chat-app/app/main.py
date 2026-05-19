# %% Load OPENAI / LANGSMITH APIs:
import os
import uuid
import logging
import sys
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic_models import QueryInput, QueryResponse
from langgraph_utils import get_rag_graph
from context_qa import static_qna_handler, PREDEFINED_QNA
from db_utils import get_chat_history, AIMessage, HumanMessage, insert_message, get_current_q_index, increment_q_index
from chromadb import HttpClient
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display

load_dotenv()
load_dotenv("../../.env")  # remove in Docker
log_file_path = os.path.join(os.path.dirname(__file__), 'app.log')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
app = FastAPI()
memory = MemorySaver()

CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")
client = HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
    
@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    model = query_input.model 
    input = query_input.query

    # Proceed with static Q&A logic
    if get_current_q_index(session_id) <= len(PREDEFINED_QNA):
        static_response = static_qna_handler(input, model, session_id)
        return static_response
    
    chat_history = get_chat_history(session_id)
    # logging.info(f"+++ chat_history in main: {chat_history} +++")
    # If all static questions are done, proceed to LLM logic
    user_msg = HumanMessage(content=input)
    # print("current user_msg:", user_msg)
    
    messages = chat_history + [user_msg]
    insert_message(session_id, user_msg, model=model)

    # graph = get_rag_graph(model=model).compile(checkpointer=memory)
    graph = get_rag_graph(model=model).compile()
    # config = {"configurable": {"thread_id": "1"}}
    image_data = graph.get_graph().draw_mermaid_png()
    with open("output.png", "wb") as f:
        f.write(image_data)
        
    # result = graph.invoke({       
    #     "messages": messages,
    #     "session_id": session_id,
    #     "user_input": input
    # }, config)

    result = graph.invoke({       
        "messages": messages,
        "session_id": session_id,
        "user_input": input
    })


    final_msg = result["messages"][-1]
    sources = []

    for msg in result["messages"]:
        if msg.type == "tool" and hasattr(msg, "artifact"):
            sources = msg.artifact
            break

    return QueryResponse(
        response=final_msg.content,
        session_id=session_id,
        model=model,
        sources=sources,
        is_done=True
    )

if __name__ == "__main__":
    import subprocess
    import uvicorn

    subprocess.Popen(["streamlit", "run", "./streamlit_app/streamlit_app.py"])
    uvicorn.run(app, host="0.0.0.0", port=8001)
