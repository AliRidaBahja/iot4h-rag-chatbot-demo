# langgraph_utils.py
import time 
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage
from typing import TypedDict, List, Optional
from similarity import custom_similarity_search
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from db_utils import insert_message
import logging
from db_utils import save_retrieved_docs
from db_utils import get_retrieved_docs

# logging.basicConfig(filename='app.log', level=logging.INFO)

class CustomState(TypedDict):
    messages: List[BaseMessage]
    session_id: str
    user_input: Optional[str]
    retrieved_docs: Optional[List[dict]]

def print_state(state: CustomState):
    print("\n=== GRAPH STATE ===")
    print(f"Session ID: {state.get('session_id')}")
    print(f"Query: {state.get('user_input')}")

    print("\nMessages:")
    for m in state.get("messages", []):
        msg_type = m.__class__.__name__
        print(f"[{msg_type}] {getattr(m, 'content', '')}")

    if "retrieved_docs" in state:
        if state.get('retrieved_docs') is not None:
            print(f"\nRetrieved Docs: {len(state['retrieved_docs'])}")
            for i, doc in enumerate(state["retrieved_docs"]):
                content = doc.get("page_content", "")[:80].replace("\n", " ") + "..."
                score = doc.get("score", "N/A")
                print(f"  {i+1}. Score: {score:.2f} - {content}")
        print("====================\n")
    
def get_rag_graph(model):

    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve relevant IoT use cases based on the user query."""
        docs_with_scores = custom_similarity_search(query, 1, 0.5)
        artifact = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in docs_with_scores
        ]
        print(" ********************** retrieve ****************** ") 
        # time.sleep(2)
        return "returned docs", artifact  
    
    def query_or_retrieve(state: CustomState):
        print(" ********************** query_or_retrieve ****************** ") 
        # print_state(state)
        # time.sleep(2)

        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])

        insert_message(state["session_id"], response, model=llm.model_name)
        # user_msg = next((m for m in reversed(state["messages"]) if m.type == "human"), None)

        return {
            "messages": state["messages"] + [response],
            "session_id": state["session_id"],
            # "user_input": user_msg.content if user_msg else "",
            # "retrieved_docs": retrieved_docs
        }

    def maybe_answer_from_docs(state: CustomState):
        print(" ********************** maybe_answer_from_docs ****************** ") 
        # print_state(state)
        # time.sleep(2)

        """Check if this is a follow-up and retrieved_docs exist — then reuse them."""
        session_id = state["session_id"]

        previous_docs = get_retrieved_docs(session_id)
        latest_query = next((m for m in reversed(state["messages"]) if m.type == "human"), None)

        if previous_docs and latest_query:
            prompt = [
            SystemMessage(
                "The user is going to ask questions about iot in Handcraft (Handwerk in German). "
                "You are to be given the full conversation between the user and the LLM model, and you should assess if the latest user query is asking follow-up questions or a completely different topic (for example, a different field of Handcraft). "
                "If the document is not written in english, then translate it first before analyzing. "
                f"Document:\n{previous_docs}."
                "It is very important to return exactly the word: 'None' If there is no corrolation, i-e without any explanation." 
                "If there is corrolation, return exactly the word 'True'."
            ),
            *[
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)]
                ]

            response = llm.invoke(prompt).content.strip()
            if response == "None" or response == None:
                return {
                     "retrieved_docs": None
                    }
            
        return {
            "retrieved_docs": previous_docs,
        }
    
    def followup_generate(state: CustomState):
        """Answer follow-up questions using previously stored docs."""
        print(" ********************** followup_generate ****************** ") 
        # print_state(state)
        # time.sleep(2)

        retrieved_docs = get_retrieved_docs(state["session_id"])

        docs_text = "\n\n".join(
            f"Title: {doc.get('metadata', {}).get('title', 'N/A')}\n"
            f"Link: {doc.get('metadata', {}).get('link', 'N/A')}\n"
            f"Gewerke: {doc.get('metadata', {}).get('gewerke', 'N/A')}\n"
            f"{doc.get('page_content', '')}"
            for doc in retrieved_docs
        )

        system_prompt = (
        "This is a follow-up question. Use the following information (if present) to answer. Be concise, stay in the user's language.\n\n"
        f"{docs_text}" if docs_text else "No relevant documents found."
        )

        conversation = [m for m in state["messages"] if m.type in ("human", "ai")]
        prompt = [SystemMessage(system_prompt)] + conversation
        # print("prompt....", prompt)
        response = llm.invoke(prompt)
        insert_message(state["session_id"], response, model=llm.model_name)

        return {
            "messages": state["messages"] + [response],
            "session_id": state["session_id"],
            "retrieved_docs": retrieved_docs,
        }

    def generate(state: CustomState):

        """Generate answer."""
        # Get generated ToolMessages
        print(" ********************** generate ****************** ") 
        print_state(state)
        # time.sleep(2)

        retrieved_docs  = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                if hasattr(message, "artifact") and isinstance(message.artifact, list):
                    retrieved_docs.extend(message.artifact)
            else:
                break  # Stop at the first non-tool message

        save_retrieved_docs(state["session_id"], retrieved_docs)

        docs_text = "\n\n".join(
            f"Title: {doc.get('metadata', {}).get('title', 'N/A')}\n"
            f"Link: {doc.get('metadata', {}).get('link', 'N/A')}\n"
            f"Gewerke: {doc.get('metadata', {}).get('gewerke', 'N/A')}\n"
            f"{doc.get('page_content', '')}"
            for doc in retrieved_docs
        )

        prompt = [

            SystemMessage(
                "You are an assistant for craftspeople exploring IoT in their field of work. They are going to ask you questions about the topic and you will be given some documents to help you answer. "
                "If a retrieved document is available, answer with the following instructions, but if no retrieved document is available then answer the question, and in this case only, IGNORE the following instructions (Please keep the language in German):"
                "- 1. Ignoring the retrieved document, answer directly in your own words."
                "- 2. From the document, extract only the sentences and parts relevant to the query."
                # "- 3. Output step 1 and step 2 with a line between them. This is very important."
                "- 3. Ground the output from step 1 with the information in step 2."
                "- 4. Rephrase the output of step 3 such that it adhers to an interactive conversational flow of communication, such that the output is short enough and concise and maybe asks follow-up questions."
                    " for example, the user asks: How can I iot in meinem Beruf nutze?"
                    " AI: Als Bauarbeiter, kannst du iot für Betonmonitoring anwenden. Interessierts du dafür? oder brauchst du andere Anwendüngsfälle?"
                f"Here is the Retrieved docs:\n{docs_text}" if docs_text else "no relevant documents retrieved"

            ),
            *[
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)]
        ]

        response = llm.invoke(prompt)


        
        insert_message(state["session_id"], response, model=llm.model_name)
        return {"messages": state["messages"] + [response], "session_id": state["session_id"], "retrieved_docs": retrieved_docs
                }

    graph = StateGraph(CustomState)
    llm = ChatOpenAI(model=model)

    graph.add_node("query_or_retrieve", query_or_retrieve)
    graph.add_node("maybe_answer_from_docs", maybe_answer_from_docs)
    graph.add_node("followup_generate", followup_generate)
    graph.add_node("tools", ToolNode([retrieve]))
    graph.add_node("generate", generate)

    graph.set_entry_point("maybe_answer_from_docs")
    # Route from maybe_answer_from_docs based on presence of docs
    def should_use_followup_docs(state: CustomState) -> str:
        print("# HERE: ", state.get("retrieved_docs"))
        return "followup_generate" if state.get("retrieved_docs") else "query_or_retrieve"

    graph.add_conditional_edges("maybe_answer_from_docs", should_use_followup_docs, {
        "followup_generate": "followup_generate",
        "query_or_retrieve": "query_or_retrieve"
    })    

    graph.add_conditional_edges(
        "query_or_retrieve",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph.add_edge("tools", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("followup_generate", END)

    return graph

