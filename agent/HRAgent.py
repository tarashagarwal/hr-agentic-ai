# Global in-memory session store
import pdb
SESSION_STORE = {}


# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import random
from flask import Flask, request, jsonify
import uuid

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from agent.agentic_logic import (
    llm,
    agent_prompt,
    agent_runnable,
    build_initial_state,
    run_agent,
    handle_general_query,
    collect_job_details,
    are_job_details_missing,
    generate_job_description,
    AgentState
)

# ─── Load environment ──────────────────────────────────────────────────────────
from dotenv import load_dotenv, find_dotenv

# look specifically for “.env.local” up the directory tree
dotenv_path = find_dotenv(".env.local")
if not dotenv_path:
    raise FileNotFoundError("Could not find .env.local in any parent folder")
load_dotenv(dotenv_path)

os.environ["OPENAI_API_KEY"]    = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "LangGraph_HRAgent"


# ─── LangGraph Workflow ────────────────────────────────────────────────────────
checkpointer = MemorySaver()
workflow = StateGraph(AgentState)
workflow.add_node("initialize_agent", run_agent)
workflow.add_node("handle_general_query", handle_general_query)
workflow.add_node("collect_job_details", collect_job_details)
# workflow.add_node("are_job_details_missing", are_job_details_missing)
workflow.add_node("generate_job_description", generate_job_description)


workflow.add_conditional_edges(
    "initialize_agent",
    lambda s: (
        "collect_job_details" if s["intent"] == "hiring"
        else "handle_general_query" if s["intent"] == "general_query"
        else "dummy_resume" if s.get("intent") == "__make_graph_happy__"
        else "end"
    )
)


# workflow.add_edge("collect_job_details", "job_details_missing")

# Normal internal flow after resume
workflow.add_conditional_edges(
    "collect_job_details",
    lambda s: "collect_job_details" if s.get("job_details_missing") else "generate_job_description"
)

# workflow.add_edge("get_skills", "get_pay")
# workflow.add_edge("get_pay", "generate_job_description")

workflow.add_conditional_edges(
    "handle_general_query",
    lambda s: "end",
    {"end": END}
)

workflow.add_conditional_edges(
    "generate_job_description",
    lambda s: "end",
    {"end": END}
)
workflow.set_entry_point("initialize_agent")
graph = workflow.compile(
    checkpointer=checkpointer, #Needed for resuming sessions
)

# ─── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        data = request.get_json(force=True)
        user_input = data.get("chat", "")
        session_id = data.get("session_id")
    else:
        user_input = request.args.get("chat", "")
        session_id = request.args.get("session_id")

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    # Start or resume session
    if session_id not in SESSION_STORE:
        state = build_initial_state(user_input)
    else:
        state = SESSION_STORE[session_id]
        # Resume interrupted flow with user's response        #same as above
    
    state["input"] = user_input
    state["user_inputs"].append(user_input)
    state["user_messages"].append(HumanMessage(content=user_input))
    
    # Run or resume graph
    interrupt_info = state.get("__interrupt__")
    thread_config = {"configurable": {"thread_id": session_id}}
    if interrupt_info and isinstance(interrupt_info, list) and interrupt_info[0].resumable:
        resume_value = state["input"]  # e.g., "Java"
        resume_ns = state["__interrupt__"][0].ns[0]
        command = Command(resume=resume_value)
        result = graph.invoke(command, config=thread_config)
    else:
        state["agent_scratchpad"] = "" #Dont want llm to misclassify the intent based on previous activity
        state["intermediate_steps"] = []
        state["intent"] = ""    
        result = graph.invoke(state, config=thread_config)
        
    print(state)
    
    # Handle interrupt
    if "__interrupt__" in result:
        SESSION_STORE[session_id] = result
        response = jsonify({
            "interrupted": True,
            "question": result["__interrupt__"],
            "session_id": session_id,
            "user_messages": [
                f"system: {m.content}" if isinstance(m, (AIMessage, SystemMessage)) else f"human: {m.content}"
                for m in result["user_messages"]
            ]
        })
    else:

        # Normal response
        SESSION_STORE[session_id] = result
        reply = result["user_messages"][-1].content if result["user_messages"] else ""

        response = jsonify({
            "interrupted": False,
            "reply": reply,
            "session_id": session_id,
            "user_messages": [
                f"system: {m.content}" if isinstance(m, (AIMessage, SystemMessage)) else f"human: {m.content}"
                for m in result["user_messages"]
            ]
        })

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
