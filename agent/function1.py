import os
import random
import operator
from typing import Annotated, Union, List, TypedDict

from dotenv import load_dotenv
from flask import Flask, request, jsonify

# ─── Load environment from parent dir ─────────────────────────────────────────
load_dotenv("../.env.local")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "LangGraph_HRAgent"

# ─── LangGraph & LangChain Core ───────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage

# ─── LangChain Agent + Tools ─────────────────────────────────────────────────
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain import hub

# ─── Agent State Schema ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]

# ─── Tools ────────────────────────────────────────────────────────────────────
@tool("lower_case", return_direct=True)
def to_lower_case(input: str) -> str:
    """Convert the given text to all lowercase letters."""
    return input.lower()

@tool("random_number", return_direct=True)
def random_number_maker(input: str) -> str:
    """Return a random integer between 0 and 100 as a string."""
    return str(random.randint(0, 100))

tools = [to_lower_case, random_number_maker]
tool_executor = ToolExecutor(tools)

# ─── LLM Agent Setup ───────────────────────────────────────────────────────────
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-4")  # no streaming for simplicity
agent_runnable = create_openai_functions_agent(llm, tools, prompt)

# ─── Node Functions (old-style) ────────────────────────────────────────────────
def run_agent(data: AgentState):
    return {"agent_outcome": agent_runnable.invoke(data)}

def execute_tools(data: AgentState):
    action = data["agent_outcome"]
    result = tool_executor.invoke(action)
    return {"intermediate_steps": [(action, str(result))]}

def should_continue(data: AgentState) -> str:
    return "end" if isinstance(data["agent_outcome"], AgentFinish) else "continue"

# ─── Build the LangGraph workflow ──────────────────────────────────────────────
workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "end": END}
)
workflow.add_edge("action", "agent")

graph = workflow.compile()

# ─── Helper to build initial state ─────────────────────────────────────────────
def build_initial_state(user_input: str) -> AgentState:
    return {
        "input": user_input,
        "chat_history": [],
        "agent_outcome": None,
        "intermediate_steps": [],
    }

# ─── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True)
    user_input = body.get("input", "")
    state = build_initial_state(user_input)
    result = graph.invoke(state)
    reply = result["agent_outcome"].return_values["output"]
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
