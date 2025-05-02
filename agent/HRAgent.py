import os
import random
import operator
from typing import Annotated, Union, List, Optional, TypedDict
from langchain.prompts import ChatPromptTemplate

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
    skills: Optional[str]
    input: str
    user_inputs: List[str]
    messages: List[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]

# ─── Tools ────────────────────────────────────────────────────────────────────
@tool("lower_case", return_direct=True)
def to_lower_case(input_text: str) -> str:
    """Convert the given text to all lowercase letters."""
    return input_text.lower()

@tool("random_number", return_direct=True)
def random_number_maker(input_text: str) -> str:
    """Return a random integer between 0 and 100 as a string."""
    return str(random.randint(0, 100))

tools = [to_lower_case, random_number_maker]
tool_executor = ToolExecutor(tools)

# ─── LLM Agent Setup ───────────────────────────────────────────────────────────
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an HR assistant chatbot. Your job is to assist users with HR-related queries such as hiring, skills assessment, and company policies. "
     "If the user hasn't mentioned required skills yet, your first response should be:\n"
     "\"What are the required skills for this position?\"\n"
     "Once skills are provided, proceed with hiring assistance. Respond professionally and concisely."),
    ("user", "{input}"),
    ("system", "{agent_scratchpad}")
])

llm = ChatOpenAI(model="gpt-4")
agent_runnable = create_openai_functions_agent(llm, tools, agent_prompt)

# ─── Helper Functions ─────────────────────────────────────────────────────────
def build_initial_state(user_input: str) -> AgentState:
    return {
        "input": user_input,
        "user_inputs": [user_input],
        "messages": [],
        "skills": None,
        "agent_outcome": None,
        "agent_scratchpad": "",
        "intermediate_steps": []  # <-- this line is crucial
    }


def should_continue(state: AgentState) -> str:
    return "end" if isinstance(state["agent_outcome"], AgentFinish) else "continue"

# ─── State Functions ───────────────────────────────────────────────────────────
def run_agent(state: AgentState):
    print("Reached Here 3")
    return {"agent_outcome": agent_runnable.invoke(state)}

def execute_tools(state: AgentState):
    action = state["agent_outcome"]
    result = tool_executor.invoke(action)
    return {"intermediate_steps": [(action, str(result))]}

def prompt_skills(state: AgentState) -> AgentState:
    state = increment_field_counter(state, "skills")

    skills = state.get("skills")
    if not skills or len(skills) == 0:
        state["messages"].append("Please enter at least one skill.")
        return state
    else:
        state["messages"].append(f"Thanks! You entered: {', '.join(skills)}")
        return state

def get_skills_needed(state: AgentState) -> AgentState:
    txt = input("Can you please enter the skills needed?").strip()
    state["skills"] = txt
    state["last_input"] = txt
    state["intermediate_steps"].append(f"Collected Skills: {txt}")
    return state

def increment_field_counter(state: AgentState, field_name: str) -> AgentState:
    if "field_counters" not in state:
        state["field_counters"] = {}
    state["field_counters"][field_name] = state["field_counters"].get(field_name, 0) + 1
    return state

# ─── LangGraph Workflow ────────────────────────────────────────────────────────
workflow = StateGraph(AgentState)

workflow.add_node("initialize_agent", run_agent)
workflow.add_node("get_skills", prompt_skills)
workflow.add_edge("get_skills", "initialize_agent")


workflow.set_entry_point("initialize_agent")

workflow.add_conditional_edges(
    "initialize_agent",
    should_continue,
    {"continue": "get_skills", "end": END}
)

graph = workflow.compile()

# ─── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.get_json(force=True).get("input", "")
    else:  # GET request
        user_input = request.args.get("chat", "")

    state = build_initial_state(user_input)
    print("Reached here 1")
    result = graph.invoke(state)
    print("Reached here 2")
    reply = result["agent_outcome"].return_values["output"]
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
