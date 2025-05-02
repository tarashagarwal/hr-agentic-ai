# Global in-memory session store
SESSION_STORE = {}

# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import random
import operator
from typing import Annotated, Union, List, Optional, TypedDict
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from flask import Flask, request, jsonify
import uuid

from dotenv import load_dotenv
from flask import Flask, request, jsonify

# ─── Load environment from parent dir ─────────────────────────────────────────
load_dotenv("../.env.local")
os.environ["OPENAI_API_KEY"]    = os.getenv("OPENAI_API_KEY")
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
     "You are an HR assistant chatbot. Your job is to assist users with HR-related queries such as hiring, skills assessment, and company policies."
     "Whenver user ask for Anything. Please provide them with the best possible answer even if general question but in the end remind them that your best"
     "Primary Task is to help with Hiring Related tasks."),
    ("user", "{input}"),
    ("system", "{agent_scratchpad}")
])

llm = ChatOpenAI(model=os.getenv("GPT_MODEL_NAME"))
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
        "intermediate_steps": []
    }


def should_continue(state: AgentState) -> str:
    return "end" if isinstance(state["agent_outcome"], AgentFinish) else "continue"

# ─── State Functions ───────────────────────────────────────────────────────────
def run_agent(state: AgentState):
    agent_outcome = agent_runnable.invoke(state)
    print("Reached Here 3 ************", agent_outcome)

    # Check if the outcome is an action
    if isinstance(agent_outcome, AgentAction):
        # Execute the tool and update the state
        result = tool_executor.invoke(agent_outcome)
        state["intermediate_steps"].append((agent_outcome, str(result)))
        return {"agent_outcome": agent_outcome}

    state["messages"].append(AIMessage(content=agent_outcome.return_values["output"]))
    # If the outcome is a finish, return it directly
    return {"agent_outcome": agent_outcome}

def execute_tools(state: AgentState):
    action = state["agent_outcome"]
    result = tool_executor.invoke(action)
    return {"intermediate_steps": [(action, str(result))]}

def generate_job_description(state: AgentState) -> AgentState:
    # Ensure skills are provided
    skills = state.get("skills")
    if not skills:
        state["messages"].append(SystemMessage(content="No skills provided. Cannot prepare a job description."))
        return state

    # Query the LLM to generate a job description
    prompt = f"Using the following skills: {skills}, generate a professional job description for an HR assistant role."
    response = llm.predict(prompt)

    # Update the state with the generated job description
    state["job_description"] = response
    state["messages"].append(AIMessage(content=f"Generated Job Description: {response}"))
    print("Generated Job Description:", response)
    return state

def update_skills(state: AgentState) -> AgentState:
    # Extract the user's input
    user_input = state["input"]

    # Update the skills field based on the user's input
    if "skills" not in state or not state["skills"]:
        state["skills"] = user_input
    else:
        # Append new skills to the existing list
        state["skills"] += f", {user_input}"

    # Add a message to indicate the update
    state["messages"].append(SystemMessage(content=f"Updated skills: {state['skills']}"))
    print("Updated skills:", state["skills"])
    return state


def generate_job_description(state: AgentState) -> AgentState:
    # Extract the user's input
    user_input = state["input"]

    # Update the skills field based on the user's input
    if "skills" not in state or not state["skills"]:
        state["skills"] = user_input
    else:
        # Append new skills to the existing list
        state["skills"] += f", {user_input}"

    # Add a message to indicate the update
    state["messages"].append(f"Updated skills: {state['skills']}")
    print("Updated skills:", state["skills"])
    return state

# def increment_field_counter(state: AgentState, field_name: str) -> AgentState:
#     if "field_counters" not in state:
#         state["field_counters"] = {}
#     state["field_counters"][field_name] = state["field_counters"].get(field_name, 0) + 1
#     return state

# ─── LangGraph Workflow ────────────────────────────────────────────────────────
workflow = StateGraph(AgentState)

workflow.add_node("initialize_agent", run_agent)
workflow.add_node("update_skills", update_skills)
workflow.add_node("generate_job_description", generate_job_description)

workflow.add_conditional_edges(
    "initialize_agent",
    should_continue,
    {"continue": "update_skills", "end": END}
)

workflow.add_edge("update_skills", "generate_job_description")


workflow.add_conditional_edges(
    "generate_job_description",
    lambda state: "end",  # End the workflow after generating the job description
    {"end": END}
)

workflow.set_entry_point("initialize_agent")

graph = workflow.compile()

# ─── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
SESSION_STORE = {}  # Global dictionary for session persistence

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        data = request.get_json(force=True)
        user_input = data.get("input", "")
        session_id = data.get("session_id")
    else:
        user_input = request.args.get("chat", "")
        session_id = request.args.get("session_id")

    # Start a new session if not provided or not found
    if not session_id or session_id not in SESSION_STORE:
        session_id = session_id or str(uuid.uuid4())
        state = build_initial_state(user_input)
    else:
        state = SESSION_STORE[session_id]
      
    state["input"] = user_input
    state["user_inputs"].append(user_input)
    state["messages"].append(HumanMessage(content=user_input))

    # Run the LangGraph logic
    result = graph.invoke(state)
    agent_outcome = result["agent_outcome"]

    print("State:")
    print(state)

    # Handle AgentFinish or AgentAction
    if isinstance(agent_outcome, AgentFinish):
        reply = state["messages"][-1] if state["messages"] else ""
    elif isinstance(agent_outcome, AgentAction):
        reply = f"Action required: {agent_outcome.tool} with input {agent_outcome.tool_input}"
    else:
        reply = "Unexpected outcome from the agent."

    # Save updated state in memory
    SESSION_STORE[session_id] = result

    messages = [message.content for message in state["messages"]]

    return jsonify({
        "user_input": user_input if user_input else "",
        "reply": messages[-1] if messages else "",
        "messages": messages,
        "session_id": session_id
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)