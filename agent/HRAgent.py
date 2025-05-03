# Global in-memory session store
import pdb
SESSION_STORE = {}
COMPANY_NAME= "Square Lift"
LOCATION = "California, USA"

# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import random
import operator
from typing import Annotated, Union, List, Optional
from typing_extensions import TypedDict, Literal

from flask import Flask, request, jsonify
from dotenv import load_dotenv
import uuid

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.agents import AgentAction, AgentFinish

from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver



# ─── Load environment ──────────────────────────────────────────────────────────
load_dotenv("../.env.local")
os.environ["OPENAI_API_KEY"]    = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "LangGraph_HRAgent"

# ─── Agent State Schema ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    input: str
    next_expected_node: str
    user_inputs: List[str]
    messages: List[BaseMessage]
    skills: Optional[str]
    intent: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    agent_scratchpad: str
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]
    job_description: Optional[str]

# ─── Tools ────────────────────────────────────────────────────────────────────
@tool("classify_intent", return_direct=True)
def classify_intent(input_text: str) -> str:
    """Classify the user's intent into one of: hiring, general_query, greeting, feedback, unknown."""
    prompt = f"""
You are an intent classifier. Read the following user input and classify the intent into one of these categories:
"hiring", "general_query", "greeting", "feedback", "unknown". Anything that is related to job, finding employes etc should be classified as hiring

User input: "{input_text}"

Just return one of the five categories, nothing else.
"""
    result = llm.invoke(prompt)
    pdb.set_trace()
    return result.content.strip().lower()



# @tool("lower_case", return_direct=True)
# def to_lower_case(input_text: str) -> str:
#     """Convert the given text to all lowercase letters."""
#     return input_text.lower()

@tool("random_number", return_direct=True)
def random_number_maker(input_text: str) -> str:
    """Return a random integer between 0 and 100 as a string."""
    return "Generated Random Number = " + str(random.randint(0, 100))

tools = [random_number_maker, classify_intent]
tools_map = { f.name: f for f in tools } 

# ─── LLM and Agent Setup ───────────────────────────────────────────────────────
llm = ChatOpenAI(model=os.getenv("GPT_MODEL_NAME"))
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "If the use ask to perform a general task do it first, but mention that your skills are best used for hiring related purposes"),
    ("user", "{input}"),
    ("system", "{agent_scratchpad}")
])

agent_runnable = create_openai_functions_agent(llm, tools, agent_prompt)

# ─── Helper Functions ─────────────────────────────────────────────────────────
def build_initial_state(user_input: str) -> AgentState:
    return {
        "input": user_input,
        "user_inputs": [],
        "messages": [],
        "skills": None,
        "intent": "",
        "agent_outcome": None,
        "agent_scratchpad": "",
        "intermediate_steps": [],
        "job_description": None,
    }

# ─── State Functions ───────────────────────────────────────────────────────────
def run_agent(state: AgentState) -> AgentState:
    agent_outcome = agent_runnable.invoke(state)
    state["agent_outcome"] = agent_outcome

    if isinstance(agent_outcome, AgentAction):

        tool_name = agent_outcome.tool
        result = tools_map[tool_name].invoke(agent_outcome.tool_input)
        state["intermediate_steps"].append((agent_outcome, str(result)))

        if agent_outcome.tool == "classify_intent":
            state["intent"] = str(result).strip().lower()
        else: 
            state["messages"].append(AIMessage(content=f"{result}"))

        state["agent_scratchpad"] += f"Executed tool: {agent_outcome.tool}, Result: {result}\n"
        return state

    if isinstance(agent_outcome, AgentFinish):
        state["messages"].append(AIMessage(content=agent_outcome.return_values["output"]))
        return state

    return state

def handle_general_query(state: AgentState) -> AgentState:
    """Handle general queries by passing the input back to the LLM using the agent prompt."""
    user_input = state["input"]
    # Ensure agent_scratchpad is initialized
    if "agent_scratchpad" not in state:
        state["agent_scratchpad"] = ""  # Initialize if missing

    # Use the agent prompt to guide the LLM's response
    prompt = agent_prompt.format(input=user_input, agent_scratchpad=state["agent_scratchpad"])
    response = llm.predict(prompt)

    # Add the response to the state messages
    state["messages"].append(AIMessage(content=response))
    print(f"General query response: {response}")  # Debugging
    return state

def get_skills(state: AgentState) -> AgentState:
    if not state.get("skills"):
        #prompt = "Generate a polite system message asking the user to provide the set of skills required for the job."
        response = "Please provide the set of skills required for the job."
        state["messages"].append(SystemMessage(content=response))
        state[ "skills"] = interrupt(response) # Initialize skills if not present
    return state

def update_skills(state: AgentState) -> AgentState:

    user_input = state["input"]
    if not state.get("skills"):
        state["skills"] = user_input
    else:
        state["skills"] += f", {user_input}"
    state["next_expected_node"] = "generate_job_description"
    return state

def generate_job_description(state: AgentState) -> AgentState:
    skills = state.get("skills")
    if not skills:
        state["messages"].append(SystemMessage(content="No skills provided. Cannot prepare a job description."))
        return state
    prompt = f"Using the following skills: {skills}, generate a professional job description for an HR assistant role. for {COMPAN_NAME} located at {LOCATION}"
    response = llm.predict(prompt)
    state["job_description"] = response
    state["messages"].append(AIMessage(content=f"Generated Job Description: {response}"))
    return state


# ─── LangGraph Workflow ────────────────────────────────────────────────────────
checkpointer = MemorySaver()
workflow = StateGraph(AgentState)
workflow.add_node("initialize_agent", run_agent)
workflow.add_node("handle_general_query", handle_general_query)
workflow.add_node("get_skills", get_skills)
workflow.add_node("update_skills", update_skills)
workflow.add_node("generate_job_description", generate_job_description)


workflow.add_conditional_edges(
    "initialize_agent",
    lambda s: (
        "get_skills" if s["intent"] == "hiring"
        else "handle_general_query" if s["intent"] == "general_query"
        else "dummy_resume" if s.get("intent") == "__make_graph_happy__"
        else "end"
    ),
    {
        "get_skills": "get_skills",
        "handle_general_query": "handle_general_query",
        "end": END
    }
)

workflow.add_edge("get_skills", "update_skills")
# Normal internal flow after resume
workflow.add_edge("update_skills", "generate_job_description")

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
        user_input = data.get("input", "")
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
        # Resume interrupted flow with user's response
        state["input"] = user_input
        state["user_inputs"].append(user_input)
        state["messages"].append(HumanMessage(content=user_input))
    # Run or resume graph
    interrupt_info = state.get("__interrupt__")
    thread_config = {"configurable": {"thread_id": session_id}}
    if interrupt_info and isinstance(interrupt_info, list) and interrupt_info[0].resumable:
        resume_value = state["input"]  # e.g., "Java"
        resume_ns = state["__interrupt__"][0].ns[0]
        command = Command(resume=resume_value)
        result = graph.invoke(command, config=thread_config)
    else:
        result = graph.invoke(state, config=thread_config)
    print(state)
    # Handle interrupt
    if "__interrupt__" in result:
        SESSION_STORE[session_id] = result
        return jsonify({
            "interrupted": True,
            "question": result["__interrupt__"],
            "session_id": session_id,
            "messages": [
                f"system: {m.content}" if isinstance(m, (AIMessage, SystemMessage)) else f"user: {m.content}"
                for m in result["messages"]
            ]
        })

    # Normal response
    SESSION_STORE[session_id] = result
    reply = result["messages"][-1].content if result["messages"] else ""

    return jsonify({
        "interrupted": False,
        "reply": reply,
        "session_id": session_id,
        "messages": [
            f"system: {m.content}" if isinstance(m, (AIMessage, SystemMessage)) else f"user: {m.content}"
            for m in result["messages"]
        ]
    })



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
