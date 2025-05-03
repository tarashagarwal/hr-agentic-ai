COMPANY_NAME= "Square Lift"
LOCATION = "California, USA"

import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from langgraph.types import interrupt
from langchain.tools import tool
from typing import Annotated, Union, List, Optional
from typing_extensions import TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
import operator

from agent.tools import tools, tools_map

# ─── Load environment and instantiate LLM ─────────────────────────────────────
from dotenv import load_dotenv, find_dotenv

# look specifically for “.env.local” up the directory tree
dotenv_path = find_dotenv(".env.local")
if not dotenv_path:
    raise FileNotFoundError("Could not find .env.local in any parent folder")
load_dotenv(dotenv_path)

llm = ChatOpenAI(model=os.getenv("GPT_MODEL_NAME"))

# ─── Agent prompt and runnable setup ─────────────────────────────────────────
agent_prompt = ChatPromptTemplate.from_messages([
    ("system",  "If the user asks to perform a general task do it first, but mention that your skills are best used for hiring related purposes"),
    ("user",    "{input}"),
    ("system",  "{agent_scratchpad}")
])
agent_runnable = create_openai_functions_agent(llm, tools, agent_prompt)

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

# ─── Helper for initial state ─────────────────────────────────────────────────
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

# ─── State transition functions ────────────────────────────────────────────────
def run_agent(state: AgentState) -> AgentState:
    outcome = agent_runnable.invoke(state)
    state["agent_outcome"] = outcome

    if isinstance(outcome, AgentAction):
        name = outcome.tool
        result = tools_map[name].invoke(outcome.tool_input)
        state["intermediate_steps"].append((outcome, str(result)))

        if name == "classify_intent":
            state["intent"] = result.strip().lower()
        else:
            state["messages"].append(AIMessage(content=str(result)))

        state["agent_scratchpad"] += f"Executed tool: {name}, Result: {result}\n"
        return state

    if isinstance(outcome, AgentFinish):
        state["messages"].append(AIMessage(content=outcome.return_values["output"]))
        return state

    return state


def handle_general_query(state: AgentState) -> AgentState:
    prompt = agent_prompt.format(
        input=state["input"],
        agent_scratchpad=state.get("agent_scratchpad", "")
    )
    resp = llm.predict(prompt)
    state["messages"].append(AIMessage(content=resp))
    return state


def get_skills(state: AgentState) -> AgentState:
    if not state["skills"]:
        ask = "Please provide the set of skills required for the job."
        state["messages"].append(SystemMessage(content=ask))
        state["skills"] = interrupt(ask)
    return state


def update_skills(state: AgentState) -> AgentState:
    incoming = state["input"]
    state["skills"] = incoming if not state["skills"] else f"{state['skills']}, {incoming}"
    state["next_expected_node"] = "generate_job_description"
    return state


def generate_job_description(state: AgentState) -> AgentState:
    skills = state["skills"]
    if not skills:
        state["messages"].append(SystemMessage(content="No skills provided. Cannot prepare a job description."))
        return state

    prompt = (
        f"Using the following skills: {skills}, "
        f"generate a professional job description for an HR assistant role "
        f"for {COMPANY_NAME} located at {LOCATION}"
    )
    resp = llm.predict(prompt)
    state["job_description"] = resp
    state["messages"].append(AIMessage(content=f"Generated Job Description: {resp}"))
    return state
