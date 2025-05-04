
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
import pdb
# ─── Constants ────────────────────────────────────────────────────────────────
from agent.tools import tools, tools_map
from agent.config import COMPANY_DETAILS

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
    ("system",  "Answer what ever the user is asking "
    "but mention and remember that your skills are best" 
    "used for hiring related purposes"),
    ("user",    "{input}"),
    ("system",  "{agent_scratchpad}")
])


system_query_prompt = ChatPromptTemplate.from_messages([
    ("system",  "Do as asked."),
    ("user",    "{input}"),
    ("system",  "{agent_scratchpad}")
])


agent_runnable = create_openai_functions_agent(llm, tools, agent_prompt)

# ─── Agent State Schema ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    input: str
    interrupt_count: int
    terminate: bool
    next_expected_node: str
    user_inputs: List[str]
    messages: List[BaseMessage]
    user_messages: List[BaseMessage]
    ################job attributes#################
    job_title: Optional[str]  # Added job_title for clarity
    skills: Optional[str]
    budget: Optional[str]
    timeline: Optional[str]
    educational_requirements: Optional[str]
    job_type: Optional[str]
    key_responsibilities: Optional[str]
    ###############################################
    intent: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    agent_scratchpad: str
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]
    job_description: Optional[str]


# ─── Helper for initial state ─────────────────────────────────────────────────
def build_initial_state(user_input: str) -> AgentState:
    return {
        "input": user_input,
        "interrupt_count": 0,
        "terminate": False,
        "user_inputs": [],
        "messages": [],
        "user_messages": [],
        "skills": None,
        "intent": "",
        "agent_outcome": None,
        "agent_scratchpad": "",
        "intermediate_steps": [],
        "job_description": None,
    }

def askLLM(query: str) -> str:
    try:
        prompt = system_query_prompt.format(
            input=query,
            agent_scratchpad=""
        )
        
        resp = llm.predict(prompt)
        return resp.strip()

    except Exception as e:
        print(f"[Error]: {str(e)}")

    


# ─── State transition functions ────────────────────────────────────────────────
def run_agent(state: AgentState) -> AgentState:
    outcome = agent_runnable.invoke(state)
    state["agent_outcome"] = outcome

    if isinstance(outcome, AgentAction):
        name = outcome.tool
        result = tools_map[name].invoke(outcome.tool_input)
        #state["intermediate_steps"].append((outcome, str(result)))

        if name == "classify_intent":
            state["intent"] = result.strip().lower()
        else:
            state["user_messages"].append(AIMessage(content=str(result)))

        state["agent_scratchpad"] += f"Executed tool: {name}, Result: {result}\n"
        return state

    if isinstance(outcome, AgentFinish):
        state["user_messages"].append(AIMessage(content=outcome.return_values["output"]))
        return state

    return state


def handle_general_query(state: AgentState) -> AgentState:
    prompt = agent_prompt.format(
        input=state["input"],
        agent_scratchpad=state.get("agent_scratchpad", "")
    )
    resp = llm.predict(prompt)
    state["user_messages"].append(AIMessage(content=resp))
    return state

def get_jobtitle(state: AgentState) -> AgentState:
    prompts = [
        "I will write a job description for this task but before that I need to know the job title. Can you give me one?",
        "Not a valid Job Title. Please provide valid Job Title; provide some job title such as Software Engineer, Data Scientist, etc.",
        "Still seems not a valid Job Title, check again, however I will proceed with the job description generation this time with whatever you provide."
    ]
    max_attempts = 3
    state["interrupt_count"] = 0

    for attempt in range(max_attempts):
        # Choose initial vs. retry prompt
        prompt = prompts[state["interrupt_count"]]
        state["user_messages"].append(SystemMessage(content=prompt))

        job_title = interrupt(prompt)
        state["interrupt_count"] += 1
        job_title = askLLM(
                f"From this text extract the job title if present or make one from this information{job_title}. Return 'no' only if job title is not appropriate or not there"
            ).strip()
        # Validate via LLM; normalize the answer
        pdb.set_trace()
        is_valid = ( job_title.lower() != "no")
        if is_valid:
            state["job_title"] = job_title
            return state  # done as soon as we have valid skills

    state["interrupt_count"] = 0
    return state


def get_skills(state: AgentState) -> AgentState:
    prompts = [
        "Now I need Skills for the job. Please give me a complete list of skills required.",
        "Not a valid skill. Please provide valid skills such as Java, Python, etc.",
        "Still seems not a valid skill, check again, however I will proceed with the job description generation this time with whatever you provide."
    ]
    max_attempts = 3
    state["interrupt_count"] = 0

    for attempt in range(max_attempts):
        # Choose initial vs. retry prompt
        prompt = prompts[state["interrupt_count"]]
        state["user_messages"].append(SystemMessage(content=prompt))

        skills_required = interrupt(prompt)
        state["interrupt_count"] += 1
        skills_required = askLLM(
                f"From this text extract all the possible skills for a job: {skills_required}. Return 'no' only if skills are not there"
            ).strip()
        # Validate via LLM; normalize the answer
        is_valid = ( skills_required.lower() != "no")
        if is_valid:
            state["skills"] = skills_required
            return state  # done as soon as we have valid skills
    
    state["interrupt_count"] = 0
    return state


def generate_job_description(state: AgentState) -> AgentState:
    skills = state["skills"]
    if not skills:
        state["user_messages"].append(SystemMessage(content="No skills provided. Cannot prepare a job description."))
        return state

    prompt = (
        f"Using the following skills: {skills} and these skills only, "
        f"generate a professional job description for a Software Engineer"
        f"for {COMPANY_DETAILS['NAME']} located at {COMPANY_DETAILS['LOCATION']}"
        f"having a mission of {COMPANY_DETAILS['MISSION']} and client base of {COMPANY_DETAILS['CLIENTS']}"
        f"The Company HR should be contacter at: {COMPANY_DETAILS['EMAIL']}. "
    )
    resp = llm.predict(prompt)
    state["job_description"] = resp
    state["user_messages"].append(AIMessage(content=f"Generated Job Description: {resp}"))
    return state
