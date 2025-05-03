import os
import random
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# Load environment so GPT_MODEL_NAME is available
from dotenv import load_dotenv, find_dotenv

# look specifically for “.env.local” up the directory tree
dotenv_path = find_dotenv(".env.local")
if not dotenv_path:
    raise FileNotFoundError("Could not find .env.local in any parent folder")
load_dotenv(dotenv_path)

llm = ChatOpenAI(model=os.getenv("GPT_MODEL_NAME"))

@tool("classify_intent", return_direct=True)
def classify_intent(input_text: str) -> str:
    """Classify the user's intent into one of: hiring, general_query, greeting, feedback, unknown."""
    prompt = f"""
You are an intent classifier. Read the following user input and classify the intent into one of these categories:
"hiring", "general_query".

User input: "{input_text}"

Just return one of the two categories, nothing else.
"""
    result = llm.invoke(prompt)
    return result.content.strip().lower()

@tool("random_number", return_direct=True)
def random_number_maker(input_text: str) -> str:
    """Return a random integer between 0 and 100 as a string."""
    return "Generated Random Number = " + str(random.randint(0, 100))

tools = [classify_intent, random_number_maker]
tools_map = {t.name: t for t in tools}
