import os
import random
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import json

# Load environment so GPT_MODEL_NAME is available
from dotenv import load_dotenv, find_dotenv

# look specifically for “.env.local” up the directory tree
dotenv_path = find_dotenv(".env.local")
if not dotenv_path:
    raise FileNotFoundError("Could not find .env.local in any parent folder")
load_dotenv(dotenv_path)

llm = ChatOpenAI(model=os.getenv("GPT_MODEL_NAME"))

here = os.path.dirname(__file__)
hiring_intent_data_file = os.path.join(here, "data", "hiring_intent.txt")

@tool("classify_intent", return_direct=True)
def classify_intent(input_text: str) -> str:
    """Classify the user's intent into one of: hiring, general_query."""

    examples = []
    try:
        with open(hiring_intent_data_file, "r") as f:
            examples = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print("⚠️ hiring_intent.txt file not found. Proceeding without examples.")

    examples_text = "\n".join(f"- {ex}" for ex in examples)

    prompt = f"""
You are an intent classifier. Read the following user input and classify the intent into one of these categories:
"hiring" or "general_query".

Examples of *hiring* intent:
{examples_text}

Now classify this user input:
"{input_text}"

Just return one of the two categories: hiring or general_query.
"""

    result = llm.invoke(prompt)
    return result.content.strip().lower()

@tool("random_number", return_direct=True)
def random_number_maker(input_text: str) -> str:
    """Return a random integer between 0 and 100 as a string."""
    return "Generated Random Number = " + str(random.randint(0, 100))


from langchain.tools import tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain.tools import tool
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@tool("match_profile_to_job", return_direct=False)
def match_profile_to_job(input: str) -> str:
    """
    Compares a candidate profile with a job description and returns a match score.

    **Input**: A JSON‐encoded string with two keys:
      - "job_description": the text of the job description
      - "user_profile": the text of the candidate’s profile

    **Returns**: 
      - If both texts are present, a float similarity score between 0 and 1.
      - Otherwise, an error message string.
    """
    try:
        data = json.loads(input)
        jd = data.get("job_description", "")
        profile = data.get("user_profile", "")

        if not jd.strip() or not profile.strip():
            return "Job description or profile text is missing."

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([jd, profile])
        similarity = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])

        return str(similarity)

    except Exception as e:
        return f"Error occurred: {str(e)}"

