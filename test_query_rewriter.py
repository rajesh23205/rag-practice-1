import os

from dotenv import load_dotenv
from openai import OpenAI

from query_rewriter import rewrite_query

load_dotenv()
client = OpenAI()

model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-nano")

conversation_history = [
    {
        "question": "What is a Python function?",
        "answer": (
            "A function is a reusable block of code. "
            "For example, def add(a, b): return a + b"
        ),
    }
]

question = "Can you show me an example of one?"

rewritten_query = rewrite_query(
    question=question,
    conversation_history=conversation_history,
    client=client,
    model=model,
)

print("Original question:", question)
print("Rewritten query:", rewritten_query)
