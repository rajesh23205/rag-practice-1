import os
import sys

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL")

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY is missing. Please configure it in your .env file.")
    sys.exit(1)

if not OPENAI_DEFAULT_MODEL:
    print(
        "Error: OPENAI_DEFAULT_MODEL is missing. Please configure it in your .env file."
    )
    sys.exit(1)
