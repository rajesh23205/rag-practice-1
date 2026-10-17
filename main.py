from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_DEFAULT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.responses.create(
    model=OPENAI_DEFAULT_MODEL,
    input="Say: We are up and running!",
)

print(response.output_text)
