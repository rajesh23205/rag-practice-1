from dotenv import load_dotenv
from openai import OpenAI

# Load the API key from the .env file.
load_dotenv()

# Define the embedding model in one place.
EMBEDDING_MODEL = "text-embedding-3-small"

# Create the OpenAI client.
client = OpenAI()


def generate_embedding(text):
    # Generate an embedding using the configured model.
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    # Extract the numerical vector.
    return response.data[0].embedding


if __name__ == "__main__":
    # Text we want to convert into a numerical vector.
    text = "Python is useful for building AI applications."
    embedding = generate_embedding(text)
    print("Original text:", text)
    print("Embedding model:", EMBEDDING_MODEL)
    print("Number of dimensions:", len(embedding))
    print("First 10 numbers:", embedding[:10])
