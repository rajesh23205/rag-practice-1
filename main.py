import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from document_loader import load_document
from embedding_generator import EMBEDDING_MODEL, generate_embedding
from query_rewriter import rewrite_query
from similarity import cosine_similarity
from text_chunker import split_text

# Load environment variables and create the OpenAI client.
load_dotenv()
client = OpenAI()

MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-nano")
CACHE_FILE = Path("data/embedding_cache.json")


# Configuration
CHUNK_SIZE = 500
OVERLAP_SIZE = 50
TOP_K = 3
SIMILARITY_THRESHOLD = 0.3


def load_embedding_cache():
    if not CACHE_FILE.exists():
        return {}

    with CACHE_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_embedding_cache(cache):
    with CACHE_FILE.open("w", encoding="utf-8") as file:
        json.dump(cache, file)


def get_embedding(text, cache):
    # Identify the embedding using both model and text.
    cache_key = f"{EMBEDDING_MODEL}::{text}"

    # Check for the new model-aware cache key.
    if cache_key in cache:
        print("Cache HIT: Reusing saved embedding.")
        return cache[cache_key]

    # Migrate an existing entry from the old cache format.
    if text in cache:
        print("Cache HIT: Migrating existing embedding.")
        embedding = cache.pop(text)
        cache[cache_key] = embedding
        return embedding

    # Generate an embedding if no matching entry exists.
    print("Cache MISS: Generating new embedding.")
    embedding = generate_embedding(text)
    cache[cache_key] = embedding

    return embedding


def main():
    # Step 1: Load and prepare the knowledge base.
    doc = load_document()

    chunk_details = split_text(
        doc,
        chunk_size=CHUNK_SIZE,
        overlap_size=OVERLAP_SIZE,
    )

    if not chunk_details:
        print("No chunks were created.")
        return

    # Step 2: Load cached embeddings.
    embedding_cache = load_embedding_cache()

    # Generate embeddings only for chunks not already cached.
    chunk_embeddings = []

    for chunk in chunk_details:
        embedding = get_embedding(chunk, embedding_cache)
        chunk_embeddings.append(embedding)

    # Save newly generated embeddings.
    save_embedding_cache(embedding_cache)

    # Step 3: Create memory for this conversation session.
    conversation_history = []

    print("\nAI Knowledge Assistant")
    print("Ask questions about your knowledge base.")
    print("Type 'exit' to end the session.")

    # Step 4: Keep accepting questions.
    while True:
        question = input("\nYou: ").strip()

        if not question:
            print("Please enter a question.")
            continue

        if question.lower() in ("exit", "quit"):
            print("Session ended.")
            break

        # Step 5: Rewrite the question using conversation history.
        search_query = rewrite_query(
            question=question,
            conversation_history=conversation_history,
            client=client,
            model=MODEL,
        )

        print(f"Search query: {search_query}")

        # Step 6: Generate an embedding for the search query.
        question_embedding = generate_embedding(search_query)

        # Step 7: Find relevant chunks.
        scores = []

        for index, chunk_embedding in enumerate(chunk_embeddings):
            score = cosine_similarity(
                question_embedding,
                chunk_embedding,
            )

            if score > SIMILARITY_THRESHOLD:
                scores.append((score, index))

        # Step 8: Sort scores and select the top matching chunks.
        scores.sort(reverse=True)
        top_results = scores[:TOP_K]

        # Step 9: Build context from the retrieved chunks.
        if top_results:
            context = "\n\n".join(chunk_details[index] for score, index in top_results)
        else:
            context = (
                "No relevant knowledge-base chunks were retrieved. "
                "Use the conversation history if it contains enough "
                "information to answer the question."
            )

        # Step 10: Format the previous conversation.
        history_text = ""

        for turn in conversation_history:
            history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n\n"

        # Step 11: Generate an answer using context and history.
        response = client.responses.create(
            model=MODEL,
            instructions=(
                "You are a helpful knowledge assistant. "
                "Use the retrieved context to answer questions "
                "about the knowledge base. Use conversation history "
                "to understand follow-up questions and references "
                "to earlier messages. Do not invent facts that are "
                "not supported by the context or conversation history. "
                "If there is not enough information, say so."
            ),
            input=(
                f"Conversation history:\n{history_text}\n"
                f"Retrieved context:\n{context}\n\n"
                f"Current question:\n{question}"
            ),
        )

        answer = response.output_text

        # Step 12: Display the answer.
        print(f"\nAssistant: {answer}")

        # Step 13: Save this exchange in session memory.
        conversation_history.append(
            {
                "question": question,
                "answer": answer,
            }
        )


if __name__ == "__main__":
    main()
