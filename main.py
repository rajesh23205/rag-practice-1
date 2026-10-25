import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from embedding_cache import (
    get_embedding,
    load_embedding_cache,
    save_embedding_cache,
)
from query_rewriter import rewrite_query

# Load environment variables and create the OpenAI client.
load_dotenv()
client = OpenAI()

MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-nano")

# Configuration
TOP_K = 3
SIMILARITY_THRESHOLD = 0.3


def main():
    # Step 1: Connect to the existing ChromaDB collection.
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")

    try:
        collection = chroma_client.get_collection(name="knowledge_chunks")
    except Exception:
        print("Knowledge base not found. Run: python index_knowledge.py")
        return

    if collection.count() == 0:
        print("Knowledge base is empty. Run: python index_knowledge.py")
        return

    print("Knowledge-base chunks available:", collection.count())

    # Step 2: Load the embedding cache.
    embedding_cache = load_embedding_cache()

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
            save_embedding_cache(embedding_cache)
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

        # Step 6: Generate or reuse the question embedding.
        question_embedding = get_embedding(
            search_query,
            embedding_cache,
        )
        save_embedding_cache(embedding_cache)

        # Step 7: Search the existing ChromaDB collection.
        search_results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(collection.count(), max(TOP_K, 10)),
            include=["documents", "distances", "metadatas"],
        )

        # Step 8: Filter results by cosine similarity.
        scores = []

        result_ids = search_results["ids"][0]
        result_documents = search_results["documents"][0]
        result_distances = search_results["distances"][0]
        result_metadatas = search_results["metadatas"][0]

        for chunk_id, document, distance, metadata in zip(
            result_ids,
            result_documents,
            result_distances,
            result_metadatas,
        ):
            score = 1 - distance

            if score > SIMILARITY_THRESHOLD:
                source = metadata["source"]
                scores.append((score, chunk_id, document, source))

        # Step 9: Select the top matching chunks.
        scores.sort(key=lambda result: result[0], reverse=True)
        top_results = scores[:TOP_K]

        if not top_results:
            print(
                "\nAssistant: I couldn't find relevant information "
                "in the knowledge base to answer that question."
            )
            continue

        print("\nRetrieved chunks:")

        for score, chunk_id, document, source in top_results:
            print(f"\nSimilarity: {score:.4f}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Source: {source}")
            print(document)

        # Step 10: Build context from the retrieved documents.
        if top_results:
            context = "\n\n".join(
                f"Source: {source}\n{document}"
                for score, chunk_id, document, source in top_results
            )
        else:
            context = (
                "No relevant knowledge-base chunks were retrieved. "
                "Use conversation history if it contains enough "
                "information to answer the question."
            )

        # Step 11: Format the previous conversation.
        history_text = ""

        for turn in conversation_history:
            history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n\n"

        # Step 12: Generate an answer using context and history.
        response = client.responses.create(
            model=MODEL,
            instructions=(
                "You are a helpful knowledge assistant. "
                "Use the retrieved context to answer questions "
                "about the knowledge base. Use conversation history "
                "to understand follow-up questions and references "
                "to earlier messages. Do not invent facts that are "
                "not supported by the context or conversation history. "
                "If there is not enough information, say so. "
                "When using information from the retrieved context, "
                "cite the relevant source filename in your answer."
            ),
            input=(
                f"Conversation history:\n{history_text}\n"
                f"Retrieved context:\n{context}\n\n"
                f"Current question:\n{question}"
            ),
        )

        answer = response.output_text

        # Step 13: Display the answer.
        print(f"\nAssistant: {answer}")

        # Step 14: Save this exchange in session memory.
        conversation_history.append(
            {
                "question": question,
                "answer": answer,
            }
        )


if __name__ == "__main__":
    main()
