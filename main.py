import os

from dotenv import load_dotenv
from openai import OpenAI

from document_loader import load_document
from embedding_generator import generate_embedding
from similarity import cosine_similarity
from text_chunker import split_text

# Load environment variables from .env.
load_dotenv()

# Create the OpenAI client.
client = OpenAI()


# Configuration
CHUNK_SIZE = 500
OVERLAP_SIZE = 50
TOP_K = 3
SIMILARITY_THRESHOLD = 0.3


def main():
    # Step 1: Load the knowledge document.
    doc = load_document()

    # Step 2: Split the document into chunks.
    chunk_details = split_text(
        doc,
        chunk_size=CHUNK_SIZE,
        overlap_size=OVERLAP_SIZE,
    )

    if not chunk_details:
        print("No chunks were created.")
        return

    # Step 3: Generate an embedding for each chunk.
    chunk_embeddings = []

    for chunk in chunk_details:
        embedding = generate_embedding(chunk)
        chunk_embeddings.append(embedding)

    # Step 4: Generate an embedding for the user's question.
    question = input("\nEnter your question: ").strip()

    if not question:
        print("Please enter a question.")
        return
    question_embedding = generate_embedding(question)

    # Step 5: Calculate similarity between the question
    # embedding and each chunk embedding.
    scores = []

    for index, chunk_embedding in enumerate(chunk_embeddings):
        score = cosine_similarity(
            question_embedding,
            chunk_embedding,
        )

        # Keep chunks that pass the similarity threshold.
        if score > SIMILARITY_THRESHOLD:
            scores.append((score, index))

    if not scores:
        print("No information available")
        return

    # Step 6: Sort by similarity, highest first.
    scores.sort(reverse=True)

    # Step 7: Retrieve and display the top matching chunks.
    top_results = scores[:TOP_K]

    print(f"\nQuestion: {question}")
    print(f"Total chunks: {len(chunk_details)}")
    print(f"Showing top {len(top_results)} results")

    for rank, (score, index) in enumerate(top_results, start=1):
        print(f"\n{'=' * 50}")
        print(f"Rank: {rank}")
        print(f"Similarity score: {score:.4f}")
        print(f"Chunk index: {index}")
        print(f"{'=' * 50}")
        print(chunk_details[index])

    # Step 8: Build context from the retrieved chunks.
    context = "\n\n".join(chunk_details[index] for score, index in top_results)

    # Step 9: Generate an answer using the retrieved context.
    response = client.responses.create(
        model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-nano"),
        instructions=(
            "Answer the user's question using only the provided context. "
            "Explain the answer clearly and simply. "
            "If the context does not contain enough information, "
            "say that the knowledge base does not contain enough "
            "information to answer the question."
        ),
        input=(f"Context:\n{context}\n\nQuestion:\n{question}"),
    )

    # Step 10: Display the generated answer.
    print(f"\n{'=' * 50}")
    print("Generated Answer")
    print(f"{'=' * 50}")
    print(response.output_text)

    # Step 11: Display summary information.
    print(f"\n{'=' * 50}")
    print("Pipeline Summary")
    print(f"{'=' * 50}")
    print(f"Total characters: {len(doc)}")
    print(f"Total chunks: {len(chunk_details)}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Total embeddings: {len(chunk_embeddings)}")
    print(f"Embedding dimensions: {len(chunk_embeddings[0])}")


if __name__ == "__main__":
    main()
