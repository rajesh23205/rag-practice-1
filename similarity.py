import numpy as np


def cosine_similarity(
    question_embedding: list[float],
    chunk_embedding: list[float],
) -> float:
    """Calculate cosine similarity between two embedding vectors."""

    question = np.asarray(question_embedding, dtype=float)
    chunk = np.asarray(chunk_embedding, dtype=float)

    # Both inputs must be one-dimensional vectors.
    if question.ndim != 1 or chunk.ndim != 1:
        raise ValueError("Both embeddings must be one-dimensional vectors.")

    # Both vectors must have the same number of values.
    if question.shape != chunk.shape:
        raise ValueError("Embeddings must have the same dimensions.")

    # Embeddings must contain finite numbers.
    if not np.all(np.isfinite(question)) or not np.all(np.isfinite(chunk)):
        raise ValueError("Embeddings must contain only finite numbers.")

    # Calculate the magnitude of each vector.
    question_magnitude = np.linalg.norm(question)
    chunk_magnitude = np.linalg.norm(chunk)

    # Prevent division by zero.
    if question_magnitude == 0 or chunk_magnitude == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")

    # Calculate cosine similarity.
    similarity = np.dot(question, chunk) / (question_magnitude * chunk_magnitude)

    return float(similarity)


if __name__ == "__main__":
    print(cosine_similarity([1, 0], [1, 0]))  # 1.0
    print(cosine_similarity([1, 0], [0, 1]))  # 0.0
    print(cosine_similarity([1, 0], [-1, 0]))  # -1.0
