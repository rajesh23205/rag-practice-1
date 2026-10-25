import hashlib
from pathlib import Path

import chromadb

from document_loader import load_document
from embedding_cache import (
    get_embedding,
    load_embedding_cache,
    save_embedding_cache,
)
from text_chunker import split_text

CHUNK_SIZE = 500
OVERLAP_SIZE = 50


def get_hash_file_path(filename: str) -> Path:
    return Path("data/hashes") / f"{filename}.hash"


def main(filename: str):
    # Step 1: Load the knowledge document.
    document = load_document(filename)
    hash_file = get_hash_file_path(filename)

    chunks = split_text(
        document,
        chunk_size=CHUNK_SIZE,
        overlap_size=OVERLAP_SIZE,
    )

    document_hash = hashlib.sha256(document.encode("utf-8")).hexdigest()

    print("Document hash:", document_hash)

    if hash_file.exists():
        previous_hash = hash_file.read_text(encoding="utf-8").strip()
    else:
        previous_hash = None

    print("Previous hash:", previous_hash)

    if previous_hash == document_hash:
        print("DEBUG: Document hashes match.")
        chroma_client = chromadb.PersistentClient(path="data/chroma_db")

        collection = chroma_client.get_or_create_collection(
            name="knowledge_chunks",
            configuration={"hnsw": {"space": "cosine"}},
        )

        expected_ids = [f"{filename}_chunk_{index}" for index in range(len(chunks))]

        existing_chunks = collection.get(where={"source": filename})

        if set(existing_chunks["ids"]) == set(expected_ids):
            print("Knowledge base unchanged. Skipping indexing.")
            return

    # Step 2: Split the document into chunks.
    chunks = split_text(
        document,
        chunk_size=CHUNK_SIZE,
        overlap_size=OVERLAP_SIZE,
    )

    if not chunks:
        print("No chunks were created.")
        return

    print(f"Total chunks created: {len(chunks)}")

    # Step 3: Generate or reuse embeddings.
    embedding_cache = load_embedding_cache()

    embeddings = [get_embedding(chunk, embedding_cache) for chunk in chunks]

    # Step 4: Connect to the persistent ChromaDB database.
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")

    collection = chroma_client.get_or_create_collection(
        name="knowledge_chunks",
        configuration={"hnsw": {"space": "cosine"}},
    )

    # Step 5: Replace the existing index with the current chunks.
    collection.delete(where={"source": filename})

    collection.upsert(
        ids=[f"{filename}_chunk_{index}" for index in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "source": filename,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ],
    )

    # Step 6: Save the embedding cache.
    save_embedding_cache(embedding_cache)

    # Save the hash of the successfully indexed document.
    hash_file.parent.mkdir(parents=True, exist_ok=True)
    hash_file.write_text(document_hash, encoding="utf-8")

    print(collection.get(limit=2, include=["metadatas"]))

    print(f"Chunks stored in ChromaDB: {collection.count()}")
    print("Knowledge-base indexing completed.")


if __name__ == "__main__":
    main("knowledge.txt")
    main("javascript.txt")
