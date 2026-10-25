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
HASH_FILE = Path("data/knowledge_hash.txt")


def main():
    # Step 1: Load the knowledge document.
    document = load_document()

    document_hash = hashlib.sha256(document.encode("utf-8")).hexdigest()

    print("Document hash:", document_hash)

    if HASH_FILE.exists():
        previous_hash = HASH_FILE.read_text(encoding="utf-8").strip()
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

        if collection.count() > 0:
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
    existing_ids = collection.get()["ids"]

    if existing_ids:
        collection.delete(ids=existing_ids)

    collection.upsert(
        ids=[f"chunk_{index}" for index in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "source": "knowledge.txt",
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ],
    )

    # Step 6: Save the embedding cache.
    save_embedding_cache(embedding_cache)

    # Save the hash of the successfully indexed document.
    HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    HASH_FILE.write_text(document_hash, encoding="utf-8")

    print(collection.get(limit=2, include=["metadatas"]))

    print(f"Chunks stored in ChromaDB: {collection.count()}")
    print("Knowledge-base indexing completed.")


if __name__ == "__main__":
    main()
