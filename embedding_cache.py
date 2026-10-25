import json
from pathlib import Path

from embedding_generator import EMBEDDING_MODEL, generate_embedding

CACHE_FILE = Path("data/embedding_cache.json")


def load_embedding_cache():
    if not CACHE_FILE.exists():
        return {}

    with CACHE_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_embedding_cache(cache):
    with CACHE_FILE.open("w", encoding="utf-8") as file:
        json.dump(cache, file)


def get_embedding(text, cache):
    # Include the model and text to identify an embedding.
    cache_key = f"{EMBEDDING_MODEL}::{text}"

    if cache_key in cache:
        print("Cache HIT: Reusing saved embedding.")
        return cache[cache_key]

    # Migrate entries created using the older cache format.
    if text in cache:
        print("Cache HIT: Migrating existing embedding.")
        embedding = cache.pop(text)
        cache[cache_key] = embedding
        return embedding

    print("Cache MISS: Generating new embedding.")
    embedding = generate_embedding(text)
    cache[cache_key] = embedding

    return embedding
