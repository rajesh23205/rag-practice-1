import chromadb

from embedding_generator import generate_embedding

# Create an in-memory ChromaDB client.
client = chromadb.PersistentClient(path="data/chroma_db")

# Create a collection that uses cosine distance.
collection = client.get_or_create_collection(
    name="python_knowledge", configuration={"hnsw": {"space": "cosine"}}
)

# Define our sample documents.
documents = [
    "Python functions are reusable blocks of code.",
    "Python lists store multiple items in a single variable.",
    "Python dictionaries store data as key-value pairs.",
]

# Generate OpenAI embeddings for each document.
embeddings = [generate_embedding(document) for document in documents]

# Store documents and their embeddings in ChromaDB.
collection.upsert(
    ids=["doc1", "doc2", "doc3"],
    documents=documents,
    embeddings=embeddings,
)

print("Documents stored:", collection.count())


# Define the question.
question = "How do I group reusable code?"

# Generate an OpenAI embedding for the question.
question_embedding = generate_embedding(question)

# Search using the question embedding.
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2,
)

print("Distances:", results["distances"][0])

print("\nSearch results:")
for document in results["documents"][0]:
    print("-", document)
