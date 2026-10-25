import chromadb

client = chromadb.PersistentClient(path="data/chroma_db")

collection = client.get_or_create_collection(
    name="document_management_test",
    configuration={"hnsw": {"space": "cosine"}},
)

# collection.add(
#     ids=["test_1"],
#     documents=["Python supports reusable functions."],
#     embeddings=[[0.1, 0.2, 0.3]],
#     metadatas=[
#         {
#             "source": "test_document.txt",
#             "topic": "functions",
#         }
#     ],
# )

collection.upsert(
    ids=["test_1"],
    documents=["Python supports reusable functions and modules."],
    embeddings=[[0.2, 0.3, 0.4]],
    metadatas=[
        {
            "source": "test_document.txt",
            "topic": "functions",
        }
    ],
)

collection.upsert(
    ids=["test_2"],
    documents=["JavaScript supports reusable functions."],
    embeddings=[[0.4, 0.5, 0.6]],
    metadatas=[
        {
            "source": "javascript.txt",
            "topic": "functions",
        }
    ],
)

collection.delete(where={"source": "test_document.txt"})

print(
    collection.get(
        include=["documents", "metadatas"],
    )
)
