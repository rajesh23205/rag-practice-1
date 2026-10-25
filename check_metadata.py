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

collection.delete(ids=["test_1"])

print(
    collection.get(
        where={"source": "test_document.txt"},
        include=["documents", "metadatas"],
    )
)
