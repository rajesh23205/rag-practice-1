from document_loader import load_document
from text_chunker import split_text

CHUNK_SIZE = 500


def main():
    doc = load_document()
    chunk_details = split_text(doc, CHUNK_SIZE)

    print(f"Total characters in document: {len(doc)}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Total number of chunks: {len(chunk_details)}")

    if chunk_details:
        print(f"First chunk length: {len(chunk_details[0])}")
        print("First 100 characters of the first chunk:")
        print(chunk_details[0][:100])
        print(f"Last chunk length: {len(chunk_details[-1])}")
    else:
        print("No chunks were created.")


if __name__ == "__main__":
    main()
