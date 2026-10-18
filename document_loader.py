from pathlib import Path


def load_document() -> str:
    """Load the knowledge document and return its text."""
    file_path = Path(__file__).parent / "data" / "knowledge.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Knowledge document not found: {file_path}")

    if not file_path.is_file():
        raise OSError(f"The knowledge path is not a file: {file_path}")

    text = file_path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("The knowledge document is empty.")

    return text


if __name__ == "__main__":
    try:
        text = load_document()

        print("Document loaded successfully!")
        print(f"Total characters: {len(text)}")
        print("\nPreview (first 100 characters):")
        print(text[:100])

    except (OSError, ValueError) as error:
        print(f"Error: {error}")
