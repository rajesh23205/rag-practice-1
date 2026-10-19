def split_text(text: str, chunk_size: int) -> list[str]:
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    chunks = []

    for start in range(0, len(text), chunk_size):
        chunk = text[start : start + chunk_size]
        chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    print(split_text("abcdefghijklmnopqrstuvwxyz", 4))
    print(split_text("", 4))
