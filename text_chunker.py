import re


def split_text(
    text: str,
    chunk_size: int,
    overlap_size: int = 50,
) -> list[str]:
    """
    Split text into chunks while preserving sentence and heading
    boundaries where possible.

    Args:
        text: The document text to split.
        chunk_size: Target maximum number of characters per chunk.
        overlap_size: Approximate number of overlapping characters.

    Returns:
        A list of text chunks.
    """

    # Step 1: Validate the input.
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap_size < 0:
        raise ValueError("overlap_size cannot be negative")

    if overlap_size >= chunk_size:
        raise ValueError("overlap_size must be smaller than chunk_size")

    # Step 2: Split the text into sentences and Markdown headings.
    # Keep headings as separate units so they are not accidentally
    # attached to the end of an unrelated sentence.
    units = re.split(
        r"(?m)(?=^#{1,6}\s)|(?<=[.!?])\s+",
        text.strip(),
    )

    units = [unit.strip() for unit in units if unit.strip()]

    chunks: list[str] = []
    current_chunk = ""

    # Step 3: Process each sentence or heading.
    for unit in units:
        # Handle a single unit longer than the target chunk size.
        if len(unit) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            start = 0

            while start < len(unit):
                end = min(start + chunk_size, len(unit))

                # Prefer a word boundary when splitting long text.
                if end < len(unit):
                    boundary = unit.rfind(" ", start, end)

                    if boundary > start:
                        end = boundary

                piece = unit[start:end].strip()

                if piece:
                    chunks.append(piece)

                if end >= len(unit):
                    break

                # Find the next starting position without splitting a word.
                next_start = max(start + 1, end - overlap_size)

                if next_start > 0 and not unit[next_start - 1].isspace():
                    next_space = unit.find(" ", next_start)

                    if next_space != -1 and next_space < end:
                        next_start = next_space + 1

                start = next_start

            continue

        # Step 4: Add the unit to the current chunk if it fits.
        candidate = f"{current_chunk}\n{unit}".strip() if current_chunk else unit

        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue

        # Step 5: Save the completed chunk.
        if current_chunk:
            chunks.append(current_chunk)

        # Step 6: Create overlap from the previous chunk.
        overlap = ""

        if chunks and overlap_size > 0:
            previous_chunk = chunks[-1]

            # Find an approximate overlap starting position.
            start = max(0, len(previous_chunk) - overlap_size)

            # Move forward to the next word boundary.
            if start > 0:
                space_index = previous_chunk.find(" ", start)

                if space_index != -1:
                    start = space_index + 1

            overlap = previous_chunk[start:].strip()

        # Step 7: Start the next chunk with the overlap and new unit.
        current_chunk = f"{overlap}\n{unit}".strip() if overlap else unit

        # If overlap makes the chunk too large, prioritize the new unit.
        if len(current_chunk) > chunk_size:
            current_chunk = unit

    # Step 8: Save the final chunk.
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


if __name__ == "__main__":
    sample_text = (
        "# Python Basics\n"
        "Python is easy to learn. "
        "Functions organize code into reusable blocks. "
        "Exceptions help handle errors safely. "
        "# File Handling\n"
        "Files allow programs to store information."
    )

    print("Test 1: Normal text")
    for index, chunk in enumerate(
        split_text(sample_text, chunk_size=80, overlap_size=15),
        start=1,
    ):
        print(f"\nChunk {index} ({len(chunk)} characters):")
        print(chunk)

    print("\nTest 2: Empty text")
    print(split_text("", chunk_size=80, overlap_size=15))

    print("\nTest 3: Whitespace-only text")
    print(split_text("   \n  ", chunk_size=80, overlap_size=15))

    print("\nTest 4: One sentence")
    print(split_text("Python is easy to learn.", chunk_size=80))

    print("\nTest 5: Invalid overlap")
    try:
        split_text(sample_text, chunk_size=20, overlap_size=20)
    except ValueError as error:
        print(error)

    print("\nTest 6: Invalid chunk size")
    try:
        split_text(sample_text, chunk_size=0)
    except ValueError as error:
        print(error)
