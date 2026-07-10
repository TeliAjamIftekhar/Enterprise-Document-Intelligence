import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import CHUNK_SIZE, CHUNK_OVERLAP


def create_chunks(text):
    """
    Split textbook text into overlapping chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end]

        chunks.append(chunk)

        start += (CHUNK_SIZE - CHUNK_OVERLAP)

    return chunks


if __name__ == "__main__":

    sample_text = "A" * 5000

    chunks = create_chunks(sample_text)

    print(f"Chunks Created: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1} Length: {len(chunk)}")