import json
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import (
    CHUNKS_PATH,
    TOP_K
)

from src.embeddings import get_embedding
from src.vector_store import (
    load_faiss_index,
    search_faiss_index
)


def load_chunks():
    """
    Load chunks metadata from chunks.json.
    """

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks


def retrieve_relevant_chunks(question, top_k=TOP_K):
    """
    Retrieve most relevant textbook chunks for a user question.
    """

    print("Loading FAISS index...")

    index = load_faiss_index()

    print("Loading chunks metadata...")

    chunks = load_chunks()

    print("Generating embedding for question...")

    question_embedding = get_embedding(question)

    print("Searching FAISS index...")

    scores, indices = search_faiss_index(
        index=index,
        query_embedding=question_embedding,
        top_k=top_k
    )

    results = []

    for score, chunk_index in zip(scores, indices):

        if chunk_index == -1:
            continue

        chunk = chunks[int(chunk_index)]

        results.append({
            "score": float(score),
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"]
        })

    return results


if __name__ == "__main__":

    question = "What is the textbook about?"

    results = retrieve_relevant_chunks(
        question=question,
        top_k=5
    )

    print("\nQuestion:")
    print(question)

    print("\nTop Retrieved Chunks:")

    for i, result in enumerate(results, start=1):
        print("\n" + "-" * 60)
        print(f"Result {i}")
        print(f"Score : {result['score']}")
        print(f"Page  : {result['page']}")
        print(f"Source: {result['source']}")
        print("\nText Preview:")
        print(result["text"][:700])