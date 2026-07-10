import os
import faiss
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import FAISS_INDEX_PATH


def normalize_vectors(vectors):
    """
    Normalize vectors so FAISS can perform cosine similarity search.
    """

    vectors = np.array(vectors).astype("float32")

    faiss.normalize_L2(vectors)

    return vectors


def create_faiss_index(embeddings):
    """
    Create a FAISS index from embedding vectors.
    """

    vectors = normalize_vectors(embeddings)

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    return index


def save_faiss_index(index):
    """
    Save FAISS index to local disk.
    """

    os.makedirs(
        os.path.dirname(FAISS_INDEX_PATH),
        exist_ok=True
    )

    faiss.write_index(
        index,
        FAISS_INDEX_PATH
    )

    print(f"FAISS index saved to: {FAISS_INDEX_PATH}")


def load_faiss_index():
    """
    Load FAISS index from local disk.
    """

    index = faiss.read_index(
        FAISS_INDEX_PATH
    )

    return index


def search_faiss_index(index, query_embedding, top_k=5):
    """
    Search FAISS index using a query embedding.
    """

    query_vector = normalize_vectors([query_embedding])

    scores, indices = index.search(
        query_vector,
        top_k
    )

    return scores[0], indices[0]


if __name__ == "__main__":

    sample_embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.1, 0.4, 0.3],
        [0.9, 0.8, 0.1, 0.2],
        [0.8, 0.9, 0.2, 0.1],
    ]

    print("Creating FAISS index...")

    index = create_faiss_index(sample_embeddings)

    print(f"Total vectors in index: {index.ntotal}")

    save_faiss_index(index)

    print("Loading FAISS index...")

    loaded_index = load_faiss_index()

    query_embedding = [0.1, 0.2, 0.3, 0.4]

    scores, indices = search_faiss_index(
        loaded_index,
        query_embedding,
        top_k=2
    )

    print("\nSearch Results:")

    for score, idx in zip(scores, indices):
        print(f"Index: {idx}, Score: {score}")