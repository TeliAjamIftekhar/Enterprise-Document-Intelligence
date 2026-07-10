import os
import sys
import json
import time
import argparse
from pathlib import Path

import fitz

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import (
    S3_KEY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNKS_PATH
)

from src.pdf_loader import download_pdf_from_s3
from src.embeddings import get_embedding
from src.vector_store import create_faiss_index, save_faiss_index


def extract_page_texts(pdf_path):
    """
    Extract text page by page from the PDF.
    This helps us keep page numbers for source citations.
    """

    print("\nOpening PDF for page-wise extraction...")

    document = fitz.open(pdf_path)

    page_texts = []

    print(f"Total Pages: {len(document)}")

    for page_index in range(len(document)):
        page = document[page_index]

        text = page.get_text("text")

        page_texts.append({
            "page": page_index + 1,
            "text": text
        })

        if page_index % 25 == 0:
            print(f"Extracted page {page_index + 1}")

    document.close()

    return page_texts


def create_page_chunks(page_texts):
    """
    Create chunks from each page separately.
    Each chunk stores:
    - chunk_id
    - page number
    - source file
    - chunk text
    """

    chunks = []

    step_size = CHUNK_SIZE - CHUNK_OVERLAP

    if step_size <= 0:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    for page_item in page_texts:
        page_number = page_item["page"]

        text = page_item["text"]

        # Clean excessive whitespace
        text = " ".join(text.split())

        if len(text) < 50:
            continue

        start = 0

        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= 50:
                chunks.append({
                    "chunk_id": len(chunks),
                    "source": S3_KEY,
                    "page": page_number,
                    "char_start": start,
                    "char_end": end,
                    "text": chunk_text
                })

            if end == len(text):
                break

            start += step_size

    return chunks


def get_embedding_with_retry(text, max_retries=5):
    """
    Generate embedding with retry handling.
    Useful if Bedrock temporarily throttles requests.
    """

    for attempt in range(max_retries):
        try:
            return get_embedding(text)

        except Exception as error:
            wait_time = 2 ** attempt

            print(f"Embedding error: {error}")
            print(f"Retrying in {wait_time} seconds...")

            time.sleep(wait_time)

    raise RuntimeError("Failed to generate embedding after retries")


def generate_embeddings_for_chunks(chunks):
    """
    Generate embeddings for all chunks.
    """

    embeddings = []

    total = len(chunks)

    print(f"\nGenerating embeddings for {total} chunks...")

    for index, chunk in enumerate(chunks):
        embedding = get_embedding_with_retry(chunk["text"])

        embeddings.append(embedding)

        if (index + 1) % 10 == 0 or index == 0:
            print(f"Embedded {index + 1}/{total} chunks")

        # Small pause to reduce throttling risk
        time.sleep(0.05)

    return embeddings


def save_chunks_json(chunks):
    """
    Save chunks metadata as JSON.
    This is needed later for citations.
    """

    os.makedirs(
        os.path.dirname(CHUNKS_PATH),
        exist_ok=True
    )

    with open(CHUNKS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Chunks metadata saved to: {CHUNKS_PATH}")


def main(max_chunks=None):
    """
    Full textbook processing pipeline.
    """

    print("=" * 70)
    print("TEXTBOOK PROCESSING PIPELINE STARTED")
    print("=" * 70)

    pdf_path = download_pdf_from_s3()

    page_texts = extract_page_texts(pdf_path)

    chunks = create_page_chunks(page_texts)

    print(f"\nTotal chunks created from textbook: {len(chunks)}")

    if max_chunks is not None:
        chunks = chunks[:max_chunks]
        print(f"TEST MODE: Using only first {len(chunks)} chunks")

    if len(chunks) == 0:
        raise ValueError("No chunks were created. Check PDF text extraction.")

    embeddings = generate_embeddings_for_chunks(chunks)

    print("\nCreating FAISS index...")

    index = create_faiss_index(embeddings)

    print(f"Total vectors in FAISS index: {index.ntotal}")

    save_faiss_index(index)

    save_chunks_json(chunks)

    print("\n" + "=" * 70)
    print("TEXTBOOK PROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"FAISS vectors: {index.ntotal}")
    print(f"Chunks saved : {len(chunks)}")
    print(f"Chunk file   : {CHUNKS_PATH}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Use only first N chunks for testing"
    )

    args = parser.parse_args()

    main(max_chunks=args.max_chunks)