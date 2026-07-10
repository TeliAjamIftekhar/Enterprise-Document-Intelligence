import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import TOP_K

from src.retrieval import retrieve_relevant_chunks
from src.llm import generate_answer


def build_context(retrieved_chunks):
    """
    Convert retrieved chunks into a clean context block for the LLM.
    """

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):

        context_part = f"""
Source {index}
Page: {chunk['page']}
Chunk ID: {chunk['chunk_id']}
Text:
{chunk['text']}
"""

        context_parts.append(context_part)

    return "\n\n".join(context_parts)


def build_rag_prompt(question, retrieved_chunks):
    """
    Build the final RAG prompt that goes to the LLM.
    """

    context = build_context(retrieved_chunks)

    prompt = f"""
You are a helpful AI Textbook Assistant.

Your job is to answer the student's question using ONLY the textbook context provided below.

Rules:
1. Use only the provided textbook context.
2. Do not use outside knowledge.
3. If the answer is not found in the context, say:
   "I could not find this answer in the provided textbook context."
4. Write the answer in simple student-friendly language.
5. Include page citations in the answer wherever useful, like this: (Page 6).

TEXTBOOK CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""

    return prompt


def answer_question(question, top_k=TOP_K):
    """
    Full RAG pipeline:
    question -> retrieval -> prompt -> LLM answer -> citations
    """

    retrieved_chunks = retrieve_relevant_chunks(
        question=question,
        top_k=top_k
    )

    prompt = build_rag_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks
    )

    answer = generate_answer(prompt)

    citations = []

    for chunk in retrieved_chunks:
        citations.append({
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "score": chunk["score"],
            "preview": chunk["text"][:400]
        })

    return {
        "question": question,
        "answer": answer,
        "citations": citations
    }


if __name__ == "__main__":

    question = "What is the Kaveri textbook designed to help learners do?"

    result = answer_question(
        question=question,
        top_k=5
    )

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(result["question"])

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])

    print("\n" + "=" * 70)
    print("SOURCE CITATIONS")
    print("=" * 70)

    for index, citation in enumerate(result["citations"], start=1):
        print("\n" + "-" * 60)
        print(f"Source {index}")
        print(f"Page    : {citation['page']}")
        print(f"Chunk ID: {citation['chunk_id']}")
        print(f"Score   : {citation['score']}")
        print("Preview :")
        print(citation["preview"])