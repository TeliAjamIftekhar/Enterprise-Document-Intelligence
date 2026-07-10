import json
import boto3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import (
    AWS_REGION,
    EMBEDDING_MODEL_ID
)


def get_embedding(text):
    """
    Generate embedding using Amazon Titan Embeddings V2
    """

    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION
    )

    body = json.dumps({
        "inputText": text
    })

    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(
        response["body"].read()
    )

    embedding = response_body["embedding"]

    return embedding


if __name__ == "__main__":

    sample_text = """
    The sun rises in the east and sets in the west.
    """

    embedding = get_embedding(sample_text)

    print("\nEmbedding Generated Successfully")

    print(f"Vector Length: {len(embedding)}")

    print("\nFirst 10 Values:")

    print(embedding[:10])