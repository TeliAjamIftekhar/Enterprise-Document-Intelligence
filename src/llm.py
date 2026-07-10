import sys
from pathlib import Path

import boto3

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import (
    AWS_REGION,
    LLM_MODEL_ID
)


def generate_answer(prompt):
    """
    Generate an answer using Amazon Bedrock Nova Lite.
    """

    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION
    )

    response = bedrock.converse(
        modelId=LLM_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        inferenceConfig={
            "maxTokens": 700,
            "temperature": 0.2,
            "topP": 0.9
        }
    )

    answer = response["output"]["message"]["content"][0]["text"]

    return answer


if __name__ == "__main__":

    test_prompt = """
    Explain in simple words what an English textbook is used for.
    """

    answer = generate_answer(test_prompt)

    print("\nAnswer Generated Successfully\n")

    print(answer)