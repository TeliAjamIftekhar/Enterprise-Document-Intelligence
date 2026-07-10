import os
import boto3
import fitz
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import (
    AWS_REGION,
    S3_BUCKET,
    S3_KEY,
    LOCAL_PDF_PATH
)


def download_pdf_from_s3():
    """
    Download textbook PDF from S3.
    """

    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION
    )

    os.makedirs(
        os.path.dirname(LOCAL_PDF_PATH),
        exist_ok=True
    )

    s3.download_file(
        S3_BUCKET,
        S3_KEY,
        LOCAL_PDF_PATH
    )

    print(f"PDF downloaded: {LOCAL_PDF_PATH}")

    return LOCAL_PDF_PATH


def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF using PyMuPDF.
    """

    print("\nOpening PDF...")

    document = fitz.open(pdf_path)

    print(f"Total Pages: {len(document)}")

    all_text = []

    for page_num in range(len(document)):

        page = document[page_num]

        page_text = page.get_text()

        all_text.append(page_text)

        if page_num % 25 == 0:
            print(f"Processed page {page_num + 1}")

    document.close()

    full_text = "\n".join(all_text)

    print("\nText extraction completed")

    print(f"Total Characters: {len(full_text):,}")

    return full_text


if __name__ == "__main__":

    pdf_path = download_pdf_from_s3()

    text = extract_text_from_pdf(pdf_path)

    print("\nPreview:\n")

    print(text[:2000])