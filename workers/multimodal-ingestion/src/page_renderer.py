from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

from src.models import PageArtifact


def calculate_file_sha256(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def build_page_filename(page_number: int) -> str:
    if page_number < 1:
        raise ValueError("Page number must be at least 1.")

    return f"page-{page_number:04d}"


def render_page_as_jpeg(
    page: fitz.Page,
    output_path: Path,
    dpi: int,
    jpeg_quality: int,
) -> tuple[int, int]:
    """
    Render a PDF page in RGB and save it as an optimized JPEG.

    Full-page JPEG files are used as reference images. Higher-quality
    PNG crops will be created later for figures and diagrams.
    """
    if dpi < 72:
        raise ValueError("DPI must be at least 72.")

    if not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100.")

    pixmap = page.get_pixmap(
        dpi=dpi,
        colorspace=fitz.csRGB,
        alpha=False,
        annots=True,
    )

    image = Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    image.save(
        output_path,
        format="JPEG",
        quality=jpeg_quality,
        optimize=True,
        progressive=True,
        dpi=(dpi, dpi),
    )

    image.close()

    return pixmap.width, pixmap.height


def render_pages(
    pdf_path: Path,
    output_root: Path,
    bucket: str,
    source_pdf_uri: str,
    book_id: str,
    book_version: str,
    derived_prefix: str,
    start_page: int,
    end_page: int,
    dpi: int = 150,
    jpeg_quality: int = 88,
    overwrite: bool = False,
) -> list[PageArtifact]:
    """
    Render an inclusive, one-based page range.

    Example:
        start_page=1, end_page=5
        renders PDF indexes 0 through 4.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if start_page < 1:
        raise ValueError("start_page must be at least 1.")

    if end_page < start_page:
        raise ValueError("end_page must be greater than or equal to start_page.")

    page_artifacts: list[PageArtifact] = []

    image_dir = output_root / "pages"
    metadata_dir = output_root / "metadata" / "pages"

    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as document:
        if document.needs_pass:
            raise RuntimeError("The PDF is password protected.")

        if end_page > document.page_count:
            raise ValueError(
                f"Requested end page {end_page} exceeds "
                f"PDF page count {document.page_count}."
            )

        for page_number in range(start_page, end_page + 1):
            page_index = page_number - 1
            page = document.load_page(page_index)

            base_name = build_page_filename(page_number)
            image_path = image_dir / f"{base_name}.jpg"
            metadata_path = metadata_dir / f"{base_name}.json"

            image_s3_key = (
                f"{derived_prefix}/pages/{base_name}.jpg"
            )
            metadata_s3_key = (
                f"{derived_prefix}/metadata/pages/{base_name}.json"
            )

            if image_path.exists() and not overwrite:
                print(
                    f"Page {page_number}: local image exists; "
                    "render skipped."
                )

                with Image.open(image_path) as existing_image:
                    pixel_width, pixel_height = existing_image.size

            else:
                print(f"Page {page_number}: rendering...")

                pixel_width, pixel_height = render_page_as_jpeg(
                    page=page,
                    output_path=image_path,
                    dpi=dpi,
                    jpeg_quality=jpeg_quality,
                )

            image_size = image_path.stat().st_size
            image_sha256 = calculate_file_sha256(image_path)

            page_rect = page.rect

            artifact = PageArtifact(
                book_id=book_id,
                book_version=book_version,
                page_number=page_number,
                source_pdf_uri=source_pdf_uri,
                local_image_path=str(image_path),
                image_s3_key=image_s3_key,
                metadata_s3_key=metadata_s3_key,
                image_format="jpeg",
                dpi=dpi,
                pixel_width=pixel_width,
                pixel_height=pixel_height,
                pdf_width_points=float(page_rect.width),
                pdf_height_points=float(page_rect.height),
                file_size_bytes=image_size,
                image_sha256=image_sha256,
            )

            metadata_path.write_text(
                json.dumps(
                    artifact.model_dump(mode="json"),
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            page_artifacts.append(artifact)

    return page_artifacts


def upload_page_artifacts(
    s3_client: Any,
    bucket: str,
    output_root: Path,
    artifacts: list[PageArtifact],
) -> None:
    for artifact in artifacts:
        image_path = Path(artifact.local_image_path)

        metadata_path = (
            output_root
            / "metadata"
            / "pages"
            / f"{build_page_filename(artifact.page_number)}.json"
        )

        print(
            f"Page {artifact.page_number}: uploading image and metadata..."
        )

        s3_client.upload_file(
            str(image_path),
            bucket,
            artifact.image_s3_key,
            ExtraArgs={
                "ContentType": "image/jpeg",
                "ServerSideEncryption": "AES256",
                "Metadata": {
                    "book-id": artifact.book_id,
                    "book-version": artifact.book_version,
                    "page-number": str(artifact.page_number),
                    "artifact-type": "page-image",
                    "sha256": artifact.image_sha256,
                },
            },
        )

        s3_client.upload_file(
            str(metadata_path),
            bucket,
            artifact.metadata_s3_key,
            ExtraArgs={
                "ContentType": "application/json",
                "ServerSideEncryption": "AES256",
                "Metadata": {
                    "book-id": artifact.book_id,
                    "book-version": artifact.book_version,
                    "page-number": str(artifact.page_number),
                    "artifact-type": "page-metadata",
                },
            },
        )
