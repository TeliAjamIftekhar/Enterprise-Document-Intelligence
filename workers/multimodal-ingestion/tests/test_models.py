import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models import BookManifest, ContentUnit, Modality
from src.models import BookManifest, ContentUnit, Modality


def test_content_unit_creation():
    unit = ContentUnit(
        unit_id="grade-9-english-kaveri-v1-page-27-unit-001",
        book_id="grade-9-english-kaveri",
        book_version="v1",
        page_number=27,
        sequence=1,
        modality=Modality.PARAGRAPH,
        language="english",
        text="Example textbook paragraph.",
        source_pdf_uri=(
            "s3://edi-documents-ajam-2026/"
            "source-documents/grade-9/"
            "grade-9-english-kaveri/versions/v1/textbook.pdf"
        ),
    )

    assert unit.page_number == 27
    assert unit.modality == Modality.PARAGRAPH


def test_manifest_creation():
    manifest = BookManifest(
        book_id="grade-9-english-kaveri",
        title="Kaveri English Textbook",
        grade=9,
        subject="english",
        language="english",
        board="maharashtra-state-board",
        book_version="v1",
        source_pdf_uri=(
            "s3://edi-documents-ajam-2026/"
            "source-documents/grade-9/"
            "grade-9-english-kaveri/versions/v1/textbook.pdf"
        ),
        page_count=300,
        content_units_key=(
            "derived-artifacts/grade-9/"
            "grade-9-english-kaveri/v1/metadata/content-units.json"
        ),
        page_images_prefix=(
            "derived-artifacts/grade-9/"
            "grade-9-english-kaveri/v1/pages/"
        ),
        figures_prefix=(
            "derived-artifacts/grade-9/"
            "grade-9-english-kaveri/v1/figures/"
        ),
        tables_prefix=(
            "derived-artifacts/grade-9/"
            "grade-9-english-kaveri/v1/tables/"
        ),
        ingestion_status="processing",
    )

    assert manifest.grade == 9
    assert manifest.book_version == "v1"