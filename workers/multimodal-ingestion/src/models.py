from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Modality(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    DIAGRAM = "diagram"
    FLOWCHART = "flowchart"
    GRAPH = "graph"
    ILLUSTRATION = "illustration"
    INFOGRAPHIC = "infographic"
    EQUATION = "equation"
    PAGE_SUMMARY = "page_summary"


class BoundingBox(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class ContentUnit(BaseModel):
    unit_id: str
    book_id: str
    book_version: str
    page_number: int = Field(ge=1)
    sequence: int = Field(ge=0)

    modality: Modality
    language: str

    chapter_id: Optional[str] = None
    chapter_title: Optional[str] = None
    section_id: Optional[str] = None

    text: Optional[str] = None
    caption: Optional[str] = None
    visual_description: Optional[str] = None
    ocr_text: Optional[str] = None
    table_markdown: Optional[str] = None

    figure_number: Optional[str] = None
    table_number: Optional[str] = None
    diagram_number: Optional[str] = None

    bbox: Optional[BoundingBox] = None

    source_pdf_uri: str
    page_image_uri: Optional[str] = None
    crop_uri: Optional[str] = None

    extraction_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )
    visual_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    content_hash: Optional[str] = None
    is_active: bool = True


class BookManifest(BaseModel):
    book_id: str
    title: str
    grade: int
    subject: str
    language: str
    board: str

    book_version: str
    source_pdf_uri: str
    page_count: int = Field(ge=1)

    original_source_key: Optional[str] = None
    source_size_bytes: Optional[int] = Field(default=None, ge=1)
    source_sha256: Optional[str] = None
    source_etag: Optional[str] = None

    content_units_key: str
    page_images_prefix: str
    figures_prefix: str
    tables_prefix: str

    ingestion_status: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

class PageArtifact(BaseModel):
    book_id: str
    book_version: str
    page_number: int = Field(ge=1)

    source_pdf_uri: str
    local_image_path: str
    image_s3_key: str
    metadata_s3_key: str

    image_format: str
    dpi: int = Field(gt=0)
    pixel_width: int = Field(gt=0)
    pixel_height: int = Field(gt=0)

    pdf_width_points: float = Field(gt=0)
    pdf_height_points: float = Field(gt=0)

    file_size_bytes: int = Field(gt=0)
    image_sha256: str

    rendered_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
