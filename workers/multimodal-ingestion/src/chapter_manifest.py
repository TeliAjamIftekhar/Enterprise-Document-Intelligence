from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.book_config import BookConfig


SLUG_PATTERN = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
)

VERSION_PATTERN = re.compile(
    r"^v[0-9]+(?:-[a-z0-9]+)*$"
)

S3_BUCKET_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
)


def validate_relative_posix_path(
    value: str,
    *,
    label: str,
) -> str:
    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"{label} cannot be empty."
        )

    if "\\" in normalized:
        raise ValueError(
            f"{label} must use forward slashes."
        )

    path = PurePosixPath(normalized)

    if path.is_absolute():
        raise ValueError(
            f"{label} must be relative."
        )

    if ".." in path.parts:
        raise ValueError(
            f"{label} cannot contain '..'."
        )

    return path.as_posix()


class ImmutableManifestModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class SourceArchiveConfig(
    ImmutableManifestModel
):
    bucket: str
    key: str
    archive_root: str
    supplementary_assets: list[str] = Field(
        default_factory=list
    )

    @field_validator("bucket")
    @classmethod
    def validate_bucket(
        cls,
        value: str,
    ) -> str:
        if not S3_BUCKET_PATTERN.fullmatch(
            value
        ):
            raise ValueError(
                "Invalid S3 bucket name."
            )

        return value

    @field_validator("key")
    @classmethod
    def validate_key(
        cls,
        value: str,
    ) -> str:
        return validate_relative_posix_path(
            value,
            label="S3 archive key",
        )

    @field_validator("archive_root")
    @classmethod
    def validate_archive_root(
        cls,
        value: str,
    ) -> str:
        return validate_relative_posix_path(
            value,
            label="Archive root",
        )

    @field_validator(
        "supplementary_assets"
    )
    @classmethod
    def validate_supplementary_assets(
        cls,
        values: list[str],
    ) -> list[str]:
        normalized = [
            validate_relative_posix_path(
                value,
                label="Supplementary asset",
            )
            for value in values
        ]

        casefolded = [
            value.casefold()
            for value in normalized
        ]

        if (
            len(casefolded)
            != len(set(casefolded))
        ):
            raise ValueError(
                "Supplementary assets must "
                "be unique."
            )

        return normalized

    @property
    def s3_uri(self) -> str:
        return (
            f"s3://{self.bucket}/{self.key}"
        )


class CanonicalLayoutConfig(
    ImmutableManifestModel
):
    leading_blank_pages: int = Field(
        ge=0
    )
    source_document_pages: int = Field(
        ge=1
    )
    trailing_blank_pages: int = Field(
        ge=0
    )
    canonical_page_count: int = Field(
        ge=1
    )
    source_to_canonical_page_offset: int = (
        Field(ge=0)
    )

    @model_validator(mode="after")
    def validate_layout(
        self,
    ) -> "CanonicalLayoutConfig":
        calculated_page_count = (
            self.leading_blank_pages
            + self.source_document_pages
            + self.trailing_blank_pages
        )

        if (
            calculated_page_count
            != self.canonical_page_count
        ):
            raise ValueError(
                "Canonical layout page count "
                "mismatch."
            )

        if (
            self.source_to_canonical_page_offset
            != self.leading_blank_pages
        ):
            raise ValueError(
                "Source-to-canonical offset must "
                "equal leading blank pages."
            )

        return self


class ChapterRange(
    ImmutableManifestModel
):
    chapter_id: str
    chapter_title: str = Field(
        min_length=1,
        max_length=300,
    )
    source_start_page: int = Field(ge=1)
    source_end_page: int = Field(ge=1)
    canonical_start_page: int = Field(ge=1)
    canonical_end_page: int = Field(ge=1)

    @field_validator("chapter_id")
    @classmethod
    def validate_chapter_id(
        cls,
        value: str,
    ) -> str:
        if not SLUG_PATTERN.fullmatch(value):
            raise ValueError(
                "Chapter ID must be a lowercase "
                "hyphen-separated slug."
            )

        return value

    @model_validator(mode="after")
    def validate_ranges(
        self,
    ) -> "ChapterRange":
        if (
            self.source_end_page
            < self.source_start_page
        ):
            raise ValueError(
                "Chapter source range is invalid."
            )

        if (
            self.canonical_end_page
            < self.canonical_start_page
        ):
            raise ValueError(
                "Chapter canonical range is "
                "invalid."
            )

        source_length = (
            self.source_end_page
            - self.source_start_page
            + 1
        )

        canonical_length = (
            self.canonical_end_page
            - self.canonical_start_page
            + 1
        )

        if source_length != canonical_length:
            raise ValueError(
                "Chapter source and canonical "
                "ranges must have the same length."
            )

        return self

    def contains_canonical_page(
        self,
        page_number: int,
    ) -> bool:
        return (
            self.canonical_start_page
            <= page_number
            <= self.canonical_end_page
        )


class SourceDocument(
    ImmutableManifestModel
):
    order: int = Field(ge=1)
    document_id: str
    document_type: Literal[
        "front_matter",
        "unit",
        "appendix",
        "supplementary",
    ]
    unit_number: int | None = Field(
        default=None,
        ge=1,
    )
    source_filename: str
    source_page_count: int = Field(ge=1)
    canonical_start_page: int = Field(ge=1)
    canonical_end_page: int = Field(ge=1)
    title: str = Field(
        min_length=1,
        max_length=300,
    )
    chapters: list[ChapterRange] = Field(
        default_factory=list
    )

    @field_validator("document_id")
    @classmethod
    def validate_document_id(
        cls,
        value: str,
    ) -> str:
        if not SLUG_PATTERN.fullmatch(value):
            raise ValueError(
                "Document ID must be a lowercase "
                "hyphen-separated slug."
            )

        return value

    @field_validator("source_filename")
    @classmethod
    def validate_source_filename(
        cls,
        value: str,
    ) -> str:
        path = PurePosixPath(value)

        if (
            path.name != value
            or "\\" in value
            or path.suffix.casefold() != ".pdf"
        ):
            raise ValueError(
                "Source filename must be a plain "
                "PDF filename."
            )

        return value

    @model_validator(mode="after")
    def validate_document(
        self,
    ) -> "SourceDocument":
        if (
            self.document_type == "unit"
            and self.unit_number is None
        ):
            raise ValueError(
                "Unit documents require "
                "unit_number."
            )

        if (
            self.document_type != "unit"
            and self.unit_number is not None
        ):
            raise ValueError(
                "Only unit documents can define "
                "unit_number."
            )

        canonical_length = (
            self.canonical_end_page
            - self.canonical_start_page
            + 1
        )

        if canonical_length != (
            self.source_page_count
        ):
            raise ValueError(
                "Document canonical range does "
                "not match source_page_count."
            )

        chapter_ids = [
            chapter.chapter_id
            for chapter in self.chapters
        ]

        if len(chapter_ids) != len(
            set(chapter_ids)
        ):
            raise ValueError(
                "Chapter IDs must be unique "
                "within a document."
            )

        if not self.chapters:
            return self

        expected_source_start = 1
        expected_canonical_start = (
            self.canonical_start_page
        )

        for chapter in self.chapters:
            if (
                chapter.source_start_page
                != expected_source_start
            ):
                raise ValueError(
                    "Chapter source ranges must "
                    "be contiguous and start at 1."
                )

            if (
                chapter.canonical_start_page
                != expected_canonical_start
            ):
                raise ValueError(
                    "Chapter canonical ranges "
                    "must be contiguous."
                )

            expected_from_offset = (
                self.canonical_start_page
                + chapter.source_start_page
                - 1
            )

            if (
                chapter.canonical_start_page
                != expected_from_offset
            ):
                raise ValueError(
                    "Chapter canonical offset "
                    "does not match its source "
                    "page."
                )

            expected_source_start = (
                chapter.source_end_page + 1
            )
            expected_canonical_start = (
                chapter.canonical_end_page + 1
            )

        if (
            self.chapters[-1].source_end_page
            != self.source_page_count
        ):
            raise ValueError(
                "Chapter source ranges must "
                "cover the complete document."
            )

        if (
            self.chapters[-1]
            .canonical_end_page
            != self.canonical_end_page
        ):
            raise ValueError(
                "Chapter canonical ranges must "
                "cover the complete document."
            )

        return self

    def contains_canonical_page(
        self,
        page_number: int,
    ) -> bool:
        return (
            self.canonical_start_page
            <= page_number
            <= self.canonical_end_page
        )


class ChapterManifest(
    ImmutableManifestModel
):
    schema_version: Literal["1.0"]
    book_id: str
    book_version: str
    title: str = Field(
        min_length=1,
        max_length=300,
    )
    ordering_strategy: Literal["manifest"]
    source_archive: SourceArchiveConfig
    canonical_layout: CanonicalLayoutConfig
    documents: list[SourceDocument] = Field(
        min_length=1
    )

    @field_validator("book_id")
    @classmethod
    def validate_book_id(
        cls,
        value: str,
    ) -> str:
        if not SLUG_PATTERN.fullmatch(value):
            raise ValueError(
                "Book ID must be a lowercase "
                "hyphen-separated slug."
            )

        return value

    @field_validator("book_version")
    @classmethod
    def validate_book_version(
        cls,
        value: str,
    ) -> str:
        if not VERSION_PATTERN.fullmatch(value):
            raise ValueError(
                "Book version must look like "
                "v1 or v1-chapter-test."
            )

        return value

    @model_validator(mode="after")
    def validate_manifest(
        self,
    ) -> "ChapterManifest":
        expected_orders = list(
            range(1, len(self.documents) + 1)
        )

        actual_orders = [
            document.order
            for document in self.documents
        ]

        if actual_orders != expected_orders:
            raise ValueError(
                "Document order must be "
                "consecutive and start at 1."
            )

        document_ids = [
            document.document_id
            for document in self.documents
        ]

        if len(document_ids) != len(
            set(document_ids)
        ):
            raise ValueError(
                "Document IDs must be unique."
            )

        filenames = [
            document.source_filename.casefold()
            for document in self.documents
        ]

        if len(filenames) != len(
            set(filenames)
        ):
            raise ValueError(
                "Source PDF filenames must "
                "be unique."
            )

        source_page_total = sum(
            document.source_page_count
            for document in self.documents
        )

        if source_page_total != (
            self.canonical_layout
            .source_document_pages
        ):
            raise ValueError(
                "Manifest source page total "
                "does not match canonical "
                "layout."
            )

        expected_canonical_start = (
            self.canonical_layout
            .leading_blank_pages
            + 1
        )

        all_chapter_ids: list[str] = []

        for document in self.documents:
            if (
                document.canonical_start_page
                != expected_canonical_start
            ):
                raise ValueError(
                    "Document canonical ranges "
                    "must be contiguous."
                )

            expected_canonical_start = (
                document.canonical_end_page + 1
            )

            all_chapter_ids.extend(
                chapter.chapter_id
                for chapter in document.chapters
            )

        expected_source_end = (
            self.canonical_layout
            .canonical_page_count
            - self.canonical_layout
            .trailing_blank_pages
        )

        if (
            self.documents[-1]
            .canonical_end_page
            != expected_source_end
        ):
            raise ValueError(
                "Final document page does not "
                "match the canonical layout."
            )

        if len(all_chapter_ids) != len(
            set(all_chapter_ids)
        ):
            raise ValueError(
                "Chapter IDs must be unique "
                "across the complete book."
            )

        return self

    @property
    def chapter_count(self) -> int:
        return sum(
            len(document.chapters)
            for document in self.documents
        )

    def document_for_canonical_page(
        self,
        page_number: int,
    ) -> SourceDocument | None:
        if (
            page_number < 1
            or page_number
            > self.canonical_layout
            .canonical_page_count
        ):
            raise ValueError(
                "Canonical page number is "
                "outside the textbook."
            )

        for document in self.documents:
            if document.contains_canonical_page(
                page_number
            ):
                return document

        return None

    def chapter_for_canonical_page(
        self,
        page_number: int,
    ) -> ChapterRange | None:
        document = (
            self.document_for_canonical_page(
                page_number
            )
        )

        if document is None:
            return None

        for chapter in document.chapters:
            if chapter.contains_canonical_page(
                page_number
            ):
                return chapter

        return None


def load_chapter_manifest(
    path: str | Path,
) -> ChapterManifest:
    manifest_path = Path(path)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Chapter manifest not found: "
            f"{manifest_path}"
        )

    try:
        raw_manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "Chapter manifest is not valid "
            f"JSON: {manifest_path}"
        ) from error

    return ChapterManifest.model_validate(
        raw_manifest
    )


def validate_manifest_for_book_config(
    manifest: ChapterManifest,
    config: BookConfig,
) -> None:
    errors: list[str] = []

    if manifest.book_id != config.book.book_id:
        errors.append(
            "book ID mismatch"
        )

    if (
        manifest.book_version
        != config.book.version
    ):
        errors.append(
            "book version mismatch"
        )

    if manifest.title != config.book.title:
        errors.append(
            "book title mismatch"
        )

    if (
        manifest.canonical_layout
        .canonical_page_count
        != config.book.page_count
    ):
        errors.append(
            "page count mismatch"
        )

    if (
        manifest.source_archive.bucket
        != config.aws.bucket
    ):
        errors.append(
            "S3 bucket mismatch"
        )

    if config.source.mode != "chapter_folder":
        errors.append(
            "source mode must be "
            "chapter_folder"
        )

    if (
        config.source.chapter_order
        != "manifest"
    ):
        errors.append(
            "chapter order must be manifest"
        )

    if not config.source.chapter_manifest:
        errors.append(
            "chapter manifest path is missing"
        )

    if errors:
        raise ValueError(
            "Chapter manifest does not match "
            "book configuration: "
            + "; ".join(errors)
        )
