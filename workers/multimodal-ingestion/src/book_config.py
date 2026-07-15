from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


SLUG_PATTERN = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
)

VERSION_PATTERN = re.compile(
    r"^v[0-9]+(?:-[a-z0-9]+)*$"
)

S3_BUCKET_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
)


class ImmutableConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class BookIdentityConfig(
    ImmutableConfigModel
):
    book_id: str
    title: str = Field(
        min_length=1,
        max_length=300,
    )
    grade: int = Field(
        ge=1,
        le=12,
    )
    subject: str
    language: str
    board: str
    version: str
    page_count: int = Field(ge=1)
    academic_year: str | None = None
    status: Literal[
        "draft",
        "registered",
        "processing",
        "ready",
        "archived",
    ] = "draft"

    @field_validator(
        "book_id",
        "subject",
        "language",
        "board",
    )
    @classmethod
    def validate_slug(
        cls,
        value: str,
    ) -> str:
        if not SLUG_PATTERN.fullmatch(value):
            raise ValueError(
                "Value must be a lowercase "
                "hyphen-separated slug."
            )

        return value

    @field_validator("version")
    @classmethod
    def validate_version(
        cls,
        value: str,
    ) -> str:
        if not VERSION_PATTERN.fullmatch(value):
            raise ValueError(
                "Version must look like v1, "
                "v2, or v1-draft."
            )

        return value

    @property
    def grade_slug(self) -> str:
        return f"grade-{self.grade}"


class AwsConfig(ImmutableConfigModel):
    region: str
    bucket: str

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


class BdaConfig(ImmutableConfigModel):
    project_arn: str
    profile_arn: str
    stage: Literal[
        "DEVELOPMENT",
        "LIVE",
    ] = "DEVELOPMENT"

    @field_validator(
        "project_arn",
        "profile_arn",
    )
    @classmethod
    def validate_bedrock_arn(
        cls,
        value: str,
    ) -> str:
        if not value.startswith(
            "arn:aws:bedrock:"
        ):
            raise ValueError(
                "Expected a Bedrock ARN."
            )

        return value


class OpenSearchConfig(
    ImmutableConfigModel
):
    collection_endpoint: str
    index_name: str
    vector_field: str = "embedding"

    @field_validator(
        "collection_endpoint"
    )
    @classmethod
    def validate_endpoint(
        cls,
        value: str,
    ) -> str:
        if not value.startswith("https://"):
            raise ValueError(
                "Collection endpoint must "
                "use HTTPS."
            )

        return value.rstrip("/")

    @field_validator("index_name")
    @classmethod
    def validate_index_name(
        cls,
        value: str,
    ) -> str:
        if not SLUG_PATTERN.fullmatch(value):
            raise ValueError(
                "Index name must be a lowercase "
                "hyphen-separated slug."
            )

        return value


class SourceInputConfig(
    ImmutableConfigModel
):
    mode: Literal[
        "single_pdf",
        "chapter_folder",
    ] = "single_pdf"

    chapter_directory: str | None = None

    chapter_order: Literal[
        "filename",
        "manifest",
    ] = "filename"

    chapter_manifest: str | None = None

    merged_pdf_name: str = "textbook.pdf"

    @field_validator(
        "chapter_directory",
        "chapter_manifest",
    )
    @classmethod
    def validate_optional_relative_path(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        path = Path(value)

        if path.is_absolute():
            raise ValueError(
                "Chapter paths must be "
                "repository-relative."
            )

        if ".." in path.parts:
            raise ValueError(
                "Chapter paths cannot contain '..'."
            )

        return path.as_posix()

    @field_validator("merged_pdf_name")
    @classmethod
    def validate_merged_pdf_name(
        cls,
        value: str,
    ) -> str:
        path = Path(value)

        if (
            path.name != value
            or path.suffix.casefold() != ".pdf"
        ):
            raise ValueError(
                "Merged PDF name must be a plain "
                "filename ending in .pdf."
            )

        return value

    @model_validator(mode="after")
    def validate_source_mode(
        self,
    ) -> "SourceInputConfig":
        if self.mode == "single_pdf":
            if (
                self.chapter_directory is not None
                or self.chapter_manifest is not None
            ):
                raise ValueError(
                    "Single-PDF mode cannot define "
                    "chapter directory or manifest."
                )

        if self.mode == "chapter_folder":
            if not self.chapter_directory:
                raise ValueError(
                    "Chapter-folder mode requires "
                    "chapter_directory."
                )

            if (
                self.chapter_order == "manifest"
                and not self.chapter_manifest
            ):
                raise ValueError(
                    "Manifest ordering requires "
                    "chapter_manifest."
                )

        return self


class StorageConfig(
    ImmutableConfigModel
):
    source_s3_key: str
    derived_prefix: str
    bda_input_prefix: str
    local_root: str

    @field_validator(
        "source_s3_key",
        "derived_prefix",
        "bda_input_prefix",
    )
    @classmethod
    def validate_s3_key(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip("/")

        if not normalized:
            raise ValueError(
                "S3 key or prefix cannot be empty."
            )

        if ".." in normalized.split("/"):
            raise ValueError(
                "S3 key cannot contain '..'."
            )

        return normalized

    @field_validator("local_root")
    @classmethod
    def validate_local_root(
        cls,
        value: str,
    ) -> str:
        path = Path(value)

        if path.is_absolute():
            raise ValueError(
                "Local root must be repository-relative."
            )

        if ".." in path.parts:
            raise ValueError(
                "Local root cannot contain '..'."
            )

        return path.as_posix()


class EmbeddingModelConfig(
    ImmutableConfigModel
):
    model_id: str
    dimensions: int = Field(
        ge=256,
        le=4096,
    )
    normalize: bool = True


class GenerationModelConfig(
    ImmutableConfigModel
):
    model_id: str
    maximum_output_tokens: int = Field(
        ge=100,
        le=4096,
    )
    temperature: float = Field(
        ge=0,
        le=1,
    )


class FoundationModelsConfig(
    ImmutableConfigModel
):
    embedding: EmbeddingModelConfig
    generation: GenerationModelConfig


class ProcessingConfig(
    ImmutableConfigModel
):
    page_batch_size: int = Field(
        ge=1,
        le=100,
    )
    minimum_text_similarity: float = Field(
        ge=0,
        le=1,
    )
    embedding_checkpoint_interval: int = (
        Field(
            ge=1,
            le=1000,
        )
    )


class RetrievalConfig(
    ImmutableConfigModel
):
    vector_candidate_limit: int = Field(
        ge=1,
        le=100,
    )
    bm25_candidate_limit: int = Field(
        ge=1,
        le=100,
    )
    result_limit: int = Field(
        ge=1,
        le=20,
    )
    rrf_constant: int = Field(
        ge=1,
        le=1000,
    )
    vector_weight: float = Field(
        gt=0,
        le=100,
    )
    bm25_weight: float = Field(
        gt=0,
        le=100,
    )


class BookConfig(ImmutableConfigModel):
    schema_version: Literal["1.0"]
    book: BookIdentityConfig
    aws: AwsConfig
    bda: BdaConfig
    opensearch: OpenSearchConfig
    source: SourceInputConfig = Field(
        default_factory=SourceInputConfig
    )
    storage: StorageConfig
    models: FoundationModelsConfig
    processing: ProcessingConfig
    retrieval: RetrievalConfig

    @model_validator(mode="after")
    def validate_consistency(
        self,
    ) -> "BookConfig":
        expected_index_name = (
            f"{self.book.book_id}-"
            f"{self.book.version}"
        )

        if (
            self.opensearch.index_name
            != expected_index_name
        ):
            raise ValueError(
                "OpenSearch index mismatch. "
                f"Expected: {expected_index_name}"
            )

        expected_source_suffix = (
            f"{self.book.book_id}/versions/"
            f"{self.book.version}/"
            f"{self.source.merged_pdf_name}"
        )

        if not self.storage.source_s3_key.endswith(
            expected_source_suffix
        ):
            raise ValueError(
                "Source S3 key does not match "
                "book ID and version."
            )

        expected_local_root = (
            "data/multimodal-output/"
            f"{self.book.book_id}/"
            f"{self.book.version}"
        )

        if (
            self.storage.local_root
            != expected_local_root
        ):
            raise ValueError(
                "Local root mismatch. "
                f"Expected: {expected_local_root}"
            )

        identity_path = (
            f"{self.book.grade_slug}/"
            f"{self.book.book_id}/"
            f"{self.book.version}"
        )

        if identity_path not in (
            self.storage.derived_prefix
        ):
            raise ValueError(
                "Derived prefix does not contain "
                "the configured book identity."
            )

        if identity_path not in (
            self.storage.bda_input_prefix
        ):
            raise ValueError(
                "BDA input prefix does not contain "
                "the configured book identity."
            )

        if (
            self.retrieval.result_limit
            > self.retrieval.vector_candidate_limit
        ):
            raise ValueError(
                "Result limit cannot exceed "
                "vector candidate limit."
            )

        if (
            self.retrieval.result_limit
            > self.retrieval.bm25_candidate_limit
        ):
            raise ValueError(
                "Result limit cannot exceed "
                "BM25 candidate limit."
            )

        return self

    @property
    def source_s3_uri(self) -> str:
        return (
            f"s3://{self.aws.bucket}/"
            f"{self.storage.source_s3_key}"
        )

    @property
    def local_root(self) -> Path:
        return Path(
            self.storage.local_root
        )

    @property
    def source_pdf_path(self) -> Path:
        return (
            self.local_root
            / "source"
            / self.source.merged_pdf_name
        )

    @property
    def chapter_directory_path(
        self,
    ) -> Path | None:
        if not self.source.chapter_directory:
            return None

        return Path(
            self.source.chapter_directory
        )

    @property
    def chapter_manifest_path(
        self,
    ) -> Path | None:
        if not self.source.chapter_manifest:
            return None

        return Path(
            self.source.chapter_manifest
        )

    @property
    def full_book_root(self) -> Path:
        return (
            self.local_root
            / "full-book"
        )

    @property
    def batch_manifest_path(self) -> Path:
        return (
            self.full_book_root
            / "full-book-batch-manifest.json"
        )

    @property
    def jobs_dir(self) -> Path:
        return (
            self.full_book_root
            / "bda-jobs"
        )

    @property
    def results_root(self) -> Path:
        return (
            self.full_book_root
            / "bda-results"
        )


def load_book_config(
    path: str | Path,
) -> BookConfig:
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Book config not found: "
            f"{config_path}"
        )

    try:
        raw_value = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {config_path}: "
            f"{exc}"
        ) from exc

    if not isinstance(raw_value, dict):
        raise ValueError(
            "Book configuration must be "
            "a JSON object."
        )

    return BookConfig.model_validate(
        raw_value
    )
