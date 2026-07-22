from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import (
    BadZipFile,
    ZipFile,
    ZipInfo,
)

import fitz

from src.chapter_manifest import (
    ChapterManifest,
)


DEFAULT_MAX_TOTAL_UNCOMPRESSED_BYTES = (
    2 * 1024 * 1024 * 1024
)

DEFAULT_MAX_COMPRESSION_RATIO = 200.0


def calculate_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        while block := file_handle.read(
            1024 * 1024
        ):
            digest.update(block)

    return digest.hexdigest()


def validate_member_path(
    member_name: str,
) -> PurePosixPath:
    if "\\" in member_name:
        raise ValueError(
            "ZIP member names must use "
            f"forward slashes: {member_name}"
        )

    member_path = PurePosixPath(
        member_name
    )

    if member_path.is_absolute():
        raise ValueError(
            "ZIP contains an absolute path: "
            f"{member_name}"
        )

    if ".." in member_path.parts:
        raise ValueError(
            "ZIP contains an unsafe parent "
            f"path: {member_name}"
        )

    if not member_path.parts:
        raise ValueError(
            "ZIP contains an empty member path."
        )

    return member_path


def archive_member_name(
    archive_root: str | None,
    relative_name: str,
) -> str:
    relative_path = PurePosixPath(
        relative_name
    )

    if archive_root is None:
        return relative_path.as_posix()

    return (
        PurePosixPath(archive_root)
        / relative_path
    ).as_posix()


def expected_archive_members(
    manifest: ChapterManifest,
) -> set[str]:
    archive_root = (
        manifest.source_archive.archive_root
    )

    expected = {
        archive_member_name(
            archive_root,
            document.source_filename,
        )
        for document in manifest.documents
    }

    expected.update(
        archive_member_name(
            archive_root,
            asset,
        )
        for asset in (
            manifest.source_archive
            .supplementary_assets
        )
    )

    return expected


def validate_archive_inventory(
    archive: ZipFile,
    manifest: ChapterManifest,
    *,
    max_total_uncompressed_bytes: int = (
        DEFAULT_MAX_TOTAL_UNCOMPRESSED_BYTES
    ),
    max_compression_ratio: float = (
        DEFAULT_MAX_COMPRESSION_RATIO
    ),
) -> dict[str, ZipInfo]:
    members = [
        info
        for info in archive.infolist()
        if not info.is_dir()
    ]

    if not members:
        raise ValueError(
            "ZIP archive contains no files."
        )

    member_by_name: dict[
        str,
        ZipInfo,
    ] = {}

    casefold_names: set[str] = set()
    total_uncompressed_bytes = 0

    for info in members:
        validate_member_path(
            info.filename
        )

        casefold_name = (
            info.filename.casefold()
        )

        if casefold_name in casefold_names:
            raise ValueError(
                "ZIP contains duplicate or "
                "case-conflicting member names: "
                f"{info.filename}"
            )

        casefold_names.add(casefold_name)

        if info.flag_bits & 0x1:
            raise ValueError(
                "Encrypted ZIP members are not "
                f"supported: {info.filename}"
            )

        total_uncompressed_bytes += (
            info.file_size
        )

        if (
            total_uncompressed_bytes
            > max_total_uncompressed_bytes
        ):
            raise ValueError(
                "ZIP uncompressed size exceeds "
                "the configured safety limit."
            )

        if (
            info.file_size > 0
            and info.compress_size == 0
        ):
            raise ValueError(
                "ZIP member has an invalid "
                "compressed size: "
                f"{info.filename}"
            )

        if info.compress_size > 0:
            compression_ratio = (
                info.file_size
                / info.compress_size
            )

            if (
                compression_ratio
                > max_compression_ratio
            ):
                raise ValueError(
                    "ZIP member compression ratio "
                    "exceeds the safety limit: "
                    f"{info.filename}"
                )

        member_by_name[
            info.filename
        ] = info

    expected = expected_archive_members(
        manifest
    )
    actual = set(member_by_name)

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)

    if missing:
        raise ValueError(
            "ZIP is missing manifest files: "
            + ", ".join(missing)
        )

    if unexpected:
        raise ValueError(
            "ZIP contains unexpected files: "
            + ", ".join(unexpected)
        )

    corrupt_member = archive.testzip()

    if corrupt_member is not None:
        raise ValueError(
            "ZIP contains a corrupt member: "
            f"{corrupt_member}"
        )

    return member_by_name


def validate_pdf(
    path: Path,
    *,
    expected_page_count: int,
) -> dict[str, object]:
    try:
        with fitz.open(path) as document:
            if document.needs_pass:
                raise ValueError(
                    "PDF is password protected: "
                    f"{path.name}"
                )

            page_count = (
                document.page_count
            )

            if page_count < 1:
                raise ValueError(
                    "PDF contains no pages: "
                    f"{path.name}"
                )

            if (
                page_count
                != expected_page_count
            ):
                raise ValueError(
                    "PDF page count mismatch for "
                    f"{path.name}: "
                    f"{page_count} != "
                    f"{expected_page_count}"
                )

            # Force access to the first and last
            # pages to detect malformed objects.
            document[0].get_text("text")

            if page_count > 1:
                document[
                    page_count - 1
                ].get_text("text")

    except fitz.FileDataError as error:
        raise ValueError(
            f"PyMuPDF cannot read {path.name}."
        ) from error

    return {
        "page_count": page_count,
        "size_bytes": path.stat().st_size,
        "sha256": calculate_sha256(path),
    }


def extract_chapter_archive(
    archive_path: Path,
    target_directory: Path,
    manifest: ChapterManifest,
    *,
    report_path: Path,
    replace: bool = False,
) -> dict[str, object]:
    if not archive_path.is_file():
        raise FileNotFoundError(
            "Chapter archive not found: "
            f"{archive_path}"
        )

    target_directory = (
        target_directory.resolve()
    )
    target_parent = (
        target_directory.parent
    )

    target_parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if target_directory.exists():
        if not replace:
            raise FileExistsError(
                "Extraction directory already "
                "exists. Use replace=True only "
                "after reviewing its contents: "
                f"{target_directory}"
            )

        shutil.rmtree(
            target_directory
        )

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=(
                f".{target_directory.name}-"
                "extract-"
            ),
            dir=target_parent,
        )
    )

    try:
        try:
            archive = ZipFile(
                archive_path
            )
        except BadZipFile as error:
            raise ValueError(
                "Source archive is not a valid "
                f"ZIP file: {archive_path}"
            ) from error

        with archive:
            member_by_name = (
                validate_archive_inventory(
                    archive,
                    manifest,
                )
            )

            archive_root = (
                manifest.source_archive
                .archive_root
            )

            extraction_records = []

            for document in (
                manifest.documents
            ):
                archive_member = (
                    archive_member_name(
                        archive_root,
                        document.source_filename,
                    )
                )

                info = member_by_name[
                    archive_member
                ]

                destination = (
                    temporary_directory
                    / document.source_filename
                )

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with archive.open(
                    info,
                    "r",
                ) as source_handle:
                    with destination.open(
                        "wb"
                    ) as destination_handle:
                        shutil.copyfileobj(
                            source_handle,
                            destination_handle,
                            length=1024 * 1024,
                        )

                validation = validate_pdf(
                    destination,
                    expected_page_count=(
                        document
                        .source_page_count
                    ),
                )

                extraction_records.append({
                    "order": document.order,
                    "document_id": (
                        document.document_id
                    ),
                    "document_type": (
                        document.document_type
                    ),
                    "source_filename": (
                        document.source_filename
                    ),
                    "canonical_start_page": (
                        document
                        .canonical_start_page
                    ),
                    "canonical_end_page": (
                        document
                        .canonical_end_page
                    ),
                    **validation,
                })

            asset_records = []

            for asset in (
                manifest.source_archive
                .supplementary_assets
            ):
                archive_member = (
                    archive_member_name(
                        archive_root,
                        asset,
                    )
                )

                info = member_by_name[
                    archive_member
                ]

                destination = (
                    temporary_directory
                    / asset
                )

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with archive.open(
                    info,
                    "r",
                ) as source_handle:
                    with destination.open(
                        "wb"
                    ) as destination_handle:
                        shutil.copyfileobj(
                            source_handle,
                            destination_handle,
                            length=1024 * 1024,
                        )

                asset_records.append({
                    "source_filename": asset,
                    "size_bytes": (
                        destination
                        .stat()
                        .st_size
                    ),
                    "sha256": calculate_sha256(
                        destination
                    ),
                })

        temporary_directory.rename(
            target_directory
        )

        report = {
            "schema_version": "1.0",
            "status": "VALID",
            "book_id": manifest.book_id,
            "book_version": (
                manifest.book_version
            ),
            "source_archive": {
                "local_path": str(
                    archive_path
                ),
                "s3_uri": (
                    manifest.source_archive
                    .s3_uri
                ),
                "size_bytes": (
                    archive_path
                    .stat()
                    .st_size
                ),
                "sha256": calculate_sha256(
                    archive_path
                ),
            },
            "target_directory": str(
                target_directory
            ),
            "document_count": len(
                extraction_records
            ),
            "source_page_count": sum(
                int(record["page_count"])
                for record
                in extraction_records
            ),
            "documents": (
                extraction_records
            ),
            "supplementary_assets": (
                asset_records
            ),
            "aws_calls": 0,
        }

        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        return report

    except Exception:
        if temporary_directory.exists():
            shutil.rmtree(
                temporary_directory
            )

        raise
