import json
import sys
from pathlib import Path

from scripts.prepare_embedding_records import (
    build_context_citation_label,
    copy_context_metadata,
    main,
)


def write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def read_jsonl(
    path: Path,
) -> list[dict]:
    if not path.is_file():
        return []

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def base_content_unit(
    unit_id: str,
    text: str,
    pages: list[int],
) -> dict:
    return {
        "schema_version": "1.0",
        "unit_id": unit_id,
        "book_id": "test-book",
        "book_version": "v1-test",
        "element_index": 1,
        "element_type": "TEXT",
        "element_sub_type": "PARAGRAPH",
        "modality": "paragraph",
        "retrieval_priority": "normal",
        "source_page_numbers": pages,
        "locations": [],
        "raw_text": text,
        "markdown_text": text,
        "search_text": text,
        "generated_title": None,
        "generated_summary": None,
        "asset_s3_uris": [],
        "asset_local_paths": [],
        "quality_flags": [],
    }


def chapter_context(
    *,
    chapter_status: str,
    chapter_ids: list[str],
    chapter_titles: list[str],
    chapter_id: str | None,
    chapter_title: str | None,
) -> dict:
    return {
        "page_context_status": "resolved",
        "chapter_context_status": (
            chapter_status
        ),
        "unresolved_source_page_numbers": [],
        "page_types": ["unit"],
        "document_ids": ["unit-1"],
        "document_titles": ["Unit 1"],
        "document_types": ["unit"],
        "unit_numbers": [1],
        "chapter_ids": chapter_ids,
        "chapter_titles": chapter_titles,
        "source_filenames": ["unit-1.pdf"],
        "document_id": "unit-1",
        "document_title": "Unit 1",
        "document_type": "unit",
        "unit_number": 1,
        "chapter_id": chapter_id,
        "chapter_title": chapter_title,
    }


def test_builds_context_citation_labels():
    assert build_context_citation_label(
        "test-book",
        [41],
        chapter_titles=["Chapter A"],
        document_titles=["Unit 1"],
    ) == (
        "Chapter A "
        "(test-book, page 41)"
    )

    assert build_context_citation_label(
        "test-book",
        [42, 43],
        chapter_titles=[
            "Chapter A",
            "Chapter B",
        ],
        document_titles=["Unit 1"],
    ) == (
        "Chapter A / Chapter B "
        "(test-book, pages 42-43)"
    )

    assert build_context_citation_label(
        "test-book",
        [44],
        chapter_titles=[],
        document_titles=["Appendix"],
    ) == (
        "Appendix "
        "(test-book, page 44)"
    )


def test_copy_context_preserves_only_present_fields():
    content_unit = {
        "unit_id": "u1",
        "chapter_id": "chapter-a",
        "chapter_title": "Chapter A",
        "document_id": "unit-1",
        "raw_text": "Example",
    }

    copied = copy_context_metadata(
        content_unit
    )

    assert copied == {
        "document_id": "unit-1",
        "chapter_id": "chapter-a",
        "chapter_title": "Chapter A",
    }


def test_embedding_preparation_propagates_context(
    tmp_path: Path,
    monkeypatch,
):
    normalized_dir = (
        tmp_path / "normalized"
    )
    output_dir = (
        tmp_path / "embedding-ready"
    )

    normalized_dir.mkdir()

    single = base_content_unit(
        "single-chapter",
        (
            "This is a detailed educational "
            "paragraph belonging to Chapter A."
        ),
        [41],
    )

    single.update(
        chapter_context(
            chapter_status="single",
            chapter_ids=["chapter-a"],
            chapter_titles=["Chapter A"],
            chapter_id="chapter-a",
            chapter_title="Chapter A",
        )
    )

    multiple = base_content_unit(
        "cross-chapter",
        (
            "This educational explanation spans "
            "Chapter A and Chapter B."
        ),
        [42, 43],
    )

    multiple.update(
        chapter_context(
            chapter_status="multiple",
            chapter_ids=[
                "chapter-a",
                "chapter-b",
            ],
            chapter_titles=[
                "Chapter A",
                "Chapter B",
            ],
            chapter_id=None,
            chapter_title=None,
        )
    )

    skipped = base_content_unit(
        "skipped-chapter",
        (
            "This educational paragraph is "
            "intentionally low priority."
        ),
        [44],
    )

    skipped["retrieval_priority"] = "low"

    skipped.update(
        chapter_context(
            chapter_status="single",
            chapter_ids=["chapter-b"],
            chapter_titles=["Chapter B"],
            chapter_id="chapter-b",
            chapter_title="Chapter B",
        )
    )

    legacy = base_content_unit(
        "legacy-unit",
        (
            "This legacy educational paragraph "
            "contains no chapter metadata."
        ),
        [1],
    )

    write_jsonl(
        normalized_dir
        / "content-units.jsonl",
        [
            single,
            multiple,
            skipped,
            legacy,
        ],
    )

    write_jsonl(
        normalized_dir / "figures.jsonl",
        [],
    )

    write_jsonl(
        normalized_dir / "tables.jsonl",
        [],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_embedding_records.py",
            str(normalized_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert main() == 0

    records = read_jsonl(
        output_dir
        / "embedding-records.jsonl"
    )

    skipped_records = read_jsonl(
        output_dir
        / "skipped-records.jsonl"
    )

    by_unit_id = {
        record["source_unit_id"]: record
        for record in records
    }

    single_record = by_unit_id[
        "single-chapter"
    ]

    assert single_record[
        "chapter_id"
    ] == "chapter-a"

    assert single_record[
        "chapter_title"
    ] == "Chapter A"

    assert single_record[
        "document_id"
    ] == "unit-1"

    assert single_record[
        "citation_label"
    ] == "test-book, page 41"

    assert single_record[
        "context_citation_label"
    ] == (
        "Chapter A "
        "(test-book, page 41)"
    )

    multiple_record = by_unit_id[
        "cross-chapter"
    ]

    assert multiple_record[
        "chapter_context_status"
    ] == "multiple"

    assert multiple_record[
        "chapter_id"
    ] is None

    assert multiple_record[
        "chapter_ids"
    ] == [
        "chapter-a",
        "chapter-b",
    ]

    assert multiple_record[
        "context_citation_label"
    ] == (
        "Chapter A / Chapter B "
        "(test-book, pages 42-43)"
    )

    legacy_record = by_unit_id[
        "legacy-unit"
    ]

    assert (
        "chapter_id"
        not in legacy_record
    )

    assert (
        "document_id"
        not in legacy_record
    )

    assert (
        "context_citation_label"
        not in legacy_record
    )

    assert len(skipped_records) == 1

    skipped_record = skipped_records[0]

    assert skipped_record[
        "unit_id"
    ] == "skipped-chapter"

    assert skipped_record[
        "chapter_id"
    ] == "chapter-b"

    assert skipped_record[
        "context_citation_label"
    ] == (
        "Chapter B "
        "(test-book, page 44)"
    )
