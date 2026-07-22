import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


RUNNER_PATH = Path(
    "workers/multimodal-ingestion/"
    "scripts/run_all_textbooks.py"
)


def load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_all_textbooks_ocr_planning_test",
        RUNNER_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


def write_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def build_paths(
    tmp_path: Path,
) -> dict[str, Path]:
    return {
        "archive": tmp_path / "source.zip",
        "inspection": tmp_path / "inspection.json",
        "config": tmp_path / "book.json",
        "manifest": tmp_path / "manifest.json",
        "generation_report": (
            tmp_path / "generation-report.json"
        ),
        "extracted": tmp_path / "extracted",
        "extraction_report": (
            tmp_path / "extraction-report.json"
        ),
        "canonical_pdf": (
            tmp_path / "source/textbook.pdf"
        ),
        "page_map": (
            tmp_path
            / "source/chapter-page-map.json"
        ),
        "merge_report": (
            tmp_path
            / "source/chapter-merge-report.json"
        ),
        "bda_normalized": (
            tmp_path
            / "pipeline/bda/normalized"
        ),
        "ocr_plan": (
            tmp_path
            / "pipeline/ocr/fallback-plan.json"
        ),
        "ocr_fallback_root": (
            tmp_path / "pipeline/ocr/surya"
        ),
        "ocr_fallback_state": (
            tmp_path
            / "pipeline/ocr/surya/run-state.json"
        ),
        "ocr_fallback_report": (
            tmp_path
            / "pipeline/ocr/surya/verified/"
            "surya-fallback-report.json"
        ),
        "ocr_fallback_marker": (
            tmp_path
            / "pipeline/ocr/surya/verified/"
            "SURYA_OCR_FALLBACK_VERIFIED"
        ),
        "ocr_approval": (
            tmp_path / "surya-approval.json"
        ),
        "log": tmp_path / "pipeline.log",
    }


def prepare_book_artifacts(
    paths: dict[str, Path],
    *,
    language: str,
    subject: str,
    page_count: int,
    normalized_text: str,
) -> None:
    paths["archive"].write_bytes(
        b"fake-zip"
    )

    write_json(
        paths["inspection"],
        {
            "inspection_status": "PASSED",
        },
    )

    write_json(
        paths["config"],
        {
            "book": {
                "language": language,
                "subject": subject,
                "page_count": page_count,
            }
        },
    )

    write_json(
        paths["manifest"],
        {
            "chapters": [],
        },
    )

    write_json(
        paths["generation_report"],
        {
            "status": "READY",
        },
    )

    paths["extracted"].mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        paths["extraction_report"],
        {
            "status": "VALID",
        },
    )

    paths["canonical_pdf"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths["canonical_pdf"].write_bytes(
        b"%PDF-test"
    )

    write_json(
        paths["page_map"],
        {
            "pages": [],
        },
    )

    write_json(
        paths["merge_report"],
        {
            "status": "VALID",
        },
    )

    paths["bda_normalized"].mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        paths["bda_normalized"]
        / "pages.jsonl"
    ).write_text(
        json.dumps(
            {
                "canonical_page": 1,
                "text": normalized_text,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def create_state(
    runner: ModuleType,
    state_path: Path,
) -> dict:
    return runner.load_or_create_state(
        state_path,
        resume=False,
        bucket="test-bucket",
        prefix="NCERT I-X/",
        grades={1},
    )


def test_mathematics_uses_mixed_quality_gate(
    tmp_path: Path,
) -> None:
    runner = load_runner()

    config = tmp_path / "book.json"

    write_json(
        config,
        {
            "book": {
                "language": "English",
                "subject": "Mathematics",
                "page_count": 220,
            }
        },
    )

    language, page_count = (
        runner.load_book_processing_metadata(
            {},
            config,
        )
    )

    assert language == "Mathematics"
    assert page_count == 220


def test_process_book_accepts_valid_bda_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    paths = build_paths(tmp_path)

    prepare_book_artifacts(
        paths,
        language="English",
        subject="English",
        page_count=1,
        normalized_text=(
            "This textbook page contains a complete "
            "English paragraph for students. It has "
            "enough meaningful content to pass the "
            "language-aware text quality validation."
        ),
    )

    monkeypatch.setattr(
        runner,
        "paths_for_book",
        lambda book_id, version: paths,
    )

    state_path = tmp_path / "state.json"
    state = create_state(
        runner,
        state_path,
    )

    result = runner.process_book(
        {
            "book_id": "grade-1-english-test",
            "grade": 1,
            "title": "Grade 1 English Test Textbook",
            "source_bucket": "test-bucket",
            "source_zip_key": "NCERT I-X/Standard-I/English-Test.zip",
            "language": "English",
            "subject": "English",
        },
        registry_path=tmp_path / "registry.json",
        state_path=state_path,
        state=state,
        maximum_retries=0,
        dry_run=False,
    )

    assert result == "OCR_QUALITY_CHECKED"
    assert paths["ocr_plan"].is_file()

    plan = json.loads(
        paths["ocr_plan"].read_text(
            encoding="utf-8"
        )
    )

    assert (
        plan["classification"]
        == "BDA_ACCEPTED"
    )

    assert plan["fallback_pages"] == []


def test_process_book_stops_at_fallback_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    paths = build_paths(tmp_path)

    prepare_book_artifacts(
        paths,
        language="Urdu",
        subject="Urdu",
        page_count=1,
        normalized_text=(
            "This is corrupted English OCR output "
            "instead of the expected Urdu script."
        ),
    )

    write_json(
        paths["ocr_approval"],
        {
            "ocr_engine": "surya-ocr",
            "model": "datalab-to/surya-ocr-2",
            "approved_for_pipeline_integration": True,
            "full_book_run_authorized": False,
        },
    )

    monkeypatch.setattr(
        runner,
        "paths_for_book",
        lambda book_id, version: paths,
    )

    state_path = tmp_path / "state.json"
    state = create_state(
        runner,
        state_path,
    )

    result = runner.process_book(
        {
            "book_id": "grade-1-urdu-test",
            "grade": 1,
            "title": "Grade 1 Urdu Test Textbook",
            "source_bucket": "test-bucket",
            "source_zip_key": "NCERT I-X/Standard-I/Urdu-Test.zip",
            "language": "Urdu",
            "subject": "Urdu",
        },
        registry_path=tmp_path / "registry.json",
        state_path=state_path,
        state=state,
        maximum_retries=0,
        dry_run=False,
    )

    assert result == "OCR_FALLBACK_REQUIRED"

    plan = json.loads(
        paths["ocr_plan"].read_text(
            encoding="utf-8"
        )
    )

    assert (
        plan["classification"]
        == "OCR_FALLBACK_REQUIRED"
    )

    assert plan["fallback_pages"] == [1]

    assert (
        state["books"][
            "grade-1-urdu-test"
        ]["status"]
        == "OCR_FALLBACK_REQUIRED"
    )


def test_existing_valid_plan_is_resume_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    paths = build_paths(tmp_path)

    prepare_book_artifacts(
        paths,
        language="Urdu",
        subject="Urdu",
        page_count=1,
        normalized_text=(
            "Corrupted English output."
        ),
    )

    write_json(
        paths["ocr_approval"],
        {
            "approved_for_pipeline_integration": True,
            "full_book_run_authorized": False,
        },
    )

    write_json(
        paths["ocr_plan"],
        {
            "classification": (
                "OCR_FALLBACK_REQUIRED"
            ),
            "fallback_pages": [1],
            "accepted_bda_pages": [],
        },
    )

    monkeypatch.setattr(
        runner,
        "paths_for_book",
        lambda book_id, version: paths,
    )

    def unexpected_command(*args, **kwargs):
        raise AssertionError(
            "Planner must not rerun for a "
            "valid existing plan."
        )

    monkeypatch.setattr(
        runner,
        "run_command",
        unexpected_command,
    )

    state_path = tmp_path / "state.json"
    state = create_state(
        runner,
        state_path,
    )

    result = runner.process_book(
        {
            "book_id": "grade-1-urdu-test",
            "grade": 1,
            "title": "Grade 1 Urdu Test Textbook",
            "source_bucket": "test-bucket",
            "source_zip_key": "NCERT I-X/Standard-I/Urdu-Test.zip",
            "language": "Urdu",
            "subject": "Urdu",
        },
        registry_path=tmp_path / "registry.json",
        state_path=state_path,
        state=state,
        maximum_retries=0,
        dry_run=False,
    )

    assert result == "OCR_FALLBACK_REQUIRED"


def test_invalid_existing_plan_is_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    paths = build_paths(tmp_path)

    prepare_book_artifacts(
        paths,
        language="Urdu",
        subject="Urdu",
        page_count=1,
        normalized_text="Corrupted English output.",
    )

    write_json(
        paths["ocr_plan"],
        {
            "classification": "UNKNOWN",
        },
    )

    monkeypatch.setattr(
        runner,
        "paths_for_book",
        lambda book_id, version: paths,
    )

    state_path = tmp_path / "state.json"
    state = create_state(
        runner,
        state_path,
    )

    with pytest.raises(
        RuntimeError,
        match="invalid or incomplete",
    ):
        runner.process_book(
            {
                "book_id": "grade-1-urdu-test",
                "grade": 1,
                "title": "Grade 1 Urdu Test Textbook",
                "source_bucket": "test-bucket",
                "source_zip_key": "NCERT I-X/Standard-I/Urdu-Test.zip",
                "language": "Urdu",
                "subject": "Urdu",
            },
            registry_path=(
                tmp_path / "registry.json"
            ),
            state_path=state_path,
            state=state,
            maximum_retries=0,
            dry_run=False,
        )


def test_format_fallback_page_spec() -> None:
    runner = load_runner()

    assert runner.format_fallback_page_spec(
        (1, 5, 17, 60)
    ) == "1,5,17,60"


def test_build_surya_fallback_command() -> None:
    runner = load_runner()

    command = runner.build_surya_fallback_command(
        book_id="grade-1-urdu-test",
        version="v1",
        canonical_pdf=Path(
            "/tmp/textbook.pdf"
        ),
        output_root=Path(
            "/tmp/surya"
        ),
        expected_language="Urdu",
        fallback_pages=(1, 5, 17),
        approval_record=Path(
            "/tmp/approval.json"
        ),
    )

    assert (
        "run_surya_ocr_fallback.py"
        in command[1]
    )

    assert command[
        command.index("--pages") + 1
    ] == "1,5,17"

    assert "--resume" in command


def test_authorized_fallback_invokes_surya(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    paths = build_paths(tmp_path)

    prepare_book_artifacts(
        paths,
        language="Urdu",
        subject="Urdu",
        page_count=1,
        normalized_text=(
            "Corrupted English OCR output "
            "instead of Urdu script."
        ),
    )

    write_json(
        paths["ocr_plan"],
        {
            "classification": (
                "OCR_FALLBACK_REQUIRED"
            ),
            "fallback_pages": [1],
            "accepted_bda_pages": [],
        },
    )

    write_json(
        paths["ocr_approval"],
        {
            "ocr_engine": "surya-ocr",
            "model": "datalab-to/surya-ocr-2",
            "approved_for_pipeline_integration": True,
            "full_book_run_authorized": True,
        },
    )

    monkeypatch.setattr(
        runner,
        "paths_for_book",
        lambda book_id, version: paths,
    )

    recorded_commands: list[list[str]] = []

    def fake_run_command(
        command: list[str],
        *,
        log_path: Path,
        maximum_retries: int,
        dry_run: bool,
    ) -> None:
        del log_path
        del maximum_retries

        assert dry_run is False
        recorded_commands.append(command)

        write_json(
            paths["ocr_fallback_report"],
            {
                "classification": "PASS",
                "accepted_for_pipeline": True,
                "expected_pages": [1],
                "missing_pages": [],
                "passed": 1,
                "review": 0,
                "failed": 0,
                "pages": [],
            },
        )

        paths[
            "ocr_fallback_marker"
        ].parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        paths[
            "ocr_fallback_marker"
        ].write_text(
            "PASS\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        runner,
        "run_command",
        fake_run_command,
    )

    state_path = tmp_path / "state.json"

    state = create_state(
        runner,
        state_path,
    )

    result = runner.process_book(
        {
            "book_id": "grade-1-urdu-test",
            "grade": 1,
            "title": "Grade 1 Urdu Test Textbook",
            "language": "Urdu",
            "subject": "Urdu",
            "source_bucket": "test-bucket",
            "source_zip_key": (
                "NCERT I-X/Standard-I/"
                "Urdu-Test.zip"
            ),
        },
        registry_path=tmp_path / "registry.json",
        state_path=state_path,
        state=state,
        maximum_retries=0,
        dry_run=False,
    )

    assert result == "OCR_FALLBACK_VERIFIED"
    assert len(recorded_commands) == 1

    command = recorded_commands[0]

    assert (
        "run_surya_ocr_fallback.py"
        in command[1]
    )

    assert command[
        command.index("--pages") + 1
    ] == "1"

    assert (
        state["books"][
            "grade-1-urdu-test"
        ]["status"]
        == "OCR_FALLBACK_VERIFIED"
    )


def test_downstream_command_builders(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    paths = build_paths(tmp_path)

    roots = (
        tmp_path / "normalized-a",
        tmp_path / "normalized-b",
    )

    command = (
        runner.build_unified_merge_command(
            paths=paths,
            normalized_roots=roots,
        )
    )

    assert (
        "merge_bda_surya_records.py"
        in command[1]
    )

    assert command.count(
        "--normalized-root"
    ) == 2

    assert "--ocr-plan" in command
    assert "--page-map" in command
    assert "--output-dir" in command

    embedding_command = (
        runner
        .build_embedding_preparation_command(
            paths=paths
        )
    )

    assert (
        "prepare_embedding_records.py"
        in embedding_command[1]
    )

    artifacts = (
        runner.downstream_artifact_paths(
            paths
        )
    )

    assert (
        str(artifacts["unified_root"])
        in embedding_command
    )

    assert (
        str(artifacts["embedding_root"])
        in embedding_command
    )


def test_discover_normalized_record_roots(
    tmp_path: Path,
) -> None:
    runner = load_runner()

    first = (
        tmp_path
        / "batch-0001"
        / "normalized"
    )

    second = (
        tmp_path
        / "batch-0002"
        / "normalized"
    )

    ignored = (
        tmp_path
        / "unified-normalized"
    )

    for directory in (
        first,
        second,
        ignored,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            directory
            / "content-units.jsonl"
        ).write_text(
            '{"unit_id":"x"}\n',
            encoding="utf-8",
        )

    discovered = (
        runner
        .discover_normalized_record_roots(
            tmp_path
        )
    )

    assert discovered == tuple(
        sorted(
            (
                first.resolve(),
                second.resolve(),
            ),
            key=str,
        )
    )


def test_process_book_prepares_unified_and_embedding_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    paths = build_paths(tmp_path)

    prepare_book_artifacts(
        paths,
        language="English",
        subject="English",
        page_count=1,
        normalized_text=(
            "This is a complete English textbook "
            "paragraph that passes the quality gate."
        ),
    )

    normalized_root = (
        paths["bda_normalized"]
        / "batch-0001"
        / "normalized"
    )

    normalized_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        normalized_root
        / "content-units.jsonl"
    ).write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "unit_id": "unit-page-1",
                "book_id": (
                    "grade-1-english-test"
                ),
                "book_version": "v1",
                "source_kind": (
                    "bda_standard_output"
                ),
                "source_pdf": str(
                    paths["canonical_pdf"]
                ),
                "bda_element_id": "element-1",
                "element_index": 1,
                "element_type": "TEXT",
                "element_sub_type": (
                    "PARAGRAPH"
                ),
                "modality": "paragraph",
                "reading_order": 1,
                "source_page_numbers": [1],
                "locations": [],
                "raw_text": "Valid English text",
                "markdown": "Valid English text",
                "search_text": (
                    "Valid English text"
                ),
                "asset_s3_uris": [],
                "asset_local_paths": [],
                "quality_flags": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = (
        runner.downstream_artifact_paths(
            paths
        )
    )

    titan_artifacts = (
        runner.titan_artifact_paths(
            paths
        )
    )

    search_artifacts = (
        runner.opensearch_artifact_paths(
            paths
        )
    )

    write_json(
        artifacts["page_map"],
        {
            "schema_version": "1.0",
            "book_id": (
                "grade-1-english-test"
            ),
            "book_version": "v1",
            "canonical_page_count": 1,
            "pages": [
                {
                    "canonical_page": 1,
                    "page_type": (
                        "front_matter"
                    ),
                    "document_order": 1,
                    "document_id": (
                        "front-matter"
                    ),
                    "document_type": (
                        "front_matter"
                    ),
                    "document_title": (
                        "Front Matter"
                    ),
                    "source_filename": (
                        "source.pdf"
                    ),
                    "source_page": 1,
                    "unit_number": None,
                    "chapter_id": None,
                    "chapter_title": None,
                    "chapter_page": None,
                }
            ],
        },
    )

    write_json(
        paths["ocr_plan"],
        {
            "classification": (
                "BDA_ACCEPTED"
            ),
            "fallback_pages": [],
            "accepted_bda_pages": [1],
            "assessments": [],
        },
    )

    monkeypatch.setattr(
        runner,
        "paths_for_book",
        lambda book_id, version: paths,
    )

    commands: list[list[str]] = []

    def fake_run_command(
        command: list[str],
        *,
        log_path: Path,
        maximum_retries: int,
        dry_run: bool,
    ) -> None:
        del log_path
        del maximum_retries

        assert dry_run is False
        commands.append(command)

        if (
            "merge_bda_surya_records.py"
            in command[1]
        ):
            artifacts[
                "unified_root"
            ].mkdir(
                parents=True,
                exist_ok=True,
            )

            artifacts[
                "unified_content_units"
            ].write_text(
                '{"unit_id":"unified-1"}\n',
                encoding="utf-8",
            )

            artifacts[
                "unified_figures"
            ].write_text(
                "",
                encoding="utf-8",
            )

            artifacts[
                "unified_tables"
            ].write_text(
                "",
                encoding="utf-8",
            )

            write_json(
                artifacts[
                    "unified_report"
                ],
                {
                    "status": "VALID",
                },
            )

            artifacts[
                "unified_marker"
            ].write_text(
                "VALID\n",
                encoding="utf-8",
            )

            return

        if (
            "prepare_embedding_records.py"
            in command[1]
        ):
            artifacts[
                "embedding_root"
            ].mkdir(
                parents=True,
                exist_ok=True,
            )

            artifacts[
                "embedding_records"
            ].write_text(
                '{"record_id":"record-1"}\n',
                encoding="utf-8",
            )

            artifacts[
                "embedding_skipped"
            ].write_text(
                "",
                encoding="utf-8",
            )

            write_json(
                artifacts[
                    "embedding_report"
                ],
                {
                    "status": "VALID",
                    "prepared_records": 1,
                },
            )

            return

        if (
            "embed_records_titan_v2.py"
            in command[1]
        ):
            titan_artifacts[
                "root"
            ].mkdir(
                parents=True,
                exist_ok=True,
            )

            titan_artifacts[
                "embeddings"
            ].write_text(
                '{"record_id":"record-1"}\n',
                encoding="utf-8",
            )

            write_json(
                titan_artifacts[
                    "manifest"
                ],
                {
                    "status": "COMPLETED",
                    "input_record_count": 1,
                    "completed_record_count": 1,
                },
            )

            return

        if (
            "prepare_opensearch_bulk.py"
            in command[1]
        ):
            search_artifacts[
                "bulk_root"
            ].mkdir(
                parents=True,
                exist_ok=True,
            )

            search_artifacts[
                "bulk_payload"
            ].write_text(
                '{"index":{"_id":"record-1"}}\n'
                '{"record_id":"record-1"}\n',
                encoding="utf-8",
            )

            write_json(
                search_artifacts[
                    "bulk_preparation_report"
                ],
                {
                    "status": "PREPARED",
                    "validation": {
                        "document_count": 1,
                        "errors": [],
                    },
                },
            )

            return

        if (
            "create_textbook_index.py"
            in command[1]
        ):
            search_artifacts[
                "root"
            ].mkdir(
                parents=True,
                exist_ok=True,
            )

            write_json(
                search_artifacts[
                    "index_report"
                ],
                {
                    "status": "PROVISIONED",
                    "action": "matching",
                },
            )

            return

        if (
            "upload_opensearch_bulk.py"
            in command[1]
        ):
            write_json(
                search_artifacts[
                    "upload_report"
                ],
                {
                    "status": "COMPLETED",
                    "prepared_document_count": 1,
                },
            )

            return

        raise AssertionError(
            f"Unexpected command: {command}"
        )

    monkeypatch.setattr(
        runner,
        "run_command",
        fake_run_command,
    )

    state_path = tmp_path / "state.json"

    state = create_state(
        runner,
        state_path,
    )

    result = runner.process_book(
        {
            "book_id": (
                "grade-1-english-test"
            ),
            "grade": 1,
            "title": (
                "Grade 1 English Test Textbook"
            ),
            "language": "English",
            "subject": "English",
            "source_bucket": "test-bucket",
            "source_zip_key": (
                "NCERT I-X/Standard-I/"
                "English-Test.zip"
            ),
        },
        registry_path=tmp_path / "registry.json",
        state_path=state_path,
        state=state,
        maximum_retries=0,
        dry_run=False,
    )

    assert (
        result
        == "INDEXED"
    )

    assert len(commands) == 6

    assert (
        state["books"][
            "grade-1-english-test"
        ]["status"]
        == "INDEXED"
    )

    history = [
        item["status"]
        for item in state["books"][
            "grade-1-english-test"
        ]["history"]
    ]

    assert (
        "UNIFIED_RECORDS_PREPARED"
        in history
    )

    assert (
        "EMBEDDING_RECORDS_PREPARED"
        in history
    )


def test_process_book_runs_titan_and_opensearch_to_indexed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    paths = build_paths(tmp_path)

    prepare_book_artifacts(
        paths,
        language="English",
        subject="English",
        page_count=1,
        normalized_text=(
            "This is a complete English textbook "
            "paragraph that passes the quality gate."
        ),
    )

    normalized_root = (
        paths["bda_normalized"]
        / "batch-0001"
        / "normalized"
    )

    normalized_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        normalized_root
        / "content-units.jsonl"
    ).write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "unit_id": "unit-page-1",
                "book_id": (
                    "grade-1-english-test"
                ),
                "book_version": "v1",
                "source_kind": (
                    "bda_standard_output"
                ),
                "source_pdf": str(
                    paths["canonical_pdf"]
                ),
                "bda_element_id": "element-1",
                "element_index": 1,
                "element_type": "TEXT",
                "element_sub_type": (
                    "PARAGRAPH"
                ),
                "modality": "paragraph",
                "reading_order": 1,
                "source_page_numbers": [1],
                "locations": [],
                "raw_text": "Valid English text",
                "markdown": "Valid English text",
                "search_text": (
                    "Valid English text"
                ),
                "asset_s3_uris": [],
                "asset_local_paths": [],
                "quality_flags": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = (
        runner.downstream_artifact_paths(
            paths
        )
    )

    write_json(
        artifacts["page_map"],
        {
            "schema_version": "1.0",
            "book_id": (
                "grade-1-english-test"
            ),
            "book_version": "v1",
            "canonical_page_count": 1,
            "pages": [
                {
                    "canonical_page": 1,
                    "page_type": (
                        "front_matter"
                    ),
                    "document_order": 1,
                    "document_id": (
                        "front-matter"
                    ),
                    "document_type": (
                        "front_matter"
                    ),
                    "document_title": (
                        "Front Matter"
                    ),
                    "source_filename": (
                        "source.pdf"
                    ),
                    "source_page": 1,
                    "unit_number": None,
                    "chapter_id": None,
                    "chapter_title": None,
                    "chapter_page": None,
                }
            ],
        },
    )

    write_json(
        paths["ocr_plan"],
        {
            "classification": (
                "BDA_ACCEPTED"
            ),
            "fallback_pages": [],
            "accepted_bda_pages": [1],
            "assessments": [],
        },
    )

    artifacts[
        "unified_root"
    ].mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts[
        "unified_content_units"
    ].write_text(
        '{"unit_id":"unified-1"}\n',
        encoding="utf-8",
    )

    artifacts[
        "unified_figures"
    ].write_text(
        "",
        encoding="utf-8",
    )

    artifacts[
        "unified_tables"
    ].write_text(
        "",
        encoding="utf-8",
    )

    write_json(
        artifacts["unified_report"],
        {
            "status": "VALID",
        },
    )

    artifacts[
        "unified_marker"
    ].write_text(
        "VALID\n",
        encoding="utf-8",
    )

    artifacts[
        "embedding_root"
    ].mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts[
        "embedding_records"
    ].write_text(
        '{"record_id":"record-1"}\n',
        encoding="utf-8",
    )

    artifacts[
        "embedding_skipped"
    ].write_text(
        "",
        encoding="utf-8",
    )

    write_json(
        artifacts["embedding_report"],
        {
            "status": "VALID",
            "prepared_records": 1,
        },
    )

    monkeypatch.setattr(
        runner,
        "paths_for_book",
        lambda book_id, version: paths,
    )

    completed = {
        "titan": False,
        "bulk": False,
        "index": False,
        "upload": False,
    }

    monkeypatch.setattr(
        runner,
        "titan_embeddings_are_valid",
        lambda *args, **kwargs: (
            completed["titan"]
        ),
    )

    monkeypatch.setattr(
        runner,
        "opensearch_bulk_is_valid",
        lambda *args, **kwargs: (
            completed["bulk"]
        ),
    )

    monkeypatch.setattr(
        runner,
        "opensearch_index_is_valid",
        lambda *args, **kwargs: (
            completed["index"]
        ),
    )

    monkeypatch.setattr(
        runner,
        "opensearch_upload_is_valid",
        lambda *args, **kwargs: (
            completed["upload"]
        ),
    )

    commands: list[str] = []

    def fake_run_command(
        command: list[str],
        *,
        log_path: Path,
        maximum_retries: int,
        dry_run: bool,
    ) -> None:
        del log_path
        del maximum_retries

        assert dry_run is False

        script_name = Path(
            command[1]
        ).name

        commands.append(script_name)

        if (
            script_name
            == "embed_records_titan_v2.py"
        ):
            assert completed == {
                "titan": False,
                "bulk": False,
                "index": False,
                "upload": False,
            }

            completed["titan"] = True
            return

        if (
            script_name
            == "prepare_opensearch_bulk.py"
        ):
            assert completed["titan"]
            assert not completed["bulk"]

            completed["bulk"] = True
            return

        if script_name in {
            "create_textbook_index.py",
            "create_opensearch_index.py",
        }:
            assert completed["bulk"]
            assert not completed["index"]

            completed["index"] = True
            return

        if (
            script_name
            == "upload_opensearch_bulk.py"
        ):
            assert completed["index"]
            assert not completed["upload"]

            completed["upload"] = True
            return

        raise AssertionError(
            f"Unexpected command: {command}"
        )

    monkeypatch.setattr(
        runner,
        "run_command",
        fake_run_command,
    )

    state_path = tmp_path / "state.json"

    state = create_state(
        runner,
        state_path,
    )

    result = runner.process_book(
        {
            "book_id": (
                "grade-1-english-test"
            ),
            "grade": 1,
            "title": (
                "Grade 1 English Test Textbook"
            ),
            "language": "English",
            "subject": "English",
            "source_bucket": "test-bucket",
            "source_zip_key": (
                "NCERT I-X/Standard-I/"
                "English-Test.zip"
            ),
        },
        registry_path=(
            tmp_path / "registry.json"
        ),
        state_path=state_path,
        state=state,
        maximum_retries=0,
        dry_run=False,
    )

    assert result == "INDEXED"

    assert completed == {
        "titan": True,
        "bulk": True,
        "index": True,
        "upload": True,
    }

    assert commands == [
        "embed_records_titan_v2.py",
        "prepare_opensearch_bulk.py",
        "create_textbook_index.py",
        "upload_opensearch_bulk.py",
    ]



def test_process_book_advances_from_indexed_to_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    paths = build_paths(tmp_path)

    prepare_book_artifacts(
        paths,
        language="English",
        subject="English",
        page_count=1,
        normalized_text=(
            "This is a complete English textbook "
            "paragraph that passes the quality gate."
        ),
    )

    write_json(
        paths["ocr_plan"],
        {
            "classification": "BDA_ACCEPTED",
            "fallback_pages": [],
            "accepted_bda_pages": [1],
            "assessments": [],
        },
    )

    monkeypatch.setattr(
        runner,
        "paths_for_book",
        lambda book_id, version: paths,
    )

    monkeypatch.setattr(
        runner,
        "embedding_records_are_valid",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        runner,
        "titan_embeddings_are_valid",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        runner,
        "opensearch_bulk_is_valid",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        runner,
        "opensearch_index_is_valid",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        runner,
        "opensearch_upload_is_valid",
        lambda *args, **kwargs: True,
    )

    artifacts = (
        runner.final_verification_artifact_paths(
            paths
        )
    )

    for report_path in (
        artifacts["bulk_upload_report"],
        artifacts["vector_report"],
        artifacts["hybrid_report"],
        artifacts["rag_report"],
    ):
        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        write_json(
            report_path,
            {
                "status": "PASSED",
            },
        )

    completed = {
        "verified": False,
    }

    monkeypatch.setattr(
        runner,
        "final_verification_is_valid",
        lambda *args, **kwargs: (
            completed["verified"]
        ),
    )

    commands: list[str] = []

    def fake_run_command(
        command: list[str],
        *,
        log_path: Path,
        maximum_retries: int,
        dry_run: bool,
    ) -> None:
        del log_path
        del maximum_retries

        assert dry_run is False

        script_name = Path(
            command[1]
        ).name

        commands.append(script_name)

        assert (
            script_name
            == "verify_indexed_textbook.py"
        )

        artifacts[
            "verification_root"
        ].mkdir(
            parents=True,
            exist_ok=True,
        )

        write_json(
            artifacts[
                "verification_report"
            ],
            {
                "status": "VERIFIED",
                "all_checks_passed": True,
            },
        )

        artifacts[
            "verification_marker"
        ].write_text(
            "VERIFIED\n",
            encoding="utf-8",
        )

        completed["verified"] = True

    monkeypatch.setattr(
        runner,
        "run_command",
        fake_run_command,
    )

    state_path = tmp_path / "state.json"

    state = create_state(
        runner,
        state_path,
    )

    result = runner.process_book(
        {
            "book_id": (
                "grade-1-english-test"
            ),
            "grade": 1,
            "title": (
                "Grade 1 English Test Textbook"
            ),
            "language": "English",
            "subject": "English",
            "source_bucket": "test-bucket",
            "source_zip_key": (
                "NCERT I-X/Standard-I/"
                "English-Test.zip"
            ),
        },
        registry_path=(
            tmp_path / "registry.json"
        ),
        state_path=state_path,
        state=state,
        maximum_retries=0,
        dry_run=False,
    )

    assert result == "VERIFIED"
    assert completed["verified"] is True

    assert commands == [
        "verify_indexed_textbook.py",
    ]



def test_process_book_runs_evaluations_then_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    paths = build_paths(tmp_path)

    prepare_book_artifacts(
        paths,
        language="English",
        subject="English",
        page_count=1,
        normalized_text=(
            "This is a complete English textbook "
            "paragraph that passes the quality gate."
        ),
    )

    write_json(
        paths["ocr_plan"],
        {
            "classification": "BDA_ACCEPTED",
            "fallback_pages": [],
            "accepted_bda_pages": [1],
            "assessments": [],
        },
    )

    monkeypatch.setattr(
        runner,
        "paths_for_book",
        lambda book_id, version: paths,
    )

    monkeypatch.setattr(
        runner,
        "embedding_records_are_valid",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        runner,
        "titan_embeddings_are_valid",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        runner,
        "opensearch_bulk_is_valid",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        runner,
        "opensearch_index_is_valid",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        runner,
        "opensearch_upload_is_valid",
        lambda *args, **kwargs: True,
    )

    evaluation_cases = {
        "vector": (
            tmp_path / "vector-tests.json"
        ),
        "hybrid": (
            tmp_path / "hybrid-tests.json"
        ),
        "rag": (
            tmp_path / "rag-tests.json"
        ),
    }

    for case_path in (
        evaluation_cases.values()
    ):
        write_json(
            case_path,
            {
                "schema_version": "1.0",
                "book_id": (
                    "grade-1-english-test"
                ),
                "book_version": "v1",
                "tests": [
                    {
                        "test_id": "test-1",
                        "question": (
                            "Test question"
                        ),
                    }
                ],
            },
        )

    monkeypatch.setattr(
        runner,
        "evaluation_test_case_paths",
        lambda **kwargs: evaluation_cases,
    )

    artifacts = (
        runner.final_verification_artifact_paths(
            paths
        )
    )

    # bulk_upload_fixture_created
    expected_upload_report = (
        runner.opensearch_artifact_paths(
            paths
        )["upload_report"]
    )

    assert (
        artifacts["bulk_upload_report"]
        == expected_upload_report
    )

    expected_upload_report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        expected_upload_report,
        {
            "status": "COMPLETED",
            "uploaded": True,
            "prepared_document_count": 1,
            "expected_final_count": 1,
            "final_count": 1,
            "bulk_result": {
                "failure_count": 0,
            },
        },
    )

    commands: list[str] = []

    def write_passing_evaluation(
        report_path: Path,
    ) -> None:
        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_json(
            report_path,
            {
                "status": "PASSED",
                "all_tests_passed": True,
                "test_count": 1,
                "passed_test_count": 1,
                "failed_test_count": 0,
            },
        )

    def fake_run_command(
        command: list[str],
        *,
        log_path: Path,
        maximum_retries: int,
        dry_run: bool,
    ) -> None:
        del log_path
        del maximum_retries

        assert dry_run is False

        script_name = Path(
            command[1]
        ).name

        commands.append(script_name)

        if (
            script_name
            == (
                "evaluate_opensearch_"
                "vector_retrieval.py"
            )
        ):
            write_passing_evaluation(
                artifacts["vector_report"]
            )
            return

        if (
            script_name
            == (
                "evaluate_opensearch_"
                "hybrid_retrieval.py"
            )
        ):
            write_passing_evaluation(
                artifacts["hybrid_report"]
            )
            return

        if (
            script_name
            == "evaluate_book_rag.py"
        ):
            write_passing_evaluation(
                artifacts["rag_report"]
            )
            return

        if (
            script_name
            == "verify_indexed_textbook.py"
        ):
            artifacts[
                "verification_root"
            ].mkdir(
                parents=True,
                exist_ok=True,
            )

            write_json(
                artifacts[
                    "verification_report"
                ],
                {
                    "status": "VERIFIED",
                    "all_checks_passed": True,
                },
            )

            artifacts[
                "verification_marker"
            ].write_text(
                "VERIFIED\n",
                encoding="utf-8",
            )
            return

        raise AssertionError(
            f"Unexpected command: {command}"
        )

    monkeypatch.setattr(
        runner,
        "run_command",
        fake_run_command,
    )

    state_path = tmp_path / "state.json"

    state = create_state(
        runner,
        state_path,
    )

    result = runner.process_book(
        {
            "book_id": (
                "grade-1-english-test"
            ),
            "grade": 1,
            "title": (
                "Grade 1 English Test Textbook"
            ),
            "language": "English",
            "subject": "English",
            "source_bucket": "test-bucket",
            "source_zip_key": (
                "NCERT I-X/Standard-I/"
                "English-Test.zip"
            ),
        },
        registry_path=(
            tmp_path / "registry.json"
        ),
        state_path=state_path,
        state=state,
        maximum_retries=0,
        dry_run=False,
    )

    assert result == "VERIFIED"

    assert commands == [
        (
            "evaluate_opensearch_"
            "vector_retrieval.py"
        ),
        (
            "evaluate_opensearch_"
            "hybrid_retrieval.py"
        ),
        "evaluate_book_rag.py",
        "verify_indexed_textbook.py",
    ]



def test_final_verification_reuses_opensearch_upload_report_path(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    paths = build_paths(tmp_path)

    opensearch_artifacts = (
        runner.opensearch_artifact_paths(
            paths
        )
    )

    verification_artifacts = (
        runner.final_verification_artifact_paths(
            paths
        )
    )

    assert (
        verification_artifacts[
            "bulk_upload_report"
        ]
        == opensearch_artifacts[
            "upload_report"
        ]
    )



def test_missing_evaluation_cases_retain_indexed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    paths = build_paths(tmp_path)

    prepare_book_artifacts(
        paths,
        language="English",
        subject="English",
        page_count=1,
        normalized_text=(
            "This is a complete English textbook "
            "paragraph that passes the quality gate."
        ),
    )

    write_json(
        paths["ocr_plan"],
        {
            "classification": "BDA_ACCEPTED",
            "fallback_pages": [],
            "accepted_bda_pages": [1],
            "assessments": [],
        },
    )

    monkeypatch.setattr(
        runner,
        "paths_for_book",
        lambda book_id, version: paths,
    )

    monkeypatch.setattr(
        runner,
        "embedding_records_are_valid",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        runner,
        "titan_embeddings_are_valid",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        runner,
        "opensearch_bulk_is_valid",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        runner,
        "opensearch_index_is_valid",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        runner,
        "opensearch_upload_is_valid",
        lambda *args, **kwargs: True,
    )

    # missing_evaluation_bulk_fixture_created
    bulk_upload_report = (
        runner.opensearch_artifact_paths(
            paths
        )["upload_report"]
    )

    bulk_upload_report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        bulk_upload_report,
        {
            "status": "COMPLETED",
            "uploaded": True,
            "prepared_document_count": 1,
            "expected_final_count": 1,
            "final_count": 1,
            "bulk_result": {
                "failure_count": 0,
            },
        },
    )

    missing_cases = {
        "vector": tmp_path / "missing-vector.json",
        "hybrid": tmp_path / "missing-hybrid.json",
        "rag": tmp_path / "missing-rag.json",
    }

    monkeypatch.setattr(
        runner,
        "evaluation_test_case_paths",
        lambda **kwargs: missing_cases,
    )

    commands: list[list[str]] = []

    monkeypatch.setattr(
        runner,
        "run_command",
        lambda command, **kwargs: commands.append(
            command
        ),
    )

    state_path = tmp_path / "state.json"

    state = create_state(
        runner,
        state_path,
    )

    result = runner.process_book(
        {
            "book_id": "grade-1-english-test",
            "grade": 1,
            "title": "Grade 1 English Test Textbook",
            "language": "English",
            "subject": "English",
            "source_bucket": "test-bucket",
            "source_zip_key": (
                "NCERT I-X/Standard-I/"
                "English-Test.zip"
            ),
        },
        registry_path=tmp_path / "registry.json",
        state_path=state_path,
        state=state,
        maximum_retries=0,
        dry_run=False,
    )

    assert result == "INDEXED"
    assert commands == []


def test_existing_evaluation_reports_are_resume_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    paths = build_paths(tmp_path)

    prepare_book_artifacts(
        paths,
        language="English",
        subject="English",
        page_count=1,
        normalized_text=(
            "This is a complete English textbook "
            "paragraph that passes the quality gate."
        ),
    )

    write_json(
        paths["ocr_plan"],
        {
            "classification": "BDA_ACCEPTED",
            "fallback_pages": [],
            "accepted_bda_pages": [1],
            "assessments": [],
        },
    )

    monkeypatch.setattr(
        runner,
        "paths_for_book",
        lambda book_id, version: paths,
    )

    monkeypatch.setattr(
        runner,
        "embedding_records_are_valid",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        runner,
        "titan_embeddings_are_valid",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        runner,
        "opensearch_bulk_is_valid",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        runner,
        "opensearch_index_is_valid",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        runner,
        "opensearch_upload_is_valid",
        lambda *args, **kwargs: True,
    )

    cases = {
        "vector": tmp_path / "vector-tests.json",
        "hybrid": tmp_path / "hybrid-tests.json",
        "rag": tmp_path / "rag-tests.json",
    }

    for case_path in cases.values():
        write_json(
            case_path,
            {
                "schema_version": "1.0",
                "book_id": "grade-1-english-test",
                "book_version": "v1",
                "tests": [
                    {
                        "test_id": "test-1",
                        "question": "Test question",
                    }
                ],
            },
        )

    monkeypatch.setattr(
        runner,
        "evaluation_test_case_paths",
        lambda **kwargs: cases,
    )

    artifacts = (
        runner.final_verification_artifact_paths(
            paths
        )
    )

    passing_report = {
        "status": "PASSED",
        "all_tests_passed": True,
        "test_count": 1,
        "passed_test_count": 1,
        "failed_test_count": 0,
    }

    for report_path in (
        artifacts["vector_report"],
        artifacts["hybrid_report"],
        artifacts["rag_report"],
    ):
        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        write_json(
            report_path,
            passing_report,
        )

    bulk_report = (
        runner.opensearch_artifact_paths(
            paths
        )["upload_report"]
    )

    bulk_report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        bulk_report,
        {
            "status": "COMPLETED",
            "uploaded": True,
            "prepared_document_count": 1,
            "expected_final_count": 1,
            "final_count": 1,
            "bulk_result": {
                "failure_count": 0,
            },
        },
    )

    commands: list[str] = []

    def fake_run_command(
        command: list[str],
        *,
        log_path: Path,
        maximum_retries: int,
        dry_run: bool,
    ) -> None:
        del log_path
        del maximum_retries

        assert dry_run is False

        script_name = Path(
            command[1]
        ).name

        commands.append(script_name)

        assert (
            script_name
            == "verify_indexed_textbook.py"
        )

        artifacts[
            "verification_root"
        ].mkdir(
            parents=True,
            exist_ok=True,
        )

        write_json(
            artifacts[
                "verification_report"
            ],
            {
                "status": "VERIFIED",
                "all_checks_passed": True,
            },
        )

        artifacts[
            "verification_marker"
        ].write_text(
            "VERIFIED\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        runner,
        "run_command",
        fake_run_command,
    )

    state_path = tmp_path / "state.json"

    state = create_state(
        runner,
        state_path,
    )

    result = runner.process_book(
        {
            "book_id": "grade-1-english-test",
            "grade": 1,
            "title": "Grade 1 English Test Textbook",
            "language": "English",
            "subject": "English",
            "source_bucket": "test-bucket",
            "source_zip_key": (
                "NCERT I-X/Standard-I/"
                "English-Test.zip"
            ),
        },
        registry_path=tmp_path / "registry.json",
        state_path=state_path,
        state=state,
        maximum_retries=0,
        dry_run=False,
    )

    assert result == "VERIFIED"

    assert commands == [
        "verify_indexed_textbook.py",
    ]



def test_kaveri_is_not_hardcoded_verified() -> None:
    runner = load_runner()

    assert (
        "grade-9-english-kaveri"
        not in runner.KNOWN_VERIFIED_BOOKS
    )

    cases = (
        runner.evaluation_test_case_paths(
            book_id=(
                "grade-9-english-kaveri"
            ),
            version="v1",
        )
    )

    assert all(
        path.is_file()
        for path in cases.values()
    )
