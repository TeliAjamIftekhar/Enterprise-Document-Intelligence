from __future__ import annotations

import json
import re
import shutil
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

import fitz
import numpy as np

from src.chapter_manifest import (
    ChapterManifest,
)
from src.chapter_source import (
    calculate_sha256,
    validate_pdf,
)


def normalize_text(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def build_page_map(
    manifest: ChapterManifest,
) -> dict[str, object]:
    pages: list[dict[str, object]] = []

    canonical_page_count = (
        manifest.canonical_layout
        .canonical_page_count
    )

    for page_number in range(
        1,
        canonical_page_count + 1,
    ):
        document = (
            manifest
            .document_for_canonical_page(
                page_number
            )
        )

        chapter = (
            manifest
            .chapter_for_canonical_page(
                page_number
            )
        )

        if document is None:
            if (
                page_number
                <= manifest.canonical_layout
                .leading_blank_pages
            ):
                page_type = "leading_blank"
            else:
                page_type = "trailing_blank"

            pages.append({
                "canonical_page": page_number,
                "page_type": page_type,
                "document_order": None,
                "document_id": None,
                "document_type": None,
                "document_title": None,
                "source_filename": None,
                "source_page": None,
                "unit_number": None,
                "chapter_id": None,
                "chapter_title": None,
                "chapter_page": None,
            })

            continue

        source_page = (
            page_number
            - document.canonical_start_page
            + 1
        )

        chapter_page = None

        if chapter is not None:
            chapter_page = (
                page_number
                - chapter.canonical_start_page
                + 1
            )

        pages.append({
            "canonical_page": page_number,
            "page_type": (
                document.document_type
            ),
            "document_order": (
                document.order
            ),
            "document_id": (
                document.document_id
            ),
            "document_type": (
                document.document_type
            ),
            "document_title": (
                document.title
            ),
            "source_filename": (
                document.source_filename
            ),
            "source_page": source_page,
            "unit_number": (
                document.unit_number
            ),
            "chapter_id": (
                chapter.chapter_id
                if chapter
                else None
            ),
            "chapter_title": (
                chapter.chapter_title
                if chapter
                else None
            ),
            "chapter_page": chapter_page,
        })

    return {
        "schema_version": "1.0",
        "book_id": manifest.book_id,
        "book_version": (
            manifest.book_version
        ),
        "title": manifest.title,
        "canonical_page_count": (
            canonical_page_count
        ),
        "source_document_count": len(
            manifest.documents
        ),
        "chapter_count": (
            manifest.chapter_count
        ),
        "pages": pages,
    }




RENDER_MAX_MAE = 2.0
RENDER_MAX_RMSE = 6.0
RENDER_MAX_CHANGED_ABOVE_10_PERCENT = 8.0


def page_geometry_signature(
    page: fitz.Page,
) -> tuple[object, ...]:
    def rect_values(
        rect: fitz.Rect,
    ) -> tuple[float, ...]:
        return (
            round(rect.x0, 4),
            round(rect.y0, 4),
            round(rect.x1, 4),
            round(rect.y1, 4),
        )

    return (
        rect_values(page.rect),
        rect_values(page.mediabox),
        rect_values(page.cropbox),
        page.rotation,
    )


def render_page_difference_metrics(
    source_page: fitz.Page,
    merged_page: fitz.Page,
) -> dict[str, object]:
    matrix = fitz.Matrix(1.0, 1.0)

    source_pixmap = source_page.get_pixmap(
        matrix=matrix,
        colorspace=fitz.csRGB,
        alpha=False,
    )

    merged_pixmap = merged_page.get_pixmap(
        matrix=matrix,
        colorspace=fitz.csRGB,
        alpha=False,
    )

    source_shape = (
        source_pixmap.width,
        source_pixmap.height,
        source_pixmap.n,
    )

    merged_shape = (
        merged_pixmap.width,
        merged_pixmap.height,
        merged_pixmap.n,
    )

    if source_shape != merged_shape:
        return {
            "shape_matches": False,
            "source_shape": source_shape,
            "merged_shape": merged_shape,
        }

    source_array = np.frombuffer(
        source_pixmap.samples,
        dtype=np.uint8,
    ).astype(np.int16)

    merged_array = np.frombuffer(
        merged_pixmap.samples,
        dtype=np.uint8,
    ).astype(np.int16)

    difference = np.abs(
        source_array - merged_array
    )

    squared_difference = (
        difference.astype(np.float64) ** 2
    )

    return {
        "shape_matches": True,
        "source_shape": source_shape,
        "merged_shape": merged_shape,
        "mean_absolute_difference": float(
            difference.mean()
        ),
        "root_mean_square_difference": float(
            np.sqrt(
                squared_difference.mean()
            )
        ),
        "maximum_channel_difference": int(
            difference.max()
        ),
        "changed_above_10_percent": float(
            np.mean(difference > 10)
            * 100
        ),
    }


def classify_text_extraction_variances(
    text_mismatches: list[
        dict[str, object]
    ],
    geometry_mismatches: list[
        dict[str, object]
    ],
    render_records: list[
        dict[str, object]
    ],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Separate harmless extraction variance from unsafe mismatch.

    A text mismatch is accepted only when page geometry
    matches and the rendered page is pixel-identical.
    """

    def page_key(
        record: dict[str, object],
    ) -> tuple[str, int, int]:
        return (
            str(record["source_filename"]),
            int(record["source_page"]),
            int(record["canonical_page"]),
        )

    geometry_mismatch_keys = {
        page_key(record)
        for record in geometry_mismatches
    }

    render_by_page = {
        page_key(record): record
        for record in render_records
    }

    accepted: list[dict[str, object]] = []
    unsafe: list[dict[str, object]] = []

    for mismatch in text_mismatches:
        key = page_key(mismatch)

        render_record = render_by_page.get(
            key
        )

        geometry_matches = (
            key not in geometry_mismatch_keys
        )

        exact_render = (
            render_record is not None
            and bool(
                render_record.get(
                    "shape_matches"
                )
            )
            and int(
                render_record.get(
                    "maximum_channel_difference",
                    1,
                )
            )
            == 0
        )

        if geometry_matches and exact_render:
            accepted.append({
                **mismatch,
                "reason": (
                    "text extraction variance "
                    "with exact visual equivalence"
                ),
            })
            continue

        unsafe.append({
            **mismatch,
            "geometry_matches": (
                geometry_matches
            ),
            "exact_render": exact_render,
        })

    return accepted, unsafe


def validate_source_equivalence(
    merged_pdf_path: Path,
    source_directory: Path,
    manifest: ChapterManifest,
) -> dict[str, object]:
    checked_pages = 0

    text_mismatches: list[
        dict[str, object]
    ] = []

    geometry_mismatches: list[
        dict[str, object]
    ] = []

    render_mismatches: list[
        dict[str, object]
    ] = []

    render_records: list[
        dict[str, object]
    ] = []

    exact_render_pages = 0

    with fitz.open(
        merged_pdf_path
    ) as merged_document:
        expected_page_count = (
            manifest.canonical_layout
            .canonical_page_count
        )

        if (
            merged_document.page_count
            != expected_page_count
        ):
            raise ValueError(
                "Merged PDF page count mismatch: "
                f"{merged_document.page_count} "
                f"!= {expected_page_count}"
            )

        for page_number in range(
            1,
            manifest.canonical_layout
            .leading_blank_pages
            + 1,
        ):
            text = normalize_text(
                merged_document[
                    page_number - 1
                ].get_text("text")
            )

            if text:
                raise ValueError(
                    "Leading blank page contains "
                    f"text: {page_number}"
                )

        for document in manifest.documents:
            source_path = (
                source_directory
                / document.source_filename
            )

            with fitz.open(
                source_path
            ) as source_document:
                for source_index in range(
                    source_document.page_count
                ):
                    canonical_page = (
                        document
                        .canonical_start_page
                        + source_index
                    )

                    source_page = source_document[
                        source_index
                    ]

                    merged_page = merged_document[
                        canonical_page - 1
                    ]

                    page_identity = {
                        "source_filename": (
                            document
                            .source_filename
                        ),
                        "source_page": (
                            source_index + 1
                        ),
                        "canonical_page": (
                            canonical_page
                        ),
                    }

                    checked_pages += 1

                    source_text = normalize_text(
                        source_page.get_text(
                            "text"
                        )
                    )

                    merged_text = normalize_text(
                        merged_page.get_text(
                            "text"
                        )
                    )

                    if source_text != merged_text:
                        text_mismatches.append(
                            page_identity
                        )

                    source_geometry = (
                        page_geometry_signature(
                            source_page
                        )
                    )

                    merged_geometry = (
                        page_geometry_signature(
                            merged_page
                        )
                    )

                    if (
                        source_geometry
                        != merged_geometry
                    ):
                        geometry_mismatches.append({
                            **page_identity,
                            "source_geometry": (
                                source_geometry
                            ),
                            "merged_geometry": (
                                merged_geometry
                            ),
                        })

                    metrics = (
                        render_page_difference_metrics(
                            source_page,
                            merged_page,
                        )
                    )

                    render_record = {
                        **page_identity,
                        **metrics,
                    }

                    render_records.append(
                        render_record
                    )

                    if not metrics[
                        "shape_matches"
                    ]:
                        render_mismatches.append({
                            **render_record,
                            "reason": (
                                "pixmap shape mismatch"
                            ),
                        })
                        continue

                    maximum_difference = int(
                        metrics[
                            "maximum_channel_difference"
                        ]
                    )

                    if maximum_difference == 0:
                        exact_render_pages += 1

                    mae = float(
                        metrics[
                            "mean_absolute_difference"
                        ]
                    )

                    rmse = float(
                        metrics[
                            "root_mean_square_difference"
                        ]
                    )

                    changed_above_10 = float(
                        metrics[
                            "changed_above_10_percent"
                        ]
                    )

                    if (
                        mae > RENDER_MAX_MAE
                        or rmse
                        > RENDER_MAX_RMSE
                        or changed_above_10
                        > (
                            RENDER_MAX_CHANGED_ABOVE_10_PERCENT
                        )
                    ):
                        render_mismatches.append({
                            **render_record,
                            "reason": (
                                "visual tolerance "
                                "exceeded"
                            ),
                        })

        trailing_start = (
            manifest.canonical_layout
            .canonical_page_count
            - manifest.canonical_layout
            .trailing_blank_pages
            + 1
        )

        for page_number in range(
            trailing_start,
            manifest.canonical_layout
            .canonical_page_count
            + 1,
        ):
            text = normalize_text(
                merged_document[
                    page_number - 1
                ].get_text("text")
            )

            if text:
                raise ValueError(
                    "Trailing blank page contains "
                    f"text: {page_number}"
                )

    (
        accepted_text_variances,
        unsafe_text_mismatches,
    ) = classify_text_extraction_variances(
        text_mismatches,
        geometry_mismatches,
        render_records,
    )

    if geometry_mismatches:
        raise ValueError(
            "Merged PDF geometry equivalence "
            f"failed for "
            f"{len(geometry_mismatches)} "
            "source page(s)."
        )

    if render_mismatches:
        sample = json.dumps(
            render_mismatches[:3],
            ensure_ascii=False,
        )

        raise ValueError(
            "Merged PDF visual equivalence "
            f"failed for "
            f"{len(render_mismatches)} "
            f"source page(s). Sample: {sample}"
        )

    if unsafe_text_mismatches:
        sample = json.dumps(
            unsafe_text_mismatches[:3],
            ensure_ascii=False,
        )

        raise ValueError(
            "Merged PDF text equivalence failed "
            f"for "
            f"{len(unsafe_text_mismatches)} "
            f"source page(s). Sample: {sample}"
        )

    comparable_records = [
        record
        for record in render_records
        if record["shape_matches"]
    ]

    page_mae_values = [
        float(
            record[
                "mean_absolute_difference"
            ]
        )
        for record in comparable_records
    ]

    page_rmse_values = [
        float(
            record[
                "root_mean_square_difference"
            ]
        )
        for record in comparable_records
    ]

    changed_above_10_values = [
        float(
            record[
                "changed_above_10_percent"
            ]
        )
        for record in comparable_records
    ]

    return {
        "checked_source_pages": (
            checked_pages
        ),
        "matching_source_pages": (
            checked_pages
        ),
        "mismatching_source_pages": 0,
        "matching_source_text_pages": (
            checked_pages
            - len(text_mismatches)
        ),
        "mismatching_source_text_pages": (
            len(text_mismatches)
        ),
        "accepted_text_extraction_variance_pages": (
            len(accepted_text_variances)
        ),
        "text_extraction_variances": (
            accepted_text_variances
        ),
        "matching_source_geometry_pages": (
            checked_pages
        ),
        "mismatching_source_geometry_pages": 0,
        "matching_source_render_pages": (
            checked_pages
        ),
        "mismatching_source_render_pages": 0,
        "exact_source_render_pages": (
            exact_render_pages
        ),
        "render_validation_mode": (
            "tolerance"
        ),
        "render_matrix": "1.0x1.0 RGB",
        "render_validation_thresholds": {
            "maximum_page_mae": (
                RENDER_MAX_MAE
            ),
            "maximum_page_rmse": (
                RENDER_MAX_RMSE
            ),
            "maximum_changed_above_10_percent": (
                RENDER_MAX_CHANGED_ABOVE_10_PERCENT
            ),
        },
        "observed_render_metrics": {
            "maximum_page_mae": max(
                page_mae_values,
                default=0.0,
            ),
            "mean_page_mae": (
                sum(page_mae_values)
                / len(page_mae_values)
                if page_mae_values
                else 0.0
            ),
            "maximum_page_rmse": max(
                page_rmse_values,
                default=0.0,
            ),
            "maximum_changed_above_10_percent": (
                max(
                    changed_above_10_values,
                    default=0.0,
                )
            ),
        },
        "status": "VALID",
    }


def compare_reference_pdf(
    generated_pdf_path: Path,
    reference_pdf_path: Path,
) -> dict[str, object]:
    if not reference_pdf_path.is_file():
        raise FileNotFoundError(
            "Reference PDF not found: "
            f"{reference_pdf_path}"
        )

    similarities: list[float] = []
    exact_text_pages = 0
    geometry_matching_pages = 0

    text_differences: list[
        dict[str, object]
    ] = []

    geometry_difference_sample: list[
        dict[str, object]
    ] = []

    maximum_width_difference = 0.0
    maximum_height_difference = 0.0

    with (
        fitz.open(
            generated_pdf_path
        ) as generated,
        fitz.open(
            reference_pdf_path
        ) as reference,
    ):
        if (
            generated.page_count
            != reference.page_count
        ):
            raise ValueError(
                "Generated and reference PDFs "
                "have different page counts."
            )

        for page_index in range(
            generated.page_count
        ):
            generated_page = generated[
                page_index
            ]

            reference_page = reference[
                page_index
            ]

            generated_text = normalize_text(
                generated_page.get_text(
                    "text"
                )
            )

            reference_text = normalize_text(
                reference_page.get_text(
                    "text"
                )
            )

            if generated_text == reference_text:
                similarity = 1.0
                exact_text_pages += 1
            else:
                similarity = SequenceMatcher(
                    None,
                    generated_text,
                    reference_text,
                    autojunk=False,
                ).ratio()

            similarities.append(similarity)

            if similarity < 0.999999:
                text_differences.append({
                    "page": page_index + 1,
                    "text_similarity": round(
                        similarity,
                        9,
                    ),
                })

            generated_width = (
                generated_page.rect.width
            )
            generated_height = (
                generated_page.rect.height
            )

            reference_width = (
                reference_page.rect.width
            )
            reference_height = (
                reference_page.rect.height
            )

            width_difference = abs(
                generated_width
                - reference_width
            )

            height_difference = abs(
                generated_height
                - reference_height
            )

            maximum_width_difference = max(
                maximum_width_difference,
                width_difference,
            )

            maximum_height_difference = max(
                maximum_height_difference,
                height_difference,
            )

            geometry_matches = (
                page_geometry_signature(
                    generated_page
                )
                == page_geometry_signature(
                    reference_page
                )
            )

            if geometry_matches:
                geometry_matching_pages += 1

            elif (
                len(
                    geometry_difference_sample
                )
                < 20
            ):
                geometry_difference_sample.append({
                    "page": page_index + 1,
                    "generated_size": [
                        round(
                            generated_width,
                            4,
                        ),
                        round(
                            generated_height,
                            4,
                        ),
                    ],
                    "reference_size": [
                        round(
                            reference_width,
                            4,
                        ),
                        round(
                            reference_height,
                            4,
                        ),
                    ],
                    "width_difference": round(
                        width_difference,
                        6,
                    ),
                    "height_difference": round(
                        height_difference,
                        6,
                    ),
                })

        page_count = generated.page_count

    return {
        "reference_path": str(
            reference_pdf_path
        ),
        "page_count": page_count,
        "exact_text_pages": (
            exact_text_pages
        ),
        "text_differing_page_count": (
            len(text_differences)
        ),
        "text_differing_pages": (
            text_differences
        ),
        "minimum_text_similarity": round(
            min(similarities),
            9,
        ),
        "mean_text_similarity": round(
            sum(similarities)
            / len(similarities),
            9,
        ),
        "geometry_matching_pages": (
            geometry_matching_pages
        ),
        "geometry_differing_page_count": (
            page_count
            - geometry_matching_pages
        ),
        "maximum_width_difference": round(
            maximum_width_difference,
            6,
        ),
        "maximum_height_difference": round(
            maximum_height_difference,
            6,
        ),
        "geometry_difference_sample": (
            geometry_difference_sample
        ),
        "geometry_difference_is_failure": (
            False
        ),
        "geometry_note": (
            "Reference geometry is informational. "
            "The generated PDF is validated "
            "against the original chapter PDFs."
        ),
    }


def write_json_atomically(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(
            temporary_file.name
        )

        json.dump(
            payload,
            temporary_file,
            indent=2,
            ensure_ascii=False,
        )
        temporary_file.write("\n")

    temporary_path.replace(path)


def build_chapter_textbook(
    source_directory: Path,
    output_pdf_path: Path,
    page_map_path: Path,
    report_path: Path,
    manifest: ChapterManifest,
    *,
    reference_pdf_path: Path | None = None,
    replace: bool = False,
) -> dict[str, object]:
    if not source_directory.is_dir():
        raise FileNotFoundError(
            "Chapter source directory not "
            f"found: {source_directory}"
        )

    for path in (
        output_pdf_path,
        page_map_path,
        report_path,
    ):
        if path.exists() and not replace:
            raise FileExistsError(
                "Output already exists. Use "
                f"replace=True after review: {path}"
            )

    output_pdf_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_validations = []

    for document in manifest.documents:
        source_path = (
            source_directory
            / document.source_filename
        )

        if not source_path.is_file():
            raise FileNotFoundError(
                "Manifest source PDF not found: "
                f"{source_path}"
            )

        validation = validate_pdf(
            source_path,
            expected_page_count=(
                document.source_page_count
            ),
        )

        source_validations.append({
            "order": document.order,
            "document_id": (
                document.document_id
            ),
            "source_filename": (
                document.source_filename
            ),
            **validation,
        })

    first_source_path = (
        source_directory
        / manifest.documents[0]
        .source_filename
    )

    last_source_path = (
        source_directory
        / manifest.documents[-1]
        .source_filename
    )

    with fitz.open(
        first_source_path
    ) as first_document:
        first_page_rect = (
            first_document[0].rect
        )

    with fitz.open(
        last_source_path
    ) as last_document:
        last_page_rect = (
            last_document[
                last_document.page_count - 1
            ].rect
        )

    with tempfile.NamedTemporaryFile(
        dir=output_pdf_path.parent,
        prefix=(
            f".{output_pdf_path.stem}."
        ),
        suffix=".tmp.pdf",
        delete=False,
    ) as temporary_file:
        temporary_pdf_path = Path(
            temporary_file.name
        )

    # Preserve the first source PDF as the base
    # document. Creating a new empty PDF and using
    # insert_pdf() for every source can alter complex
    # transparency, colour and font resources.
    shutil.copy2(
        first_source_path,
        temporary_pdf_path,
    )

    merged_document = fitz.open(
        temporary_pdf_path
    )

    try:
        for _ in range(
            manifest.canonical_layout
            .leading_blank_pages
        ):
            merged_document.new_page(
                pno=0,
                width=first_page_rect.width,
                height=first_page_rect.height,
            )

        first_document = (
            manifest.documents[0]
        )

        if (
            merged_document.page_count
            != first_document.canonical_end_page
        ):
            raise ValueError(
                "Merged page offset mismatch "
                f"after "
                f"{first_document.document_id}: "
                f"{merged_document.page_count} "
                f"!= "
                f"{first_document.canonical_end_page}"
            )

        for document in manifest.documents[1:]:
            source_path = (
                source_directory
                / document.source_filename
            )

            with fitz.open(
                source_path
            ) as source_document:
                merged_document.insert_pdf(
                    source_document
                )

            if (
                merged_document.page_count
                != document.canonical_end_page
            ):
                raise ValueError(
                    "Merged page offset mismatch "
                    f"after {document.document_id}: "
                    f"{merged_document.page_count} "
                    f"!= "
                    f"{document.canonical_end_page}"
                )

        for _ in range(
            manifest.canonical_layout
            .trailing_blank_pages
        ):
            merged_document.new_page(
                width=last_page_rect.width,
                height=last_page_rect.height,
            )

        expected_page_count = (
            manifest.canonical_layout
            .canonical_page_count
        )

        if (
            merged_document.page_count
            != expected_page_count
        ):
            raise ValueError(
                "Final merged page count "
                f"mismatch: "
                f"{merged_document.page_count} "
                f"!= {expected_page_count}"
            )

        merged_document.set_metadata({
            "title": manifest.title,
            "subject": (
                "Chapter-aware canonical "
                "textbook PDF"
            ),
            "keywords": (
                f"{manifest.book_id}, "
                f"{manifest.book_version}, "
                "chapter-aware"
            ),
            "creator": (
                "Enterprise Document "
                "Intelligence"
            ),
            "producer": "PyMuPDF",
        })

        merged_document.saveIncr()

    finally:
        merged_document.close()

    try:
        source_equivalence = (
            validate_source_equivalence(
                temporary_pdf_path,
                source_directory,
                manifest,
            )
        )

        output_validation = validate_pdf(
            temporary_pdf_path,
            expected_page_count=(
                manifest.canonical_layout
                .canonical_page_count
            ),
        )

    except Exception:
        temporary_pdf_path.unlink(
            missing_ok=True
        )
        raise

    reference_comparison = None

    if reference_pdf_path is not None:
        reference_comparison = (
            compare_reference_pdf(
                temporary_pdf_path,
                reference_pdf_path,
            )
        )

    page_map = build_page_map(
        manifest
    )

    report: dict[str, object] = {
        "schema_version": "1.0",
        "status": "VALID",
        "book_id": manifest.book_id,
        "book_version": (
            manifest.book_version
        ),
        "title": manifest.title,
        "source_directory": str(
            source_directory
        ),
        "output_pdf": {
            "path": str(
                output_pdf_path
            ),
            **output_validation,
        },
        "document_count": len(
            manifest.documents
        ),
        "chapter_count": (
            manifest.chapter_count
        ),
        "leading_blank_pages": (
            manifest.canonical_layout
            .leading_blank_pages
        ),
        "source_document_pages": (
            manifest.canonical_layout
            .source_document_pages
        ),
        "trailing_blank_pages": (
            manifest.canonical_layout
            .trailing_blank_pages
        ),
        "canonical_page_count": (
            manifest.canonical_layout
            .canonical_page_count
        ),
        "source_documents": (
            source_validations
        ),
        "source_equivalence": (
            source_equivalence
        ),
        "reference_comparison": (
            reference_comparison
        ),
        "page_map_path": str(
            page_map_path
        ),
        "aws_calls": 0,
    }

    temporary_pdf_path.replace(
        output_pdf_path
    )

    write_json_atomically(
        page_map_path,
        page_map,
    )

    write_json_atomically(
        report_path,
        report,
    )

    return report
