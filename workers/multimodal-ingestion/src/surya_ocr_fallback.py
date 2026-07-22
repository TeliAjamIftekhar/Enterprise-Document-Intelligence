"""Reusable Surya OCR fallback adapter.

Responsibilities:

- Render selected canonical PDF pages at a consistent DPI.
- Build the isolated Surya/vLLM runtime environment.
- Build the Surya CLI command without invoking it implicitly.
- Locate and parse Surya ``results.json`` output.
- Apply language-aware OCR quality validation.
- Write resume-safe page text, reports and approval markers.

The main textbook pipeline remains responsible for deciding which BDA pages
need OCR fallback and for merging Surya text with BDA tables, figures and
page assets.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from src.ocr_quality import (
    OCRQualityDecision,
    OCRQualityThresholds,
    clean_ocr_text,
    evaluate_ocr_text,
)


FallbackClassification = Literal["PASS", "REVIEW", "FAIL"]


@dataclass(frozen=True)
class SuryaRuntimeConfig:
    """Runtime settings validated by the Urdu OCR pilot."""

    executable: Path
    project_root: Path

    backend: str = "vllm"
    model: str = "datalab-to/surya-ocr-2"

    gpu_device: str = "0"
    gpu_type: str = "t4"
    dtype: str = "float16"

    inference_parallel: int = 1
    gpu_memory_utilization: float = 0.80

    max_model_len: int = 8192
    context_size: int = 8192
    context_per_slot: int = 8192
    maximum_full_page_tokens: int = 4096

    startup_timeout_seconds: int = 1800
    request_timeout_seconds: int = 1200

    keep_alive: bool = False
    enable_mtp: bool = False

    render_dpi: int = 300

    @property
    def huggingface_cache(self) -> Path:
        return (
            self.project_root
            / "workers/multimodal-ingestion/.cache/huggingface"
        )

    @property
    def model_cache(self) -> Path:
        return (
            self.project_root
            / "workers/multimodal-ingestion/.cache/surya-models"
        )


@dataclass(frozen=True)
class RenderedPage:
    """One rendered canonical textbook page."""

    canonical_page: int
    image_path: Path
    width: int
    height: int
    byte_size: int
    dpi: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["image_path"] = str(self.image_path)
        return payload


@dataclass(frozen=True)
class SuryaPageResult:
    """Parsed and quality-checked Surya result for one page."""

    page_key: str
    canonical_page: int

    raw_html: str
    clean_text: str
    confidence: float | None

    source_image: Path | None
    decision: OCRQualityDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_key": self.page_key,
            "canonical_page": self.canonical_page,
            "raw_html": self.raw_html,
            "clean_text": self.clean_text,
            "confidence": self.confidence,
            "source_image": (
                str(self.source_image)
                if self.source_image is not None
                else None
            ),
            "decision": self.decision.to_dict(),
        }


@dataclass(frozen=True)
class SuryaFallbackReport:
    """Book- or batch-level OCR fallback result."""

    expected_language: str
    results_json: Path

    pages: tuple[SuryaPageResult, ...]
    expected_pages: tuple[int, ...]
    missing_pages: tuple[int, ...]

    passed: int
    review: int
    failed: int

    classification: FallbackClassification
    accepted_for_pipeline: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_language": self.expected_language,
            "results_json": str(self.results_json),
            "expected_pages": list(self.expected_pages),
            "missing_pages": list(self.missing_pages),
            "passed": self.passed,
            "review": self.review,
            "failed": self.failed,
            "classification": self.classification,
            "accepted_for_pipeline": self.accepted_for_pipeline,
            "pages": [
                page.to_dict()
                for page in self.pages
            ],
        }


def prepare_runtime_directories(
    config: SuryaRuntimeConfig,
) -> None:
    """Create persistent model-cache directories."""

    config.huggingface_cache.mkdir(
        parents=True,
        exist_ok=True,
    )

    config.model_cache.mkdir(
        parents=True,
        exist_ok=True,
    )


def validate_runtime_config(
    config: SuryaRuntimeConfig,
    *,
    require_executable: bool = True,
) -> None:
    """Validate settings before starting Surya."""

    if require_executable and not config.executable.is_file():
        raise FileNotFoundError(
            f"Surya executable not found: {config.executable}"
        )

    if config.gpu_type != config.gpu_type.lower():
        raise ValueError(
            "gpu_type must be lowercase; expected values include 't4'"
        )

    if not 0 < config.gpu_memory_utilization < 1:
        raise ValueError(
            "gpu_memory_utilization must be between 0 and 1"
        )

    if config.render_dpi < 72:
        raise ValueError(
            "render_dpi must be at least 72"
        )

    positive_values = {
        "inference_parallel": config.inference_parallel,
        "max_model_len": config.max_model_len,
        "context_size": config.context_size,
        "context_per_slot": config.context_per_slot,
        "maximum_full_page_tokens": (
            config.maximum_full_page_tokens
        ),
        "startup_timeout_seconds": (
            config.startup_timeout_seconds
        ),
        "request_timeout_seconds": (
            config.request_timeout_seconds
        ),
    }

    invalid = [
        name
        for name, value in positive_values.items()
        if value <= 0
    ]

    if invalid:
        raise ValueError(
            "Runtime values must be positive: "
            + ", ".join(invalid)
        )


def build_surya_environment(
    config: SuryaRuntimeConfig,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build environment variables for the Surya CLI."""

    validate_runtime_config(
        config,
        require_executable=False,
    )

    environment = dict(
        os.environ
        if base_environment is None
        else base_environment
    )

    environment.update(
        {
            "HF_HOME": str(config.huggingface_cache),
            "DOCKER_HF_CACHE_PATH": str(
                config.huggingface_cache
            ),
            "MODEL_CACHE_DIR": str(config.model_cache),
            "SURYA_INFERENCE_BACKEND": config.backend,
            "CUDA_VISIBLE_DEVICES": config.gpu_device,
            "VLLM_GPUS": config.gpu_device,
            "VLLM_GPU_TYPE": config.gpu_type,
            "VLLM_DTYPE": config.dtype,
            "SURYA_INFERENCE_PARALLEL": str(
                config.inference_parallel
            ),
            "VLLM_GPU_MEMORY_UTILIZATION": str(
                config.gpu_memory_utilization
            ),
            "VLLM_MAX_MODEL_LEN": str(
                config.max_model_len
            ),
            "SURYA_INFERENCE_CTX_PER_SLOT": str(
                config.context_per_slot
            ),
            "SURYA_INFERENCE_CTX_SIZE": str(
                config.context_size
            ),
            "SURYA_MAX_TOKENS_FULL_PAGE": str(
                config.maximum_full_page_tokens
            ),
            "VLLM_ENABLE_MTP": (
                "true"
                if config.enable_mtp
                else "false"
            ),
            "SURYA_INFERENCE_STARTUP_TIMEOUT": str(
                config.startup_timeout_seconds
            ),
            "SURYA_INFERENCE_TIMEOUT_SECONDS": str(
                config.request_timeout_seconds
            ),
            "SURYA_INFERENCE_KEEP_ALIVE": (
                "true"
                if config.keep_alive
                else "false"
            ),
        }
    )

    return environment


def build_surya_command(
    config: SuryaRuntimeConfig,
    *,
    input_path: Path,
    output_dir: Path,
) -> list[str]:
    """Build the exact Surya OCR CLI command."""

    return [
        str(config.executable),
        str(input_path),
        "--output_dir",
        str(output_dir),
        "--images",
    ]


def normalize_page_numbers(
    page_numbers: Iterable[int],
    *,
    page_count: int,
) -> tuple[int, ...]:
    """Validate, deduplicate and order one-based page numbers."""

    normalized: set[int] = set()

    for value in page_numbers:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                "Page numbers must be integers"
            )

        if value < 1 or value > page_count:
            raise ValueError(
                f"Page {value} is outside valid range "
                f"1-{page_count}"
            )

        normalized.add(value)

    if not normalized:
        raise ValueError(
            "At least one page must be selected"
        )

    return tuple(sorted(normalized))


def render_pdf_pages(
    pdf_path: Path,
    output_dir: Path,
    *,
    page_numbers: Iterable[int] | None = None,
    dpi: int = 300,
) -> tuple[RenderedPage, ...]:
    """Render canonical PDF pages to PNG.

    Page numbers are one-based and preserved in filenames.
    """

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"Canonical PDF not found: {pdf_path}"
        )

    if dpi < 72:
        raise ValueError(
            "dpi must be at least 72"
        )

    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "PyMuPDF is required to render OCR pages"
        ) from error

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = fitz.open(pdf_path)

    try:
        page_count = document.page_count

        if page_count <= 0:
            raise ValueError(
                f"Canonical PDF has no pages: {pdf_path}"
            )

        selected_pages = (
            tuple(range(1, page_count + 1))
            if page_numbers is None
            else normalize_page_numbers(
                page_numbers,
                page_count=page_count,
            )
        )

        scale = dpi / 72
        matrix = fitz.Matrix(scale, scale)

        rendered: list[RenderedPage] = []

        for canonical_page in selected_pages:
            page = document.load_page(
                canonical_page - 1
            )

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )

            image_path = (
                output_dir
                / f"page-{canonical_page:04d}.png"
            )

            pixmap.save(image_path)

            rendered.append(
                RenderedPage(
                    canonical_page=canonical_page,
                    image_path=image_path,
                    width=pixmap.width,
                    height=pixmap.height,
                    byte_size=image_path.stat().st_size,
                    dpi=dpi,
                )
            )

        return tuple(rendered)

    finally:
        document.close()


def page_number_from_key(page_key: str) -> int:
    """Extract canonical page number from a Surya page key."""

    matches = re.findall(r"\d+", page_key)

    if not matches:
        raise ValueError(
            f"Cannot identify page number from key: {page_key}"
        )

    return int(matches[-1])


def locate_results_json(output_dir: Path) -> Path:
    """Locate the unique Surya ``results.json`` file."""

    candidates = sorted(
        output_dir.rglob("results.json"),
        key=lambda path: (
            len(path.relative_to(output_dir).parts),
            str(path),
        ),
    )

    if not candidates:
        raise FileNotFoundError(
            f"No Surya results.json found under: {output_dir}"
        )

    if len(candidates) > 1:
        shortest_depth = len(
            candidates[0].relative_to(output_dir).parts
        )

        same_depth = [
            path
            for path in candidates
            if len(
                path.relative_to(output_dir).parts
            ) == shortest_depth
        ]

        if len(same_depth) > 1:
            raise ValueError(
                "Multiple ambiguous Surya results files found: "
                + ", ".join(
                    str(path)
                    for path in same_depth
                )
            )

    return candidates[0]


def _walk_surya_payload(
    value: Any,
    *,
    html_values: list[str],
    text_values: list[str],
    confidence_values: list[float],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).casefold()

            if (
                isinstance(child, str)
                and child.strip()
            ):
                if normalized_key == "html":
                    html_values.append(
                        child.strip()
                    )

                elif (
                    normalized_key
                    in {
                        "text",
                        "ocr_text",
                        "recognized_text",
                        "markdown",
                    }
                    or normalized_key.endswith("_text")
                ):
                    text_values.append(
                        child.strip()
                    )

            if (
                isinstance(child, (int, float))
                and not isinstance(child, bool)
                and (
                    "confidence" in normalized_key
                    or normalized_key
                    in {
                        "conf",
                        "probability",
                    }
                )
            ):
                confidence_values.append(
                    float(child)
                )

            _walk_surya_payload(
                child,
                html_values=html_values,
                text_values=text_values,
                confidence_values=confidence_values,
            )

    elif isinstance(value, list):
        for child in value:
            _walk_surya_payload(
                child,
                html_values=html_values,
                text_values=text_values,
                confidence_values=confidence_values,
            )


def parse_surya_page(
    page_key: str,
    payload: Any,
    *,
    expected_language: str,
    input_dir: Path | None = None,
    thresholds: OCRQualityThresholds | None = None,
) -> SuryaPageResult:
    """Parse and validate one Surya page payload."""

    html_values: list[str] = []
    text_values: list[str] = []
    confidence_values: list[float] = []

    _walk_surya_payload(
        payload,
        html_values=html_values,
        text_values=text_values,
        confidence_values=confidence_values,
    )

    raw_html = "\n".join(html_values)

    raw_text = (
        raw_html
        if raw_html
        else "\n".join(text_values)
    )

    clean_text = clean_ocr_text(raw_text)

    confidence = (
        statistics.mean(confidence_values)
        if confidence_values
        else None
    )

    canonical_page = page_number_from_key(
        page_key
    )

    source_image: Path | None = None

    if input_dir is not None:
        candidate = (
            input_dir
            / f"page-{canonical_page:04d}.png"
        )

        if candidate.is_file():
            source_image = candidate

    decision = evaluate_ocr_text(
        clean_text,
        expected_language=expected_language,
        source="surya",
        confidence=confidence,
        thresholds=thresholds,
    )

    return SuryaPageResult(
        page_key=page_key,
        canonical_page=canonical_page,
        raw_html=raw_html,
        clean_text=clean_text,
        confidence=confidence,
        source_image=source_image,
        decision=decision,
    )


def parse_surya_results(
    results_json: Path,
    *,
    expected_language: str,
    expected_pages: Sequence[int] | None = None,
    input_dir: Path | None = None,
    thresholds: OCRQualityThresholds | None = None,
) -> SuryaFallbackReport:
    """Parse all pages and create a batch-level decision."""

    if not results_json.is_file():
        raise FileNotFoundError(
            f"Surya results not found: {results_json}"
        )

    payload = json.loads(
        results_json.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Surya results root must be a JSON object"
        )

    pages = tuple(
        sorted(
            (
                parse_surya_page(
                    page_key,
                    page_payload,
                    expected_language=(
                        expected_language
                    ),
                    input_dir=input_dir,
                    thresholds=thresholds,
                )
                for page_key, page_payload
                in payload.items()
            ),
            key=lambda page: page.canonical_page,
        )
    )

    actual_pages = {
        page.canonical_page
        for page in pages
    }

    normalized_expected_pages = (
        tuple(sorted(set(expected_pages)))
        if expected_pages is not None
        else tuple(sorted(actual_pages))
    )

    missing_pages = tuple(
        page
        for page in normalized_expected_pages
        if page not in actual_pages
    )

    passed = sum(
        page.decision.classification == "PASS"
        for page in pages
    )

    review = sum(
        page.decision.classification == "REVIEW"
        for page in pages
    )

    failed = sum(
        page.decision.classification == "FAIL"
        for page in pages
    )

    if missing_pages or failed:
        classification: FallbackClassification = "FAIL"

    elif review or not pages:
        classification = "REVIEW"

    else:
        classification = "PASS"

    return SuryaFallbackReport(
        expected_language=expected_language,
        results_json=results_json,
        pages=pages,
        expected_pages=normalized_expected_pages,
        missing_pages=missing_pages,
        passed=passed,
        review=review,
        failed=failed,
        classification=classification,
        accepted_for_pipeline=(
            classification == "PASS"
        ),
    )


def write_fallback_artifacts(
    report: SuryaFallbackReport,
    output_dir: Path,
) -> dict[str, Path]:
    """Write clean page text, JSON report and status marker."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pages_dir = output_dir / "pages"

    pages_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for page in report.pages:
        text_path = (
            pages_dir
            / f"page-{page.canonical_page:04d}.txt"
        )

        text_path.write_text(
            page.clean_text + "\n",
            encoding="utf-8",
        )

    report_path = (
        output_dir
        / "surya-fallback-report.json"
    )

    report_path.write_text(
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    passed_marker = (
        output_dir
        / "SURYA_OCR_FALLBACK_VERIFIED"
    )

    review_marker = (
        output_dir
        / "SURYA_OCR_FALLBACK_REVIEW_REQUIRED"
    )

    failed_marker = (
        output_dir
        / "SURYA_OCR_FALLBACK_FAILED"
    )

    for marker in (
        passed_marker,
        review_marker,
        failed_marker,
    ):
        if marker.exists():
            marker.unlink()

    selected_marker: Path

    if report.classification == "PASS":
        selected_marker = passed_marker

    elif report.classification == "REVIEW":
        selected_marker = review_marker

    else:
        selected_marker = failed_marker

    selected_marker.write_text(
        report.classification + "\n",
        encoding="utf-8",
    )

    return {
        "report": report_path,
        "marker": selected_marker,
        "pages_dir": pages_dir,
    }


def load_approval_record(
    approval_path: Path,
    *,
    require_approved: bool = True,
) -> dict[str, Any]:
    """Load the manually approved OCR-engine pilot record."""

    if not approval_path.is_file():
        raise FileNotFoundError(
            f"OCR approval record not found: {approval_path}"
        )

    payload = json.loads(
        approval_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "OCR approval record must be a JSON object"
        )

    if (
        require_approved
        and not payload.get(
            "approved_for_pipeline_integration",
            False,
        )
    ):
        raise ValueError(
            "OCR engine has not been approved "
            "for pipeline integration"
        )

    return payload
# SURYA_APPROVAL_SCOPE_GUARD
def validate_approval_scope(
    approval: dict[str, Any],
    *,
    book_id: str,
    version: str,
    selected_pages: tuple[int, ...],
) -> None:
    """Ensure an OCR approval authorizes this exact run."""

    approved_book_id = str(
        approval.get("book_id", "")
    ).strip()

    approved_version = str(
        approval.get("version", "")
    ).strip()

    if approved_book_id != book_id:
        raise ValueError(
            "OCR approval book mismatch: "
            f"approved={approved_book_id!r}, "
            f"requested={book_id!r}"
        )

    if approved_version != version:
        raise ValueError(
            "OCR approval version mismatch: "
            f"approved={approved_version!r}, "
            f"requested={version!r}"
        )

    requested_pages = {
        int(page)
        for page in selected_pages
    }

    if not requested_pages:
        raise ValueError(
            "At least one OCR page is required"
        )

    if approval.get(
        "full_book_run_authorized",
        False,
    ):
        return

    raw_pilot_pages = (
        approval.get("representative_pages")
        or approval.get("pilot_pages")
        or []
    )

    if not isinstance(
        raw_pilot_pages,
        (list, tuple),
    ):
        raise ValueError(
            "OCR pilot pages must be a list"
        )

    pilot_pages: set[int] = set()

    for value in raw_pilot_pages:
        try:
            page = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid OCR pilot page: {value!r}"
            ) from error

        if page <= 0:
            raise ValueError(
                f"Invalid OCR pilot page: {page}"
            )

        pilot_pages.add(page)

    if not pilot_pages:
        raise ValueError(
            "Full-book OCR is not authorized and "
            "no representative pilot pages exist"
        )

    unauthorized_pages = sorted(
        requested_pages - pilot_pages
    )

    if unauthorized_pages:
        raise ValueError(
            "OCR pages are outside the approved "
            "pilot scope: "
            f"{unauthorized_pages}. "
            "Full-book OCR remains unauthorized."
        )


