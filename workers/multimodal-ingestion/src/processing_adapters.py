from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


AdapterKey = Literal[
    "english-general",
    "mathematics",
    "hindi",
    "sanskrit",
    "urdu",
]

ReadingDirection = Literal[
    "ltr",
    "rtl",
]


class UnsupportedProcessingAdapter(
    ValueError
):
    """Raised when textbook metadata cannot be routed safely."""


@dataclass(frozen=True)
class ProcessingAdapter:
    """Runtime profile for one textbook processing category."""

    key: AdapterKey
    display_name: str
    representative_book_id: str

    expected_language: str
    expected_script: str
    reading_direction: ReadingDirection

    default_processing_profile: str
    title_fallback_label: str

    preserve_unicode: bool = True
    preserve_mathematical_notation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ADAPTERS: dict[AdapterKey, ProcessingAdapter] = {
    "english-general": ProcessingAdapter(
        key="english-general",
        display_name="English and General Subjects",
        representative_book_id="grade-9-english-kaveri",
        expected_language="English",
        expected_script="latin",
        reading_direction="ltr",
        default_processing_profile="multilingual-language",
        title_fallback_label="Unit",
    ),
    "mathematics": ProcessingAdapter(
        key="mathematics",
        display_name="Mathematics Multimodal",
        representative_book_id=(
            "grade-6-mathematics-ganita-prakash"
        ),
        expected_language="Mathematics",
        expected_script="mixed",
        reading_direction="ltr",
        default_processing_profile="mathematics-multimodal",
        title_fallback_label="Unit",
        preserve_mathematical_notation=True,
    ),
    "hindi": ProcessingAdapter(
        key="hindi",
        display_name="Hindi",
        representative_book_id="grade-6-hindi-malhar",
        expected_language="Hindi",
        expected_script="devanagari",
        reading_direction="ltr",
        default_processing_profile="multilingual-language",
        title_fallback_label="इकाई",
    ),
    "sanskrit": ProcessingAdapter(
        key="sanskrit",
        display_name="Sanskrit",
        representative_book_id="grade-6-sanskrit-deepakam",
        expected_language="Sanskrit",
        expected_script="devanagari",
        reading_direction="ltr",
        default_processing_profile="multilingual-language",
        title_fallback_label="इकाई",
    ),
    "urdu": ProcessingAdapter(
        key="urdu",
        display_name="Urdu",
        representative_book_id="grade-1-urdu-shahnai",
        expected_language="Urdu",
        expected_script="arabic",
        reading_direction="rtl",
        default_processing_profile="multilingual-language",
        title_fallback_label="اکائی",
    ),
}


def normalize_metadata_value(
    value: object,
) -> str:
    """Normalize registry/config metadata for routing."""

    return str(
        value or ""
    ).strip().casefold()


def resolve_processing_adapter(
    *,
    subject: object,
    language: object,
    script: object,
    processing_profile: object = None,
) -> ProcessingAdapter:
    """Select exactly one adapter using stable metadata."""

    normalized_subject = normalize_metadata_value(
        subject
    )
    normalized_language = normalize_metadata_value(
        language
    )
    normalized_script = normalize_metadata_value(
        script
    )
    normalized_profile = normalize_metadata_value(
        processing_profile
    )

    if (
        normalized_subject
        in {
            "mathematics",
            "math",
            "maths",
        }
        or normalized_profile
        == "mathematics-multimodal"
    ):
        return ADAPTERS["mathematics"]

    if (
        normalized_language == "urdu"
        or normalized_script == "arabic"
    ):
        return ADAPTERS["urdu"]

    if normalized_language == "sanskrit":
        return ADAPTERS["sanskrit"]

    if normalized_language == "hindi":
        return ADAPTERS["hindi"]

    if normalized_language == "english":
        return ADAPTERS["english-general"]

    raise UnsupportedProcessingAdapter(
        "No safe processing adapter matches "
        f"subject={subject!r}, "
        f"language={language!r}, "
        f"script={script!r}, "
        f"processing_profile={processing_profile!r}."
    )


def resolve_processing_adapter_for_book(
    book: Mapping[str, Any],
) -> ProcessingAdapter:
    """Resolve an adapter from one registry/config book object."""

    return resolve_processing_adapter(
        subject=book.get("subject"),
        language=book.get("language"),
        script=book.get("script"),
        processing_profile=book.get(
            "processing_profile"
        ),
    )


def adapter_runtime_metadata(
    book: Mapping[str, Any],
) -> dict[str, Any]:
    """Return notebook/state-friendly adapter information."""

    adapter = resolve_processing_adapter_for_book(
        book
    )

    return {
        "adapter_key": adapter.key,
        "adapter_name": adapter.display_name,
        "representative_book_id": (
            adapter.representative_book_id
        ),
        "expected_language": (
            adapter.expected_language
        ),
        "expected_script": adapter.expected_script,
        "reading_direction": (
            adapter.reading_direction
        ),
        "processing_profile": (
            normalize_metadata_value(
                book.get("processing_profile")
            )
            or adapter.default_processing_profile
        ),
        "title_fallback_label": (
            adapter.title_fallback_label
        ),
        "preserve_unicode": (
            adapter.preserve_unicode
        ),
        "preserve_mathematical_notation": (
            adapter.preserve_mathematical_notation
        ),
    }
