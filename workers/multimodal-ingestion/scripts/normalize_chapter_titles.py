from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import fitz


DEVANAGARI_WORD_PATTERN = re.compile(
    r"[\u0900-\u097F]+"
)

TITLE_SEGMENT_PATTERN = re.compile(
    r"[\u0900-\u097F]+|"
    r"[^\u0900-\u097F]+"
)

ZERO_WIDTH_CHARACTERS = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
    "\u00ad",
}

VOWEL_SIGNS = set(
    "ािीुूृॄॅेैॉोौ"
)


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return value


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def normalize_spaces(
    value: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def deduplicate_combining_marks(
    value: str,
) -> str:
    result: list[str] = []
    seen_marks: set[str] = set()

    for character in value:
        category = unicodedata.category(
            character
        )

        if category.startswith("M"):
            if character in seen_marks:
                continue

            seen_marks.add(character)

        else:
            seen_marks = set()

        result.append(character)

    return "".join(result)


def clean_unicode_text(
    value: str,
) -> str:
    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    for character in ZERO_WIDTH_CHARACTERS:
        value = value.replace(
            character,
            "",
        )

    value = value.replace(
        "\ufffd",
        "",
    )

    # Remove an erroneous space before a
    # Devanagari combining mark.
    value = re.sub(
        r"(?<=[\u0900-\u097F])"
        r"\s+"
        r"(?=[\u093A-\u094D"
        r"\u0951-\u0957])",
        "",
        value,
    )

    # OCR may incorrectly place whitespace after
    # a virama, for example "षष् ठः". Joining it
    # restores the intended conjunct "षष्ठः".
    value = re.sub(
        r"(?<=्)"
        r"\s+"
        r"(?=[\u0900-\u097F])",
        "",
        value,
    )

    value = deduplicate_combining_marks(
        value
    )

    value = re.sub(
        r"\s*[-–—]\s*",
        "-",
        value,
    )

    return normalize_spaces(value)


def first_letter(
    value: str,
) -> str | None:
    for character in value:
        if unicodedata.category(
            character
        ).startswith("L"):
            return character

    return None


def word_anomaly_score(
    value: str,
) -> int:
    word = clean_unicode_text(value)

    score = 0

    if not word:
        return 100

    if "\ufffd" in value:
        score += 20

    # Detect accidental adjacent duplicate
    # letters, such as खततरे or किततनी.
    #
    # Sanskrit can validly start with a repeated
    # consonant followed by virama, as in षष्ठः.
    for index, (
        previous,
        current,
    ) in enumerate(
        zip(
            word,
            word[1:],
        )
    ):
        if not (
            previous == current
            and unicodedata.category(
                current
            ).startswith("L")
        ):
            continue

        valid_initial_conjunct = (
            index == 0
            and len(word) > 2
            and word[2] == "्"
        )

        if not valid_initial_conjunct:
            score += 3

    # Visarga is normally word-final. An internal
    # visarga usually indicates an OCR duplicated
    # tail, for example पाठःठः.
    if "ः" in word[:-1]:
        score += 3

    # Repeated syllables are valid in many
    # languages, including Hindi words such as
    # "मामा" and "दादा-दादी". Do not treat
    # repeated multi-character fragments as an
    # automatic spelling anomaly.

    vowel_sign_count = 0
    previous_character = ""

    for character in word:
        category = unicodedata.category(
            character
        )

        if category.startswith("L"):
            vowel_sign_count = 0

        elif category.startswith("M"):
            if character in VOWEL_SIGNS:
                vowel_sign_count += 1

                if vowel_sign_count > 1:
                    score += 3

            if previous_character == "्":
                score += 3

        else:
            vowel_sign_count = 0

        previous_character = character

    return score


def title_anomaly_score(
    value: str,
) -> int:
    return sum(
        word_anomaly_score(word)
        for word in DEVANAGARI_WORD_PATTERN.findall(
            clean_unicode_text(value)
        )
    )


def normalized_match_key(
    value: str,
) -> str:
    return "".join(
        character
        for character in clean_unicode_text(
            value
        ).casefold()
        if character.isalnum()
    )


def title_similarity(
    first: str,
    second: str,
) -> float:
    first_key = normalized_match_key(
        first
    )

    second_key = normalized_match_key(
        second
    )

    if not first_key or not second_key:
        return 0.0

    return SequenceMatcher(
        None,
        first_key,
        second_key,
    ).ratio()


def build_corpus_frequency(
    extraction_report: dict[str, Any],
) -> Counter[str]:
    target_directory = Path(
        extraction_report[
            "target_directory"
        ]
    )

    frequencies: Counter[str] = Counter()

    for document in extraction_report[
        "documents"
    ]:
        pdf_path = (
            target_directory
            / document["source_filename"]
        )

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"Extracted PDF missing: "
                f"{pdf_path}"
            )

        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text = clean_unicode_text(
                    page.get_text("text")
                )

                for word in (
                    DEVANAGARI_WORD_PATTERN
                    .findall(text)
                ):
                    cleaned_word = (
                        clean_unicode_text(
                            word
                        )
                    )

                    if len(cleaned_word) >= 2:
                        frequencies[
                            cleaned_word
                        ] += 1

    return frequencies


def build_candidate_index(
    frequencies: Counter[str],
) -> dict[str, list[str]]:
    index: dict[
        str,
        list[str]
    ] = {}

    for word in frequencies:
        letter = first_letter(word)

        if letter is None:
            continue

        index.setdefault(
            letter,
            [],
        ).append(word)

    return index


def correct_word_from_corpus(
    word: str,
    frequencies: Counter[str],
    candidate_index: dict[
        str,
        list[str]
    ],
) -> str:
    original = clean_unicode_text(
        word
    )

    letter = first_letter(original)

    if (
        letter is None
        or len(original) < 4
    ):
        return original

    original_frequency = frequencies[
        original
    ]

    original_anomaly = (
        word_anomaly_score(original)
    )

    visarga_index = original.find("ः")

    if (
        0
        <= visarga_index
        < len(original) - 1
    ):
        base = original[:visarga_index]
        valid_prefix = original[
            :visarga_index + 1
        ]
        duplicated_tail = original[
            visarga_index + 1:
        ]

        if (
            base
            and duplicated_tail
            and duplicated_tail[0]
            == base[-1]
            and frequencies[valid_prefix]
            >= max(
                1,
                original_frequency,
            )
        ):
            return valid_prefix

    best_word = original
    best_rank = (
        original_anomaly,
        -original_frequency,
        -1.0,
    )

    for candidate in candidate_index.get(
        letter,
        [],
    ):
        if candidate == original:
            continue

        if abs(
            len(candidate) - len(original)
        ) > 2:
            continue

        similarity = SequenceMatcher(
            None,
            original,
            candidate,
        ).ratio()

        if similarity < 0.84:
            continue

        candidate_frequency = (
            frequencies[candidate]
        )

        candidate_anomaly = (
            word_anomaly_score(candidate)
        )

        anomaly_improved = (
            candidate_anomaly
            < original_anomaly
        )

        frequency_supports_change = (
            candidate_frequency
            >= max(
                1,
                original_frequency,
            )
        )

        if not (
            anomaly_improved
            and frequency_supports_change
        ):
            continue

        rank = (
            candidate_anomaly,
            -candidate_frequency,
            -similarity,
        )

        if rank < best_rank:
            best_word = candidate
            best_rank = rank

    return best_word


def correct_title_from_corpus(
    title: str,
    frequencies: Counter[str],
    candidate_index: dict[
        str,
        list[str]
    ],
) -> str:
    segments = TITLE_SEGMENT_PATTERN.findall(
        clean_unicode_text(title)
    )

    corrected: list[str] = []

    for segment in segments:
        if DEVANAGARI_WORD_PATTERN.fullmatch(
            segment
        ):
            corrected.append(
                correct_word_from_corpus(
                    segment,
                    frequencies,
                    candidate_index,
                )
            )

        else:
            corrected.append(segment)

    return clean_unicode_text(
        "".join(corrected)
    )


def mark_richness(
    value: str,
) -> int:
    text = clean_unicode_text(value)

    return (
        text.count("्")
        + text.count("़")
    )


def word_frequency(
    word: str,
    frequencies: Counter[str],
) -> int:
    return frequencies[
        clean_unicode_text(word)
    ]


def choose_word(
    toc_word: str,
    internal_word: str,
    frequencies: Counter[str],
) -> str:
    toc_word = clean_unicode_text(
        toc_word
    )

    internal_word = clean_unicode_text(
        internal_word
    )

    if toc_word == internal_word:
        return toc_word

    similarity = SequenceMatcher(
        None,
        toc_word,
        internal_word,
    ).ratio()

    if similarity < 0.72:
        return toc_word

    toc_anomaly = word_anomaly_score(
        toc_word
    )

    internal_anomaly = (
        word_anomaly_score(
            internal_word
        )
    )

    if toc_anomaly < internal_anomaly:
        return toc_word

    if internal_anomaly < toc_anomaly:
        return internal_word

    toc_frequency = word_frequency(
        toc_word,
        frequencies,
    )

    internal_frequency = word_frequency(
        internal_word,
        frequencies,
    )

    if (
        internal_frequency
        >= toc_frequency + 2
    ):
        return internal_word

    if (
        toc_frequency
        >= internal_frequency + 2
    ):
        return toc_word

    # Prefer the candidate preserving a valid
    # conjunct or nukta when both forms are
    # otherwise very similar.
    if similarity >= 0.86:
        toc_marks = mark_richness(
            toc_word
        )

        internal_marks = mark_richness(
            internal_word
        )

        if internal_marks > toc_marks:
            return internal_word

        if toc_marks > internal_marks:
            return toc_word

    return toc_word


def reconcile_titles(
    toc_title: str,
    internal_title: str | None,
    frequencies: Counter[str],
    candidate_index: dict[
        str,
        list[str]
    ],
) -> tuple[str, str]:
    toc_clean = correct_title_from_corpus(
        toc_title,
        frequencies,
        candidate_index,
    )

    if not internal_title:
        return (
            toc_clean,
            "toc-corpus-normalized",
        )

    internal_clean = (
        correct_title_from_corpus(
            internal_title,
            frequencies,
            candidate_index,
        )
    )

    if (
        normalized_match_key(toc_clean)
        == normalized_match_key(
            internal_clean
        )
    ):
        if (
            "-" in internal_clean
            and "-" not in toc_clean
        ):
            return (
                internal_clean,
                "internal-layout-punctuation",
            )

        if (
            mark_richness(internal_clean)
            > mark_richness(toc_clean)
        ):
            return (
                internal_clean,
                "internal-layout-orthography",
            )

        return (
            toc_clean,
            "toc-corpus-normalized",
        )

    toc_words = (
        DEVANAGARI_WORD_PATTERN.findall(
            toc_clean
        )
    )

    internal_words = (
        DEVANAGARI_WORD_PATTERN.findall(
            internal_clean
        )
    )

    if (
        len(toc_words)
        == len(internal_words)
        and toc_words
        and all(
            SequenceMatcher(
                None,
                first,
                second,
            ).ratio()
            >= 0.65
            for first, second in zip(
                toc_words,
                internal_words,
            )
        )
    ):
        selected_words = [
            choose_word(
                toc_word,
                internal_word,
                frequencies,
            )
            for toc_word, internal_word
            in zip(
                toc_words,
                internal_words,
            )
        ]

        selected_index = 0
        output_segments: list[str] = []

        for segment in (
            TITLE_SEGMENT_PATTERN.findall(
                toc_clean
            )
        ):
            if (
                DEVANAGARI_WORD_PATTERN
                .fullmatch(segment)
            ):
                output_segments.append(
                    selected_words[
                        selected_index
                    ]
                )

                selected_index += 1

            else:
                output_segments.append(
                    segment
                )

        merged = correct_title_from_corpus(
            "".join(output_segments),
            frequencies,
            candidate_index,
        )

        return (
            merged,
            "toc-internal-corpus-ensemble",
        )

    return (
        toc_clean,
        "toc-corpus-normalized",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize resolved chapter titles "
            "using textbook corpus frequency and "
            "internal layout evidence."
        )
    )

    parser.add_argument(
        "--chapter-structure",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--extraction-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    structure = load_json_object(
        args.chapter_structure
    )

    extraction = load_json_object(
        args.extraction_report
    )

    if structure.get("status") != "READY":
        raise ValueError(
            "Input chapter structure must "
            "have status READY."
        )

    if (
        structure.get("book_id")
        != extraction.get("book_id")
    ):
        raise ValueError(
            "Book ID mismatch between chapter "
            "structure and extraction report."
        )

    if (
        structure.get("book_version")
        != extraction.get("book_version")
    ):
        raise ValueError(
            "Book version mismatch."
        )

    frequencies = build_corpus_frequency(
        extraction
    )

    candidate_index = (
        build_candidate_index(
            frequencies
        )
    )

    changes: list[
        dict[str, Any]
    ] = []

    review_items: list[
        dict[str, Any]
    ] = []

    for chapter in structure["chapters"]:
        original_title = str(
            chapter["chapter_title"]
        )

        internal_title = chapter.get(
            "matched_internal_title"
        )

        (
            normalized_title,
            normalization_source,
        ) = reconcile_titles(
            original_title,
            (
                str(internal_title)
                if internal_title
                else None
            ),
            frequencies,
            candidate_index,
        )

        chapter[
            "raw_chapter_title"
        ] = original_title

        chapter[
            "chapter_title"
        ] = normalized_title

        chapter[
            "title_normalization_source"
        ] = normalization_source

        chapter[
            "title_anomaly_score"
        ] = title_anomaly_score(
            normalized_title
        )

        if normalized_title != original_title:
            changes.append({
                "chapter_id": (
                    chapter["chapter_id"]
                ),
                "before": original_title,
                "after": normalized_title,
                "source": (
                    normalization_source
                ),
            })

        if chapter[
            "title_anomaly_score"
        ] > 0:
            review_items.append({
                "chapter_id": (
                    chapter["chapter_id"]
                ),
                "chapter_title": (
                    normalized_title
                ),
                "anomaly_score": (
                    chapter[
                        "title_anomaly_score"
                    ]
                ),
            })

    structure["status"] = (
        "READY"
        if not review_items
        else "NEEDS_TITLE_REVIEW"
    )

    structure[
        "title_normalization"
    ] = {
        "method": (
            "corpus-frequency-and-"
            "layout-ensemble"
        ),
        "corpus_vocabulary_size": (
            len(frequencies)
        ),
        "changed_title_count": (
            len(changes)
        ),
        "review_item_count": (
            len(review_items)
        ),
        "changes": changes,
        "review_items": review_items,
        "safety": {
            "aws_calls": 0,
            "s3_writes": 0,
            "bedrock_calls": 0,
            "opensearch_calls": 0,
        },
    }

    atomic_write_json(
        args.output,
        structure,
    )

    print("=" * 100)
    print("CHAPTER TITLE NORMALIZATION")
    print("=" * 100)
    print(
        "Book ID:             ",
        structure["book_id"],
    )
    print(
        "Chapters/readings:   ",
        len(structure["chapters"]),
    )
    print(
        "Corpus vocabulary:   ",
        len(frequencies),
    )
    print(
        "Titles changed:      ",
        len(changes),
    )
    print(
        "Titles needing review:",
        len(review_items),
    )
    print(
        "Status:              ",
        structure["status"],
    )

    print()
    print("FINAL CHAPTER TITLES")
    print("-" * 100)

    for chapter in structure["chapters"]:
        number = chapter.get(
            "lesson_number"
        )

        number_text = (
            str(number)
            if number is not None
            else "*"
        )

        print(
            f"{number_text:>2} | "
            f"{chapter['chapter_title']} | "
            f"{chapter['source_filename']} | "
            f"canonical="
            f"{chapter['canonical_start_page']}-"
            f"{chapter['canonical_end_page']}"
        )

    if changes:
        print()
        print("AUTOMATIC CORRECTIONS")
        print("-" * 100)

        for change in changes:
            print(
                f"{change['chapter_id']} | "
                f"{change['before']} "
                f"→ {change['after']} | "
                f"{change['source']}"
            )

    if review_items:
        print()
        print("TITLE REVIEW REQUIRED")
        print("-" * 100)

        for item in review_items:
            print(
                f"{item['chapter_id']} | "
                f"{item['chapter_title']} | "
                f"score="
                f"{item['anomaly_score']}"
            )

    print()
    print("Output:", args.output)
    print("AWS calls: 0")

    return (
        0
        if structure["status"] == "READY"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())