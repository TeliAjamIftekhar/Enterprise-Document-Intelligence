from normalize_chapter_titles import (
    title_anomaly_score,
    word_anomaly_score,
)


def test_valid_repeated_hindi_syllables() -> None:
    assert title_anomaly_score(
        "चंदा मामा दूर के"
    ) == 0

    assert title_anomaly_score(
        "दादा-दादी"
    ) == 0


def test_detects_adjacent_duplicate_letter() -> None:
    assert word_anomaly_score(
        "खततरे"
    ) > 0

    assert word_anomaly_score(
        "किततनी"
    ) > 0


def test_normal_hindi_titles_have_no_anomaly() -> None:
    titles = [
        "मीना का परिवार",
        "वाह, मेरे घोड़े!",
        "झूलम-झूली",
        "कितनी प्यारी है ये दुनिया",
        "चाँद का बच्चा",
    ]

    for title in titles:
        assert title_anomaly_score(title) == 0


def test_normalizes_sanskrit_virama_spacing() -> None:
    from normalize_chapter_titles import (
        clean_unicode_text,
    )

    assert clean_unicode_text(
        "षष् ठः पाठः"
    ) == "षष्ठः पाठः"


def test_valid_sanskrit_titles_have_no_anomaly() -> None:
    titles = [
        "प्रथमः पाठः",
        "द्वितीयः पाठः",
        "तृतीयः पाठः",
        "चतुर्थः पाठः",
        "पञ्चमः पाठः",
        "षष्ठः पाठः",
        "एकादशः पाठः",
        "द्वादशः पाठः",
        "षोडशः पाठः",
    ]

    for title in titles:
        assert title_anomaly_score(title) == 0


def test_preserves_valid_sanskrit_ordinal() -> None:
    assert word_anomaly_score(
        "षष्ठः"
    ) == 0


def test_detects_duplicated_visarga_tail() -> None:
    assert word_anomaly_score(
        "पाठःठः"
    ) > 0


def test_safe_sanskrit_corpus_correction() -> None:
    from collections import Counter

    from normalize_chapter_titles import (
        build_candidate_index,
        correct_word_from_corpus,
    )

    frequencies = Counter({
        "प्रथमः": 2,
        "प्रथम": 20,
        "द्विततीयः": 1,
        "द्वितीयः": 4,
        "पाठःठः": 1,
        "पाठः": 20,
    })

    candidate_index = (
        build_candidate_index(
            frequencies
        )
    )

    assert correct_word_from_corpus(
        "प्रथमः",
        frequencies,
        candidate_index,
    ) == "प्रथमः"

    assert correct_word_from_corpus(
        "द्विततीयः",
        frequencies,
        candidate_index,
    ) == "द्वितीयः"

    assert correct_word_from_corpus(
        "पाठःठः",
        frequencies,
        candidate_index,
    ) == "पाठः"
