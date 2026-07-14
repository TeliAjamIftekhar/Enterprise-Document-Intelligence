import pytest

from src.page_renderer import build_page_filename


@pytest.mark.parametrize(
    ("page_number", "expected"),
    [
        (1, "page-0001"),
        (9, "page-0009"),
        (27, "page-0027"),
        (300, "page-0300"),
    ],
)
def test_build_page_filename(page_number, expected):
    assert build_page_filename(page_number) == expected


def test_build_page_filename_rejects_zero():
    with pytest.raises(ValueError):
        build_page_filename(0)
