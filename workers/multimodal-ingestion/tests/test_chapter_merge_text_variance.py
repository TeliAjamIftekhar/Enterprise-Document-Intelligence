from src.chapter_merge import (
    classify_text_extraction_variances,
)


def identity():
    return {
        "source_filename": "unit.pdf",
        "source_page": 4,
        "canonical_page": 12,
    }


def render_record(
    *,
    maximum_difference: int,
):
    return {
        **identity(),
        "shape_matches": True,
        "maximum_channel_difference": (
            maximum_difference
        ),
    }


def test_accepts_pixel_identical_text_variance():
    accepted, unsafe = (
        classify_text_extraction_variances(
            [identity()],
            [],
            [
                render_record(
                    maximum_difference=0
                )
            ],
        )
    )

    assert len(accepted) == 1
    assert unsafe == []
    assert accepted[0]["reason"] == (
        "text extraction variance "
        "with exact visual equivalence"
    )


def test_rejects_text_variance_with_visual_change():
    accepted, unsafe = (
        classify_text_extraction_variances(
            [identity()],
            [],
            [
                render_record(
                    maximum_difference=1
                )
            ],
        )
    )

    assert accepted == []
    assert len(unsafe) == 1
    assert unsafe[0]["exact_render"] is False


def test_rejects_text_variance_with_geometry_change():
    accepted, unsafe = (
        classify_text_extraction_variances(
            [identity()],
            [{
                **identity(),
                "source_geometry": {},
                "merged_geometry": {},
            }],
            [
                render_record(
                    maximum_difference=0
                )
            ],
        )
    )

    assert accepted == []
    assert len(unsafe) == 1
    assert unsafe[0]["geometry_matches"] is False
