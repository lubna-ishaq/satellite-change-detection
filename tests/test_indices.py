import numpy as np
import pytest

from ndvi_core import (
    DEFAULT_INDEX,
    INDICES,
    calculate_ndvi,
    get_index,
    normalized_difference,
)


def test_every_index_is_a_normalised_difference_of_two_distinct_bands():
    for name, spec in INDICES.items():
        assert spec.name == name
        assert spec.band_a != spec.band_b, name
        assert spec.band_a.startswith("B") and spec.band_b.startswith("B"), name
        assert spec.display_min < spec.display_max, name
        assert spec.increase_label and spec.decrease_label, name


@pytest.mark.parametrize(
    "name, band_a, band_b",
    [
        ("NDVI", "B08", "B04"),  # NIR, Red
        ("NDWI", "B03", "B08"),  # Green, NIR — McFeeters, water positive
        ("NBR", "B08", "B12"),  # NIR, SWIR2
    ],
)
def test_band_pairs_match_the_published_definitions(name, band_a, band_b):
    spec = get_index(name)
    assert spec.bands == (band_a, band_b)


def test_index_lookup_is_case_insensitive():
    assert get_index("ndvi") is get_index("NDVI")


def test_unknown_index_names_the_alternatives():
    with pytest.raises(KeyError, match="NDVI"):
        get_index("NDXX")


def test_default_index_exists():
    assert DEFAULT_INDEX in INDICES


def test_normalized_difference_matches_the_formula():
    a = np.array([[3.0, 1.0]])
    b = np.array([[1.0, 1.0]])
    np.testing.assert_almost_equal(normalized_difference(a, b), np.array([[0.5, 0.0]]))


def test_normalized_difference_is_antisymmetric():
    a = np.array([[0.35, 0.10]])
    b = np.array([[0.08, 0.40]])
    np.testing.assert_almost_equal(
        normalized_difference(a, b), -normalized_difference(b, a)
    )


def test_calculate_ndvi_puts_the_bands_in_vegetation_order():
    """NDVI must be (NIR - Red)/(NIR + Red), not the reverse."""
    red, nir = np.array([[0.10]]), np.array([[0.40]])
    assert calculate_ndvi(red, nir)[0, 0] == pytest.approx(0.6)
    assert calculate_ndvi(red, nir) == pytest.approx(normalized_difference(nir, red))


def test_ndwi_is_positive_over_water():
    """Water: high green, very low NIR."""
    spec = get_index("NDWI")
    green, nir = np.array([[0.08]]), np.array([[0.02]])
    value = normalized_difference(green, nir)
    assert value[0, 0] > 0
    assert spec.bands == ("B03", "B08")


def test_nbr_falls_after_a_burn():
    """Fire lowers NIR and raises SWIR2, so NBR must drop."""
    unburnt = normalized_difference(np.array([[0.30]]), np.array([[0.10]]))
    burnt = normalized_difference(np.array([[0.12]]), np.array([[0.25]]))
    assert burnt[0, 0] < unburnt[0, 0]


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="share a grid"):
        normalized_difference(np.zeros((2, 2)), np.zeros((3, 3)))
