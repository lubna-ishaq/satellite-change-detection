import datetime as dt

import numpy as np
import pytest

from ndvi_core import (
    BOA_ADD_OFFSET,
    BOA_QUANTIFICATION_VALUE,
    DEFAULT_VALID_SCL_CLASSES,
    calculate_ndvi,
    calculate_ndvi_delta,
    change_statistics,
    median_composite,
    needs_boa_offset,
    scl_valid_mask,
    to_reflectance,
)

# --- NDVI ----------------------------------------------------------------


def test_calculate_ndvi_basic():
    red = np.array([[100, 200]])
    nir = np.array([[300, 200]])
    # (300-100)/(300+100) = 0.5 ; (200-200)/(200+200) = 0.0
    np.testing.assert_almost_equal(calculate_ndvi(red, nir), np.array([[0.5, 0.0]]))


def test_zero_denominator_is_nan_not_zero():
    """0 is a real NDVI value (bare soil), so it must not double as no-data."""
    result = calculate_ndvi(np.array([[0, 100]]), np.array([[0, 300]]))
    assert np.isnan(result[0, 0])
    assert result[0, 1] == pytest.approx(0.5)


def test_nan_inputs_propagate():
    result = calculate_ndvi(np.array([[np.nan]]), np.array([[0.3]]))
    assert np.isnan(result[0, 0])


def test_ndvi_is_clipped_to_physical_range():
    result = calculate_ndvi(np.array([[-0.5]]), np.array([[0.1]]))
    assert -1.0 <= result[0, 0] <= 1.0


def test_mismatched_band_shapes_raise():
    with pytest.raises(ValueError, match="share a grid"):
        calculate_ndvi(np.zeros((2, 2)), np.zeros((3, 3)))


def test_mask_is_applied():
    red = np.array([[100, 100]])
    nir = np.array([[300, 300]])
    mask = np.array([[True, False]])
    result = calculate_ndvi(red, nir, valid_mask=mask)
    assert result[0, 0] == pytest.approx(0.5)
    assert np.isnan(result[0, 1])


def test_mismatched_mask_shape_raises():
    with pytest.raises(ValueError, match="Mask shape"):
        calculate_ndvi(np.zeros((2, 2)), np.zeros((2, 2)), valid_mask=np.zeros((3, 3)))


# --- Radiometric harmonisation -------------------------------------------


@pytest.mark.parametrize(
    "acquired, expected",
    [
        (dt.date(2021, 8, 13), False),
        (dt.date(2022, 1, 24), False),
        (dt.date(2022, 1, 25), True),  # Baseline 04.00 cutover
        (dt.date(2024, 8, 29), True),
        ("2024-08-29", True),
        ("2021-08-13T10:12:00Z", False),
    ],
)
def test_needs_boa_offset(acquired, expected):
    assert needs_boa_offset(acquired) is expected


def test_to_reflectance_applies_offset():
    dn = np.array([[3000.0]])
    without = to_reflectance(dn, apply_offset=False)
    with_offset = to_reflectance(dn, apply_offset=True)
    assert without[0, 0] == pytest.approx(0.3)
    assert with_offset[0, 0] == pytest.approx(
        (3000.0 + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE
    )


def test_to_reflectance_treats_zero_as_nodata():
    assert np.isnan(to_reflectance(np.array([[0]]), apply_offset=True)[0, 0])


def test_missing_offset_biases_ndvi():
    """Regression guard for the bug this correction exists to prevent.

    The same ground target, imaged before and after the Baseline 04.00
    cutover, must yield the same NDVI once harmonised — and a visibly
    different one if the offset is skipped.
    """
    red_old, nir_old = np.array([[1200.0]]), np.array([[3500.0]])
    red_new, nir_new = red_old + 1000.0, nir_old + 1000.0  # same target, new baseline

    harmonised = calculate_ndvi(
        to_reflectance(red_new, apply_offset=True),
        to_reflectance(nir_new, apply_offset=True),
    )
    reference = calculate_ndvi(
        to_reflectance(red_old, apply_offset=False),
        to_reflectance(nir_old, apply_offset=False),
    )
    naive = calculate_ndvi(
        to_reflectance(red_new, apply_offset=False),
        to_reflectance(nir_new, apply_offset=False),
    )

    np.testing.assert_almost_equal(harmonised, reference)
    assert abs(naive[0, 0] - reference[0, 0]) > 0.1


# --- SCL masking ---------------------------------------------------------


def test_scl_valid_mask_rejects_cloud_shadow_and_snow():
    scl = np.array([[0, 3, 4, 5, 6, 8, 9, 10, 11]])
    mask = scl_valid_mask(scl)
    np.testing.assert_array_equal(
        mask, np.array([[False, False, True, True, True, False, False, False, False]])
    )


def test_water_is_kept_by_default():
    assert 6 in DEFAULT_VALID_SCL_CLASSES


# --- Delta ---------------------------------------------------------------


def test_calculate_ndvi_delta():
    baseline = np.array([[0.2, 0.5]])
    comparison = np.array([[0.6, 0.3]])
    np.testing.assert_almost_equal(
        calculate_ndvi_delta(baseline, comparison), np.array([[0.4, -0.2]])
    )


def test_delta_refuses_to_broadcast_mismatched_grids():
    """Silent broadcasting would compare unrelated ground locations."""
    with pytest.raises(ValueError, match="same grid"):
        calculate_ndvi_delta(np.zeros((4, 4)), np.zeros((4, 1)))


# --- Compositing ---------------------------------------------------------


def test_median_composite_ignores_nan():
    stack = np.array([[[0.4]], [[np.nan]], [[0.6]]])
    np.testing.assert_almost_equal(median_composite(stack), np.array([[0.5]]))


def test_median_composite_all_nan_pixel_stays_nan():
    with np.errstate(all="ignore"):
        result = median_composite(np.full((3, 1, 1), np.nan))
    assert np.isnan(result[0, 0])


def test_median_composite_rejects_2d_input():
    with pytest.raises(ValueError, match="time, y, x"):
        median_composite(np.zeros((4, 4)))


# --- Statistics ----------------------------------------------------------


def test_change_statistics_counts_only_valid_pixels():
    delta = np.array([[-0.3, 0.0, 0.3, np.nan]])
    stats = change_statistics(delta, threshold=0.1)
    assert stats["valid_pixels"] == 3
    assert stats["total_pixels"] == 4
    assert stats["loss_fraction"] == pytest.approx(1 / 3)
    assert stats["gain_fraction"] == pytest.approx(1 / 3)
    assert stats["stable_fraction"] == pytest.approx(1 / 3)


def test_change_statistics_threshold_is_inclusive():
    stats = change_statistics(np.array([[-0.1, 0.1]]), threshold=0.1)
    assert stats["loss_fraction"] == pytest.approx(0.5)
    assert stats["gain_fraction"] == pytest.approx(0.5)


def test_change_statistics_all_masked():
    stats = change_statistics(np.full((2, 2), np.nan))
    assert stats["valid_pixels"] == 0
    assert np.isnan(stats["mean_delta"])
