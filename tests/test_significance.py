"""Per-pixel observation counts, noise estimates and adaptive thresholds."""

import numpy as np
import pytest

from ndvi_core import (
    MAD_TO_SIGMA,
    adaptive_threshold,
    change_statistics,
    combine_noise,
    observation_count,
    pairwise_observations,
    robust_scale,
)


def _stack(*scenes):
    return np.stack([np.asarray(s, dtype="float64") for s in scenes])


# --- observation counts --------------------------------------------------


def test_counts_only_finite_observations():
    stack = _stack([[0.4, 0.4]], [[0.5, np.nan]], [[0.6, np.nan]])
    np.testing.assert_array_equal(observation_count(stack), np.array([[3, 1]]))


def test_a_fully_clouded_pixel_counts_zero():
    stack = _stack([[np.nan]], [[np.nan]])
    assert observation_count(stack)[0, 0] == 0


def test_counts_reject_a_2d_input():
    with pytest.raises(ValueError, match="time, y, x"):
        observation_count(np.zeros((4, 4)))


def test_a_delta_is_only_as_observed_as_its_thinner_season():
    """Six scenes in 2024 do not rescue one scene in 2021."""
    base = np.array([[1, 6]])
    comp = np.array([[6, 6]])
    np.testing.assert_array_equal(pairwise_observations(base, comp), np.array([[1, 6]]))


def test_pairwise_observations_rejects_mismatched_grids():
    with pytest.raises(ValueError, match="share a grid"):
        pairwise_observations(np.zeros((2, 2)), np.zeros((3, 3)))


# --- noise ---------------------------------------------------------------


def test_robust_scale_matches_the_mad_definition():
    stack = _stack([[0.0]], [[1.0]], [[2.0]], [[3.0]])
    # median 1.5, absolute deviations {1.5, 0.5, 0.5, 1.5}, median 1.0
    assert robust_scale(stack)[0, 0] == pytest.approx(MAD_TO_SIGMA)


def test_a_constant_pixel_has_zero_noise():
    stack = _stack([[0.5]], [[0.5]], [[0.5]])
    assert robust_scale(stack)[0, 0] == pytest.approx(0.0)


def test_a_single_observation_has_no_noise_estimate():
    """One sample says nothing about spread; zero would be a lie."""
    stack = _stack([[0.5]], [[np.nan]], [[np.nan]])
    assert np.isnan(robust_scale(stack)[0, 0])


def test_noise_resists_one_bad_scene():
    """A missed cloud edge must not dominate the estimate."""
    clean = _stack([[0.50]], [[0.51]], [[0.49]], [[0.50]])
    with_outlier = _stack([[0.50]], [[0.51]], [[0.49]], [[0.95]])
    assert robust_scale(with_outlier)[0, 0] < 4 * robust_scale(clean)[0, 0] + 0.05


def test_combine_noise_adds_in_quadrature():
    a = np.array([[3.0]])
    b = np.array([[4.0]])
    assert combine_noise(a, b)[0, 0] == pytest.approx(5.0)


def test_combine_noise_rejects_mismatched_grids():
    with pytest.raises(ValueError, match="share a grid"):
        combine_noise(np.zeros((2, 2)), np.zeros((3, 3)))


# --- adaptive threshold --------------------------------------------------


def test_threshold_scales_with_noise():
    thresholds = adaptive_threshold(np.array([[0.02, 0.10]]), sigma=2.0, floor=0.01)
    assert thresholds[0, 1] > thresholds[0, 0]
    assert thresholds[0, 1] == pytest.approx(0.20)


def test_floor_protects_an_implausibly_quiet_pixel():
    assert adaptive_threshold(np.array([[0.0]]), floor=0.05)[0, 0] == pytest.approx(0.05)


def test_unmeasured_pixels_get_the_typical_bar_not_the_easiest():
    """Regression guard.

    An earlier version sent pixels with no noise estimate to the floor,
    which handed the *least* observed pixels the *easiest* threshold — the
    opposite of the intended scepticism.
    """
    noise = np.array([[0.10, 0.10, np.nan]])
    thresholds = adaptive_threshold(noise, sigma=2.0, floor=0.05)
    assert thresholds[0, 2] == pytest.approx(thresholds[0, 0])
    assert thresholds[0, 2] > 0.05


def test_all_unmeasured_falls_back_to_the_floor():
    thresholds = adaptive_threshold(np.full((2, 2), np.nan), floor=0.07)
    assert np.allclose(thresholds, 0.07)


# --- statistics with a per-pixel threshold -------------------------------


def test_statistics_accept_a_threshold_field():
    delta = np.array([[0.15, 0.15]])
    limits = np.array([[0.10, 0.30]])  # left pixel quiet, right pixel noisy
    stats = change_statistics(delta, threshold=limits)
    assert stats["gain_fraction"] == pytest.approx(0.5)
    assert stats["adaptive_threshold"] is True


def test_a_scalar_threshold_is_still_reported_as_fixed():
    stats = change_statistics(np.array([[0.2]]), threshold=0.1)
    assert stats["adaptive_threshold"] is False
    assert stats["threshold"] == pytest.approx(0.1)


def test_reported_threshold_is_the_median_of_a_field():
    limits = np.array([[0.1, 0.2, 0.3]])
    stats = change_statistics(np.zeros((1, 3)), threshold=limits)
    assert stats["threshold"] == pytest.approx(0.2)


def test_threshold_field_must_match_the_delta_grid():
    with pytest.raises(ValueError, match="does not match"):
        change_statistics(np.zeros((2, 2)), threshold=np.zeros((3, 3)))


def test_masked_pixels_are_ignored_by_a_threshold_field():
    delta = np.array([[0.5, np.nan]])
    limits = np.array([[0.1, 0.1]])
    stats = change_statistics(delta, threshold=limits)
    assert stats["valid_pixels"] == 1
    assert stats["gain_fraction"] == pytest.approx(1.0)


def test_adaptive_is_stricter_where_the_ground_is_noisy():
    """The point of the whole mechanism, in one assertion.

    A 0.12 delta counts as change over a stable canopy and does not over
    scrubland that scatters by that much on its own — a single constant
    cannot express both.
    """
    delta = np.full((1, 2), 0.12)
    noise = np.array([[0.01, 0.09]])  # canopy, scrub
    limits = adaptive_threshold(noise, sigma=2.0, floor=0.05)
    stats = change_statistics(delta, threshold=limits)
    assert stats["gain_fraction"] == pytest.approx(0.5)
    # A single constant calls both pixels changed, or neither.
    assert change_statistics(delta, 0.10)["gain_fraction"] == pytest.approx(1.0)
    assert change_statistics(delta, 0.20)["gain_fraction"] == pytest.approx(0.0)
