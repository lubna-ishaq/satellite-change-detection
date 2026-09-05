import numpy as np
from ndvi_core import calculate_ndvi, calculate_ndvi_delta


def test_calculate_ndvi_basic():
    red = np.array([[100, 200]])
    nir = np.array([[300, 200]])
    # (300-100)/(300+100) = 200/400 = 0.5
    # (200-200)/(200+200) = 0
    expected = np.array([[0.5, 0.0]])
    result = calculate_ndvi(red, nir)
    np.testing.assert_almost_equal(result, expected)


def test_calculate_ndvi_zero_division():
    red = np.array([[0, 0]])
    nir = np.array([[0, 0]])
    result = calculate_ndvi(red, nir)
    assert result[0, 0] == 0


def test_calculate_ndvi_delta():
    baseline = np.array([[0.2, 0.5]])
    comparison = np.array([[0.6, 0.3]])
    expected = np.array([[0.4, -0.2]])
    result = calculate_ndvi_delta(baseline, comparison)
    np.testing.assert_almost_equal(result, expected)