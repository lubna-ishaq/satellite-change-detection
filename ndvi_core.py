import numpy as np


def calculate_ndvi(red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """Calculates the Normalized Difference Vegetation Index (NDVI)

    from Red and Near-Infrared (NIR) spectral bands.
    """
    numerator = nir_band.astype(float) - red_band.astype(float)
    denominator = nir_band.astype(float) + red_band.astype(float)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(denominator == 0, 0, numerator / denominator)

    return ndvi


def calculate_ndvi_delta(
    baseline_ndvi: np.ndarray, comparison_ndvi: np.ndarray
) -> np.ndarray:
    """Calculates the spatial delta matrix between two NDVI arrays."""
    return comparison_ndvi - baseline_ndvi