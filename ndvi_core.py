"""Calculations for the change detection (indices, compositing, statistics).

No network access in this file, so everything here can be unit tested.
Data loading is in data_access.py, the CLI is main.py and the app is app.py.

Two things here matter a lot for correct results:

1. Reflectance offset: since ESA processing baseline 04.00 (data from
   2022-01-25 on) the L2A values have an offset of -1000. If you don't
   correct it, comparing 2021 with 2024 shows a fake change over the whole
   image.
2. Cloud masking per pixel: the scene cloud cover says nothing about a
   single pixel, so clouds, shadows, snow etc. are removed with the SCL band.

Missing values are always NaN and never 0, because 0 is a real index value
(bare soil, rock) and would mess up means and counts.
"""

from __future__ import annotations

import datetime as _dt
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

# --- Sentinel-2 L2A radiometry -------------------------------------------

# First acquisition date with processing baseline 04.00
BASELINE_04_00_CUTOFF = _dt.date(2022, 1, 25)

# Offset that ESA adds to the reflectance values since baseline 04.00
BOA_ADD_OFFSET = -1000.0

# Divide the raw values by this to get reflectance (0-1)
BOA_QUANTIFICATION_VALUE = 10000.0

# Converts the median absolute deviation (MAD) to the same scale as a
# standard deviation (for normally distributed data).
MAD_TO_SIGMA = 1.4826

# --- Scene Classification Layer (SCL) ------------------------------------

SCL_NO_DATA = 0
SCL_SATURATED = 1
SCL_DARK_AREA = 2
SCL_CLOUD_SHADOW = 3
SCL_VEGETATION = 4
SCL_NOT_VEGETATED = 5
SCL_WATER = 6
SCL_UNCLASSIFIED = 7
SCL_CLOUD_MEDIUM_PROB = 8
SCL_CLOUD_HIGH_PROB = 9
SCL_THIN_CIRRUS = 10
SCL_SNOW = 11

# SCL classes that are kept. Water is kept on purpose, otherwise shrinking
# lakes and reservoirs could not be detected.
DEFAULT_VALID_SCL_CLASSES: tuple[int, ...] = (
    SCL_VEGETATION,
    SCL_NOT_VEGETATED,
    SCL_WATER,
    SCL_UNCLASSIFIED,
)


# --- Spectral index registry ---------------------------------------------


@dataclass(frozen=True)
class SpectralIndex:
    """A normalised difference index: (a - b) / (a + b).

    increase_label and decrease_label say what a positive or negative delta
    means for this index (e.g. "Water gain" for NDWI), so the plots and the
    app can show the right text.
    """

    name: str
    band_a: str
    band_b: str
    description: str
    increase_label: str
    decrease_label: str
    display_min: float = -0.2
    display_max: float = 0.8

    @property
    def bands(self) -> tuple[str, str]:
        return (self.band_a, self.band_b)


INDICES: dict[str, SpectralIndex] = {
    "NDVI": SpectralIndex(
        name="NDVI",
        band_a="B08",  # NIR, reflected by leaf mesophyll
        band_b="B04",  # Red, absorbed by chlorophyll
        description="Normalised Difference Vegetation Index: vegetation health.",
        increase_label="Vegetation gain",
        decrease_label="Vegetation loss",
        display_min=-0.2,
        display_max=0.8,
    ),
    "NDWI": SpectralIndex(
        name="NDWI",
        band_a="B03",  # Green
        band_b="B08",  # NIR, strongly absorbed by water
        description=(
            "Normalised Difference Water Index (McFeeters 1996): open water "
            "extent. Positive values indicate water."
        ),
        increase_label="Water gain / flooding",
        decrease_label="Water loss / drying",
        display_min=-0.6,
        display_max=0.6,
    ),
    "NBR": SpectralIndex(
        name="NBR",
        band_a="B08",  # NIR, drops after fire
        band_b="B12",  # SWIR 2, rises after fire
        description=(
            "Normalised Burn Ratio: fire severity and post-fire recovery. "
            "Note the sign: this pipeline reports comparison minus baseline, "
            "so a burn appears as a NEGATIVE delta. The dNBR convention in "
            "the literature is the reverse (baseline minus comparison)."
        ),
        increase_label="Regrowth",
        decrease_label="Burn / vegetation loss",
        display_min=-0.2,
        display_max=0.8,
    ),
}

DEFAULT_INDEX = "NDVI"


def get_index(name: str) -> SpectralIndex:
    """Get an index by name (not case sensitive)."""
    try:
        return INDICES[name.upper()]
    except KeyError:
        raise KeyError(
            f"Unknown index {name!r}. Available: {', '.join(sorted(INDICES))}"
        ) from None


# --- Radiometry ----------------------------------------------------------


def needs_boa_offset(
    acquired: _dt.date | _dt.datetime | str,
    processing_baseline: str | float | None = None,
) -> bool:
    """Check if a scene has the -1000 reflectance offset.

    The best way to know is the processing baseline of the scene (STAC field
    "s2:processing_baseline", e.g. "05.09"). ESA reprocessed some older
    scenes with a newer baseline, and those have the offset too, even though
    they are from before 2022-01-25. Only if the baseline is missing or
    can't be read, the acquisition date is used.

    acquired can be a date, a datetime or an ISO string like "2024-08-29".
    """
    if processing_baseline is not None:
        try:
            return float(processing_baseline) >= 4.0
        except (TypeError, ValueError):
            pass  # can't read the value, use the date instead
    if isinstance(acquired, str):
        acquired = _dt.datetime.fromisoformat(acquired.replace("Z", "+00:00"))
    if isinstance(acquired, _dt.datetime):
        acquired = acquired.date()
    return acquired >= BASELINE_04_00_CUTOFF


def to_reflectance(digital_numbers: np.ndarray, apply_offset: bool) -> np.ndarray:
    """Convert raw L2A values to reflectance (roughly 0 to 1).

    Raw value 0 means no data in Sentinel-2, so it becomes NaN.
    apply_offset should be the result of needs_boa_offset().
    """
    dn = np.asarray(digital_numbers, dtype="float64")
    reflectance = np.where(dn == SCL_NO_DATA, np.nan, dn)
    if apply_offset:
        reflectance = reflectance + BOA_ADD_OFFSET
    return reflectance / BOA_QUANTIFICATION_VALUE


def scl_valid_mask(
    scl: np.ndarray, valid_classes: Iterable[int] = DEFAULT_VALID_SCL_CLASSES
) -> np.ndarray:
    """True where the SCL says the pixel is usable (land or water),
    False for clouds, cirrus, shadow, snow and no data."""
    return np.isin(np.asarray(scl), tuple(valid_classes))


# --- Index computation ---------------------------------------------------


def normalized_difference(
    band_a: np.ndarray,
    band_b: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate (a - b) / (a + b).

    Both bands have to be converted with to_reflectance() first. Because
    this is a ratio, a missing offset correction does NOT cancel out.

    The result is NaN where a + b is 0, where an input is NaN or where
    valid_mask is False. Values are clipped to [-1, 1].
    """
    a = np.asarray(band_a, dtype="float64")
    b = np.asarray(band_b, dtype="float64")

    if a.shape != b.shape:
        raise ValueError(f"Bands must share a grid, got {a.shape} and {b.shape}")

    denominator = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator == 0, np.nan, (a - b) / denominator)

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != result.shape:
            raise ValueError(
                f"Mask shape {valid_mask.shape} does not match band shape {result.shape}"
            )
        result = np.where(valid_mask, result, np.nan)

    return np.clip(result, -1.0, 1.0)


def calculate_ndvi(
    red_band: np.ndarray,
    nir_band: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red).

    Just a shortcut for normalized_difference with NIR first.
    """
    return normalized_difference(nir_band, red_band, valid_mask=valid_mask)


def calculate_index_delta(baseline: np.ndarray, comparison: np.ndarray) -> np.ndarray:
    """Change per pixel: comparison minus baseline.

    Raises an error if the shapes don't match. Otherwise numpy might
    broadcast the arrays and compare pixels that aren't the same place.
    """
    base = np.asarray(baseline, dtype="float64")
    comp = np.asarray(comparison, dtype="float64")

    if base.shape != comp.shape:
        raise ValueError(
            "Baseline and comparison rasters are not on the same grid: "
            f"{base.shape} vs {comp.shape}. Load both scenes with a "
            "shared GeoBox (see data_access.build_geobox)."
        )

    return comp - base


# Old name from when the project only did NDVI
calculate_ndvi_delta = calculate_index_delta


def median_composite(stack: np.ndarray) -> np.ndarray:
    """Median over time for every pixel, ignoring NaN.

    Input shape is (time, y, x). A median of several scenes is much more
    stable than a single date, which mostly shows that day's conditions.
    """
    stack = np.asarray(stack, dtype="float64")
    if stack.ndim < 3:
        raise ValueError(f"Expected a (time, y, x) stack, got shape {stack.shape}")
    clean = np.where(np.isfinite(stack), stack, np.nan)
    with warnings.catch_warnings():
        # a pixel can be cloudy in every scene, that's fine
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmedian(clean, axis=0)


def observation_count(stack: np.ndarray) -> np.ndarray:
    """Number of valid scenes per pixel.

    In the composite you can't see if a pixel is based on 1 scene or 6,
    but the 1-scene pixels are less reliable. This count makes it visible.
    """
    stack = np.asarray(stack, dtype="float64")
    if stack.ndim < 3:
        raise ValueError(f"Expected a (time, y, x) stack, got shape {stack.shape}")
    return np.isfinite(stack).sum(axis=0).astype("int32")


def robust_scale(stack: np.ndarray) -> np.ndarray:
    """Noise per pixel, measured as scaled median absolute deviation (MAD).

    I use the MAD instead of the standard deviation because with only a few
    scenes, one missed cloud edge would blow up the standard deviation.
    Pixels with less than 2 observations get NaN, because you can't
    measure spread from one value.
    """
    stack = np.asarray(stack, dtype="float64")
    if stack.ndim < 3:
        raise ValueError(f"Expected a (time, y, x) stack, got shape {stack.shape}")

    clean = np.where(np.isfinite(stack), stack, np.nan)
    counts = np.isfinite(clean).sum(axis=0)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        centre = np.nanmedian(clean, axis=0)
        deviation = np.abs(clean - centre)
        mad = np.nanmedian(deviation, axis=0)

    return np.where(counts >= 2, mad * MAD_TO_SIGMA, np.nan)


def pairwise_observations(
    baseline_counts: np.ndarray, comparison_counts: np.ndarray
) -> np.ndarray:
    """Observations for the delta = the smaller count of the two years.

    If 2021 has 1 scene and 2024 has 6, the delta is still based on 1 scene
    for the 2021 side.
    """
    a = np.asarray(baseline_counts)
    b = np.asarray(comparison_counts)
    if a.shape != b.shape:
        raise ValueError(f"Count fields must share a grid, got {a.shape} and {b.shape}")
    return np.minimum(a, b).astype("int32")


def combine_noise(baseline_noise: np.ndarray, comparison_noise: np.ndarray):
    """Noise of the delta: sqrt(a^2 + b^2), since the two years are independent."""
    a = np.asarray(baseline_noise, dtype="float64")
    b = np.asarray(comparison_noise, dtype="float64")
    if a.shape != b.shape:
        raise ValueError(f"Noise fields must share a grid, got {a.shape} and {b.shape}")
    return np.sqrt(a**2 + b**2)


def adaptive_threshold(
    noise: np.ndarray, sigma: float = 2.0, floor: float = 0.05
) -> np.ndarray:
    """Threshold per pixel: sigma * noise, but at least floor.

    One fixed threshold for the whole image doesn't work well: on noisy
    bare ground it counts noise as change, and on stable forest it misses
    small real changes. So every pixel is compared with its own noise.
    The floor stops very quiet pixels from getting a tiny threshold.

    Pixels where the noise is unknown (too few scenes) get the median
    threshold of the image, not the floor. Otherwise the least reliable
    pixels would get the easiest threshold.
    """
    noise = np.asarray(noise, dtype="float64")
    scaled = sigma * noise
    measured = np.isfinite(scaled)

    typical = max(float(np.median(scaled[measured])), floor) if measured.any() else floor
    return np.where(measured, np.maximum(scaled, floor), typical)


def change_statistics(delta: np.ndarray, threshold=0.1) -> dict:
    """Summary statistics for a delta array.

    threshold: changes smaller than this count as noise (0.1 is a common
    value for Sentinel-2). It can also be an array with one threshold per
    pixel (see adaptive_threshold).

    loss_fraction and gain_fraction are shares of the valid pixels, not of
    the whole image. The names fit NDVI; for other indices use
    decrease_label / increase_label when showing them.
    """
    delta = np.asarray(delta, dtype="float64")
    finite = np.isfinite(delta)
    valid = int(finite.sum())
    total = int(delta.size)

    limits = np.asarray(threshold, dtype="float64")
    if limits.ndim and limits.shape != delta.shape:
        raise ValueError(
            f"Per-pixel threshold shape {limits.shape} does not match the "
            f"delta shape {delta.shape}"
        )
    adaptive = bool(limits.ndim)
    reported = float(np.nanmedian(limits)) if adaptive else float(limits)

    if valid == 0:
        return {
            "valid_pixels": 0,
            "total_pixels": total,
            "valid_fraction": 0.0,
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
            "loss_fraction": float("nan"),
            "gain_fraction": float("nan"),
            "stable_fraction": float("nan"),
            "threshold": reported,
            "adaptive_threshold": adaptive,
        }

    values = delta[finite]
    limit_values = limits[finite] if adaptive else limits
    loss = float((values <= -limit_values).sum()) / valid
    gain = float((values >= limit_values).sum()) / valid

    return {
        "valid_pixels": valid,
        "total_pixels": total,
        "valid_fraction": valid / total,
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "loss_fraction": loss,
        "gain_fraction": gain,
        "stable_fraction": 1.0 - loss - gain,
        "threshold": reported,
        "adaptive_threshold": adaptive,
    }
