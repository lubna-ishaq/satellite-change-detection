"""Core spectral-index mathematics for Sentinel-2 L2A change detection.

This module is deliberately free of I/O and network calls so that every
numerical decision can be unit tested. See ``data_access.py`` for the STAC
layer and ``main.py`` / ``app.py`` for the two front ends.

Two corrections implemented here are what separate a plausible-looking
index delta from a defensible one:

1. **Radiometric harmonisation.** From ESA Processing Baseline 04.00
   (products acquired on or after 2022-01-25) Sentinel-2 L2A surface
   reflectance is stored with an additive offset of -1000 DN. Comparing a
   2021 scene against a 2024 scene without applying it produces a large,
   spatially coherent bias across the whole image that is easily mistaken
   for real land cover change.
2. **Per-pixel cloud masking.** A scene-level ``eo:cloud_cover`` filter says
   nothing about the individual pixel. Clouds, cirrus, shadow and snow are
   removed via the Scene Classification Layer (SCL) before any statistics
   are computed.

Invalid pixels are represented as ``NaN``, never as ``0``: zero is a
legitimate index value (bare soil, rock, concrete), so using it as a
no-data flag silently contaminates every mean, histogram and change count.
"""

from __future__ import annotations

import datetime as _dt
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

# --- Sentinel-2 L2A radiometry -------------------------------------------

#: First acquisition date processed with Baseline 04.00.
BASELINE_04_00_CUTOFF = _dt.date(2022, 1, 25)

#: Additive offset (in DN) applied to L2A BOA reflectance from Baseline 04.00.
BOA_ADD_OFFSET = -1000.0

#: DN-to-reflectance scale factor.
BOA_QUANTIFICATION_VALUE = 10000.0

#: Scales a median absolute deviation to a standard deviation for normal
#: data, so a noise field reads in the same units as the index itself.
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

#: Classes kept for analysis. Water is retained on purpose: shrinking lakes
#: and reservoirs are one of the changes this pipeline is meant to surface.
DEFAULT_VALID_SCL_CLASSES: tuple[int, ...] = (
    SCL_VEGETATION,
    SCL_NOT_VEGETATED,
    SCL_WATER,
    SCL_UNCLASSIFIED,
)


# --- Spectral index registry ---------------------------------------------


@dataclass(frozen=True)
class SpectralIndex:
    """A normalised difference index ``(a - b) / (a + b)``.

    ``increase_label`` / ``decrease_label`` name what a positive or negative
    delta physically means, so the UI never has to hard-code "vegetation"
    for an index that measures water.
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
        description="Normalised Difference Vegetation Index — vegetation vigour.",
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
            "Normalised Difference Water Index (McFeeters 1996) — open water "
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
            "Normalised Burn Ratio — fire severity and post-fire recovery. "
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
    """Look up an index by name, case-insensitively."""
    try:
        return INDICES[name.upper()]
    except KeyError:
        raise KeyError(
            f"Unknown index {name!r}. Available: {', '.join(sorted(INDICES))}"
        ) from None


# --- Radiometry ----------------------------------------------------------


def needs_boa_offset(acquired: _dt.date | _dt.datetime | str) -> bool:
    """Return ``True`` if a scene acquired at ``acquired`` carries the
    Baseline 04.00 reflectance offset.

    ``acquired`` may be a ``date``, a ``datetime`` or an ISO 8601 string
    (with or without a trailing ``Z``).
    """
    if isinstance(acquired, str):
        acquired = _dt.datetime.fromisoformat(acquired.replace("Z", "+00:00"))
    if isinstance(acquired, _dt.datetime):
        acquired = acquired.date()
    return acquired >= BASELINE_04_00_CUTOFF


def to_reflectance(digital_numbers: np.ndarray, apply_offset: bool) -> np.ndarray:
    """Convert raw L2A digital numbers to surface reflectance.

    Applies to every BOA reflectance band, not just red and NIR. Pixels
    equal to 0 are Sentinel-2's no-data value and become ``NaN``.
    ``apply_offset`` should come from :func:`needs_boa_offset`.
    """
    dn = np.asarray(digital_numbers, dtype="float64")
    reflectance = np.where(dn == SCL_NO_DATA, np.nan, dn)
    if apply_offset:
        reflectance = reflectance + BOA_ADD_OFFSET
    return reflectance / BOA_QUANTIFICATION_VALUE


def scl_valid_mask(
    scl: np.ndarray, valid_classes: Iterable[int] = DEFAULT_VALID_SCL_CLASSES
) -> np.ndarray:
    """Boolean mask that is ``True`` where the SCL band marks usable land or
    water and ``False`` over cloud, cirrus, shadow, snow and no-data."""
    return np.isin(np.asarray(scl), tuple(valid_classes))


# --- Index computation ---------------------------------------------------


def normalized_difference(
    band_a: np.ndarray,
    band_b: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """``(a - b) / (a + b)`` with the no-data discipline this project needs.

    Both bands must already be in reflectance units (see
    :func:`to_reflectance`); a normalised difference is a ratio, so a
    missing offset correction does not cancel out.

    Pixels where the denominator is zero, where either input is ``NaN``, or
    where ``valid_mask`` is ``False`` are returned as ``NaN``. The result is
    clipped to the physical range [-1, 1].
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
    """NDVI, ``(NIR - Red) / (NIR + Red)``.

    Kept as a named convenience because NDVI is the default index; it is
    :func:`normalized_difference` with the arguments in vegetation order.
    """
    return normalized_difference(nir_band, red_band, valid_mask=valid_mask)


def calculate_index_delta(baseline: np.ndarray, comparison: np.ndarray) -> np.ndarray:
    """Spatial delta matrix ``comparison - baseline``.

    Raises rather than broadcasting when the two rasters sit on different
    grids: a silent broadcast would compare unrelated ground locations.
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


#: Backwards-compatible alias from when this pipeline only handled NDVI.
calculate_ndvi_delta = calculate_index_delta


def median_composite(stack: np.ndarray) -> np.ndarray:
    """NaN-aware median over the leading (time) axis of an index stack.

    A median across every low-cloud scene in the season is far more stable
    than a single acquisition date, which mostly measures that day's
    weather and sun angle.
    """
    stack = np.asarray(stack, dtype="float64")
    if stack.ndim < 3:
        raise ValueError(f"Expected a (time, y, x) stack, got shape {stack.shape}")
    clean = np.where(np.isfinite(stack), stack, np.nan)
    with warnings.catch_warnings():
        # An all-NaN column is legitimate: a pixel clouded in every scene.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmedian(clean, axis=0)


def observation_count(stack: np.ndarray) -> np.ndarray:
    """How many scenes actually contributed to each pixel of a composite.

    A pixel built from one observation and one built from six look identical
    in the composite but are not equally trustworthy. Carrying the count
    lets a reader tell them apart — and makes tile seams, where coverage
    changes abruptly, visible instead of silent.
    """
    stack = np.asarray(stack, dtype="float64")
    if stack.ndim < 3:
        raise ValueError(f"Expected a (time, y, x) stack, got shape {stack.shape}")
    return np.isfinite(stack).sum(axis=0).astype("int32")


def robust_scale(stack: np.ndarray) -> np.ndarray:
    """Per-pixel noise estimate: the median absolute deviation, scaled to
    be comparable with a standard deviation for normal data.

    The MAD is used rather than the standard deviation because a composite
    of a handful of scenes is exactly where one undetected cloud edge would
    dominate a variance. Pixels with fewer than two observations get ``NaN``:
    a single sample says nothing about spread, and pretending it means zero
    noise would make every such pixel look infinitely significant.
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
    """Observations behind a *difference*: the weaker of the two seasons.

    A delta is only as well observed as its thinner side — six scenes in
    2024 do not rescue a single scene in 2021.
    """
    a = np.asarray(baseline_counts)
    b = np.asarray(comparison_counts)
    if a.shape != b.shape:
        raise ValueError(f"Count fields must share a grid, got {a.shape} and {b.shape}")
    return np.minimum(a, b).astype("int32")


def combine_noise(baseline_noise: np.ndarray, comparison_noise: np.ndarray):
    """Noise of a difference of two independent estimates: added in quadrature."""
    a = np.asarray(baseline_noise, dtype="float64")
    b = np.asarray(comparison_noise, dtype="float64")
    if a.shape != b.shape:
        raise ValueError(f"Noise fields must share a grid, got {a.shape} and {b.shape}")
    return np.sqrt(a**2 + b**2)


def adaptive_threshold(
    noise: np.ndarray, sigma: float = 2.0, floor: float = 0.05
) -> np.ndarray:
    """Turn a per-pixel noise field into a per-pixel change threshold.

    A single fixed threshold is wrong in both directions at once: over noisy
    bare ground it lets scatter through as "change", and over a stable
    canopy it hides real change smaller than the constant. This asks the
    same question everywhere instead — is the difference larger than this
    pixel's own scatter? — with a floor so that an implausibly quiet pixel
    cannot make trivial differences look significant.

    Pixels with too few observations to estimate scatter fall back to the
    scene's typical threshold, never to the floor. Falling back to the floor
    would hand the *least* observed pixels the *easiest* bar to clear, which
    is precisely backwards: those are the pixels to be most sceptical about.
    """
    noise = np.asarray(noise, dtype="float64")
    scaled = sigma * noise
    measured = np.isfinite(scaled)

    typical = max(float(np.median(scaled[measured])), floor) if measured.any() else floor
    return np.where(measured, np.maximum(scaled, floor), typical)


def change_statistics(delta: np.ndarray, threshold=0.1) -> dict:
    """Quantify an index delta matrix.

    ``threshold`` is the magnitude below which a change is treated as noise.
    0.1 is a common conservative choice for Sentinel-2 summer pairs. It may
    also be a **per-pixel field** of the same shape as ``delta`` — see
    :func:`adaptive_threshold` — in which case each pixel is judged against
    its own scatter instead of one constant for the whole scene.

    Returns valid-pixel counts and the share of the *valid* area (not of the
    full raster) that decreased or increased. The ``loss_``/``gain_`` keys
    read naturally for NDVI; for other indices use the index's
    ``decrease_label`` / ``increase_label`` when presenting them.
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
