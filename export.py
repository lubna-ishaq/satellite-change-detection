"""Georeferenced raster export.

A PNG is a picture; a GeoTIFF is data. Writing the delta with its CRS and
affine transform is what lets the result be opened in QGIS or ArcGIS,
stacked against other layers, and measured — which is the difference
between a demo and something an analyst can use.
"""

from __future__ import annotations

import io

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

#: float32 keeps files half the size of float64 at ~7 significant digits,
#: far more than a normalised index (range [-1, 1]) can carry.
DTYPE = "float32"

#: NaN as nodata is what GDAL-aware software expects for float rasters.
NODATA = float("nan")


def _as_bands(arrays) -> list:
    """Accept a single raster or a list of them."""
    if isinstance(arrays, np.ndarray):
        return [arrays]
    return list(arrays)


def _profile(geobox, count: int = 1) -> dict:
    return {
        "driver": "GTiff",
        "height": geobox.shape.y,
        "width": geobox.shape.x,
        "count": count,
        "dtype": DTYPE,
        "crs": CRS.from_user_input(str(geobox.crs)),
        "transform": Affine(*geobox.transform[:6]),
        "nodata": NODATA,
        "compress": "deflate",
        "predictor": 3,  # floating-point predictor, good for smooth rasters
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }


def _validate(bands, geobox) -> list:
    expected = (geobox.shape.y, geobox.shape.x)
    prepared = []
    for array in bands:
        data = np.asarray(array, dtype=DTYPE)
        if data.shape != expected:
            raise ValueError(
                f"Array shape {data.shape} does not match the GeoBox {expected}"
            )
        prepared.append(data)
    return prepared


def _write(dst, bands, descriptions, metadata) -> None:
    for position, data in enumerate(bands, start=1):
        dst.write(data, position)
        if position <= len(descriptions):
            dst.set_band_description(position, descriptions[position - 1])
    if metadata:
        dst.update_tags(**{k: str(v) for k, v in metadata.items()})


def write_geotiff(
    path,
    array,
    geobox,
    band_description="delta",
    metadata: dict | None = None,
) -> None:
    """Write a georeferenced raster to ``path``.

    ``array`` may be one raster or several. Passing the delta together with
    the per-pixel observation count is what lets a reader in QGIS tell a
    confident pixel from a thinly observed one.
    """
    bands = _validate(_as_bands(array), geobox)
    descriptions = (
        [band_description]
        if isinstance(band_description, str)
        else list(band_description)
    )

    with rasterio.open(path, "w", **_profile(geobox, count=len(bands))) as dst:
        _write(dst, bands, descriptions, metadata)


def geotiff_bytes(
    array,
    geobox,
    band_description="delta",
    metadata: dict | None = None,
) -> bytes:
    """Same as :func:`write_geotiff` but into memory, for a download button."""
    bands = _validate(_as_bands(array), geobox)
    descriptions = (
        [band_description]
        if isinstance(band_description, str)
        else list(band_description)
    )

    buffer = io.BytesIO()
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**_profile(geobox, count=len(bands))) as dst:
            _write(dst, bands, descriptions, metadata)
        buffer.write(memfile.read())
    return buffer.getvalue()
