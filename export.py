"""Export the result as a GeoTIFF.

A PNG only shows the result. A GeoTIFF also stores where each pixel is on
the ground (CRS and transform), so it can be opened in QGIS or ArcGIS and
combined with other map layers.
"""

from __future__ import annotations

import io

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

# float32 is precise enough for index values between -1 and 1
# and makes the file half as big as float64.
DTYPE = "float32"

# GIS tools expect NaN as the no-data value for float rasters.
NODATA = float("nan")


def _as_bands(arrays) -> list:
    """Accept either one array or a list of arrays."""
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
        "predictor": 3,  # compresses float data better
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
    """Write one or more bands as a GeoTIFF file.

    I usually pass the delta plus the observation count, so in QGIS you can
    check how many scenes each pixel is based on.
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
    """Same as write_geotiff, but returns the file as bytes (for the app download)."""
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
