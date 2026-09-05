"""STAC access layer: scene discovery, grid-aligned loading, compositing.

Everything that touches the network lives here. The key invariant is that
both seasons are read onto **one shared GeoBox**, so the delta in
``ndvi_core.calculate_index_delta`` compares identical ground locations
pixel for pixel. Loading each scene on its own native grid (the obvious but
wrong approach) yields rasters of different shapes and origins that either
crash on subtraction or, worse, broadcast into a meaningless result.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import numpy as np
import odc.stac
import planetary_computer as pc
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox
from pystac_client import Client

from ndvi_core import (
    DEFAULT_INDEX,
    SpectralIndex,
    get_index,
    median_composite,
    needs_boa_offset,
    normalized_difference,
    observation_count,
    robust_scale,
    scl_valid_mask,
    to_reflectance,
)

LOGGER = logging.getLogger(__name__)

STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

#: Peak vegetation window in the northern hemisphere, used for both seasons.
#: Override it for southern-hemisphere or dry-season analysis.
DEFAULT_SEASON = ("06-01", "08-31")

#: Guard against an area of interest that would exhaust memory. A 40 MP
#: raster at float64 is ~320 MB per band-year, and hosted Streamlit
#: instances typically cap out around 1 GB.
MAX_PIXELS = 40_000_000

#: A few ready-made areas of interest (west, south, east, north in EPSG:4326).
AOI_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "Graz, Austria": (15.35, 47.01, 15.50, 47.12),
    "Neusiedler See, Austria": (16.62, 47.70, 16.85, 47.90),
    "Lake Chad, Chad": (14.00, 12.90, 14.50, 13.40),
    "Doñana wetlands, Spain": (-6.50, 36.90, -6.20, 37.15),
}


class AreaTooLargeError(ValueError):
    """The requested bounding box and resolution exceed the pixel budget."""


def utm_epsg_for_bbox(bbox: tuple[float, float, float, float]) -> str:
    """Pick the UTM zone containing the centre of ``bbox``.

    Working in metres matters: a metric resolution applied in EPSG:3857 is
    stretched by 1/cos(latitude), which at 47 degrees north means a nominal
    "20 m" pixel actually covers about 13.6 m on the ground.
    """
    west, south, east, north = bbox
    lon = (west + east) / 2.0
    lat = (south + north) / 2.0
    zone = int((lon + 180.0) // 6.0) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    """Reject a malformed bounding box before any network call."""
    west, south, east, north = bbox
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError(f"Longitude out of range in {bbox}")
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise ValueError(f"Latitude out of range in {bbox}")
    if west >= east:
        raise ValueError(f"West ({west}) must be smaller than east ({east})")
    if south >= north:
        raise ValueError(f"South ({south}) must be smaller than north ({north})")


def build_geobox(
    bbox: tuple[float, float, float, float],
    resolution: float = 20.0,
    max_pixels: int = MAX_PIXELS,
) -> GeoBox:
    """One deterministic analysis grid, reused by every scene and season."""
    validate_bbox(bbox)
    crs = utm_epsg_for_bbox(bbox)
    projected = BoundingBox(*bbox, crs="EPSG:4326").to_crs(crs)
    geobox = GeoBox.from_bbox(projected, resolution=resolution, tight=True)

    pixels = geobox.shape.x * geobox.shape.y
    if pixels > max_pixels:
        raise AreaTooLargeError(
            f"{geobox.shape.x} x {geobox.shape.y} = {pixels:,} pixels exceeds the "
            f"{max_pixels:,} budget. Use a coarser --resolution or a smaller area."
        )
    return geobox


@dataclass
class SeasonComposite:
    """A cloud-masked, grid-aligned index composite for a single season."""

    year: int
    index: SpectralIndex
    values: np.ndarray
    geobox: GeoBox | None = None
    dates: list[str] = field(default_factory=list)
    scene_count: int = 0
    cloud_cover: list[float] = field(default_factory=list)
    #: Per-pixel number of scenes that contributed to the composite.
    observations: np.ndarray | None = None
    #: Per-pixel scatter across those scenes, in index units.
    noise: np.ndarray | None = None

    @property
    def mean_doy(self) -> float | None:
        """Mean day of year of the contributing acquisitions.

        Comparing a mid-June composite against a late-August one measures
        the growing season as much as it measures change, so the pipeline
        reports this and lets the reader judge.
        """
        if not self.dates:
            return None
        days = [dt.date.fromisoformat(d).timetuple().tm_yday for d in self.dates]
        return sum(days) / len(days)

    @property
    def date_range(self) -> str:
        if not self.dates:
            return "n/a"
        if len(self.dates) == 1:
            return self.dates[0]
        return f"{self.dates[0]} … {self.dates[-1]} ({self.scene_count} scenes)"

    @property
    def ndvi(self) -> np.ndarray:
        """Backwards-compatible alias from the NDVI-only version."""
        return self.values


#: Old name, kept so existing imports keep working.
YearComposite = SeasonComposite


def open_catalog(endpoint: str = STAC_ENDPOINT) -> Client:
    return Client.open(endpoint)


def scene_tile(item) -> str:
    """The MGRS tile a STAC item belongs to.

    Providers spell this differently, so try the known keys in turn and fall
    back to a per-item key, which degrades to the old one-group behaviour
    rather than mis-grouping scenes that cover different ground.
    """
    props = getattr(item, "properties", {}) or {}
    for key in ("s2:mgrs_tile", "grid:code", "sentinel:grid_square"):
        value = props.get(key)
        if value:
            return str(value)
    return f"unknown:{getattr(item, 'id', id(item))}"


def search_scenes(
    catalog: Client,
    bbox: tuple[float, float, float, float],
    year: int,
    max_cloud: float = 10.0,
    season: tuple[str, str] = DEFAULT_SEASON,
    max_scenes: int = 6,
) -> list:
    """Return the least-cloudy scenes for ``year``, grouped so that every
    Sentinel-2 tile touching the area is represented.

    ``max_scenes`` is a budget **per MGRS tile**, not per request. That
    distinction matters: an area of interest wider than ~110 km spans several
    tiles, and a single global sort by ``eo:cloud_cover`` can return N scenes
    that all belong to the same tile. The rest of the area then has no
    observations at all and is silently masked away as no-data — which shows
    up as a low valid-pixel fraction and, where two tiles do meet, as a
    straight-edged seam in the delta.

    Within each tile the scenes are still ordered cloudiest last, so a
    single-tile area behaves exactly as before.
    """
    start, end = season
    search = catalog.search(
        collections=[COLLECTION],
        bbox=list(bbox),
        datetime=f"{year}-{start}/{year}-{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100.0))

    per_tile: dict[str, list] = {}
    for item in items:
        per_tile.setdefault(scene_tile(item), []).append(item)

    selected = [it for group in per_tile.values() for it in group[:max_scenes]]
    selected.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100.0))

    LOGGER.info(
        "Found %d candidate scenes for %d across %d tile(s); kept %d",
        len(items),
        year,
        len(per_tile),
        len(selected),
    )
    return selected


def load_season_composite(
    catalog: Client,
    bbox: tuple[float, float, float, float],
    year: int,
    geobox: GeoBox,
    max_cloud: float = 10.0,
    season: tuple[str, str] = DEFAULT_SEASON,
    max_scenes: int = 6,
    index: str | SpectralIndex = DEFAULT_INDEX,
) -> SeasonComposite:
    """Search, load, harmonise, cloud-mask and composite one season."""
    spec = index if isinstance(index, SpectralIndex) else get_index(index)

    items = search_scenes(
        catalog, bbox, year, max_cloud=max_cloud, season=season, max_scenes=max_scenes
    )
    if not items:
        raise LookupError(
            f"No Sentinel-2 scene for {year} in {season[0]}–{season[1]} "
            f"below {max_cloud}% cloud cover. Try raising the cloud threshold "
            f"or widening the season."
        )

    bands = [spec.band_a, spec.band_b, "SCL"]
    dataset = odc.stac.load(
        items,
        bands=bands,
        geobox=geobox,
        groupby="solar_day",
        resampling={"SCL": "nearest", "*": "bilinear"},
        patch_url=pc.sign_url,
        chunks=None,
    )

    stack = []
    dates: list[str] = []
    for step in range(dataset.sizes["time"]):
        scene = dataset.isel(time=step)
        acquired = str(np.datetime_as_string(scene["time"].values, unit="D"))
        offset = needs_boa_offset(acquired)

        a = to_reflectance(scene[spec.band_a].values, apply_offset=offset)
        b = to_reflectance(scene[spec.band_b].values, apply_offset=offset)
        mask = scl_valid_mask(scene["SCL"].values)

        stack.append(normalized_difference(a, b, valid_mask=mask))
        dates.append(acquired)

    cube = np.stack(stack)
    composite = stack[0] if len(stack) == 1 else median_composite(cube)

    return SeasonComposite(
        year=year,
        index=spec,
        values=composite,
        geobox=geobox,
        dates=sorted(dates),
        scene_count=len(stack),
        observations=observation_count(cube),
        noise=robust_scale(cube),
        cloud_cover=[
            float(it.properties.get("eo:cloud_cover", float("nan"))) for it in items
        ],
    )


#: Old name, kept so existing imports keep working.
load_year_composite = load_season_composite
