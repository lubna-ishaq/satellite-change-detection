"""Loading Sentinel-2 data: scene search, loading onto one grid, compositing.

All network access happens in this file. The most important point: both
years are loaded onto the same grid (GeoBox), so pixel (i, j) is the same
place on the ground in both years. If every scene is loaded on its own
grid, the arrays have different shapes and the subtraction either fails
or compares the wrong pixels.
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

# Summer in the northern hemisphere, used for both years.
# Change it for the southern hemisphere or dry-season analysis.
DEFAULT_SEASON = ("06-01", "08-31")

# Maximum grid size in pixels. Note: this limits the raster size, not the
# total memory. All scenes of a composite are in memory at the same time,
# so memory also grows with the number of scenes (see estimate_peak_memory).
MAX_PIXELS = 40_000_000

# Rough memory use per pixel per scene while building a composite:
# 2 bands as uint16 + SCL as uint8, the float64 index stack, and the
# temporary copies from nanmedian and the MAD.
BYTES_PER_PIXEL_SCENE = 37

# About the memory of a free Streamlit Community Cloud app
HOSTED_MEMORY_BYTES = 1_000_000_000

# Some example areas (west, south, east, north in EPSG:4326)
AOI_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "Graz, Austria": (15.35, 47.01, 15.50, 47.12),
    "Neusiedler See, Austria": (16.62, 47.70, 16.85, 47.90),
    "Lake Chad, Chad": (14.00, 12.90, 14.50, 13.40),
    "Doñana wetlands, Spain": (-6.50, 36.90, -6.20, 37.15),
}


class AreaTooLargeError(ValueError):
    """The area is too big for the chosen resolution."""


def utm_epsg_for_bbox(bbox: tuple[float, float, float, float]) -> str:
    """Return the UTM zone (EPSG code) for the centre of the bbox.

    I use UTM because it's in metres. In Web Mercator (EPSG:3857) a "20 m"
    pixel is only about 13.6 m on the ground at 47 degrees north.
    """
    west, south, east, north = bbox
    lon = (west + east) / 2.0
    lat = (south + north) / 2.0
    zone = int((lon + 180.0) // 6.0) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    """Check the bbox before doing any network requests."""
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
    """Create the grid that all scenes of both years are loaded onto."""
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


def estimate_peak_memory(geobox: GeoBox, scenes: int) -> int:
    """Rough estimate of the peak memory (bytes) for one year's composite."""
    pixels = geobox.shape.x * geobox.shape.y
    return pixels * max(scenes, 1) * BYTES_PER_PIXEL_SCENE


@dataclass
class SeasonComposite:
    """Result for one year: the index composite plus some extra info."""

    year: int
    index: SpectralIndex
    values: np.ndarray
    geobox: GeoBox | None = None
    dates: list[str] = field(default_factory=list)
    scene_count: int = 0
    cloud_cover: list[float] = field(default_factory=list)
    # number of valid scenes per pixel
    observations: np.ndarray | None = None
    # noise (scaled MAD) per pixel
    noise: np.ndarray | None = None

    @property
    def mean_doy(self) -> float | None:
        """Average day of the year of the scenes.

        If one year is mostly June and the other mostly August, part of the
        difference is just plant growth. This value is used for a warning.
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
        return f"{self.dates[0]} … {self.dates[-1]}, {self.scene_count} scenes"

    @property
    def ndvi(self) -> np.ndarray:
        """Old name from the NDVI-only version."""
        return self.values


# old name, still used in some places
YearComposite = SeasonComposite


def open_catalog(endpoint: str = STAC_ENDPOINT) -> Client:
    return Client.open(endpoint)


def scene_tile(item) -> str:
    """Return the Sentinel-2 tile ID (MGRS) of a scene.

    Different STAC catalogs use different field names, so I try all of
    them. If none is there, the scene gets its own unique key, so it never
    gets grouped with scenes from other places by mistake.
    """
    props = getattr(item, "properties", {}) or {}
    for key in ("s2:mgrs_tile", "grid:code", "sentinel:grid_square"):
        value = props.get(key)
        if value:
            return str(value)
    return f"unknown:{getattr(item, 'id', id(item))}"


def scene_day(item) -> str:
    """Return the acquisition date (UTC) of a scene as "YYYY-MM-DD".

    Scenes without a date get a unique key, so they are never counted as
    the same day.
    """
    stamp = getattr(item, "datetime", None)
    if stamp is not None:
        return stamp.date().isoformat()
    props = getattr(item, "properties", {}) or {}
    raw = props.get("datetime")
    if raw:
        return str(raw)[:10]
    return f"undated:{getattr(item, 'id', id(item))}"


def _cloud(item) -> float:
    return item.properties.get("eo:cloud_cover", 100.0)


def search_scenes(
    catalog: Client,
    bbox: tuple[float, float, float, float],
    year: int,
    max_cloud: float = 10.0,
    season: tuple[str, str] = DEFAULT_SEASON,
    max_scenes: int = 6,
) -> list:
    """Choose which scenes to use for one year.

    I select whole days instead of single scenes. If the area covers more
    than one tile, all tiles need the same dates. In an older version each
    tile picked its own least cloudy scenes, and the delta image had
    rectangular blocks at the tile borders (easy to see over the lake).

    How it works:
    1. Find days where every tile has a scene. Take the least cloudy ones,
       up to max_scenes. If there is at least one such day, only these days
       are used, even if that's fewer than max_scenes.
    2. If there is no such day at all (e.g. the area is between two
       orbits), each tile gets up to max_scenes days of its own. Then the
       whole area is covered, but there can be visible borders.

    For an area with only one tile, this simply returns the max_scenes
    least cloudy scenes.
    """
    start, end = season
    search = catalog.search(
        collections=[COLLECTION],
        bbox=list(bbox),
        datetime=f"{year}-{start}/{year}-{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())

    per_day: dict[str, list] = {}
    for item in items:
        per_day.setdefault(scene_day(item), []).append(item)
    tiles = {scene_tile(item) for item in items}

    def day_tiles(day: str) -> set[str]:
        return {scene_tile(it) for it in per_day[day]}

    def mean_cloud(day: str) -> float:
        group = per_day[day]
        return sum(_cloud(it) for it in group) / len(group)

    full_days = sorted((d for d in per_day if day_tiles(d) == tiles), key=mean_cloud)
    if full_days:
        chosen = full_days[:max_scenes]
        if len(chosen) < max_scenes:
            LOGGER.info(
                "Only %d day(s) in %d cover all %d tiles; using just those",
                len(chosen),
                year,
                len(tiles),
            )
    else:
        # no day covers all tiles, so pick days per tile instead
        used = dict.fromkeys(tiles, 0)
        chosen = []
        ranked = sorted(per_day, key=lambda d: (-len(day_tiles(d)), mean_cloud(d)))
        for day in ranked:
            if any(used[t] < max_scenes for t in day_tiles(day)):
                chosen.append(day)
                for tile in day_tiles(day):
                    used[tile] += 1

    selected = [it for day in chosen for it in per_day[day]]
    selected.sort(key=_cloud)

    LOGGER.info(
        "Found %d candidate scenes for %d across %d tile(s); kept %d from %d day(s)",
        len(items),
        year,
        len(tiles),
        len(selected),
        len({scene_day(it) for it in selected}),
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
    """Build the composite for one year: search, load, correct, mask, median."""
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

    # Processing baseline per day. Needed for the offset check, because some
    # older scenes were reprocessed by ESA and have the offset too.
    baselines: dict[str, str] = {}
    for item in items:
        value = item.properties.get("s2:processing_baseline")
        if value is not None:
            baselines.setdefault(scene_day(item), str(value))

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
        offset = needs_boa_offset(acquired, baselines.get(acquired))

        a = to_reflectance(scene[spec.band_a].values, apply_offset=offset)
        b = to_reflectance(scene[spec.band_b].values, apply_offset=offset)
        mask = scl_valid_mask(scene["SCL"].values)

        stack.append(normalized_difference(a, b, valid_mask=mask))
        dates.append(acquired)

    cube = np.stack(stack)
    del stack  # free memory, cube already has a copy
    composite = cube[0] if len(cube) == 1 else median_composite(cube)

    return SeasonComposite(
        year=year,
        index=spec,
        values=composite,
        geobox=geobox,
        dates=sorted(dates),
        scene_count=len(cube),
        observations=observation_count(cube),
        noise=robust_scale(cube),
        cloud_cover=[
            float(it.properties.get("eo:cloud_cover", float("nan"))) for it in items
        ],
    )


# old name, still used in some places
load_year_composite = load_season_composite
