"""Search for a place by name instead of typing coordinates.

Uses the OpenStreetMap Nominatim API. Their usage policy asks for a
User-Agent that identifies the project and at most one request per second,
so this is only meant for single searches from the app, not bulk requests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

LOGGER = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Nominatim wants every client to send a name and a project URL.
USER_AGENT = (
    "satellite-change-detection/1.0 "
    "(+https://github.com/lubna-ishaq/satellite-change-detection)"
)

# Minimum box size in degrees. A small village gives a box of a few hundred
# metres, which is too small to see anything, so it gets enlarged.
MIN_SPAN_DEG = 0.05

# Maximum box size in degrees. Anything bigger (e.g. a whole country) would
# be too many pixels and take forever to download.
MAX_SPAN_DEG = 1.5


class GeocodingError(RuntimeError):
    """Raised when the search service is not reachable or returns bad data."""


@dataclass(frozen=True)
class Place:
    name: str
    bbox: tuple[float, float, float, float]  # west, south, east, north

    @property
    def span(self) -> tuple[float, float]:
        west, south, east, north = self.bbox
        return (east - west, north - south)


def _clamp_bbox(
    west: float, south: float, east: float, north: float
) -> tuple[float, float, float, float]:
    """Make a box bigger or smaller if needed, keeping the same centre."""
    centre_lon = (west + east) / 2.0
    centre_lat = (south + north) / 2.0

    span_lon = min(max(east - west, MIN_SPAN_DEG), MAX_SPAN_DEG)
    span_lat = min(max(north - south, MIN_SPAN_DEG), MAX_SPAN_DEG)

    return (
        round(centre_lon - span_lon / 2, 6),
        round(max(centre_lat - span_lat / 2, -90.0), 6),
        round(centre_lon + span_lon / 2, 6),
        round(min(centre_lat + span_lat / 2, 90.0), 6),
    )


def search_place(query: str, limit: int = 5, timeout: float = 10.0) -> list[Place]:
    """Search for a place and return matches with a usable bounding box.

    Raises GeocodingError if the service can't be reached.
    Returns an empty list if nothing was found.
    """
    query = query.strip()
    if not query:
        return []

    try:
        response = requests.get(
            NOMINATIM_URL,
            params={"q": query, "format": "jsonv2", "limit": limit},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise GeocodingError(f"Could not reach the place-name service: {exc}") from exc
    except ValueError as exc:
        raise GeocodingError("The place-name service returned invalid data") from exc

    places: list[Place] = []
    for entry in payload:
        raw = entry.get("boundingbox")
        if not raw or len(raw) != 4:
            continue
        try:
            # Nominatim returns [south, north, west, east] as strings
            south, north, west, east = (float(v) for v in raw)
        except (TypeError, ValueError):
            continue
        name = entry.get("display_name") or query
        places.append(Place(name=name, bbox=_clamp_bbox(west, south, east, north)))

    LOGGER.info("Geocoded %r to %d usable places", query, len(places))
    return places
