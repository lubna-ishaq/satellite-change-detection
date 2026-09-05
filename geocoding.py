"""Place-name search, so a user does not have to type four coordinates.

Uses the OpenStreetMap Nominatim service. Nominatim's usage policy requires
an identifying User-Agent and at most one request per second, and forbids
bulk or automated harvesting — this module is for interactive lookups only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

LOGGER = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

#: Nominatim asks every client to identify itself with a contact or project URL.
USER_AGENT = (
    "satellite-change-detection/1.0 "
    "(+https://github.com/lubna-ishaq/satellite-change-detection)"
)

#: Smallest analysis window in degrees. A village returns a bounding box a few
#: hundred metres across, which is too small to see change in; widen it so the
#: result is always a usable scene footprint.
MIN_SPAN_DEG = 0.05

#: Largest span accepted. A country-sized box would blow the pixel budget and
#: take many minutes to download.
MAX_SPAN_DEG = 1.5


class GeocodingError(RuntimeError):
    """The geocoding service could not be reached or returned nothing usable."""


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
    """Grow a too-small box and shrink a too-large one, keeping its centre."""
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
    """Look up ``query`` and return candidate places with usable bounding boxes.

    Raises :class:`GeocodingError` if the service is unreachable; returns an
    empty list if it simply found nothing.
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
            # Nominatim order is [south, north, west, east], all as strings.
            south, north, west, east = (float(v) for v in raw)
        except (TypeError, ValueError):
            continue
        name = entry.get("display_name") or query
        places.append(Place(name=name, bbox=_clamp_bbox(west, south, east, north)))

    LOGGER.info("Geocoded %r to %d usable places", query, len(places))
    return places
