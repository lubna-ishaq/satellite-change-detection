"""Geocoding: parsing and bbox shaping, with the network stubbed out."""

import pytest
import requests

import geocoding
from geocoding import (
    MAX_SPAN_DEG,
    MIN_SPAN_DEG,
    GeocodingError,
    _clamp_bbox,
    search_place,
)


class _Response:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _stub(monkeypatch, payload, capture=None):
    def fake_get(url, params=None, headers=None, timeout=None):
        if capture is not None:
            capture.update({"url": url, "params": params, "headers": headers})
        if isinstance(payload, Exception):
            raise payload
        return _Response(payload)

    monkeypatch.setattr(geocoding.requests, "get", fake_get)


# --- bbox shaping --------------------------------------------------------


def test_a_tiny_place_is_widened_to_a_usable_window():
    """A village returns a few-hundred-metre box, too small to analyse."""
    west, south, east, north = _clamp_bbox(15.4400, 47.0700, 15.4405, 47.0705)
    assert east - west == pytest.approx(MIN_SPAN_DEG, abs=1e-6)
    assert north - south == pytest.approx(MIN_SPAN_DEG, abs=1e-6)


def test_a_huge_place_is_shrunk_to_the_budget():
    west, south, east, north = _clamp_bbox(-10.0, 35.0, 30.0, 60.0)
    assert east - west == pytest.approx(MAX_SPAN_DEG, abs=1e-6)
    assert north - south == pytest.approx(MAX_SPAN_DEG, abs=1e-6)


def test_clamping_keeps_the_centre():
    west, south, east, north = _clamp_bbox(15.40, 47.00, 15.41, 47.01)
    assert (west + east) / 2 == pytest.approx(15.405, abs=1e-6)
    assert (south + north) / 2 == pytest.approx(47.005, abs=1e-6)


def test_a_reasonable_box_is_left_alone():
    original = (15.35, 47.01, 15.55, 47.21)
    assert _clamp_bbox(*original) == pytest.approx(original, abs=1e-6)


def test_latitude_never_leaves_the_globe():
    _, south, _, north = _clamp_bbox(0.0, 89.99, 0.1, 89.999)
    assert south >= -90.0 and north <= 90.0


# --- request and parsing -------------------------------------------------


def test_result_order_is_converted_from_nominatim(monkeypatch):
    """Nominatim returns [south, north, west, east] as strings."""
    _stub(
        monkeypatch,
        [
            {
                "display_name": "Neusiedler See, Burgenland, Austria",
                "boundingbox": ["47.70", "47.90", "16.62", "16.85"],
            }
        ],
    )
    (place,) = search_place("Neusiedler See")
    west, south, east, north = place.bbox
    assert (west, south, east, north) == pytest.approx(
        (16.62, 47.70, 16.85, 47.90), abs=1e-6
    )
    assert place.name.startswith("Neusiedler See")


def test_entries_without_a_usable_bbox_are_skipped(monkeypatch):
    _stub(
        monkeypatch,
        [
            {"display_name": "no box"},
            {"display_name": "short", "boundingbox": ["1", "2"]},
            {"display_name": "bad numbers", "boundingbox": ["a", "b", "c", "d"]},
            {"display_name": "good", "boundingbox": ["47.0", "47.2", "15.3", "15.5"]},
        ],
    )
    places = search_place("mixed")
    assert [p.name for p in places] == ["good"]


def test_identifies_itself_as_nominatim_requires(monkeypatch):
    captured = {}
    _stub(monkeypatch, [], capture=captured)
    search_place("anywhere")
    assert "satellite-change-detection" in captured["headers"]["User-Agent"]
    assert captured["params"]["q"] == "anywhere"


def test_empty_query_makes_no_request(monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("no request should be made for an empty query")

    monkeypatch.setattr(geocoding.requests, "get", explode)
    assert search_place("   ") == []


def test_network_failure_becomes_a_clear_error(monkeypatch):
    _stub(monkeypatch, requests.ConnectionError("no route to host"))
    with pytest.raises(GeocodingError, match="Could not reach"):
        search_place("Graz")


def test_invalid_json_becomes_a_clear_error(monkeypatch):
    _stub(monkeypatch, ValueError("not json"))
    with pytest.raises(GeocodingError, match="invalid data"):
        search_place("Graz")


def test_no_results_is_not_an_error(monkeypatch):
    _stub(monkeypatch, [])
    assert search_place("qwertyuiop") == []
