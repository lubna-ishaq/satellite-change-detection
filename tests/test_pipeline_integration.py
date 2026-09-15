"""Tests for load_year_composite with fake STAC and odc data.

Main test: the same unchanged ground in 2021 and 2024 must give a delta of
about 0. Without the offset correction there is a big fake change over the
whole image (that was the bug in my first version).
"""

import numpy as np
import odc.stac
import pytest
import xarray as xr

import data_access
from data_access import build_geobox, load_year_composite
from ndvi_core import calculate_ndvi_delta, change_statistics

GRAZ = data_access.AOI_PRESETS["Graz, Austria"]
SHAPE = (40, 48)  # width divisible by the 4-class landscape pattern

# fake landscape: forest, field, bare soil, water
RED_DN = np.tile(np.array([600.0, 1200.0, 2200.0, 900.0]), (SHAPE[0], SHAPE[1] // 4))
NIR_DN = np.tile(np.array([4000.0, 2600.0, 2400.0, 500.0]), (SHAPE[0], SHAPE[1] // 4))


class _Item:
    def __init__(self, cloud=1.0):
        self.properties = {"eo:cloud_cover": cloud}


class _Catalog:
    def search(self, **kwargs):
        return type("S", (), {"items": lambda self_: iter([_Item(), _Item(2.0)])})()


def _fake_dataset(dates, offset_applied, scl_value=4):
    """Fake version of what odc.stac.load returns.

    offset_applied=True stores the same values +1000, like ESA does since
    baseline 04.00.
    """
    shift = 1000.0 if offset_applied else 0.0
    red = np.stack([RED_DN + shift] * len(dates))
    nir = np.stack([NIR_DN + shift] * len(dates))
    scl = np.full((len(dates), *SHAPE), scl_value, dtype="uint8")
    coords = {"time": np.array(dates, dtype="datetime64[ns]")}
    dims = ("time", "y", "x")
    return xr.Dataset(
        {"B04": (dims, red), "B08": (dims, nir), "SCL": (dims, scl)}, coords=coords
    )


@pytest.fixture
def stub_loader(monkeypatch):
    """Replace odc.stac.load with fake data for the requested year."""
    calls = {}

    def fake_load(items, **kwargs):
        year = calls["year"]
        dates = [f"{year}-07-05", f"{year}-08-12"]
        # from 2022-01-25 the values have the +1000 offset
        return _fake_dataset(dates, offset_applied=year >= 2022, scl_value=calls["scl"])

    monkeypatch.setattr(odc.stac, "load", fake_load)
    return calls


def _composite(stub, year, scl=4):
    stub["year"] = year
    stub["scl"] = scl
    return load_year_composite(
        _Catalog(), GRAZ, year, build_geobox(GRAZ, 20.0), max_scenes=2
    )


def test_unchanged_target_across_the_baseline_cutover_yields_zero_delta(stub_loader):
    """Same ground, different encoding: no change should be detected."""
    baseline = _composite(stub_loader, 2021)
    comparison = _composite(stub_loader, 2024)

    delta = calculate_ndvi_delta(baseline.ndvi, comparison.ndvi)
    np.testing.assert_allclose(delta, 0.0, atol=1e-12)

    stats = change_statistics(delta)
    assert stats["loss_fraction"] == 0.0
    assert stats["gain_fraction"] == 0.0


def test_skipping_the_offset_would_have_faked_a_large_change():
    """Shows how big the error is without the offset correction."""
    from ndvi_core import calculate_ndvi, to_reflectance

    correct_2021 = calculate_ndvi(
        to_reflectance(RED_DN, False), to_reflectance(NIR_DN, False)
    )
    naive_2024 = calculate_ndvi(
        to_reflectance(RED_DN + 1000, False), to_reflectance(NIR_DN + 1000, False)
    )
    bogus_delta = calculate_ndvi_delta(correct_2021, naive_2024)

    # the whole image shifts by more than the 0.1 threshold
    assert np.abs(bogus_delta).max() > 0.1
    assert change_statistics(bogus_delta)["loss_fraction"] > 0.4


def test_composite_metadata_is_reported(stub_loader):
    composite = _composite(stub_loader, 2024)
    assert composite.year == 2024
    assert composite.scene_count == 2
    assert composite.dates == ["2024-07-05", "2024-08-12"]
    assert composite.date_range == "2024-07-05 … 2024-08-12, 2 scenes"


def test_composite_lands_on_the_requested_grid(stub_loader):
    composite = _composite(stub_loader, 2021)
    assert composite.ndvi.shape == SHAPE


def test_fully_clouded_season_masks_everything(stub_loader):
    """SCL 9 = cloud (high probability), so everything is masked."""
    composite = _composite(stub_loader, 2024, scl=9)
    assert np.isnan(composite.ndvi).all()
    assert change_statistics(composite.ndvi - composite.ndvi)["valid_pixels"] == 0


def test_no_scenes_raises_lookup_error(monkeypatch):
    class _Empty:
        def search(self, **kwargs):
            return type("S", (), {"items": lambda self_: iter([])})()

    with pytest.raises(LookupError, match="No Sentinel-2 scene"):
        load_year_composite(_Empty(), GRAZ, 2024, build_geobox(GRAZ, 20.0))


def test_water_is_not_masked_away(stub_loader):
    """SCL 6 = water. It must be kept, otherwise lake changes can't be detected."""
    composite = _composite(stub_loader, 2024, scl=6)
    assert np.isfinite(composite.ndvi).all()
    # water column must have negative NDVI
    assert composite.ndvi[0, 3] < 0


def test_reprocessed_old_scenes_are_corrected_by_their_baseline(monkeypatch):
    """A 2019 scene reprocessed with baseline 05.00 carries the offset too."""
    import datetime as dt

    class _Reprocessed:
        def __init__(self, day):
            self.id = day
            self.datetime = dt.datetime.fromisoformat(f"{day}T10:00:00+00:00")
            self.properties = {"eo:cloud_cover": 1.0, "s2:processing_baseline": "05.00"}

    class _Cat:
        def search(self, **kwargs):
            items = [_Reprocessed("2019-07-05"), _Reprocessed("2019-08-12")]
            return type("S", (), {"items": lambda self_: iter(items)})()

    monkeypatch.setattr(
        odc.stac,
        "load",
        lambda items, **kw: _fake_dataset(["2019-07-05", "2019-08-12"], True),
    )
    reprocessed = load_year_composite(_Cat(), GRAZ, 2019, build_geobox(GRAZ, 20.0))

    monkeypatch.setattr(
        odc.stac,
        "load",
        lambda items, **kw: _fake_dataset(["2019-07-05", "2019-08-12"], False),
    )
    original = load_year_composite(_Catalog(), GRAZ, 2019, build_geobox(GRAZ, 20.0))

    np.testing.assert_allclose(reprocessed.values, original.values, atol=1e-12)
