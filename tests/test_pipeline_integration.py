"""End-to-end test of load_year_composite with a stubbed STAC/odc layer.

This is the regression test for the defect that motivated the rewrite: the
same unchanged ground target, imaged in 2021 and in 2024, must produce a
delta of ~0. Without the Baseline 04.00 offset correction it produces a
large, spatially coherent shift that reads as continent-scale vegetation
change.
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

# One synthetic landscape: dense canopy, sparse field, bare soil, water.
RED_DN = np.tile(np.array([600.0, 1200.0, 2200.0, 900.0]), (SHAPE[0], SHAPE[1] // 4))
NIR_DN = np.tile(np.array([4000.0, 2600.0, 2400.0, 500.0]), (SHAPE[0], SHAPE[1] // 4))


class _Item:
    def __init__(self, cloud=1.0):
        self.properties = {"eo:cloud_cover": cloud}


class _Catalog:
    def search(self, **kwargs):
        return type("S", (), {"items": lambda self_: iter([_Item(), _Item(2.0)])})()


def _fake_dataset(dates, offset_applied, scl_value=4):
    """Build what odc.stac.load would return for the given acquisition dates.

    ``offset_applied`` mimics ESA storing reflectance +1000 DN from Baseline
    04.00 onwards, i.e. identical ground truth encoded differently.
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
    """Route odc.stac.load to a canned dataset chosen by the requested year."""
    calls = {}

    def fake_load(items, **kwargs):
        year = calls["year"]
        dates = [f"{year}-07-05", f"{year}-08-12"]
        # 2022-01-25 onwards the DN carry the +1000 encoding.
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
    """The core regression: same ground, different encoding, no false change."""
    baseline = _composite(stub_loader, 2021)
    comparison = _composite(stub_loader, 2024)

    delta = calculate_ndvi_delta(baseline.ndvi, comparison.ndvi)
    np.testing.assert_allclose(delta, 0.0, atol=1e-12)

    stats = change_statistics(delta)
    assert stats["loss_fraction"] == 0.0
    assert stats["gain_fraction"] == 0.0


def test_skipping_the_offset_would_have_faked_a_large_change():
    """Documents the magnitude of the bug the correction removes."""
    from ndvi_core import calculate_ndvi, to_reflectance

    correct_2021 = calculate_ndvi(
        to_reflectance(RED_DN, False), to_reflectance(NIR_DN, False)
    )
    naive_2024 = calculate_ndvi(
        to_reflectance(RED_DN + 1000, False), to_reflectance(NIR_DN + 1000, False)
    )
    bogus_delta = calculate_ndvi_delta(correct_2021, naive_2024)

    # A whole-scene bias far above the 0.1 "real change" threshold.
    assert np.abs(bogus_delta).max() > 0.1
    assert change_statistics(bogus_delta)["loss_fraction"] > 0.4


def test_composite_metadata_is_reported(stub_loader):
    composite = _composite(stub_loader, 2024)
    assert composite.year == 2024
    assert composite.scene_count == 2
    assert composite.dates == ["2024-07-05", "2024-08-12"]
    assert "2 scenes" in composite.date_range


def test_composite_lands_on_the_requested_grid(stub_loader):
    composite = _composite(stub_loader, 2021)
    assert composite.ndvi.shape == SHAPE


def test_fully_clouded_season_masks_everything(stub_loader):
    """SCL 9 = high-probability cloud: nothing should survive."""
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
    """SCL 6 is water; it must survive so shrinking lakes stay detectable."""
    composite = _composite(stub_loader, 2024, scl=6)
    assert np.isfinite(composite.ndvi).all()
    # The water column of the synthetic landscape must read negative.
    assert composite.ndvi[0, 3] < 0
