"""GeoTIFF export: the georeferencing has to survive the round trip."""

import numpy as np
import pytest
import rasterio

from data_access import build_geobox
from export import geotiff_bytes, write_geotiff

BBOX = (15.35, 47.01, 15.50, 47.12)


@pytest.fixture
def geobox():
    return build_geobox(BBOX, resolution=60.0)


@pytest.fixture
def delta(geobox):
    rng = np.random.default_rng(0)
    arr = rng.normal(0, 0.2, size=(geobox.shape.y, geobox.shape.x))
    arr[:4, :4] = np.nan  # a cloud gap
    return arr


def test_written_raster_keeps_crs_and_bounds(tmp_path, geobox, delta):
    path = tmp_path / "delta.tif"
    write_geotiff(path, delta, geobox)

    with rasterio.open(path) as src:
        assert src.crs.to_string() == str(geobox.crs)
        assert (src.width, src.height) == (geobox.shape.x, geobox.shape.y)
        assert np.allclose(src.bounds, tuple(geobox.extent.boundingbox), atol=1e-6)


def test_values_survive_the_round_trip(tmp_path, geobox, delta):
    path = tmp_path / "delta.tif"
    write_geotiff(path, delta, geobox)

    with rasterio.open(path) as src:
        read_back = src.read(1)

    # float32 storage, so compare at float32 precision.
    np.testing.assert_allclose(read_back, delta, atol=1e-6, equal_nan=True)


def test_masked_pixels_stay_nodata_not_zero(tmp_path, geobox, delta):
    """0 is a real index value; a cloud gap must not become one."""
    path = tmp_path / "delta.tif"
    write_geotiff(path, delta, geobox)

    with rasterio.open(path) as src:
        assert np.isnan(src.nodata)
        assert np.isnan(src.read(1)[0, 0])


def test_metadata_and_band_description_are_written(tmp_path, geobox, delta):
    path = tmp_path / "delta.tif"
    write_geotiff(
        path,
        delta,
        geobox,
        band_description="NDVI delta",
        metadata={"index": "NDVI", "baseline_year": 2021},
    )

    with rasterio.open(path) as src:
        assert src.descriptions[0] == "NDVI delta"
        tags = src.tags()
        assert tags["index"] == "NDVI"
        assert tags["baseline_year"] == "2021"


def test_shape_mismatch_is_refused(tmp_path, geobox):
    with pytest.raises(ValueError, match="does not match the GeoBox"):
        write_geotiff(tmp_path / "bad.tif", np.zeros((3, 3)), geobox)


def test_in_memory_export_is_a_valid_tiff(geobox, delta):
    payload = geotiff_bytes(delta, geobox, metadata={"index": "NBR"})
    assert payload[:4] in (b"II*\x00", b"MM\x00*")  # little/big-endian TIFF magic

    with rasterio.MemoryFile(payload) as memfile, memfile.open() as src:
        assert src.crs.to_string() == str(geobox.crs)
        assert src.tags()["index"] == "NBR"
        np.testing.assert_allclose(src.read(1), delta, atol=1e-6, equal_nan=True)


def test_in_memory_export_refuses_a_mismatched_array(geobox):
    with pytest.raises(ValueError, match="does not match the GeoBox"):
        geotiff_bytes(np.zeros((2, 2)), geobox)


# --- Multi-band export ---------------------------------------------------


def test_delta_and_observations_travel_together(tmp_path, geobox, delta):
    """A QGIS reader must be able to tell a thin pixel from a solid one."""
    observations = np.full_like(delta, 4.0)
    observations[:10, :] = 1.0
    path = tmp_path / "delta.tif"

    write_geotiff(
        path,
        [delta, observations],
        geobox,
        band_description=["NDVI delta", "observations"],
    )

    with rasterio.open(path) as src:
        assert src.count == 2
        assert src.descriptions == ("NDVI delta", "observations")
        np.testing.assert_allclose(src.read(2), observations, atol=1e-6)


def test_a_single_array_still_writes_one_band(tmp_path, geobox, delta):
    path = tmp_path / "one.tif"
    write_geotiff(path, delta, geobox, band_description="NDVI delta")
    with rasterio.open(path) as src:
        assert src.count == 1


def test_in_memory_export_handles_several_bands(geobox, delta):
    payload = geotiff_bytes(
        [delta, np.full_like(delta, 3.0)], geobox, ["delta", "observations"]
    )
    with rasterio.MemoryFile(payload) as memfile, memfile.open() as src:
        assert src.count == 2
        assert src.read(2)[0, 0] == pytest.approx(3.0)


def test_a_mismatched_band_is_refused_before_writing(tmp_path, geobox, delta):
    with pytest.raises(ValueError, match="does not match the GeoBox"):
        write_geotiff(tmp_path / "bad.tif", [delta, np.zeros((3, 3))], geobox)
