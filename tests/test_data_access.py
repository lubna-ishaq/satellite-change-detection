"""Tests for the grid and scene selection. No network needed."""

import datetime as dt

import pytest

from data_access import AOI_PRESETS, build_geobox, search_scenes, utm_epsg_for_bbox

GRAZ = AOI_PRESETS["Graz, Austria"]


@pytest.mark.parametrize(
    "bbox, expected",
    [
        (GRAZ, "EPSG:32633"),  # Austria, UTM 33N
        ((-6.50, 36.90, -6.20, 37.15), "EPSG:32629"),  # Spain, UTM 29N
        ((-58.5, -34.7, -58.3, -34.5), "EPSG:32721"),  # Argentina, UTM 21S
    ],
)
def test_utm_zone_selection(bbox, expected):
    assert utm_epsg_for_bbox(bbox) == expected


def test_geobox_is_deterministic_across_years():
    """Both years must get exactly the same grid."""
    assert build_geobox(GRAZ, 20.0) == build_geobox(GRAZ, 20.0)


def test_geobox_is_metric_not_degrees():
    geobox = build_geobox(GRAZ, resolution=20.0)
    assert geobox.crs.units[0] in ("metre", "m")
    assert abs(geobox.resolution.x) == pytest.approx(20.0)


def test_geobox_shape_matches_ground_extent():
    """~11.4 km x ~12.3 km at 20 m should be roughly 570 x 615 pixels."""
    geobox = build_geobox(GRAZ, resolution=20.0)
    assert 550 < geobox.shape.x < 600
    assert 590 < geobox.shape.y < 640


def test_finer_resolution_yields_more_pixels():
    coarse = build_geobox(GRAZ, resolution=60.0)
    fine = build_geobox(GRAZ, resolution=10.0)
    assert fine.shape.x > coarse.shape.x * 5


def test_presets_are_well_formed():
    for name, (west, south, east, north) in AOI_PRESETS.items():
        assert west < east, name
        assert south < north, name
        assert -180 <= west <= 180 and -90 <= south <= 90, name


# Scene selection (stubbed catalog, no network)


class _FakeItem:
    def __init__(self, cloud):
        self.properties = {"eo:cloud_cover": cloud}


class _FakeSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class _FakeCatalog:
    """Saves the search parameters so the tests can check them."""

    def __init__(self, items):
        self._items = items
        self.last_kwargs = None

    def search(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeSearch(self._items)


def test_search_scenes_sorts_by_cloud_cover():
    """The first version just used items[0], whatever the API returned first."""
    catalog = _FakeCatalog([_FakeItem(9.5), _FakeItem(0.4), _FakeItem(4.2)])
    result = search_scenes(catalog, GRAZ, 2024)
    assert [it.properties["eo:cloud_cover"] for it in result] == [0.4, 4.2, 9.5]


def test_search_scenes_respects_max_scenes_within_one_tile():
    """All items are on one tile here, so max_scenes is the total."""
    catalog = _FakeCatalog([_TiledItem(c, "33UWP") for c in (5, 1, 3, 2, 4)])
    assert len(search_scenes(catalog, GRAZ, 2024, max_scenes=2)) == 2


def test_search_scenes_builds_summer_window_for_the_requested_year():
    catalog = _FakeCatalog([])
    search_scenes(catalog, GRAZ, 2021, max_cloud=15.0)
    assert catalog.last_kwargs["datetime"] == "2021-06-01/2021-08-31"
    assert catalog.last_kwargs["query"] == {"eo:cloud_cover": {"lt": 15.0}}


def test_search_scenes_handles_missing_cloud_metadata():
    bad = _FakeItem(2.0)
    bad.properties = {}  # some providers omit eo:cloud_cover
    result = search_scenes(_FakeCatalog([bad, _FakeItem(1.0)]), GRAZ, 2024)
    assert result[0].properties.get("eo:cloud_cover") == 1.0


# Bounding-box validation and the pixel budget


@pytest.mark.parametrize(
    "bbox, message",
    [
        ((15.5, 47.0, 15.3, 47.1), "West"),
        ((15.3, 47.2, 15.5, 47.1), "South"),
        ((-190.0, 47.0, 15.5, 47.1), "Longitude"),
        ((15.3, -95.0, 15.5, 47.1), "Latitude"),
    ],
)
def test_malformed_boxes_are_rejected_before_any_network_call(bbox, message):
    from data_access import validate_bbox

    with pytest.raises(ValueError, match=message):
        validate_bbox(bbox)


def test_a_sane_box_passes_validation():
    from data_access import validate_bbox

    assert validate_bbox(GRAZ) is None


def test_pixel_budget_rejects_a_country_sized_request():
    """A huge area at 10 m would need far too much memory."""
    from data_access import AreaTooLargeError
    from data_access import build_geobox as bg

    with pytest.raises(AreaTooLargeError, match="exceeds"):
        bg((5.0, 45.0, 15.0, 55.0), resolution=10.0)


def test_pixel_budget_message_suggests_a_way_out():
    from data_access import AreaTooLargeError
    from data_access import build_geobox as bg

    with pytest.raises(AreaTooLargeError, match="coarser --resolution"):
        bg((5.0, 45.0, 15.0, 55.0), resolution=10.0)


def test_the_same_area_fits_at_a_coarser_resolution():
    from data_access import build_geobox as bg

    geobox = bg((5.0, 45.0, 15.0, 55.0), resolution=200.0)
    assert geobox.shape.x * geobox.shape.y > 0


# Season handling


def test_season_window_is_applied_to_both_bounds():
    catalog = _FakeCatalog([])
    search_scenes(catalog, GRAZ, 2023, season=("12-01", "12-31"))
    assert catalog.last_kwargs["datetime"] == "2023-12-01/2023-12-31"


# Tile coverage


class _TiledItem:
    def __init__(self, cloud, tile, name=""):
        self.id = name or f"{tile}-{cloud}"
        self.properties = {"eo:cloud_cover": cloud, "s2:mgrs_tile": tile}


def test_scenes_are_budgeted_per_tile_not_globally():
    """Without shared days, every tile still needs its own scenes.
    Otherwise a simple sort could pick scenes from only one tile."""
    from data_access import search_scenes as search

    items = [
        _TiledItem(1.0, "33UWP"),
        _TiledItem(2.0, "33UWP"),
        _TiledItem(3.0, "33UWP"),
        _TiledItem(9.0, "33UXP"),  # the only scene for the second tile
    ]
    result = search(_FakeCatalog(items), GRAZ, 2024, max_scenes=2)
    tiles = {it.properties["s2:mgrs_tile"] for it in result}
    assert tiles == {"33UWP", "33UXP"}, "every tile must be represented"
    assert len(result) == 3  # 2 from the first tile, 1 from the second


def test_within_a_tile_the_cleanest_scenes_win():
    from data_access import search_scenes as search

    items = [_TiledItem(c, "33UWP") for c in (8.0, 1.0, 5.0, 2.0)]
    result = search(_FakeCatalog(items), GRAZ, 2024, max_scenes=2)
    assert [it.properties["eo:cloud_cover"] for it in result] == [1.0, 2.0]


def test_a_single_tile_area_behaves_as_before():
    from data_access import search_scenes as search

    items = [_TiledItem(c, "33UWP") for c in (5.0, 1.0, 3.0)]
    result = search(_FakeCatalog(items), GRAZ, 2024, max_scenes=2)
    assert len(result) == 2


def test_results_stay_sorted_cloudiest_last():
    from data_access import search_scenes as search

    items = [_TiledItem(9.0, "A"), _TiledItem(1.0, "B"), _TiledItem(4.0, "A")]
    result = search(_FakeCatalog(items), GRAZ, 2024, max_scenes=5)
    clouds = [it.properties["eo:cloud_cover"] for it in result]
    assert clouds == sorted(clouds)


@pytest.mark.parametrize(
    "props, expected",
    [
        ({"s2:mgrs_tile": "33UWP"}, "33UWP"),
        ({"grid:code": "MGRS-33UWP"}, "MGRS-33UWP"),
        ({"sentinel:grid_square": "WP"}, "WP"),
    ],
)
def test_tile_key_is_read_from_whichever_property_exists(props, expected):
    from data_access import scene_tile

    item = _TiledItem(1.0, "ignored")
    item.properties = props
    assert scene_tile(item) == expected


def test_items_without_tile_metadata_are_never_grouped_together():
    """Scenes without a tile ID must not be grouped together."""
    from data_access import scene_tile

    a, b = _TiledItem(1.0, "x", "a"), _TiledItem(1.0, "x", "b")
    a.properties, b.properties = {}, {}
    assert scene_tile(a) != scene_tile(b)


# Day-based selection (keeps tiles on the same dates)


class _DatedItem:
    def __init__(self, cloud, tile, day, baseline=None):
        self.id = f"{tile}-{day}"
        self.datetime = dt.datetime.fromisoformat(f"{day}T10:00:00+00:00")
        self.properties = {"eo:cloud_cover": cloud, "s2:mgrs_tile": tile}
        if baseline is not None:
            self.properties["s2:processing_baseline"] = baseline


def _days(items):
    return sorted({it.datetime.date().isoformat() for it in items})


def test_tiles_share_acquisition_days_when_possible():
    """If every tile picks its own days, the delta shows blocks at tile borders."""
    items = [
        # very clear days for 33UWP, but 33UXP has no scene on them
        _DatedItem(0.1, "33UWP", "2024-07-01"),
        _DatedItem(0.2, "33UWP", "2024-07-04"),
        # days with both tiles
        _DatedItem(3.0, "33UWP", "2024-07-10"),
        _DatedItem(4.0, "33UXP", "2024-07-10"),
        _DatedItem(2.0, "33UWP", "2024-07-20"),
        _DatedItem(5.0, "33UXP", "2024-07-20"),
    ]
    result = search_scenes(_FakeCatalog(items), GRAZ, 2024, max_scenes=2)
    assert _days(result) == ["2024-07-10", "2024-07-20"]
    assert {it.properties["s2:mgrs_tile"] for it in result} == {"33UWP", "33UXP"}


def test_single_tile_days_are_not_mixed_in_when_shared_days_exist():
    """Better fewer scenes than different dates per tile."""
    items = [
        _DatedItem(3.0, "A", "2024-07-10"),
        _DatedItem(3.0, "B", "2024-07-10"),
        _DatedItem(0.1, "A", "2024-07-01"),
        _DatedItem(0.2, "B", "2024-07-02"),
    ]
    result = search_scenes(_FakeCatalog(items), GRAZ, 2024, max_scenes=4)
    assert _days(result) == ["2024-07-10"]


def test_a_chosen_day_brings_all_of_its_tiles():
    items = [
        _DatedItem(1.0, "A", "2024-07-01"),
        _DatedItem(9.0, "B", "2024-07-01"),
    ]
    result = search_scenes(_FakeCatalog(items), GRAZ, 2024, max_scenes=1)
    assert len(result) == 2


def test_a_tile_without_shared_days_still_gets_its_own():
    items = [
        _DatedItem(1.0, "A", "2024-07-01"),
        _DatedItem(1.5, "A", "2024-07-02"),
        _DatedItem(8.0, "B", "2024-07-15"),
    ]
    result = search_scenes(_FakeCatalog(items), GRAZ, 2024, max_scenes=1)
    assert {it.properties["s2:mgrs_tile"] for it in result} == {"A", "B"}
    assert _days(result) == ["2024-07-01", "2024-07-15"]


def test_cleaner_shared_day_wins_among_equal_coverage():
    items = [
        _DatedItem(6.0, "A", "2024-07-01"),
        _DatedItem(6.0, "B", "2024-07-01"),
        _DatedItem(1.0, "A", "2024-07-05"),
        _DatedItem(2.0, "B", "2024-07-05"),
    ]
    result = search_scenes(_FakeCatalog(items), GRAZ, 2024, max_scenes=1)
    assert _days(result) == ["2024-07-05"]


@pytest.mark.parametrize(
    "item, expected",
    [
        (_DatedItem(1.0, "A", "2024-07-01"), "2024-07-01"),
        (
            type("I", (), {"properties": {"datetime": "2021-08-13T10:00:00Z"}})(),
            "2021-08-13",
        ),
    ],
)
def test_scene_day_reads_the_acquisition_date(item, expected):
    from data_access import scene_day

    assert scene_day(item) == expected


def test_undated_items_are_never_the_same_day():
    from data_access import scene_day

    a, b = _TiledItem(1.0, "x", "a"), _TiledItem(1.0, "x", "b")
    assert scene_day(a) != scene_day(b)


# Memory estimate


def test_memory_estimate_grows_with_scenes():
    from data_access import estimate_peak_memory

    geobox = build_geobox(GRAZ, 20.0)
    assert estimate_peak_memory(geobox, 6) == 6 * estimate_peak_memory(geobox, 1)


def test_the_pixel_cap_alone_does_not_fit_a_hosted_instance():
    """40 MP with 6 scenes is several GB, that's why the app shows a warning."""
    from data_access import (
        BYTES_PER_PIXEL_SCENE,
        HOSTED_MEMORY_BYTES,
        MAX_PIXELS,
    )

    assert MAX_PIXELS * 6 * BYTES_PER_PIXEL_SCENE > HOSTED_MEMORY_BYTES
