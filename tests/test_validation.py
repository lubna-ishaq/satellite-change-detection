"""The validation harness itself, with the imagery stubbed out.

These tests check that a case is judged correctly given a delta — not that
the real events are detected. That is what `python validation.py` does, and
it needs network access.
"""

import numpy as np
import pytest

import validation
from data_access import MAX_PIXELS, SeasonComposite, build_geobox
from ndvi_core import INDICES, get_index
from validation import CASES, CASES_BY_KEY, format_result, run_case

# --- the case definitions themselves -------------------------------------


def test_cases_are_well_formed():
    for case in CASES:
        west, south, east, north = case.bbox
        assert west < east, case.key
        assert south < north, case.key
        assert case.index in INDICES, case.key
        assert case.baseline_year < case.comparison_year, case.key
        assert case.expect in {"decrease", "increase", "stable"}, case.key
        assert 0 < case.fraction < 1, case.key
        assert case.event.strip(), case.key


def test_case_keys_are_unique():
    keys = [c.key for c in CASES]
    assert len(keys) == len(set(keys))
    assert set(CASES_BY_KEY) == set(keys)


def test_cases_start_no_earlier_than_the_sentinel2_archive():
    """Sentinel-2 L2A on the Planetary Computer does not reach before 2015."""
    for case in CASES:
        assert case.baseline_year >= 2016, case.key


def test_every_case_fits_the_pixel_budget():
    for case in CASES:
        geobox = build_geobox(case.bbox, resolution=case.resolution)
        assert geobox.shape.x * geobox.shape.y <= MAX_PIXELS, case.key


def test_there_is_a_negative_control():
    """Without one, a systematic bias would pass every other case."""
    controls = [c for c in CASES if c.expect == "stable"]
    assert controls, "the suite needs at least one no-change control"


# --- judging -------------------------------------------------------------


def _stub_composites(monkeypatch, baseline_arr, comparison_arr):
    calls = {"n": 0}

    def fake_load(catalog, bbox, year, geobox, **kwargs):
        calls["n"] += 1
        values = baseline_arr if calls["n"] == 1 else comparison_arr
        return SeasonComposite(
            year=year,
            index=get_index(kwargs.get("index", "NDVI"))
            if isinstance(kwargs.get("index", "NDVI"), str)
            else kwargs["index"],
            values=values,
            geobox=geobox,
            dates=[f"{year}-07-01"],
            scene_count=1,
        )

    monkeypatch.setattr(validation, "load_season_composite", fake_load)
    monkeypatch.setattr(validation, "build_geobox", lambda *a, **k: None)
    return calls


@pytest.fixture
def decrease_case():
    return CASES_BY_KEY["camp-fire"]


@pytest.fixture
def control_case():
    return CASES_BY_KEY["sahara-control"]


def test_a_real_decrease_passes_a_decrease_case(monkeypatch, decrease_case):
    baseline = np.full((10, 10), 0.6)
    comparison = np.full((10, 10), 0.1)  # -0.5 everywhere
    _stub_composites(monkeypatch, baseline, comparison)

    result = run_case(decrease_case, catalog=object())
    assert result.passed
    assert "100.0% decreased" in result.detail


def test_no_change_fails_a_decrease_case(monkeypatch, decrease_case):
    flat = np.full((10, 10), 0.4)
    _stub_composites(monkeypatch, flat, flat.copy())

    result = run_case(decrease_case, catalog=object())
    assert not result.passed


def test_a_quiet_control_passes(monkeypatch, control_case):
    rng = np.random.default_rng(0)
    baseline = np.full((20, 20), 0.05)
    comparison = baseline + rng.normal(0, 0.01, size=baseline.shape)
    _stub_composites(monkeypatch, baseline, comparison)

    result = run_case(control_case, catalog=object())
    assert result.passed


def test_a_biased_control_fails(monkeypatch, control_case):
    """A residual calibration offset must be caught here."""
    baseline = np.full((20, 20), 0.05)
    comparison = baseline + 0.3  # whole-scene bias, exactly the old bug
    _stub_composites(monkeypatch, baseline, comparison)

    result = run_case(control_case, catalog=object())
    assert not result.passed
    assert "mean bias" in result.detail


def test_fully_masked_imagery_fails_rather_than_passing_silently(
    monkeypatch, control_case
):
    nan = np.full((10, 10), np.nan)
    _stub_composites(monkeypatch, nan, nan.copy())

    result = run_case(control_case, catalog=object())
    assert not result.passed
    assert "masked" in result.detail


def test_mostly_clouded_imagery_fails(monkeypatch, decrease_case):
    baseline = np.full((10, 10), 0.6)
    comparison = np.full((10, 10), 0.1)
    comparison[:, :8] = np.nan  # only 20% valid
    _stub_composites(monkeypatch, baseline, comparison)

    result = run_case(decrease_case, catalog=object())
    assert not result.passed
    assert "survived masking" in result.detail


def test_missing_imagery_is_reported_not_raised(monkeypatch, decrease_case):
    def raise_lookup(*args, **kwargs):
        raise LookupError("no scenes below 20% cloud")

    monkeypatch.setattr(validation, "load_season_composite", raise_lookup)
    monkeypatch.setattr(validation, "build_geobox", lambda *a, **k: None)

    result = run_case(decrease_case, catalog=object())
    assert not result.passed
    assert "no usable imagery" in result.detail


def test_format_result_labels_pass_and_fail(monkeypatch, decrease_case):
    baseline = np.full((10, 10), 0.6)
    _stub_composites(monkeypatch, baseline, np.full((10, 10), 0.1))
    text = format_result(run_case(decrease_case, catalog=object()))
    assert text.startswith("[PASS]")
    assert decrease_case.name in text


def test_list_mode_needs_no_network(capsys):
    assert validation.main(["--list"]) == 0
    out = capsys.readouterr().out
    for case in CASES:
        assert case.key in out


def test_scatter_without_drift_passes_the_control(monkeypatch, control_case):
    """Some scatter is tolerated; a shifted mean is not.

    The tolerance is small on purpose. An earlier version of this test
    injected enough noise to produce several percent of crossings, because
    the control then stood on irrigated farmland and several percent looked
    normal. On real dune field the measured figure is 0.0%.
    """
    rng = np.random.default_rng(1)
    baseline = np.full((200, 200), 0.04)
    comparison = baseline + rng.normal(0, 0.035, size=baseline.shape)
    _stub_composites(monkeypatch, baseline, comparison)

    result = run_case(control_case, catalog=object())
    stats = result.stats
    scatter = stats["loss_fraction"] + stats["gain_fraction"]
    assert 0 < scatter <= control_case.fraction, "some scatter, within bound"
    assert abs(stats["mean_delta"]) < control_case.max_bias, "but no drift"
    assert result.passed


def test_farmland_level_scatter_now_fails_the_control(monkeypatch, control_case):
    """Regression guard for the mis-sited control.

    Cropping change of the magnitude seen at the old East Uweinat location
    must fail rather than be waved through as desert noise.
    """
    rng = np.random.default_rng(2)
    baseline = np.full((200, 200), 0.04)
    comparison = baseline + rng.normal(0, 0.06, size=baseline.shape)
    _stub_composites(monkeypatch, baseline, comparison)

    result = run_case(control_case, catalog=object())
    assert result.stats["loss_fraction"] + result.stats["gain_fraction"] > 0.02
    assert not result.passed


def test_a_small_systematic_drift_still_fails_the_control(monkeypatch, control_case):
    """A bias far below the per-pixel threshold must still be caught."""
    baseline = np.full((100, 100), 0.04)
    comparison = baseline + 0.05  # well under the 0.1 change threshold
    _stub_composites(monkeypatch, baseline, comparison)

    result = run_case(control_case, catalog=object())
    assert not result.passed
    assert result.stats["loss_fraction"] + result.stats["gain_fraction"] == 0.0


# --- Region-restricted scoring ------------------------------------------


@pytest.fixture
def water_case():
    return CASES_BY_KEY["kakhovka"]


def test_water_cases_are_scored_over_baseline_water_only(water_case):
    assert water_case.baseline_above == 0.0, (
        "a draining reservoir must be scored over the water that was there"
    )


def test_padding_the_box_no_longer_changes_the_score(monkeypatch, water_case):
    """The same event in a box with twice the farmland must score the same.

    This is the property the whole-raster metric lacked: it rewarded tight
    boxes and punished generous ones for identical physics.
    """

    def scenario(land_columns):
        # 100 columns of water that fully dries out, plus unchanged land.
        width = 100 + land_columns
        baseline = np.full((50, width), -0.4)  # land: not water
        comparison = baseline.copy()
        baseline[:, :100] = 0.5  # water before
        comparison[:, :100] = -0.3  # dry bed after
        _stub_composites(monkeypatch, baseline, comparison)
        return run_case(water_case, catalog=object())

    tight = scenario(10)
    padded = scenario(400)
    assert tight.passed and padded.passed
    assert tight.stats["loss_fraction"] == pytest.approx(padded.stats["loss_fraction"])


def test_unchanged_water_fails_the_water_case(monkeypatch, water_case):
    baseline = np.full((50, 100), 0.5)
    _stub_composites(monkeypatch, baseline, baseline.copy())
    assert not run_case(water_case, catalog=object()).passed


def test_a_box_containing_no_water_is_reported_not_scored(monkeypatch, water_case):
    """A misplaced bbox must say so, not quietly report 0% change."""
    dry = np.full((50, 100), -0.4)
    _stub_composites(monkeypatch, dry, dry.copy())

    result = run_case(water_case, catalog=object())
    assert not result.passed
    assert "may not contain the feature" in result.detail


def test_detail_names_the_scoring_region(monkeypatch, water_case):
    baseline = np.full((50, 100), 0.5)
    comparison = np.full((50, 100), -0.3)
    _stub_composites(monkeypatch, baseline, comparison)

    detail = run_case(water_case, catalog=object()).detail
    assert "of baseline NDWI > 0" in detail
    assert "valid coverage" in detail


def test_coverage_is_still_judged_on_the_whole_raster(monkeypatch, water_case):
    """Restricting the score must not hide missing imagery."""
    baseline = np.full((50, 100), 0.5)
    comparison = np.full((50, 100), -0.3)
    baseline[:, 20:] = np.nan  # 80% of the box has no data at all
    _stub_composites(monkeypatch, baseline, comparison)

    result = run_case(water_case, catalog=object())
    assert not result.passed
    assert "survived masking" in result.detail


# --- Both directions of change ------------------------------------------


def test_the_suite_tests_increase_as_well_as_decrease():
    """Without an increase case the suite only ever proved one direction."""
    directions = {c.expect for c in CASES}
    assert {"decrease", "increase", "stable"} <= directions


def test_the_increase_case_reuses_a_proven_box():
    """Same ground, opposite direction: a sign error cannot pass both."""
    burn = CASES_BY_KEY["camp-fire"]
    regrowth = CASES_BY_KEY["camp-fire-regrowth"]
    assert regrowth.bbox == burn.bbox
    assert regrowth.index == burn.index
    assert regrowth.expect == "increase" and burn.expect == "decrease"
    assert regrowth.baseline_year == burn.comparison_year


def test_a_real_increase_passes_the_increase_case(monkeypatch):
    case = CASES_BY_KEY["camp-fire-regrowth"]
    baseline = np.full((10, 10), 0.1)
    comparison = np.full((10, 10), 0.6)
    _stub_composites(monkeypatch, baseline, comparison)

    result = run_case(case, catalog=object())
    assert result.passed
    assert "increased" in result.detail


def test_a_decrease_fails_the_increase_case(monkeypatch):
    """The guard against a swapped baseline or a sign error."""
    case = CASES_BY_KEY["camp-fire-regrowth"]
    baseline = np.full((10, 10), 0.6)
    comparison = np.full((10, 10), 0.1)
    _stub_composites(monkeypatch, baseline, comparison)

    assert not run_case(case, catalog=object()).passed


def test_results_report_the_seasonal_offset(monkeypatch, decrease_case):
    baseline = np.full((10, 10), 0.6)
    _stub_composites(monkeypatch, baseline, np.full((10, 10), 0.1))
    assert "season offset" in run_case(decrease_case, catalog=object()).detail
