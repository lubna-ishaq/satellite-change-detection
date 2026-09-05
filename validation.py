"""Validation against documented change events.

Unit tests prove the arithmetic is self-consistent. They cannot prove the
pipeline detects real change on real imagery. This module closes that gap:
each case is a place and period where something well documented happened,
plus the direction and rough magnitude the pipeline is expected to report.

The suite deliberately includes a **negative control** — a hyper-arid area
where almost nothing changes between years. A pipeline with a residual
calibration bias will happily report "change" there, so a control that must
come back quiet is the single most informative case in the set.

Run it with:

    python validation.py            # all cases
    python validation.py --case camp-fire

Every bounding box below has been checked on a map against what it is
supposed to contain. That check is not optional bookkeeping: the negative
control originally sat on a centre-pivot irrigation scheme that reads as
empty desert on a satellite image, which would have made it useless at the
one thing it exists for. Boxes are still approximate — chosen to contain
the event, not to trace it — so re-check before citing any number.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from data_access import build_geobox, load_season_composite, open_catalog
from ndvi_core import calculate_index_delta, change_statistics

Expectation = str  # "decrease" | "increase" | "stable"

#: A restricted region smaller than this is not a measurement, it is noise.
MIN_REGION_PIXELS = 500


@dataclass(frozen=True)
class ValidationCase:
    key: str
    name: str
    bbox: tuple[float, float, float, float]
    index: str
    baseline_year: int
    comparison_year: int
    expect: Expectation
    #: For "decrease"/"increase": the minimum share of valid area that must
    #: move that way. For "stable": the maximum share allowed to move at all.
    fraction: float
    event: str
    #: "stable" cases only: the largest mean delta accepted. This, not the
    #: scattered fraction, is the real test — see the note in `run_case`.
    max_bias: float = 0.02
    #: Restrict scoring to pixels whose BASELINE index exceeds this value.
    #: For a water case, 0.0 selects "was water before" — see `run_case`.
    baseline_above: float | None = None
    season: tuple[str, str] = ("06-01", "08-31")
    resolution: float = 60.0
    threshold: float = 0.1
    max_cloud: float = 20.0
    max_scenes: int = 4


CASES: tuple[ValidationCase, ...] = (
    ValidationCase(
        key="camp-fire",
        name="Camp Fire burn scar, Paradise, California",
        bbox=(-121.70, 39.68, -121.50, 39.86),
        index="NBR",
        baseline_year=2018,
        comparison_year=2019,
        expect="decrease",
        fraction=0.25,
        event=(
            "The Camp Fire burned this area in November 2018, between the two "
            "summer windows compared here. NBR drops sharply over burn scars."
        ),
    ),
    ValidationCase(
        key="camp-fire-regrowth",
        name="Camp Fire regrowth, Paradise, California",
        # Deliberately the same box as camp-fire: the strongest test of an
        # increase is ground already proven to produce a clean decrease.
        # The same pixels must fall 2018->2019 and recover 2019->2024, so a
        # sign error or a swapped baseline cannot pass both.
        bbox=(-121.70, 39.68, -121.50, 39.86),
        index="NBR",
        baseline_year=2019,
        comparison_year=2024,
        expect="increase",
        fraction=0.25,
        event=(
            "Five growing seasons after the November 2018 Camp Fire, "
            "vegetation has re-established over much of the burn scar and NBR "
            "recovers. Without this case the suite only ever tested one "
            "direction of change."
        ),
    ),
    ValidationCase(
        key="kakhovka",
        name="Kakhovka reservoir, Ukraine",
        bbox=(33.60, 46.95, 34.60, 47.45),
        index="NDWI",
        baseline_year=2022,
        comparison_year=2024,
        expect="decrease",
        # Scored over baseline water only: at least half of what was water in
        # 2022 must no longer be water. Over the whole box this signal was
        # 15.5%, which says more about how much farmland the box includes
        # than about the reservoir.
        fraction=0.50,
        baseline_above=0.0,
        event=(
            "The Kakhovka dam was breached in June 2023 and the reservoir "
            "drained. NDWI falls where open water became exposed bed."
        ),
    ),
    ValidationCase(
        key="aral-sea",
        name="Eastern basin, South Aral Sea",
        bbox=(59.60, 44.90, 60.60, 45.60),
        index="NDWI",
        baseline_year=2018,
        comparison_year=2024,
        expect="decrease",
        fraction=0.40,
        baseline_above=0.0,
        max_cloud=40.0,
        max_scenes=6,
        event=(
            "The eastern basin has repeatedly desiccated over the Sentinel-2 "
            "era. Needs a larger imagery budget than the other cases: at "
            "max_cloud=20 / 4 scenes only ~10% of the box had usable pixels, "
            "at 40 / 6 it reaches ~62%. The limit is scene availability over "
            "this region, not the surface itself."
        ),
    ),
    ValidationCase(
        key="sahara-control",
        name="NEGATIVE CONTROL — Great Sand Sea, western Egypt",
        # Checked on a map before use. An earlier version of this control sat
        # at 28.40-28.70E / 22.60-22.85N, which looks like empty desert on a
        # satellite image but is the East Uweinat centre-pivot irrigation
        # scheme, complete with an airport. A control standing on active
        # farmland cannot distinguish pipeline bias from real cropping change,
        # which is the one job it has. This box is open dune field: no roads,
        # no fields, no settlements.
        bbox=(26.10, 25.35, 26.40, 25.60),
        index="NDVI",
        baseline_year=2019,
        comparison_year=2024,
        expect="stable",
        # Scatter bound RE-DERIVED after moving the site. At the old, wrongly
        # chosen location 7.7% of pixels crossed +-0.1 and that was explained
        # away as noise over bare desert; the loose 15% bound came from that
        # reasoning. On genuine dune field the measured figure is 0.0%, so the
        # scatter was the irrigation scheme's crops all along and the
        # explanation was wrong. 2% keeps headroom over a measured zero
        # without being the rubber band it was.
        #
        # The bias bound stays at 0.02. It is the primary criterion and the
        # one that must not become brittle to ordinary year-to-year variation;
        # the measured value is 0.0065.
        fraction=0.02,
        max_bias=0.02,
        event=(
            "Open sand desert with negligible vegetation and almost no cloud. "
            "Nothing here should change between years. If the MEAN delta "
            "drifts from zero, the pipeline has a residual bias — most likely "
            "an incomplete radiometric harmonisation."
        ),
    ),
)

CASES_BY_KEY = {case.key: case for case in CASES}


@dataclass
class CaseResult:
    case: ValidationCase
    passed: bool
    detail: str
    stats: dict | None = None

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


def run_case(case: ValidationCase, catalog=None) -> CaseResult:
    """Execute one case end to end and judge it against its expectation."""
    catalog = catalog or open_catalog()
    geobox = build_geobox(case.bbox, resolution=case.resolution)

    def load(year: int):
        return load_season_composite(
            catalog,
            case.bbox,
            year,
            geobox,
            max_cloud=case.max_cloud,
            season=case.season,
            max_scenes=case.max_scenes,
            index=case.index,
        )

    try:
        baseline = load(case.baseline_year)
        comparison = load(case.comparison_year)
    except LookupError as exc:
        return CaseResult(case, False, f"no usable imagery: {exc}")

    delta = calculate_index_delta(baseline.values, comparison.values)
    coverage = change_statistics(delta, threshold=case.threshold)

    # Coverage is judged on the whole raster: it answers "did we get imagery?",
    # which is a different question from "did the event show up?".
    if coverage["valid_pixels"] == 0:
        return CaseResult(case, False, "every pixel was masked as cloud", coverage)

    if coverage["valid_fraction"] < 0.3:
        return CaseResult(
            case,
            False,
            f"only {coverage['valid_fraction']:.0%} of pixels survived masking; "
            "raise max_cloud or max_scenes",
            coverage,
        )

    # Some events only make sense measured over the ground they affect. A
    # reservoir that drains changes the water, not the surrounding farmland,
    # so scoring "share of the whole box that lost water" mostly measures how
    # much padding the box has — pad it more and the same real event scores
    # lower. Restricting to pixels that were water in the baseline asks the
    # question that has a defensible answer: of the water that was there, how
    # much went away?
    scope = ""
    judged = delta
    if case.baseline_above is not None:
        region = np.asarray(baseline.values) > case.baseline_above
        judged = np.where(region, delta, np.nan)
        scope = f" of baseline {case.index} > {case.baseline_above:g}"

    stats = change_statistics(judged, threshold=case.threshold)

    if case.baseline_above is not None and stats["valid_pixels"] < MIN_REGION_PIXELS:
        return CaseResult(
            case,
            False,
            f"only {stats['valid_pixels']:,} pixels had baseline {case.index} > "
            f"{case.baseline_above:g}; the box may not contain the feature",
            stats,
        )

    down, up = stats["loss_fraction"], stats["gain_fraction"]

    if case.expect == "decrease":
        passed = down >= case.fraction
        detail = (
            f"{down:.1%} decreased{scope} (need >= {case.fraction:.0%}), "
            f"valid coverage {coverage['valid_fraction']:.0%}"
        )
    elif case.expect == "increase":
        passed = up >= case.fraction
        detail = (
            f"{up:.1%} increased{scope} (need >= {case.fraction:.0%}), "
            f"valid coverage {coverage['valid_fraction']:.0%}"
        )
    elif case.expect == "stable":
        # What a negative control is actually for is detecting *systematic*
        # bias: a calibration error shifts the whole scene one way, so it
        # shows up in the mean. Per-pixel scatter is a different thing. Over
        # bare desert both red and NIR reflectance are small, and a ratio of
        # two small numbers amplifies sensor noise, so a few percent of
        # pixels crossing a ±0.1 threshold is expected physics, not a bug.
        # Judging the control on scatter alone would fail a correct pipeline
        # and tempt the next person to "fix" it by loosening the threshold.
        # So: the mean is the criterion, with a generous scatter bound kept
        # as a second guard so the case can still fail loudly.
        bias = abs(stats["mean_delta"])
        moved = down + up
        passed = bias <= case.max_bias and moved <= case.fraction
        detail = (
            f"mean bias {bias:.4f} (allowed <= {case.max_bias:.4f}), "
            f"{moved:.1%} scattered (allowed <= {case.fraction:.0%})"
        )
    else:  # pragma: no cover - guarded by the dataclass contract
        raise ValueError(f"Unknown expectation {case.expect!r}")

    base_doy, comp_doy = baseline.mean_doy, comparison.mean_doy
    if base_doy is not None and comp_doy is not None:
        detail += f", season offset {abs(comp_doy - base_doy):.0f} d"

    return CaseResult(case, passed, detail, stats)


def format_result(result: CaseResult) -> str:
    case = result.case
    lines = [
        f"[{result.status}] {case.name}",
        f"        {case.index}, {case.baseline_year} -> {case.comparison_year}, "
        f"expect {case.expect}",
        f"        {result.detail}",
    ]
    if result.stats and result.stats["valid_pixels"]:
        lines.append(
            f"        mean delta {result.stats['mean_delta']:+.4f}, "
            f"valid {result.stats['valid_fraction']:.0%}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--case",
        choices=sorted(CASES_BY_KEY),
        action="append",
        help="Run only this case; repeatable. Default: all.",
    )
    parser.add_argument("--list", action="store_true", help="List cases and exit.")
    args = parser.parse_args(argv)

    if args.list:
        for case in CASES:
            print(f"{case.key:16s} {case.name}")
            print(f"{'':16s} {case.event}")
        return 0

    selected = [CASES_BY_KEY[k] for k in args.case] if args.case else list(CASES)

    try:
        catalog = open_catalog()
    except Exception as exc:
        print(f"error: could not reach the STAC API: {exc}", file=sys.stderr)
        return 2

    results = []
    for case in selected:
        print(f"running {case.key} …", flush=True)
        try:
            result = run_case(case, catalog=catalog)
        except Exception as exc:
            result = CaseResult(case, False, f"unexpected error: {exc}")
        results.append(result)
        print(format_result(result))
        print()

    passed = sum(r.passed for r in results)
    print(f"{passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
