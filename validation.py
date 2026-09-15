"""Check the pipeline against real, documented events.

The unit tests only check that the math is consistent. They can't show
that the pipeline finds real changes in real satellite data. So each case
here is a place where something known happened (a fire, a drained lake),
together with the direction of change I expect.

There is also a negative control: a desert area where nothing should
change. If the pipeline has a calibration error, it shows up there as
fake change, which makes it the most useful case.

Usage:

    python validation.py            # all cases
    python validation.py --case camp-fire
    python validation.py --report validation_results.md

I checked every bounding box on a map. The first control area turned out
to be an irrigation project (it looks like empty desert on the satellite
image), which would have made it useless. The boxes are still rough, so
check them on a map before quoting any number.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass

import numpy as np

from data_access import build_geobox, load_season_composite, open_catalog
from ndvi_core import calculate_index_delta, change_statistics

Expectation = str  # "decrease" | "increase" | "stable"

# If fewer pixels than this are left after filtering, the result is not meaningful
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
    # "decrease"/"increase": minimum share of pixels that must change that way.
    # "stable": maximum share of pixels that may change at all.
    fraction: float
    event: str
    # Only for "stable": maximum allowed mean delta. This is the main check
    # for the control (see the comment in run_case).
    max_bias: float = 0.02
    # Only score pixels where the baseline index is above this value.
    # For water cases 0.0 means "was water in the baseline year".
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
        # Same box as camp-fire on purpose. The same pixels have to go down
        # 2018->2019 and back up 2019->2024, so a sign error can't pass both.
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
        # Only pixels that were water in 2022 are scored: at least half of
        # them must be gone. Over the whole box it was only 15.5%, because
        # most of the box is farmland.
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
        name="NEGATIVE CONTROL: Great Sand Sea, western Egypt",
        # My first box was at 28.40-28.70E / 22.60-22.85N. It looks like
        # desert, but it's the East Uweinat irrigation project (with an
        # airport). On farmland you can't tell a pipeline error from real crop
        # changes, so I moved it here: only sand dunes, no roads or fields.
        bbox=(26.10, 25.35, 26.40, 25.60),
        index="NDVI",
        baseline_year=2019,
        comparison_year=2024,
        expect="stable",
        # At the old location 7.7% of pixels changed by more than 0.1. I
        # thought that was desert noise and allowed 15%. At the new location
        # it's 0.0%, so it was actually the crops. The limit is now 2%.
        #
        # The bias limit stays at 0.02. That's the main check and it shouldn't
        # fail because of normal year-to-year differences. Measured: 0.0013.
        fraction=0.02,
        max_bias=0.02,
        event=(
            "Open sand desert with negligible vegetation and almost no cloud. "
            "Nothing here should change between years. If the MEAN delta "
            "drifts from zero, the pipeline has a remaining bias, most likely "
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
    """Run one case with real data and check if the result is as expected."""
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

    # First check if we got enough usable pixels at all (whole image).
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

    # For water cases only the pixels that were water in the baseline count.
    # Otherwise the result depends on how much farmland is in the box: a
    # bigger box would give a lower score for the same event. The question
    # is: of the water that was there, how much is gone?
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
        # The control is mainly about systematic errors. A calibration error
        # shifts the whole image in one direction, so it shows up in the mean.
        # Single pixels can still be noisy over desert (red and NIR are both
        # small there, so the ratio is noisy). That's why the mean is the main
        # check and the share of changed pixels is only a second check.
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


def format_report(results: list[CaseResult]) -> str:
    """Create a Markdown table of the results (saved as validation_results.md)."""
    lines = [
        "# Validation results",
        "",
        f"Generated by `python validation.py --report` on {dt.date.today():%Y-%m-%d}.",
        "",
        "| Case | Index | Period | Status | Result |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        c = r.case
        lines.append(
            f"| {c.name} | {c.index} | {c.baseline_year} → {c.comparison_year} "
            f"| {r.status} | {r.detail} |"
        )
    passed = sum(r.passed for r in results)
    lines += ["", f"{passed}/{len(results)} cases passed.", ""]
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
    parser.add_argument(
        "--report", metavar="PATH", help="Also write the results as a Markdown table."
    )
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
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(format_report(results))
        print(f"Wrote {args.report}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
