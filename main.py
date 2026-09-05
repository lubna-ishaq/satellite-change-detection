"""Command line entry point for the Sentinel-2 change detection pipeline.

Every figure in the README is reproducible from this script, for example:

    python main.py --baseline-year 2021 --comparison-year 2024 \
        --aoi "Neusiedler See, Austria" --out change_detection_vergleich.png
"""

from __future__ import annotations

import argparse
import logging
import sys

from data_access import (
    AOI_PRESETS,
    DEFAULT_SEASON,
    AreaTooLargeError,
    build_geobox,
    load_season_composite,
    open_catalog,
)
from export import write_geotiff
from ndvi_core import (
    INDICES,
    adaptive_threshold,
    calculate_index_delta,
    change_statistics,
    combine_noise,
    get_index,
    pairwise_observations,
)
from plotting import build_comparison_figure, format_statistics

LOGGER = logging.getLogger("change_detection")

#: Beyond this many days between the two composites, seasonal difference
#: starts to dominate whatever real change is present.
MAX_DOY_OFFSET = 21


def _season(value: str) -> tuple[str, str]:
    """Parse ``MM-DD:MM-DD`` into a season window."""
    try:
        start, end = value.split(":")
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Season must look like 06-01:08-31, got {value!r}"
        ) from None
    for part in (start, end):
        month, _, day = part.partition("-")
        if not (month.isdigit() and day.isdigit() and len(month) == 2 and len(day) == 2):
            raise argparse.ArgumentTypeError(f"Season bounds must be MM-DD, got {part!r}")
    return (start, end)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sentinel-2 spectral-index change detection between two seasons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-year", type=int, default=2021)
    parser.add_argument("--comparison-year", type=int, default=2024)

    parser.add_argument(
        "--index",
        choices=sorted(INDICES),
        default="NDVI",
        help="Spectral index: NDVI (vegetation), NDWI (water), NBR (burn).",
    )
    parser.add_argument(
        "--season",
        type=_season,
        default=DEFAULT_SEASON,
        metavar="MM-DD:MM-DD",
        help="Season window applied to both years, e.g. 12-01:02-28.",
    )

    area = parser.add_mutually_exclusive_group()
    area.add_argument(
        "--aoi",
        choices=sorted(AOI_PRESETS),
        default="Graz, Austria",
        help="Named area of interest.",
    )
    area.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Custom bounding box in EPSG:4326, overrides --aoi.",
    )

    parser.add_argument("--resolution", type=float, default=20.0, help="Metres.")
    parser.add_argument("--max-cloud", type=float, default=10.0, help="Percent.")
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=6,
        help="Scenes per year to composite. 1 disables compositing.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Index magnitude below which a change counts as noise.",
    )
    parser.add_argument(
        "--adaptive-threshold",
        action="store_true",
        help=(
            "Judge each pixel against its own scatter across the season "
            "instead of one constant, using --sigma and --threshold-floor."
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Multiples of per-pixel scatter needed with --adaptive-threshold.",
    )
    parser.add_argument(
        "--threshold-floor",
        type=float,
        default=0.05,
        help="Smallest per-pixel threshold allowed with --adaptive-threshold.",
    )
    parser.add_argument("--out", default="change_detection_vergleich.png")
    parser.add_argument(
        "--geotiff",
        metavar="PATH",
        help="Also write the delta as a georeferenced GeoTIFF for QGIS.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.baseline_year >= args.comparison_year:
        LOGGER.warning(
            "Baseline year %d is not earlier than comparison year %d; the delta "
            "will read backwards in time.",
            args.baseline_year,
            args.comparison_year,
        )

    spec = get_index(args.index)
    bbox = tuple(args.bbox) if args.bbox else AOI_PRESETS[args.aoi]
    area_label = "custom bbox" if args.bbox else args.aoi

    # One grid, both seasons. This is what makes the subtraction meaningful.
    try:
        geobox = build_geobox(bbox, resolution=args.resolution)
    except AreaTooLargeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: invalid area: {exc}", file=sys.stderr)
        return 1

    print(
        f"Index         : {spec.name} = ({spec.band_a} - {spec.band_b}) / "
        f"({spec.band_a} + {spec.band_b})\n"
        f"Area          : {area_label} {tuple(round(v, 4) for v in bbox)}\n"
        f"Analysis grid : {geobox.shape.x} x {geobox.shape.y} px "
        f"at {args.resolution:g} m, {geobox.crs}\n"
        f"Season        : {args.season[0]} to {args.season[1]}, "
        f"cloud cover < {args.max_cloud:g}%"
    )

    try:
        catalog = open_catalog()
        baseline = load_season_composite(
            catalog,
            bbox,
            args.baseline_year,
            geobox,
            max_cloud=args.max_cloud,
            season=args.season,
            max_scenes=args.max_scenes,
            index=spec,
        )
        comparison = load_season_composite(
            catalog,
            bbox,
            args.comparison_year,
            geobox,
            max_cloud=args.max_cloud,
            season=args.season,
            max_scenes=args.max_scenes,
            index=spec,
        )
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # STAC API unreachable, signing failure, bad raster
        print(f"error: could not load satellite data: {exc}", file=sys.stderr)
        print(
            "Check your network connection and any proxy or firewall that may "
            "block planetarycomputer.microsoft.com.",
            file=sys.stderr,
        )
        return 2

    print(f"Baseline      : {baseline.date_range}")
    print(f"Comparison    : {comparison.date_range}")

    delta = calculate_index_delta(baseline.values, comparison.values)
    observations = pairwise_observations(baseline.observations, comparison.observations)

    if args.adaptive_threshold:
        noise = combine_noise(baseline.noise, comparison.noise)
        limit = adaptive_threshold(noise, sigma=args.sigma, floor=args.threshold_floor)
    else:
        limit = args.threshold

    stats = change_statistics(delta, threshold=limit)

    # A composite made from a mid-June window and one made from late August
    # differ by growing season as much as by change.
    base_doy, comp_doy = baseline.mean_doy, comparison.mean_doy
    if base_doy is not None and comp_doy is not None:
        offset = abs(comp_doy - base_doy)
        print(f"Season offset : {offset:.0f} days between the two composites")
        if offset > MAX_DOY_OFFSET:
            print(
                f"  warning: more than {MAX_DOY_OFFSET} days apart — part of this "
                "delta is phenology, not change.",
                file=sys.stderr,
            )

    thin = int((observations < 2).sum())
    if thin:
        print(
            f"Thin coverage : {thin:,} px ({thin / observations.size:.1%}) built "
            "from a single scene"
        )

    print()
    print(format_statistics(stats, index=spec))

    figure = build_comparison_figure(
        baseline.values,
        comparison.values,
        delta,
        baseline_label=f"{baseline.year} ({baseline.date_range})",
        comparison_label=f"{comparison.year} ({comparison.date_range})",
        index=spec,
    )
    figure.suptitle(
        f"Sentinel-2 {spec.name} change detection — {area_label}, "
        f"{comparison.year} vs {baseline.year}",
        y=1.02,
    )
    figure.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"\nWrote {args.out}")

    if args.geotiff:
        write_geotiff(
            args.geotiff,
            [delta, observations.astype("float32")],
            geobox,
            band_description=[
                f"{spec.name} delta ({comparison.year} minus {baseline.year})",
                "observations (min of the two seasons)",
            ],
            metadata={
                "index": spec.name,
                "baseline_year": baseline.year,
                "comparison_year": comparison.year,
                "baseline_dates": baseline.date_range,
                "comparison_dates": comparison.date_range,
                "season": f"{args.season[0]}..{args.season[1]}",
                "max_cloud_percent": args.max_cloud,
                "threshold": "adaptive" if args.adaptive_threshold else args.threshold,
                "median_threshold": round(stats["threshold"], 4),
            },
        )
        print(f"Wrote {args.geotiff}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
