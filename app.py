"""Streamlit front end for the Sentinel-2 change detection pipeline.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from data_access import (
    AOI_PRESETS,
    AreaTooLargeError,
    build_geobox,
    load_season_composite,
    open_catalog,
)
from export import geotiff_bytes
from geocoding import GeocodingError, search_place
from ndvi_core import (
    DEFAULT_INDEX,
    INDICES,
    adaptive_threshold,
    calculate_index_delta,
    change_statistics,
    combine_noise,
    get_index,
    pairwise_observations,
)
from plotting import build_comparison_figure

try:
    import folium
    from streamlit_folium import st_folium

    HAS_MAP = True
except ImportError:  # the app still works without the optional map
    HAS_MAP = False

st.set_page_config(page_title="Sentinel-2 Change Detection", layout="wide")
st.title("Sentinel-2 Satellite Change Detection")
st.caption(
    "Spectral-index composites from the Microsoft Planetary Computer, "
    "harmonised for the Baseline 04.00 reflectance offset and masked per "
    "pixel with the Scene Classification Layer."
)


@st.cache_data(show_spinner=False, ttl=3600)
def _load(bbox, year, resolution, max_cloud, max_scenes, index_name, season):
    """Cached loader.

    Scene downloads dominate the runtime, so results are memoised on the
    parameters. The GeoBox is rebuilt from the same bbox and resolution, which
    makes it deterministic and therefore safe to cache per year: both seasons
    still land on an identical grid.
    """
    geobox = build_geobox(tuple(bbox), resolution=resolution)
    composite = load_season_composite(
        open_catalog(),
        tuple(bbox),
        year,
        geobox,
        max_cloud=max_cloud,
        season=tuple(season),
        max_scenes=max_scenes,
        index=index_name,
    )
    return (
        composite.values,
        composite.date_range,
        composite.scene_count,
        composite.observations,
        composite.noise,
        composite.mean_doy,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _geocode(query: str):
    return search_place(query)


DEFAULT_PRESET = "Neusiedler See, Austria"

if "bbox" not in st.session_state:
    st.session_state.bbox = AOI_PRESETS[DEFAULT_PRESET]
    st.session_state.area_label = DEFAULT_PRESET


with st.sidebar:
    st.header("Area")
    mode = st.radio(
        "Choose the area by", ["Preset", "Place name", "Coordinates"], index=0
    )

    if mode == "Preset":
        preset_names = sorted(AOI_PRESETS)
        name = st.selectbox(
            "Preset", preset_names, index=preset_names.index(DEFAULT_PRESET)
        )
        st.session_state.bbox = AOI_PRESETS[name]
        st.session_state.area_label = name

    elif mode == "Place name":
        query = st.text_input("Search", placeholder="e.g. Lake Garda")
        if query:
            try:
                places = _geocode(query)
            except GeocodingError as exc:
                places = []
                st.error(str(exc))
            if places:
                chosen = st.selectbox("Result", places, format_func=lambda p: p.name[:70])
                st.session_state.bbox = chosen.bbox
                st.session_state.area_label = chosen.name.split(",")[0]
            elif query.strip():
                st.info("Nothing found for that search.")

    else:
        west, south, east, north = st.session_state.bbox
        cols = st.columns(2)
        west = cols[0].number_input("West", value=float(west), format="%.4f")
        south = cols[1].number_input("South", value=float(south), format="%.4f")
        east = cols[0].number_input("East", value=float(east), format="%.4f")
        north = cols[1].number_input("North", value=float(north), format="%.4f")
        st.session_state.bbox = (west, south, east, north)
        st.session_state.area_label = "custom bbox"

    st.header("Index")
    # sorted() puts NBR first alphabetically; the default must be the
    # project default, not whatever happens to sort first.
    index_names = sorted(INDICES)
    index_name = st.selectbox(
        "Spectral index", index_names, index=index_names.index(DEFAULT_INDEX)
    )
    spec = get_index(index_name)
    st.caption(spec.description)

    st.header("Period")
    years = list(range(2016, 2026))
    baseline_year = st.selectbox("Baseline year", years, index=years.index(2021))
    comparison_year = st.selectbox("Comparison year", years, index=years.index(2024))
    season = st.select_slider(
        "Season",
        options=["06-01:08-31", "03-01:05-31", "09-01:11-30", "12-01:02-28"],
        value="06-01:08-31",
        format_func=lambda s: {
            "06-01:08-31": "Summer (Jun–Aug)",
            "03-01:05-31": "Spring (Mar–May)",
            "09-01:11-30": "Autumn (Sep–Nov)",
            "12-01:02-28": "Winter (Dec–Feb)",
        }[s],
    )
    season_tuple = tuple(season.split(":"))

    st.header("Quality")
    max_cloud = st.slider("Max scene cloud cover (%)", 1, 60, 10)
    max_scenes = st.slider("Scenes per year (median composite)", 1, 10, 4)
    resolution = st.select_slider("Resolution (m)", [10, 20, 60], value=20)
    adaptive = st.toggle(
        "Adaptive threshold",
        value=False,
        help=(
            "Judge each pixel against its own scatter across the season "
            "instead of one constant for the whole scene."
        ),
    )
    if adaptive:
        sigma = st.slider("Sigma", 1.0, 4.0, 2.0, step=0.5)
        threshold = st.slider("Threshold floor", 0.02, 0.20, 0.05, step=0.01)
    else:
        sigma = 2.0
        threshold = st.slider("Change threshold", 0.02, 0.40, 0.10, step=0.01)

    if st.button("Run analysis", type="primary", use_container_width=True):
        # A st.button value lives for exactly one rerun. The folium map is a
        # custom component and triggers a rerun of its own right after the
        # click, which would silently swallow the request. Latch it in session
        # state instead, so the analysis survives that second pass.
        st.session_state.analysis_requested = True

    run = st.session_state.get("analysis_requested", False)

bbox = tuple(float(v) for v in st.session_state.bbox)
area_label = st.session_state.area_label

try:
    geobox = build_geobox(bbox, resolution=resolution)
except AreaTooLargeError as exc:
    st.error(str(exc))
    st.stop()
except ValueError as exc:
    st.error(f"Invalid area: {exc}")
    st.stop()

if baseline_year >= comparison_year:
    st.warning(
        "The baseline year is not earlier than the comparison year — the delta "
        "will read backwards in time."
    )

if not run:
    st.info(
        f"**{area_label}** — {geobox.shape.x} x {geobox.shape.y} px at "
        f"{resolution} m ({geobox.crs}). Press **Run analysis** in the sidebar."
    )
    if HAS_MAP:
        west, south, east, north = bbox
        preview = folium.Map(
            location=[(south + north) / 2, (west + east) / 2], zoom_start=9
        )
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color="#2166ac",
            fill=True,
            fill_opacity=0.15,
        ).add_to(preview)
        preview.fit_bounds([[south, west], [north, east]])
        st_folium(preview, height=420, width=None, returned_objects=[])
    else:
        st.caption("Install `folium` and `streamlit-folium` to see the area on a map.")
    st.stop()

try:
    with st.spinner(f"Loading {baseline_year} composite…"):
        baseline_values, baseline_dates, baseline_n, base_obs, base_noise, base_doy = (
            _load(
                bbox,
                baseline_year,
                resolution,
                max_cloud,
                max_scenes,
                index_name,
                season_tuple,
            )
        )
    with st.spinner(f"Loading {comparison_year} composite…"):
        (
            comparison_values,
            comparison_dates,
            comparison_n,
            comp_obs,
            comp_noise,
            comp_doy,
        ) = _load(
            bbox,
            comparison_year,
            resolution,
            max_cloud,
            max_scenes,
            index_name,
            season_tuple,
        )
except LookupError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # network, signing, or raster read failures
    st.error(f"Could not load satellite data: {exc}")
    st.stop()

delta = calculate_index_delta(baseline_values, comparison_values)
observations = pairwise_observations(base_obs, comp_obs)

if adaptive:
    limit = adaptive_threshold(
        combine_noise(base_noise, comp_noise), sigma=sigma, floor=threshold
    )
else:
    limit = threshold

stats = change_statistics(delta, threshold=limit)

if base_doy is not None and comp_doy is not None and abs(comp_doy - base_doy) > 21:
    st.warning(
        f"The two composites are {abs(comp_doy - base_doy):.0f} days apart in the "
        "season. Part of this delta is phenology, not change."
    )

thin = int((observations < 2).sum())
if thin:
    st.info(
        f"{thin:,} pixels ({thin / observations.size:.1%}) rest on a single scene. "
        "They are the least trustworthy part of this map."
    )

if stats["valid_fraction"] < 0.5:
    st.warning(
        f"Only {stats['valid_fraction']:.0%} of pixels survived cloud masking. "
        "Raise the cloud threshold or the number of scenes per year."
    )

metric_cols = st.columns(4)
metric_cols[0].metric(spec.decrease_label, f"{stats['loss_fraction']:.1%}")
metric_cols[1].metric(spec.increase_label, f"{stats['gain_fraction']:.1%}")
metric_cols[2].metric(f"Mean {spec.name} delta", f"{stats['mean_delta']:+.3f}")
metric_cols[3].metric("Valid pixels", f"{stats['valid_fraction']:.0%}")

figure = build_comparison_figure(
    baseline_values,
    comparison_values,
    delta,
    baseline_label=f"{baseline_year} ({baseline_dates})",
    comparison_label=f"{comparison_year} ({comparison_dates})",
    index=spec,
)
st.pyplot(figure)

png_buffer = io.BytesIO()
figure.savefig(png_buffer, format="png", dpi=150, bbox_inches="tight")
plt.close(figure)  # Streamlit reruns on every widget change; unclosed figures leak.

slug = f"{area_label.split(',')[0]}_{spec.name}_{comparison_year}_vs_{baseline_year}"
slug = slug.replace(" ", "_")

download_cols = st.columns(2)
download_cols[0].download_button(
    "Download figure (PNG)",
    data=png_buffer.getvalue(),
    file_name=f"{slug}.png",
    mime="image/png",
    use_container_width=True,
)
download_cols[1].download_button(
    "Download delta (GeoTIFF)",
    data=geotiff_bytes(
        [delta, observations.astype("float32")],
        geobox,
        band_description=[f"{spec.name} delta", "observations"],
        metadata={
            "index": spec.name,
            "baseline_year": baseline_year,
            "comparison_year": comparison_year,
            "baseline_dates": baseline_dates,
            "comparison_dates": comparison_dates,
            "season": season,
            "threshold": "adaptive" if adaptive else threshold,
            "median_threshold": round(stats["threshold"], 4),
        },
    ),
    file_name=f"{slug}.tif",
    mime="image/tiff",
    use_container_width=True,
    help="Georeferenced raster for QGIS, ArcGIS or rasterio.",
)

with st.expander("Delta distribution and provenance"):
    finite = delta[np.isfinite(delta)]
    hist_fig, hist_ax = plt.subplots(figsize=(8, 3))
    hist_ax.hist(finite, bins=80, color="#4a7c59")
    hist_ax.axvline(0, color="black", linewidth=0.8)
    hist_ax.axvline(-threshold, color="#b2182b", linestyle="--", linewidth=0.8)
    hist_ax.axvline(threshold, color="#2166ac", linestyle="--", linewidth=0.8)
    hist_ax.set_xlabel(f"{spec.name} delta")
    hist_ax.set_ylabel("Pixels")
    st.pyplot(hist_fig)
    plt.close(hist_fig)

    formula = f"({spec.band_a} − {spec.band_b}) / ({spec.band_a} + {spec.band_b})"
    grid = (
        f"{geobox.shape.x} × {geobox.shape.y} px at {resolution} m, "
        f"{geobox.crs} — identical for both years"
    )
    st.markdown(
        f"""
- **Index:** {spec.name} = {formula}
- **Area:** {area_label} — bbox `{tuple(round(v, 4) for v in bbox)}`
- **Season:** {season_tuple[0]} to {season_tuple[1]}, both years
- **Baseline:** {baseline_n} scene(s), {baseline_dates}
- **Comparison:** {comparison_n} scene(s), {comparison_dates}
- **Grid:** {grid}
- **Threshold:** {"adaptive, median " if adaptive else "fixed "}{stats["threshold"]:.3f}
- **Season offset:** {abs(comp_doy - base_doy):.0f} days between composites
- **Median delta:** {stats["median_delta"]:+.4f} over
  {stats["valid_pixels"]:,} valid pixels
"""
    )

st.caption(
    "Interpret a single-season delta as a candidate for change, not confirmed "
    "change: sun angle and phenology differences survive compositing."
)
