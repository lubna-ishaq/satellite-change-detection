"""Shared figure construction for the CLI and the Streamlit app."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ndvi_core import DEFAULT_INDEX, SpectralIndex, get_index

INDEX_CMAP = "YlGn"
DELTA_CMAP = "RdBu"

#: Masked pixels are drawn in a neutral grey so cloud gaps are visibly
#: different from "no change" rather than blending into the colour ramp.
NODATA_COLOR = "#d9d9d9"


def _resolve(index: str | SpectralIndex | None) -> SpectralIndex:
    if isinstance(index, SpectralIndex):
        return index
    return get_index(index or DEFAULT_INDEX)


def _cmap_with_nodata(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(NODATA_COLOR)
    return cmap


def build_comparison_figure(
    baseline_values: np.ndarray,
    comparison_values: np.ndarray,
    delta: np.ndarray,
    baseline_label: str,
    comparison_label: str,
    index: str | SpectralIndex | None = None,
    delta_limit: float = 0.5,
):
    """Three-panel baseline / comparison / delta figure.

    Both index panels share one colour scale so they can be read against each
    other, and the delta panel is symmetric around zero so that red and blue
    represent equal magnitudes.
    """
    spec = _resolve(index)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    index_cmap = _cmap_with_nodata(INDEX_CMAP)
    delta_cmap = _cmap_with_nodata(DELTA_CMAP)

    for ax, data, title in (
        (axes[0], baseline_values, f"{spec.name} Baseline\n{baseline_label}"),
        (axes[1], comparison_values, f"{spec.name} Comparison\n{comparison_label}"),
    ):
        image = ax.imshow(
            data, cmap=index_cmap, vmin=spec.display_min, vmax=spec.display_max
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    delta_image = axes[2].imshow(
        delta, cmap=delta_cmap, vmin=-delta_limit, vmax=delta_limit
    )
    axes[2].set_title(
        f"{spec.name} Delta\nBlue = {spec.increase_label} | Red = {spec.decrease_label}"
    )
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(delta_image, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


def format_statistics(stats: dict, index: str | SpectralIndex | None = None) -> str:
    """Render :func:`ndvi_core.change_statistics` output as readable text."""
    if not stats["valid_pixels"]:
        return "No valid pixels: every pixel was masked as cloud, shadow or no-data."

    spec = _resolve(index)
    width = max(len(spec.decrease_label), len(spec.increase_label)) + 20
    decrease = f"{spec.decrease_label} (delta <= -{stats['threshold']:.2f})"
    increase = f"{spec.increase_label} (delta >= +{stats['threshold']:.2f})"

    return (
        f"Valid pixels : {stats['valid_pixels']:,} of {stats['total_pixels']:,} "
        f"({stats['valid_fraction']:.1%} of the raster)\n"
        f"Mean delta   : {stats['mean_delta']:+.4f}\n"
        f"Median delta : {stats['median_delta']:+.4f}\n"
        f"{decrease:<{width}} : {stats['loss_fraction']:.1%} of valid area\n"
        f"{increase:<{width}} : {stats['gain_fraction']:.1%} of valid area\n"
        f"{'Stable':<{width}} : {stats['stable_fraction']:.1%} of valid area"
    )
