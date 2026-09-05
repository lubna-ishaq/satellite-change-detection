import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from ndvi_core import change_statistics  # noqa: E402
from plotting import build_comparison_figure, format_statistics  # noqa: E402


@pytest.fixture
def rasters():
    rng = np.random.default_rng(0)
    baseline = rng.uniform(-0.2, 0.8, size=(20, 25))
    comparison = baseline + rng.normal(0, 0.05, size=(20, 25))
    comparison[:5, :5] = np.nan  # simulate a cloud gap
    return baseline, comparison


def test_figure_has_three_panels_and_three_colorbars(rasters):
    baseline, comparison = rasters
    figure = build_comparison_figure(
        baseline, comparison, comparison - baseline, "2021", "2024"
    )
    # 3 image axes + 3 colorbar axes
    assert len(figure.axes) == 6
    plt.close(figure)


def test_figure_marks_nodata_distinctly(rasters):
    """Cloud gaps must not render as 'no change' in the delta panel."""
    baseline, comparison = rasters
    figure = build_comparison_figure(
        baseline, comparison, comparison - baseline, "2021", "2024"
    )
    delta_axis = figure.axes[2]
    bad_color = delta_axis.images[0].get_cmap().get_bad()
    assert bad_color[3] > 0  # opaque, i.e. explicitly set
    plt.close(figure)


def test_format_statistics_reports_percentages():
    delta = np.array([[-0.3, 0.0, 0.3, np.nan]])
    text = format_statistics(change_statistics(delta, threshold=0.1))
    assert "33.3%" in text
    assert "Mean delta" in text


def test_format_statistics_handles_fully_masked_input():
    text = format_statistics(change_statistics(np.full((2, 2), np.nan)))
    assert "No valid pixels" in text


# --- Index-aware labelling ----------------------------------------------


@pytest.mark.parametrize(
    "index, expected",
    [
        ("NDVI", "Vegetation"),
        ("NDWI", "Water"),
        ("NBR", "Burn"),
    ],
)
def test_delta_panel_names_what_the_index_measures(rasters, index, expected):
    """An NDWI map must not be labelled 'vegetation loss'."""
    baseline, comparison = rasters
    figure = build_comparison_figure(
        baseline, comparison, comparison - baseline, "2021", "2024", index=index
    )
    assert expected in figure.axes[2].get_title()
    plt.close(figure)


def test_index_panels_use_the_index_display_range(rasters):
    from ndvi_core import get_index

    baseline, comparison = rasters
    spec = get_index("NDWI")
    figure = build_comparison_figure(
        baseline, comparison, comparison - baseline, "a", "b", index="NDWI"
    )
    assert figure.axes[0].images[0].get_clim() == (spec.display_min, spec.display_max)
    plt.close(figure)


def test_statistics_text_follows_the_index(rasters):
    delta = np.array([[-0.3, 0.0, 0.3, np.nan]])
    text = format_statistics(change_statistics(delta, threshold=0.1), index="NBR")
    assert "Burn" in text
    assert "Regrowth" in text


def test_statistics_default_to_ndvi_wording():
    delta = np.array([[-0.3, 0.3]])
    assert "Vegetation" in format_statistics(change_statistics(delta))
