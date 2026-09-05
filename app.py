import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import planetary_computer
from pystac_client import Client
import streamlit as st

st.set_page_config(page_title="Satellite Change Detection", layout="wide")

st.title("Sentinel-2 Satellite Change Detection System")
st.write(
    "Automated processing pipeline for multispectral **Sentinel-2 L2A imagery** "
    "(ESA) to analyze vegetation indices and land cover changes over time."
)

st.sidebar.header("Configuration")
year_1 = st.sidebar.selectbox("Baseline Year:", [2020, 2021, 2022], index=1)
year_2 = st.sidebar.selectbox("Comparison Year:", [2023, 2024, 2025], index=1)


@st.cache_resource
def get_catalog():
    """Initialize STAC client session with Planetary Computer signing."""
    return Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


catalog = get_catalog()
# Target bounding box (West, South, East, North)
bbox = [16.70, 47.80, 16.90, 47.95]


def fetch_ndvi(year):
    """Query STAC API, download Red/NIR bands, and compute NDVI matrix."""
    date_range = f"{year}-06-01/{year}-08-31"
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 10}},
    )
    items = list(search.items())
    if not items:
        return None, None

    item = items[0]

    data = odc.stac.load(
        [item],
        bands=["red", "nir"],
        bbox=bbox,
        resolution=20,
    )

    red = data.red.squeeze().values.astype(float)
    nir = data.nir.squeeze().values.astype(float)
    ndvi = (nir - red) / (nir + red + 1e-10)

    return ndvi, item.datetime.strftime("%Y-%m-%d")


if st.sidebar.button("Run Analysis"):
    with st.spinner("Fetching scenes and processing spectral bands..."):
        ndvi_1, date_1 = fetch_ndvi(year_1)
        ndvi_2, date_2 = fetch_ndvi(year_2)

        if ndvi_1 is not None and ndvi_2 is not None:
            # Compute difference matrix
            ndvi_diff = ndvi_2 - ndvi_1

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Baseline scene
            im1 = axes[0].imshow(ndvi_1, cmap="YlGn", vmin=-0.2, vmax=0.8)
            axes[0].set_title(f"Baseline: {year_1} ({date_1})")
            plt.colorbar(im1, ax=axes[0])

            # Comparison scene
            im2 = axes[1].imshow(ndvi_2, cmap="YlGn", vmin=-0.2, vmax=0.8)
            axes[1].set_title(f"Target: {year_2} ({date_2})")
            plt.colorbar(im2, ax=axes[1])

            # Change map
            im3 = axes[2].imshow(ndvi_diff, cmap="bwr", vmin=-0.5, vmax=0.5)
            axes[2].set_title("NDVI Delta (Red = Decrease | Blue = Increase)")
            plt.colorbar(im3, ax=axes[2])

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.error("No valid satellite scenes found meeting the cloud cover criteria (<10%).")