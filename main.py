import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import planetary_computer
from pystac_client import Client

# Initialize STAC client with auth signing
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Target ROI (West, South, East, North) - Neusiedler See region
bbox = [16.70, 47.80, 16.90, 47.95]


def get_ndvi_for_period(date_range, label):
    """Fetches Sentinel-2 data and computes NDVI for a given date window."""
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 10}},
    )
    items = list(search.items())
    if not items:
        raise ValueError(f"No valid scenes found for {label}")

    item = items[0]
    date_str = item.datetime.strftime("%Y-%m-%d")
    print(f"[{label}] Selected scene: {date_str}")

    # Load Red (B04) and NIR (B08) bands
    data = odc.stac.load(
        [item],
        bands=["red", "nir"],
        bbox=bbox,
        resolution=20,
    )

    red = data.red.squeeze().values.astype(float)
    nir = data.nir.squeeze().values.astype(float)

    # Compute NDVI: (NIR - Red) / (NIR + Red)
    ndvi = (nir - red) / (nir + red + 1e-10)
    return ndvi, date_str


# Fetch imagery for baseline and comparative periods
ndvi_2021, date_2021 = get_ndvi_for_period("2021-06-01/2021-08-31", "2021")
ndvi_2024, date_2024 = get_ndvi_for_period("2024-06-01/2024-08-31", "2024")

# Compute absolute change matrix
ndvi_diff = ndvi_2024 - ndvi_2021

# Plot comparison pipeline
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Baseline period
im1 = axes[0].imshow(ndvi_2021, cmap="YlGn", vmin=-0.2, vmax=0.8)
axes[0].set_title(f"NDVI Baseline ({date_2021})")
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# Target period
im2 = axes[1].imshow(ndvi_2024, cmap="YlGn", vmin=-0.2, vmax=0.8)
axes[1].set_title(f"NDVI Target ({date_2024})")
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

# Change map (Red = decrease, Blue = increase)
im3 = axes[2].imshow(ndvi_diff, cmap="bwr", vmin=-0.5, vmax=0.5)
axes[2].set_title("NDVI Delta (2024 vs 2021)\nRed = Decrease | Blue = Increase")
plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("change_detection_vergleich.png", dpi=300)
plt.close()

print("\nPipeline execution complete. Output saved to 'change_detection_vergleich.png'.")