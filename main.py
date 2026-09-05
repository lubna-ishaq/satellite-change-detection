import matplotlib.pyplot as plt
import numpy as np
import planetary_computer as pc
from pystac_client import Client
import odc.stac

from ndvi_core import calculate_ndvi, calculate_ndvi_delta

# 1. Search Satellite Data
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[15.35, 47.01, 15.50, 47.12],  # Graz Region
    datetime="2023-06-01/2023-08-31",
    query={"eo:cloud_cover": {"lt": 10}},
)

items = list(search.items())
if not items:
    raise ValueError("No matching satellite imagery found.")

selected_item = items[0]

# 2. Load Red and NIR Bands
ds = odc.stac.load(
    [selected_item],
    bands=["B04", "B08"],
    resolution=20,
    patch_url=pc.sign_url,
)

red = ds["B04"].squeeze().values
nir = ds["B08"].squeeze().values

# 3. Calculate NDVI & Synthetic Delta Example
ndvi_baseline = calculate_ndvi(red, nir)
# Simulated comparison state for local test run
ndvi_comparison = ndvi_baseline * 0.95 
ndvi_delta = calculate_ndvi_delta(ndvi_baseline, ndvi_comparison)

# 4. Save Plot
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(ndvi_baseline, cmap="YlGn")
ax[0].set_title("Baseline NDVI")
ax[1].imshow(ndvi_comparison, cmap="YlGn")
ax[1].set_title("Comparison NDVI")
im = ax[2].imshow(ndvi_delta, cmap="RdBu", vmin=-0.5, vmax=0.5)
ax[2].set_title("NDVI Delta Shift")

plt.colorbar(im, ax=ax[2])
plt.tight_layout()
plt.savefig("example_output.png")
print("Successfully generated example_output.png")