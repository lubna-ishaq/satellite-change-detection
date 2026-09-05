import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import planetary_computer
from pystac_client import Client

print("Verbindung zur Satelliten-API wird hergestellt...")

# 1. API-Verbindung herstellen
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

bbox = [16.70, 47.80, 16.90, 47.95]

# Funktion zum Laden und Berechnen des NDVI für einen bestimmten Zeitraum
def get_ndvi_for_period(date_range, label):
    print(f"Suche Bild für {label} ({date_range})...")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 10}},
    )
    items = list(search.items())
    item = items[0]
    print(f"-> Geladenes Bild vom: {item.datetime.strftime('%Y-%m-%d')}")
    
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

# 2. Zwei verschiedene Jahre abfragen (Vorher vs. Nachher)
ndvi_2021, date_2021 = get_ndvi_for_period("2021-06-01/2021-08-31", "2021")
ndvi_2024, date_2024 = get_ndvi_for_period("2024-06-01/2024-08-31", "2024")

# 3. Differenz berechnen (Change Detection Matrix)
# Positiver Wert = Mehr Grün/Vegetation | Negativer Wert = Rückgang/Trockenheit/Abbau
ndvi_diff = ndvi_2024 - ndvi_2021

# 4. Dreiteilige Visualisierung erstellen
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Bild 1: Zustand 2021
im1 = axes[0].imshow(ndvi_2021, cmap="YlGn", vmin=-0.2, vmax=0.8)
axes[0].set_title(f"NDVI Status ({date_2021})")
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# Bild 2: Zustand 2024
im2 = axes[1].imshow(ndvi_2024, cmap="YlGn", vmin=-0.2, vmax=0.8)
axes[1].set_title(f"NDVI Status ({date_2024})")
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

# Bild 3: Die Differenz (Change Detection)
# 'bwr' Colormap: Rot = Rückgang (Trockenheit/Abbau), Blau = Zunahme (Wachstum/Nässe)
im3 = axes[2].imshow(ndvi_diff, cmap="bwr", vmin=-0.5, vmax=0.5)
axes[2].set_title("Veränderung (2024 vs 2021)\nRot = Rückgang | Blau = Zunahme")
plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("change_detection_vergleich.png")
plt.close()

print("\nFERTIG! Das Vergleichsbild wurde als 'change_detection_vergleich.png' gespeichert.")