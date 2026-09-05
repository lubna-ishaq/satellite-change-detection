import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import planetary_computer
from pystac_client import Client
import streamlit as st

# Seiten-Konfiguration
st.set_page_config(page_title="Satellite Change Detection", layout="wide")

st.title("🛰️ Satellite Image Change Detection System")
st.write(
    "Diese Anwendung analysiert automatisiert multispektrale **Sentinel-2 Satellitendaten** "
    "der Europäischen Weltraumorganisation (ESA) zur Erkennung von Umwelt- und Vegetationsveränderungen."
)

# Sidebar Einstellungen
st.sidebar.header("Analyse-Parameter")
jahr_1 = st.sidebar.selectbox("Basis-Jahr (Vorher):", [2020, 2021, 2022], index=1)
jahr_2 = st.sidebar.selectbox("Vergleichs-Jahr (Nachher):", [2023, 2024, 2025], index=1)

@st.cache_resource
def get_catalog():
    return Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

catalog = get_catalog()
bbox = [16.70, 47.80, 16.90, 47.95]

def fetch_ndvi(year):
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

if st.sidebar.button("Analyse starten"):
    with st.spinner("Lade Satellitenbilder und berechne NDVI..."):
        ndvi_1, date_1 = fetch_ndvi(jahr_1)
        ndvi_2, date_2 = fetch_ndvi(jahr_2)

        if ndvi_1 is not None and ndvi_2 is not None:
            ndvi_diff = ndvi_2 - ndvi_1

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            im1 = axes[0].imshow(ndvi_1, cmap="YlGn", vmin=-0.2, vmax=0.8)
            axes[0].set_title(f"Status {jahr_1} ({date_1})")
            plt.colorbar(im1, ax=axes[0])

            im2 = axes[1].imshow(ndvi_2, cmap="YlGn", vmin=-0.2, vmax=0.8)
            axes[1].set_title(f"Status {jahr_2} ({date_2})")
            plt.colorbar(im2, ax=axes[1])

            im3 = axes[2].imshow(ndvi_diff, cmap="bwr", vmin=-0.5, vmax=0.5)
            axes[2].set_title("Veränderung (Rot = Rückgang | Blau = Zunahme)")
            plt.colorbar(im3, ax=axes[2])

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.error("Es konnten keine geeigneten Bilder mit geringer Bewölkung gefunden werden.")