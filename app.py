import streamlit as st
import matplotlib.pyplot as plt
import planetary_computer as pc
from pystac_client import Client
import odc.stac

from ndvi_core import calculate_ndvi, calculate_ndvi_delta

st.title("Sentinel-2 Satellite Change Detection System")

st.sidebar.header("Parameters")
baseline_year = st.sidebar.selectbox("Baseline Year", [2020, 2021, 2022], index=0)
comparison_year = st.sidebar.selectbox("Comparison Year", [2023, 2024, 2025], index=0)

if st.sidebar.button("Run Analysis"):
    st.info("Querying Satellite Scenes from STAC API...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    
    # Load Baseline
    search_base = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[15.35, 47.01, 15.50, 47.12],
        datetime=f"{baseline_year}-06-01/{baseline_year}-08-31",
        query={"eo:cloud_cover": {"lt": 10}},
    )
    items_base = list(search_base.items())
    
    # Load Comparison
    search_comp = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[15.35, 47.01, 15.50, 47.12],
        datetime=f"{comparison_year}-06-01/{comparison_year}-08-31",
        query={"eo:cloud_cover": {"lt": 10}},
    )
    items_comp = list(search_comp.items())
    
    if items_base and items_comp:
        ds_base = odc.stac.load([items_base[0]], bands=["B04", "B08"], resolution=20, patch_url=pc.sign_url)
        ds_comp = odc.stac.load([items_comp[0]], bands=["B04", "B08"], resolution=20, patch_url=pc.sign_url)
        
        ndvi_base = calculate_ndvi(ds_base["B04"].squeeze().values, ds_base["B08"].squeeze().values)
        ndvi_comp = calculate_ndvi(ds_comp["B04"].squeeze().values, ds_comp["B08"].squeeze().values)
        delta = calculate_ndvi_delta(ndvi_base, ndvi_comp)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader(f"Baseline ({baseline_year})")
            fig, ax = plt.subplots()
            ax.imshow(ndvi_base, cmap="YlGn")
            st.pyplot(fig)
        with col2:
            st.subheader(f"Comparison ({comparison_year})")
            fig, ax = plt.subplots()
            ax.imshow(ndvi_comp, cmap="YlGn")
            st.pyplot(fig)
        with col3:
            st.subheader("NDVI Delta")
            fig, ax = plt.subplots()
            im = ax.imshow(delta, cmap="RdBu", vmin=-0.5, vmax=0.5)
            st.pyplot(fig)
    else:
        st.error("Could not find suitable low-cloud satellite scenes for the selected timeframe.")