# Sentinel-2 Satellite Change Detection System

An end-to-end geospatial data processing pipeline and web interface for detecting vegetation index shifts and land cover alterations over time using European Space Agency (ESA) Sentinel-2 multispectral satellite imagery.

## Overview

This application queries raw satellite scene metadata from the Microsoft Planetary Computer STAC API to load red and near-infrared (NIR) spectral bands. It calculates the Normalized Difference Vegetation Index (NDVI) across different acquisition periods and computes a spatial difference matrix to highlight environmental changes such as agricultural shifts, water body fluctuations, or seasonal droughts.

## Core Concepts: Baseline vs. Comparison Selection

The change detection matrix is computed by contrasting two distinct single-year satellite scenes taken during peak vegetation periods (June 1 – August 31):

* **Baseline Year (Reference State):** Represents the historical reference point selected from the primary dropdown options (2020, 2021, or 2022). It establishes the initial environmental baseline.
* **Comparison Year (Target State):** Represents the recent observation point selected from the secondary dropdown options (2023, 2024, or 2025).
* **NDVI Delta Matrix:** Calculated directly as $\Delta NDVI = NDVI_{\text{Target}} - NDVI_{\text{Baseline}}$.
  * **Positive values (Blue):** Vegetation gain, reforestation, or increased moisture.
  * **Negative values (Red):** Vegetation loss, land clearing, or reduced surface water.

## Technical Architecture

* **Language:** Python 3.12
* **Geospatial Processing:** rasterio, odc-stac, pystac-client, planetary-computer
* **Data Processing & Analytics:** numpy, matplotlib
* **Web Interface:** streamlit

## Mathematical Methodology

The Normalized Difference Vegetation Index (NDVI) leverages the differential reflection of vegetation across spectral bands:

$$NDVI = \frac{NIR - Red}{NIR + Red}$$

* **Band 4 (Red):** Absorbance by chlorophyll.
* **Band 8 (Near-Infrared / NIR):** High reflectance by leaf mesophyll structure.

Values range from -1.0 to +1.0:
* **Negative values:** Water bodies and snow cover.
* **Near 0.0:** Bare soil, rock, or urban surfaces.
* **High positive values (+0.4 to +0.8):** Moderate to dense vegetation canopy.

## Project Structure

```text
.
├── main.py        # Standalone script for automated pipeline execution and image generation
├── app.py         # Interactive Streamlit web application
├── .gitignore     # Git exclusion rules
└── README.md      # Project documentation
