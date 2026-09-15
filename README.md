# Sentinel-2 Satellite Change Detection

![Example output: NDVI change around Neusiedler See, 2021 vs 2024](change_detection_vergleich.png)

This tool compares two seasons of Sentinel-2 imagery and maps what changed:
vegetation (NDVI), open water (NDWI) or burn scars (NBR). It runs as a
Streamlit app or from the command line and can export the result as a GeoTIFF
for QGIS. The imagery comes from the Microsoft Planetary Computer, which is
free to use.

The image above was created with:

```bash
python main.py --baseline-year 2021 --comparison-year 2024 \
    --aoi "Neusiedler See, Austria" --max-scenes 4 --max-cloud 20 \
    --season 07-01:08-31
```

Both composites use 4 scenes (2021-07-06 to 2021-08-10 and 2024-07-30 to
2024-08-29). Their average dates are 24 days apart, so the tool prints a
phenology warning: part of the change on the fields is simply a different
point in the growing season.

## How it works

1. Search Sentinel-2 L2A scenes for the same season in both years.
2. Pick acquisition days so that every tile in the area gets the same dates.
3. Mask clouds, shadows and snow per pixel with the Scene Classification Layer (SCL).
4. Compute the index for every scene and take the median per pixel.
5. Subtract the baseline composite from the comparison composite.
6. Count pixels above a fixed or per-pixel threshold and export the result.

## Project structure

| File | Purpose |
| --- | --- |
| `ndvi_core.py` | Index calculation, offset correction, cloud mask, median, thresholds, statistics |
| `data_access.py` | Scene search and selection, loading both years onto one grid |
| `main.py` | Command line tool |
| `app.py` | Streamlit app |
| `plotting.py` | The three-panel figure |
| `export.py` | GeoTIFF export |
| `geocoding.py` | Place search (Nominatim) |
| `validation.py` | Checks against real events |
| `tests/` | Offline tests |

## Features

- Sentinel-2 L2A data from the Microsoft Planetary Computer
- Area selection by preset, place name (OpenStreetMap Nominatim) or bounding box
- NDVI, NDWI and NBR
- Per-pixel cloud masking and seasonal median composites
- Both years on one shared UTM grid
- Optional per-pixel threshold based on the scatter between scenes
- GeoTIFF export with an observation count band
- Validation against real events, including a negative control region
- 176 offline unit and integration tests, run in CI on Python 3.11 and 3.12

## Validation

Unit tests only show that the code is consistent with itself. To check it
against reality, `validation.py` runs the full pipeline on places where
something documented happened. The latest run is saved in
[validation_results.md](validation_results.md):

| Case | Index | Period | Result |
| --- | --- | --- | --- |
| Camp Fire, Paradise CA | NBR | 2018 → 2019 | 66.6 % of the area decreased, mean −0.252 |
| Same box, regrowth | NBR | 2019 → 2024 | 50.9 % increased, mean +0.143 |
| Kakhovka reservoir | NDWI | 2022 → 2024 | 92.7 % of the 2022 water is gone, mean −0.687 |
| South Aral Sea, east basin | NDWI | 2018 → 2024 | 98.6 % of the 2018 water is gone, mean −0.693 |
| Control: Great Sand Sea | NDVI | 2019 → 2024 | mean bias 0.0013, 0.0 % of pixels changed |

All 5 cases pass their thresholds (defined in `validation.py`).

How to read this:

- **The desert control matters most.** Nothing should change there. If the
  pipeline had a calibration bias, it would show up as a mean shift across the
  whole scene, so the control is judged on the mean delta and not only on how
  many pixels passed a threshold.
- **Both Camp Fire rows use the same box.** The same ground has to drop after
  the fire and recover later, so a sign error cannot pass both.
- **Water cases are scored only over pixels that were water in the baseline.**
  Scoring the whole box would mostly measure how much farmland the box happens
  to contain.

The first desert control I chose (28.4° E / 22.6° N) turned out to be the East
Uweinat irrigation project, which looks like empty desert at first glance. I
moved the control to open dune field in the Great Sand Sea. The Aral Sea case
needs a larger imagery budget (40 % cloud, 6 scenes) because there are
few clear scenes over that region.

The bounding boxes are approximate. Check them on a map before quoting any
number.

## Setup

```bash
git clone https://github.com/lubna-ishaq/satellite-change-detection.git
cd satellite-change-detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py                                 # the app
python main.py --help                                # the CLI
python validation.py --report validation_results.md  # real-data validation
```

## Spectral indices

All three are normalised differences, `(a - b) / (a + b)`:

| Index | Bands | Shows | Negative delta means |
| --- | --- | --- | --- |
| NDVI | B08, B04 | vegetation | vegetation lost |
| NDWI | B03, B08 | open water | water lost |
| NBR | B08, B12 | burn severity | burned |

The pipeline always computes *comparison minus baseline*, so a fire gives a
negative NBR delta. The dNBR used in most papers has the opposite sign.

## Problems I had to solve

These issues all produce results that look plausible but are wrong.

**Reflectance offset.** Since ESA processing baseline 04.00, L2A reflectance
is stored with an offset of -1000. Because an index is a ratio, the offset does
not cancel out. My first version ignored it, and the 2021 vs 2024 comparison
showed a smooth bias over the whole scene that looked like real change. The
pipeline now reads the processing baseline of each scene
(`s2:processing_baseline`) and only falls back to the acquisition date
(from 2022-01-25) when that property is missing. This also covers older scenes
that ESA has reprocessed.

**Cloud cover is reported per scene.** A scene with `eo:cloud_cover < 10` can
still be cloudy over the area you care about. Every pixel goes through the SCL
mask, which removes cloud, cirrus, shadow, snow, saturated and no-data pixels.
Water is kept on purpose.

**Different grids.** Scenes from different years can come from different tiles
and grids. Both seasons are loaded onto one grid in the local UTM zone, so pixel
(i, j) is the same ground in both years. If the shapes ever differ,
`calculate_index_delta` raises an error instead of broadcasting. I use UTM
because a "20 m" pixel in Web Mercator (EPSG:3857) is only about 13.6 m on the
ground at 47° N.

**Tile edges.** Areas can span several Sentinel-2 tiles. An earlier version let
each tile pick its own clearest scenes, so each tile had different dates and
the delta showed rectangular blocks along the tile edges. This was most
visible over water. Scene selection now works on acquisition days: if there are
days on which every tile has a usable scene, only those days are used, even if
that means fewer scenes. Only when no such day exists does each tile get its
own dates.

Two smaller points: NaN is used for missing data everywhere, never 0, because 0
is a valid NDVI for bare soil. And the index is a median over several scenes,
because a single date mostly reflects that day's weather.

## Reliability layers

Each composite also stores how many scenes reached each pixel and how much
those scenes disagreed (median absolute deviation).

- The observation count is written as band 2 of the GeoTIFF, so you can see
  which pixels are based on only one scene.
- The scatter is used by `--adaptive-threshold`, which replaces the fixed ±0.1
  with `sigma × scatter` per pixel. Pixels with too few observations get the
  scene's typical threshold, not the lower floor.
- The CLI and the app warn when the average dates of the two composites are more
  than 21 days apart, because then part of the delta is seasonal growth.

## Export

Use `--geotiff PATH` on the CLI or the download button in the app. The file is
Float32 with CRS, transform, NaN as no-data, processing metadata in the tags,
and the observation count as band 2.

```bash
python main.py --index NBR --bbox -121.70 39.68 -121.50 39.86 \
    --baseline-year 2018 --comparison-year 2019 --geotiff camp_fire.tif
```

## CLI options

| Flag | Default | Notes |
| --- | --- | --- |
| `--baseline-year` | 2021 | |
| `--comparison-year` | 2024 | |
| `--index` | NDVI | NDVI, NDWI or NBR |
| `--season` | `06-01:08-31` | used for both years |
| `--aoi` | Graz, Austria | see `data_access.AOI_PRESETS` |
| `--bbox W S E N` | | EPSG:4326, replaces `--aoi` |
| `--resolution` | 20 | metres |
| `--max-cloud` | 10 | percent, per scene |
| `--max-scenes` | 6 | acquisition days per season |
| `--threshold` | 0.1 | |
| `--adaptive-threshold` | off | per-pixel threshold |
| `--sigma` | 2.0 | |
| `--threshold-floor` | 0.05 | |
| `--geotiff` | | also write a GeoTIFF |

## Deployment

The app can run on Streamlit Community Cloud: push the repo, choose `app.py`
and deploy. `requirements.txt` is used as the build manifest, and
rasterio and pyproj ship GDAL and PROJ in their wheels.

Memory is the main limit there (about 1 GB). The grid is capped at 40
megapixels (`data_access.MAX_PIXELS`), but every scene of a composite is held
in memory at once, so the peak grows with the number of scenes (roughly
pixels × scenes × 37 bytes). The app shows a warning when a request is likely
to exceed 1 GB, and the CLI prints the estimate.

## Limitations

- Indices saturate: over dense forest, large biomass changes barely move NDVI.
- Sun angle and plant growth differences remain after compositing. A delta
  from one season is a candidate for change, not proof.
- There is no terrain or BRDF correction, so steep slopes keep some
  illumination bias.
- SCL misses thin cloud edges and has trouble with shadows over water.
- The default season is northern-hemisphere summer. Use `--season` elsewhere.
- Place search uses the public Nominatim service, which is meant for occasional
  interactive requests only.

## Development

The tests run fully offline. STAC search, image loading and geocoding are
replaced with stubs. `validation.py` needs real network access and is therefore
not part of CI.

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest --cov
```

## Possible next steps

- Trends over more than two years
- More spectral indices
- Before/after slider in the app

## Stack

Python 3.11+, numpy, rasterio, odc-stac, odc-geo, pystac-client,
planetary-computer, matplotlib, streamlit, folium.

## License

MIT, see [LICENSE](LICENSE).
