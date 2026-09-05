# Sentinel-2 Satellite Change Detection

[![CI Tests](https://github.com/lubna-ishaq/satellite-change-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/lubna-ishaq/satellite-change-detection/actions/workflows/tests.yml)

![Satellite Change Detection Output](change_detection_vergleich.png)

Compares two seasons of Sentinel-2 imagery and shows what changed. Vegetation
(NDVI), open water (NDWI) or burn scars (NBR). Runs as a Streamlit app or from
the command line, and exports a GeoTIFF you can drop into QGIS.

Imagery comes from the Microsoft Planetary Computer, which is free and needs no
account.

The picture above:

```bash
python main.py --baseline-year 2021 --comparison-year 2024 \
    --aoi "Neusiedler See, Austria" --max-scenes 4
```

## Setup

```bash
git clone https://github.com/lubna-ishaq/satellite-change-detection.git
cd satellite-change-detection

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py     # the app
python main.py --help    # the CLI
python validation.py     # check it against real events
```

## The indices

All three are normalised differences, `(a - b) / (a + b)`:

| | Bands | What it shows | A negative delta means |
| --- | --- | --- | --- |
| NDVI | B08, B04 | vegetation | vegetation lost |
| NDWI | B03, B08 | open water | water lost |
| NBR | B08, B12 | burn severity | burned |

Careful with NBR: this pipeline always computes *comparison minus baseline*, so
a fire is a negative number. The dNBR you see in papers is the other way round.

## The things that will bite you

Most of the work here went into four problems. They all produce output that
looks completely reasonable and is wrong.

**The reflectance offset.** Since ESA Processing Baseline 04.00 — anything
acquired from 2022-01-25 — L2A reflectance is stored with -1000 added to it. An
index is a ratio, so this does *not* cancel out. Compare 2021 against 2024
without fixing it and you get a smooth bias across the whole scene that looks
exactly like real land cover change. This was the first version of this project
and the reason the header image had to be redone.

**Cloud cover is per scene, not per pixel.** `eo:cloud_cover < 10%` tells you
nothing about the pixel you care about. Everything goes through the Scene
Classification Layer: cloud, cirrus, shadow, snow, saturated, no-data all get
dropped. Water is kept deliberately, since a shrinking lake is the point.

**Two years, two grids.** Scenes from different years can sit on different
tiles and different native grids. Both seasons get loaded onto one GeoBox in
the local UTM zone, so pixel (i, j) is the same patch of ground in both. If
that ever breaks, `calculate_index_delta` raises instead of quietly
broadcasting garbage. UTM and not Web Mercator, by the way — a "20 m" pixel in
EPSG:3857 is about 13.6 m on the ground at 47° N.

**Scene budgets are per tile.** This one cost me an afternoon. If your area is
wider than about 110 km it spans several Sentinel-2 tiles, and sorting all
candidates by cloud cover can hand you four scenes that all belong to the same
tile. The rest of your area then has no data at all, gets masked out, and the
result still looks fine — until you notice the valid-pixel fraction is 23 % and
there's a suspiciously straight edge in the delta.

Two smaller things: NaN is the no-data value everywhere, never 0 (0 is a
perfectly good NDVI for bare soil, and using it as a flag poisons every
average), and the index is a median across several scenes rather than a single
date, because one date mostly measures that day's weather.

## Knowing how much to trust a pixel

Each composite carries two extra layers: how many scenes actually reached each
pixel, and how much those scenes disagreed.

The count ships as band 2 of the GeoTIFF. A pixel built from one observation
looks identical to one built from six in the composite itself, which is not
great.

The disagreement feeds `--adaptive-threshold`, which swaps the fixed ±0.1 for
`sigma × scatter` per pixel. One constant is wrong twice over: too low on noisy
bare ground, too high over a stable canopy where a small real change gets
buried. Pixels with too few observations to measure scatter get the scene's
typical threshold rather than the floor — giving the worst-observed pixels the
easiest bar would be backwards.

The composites also report their mean day-of-year, and warn if the two are more
than three weeks apart. A mid-June composite against a late-August one is
measuring the growing season at least as much as it's measuring change.

## Does it actually work

Unit tests only prove the arithmetic is consistent with itself. `validation.py`
runs the whole thing against places where something documented happened:

| Case | Index | Period | Result |
| --- | --- | --- | --- |
| Camp Fire, Paradise CA | NBR | 2018 → 2019 | 66.3 % of the area dropped, mean −0.251 |
| Same box, regrowth | NBR | 2019 → 2024 | 51.0 % recovered, mean +0.144 |
| Kakhovka reservoir | NDWI | 2022 → 2024 | 93.0 % of the 2022 water is gone, mean −0.711 |
| South Aral Sea, east basin | NDWI | 2018 → 2024 | 98.6 % of the 2018 water is gone, mean −0.695 |
| Control: Great Sand Sea | NDVI | 2019 → 2024 | mean bias 0.0065, zero pixels changed |

The last row is the one I'd look at first. It's empty desert, nothing should
change there, and a pipeline with a leftover calibration bias will cheerfully
report change anyway. It's judged on the mean, not on how many pixels crossed
a threshold — a 0.05 drift across the whole scene crosses no per-pixel ±0.1
rule at all, but it's exactly the kind of bug that matters.

The two Camp Fire rows use the same bounding box on purpose. The same ground
has to drop after the fire and recover five seasons later, so a sign error
can't pass both.

For the water cases the score is over pixels that *were* water in the baseline,
not over the whole box. Scoring the whole box mostly measures how much farmland
you happened to include; pad the box and the same real event scores worse.

Two things I got wrong and had to fix: the control originally sat at
28.4° E / 22.6° N, which looks like empty desert on a satellite image but is
actually the East Uweinat centre-pivot irrigation scheme, airport and all. A
control standing on farmland is useless. And the Aral Sea case needs a bigger
imagery budget than the others (40 % cloud, 6 scenes per tile) — there just
aren't many clean scenes over that region.

The boxes are approximate. Check them on a map before quoting any number.

## Exporting

`--geotiff PATH` on the CLI, or the download button in the app. Float32, proper
CRS and transform, NaN nodata, provenance in the tags, plus the observation
count as a second band.

```bash
python main.py --index NBR --bbox -121.70 39.68 -121.50 39.86 \
    --baseline-year 2018 --comparison-year 2019 --geotiff camp_fire.tif
```

## CLI flags

| Flag | Default | |
| --- | --- | --- |
| `--baseline-year` | 2021 | |
| `--comparison-year` | 2024 | |
| `--index` | NDVI | NDVI, NDWI or NBR |
| `--season` | `06-01:08-31` | applied to both years |
| `--aoi` | Graz, Austria | see `data_access.AOI_PRESETS` |
| `--bbox W S E N` | | EPSG:4326, overrides `--aoi` |
| `--resolution` | 20 | metres |
| `--max-cloud` | 10 | percent, scene level |
| `--max-scenes` | 6 | per tile |
| `--threshold` | 0.1 | |
| `--adaptive-threshold` | off | per-pixel threshold instead |
| `--sigma` | 2.0 | |
| `--threshold-floor` | 0.05 | |
| `--geotiff` | | also write a GeoTIFF |

## Deploying

Streamlit Community Cloud, free: push the repo, pick `app.py` as the main file,
deploy. `requirements.txt` is the build manifest, `.streamlit/config.toml` has
the theme. No `packages.txt` needed — rasterio and pyproj bundle GDAL and PROJ
in their wheels.

Hosted instances run out of memory easily, so requests are capped at 40
megapixels (`data_access.MAX_PIXELS`). Anything bigger gets refused with a note
suggesting a coarser resolution.

## What this won't do

- Indices saturate. Over dense canopy a big change in biomass barely moves NDVI.
- Sun angle and phenology survive compositing. Treat a single-season delta as
  a *candidate* for change, not proof of it.
- No topographic or BRDF correction, so steep terrain keeps an illumination
  bias.
- SCL misses thin cloud edges and struggles with shadow over water.
- The default season is northern-hemisphere summer. Use `--season` elsewhere.
- Place search goes through OSM Nominatim, which is for interactive lookups
  only — don't hammer it.

## Development

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest --cov
```

157 tests, no network needed — STAC, imagery and geocoding are all stubbed.
`validation.py` is the only part that goes online, and it's deliberately kept
out of CI.

## Stack

Python 3.11+, numpy, rasterio, odc-stac, odc-geo, pystac-client,
planetary-computer, matplotlib, streamlit, folium.

## License

MIT, see [LICENSE](LICENSE).
