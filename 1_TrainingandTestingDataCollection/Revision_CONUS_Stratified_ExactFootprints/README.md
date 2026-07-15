# Stratified CONUS BF/BH Archive Builders

This workflow now has two distinct stages and two distinct dataset builders:

- a reusable per-state exact-footprint cache builder rooted in raw MAv1 CSVs
- a training archive builder
- a held-out test archive builder

The design is now explicitly MSA-disjoint:

- San Diego is excluded from both train and test archive construction
- a deterministic MSA-level split is written to `msa_role_split.csv`
- the training archive is built only from `dataset_role == train`
- the internal CONUS test archive is built only from `dataset_role == test`

This keeps the train/test split free of tile leakage while preserving San Diego as a separate external comparison city.

The stratification key is now a 2D built-intensity by built-height grid:

- `bf_mean` bins: `0.01-0.05`, `0.05-0.15`, `0.15+`
- `bh_built_mean_m` bins: `<6`, `6-10`, `10-15`, `15+`
- `stratum_id = (bf_bin, height_bin)`

Quotas are assigned with a balanced hybrid rule:

- every populated stratum gets a minimum floor
- remaining tiles are allocated in proportion to residual candidate availability
- fallback scan only targets strata whose candidate pool remains below `2x` the provisional target

## Scripts

- `prepare_ma_state_polygon_cache.py`
- `build_stratified_training_archive.py`
- `build_stratified_test_archive.py`
- `stratified_archive_pipeline.py`

## Stage 1: Build per-state exact-footprint caches

The cache builder is unchanged in purpose:

- stream raw MAv1 CSVs in chunks
- parse `Footprint2D` polygons
- repair invalid geometries only when needed
- project into the NLCD CRS using the raster proj4 string
- write one `GPKG` per state under `state_cache/`

Default run:

```bash
conda run --no-capture-output -n gis python prepare_ma_state_polygon_cache.py
```

## Stage 2: Build train/test archives from the shared cache

The archive builders now use a two-speed strategy:

1. **Fast candidate scan**
   - parallelized at the MSA level
   - uses exact building-tile overlap area, but does **not** do pixel-by-pixel exact rendering during scan
   - stratifies on `bf_mean` and `bh_built_mean_m`
   - writes resumable per-MSA scan chunks

2. **Exact final render**
   - runs only on the final selected tiles
   - uses exact polygon-pixel overlap to write `central/`, `BFrac/`, and `BHeight/`
   - writes `tile_manifest_rendered.csv` with exact post-render tile statistics

This refactor keeps the scientifically important exact render while removing the expensive exact pixel loop from the candidate-discovery phase.

## Train/test split

The deterministic split is written to:

```text
msa_role_split.csv
```

Defaults:

- `master_seed = 2026`
- `test_fraction = 0.25`
- `min_test_msas = 80`

The training builder uses the remaining MSAs after the held-out test assignment.

## Archive targets

Training archive:

- `8,000` tiles total
- minimum floor: `80` per populated 2D stratum
- remaining quota: proportional to residual candidate availability

Held-out test archive:

- `2,000` tiles total
- minimum floor: `20` per populated 2D stratum
- remaining quota: proportional to residual candidate availability

Both builders enforce per-MSA total caps and dynamic per-stratum caps so a small number of metros do not dominate the archive.

## Recommended commands

Training archive, scan only:

```bash
conda run --no-capture-output -n gis python \
  build_stratified_training_archive.py \
  --stage scan \
  --workers 6
```

Training archive, full run:

```bash
conda run --no-capture-output -n gis python \
  build_stratified_training_archive.py \
  --stage all \
  --workers 6
```

Held-out test archive, full run:

```bash
conda run --no-capture-output -n gis python \
  build_stratified_test_archive.py \
  --stage all \
  --workers 6
```

## Output layout

Training output defaults to:

```text
output/
```

Held-out test output defaults to:

```text
output_test/
```

Each output directory contains:

```text
central/
BFrac/
BHeight/
scan_chunks/
render_metrics_by_msa/
candidate_tiles.csv
candidate_bin_counts.csv
msa_selection.csv
tile_manifest.csv
tile_manifest_rendered.csv
collection_manifest.json
msa_role_split.csv
```

The key stratification fields are:

- `candidate_tiles.csv`: `bf_bin`, `height_bin`, `stratum_id`
- `tile_manifest.csv`: selected scan-time `stratum_id`
- `tile_manifest_rendered.csv`: `scan_bf_bin`, `scan_height_bin`, `scan_stratum_id`

## Notes

- `Area2D` is retained only for cache QA. In the raw MAv1 CSVs it behaves like square feet, so the cache builder converts it before comparing it to exact projected polygon area.
- The builders use the NLCD raster proj4 string directly for target CRS handling.
- The scan phase is resumable at the per-MSA chunk level.
- The render phase is also resumable at the per-MSA metrics-file level.
- Exact polygon-pixel overlap is used only in the final render stage, not in scan-time stratification.
