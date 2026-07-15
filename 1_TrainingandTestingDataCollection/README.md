# 1_TrainingandTestingDataCollection

This folder now keeps both data-collection workflows relevant to the revised manuscript.

## Legacy_LA_Submission

This subfolder preserves the original Los Angeles training/testing data collection scripts from the manuscript-era code bundle:

- `1_CreateTraining.py`
- `1_CreateTesting_seed_number.py`

These are preserved without rewriting so the original workflow remains traceable.

## Revision_CONUS_Stratified_ExactFootprints

This subfolder contains the revised CONUS-scale archive builder based on:

- raw Model America v1 footprint CSVs
- exact polygon parsing and projection into the NLCD CRS
- deterministic MSA split into train and held-out test
- 2D BF x built-height stratified candidate discovery
- exact polygon-pixel overlap during final raster rendering

The curated scripts here are copied from the active revision workflow under:

`Report/GIScRSSubmission/Revision/codev3/Stratified_Archive`

Only scripts and documentation are curated here. Generated outputs, caches, and archives are intentionally not duplicated into this GitHub-facing folder.
