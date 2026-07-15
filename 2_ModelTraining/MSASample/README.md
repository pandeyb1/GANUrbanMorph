# MSASample Training Notebooks

This folder contains copies of the LA legacy training notebooks retargeted to the revised CONUS MSA-sampled training archive.

Notebook logic is intentionally unchanged apart from:

- `DRIVE_ARCHIVE_ZIP` pointing to `Archive_CONUS_M1_random_stratified_2015.zip`
- `LOCAL_ARCHIVE_ZIP` using a distinct local filename
- notebook filenames gaining an `_MSASample` suffix to distinguish them from the LA notebooks in Colab
- `OUTPUT_SUBDIR` gaining an `_MSASample` suffix to avoid overwriting LA-trained runs
- `OUTPUT_ROOT` moving to `/content/drive/MyDrive/IM3/EvalP1/revision_msa_sample`

Expected archive contents after extraction remain the same:

- `central/`
- `BFrac/`
- `BHeight/`
