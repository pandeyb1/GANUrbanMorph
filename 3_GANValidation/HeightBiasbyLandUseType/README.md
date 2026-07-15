# Height Bias by HISDAC Land-Use Type

This workflow evaluates building-height bias across HISDAC 2015 land-use classes
for CONUS-trained models on the CONUS Test Dataset. It is designed to address
the reviewer comment asking how BH bias varies with land-use type.

## Scope

- Dataset: CONUS Test Dataset, 2,000 held-out 2015 tiles.
- Training regime: CONUS Model (`MSASample`) only.
- Model classes: U-Net baseline (`1`) and single-latent cGAN (`2A`) only.
- Target: BH only.
- Learning rates: `0.0001`, `0.0002`, `0.0005`, `0.001`.
- Checkpoint: epoch 1000.
- Evaluation mode: `oracle_bf`, meaning the BH generator is conditioned on
  original/reference BF, not generated BF.

## Inputs

- CONUS test archive:
  `/Users/9oy/Documents/Projects/IM3/EvaluationP/Report/GIScRSSubmission/Revision/codev3/Stratified_Archive/output_test`
- HISDAC majority land use:
  `/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/HISDAC/Majority/Majority/Majority_2015.tif`
- CONUS model root:
  `/Volumes/HDD/Models/MSASampleModels`

The model drive must be mounted before running the aggregation script.

## Run Order

From this directory:

```bash
conda run -n pytorch python 01_aggregate_height_bias_by_hisdac_land_use.py
Rscript 02_plot_height_bias_by_hisdac_land_use.R
```

## Outputs

- `tables/height_bias_by_hisdac_land_use_tile_stats_epoch1000_conus_unet_single_latent_oracle_bf.csv`
- `tables/height_bias_by_hisdac_land_use_summary_epoch1000_conus_unet_single_latent_oracle_bf.csv`
- `tables/height_bias_by_hisdac_land_use_model_source_manifest.csv`
- `tables/height_bias_by_hisdac_land_use_overlay_coverage.csv`
- `figures/height_bias_by_hisdac_land_use_epoch1000_conus_unet_single_latent_oracle_bf.png`
- `figures/height_bias_by_hisdac_land_use_epoch1000_conus_unet_single_latent_oracle_bf.pdf`
- `text/height_bias_by_hisdac_land_use_results.md`

## Validation Notes

- The script fails if the CONUS single-latent BH `lr=0.001` model does not use
  `lr_0p001_seed_2026`.
- The script fails if `lr_0p001_seed_5026` is selected for the corrected model
  case.
- Reference BH is converted from feet to meters.
- Generated BH is decoded from the log1p-normalized model output to meters.
- The main figure uses pixels with `reference_bh_m > 0.5`.
