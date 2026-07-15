# Revised Urban Scaling Analyses

This folder contains the manuscript-ready revised analyses for the urban
scaling results sections. The workflows compare uncorrected epoch-1000
morphology-growth outputs from:

- U-Net baseline (`1`) and single-latent cGAN (`2A`)
- Los Angeles Model (`LALegacy`) and CONUS Model (`MSASample`)
- Learning rates `0.0001`, `0.0002`, `0.0005`, and `0.001`

The generated building-volume increment is computed as:

```text
predbf * predbh * 30 m * 30 m
```

No bias correction or post-hoc rescaling is applied.

The left-panel red Los Angeles marker is an overlaid LA urban-area estimate,
not the Los Angeles MSA record used in the 2015 scaling fit. The
Los Angeles-Long Beach-Anaheim MSA remains in the gray MSA point cloud from
`BuildingVolume2015MSA.gpkg`. Diagnostic tables retain both meter-consistent
LA urban-area totals and the legacy raster-derived totals that explain why the
older figure's red LA urban-area point was above `10^10`.

LandScan population sums use a cell-center inclusion rule (`touches = FALSE` in
`terra::mask`) to reproduce the older `raster::mask` behavior used in the
previous figure. The touched-cell alternative is written to
`tables/landscan_population_zonal_diagnostics.csv` for comparison.

## Building Volume-Population Scaling Run Order

From the project root:

```bash
Rscript Script/Revision/5_UrbanScaling/00_prepare_data.R
Rscript Script/Revision/5_UrbanScaling/01_compute_building_volume_population_scaling.R
Rscript Script/Revision/5_UrbanScaling/02_plot_building_volume_population_scaling.R
```

## Taylor's Power Law Scaling Run Order

From the project root, after the data-prep step:

```bash
Rscript Script/Revision/5_UrbanScaling/00_prepare_data.R
Rscript Script/Revision/5_UrbanScaling/03_compute_taylor_power_law_scaling.R
Rscript Script/Revision/5_UrbanScaling/04_plot_taylor_power_law_scaling.R
```

The Taylor workflow preserves the previous analysis logic: it fits
`log(Variance) ~ log(Mean)` from `HeteroScalingLA.csv`, reconstructs
256 x 256-window generated building-volume arrays from the paired evaldfs using
`ChangeMask2010_2020.tif`, and evaluates observed/generated variance against
the expected Taylor variance. Expected variance is computed from the fitted
Taylor relationship with 95% confidence intervals.

## Building Volume-Population Outputs

- `tables/building_volume_growth_epoch1000_unet_single_latent_no_bias_correction.csv`
- `tables/population_scaling_fit.csv`
- `tables/landscan_population_zonal_diagnostics.csv`
- `tables/model_source_manifest.csv`
- `figures/building_volume_population_scaling_epoch1000_unet_single_latent_no_bias_correction.png`
- `figures/building_volume_population_scaling_epoch1000_unet_single_latent_no_bias_correction.pdf`
- `text/building_volume_population_scaling_results.md`

## Taylor's Power Law Outputs

- `tables/taylor_power_law_fit.csv`
- `tables/taylor_power_law_window_metrics_epoch1000_unet_single_latent_no_bias_correction.csv`
- `tables/taylor_power_law_model_summary_epoch1000_unet_single_latent_no_bias_correction.csv`
- `tables/taylor_power_law_reference_summary.csv`
- `figures/taylor_power_law_scaling_epoch1000_unet_single_latent_no_bias_correction.png`
- `figures/taylor_power_law_scaling_epoch1000_unet_single_latent_no_bias_correction.pdf`
- `text/taylor_power_law_scaling_results.md`

## Provenance Guard

The population-scaling and Taylor computation scripts hard-fail unless the
CONUS Model, single-latent cGAN, BH, learning-rate `0.001` source is:

```text
/Volumes/HDD/Models/MSASampleModels/2A_BH_cGANRandomVecFixed_MSASample/lr_0p001_seed_2026/generator_epoch_1000.pth
```

It also fails if the corresponding source contains `lr_0p001_seed_5026`.

## Data Policy

The workflow copies the derived MSA building-volume table, NHGIS inputs,
LandScan raster inputs, LA morphology rasters, `HeteroScalingLA.csv`, and model
evaldfs/metrics into `data/`. The 19 GB
`Script/5_UrbanScaling/Data/MetroBuildings` directory is not duplicated; it is
documented in `data/manifests/external_data_manifest.csv`.
