# GANUrbanMorph Revision Code

### ReadMe for Python and R Code used in the Revised GAN Urban Morphology Study
### Author: Bhartendu Pandey
### Directory Structure:
```
Root
├── 1_TrainingandTestingDataCollection
│   ├── Legacy_LA_Submission
│   │   ├── 1_CreateTesting_seed_number.py
│   │   └── 1_CreateTraining.py
│   └── Revision_CONUS_Stratified_ExactFootprints
│       ├── build_stratified_test_archive.py
│       ├── build_stratified_training_archive.py
│       ├── ma_exact_footprints.py
│       ├── prepare_ma_state_polygon_cache.py
│       ├── run_full_exact_archive_pipeline.sh
│       ├── run_full_exact_test_archive_pipeline.sh
│       └── stratified_archive_pipeline.py
├── 2_ModelTraining
│   ├── LALegacy
│   │   ├── 1_BF_UNETBaseline.ipynb
│   │   ├── 1_BH_UNETBaseline.ipynb
│   │   ├── 2A_BF_cGANRandomVecFixed.ipynb
│   │   ├── 2A_BH_cGANRandomVecFixed.ipynb
│   │   ├── 3_BF_cGANMultiRandomDiversity.ipynb
│   │   └── 3_BH_cGANMultiRandomDiversity.ipynb
│   └── MSASample
│       ├── 1_BF_UNETBaseline_MSASample.ipynb
│       ├── 1_BH_UNETBaseline_MSASample.ipynb
│       ├── 2A_BF_cGANRandomVecFixed_MSASample.ipynb
│       ├── 2A_BH_cGANRandomVecFixed_MSASample.ipynb
│       ├── 2A_BH_cGANRandomVecFixed_MSASample_0.001Rerun.ipynb
│       ├── 3_BF_cGANMultiRandomDiversity_MSASample.ipynb
│       └── 3_BH_cGANMultiRandomDiversity_MSASample.ipynb
├── 3_GANValidation
│   ├── HeightBiasbyLandUseType
│   │   ├── 01_aggregate_height_bias_by_hisdac_land_use.py
│   │   └── 02_plot_height_bias_by_hisdac_land_use.R
│   ├── ManuscriptFigures
│   │   ├── make_epoch1000_focus_figures.py
│   │   ├── make_maintext_error_bias_scatter.py
│   │   ├── make_maintext_morans_error_bias_scatter.py
│   │   ├── make_osm_context_images.py
│   │   ├── make_representative_high_density_highrise_examples.py
│   │   └── make_single_vs_multi_diversity_figure.py
│   ├── OverallValidation
│   │   ├── OverallValidation_Colab_A100.ipynb
│   │   ├── run_overall_validation.py
│   │   ├── validation_config.json
│   │   └── validation_core.py
│   └── SpatialClustering
│       ├── SpatialClustering_Colab_A100.ipynb
│       ├── run_spatial_clustering.py
│       ├── spatial_clustering_config.json
│       └── spatial_clustering_core.py
├── 4_Inference
├── 5_UrbanScaling
│   ├── 00_prepare_data.R
│   ├── 01_compute_building_volume_population_scaling.R
│   ├── 02_plot_building_volume_population_scaling.R
│   ├── 03_compute_taylor_power_law_scaling.R
│   └── 04_plot_taylor_power_law_scaling.R
├── BPspatlibv0.py
└── BPspatlibv1.py
```

### Description:

1_TrainingandTestingDataCollection: This folder contains the data collection scripts used for the revised manuscript. `Legacy_LA_Submission` preserves the Los Angeles-centered scripts from the original code bundle. `Revision_CONUS_Stratified_ExactFootprints` contains the revised CONUS-scale stratified archive builders that create the CONUS Model training archive and the held-out CONUS Test Dataset using MSA-disjoint sampling, BF/BH stratification, and exact footprint rasterization.

2_ModelTraining: This folder contains notebooks for the three active model classes used in the revision: U-Net baseline (`1`), single-latent cGAN (`2A`), and multi-latent/diversity cGAN (`3`). Separate notebooks are provided for the Los Angeles Model (`LALegacy`) and the CONUS Model (`MSASample`), for both building footprint fraction (BF) and building height (BH). The corrected CONUS single-latent BH learning-rate `0.001` run is represented by `2A_BH_cGANRandomVecFixed_MSASample_0.001Rerun.ipynb`.

3_GANValidation: This folder contains the revised validation workflows. `OverallValidation` computes point metrics, tile-average metrics, tile-heterogeneity metrics, BH oracle-vs-pipeline comparisons, and stochastic diversity metrics across model classes, training regimes, held-out datasets, learning rates, and checkpoints. `SpatialClustering` computes tile-level Moran's I for BF and BH to evaluate spatial autocorrelation. `ManuscriptFigures` regenerates the main manuscript figures from the validation outputs. `HeightBiasbyLandUseType` evaluates CONUS Model BH underprediction across HISDAC 2015 land-use classes in the CONUS Test Dataset.

4_Inference: This folder is reserved for inference code and outputs associated with morphology growth analyses. The revised manuscript-facing scaling workflows consume the morphology-growth evaldfs and metrics documented under `5_UrbanScaling`.

5_UrbanScaling: This folder contains the revised building volume-population scaling and Taylor's Power Law scaling workflows. `00_prepare_data.R` prepares local inputs and manifests. `01_compute_building_volume_population_scaling.R` and `02_plot_building_volume_population_scaling.R` compute and plot uncorrected epoch-1000 building-volume growth predictions for U-Net and single-latent cGAN models. `03_compute_taylor_power_law_scaling.R` and `04_plot_taylor_power_law_scaling.R` compute and plot Taylor's Power Law consistency for the same model configurations.

BPspatlibv0.py and BPspatlibv1.py contain helper functions referenced by the training and data collection scripts.

### Notes for GitHub:

This repository should track the reproducible scripts, configuration files, and concise documentation. Large input datasets, generated archives, model checkpoints, and large generated result tables should be stored outside GitHub, for example through Zenodo or local data paths documented in each workflow. The `.gitignore` in this folder excludes generated outputs, local caches, large raster/vector inputs, archive files, and macOS metadata files that should not be committed to the code repository.
