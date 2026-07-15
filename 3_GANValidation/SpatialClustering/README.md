# Spatial Clustering Validation

This standalone bundle addresses the reviewer concern that pixel-wise error metrics do not directly evaluate spatial pattern realism. It computes tile-level Moran's I for reference and generated rasters, then summarizes whether generated building footprint (BF) and building height (BH) fields reproduce the spatial autocorrelation structure of the held-out CONUS and San Diego test datasets.

## Scope

- Model classes: `1`, `2A`, and `3`.
- Training regimes: `LALegacy` and `MSASample`.
- Test datasets: `CONUSStratifiedTest` and `SanDiegoTestNoOverlap`.
- Learning rates: `0.0001`, `0.0002`, `0.0005`, and `0.001`.
- Checkpoint: `1000` only.
- BF evaluation: standard BF generator output.
- BH evaluation: reference-BF-conditioned BH generator output (`oracle_bf` internally), matching the main model-class figures.
- Spatial statistic: Moran's I with first-order rook contiguity over each raster tile.

## Rationale

Moran's I measures whether neighboring pixels have similar values. For this analysis, each reference tile and generated tile receives a Moran's I value. Generated outputs are then compared to the reference values using Moran's I mean absolute error, mean bias, regression slope, and R² across tiles. A negative Moran's I bias indicates that generated tiles are less spatially autocorrelated than reference tiles; a positive bias indicates excessive clustering. This complements MAE, bias, slope, and R² by testing whether generated rasters preserve spatial organization rather than only matching pixel magnitudes or tile averages.

## Files

- `spatial_clustering_config.json`: local configuration.
- `run_spatial_clustering.py`: CLI entrypoint.
- `spatial_clustering_core.py`: Moran's I, inference, resumability, summaries, and figure generation.
- `SpatialClustering_Colab_A100.ipynb`: Colab workflow for staging Drive inputs and running the analysis.
- `results/`: created at runtime.

The workflow imports model loading and decoding utilities from the sibling `OverallValidation` bundle. This keeps the spatial-clustering analysis separate while ensuring the same checkpoint discovery, architecture reconstruction, latent-seed handling, and BF/BH decoding used in the main validation.

## Local Commands

Run the full epoch-1000 Moran's I workflow:

```bash
conda run -n pytorch python /Users/9oy/Documents/Projects/IM3/EvaluationP/Script/Revision/3_GANValidation/SpatialClustering/run_spatial_clustering.py all --config /Users/9oy/Documents/Projects/IM3/EvaluationP/Script/Revision/3_GANValidation/SpatialClustering/spatial_clustering_config.json --resume
```

Rebuild summaries and figures without rerunning inference:

```bash
conda run -n pytorch python /Users/9oy/Documents/Projects/IM3/EvaluationP/Script/Revision/3_GANValidation/SpatialClustering/run_spatial_clustering.py summary --config /Users/9oy/Documents/Projects/IM3/EvaluationP/Script/Revision/3_GANValidation/SpatialClustering/spatial_clustering_config.json
```

## Outputs

- `results/manifests/spatial_evaluation_matrix.csv`: deterministic matrix of Moran's I rows to be evaluated.
- `results/manifests/runtime_environment.json`: runtime and configuration snapshot.
- `results/manifests/dataset_manifests_snapshot.json`: dataset manifest snapshot where available.
- `results/metrics/tile_morans_i.csv`: one row per tile, model setting, target, and evaluation mode.
- `results/metrics/overall_morans_i.csv`: setting-level Moran's I comparison metrics.
- `results/tables/table_morans_i_summary_epoch1000.csv`: manuscript-facing summary table.
- `results/figures/morans_i_model_comparison_epoch1000.png`: compact model-class comparison figure.
- `results/logs/completed_rows.csv`: resumability log.

## Interpretation

Strong spatial-pattern support is indicated when generated Moran's I closely matches reference Moran's I across both test datasets, targets, learning rates, and model classes, with low Moran's I MAE, near-zero Moran's I bias, slope near one, and high R². Systematically negative Moran's I bias would indicate over-smoothed generated fields with weaker local clustering than the reference data; systematically positive bias would indicate overly clumped outputs. These outcomes should be interpreted alongside the existing error, bias, average, heterogeneity, and diversity analyses rather than as a replacement for them.
