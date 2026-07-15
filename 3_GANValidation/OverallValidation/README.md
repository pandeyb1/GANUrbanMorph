# Overall Validation Bundle

Fresh, reproducible manuscript-facing validation for the active model classes:

- `1`: U-Net baseline
- `2A`: single-latent cGAN (`cGANRandomVecFixed`)
- `3`: multi-random diversity cGAN (`cGANMultiRandomDiversity`)

The bundle evaluates `BF` and `BH` across:

- `LALegacy` and `MSASample` training regimes
- `SanDiegoTestNoOverlap` and `CONUSStratifiedTest`
- learning rates `0.0001`, `0.0002`, `0.0005`, `0.001`
- checkpoints `250`, `500`, `750`, `1000`

## Files

- [validation_config.json](/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/Revision/3_GANValidation/OverallValidation/validation_config.json)
- [run_overall_validation.py](/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/Revision/3_GANValidation/OverallValidation/run_overall_validation.py)
- [validation_core.py](/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/Revision/3_GANValidation/OverallValidation/validation_core.py)
- [environment.yml](/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/Revision/3_GANValidation/OverallValidation/environment.yml)
- [OverallValidation_Colab_A100.ipynb](/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/Revision/3_GANValidation/OverallValidation/OverallValidation_Colab_A100.ipynb)

## Environment

Create the environment:

```bash
cd /Users/9oy/Documents/Projects/IM3/EvaluationP/Script/Revision/3_GANValidation/OverallValidation
conda env create -f environment.yml
conda activate overall-gan-validation
```

The local `pytorch` environment already available on this machine is also compatible with the current script stack:

```bash
conda run -n pytorch python run_overall_validation.py --help
```

Reproducibility is intended for the same software stack and hardware backend. Small floating-point differences across CPU, MPS, and CUDA are acceptable and are documented by `results/manifests/runtime_environment.json`.

## Colab A100

Use [OverallValidation_Colab_A100.ipynb](/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/Revision/3_GANValidation/OverallValidation/OverallValidation_Colab_A100.ipynb) to run the same workflow on Colab. The notebook assumes the validation code folder is copied to Google Drive at `/content/drive/MyDrive/IM3/EvalP1/OverallValidation`, legacy model outputs are under `/content/drive/MyDrive/IM3/EvalP1/codev3`, and MSA-sample model outputs are under `/content/drive/MyDrive/IM3/EvalP1/revision_msa_sample`, matching the previous Colab training notebooks. Upload `Archive_SanDiego_TestNoOverlap.zip` and `Archive_CONUS_M1_random_stratified_2015_test.zip` to `/content/drive/MyDrive/IM3/EvalP1/validation_inputs`, then run the staging cells so checkpoints and extracted rasters are copied to `/content` before validation. The notebook writes `validation_config_colab.json` with `device = "cuda"` and `batch_size = 128`, then copies final `results/` back to Drive.

## Commands

Run the full workflow:

```bash
cd /Users/9oy/Documents/Projects/IM3/EvaluationP/Script/Revision/3_GANValidation/OverallValidation
python run_overall_validation.py all --config validation_config.json
```

Run point metrics only:

```bash
python run_overall_validation.py point --config validation_config.json --resume
```

Run diversity only:

```bash
python run_overall_validation.py diversity --config validation_config.json --resume
```

Rebuild tables and figures from saved CSV outputs without rerunning inference:

```bash
python run_overall_validation.py summary --config validation_config.json
```

Useful filters:

```bash
python run_overall_validation.py point \
  --config validation_config.json \
  --training-regime LALegacy \
  --family 2A \
  --dataset SanDiegoTestNoOverlap \
  --learning-rate 0.0001 \
  --checkpoint 250
```

Comma-separated values are accepted for all filter flags.

## Outputs

`results/` is created under this folder and contains:

- `manifests/`
  - `evaluation_matrix.csv`
  - `runtime_environment.json`
  - `dataset_manifests_snapshot.json`
  - `bootstrap_indices.npz`
  - `pixel_regression_sample.csv`
  - `diversity_subsets_*.csv`
- `metrics/`
  - `overall_point_metrics.csv`
  - `tile_level_mean_metrics.csv`
  - `tile_level_heterogeneity_metrics.csv`
  - `pixel_level_metrics.csv`
  - `bh_pipeline_vs_oracle.csv`
  - `bias_by_reference_bin.csv`
- `diversity/`
  - `overall_diversity_metrics.csv`
  - `tile_level_diversity_metrics.csv`
- `tables/`
  - manuscript-facing descriptive tables
  - `table_model_class_metric_summary_epoch1000.csv`
  - `table_model_class_fit_summary_epoch1000.csv`
- `figures/`
  - `lr x checkpoint` performance matrices
  - `model_class_comparison_epoch1000.png`
  - `model_class_fit_epoch1000.png`
  - bias-by-bin plots
  - diversity-accuracy frontiers
  - BH pipeline-vs-oracle plots
  - representative multi-sample panels
- `logs/`
  - `completed_rows.csv`
- `samples/`
  - intermediate `.npz` bundles used to render representative panels

## Metric Design

Point metrics:

- Tile mean: `MAE`, `RMSE`, `MBE`, slope, intercept, `R^2`
- Tile heterogeneity: `MAE`, `RMSE`, `MBE`, slope, intercept, `R^2`
- Pixel level on the active-pixel union mask: `MAE`, `RMSE`, `MBE`, slope, intercept, `R^2`
- `MAPE` is intentionally excluded from the main workflow

BH is evaluated in both modes:

- `oracle_bf`: BH conditioned on reference BF
- `pipeline_bf`: BH conditioned on generated BF

Diversity metrics:

- `div_pairwise_mae`
- `div_pixel_std`
- `div_tile_mean_sd`
- `div_tile_heterogeneity_sd`
- `oracle8_tile_mae_gain`
- `spread_error_spearman`

U-Net is recorded as a deterministic zero-diversity baseline.

Epoch-1000 model-class comparisons:

- The summary stage builds a combined multi-panel figure comparing `LALegacy` and `MSASample` families `1`, `2A`, and `3`.
- The figures show BF and BH only; BH uses the reference-BF-conditioned evaluation internally but is labeled simply as BH for manuscript-facing interpretation.
- These outputs are descriptive and emphasize effect direction, magnitude, and consistency across held-out datasets and learning rates.
- Formal pairwise significance tests are intentionally excluded from the main workflow to keep the manuscript-facing analysis interpretable.

## Rerun Notes

- Non-`--resume` runs reset the stage-specific CSV outputs before writing new rows.
- `summary` always rebuilds tables and figures from saved CSVs.
- `diversity` requires point-stage outputs to exist first.
