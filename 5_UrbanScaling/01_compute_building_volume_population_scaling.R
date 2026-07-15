rm(list = ls())
gc()

required_packages <- c("sf", "raster", "terra", "lmtest", "sandwich")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop("Missing required R package(s): ", paste(missing_packages, collapse = ", "))
}

script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
  }
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(normalizePath(sys.frames()[[1]]$ofile, mustWork = TRUE))
  }
  normalizePath("01_compute_building_volume_population_scaling.R", mustWork = FALSE)
}

SCRIPT_DIR <- dirname(script_path())
DATA_DIR <- file.path(SCRIPT_DIR, "data")
TABLE_DIR <- file.path(SCRIPT_DIR, "tables")
TEXT_DIR <- file.path(SCRIPT_DIR, "text")
dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TEXT_DIR, recursive = TRUE, showWarnings = FALSE)
terra::terraOptions(progress = 0)

lr_from_evaldf <- function(path) {
  as.numeric(gsub("Evaldf_|[.]csv", "", basename(path)))
}

finite_sum <- function(x) {
  sum(x[is.finite(x)], na.rm = TRUE)
}

terra_sum <- function(x) {
  as.numeric(terra::global(x, "sum", na.rm = TRUE)[1, 1])
}

format_range <- function(x) {
  paste0(sprintf("%.2f", min(x, na.rm = TRUE)), "-", sprintf("%.2f", max(x, na.rm = TRUE)), "%")
}

model_specs <- read.csv(file.path(DATA_DIR, "manifests", "model_input_specs.csv"), stringsAsFactors = FALSE)
expected_learning_rates <- c(0.0001, 0.0002, 0.0005, 0.001)

metrics_rows <- lapply(seq_len(nrow(model_specs)), function(i) {
  spec <- model_specs[i, ]
  metrics_path <- file.path(DATA_DIR, "model_metrics", spec$model_key, "morphology_growth_epoch1000_metrics.csv")
  if (!file.exists(metrics_path)) {
    stop("Missing copied model metrics: ", metrics_path)
  }
  metrics <- read.csv(metrics_path, stringsAsFactors = FALSE)
  metrics$model_key <- spec$model_key
  metrics$model_class <- spec$model_class
  metrics$family <- spec$model_class
  metrics$model_class_label <- spec$model_class_label
  metrics$training_regime <- spec$training_regime
  metrics$training_regime_label <- spec$training_regime_label
  metrics
})
all_metric_columns <- unique(unlist(lapply(metrics_rows, names)))
metrics_rows <- lapply(metrics_rows, function(metrics) {
  missing_columns <- setdiff(all_metric_columns, names(metrics))
  for (column in missing_columns) {
    metrics[[column]] <- NA
  }
  metrics[, all_metric_columns]
})
model_source_manifest <- do.call(rbind, metrics_rows)
write.csv(
  model_source_manifest,
  file.path(TABLE_DIR, "model_source_manifest.csv"),
  row.names = FALSE
)

correct_bh_source <- "/Volumes/HDD/Models/MSASampleModels/2A_BH_cGANRandomVecFixed_MSASample/lr_0p001_seed_2026/generator_epoch_1000.pth"
guard_rows <- model_source_manifest[
  model_source_manifest$training_regime == "MSASample" &
    model_source_manifest$model_class == "2A" &
    model_source_manifest$target == "BH" &
    abs(model_source_manifest$learning_rate - 0.001) < 1e-12,
]
if (nrow(guard_rows) == 0) {
  stop("Missing source rows for MSASample / 2A / BH / lr=0.001.")
}
if (any(grepl("lr_0p001_seed_5026", guard_rows$source, fixed = TRUE))) {
  stop("Invalid source detected for MSASample / 2A / BH / lr=0.001: seed_5026.")
}
if (!all(guard_rows$source == correct_bh_source)) {
  stop(
    "Unexpected source for MSASample / 2A / BH / lr=0.001. Expected: ",
    correct_bh_source,
    "\nObserved: ",
    paste(unique(guard_rows$source), collapse = "; ")
  )
}

msa_path <- file.path(DATA_DIR, "derived", "msa", "BuildingVolume2015MSA.gpkg")
msa_volume <- sf::st_read(msa_path, quiet = TRUE)
msa_volume <- as.data.frame(sf::st_drop_geometry(msa_volume))
msa_volume$ACK2E001 <- as.numeric(msa_volume$ACK2E001)
msa_volume$vols <- as.numeric(msa_volume$vols)
msa_volume <- msa_volume[is.finite(msa_volume$ACK2E001) & is.finite(msa_volume$vols) & msa_volume$vols > 0, ]
if (nrow(msa_volume) != 379) {
  stop("Expected 379 MSA records after filtering, found: ", nrow(msa_volume))
}

scaling_model <- lm(log(vols) ~ log(ACK2E001), data = msa_volume)
model_summary <- summary(scaling_model)
coef_table <- model_summary$coefficients
standard_ci <- confint(scaling_model, level = 0.95)
robust_test <- lmtest::coeftest(scaling_model, vcov = sandwich::vcovHC(scaling_model, type = "HC0"))
robust_ci <- confint(robust_test, level = 0.95)

population_scaling_fit <- data.frame(
  n_msa = nrow(msa_volume),
  intercept = unname(coef(scaling_model)[1]),
  beta = unname(coef(scaling_model)[2]),
  beta_se = coef_table[2, 2],
  beta_p_value = coef_table[2, 4],
  beta_ci95_low = robust_ci[2, 1],
  beta_ci95_high = robust_ci[2, 2],
  beta_ols_ci95_low = standard_ci[2, 1],
  beta_ols_ci95_high = standard_ci[2, 2],
  beta_robust_se = robust_test[2, 2],
  beta_robust_ci95_low = robust_ci[2, 1],
  beta_robust_ci95_high = robust_ci[2, 2],
  r2 = model_summary$r.squared,
  stringsAsFactors = FALSE
)
write.csv(population_scaling_fit, file.path(TABLE_DIR, "population_scaling_fit.csv"), row.names = FALSE)

la_raw_dir <- file.path(DATA_DIR, "raw", "la_morphology")
ua2010_shape <- file.path(DATA_DIR, "raw", "nhgis", "nhgis0040_shapefile_tl2010_us_urb_area_2010", "US_urb_area_2010.shp")
la_urban_area <- terra::vect(ua2010_shape)
la_urban_area <- la_urban_area[la_urban_area$GISJOIN == "G51445", ]
if (nrow(la_urban_area) != 1) {
  stop("Could not identify Los Angeles--Long Beach--Anaheim urban area G51445 in the 2010 NHGIS urban-area shapefile.")
}

landscan2010 <- terra::rast(file.path(DATA_DIR, "raw", "landscan", "landscan-global-2010.tif"))
landscan2020 <- terra::rast(file.path(DATA_DIR, "raw", "landscan", "landscan-global-2020.tif"))
la_urban_area_landscan <- suppressWarnings(terra::project(la_urban_area, terra::crs(landscan2010)))
landscan2010_crop <- terra::crop(landscan2010, la_urban_area_landscan)
landscan2020_crop <- terra::crop(landscan2020, la_urban_area_landscan)
landscan_population_2010_touched_cells <- terra_sum(
  terra::mask(landscan2010_crop, la_urban_area_landscan)
)
landscan_population_2020_touched_cells <- terra_sum(
  terra::mask(landscan2020_crop, la_urban_area_landscan)
)
landscan_population_2010 <- terra_sum(
  terra::mask(landscan2010_crop, la_urban_area_landscan, touches = FALSE)
)
landscan_population_2020 <- terra_sum(
  terra::mask(landscan2020_crop, la_urban_area_landscan, touches = FALSE)
)
landscan_population_diagnostics <- data.frame(
  geography = "Los Angeles--Long Beach--Anaheim urban area",
  population_source = "LandScan clipped to 2010 NHGIS urban area G51445",
  inclusion_rule = c("cell_center", "touched_cells"),
  population_2010 = c(landscan_population_2010, landscan_population_2010_touched_cells),
  population_2020 = c(landscan_population_2020, landscan_population_2020_touched_cells),
  log10_population_2010 = log10(c(landscan_population_2010, landscan_population_2010_touched_cells)),
  population_growth_pct = 100 *
    (c(landscan_population_2020, landscan_population_2020_touched_cells) -
       c(landscan_population_2010, landscan_population_2010_touched_cells)) /
    c(landscan_population_2010, landscan_population_2010_touched_cells),
  stringsAsFactors = FALSE
)
write.csv(
  landscan_population_diagnostics,
  file.path(TABLE_DIR, "landscan_population_zonal_diagnostics.csv"),
  row.names = FALSE
)

bf2010 <- raster::raster(file.path(la_raw_dir, "BF2010.tif"))
bh2010_raw_units <- raster::raster(file.path(la_raw_dir, "BH2010.tif"))
bh2010_raw_units[bh2010_raw_units > 75] <- 75
legacy_baseline_volume_2010_raw_units <- raster::cellStats(bf2010 * bh2010_raw_units * 900, sum, na.rm = TRUE)

bh2010 <- raster::raster(file.path(la_raw_dir, "BH2010.tif")) * 0.3048
bh2010[bh2010 > 75] <- 75
baseline_volume_m3 <- raster::cellStats(bf2010 * bh2010 * 900, sum, na.rm = TRUE)

bf2020 <- raster::raster(file.path(la_raw_dir, "BF_2020.tif"))
bh2020_raw_units <- raster::raster(file.path(la_raw_dir, "BH_2020.tif"))
bh2020_raw_units[bh2020_raw_units > 75] <- 75
bh2020 <- raster::raster(file.path(la_raw_dir, "BH_2020.tif")) * 0.3048
bh2020[bh2020 > 75] <- 75
nlcd2010 <- raster::raster(file.path(la_raw_dir, "nlcd2010LA.tif"))
nlcd2020 <- raster::raster(file.path(la_raw_dir, "nlcd2020LA.tif"))
developed_classes <- c(22, 23, 24)
change_mask <- (nlcd2020[] != nlcd2010[]) & (nlcd2020[] %in% developed_classes)
reference_delta_from_raster_m3 <- finite_sum((bf2020[] * bh2020[] * 900)[change_mask])
legacy_reference_delta_from_raster_raw_units <- finite_sum((bf2020[] * bh2020_raw_units[] * 900)[change_mask])
legacy_reference_total_2020_raw_units <- legacy_baseline_volume_2010_raw_units +
  legacy_reference_delta_from_raster_raw_units

bf2010_t <- terra::rast(file.path(la_raw_dir, "BF2010.tif"))
la_urban_area_volume <- suppressWarnings(terra::project(la_urban_area, terra::crs(bf2010_t)))
bh2010_raw_t <- terra::clamp(terra::rast(file.path(la_raw_dir, "BH2010.tif")), upper = 75, values = TRUE)
bh2020_raw_t <- terra::clamp(terra::rast(file.path(la_raw_dir, "BH_2020.tif")), upper = 75, values = TRUE)
bh2010_m_t <- terra::clamp(terra::rast(file.path(la_raw_dir, "BH2010.tif")) * 0.3048, upper = 75, values = TRUE)
bh2020_m_t <- terra::clamp(terra::rast(file.path(la_raw_dir, "BH_2020.tif")) * 0.3048, upper = 75, values = TRUE)
bf2020_t <- terra::rast(file.path(la_raw_dir, "BF_2020.tif"))
change_mask_t <- terra::rast(file.path(la_raw_dir, "ChangeMask2010_2020.tif"))
la_urban_baseline_volume_m3 <- terra_sum(
  terra::mask(terra::crop(bf2010_t * bh2010_m_t * 900, la_urban_area_volume), la_urban_area_volume)
)
la_urban_reference_delta_m3 <- terra_sum(
  terra::mask(terra::crop(bf2020_t * bh2020_m_t * 900 * change_mask_t, la_urban_area_volume), la_urban_area_volume)
)
la_urban_legacy_baseline_raw_units <- terra_sum(
  terra::mask(terra::crop(bf2010_t * bh2010_raw_t * 900, la_urban_area_volume), la_urban_area_volume)
)
la_urban_legacy_reference_delta_raw_units <- terra_sum(
  terra::mask(terra::crop(bf2020_t * bh2020_raw_t * 900 * change_mask_t, la_urban_area_volume), la_urban_area_volume)
)
if (abs(la_urban_baseline_volume_m3 - baseline_volume_m3) > max(1, baseline_volume_m3 * 1e-6)) {
  warning("LA urban-area clipped baseline differs from the raster baseline by more than one ppm.")
}
baseline_volume_m3 <- la_urban_baseline_volume_m3
reference_delta_from_raster_m3 <- la_urban_reference_delta_m3
legacy_baseline_volume_2010_raw_units <- la_urban_legacy_baseline_raw_units
legacy_reference_delta_from_raster_raw_units <- la_urban_legacy_reference_delta_raw_units
legacy_reference_total_2020_raw_units <- legacy_baseline_volume_2010_raw_units +
  legacy_reference_delta_from_raster_raw_units

growth_rows <- lapply(seq_len(nrow(model_specs)), function(i) {
  spec <- model_specs[i, ]
  evaldf_dir <- file.path(DATA_DIR, "model_evaldfs", spec$model_key)
  evaldf_paths <- list.files(evaldf_dir, pattern = "Evaldf_.*[.]csv$", full.names = TRUE)
  observed_lrs <- sort(vapply(evaldf_paths, lr_from_evaldf, numeric(1)))
  if (length(observed_lrs) != length(expected_learning_rates) ||
      any(abs(observed_lrs - expected_learning_rates) > 1e-12)) {
    stop(
      "Unexpected learning-rate coverage for ", spec$model_key,
      ". Expected ", paste(expected_learning_rates, collapse = ", "),
      "; observed ", paste(observed_lrs, collapse = ", ")
    )
  }
  do.call(rbind, lapply(evaldf_paths, function(path) {
    learning_rate <- lr_from_evaldf(path)
    frame <- read.csv(path)
    required_columns <- c("origbf", "origbh", "predbf", "predbh")
    if (!all(required_columns %in% names(frame))) {
      stop("Missing required columns in ", path)
    }
    if (nrow(frame) != 150166) {
      stop("Expected 150166 rows in ", path, ", found ", nrow(frame))
    }
    reference_delta_m3 <- finite_sum(frame$origbf * frame$origbh * 900)
    generated_delta_m3 <- finite_sum(frame$predbf * frame$predbh * 900)
    data.frame(
      model_key = spec$model_key,
      model_class = spec$model_class,
      model_class_label = spec$model_class_label,
      training_regime = spec$training_regime,
      training_regime_label = spec$training_regime_label,
      learning_rate = learning_rate,
      n_new_developed_pixels = nrow(frame),
      baseline_volume_2010_m3 = baseline_volume_m3,
      reference_delta_m3 = reference_delta_m3,
      generated_delta_m3 = generated_delta_m3,
      reference_total_2020_m3 = baseline_volume_m3 + reference_delta_m3,
      generated_total_2020_m3 = baseline_volume_m3 + generated_delta_m3,
      reference_growth_pct = 100 * reference_delta_m3 / baseline_volume_m3,
      generated_growth_pct = 100 * generated_delta_m3 / baseline_volume_m3,
      generated_to_reference_delta_ratio = generated_delta_m3 / reference_delta_m3,
      evaldf_path = normalizePath(path, mustWork = TRUE),
      stringsAsFactors = FALSE
    )
  }))
})
volume_growth <- do.call(rbind, growth_rows)
write.csv(
  volume_growth,
  file.path(TABLE_DIR, "building_volume_growth_epoch1000_unet_single_latent_no_bias_correction.csv"),
  row.names = FALSE
)

reference_summary <- data.frame(
  baseline_volume_2010_m3 = baseline_volume_m3,
  reference_delta_from_evaldf_m3 = unique(volume_growth$reference_delta_m3)[1],
  reference_delta_from_raster_m3 = reference_delta_from_raster_m3,
  reference_growth_pct = unique(volume_growth$reference_growth_pct)[1],
  reference_delta_evaldf_minus_raster_m3 = unique(volume_growth$reference_delta_m3)[1] - reference_delta_from_raster_m3,
  legacy_baseline_volume_2010_raw_units = legacy_baseline_volume_2010_raw_units,
  legacy_reference_delta_from_raster_raw_units = legacy_reference_delta_from_raster_raw_units,
  legacy_reference_total_2020_raw_units = legacy_reference_total_2020_raw_units,
  stringsAsFactors = FALSE
)
write.csv(reference_summary, file.path(TABLE_DIR, "la_reference_growth_summary.csv"), row.names = FALSE)

ua2010_csv <- file.path(DATA_DIR, "raw", "nhgis", "nhgis0040_csv", "nhgis0040_ds172_2010_urb_area.csv")
ua2020_csv <- file.path(DATA_DIR, "raw", "nhgis", "nhgis0040_csv", "nhgis0040_ds258_2020_urb_area.csv")
ua2010 <- read.csv(ua2010_csv, stringsAsFactors = FALSE)
ua2020 <- read.csv(ua2020_csv, stringsAsFactors = FALSE)
la_pop2010 <- ua2010[ua2010$GISJOIN == "G51445", c("GISJOIN", "NAME", "H7V001")]
la_pop2020 <- ua2020[ua2020$GISJOIN == "G51445", c("GISJOIN", "NAME", "U7H001")]
if (nrow(la_pop2010) != 1 || nrow(la_pop2020) != 1) {
  stop("Could not identify Los Angeles urban-area population row G51445 in NHGIS 2010/2020 CSVs.")
}
la_population_summary <- data.frame(
  geography = "Los Angeles--Long Beach--Anaheim urban area",
  population_source = "LandScan 2010/2020 clipped to 2010 NHGIS urban area G51445 using cell-center inclusion",
  population_2010 = landscan_population_2010,
  population_2020 = landscan_population_2020,
  population_growth_pct = 100 * (landscan_population_2020 - landscan_population_2010) / landscan_population_2010,
  touched_cell_population_2010 = landscan_population_2010_touched_cells,
  touched_cell_population_2020 = landscan_population_2020_touched_cells,
  touched_cell_population_growth_pct = 100 *
    (landscan_population_2020_touched_cells - landscan_population_2010_touched_cells) /
    landscan_population_2010_touched_cells,
  nhgis_tabular_population_2010 = as.numeric(la_pop2010$H7V001),
  nhgis_tabular_population_2020 = as.numeric(la_pop2020$U7H001),
  nhgis_tabular_population_growth_pct = 100 * (as.numeric(la_pop2020$U7H001) - as.numeric(la_pop2010$H7V001)) / as.numeric(la_pop2010$H7V001),
  baseline_volume_2010_m3 = baseline_volume_m3,
  reference_total_2020_m3 = baseline_volume_m3 + unique(volume_growth$reference_delta_m3)[1],
  legacy_baseline_volume_2010_raw_units = legacy_baseline_volume_2010_raw_units,
  legacy_reference_total_2020_raw_units = legacy_reference_total_2020_raw_units,
  expected_volume_2010_m3 = exp(coef(scaling_model)[1]) * landscan_population_2010 ^ coef(scaling_model)[2],
  expected_volume_2020_m3 = exp(coef(scaling_model)[1]) * landscan_population_2020 ^ coef(scaling_model)[2],
  stringsAsFactors = FALSE
)
la_population_summary$expected_growth_pct <- 100 *
  (la_population_summary$expected_volume_2020_m3 - la_population_summary$expected_volume_2010_m3) /
  la_population_summary$expected_volume_2010_m3
write.csv(la_population_summary, file.path(TABLE_DIR, "la_population_volume_summary.csv"), row.names = FALSE)

range_table <- aggregate(
  generated_growth_pct ~ model_class_label + training_regime_label,
  data = volume_growth,
  FUN = function(x) paste0(sprintf("%.2f", min(x)), "-", sprintf("%.2f", max(x)))
)

line_for <- function(model_label, regime_label) {
  vals <- volume_growth$generated_growth_pct[
    volume_growth$model_class_label == model_label &
      volume_growth$training_regime_label == regime_label
  ]
  format_range(vals)
}

result_lines <- c(
  "# Building Volume-Population Size Scaling",
  "",
  paste0(
    "The 2015 metropolitan scaling fit used ", population_scaling_fit$n_msa,
    " metropolitan statistical areas and estimated a near-linear building-volume population exponent of ",
    sprintf("%.2f", population_scaling_fit$beta),
    " (robust 95% CI: ", sprintf("%.2f", population_scaling_fit$beta_ci95_low),
    "-", sprintf("%.2f", population_scaling_fit$beta_ci95_high), ")."
  ),
  "",
  paste0(
    "For NLCD-based newly developed pixels in Los Angeles between 2010 and 2020, ",
    "the MA v2 reference data imply a ", sprintf("%.2f", unique(volume_growth$reference_growth_pct)[1]),
    "% increase in aggregate building volume relative to the 2010 baseline. ",
    "Without bias correction, the U-Net baseline predicts ",
    line_for("U-Net baseline", "Los Angeles Model"), " under the Los Angeles Model and ",
    line_for("U-Net baseline", "CONUS Model"), " under the CONUS Model. ",
    "The single-latent cGAN predicts ",
    line_for("single-latent cGAN", "Los Angeles Model"), " under the Los Angeles Model and ",
    line_for("single-latent cGAN", "CONUS Model"), " under the CONUS Model."
  ),
  "",
  paste0(
    "Figure caption: Building volume-population size scaling and uncorrected model-predicted building-volume growth. ",
    "Left: building volume scaling with population across 379 metropolitan statistical areas of the United States in 2015; ",
    "the solid red line shows the log-log least-squares fit and the dashed lines show the corresponding 95% confidence interval. ",
    "The red point marks the raster-derived Los Angeles urban-area estimate and is not included in the MSA fit; ",
    "the Los Angeles-Long Beach-Anaheim MSA remains one of the gray MSA points. ",
    "Right: 2010-to-2020 LandScan cell-center population and aggregate building-volume changes for the Los Angeles urban area, where the red arrow marks ",
    "the reference change and colored arrows mark uncorrected epoch-1000 U-Net and single-latent cGAN outputs ",
    "across learning rates and training regimes."
  )
)
writeLines(result_lines, file.path(TEXT_DIR, "building_volume_population_scaling_results.md"))

cat("Wrote volume growth table to", file.path(TABLE_DIR, "building_volume_growth_epoch1000_unet_single_latent_no_bias_correction.csv"), "\n")
cat("Wrote population scaling fit to", file.path(TABLE_DIR, "population_scaling_fit.csv"), "\n")
cat("Wrote model source manifest to", file.path(TABLE_DIR, "model_source_manifest.csv"), "\n")
cat("Provenance guard passed for corrected MSASample / 2A / BH / lr=0.001 source.\n")
