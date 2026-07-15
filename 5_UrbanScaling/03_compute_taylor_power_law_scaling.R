rm(list = ls())
gc()

required_packages <- c("raster")
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
  normalizePath("03_compute_taylor_power_law_scaling.R", mustWork = FALSE)
}

SCRIPT_DIR <- dirname(script_path())
DATA_DIR <- file.path(SCRIPT_DIR, "data")
TABLE_DIR <- file.path(SCRIPT_DIR, "tables")
TEXT_DIR <- file.path(SCRIPT_DIR, "text")
dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TEXT_DIR, recursive = TRUE, showWarnings = FALSE)

lr_from_evaldf <- function(path) {
  as.numeric(gsub("Evaldf_|[.]csv", "", basename(path)))
}

finite_positive <- function(x) {
  is.finite(x) & x > 0
}

population_variance <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(NA_real_)
  }
  mean((x - mean(x)) ^ 2)
}

format_range <- function(x, digits = 2) {
  paste0(sprintf(paste0("%.", digits, "f"), min(x, na.rm = TRUE)),
         " to ", sprintf(paste0("%.", digits, "f"), max(x, na.rm = TRUE)))
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
write.csv(model_source_manifest, file.path(TABLE_DIR, "model_source_manifest.csv"), row.names = FALSE)

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

taylor_path <- file.path(DATA_DIR, "derived", "taylor", "HeteroScalingLA.csv")
if (!file.exists(taylor_path)) {
  stop("Missing Taylor scaling input: ", taylor_path, ". Re-run 00_prepare_data.R.")
}
taylor_data <- read.csv(taylor_path, stringsAsFactors = FALSE)
required_taylor_columns <- c("Resolution", "Mean", "Variance")
if (!all(required_taylor_columns %in% names(taylor_data))) {
  stop("Taylor scaling input is missing required columns: ", paste(required_taylor_columns, collapse = ", "))
}
taylor_data <- taylor_data[
  finite_positive(taylor_data$Mean) &
    finite_positive(taylor_data$Variance) &
    is.finite(taylor_data$Resolution),
]
if (nrow(taylor_data) != 301) {
  stop("Expected 301 Taylor scaling rows, found: ", nrow(taylor_data))
}

taylor_model <- lm(log(Variance) ~ log(Mean), data = taylor_data)
taylor_summary <- summary(taylor_model)
taylor_ci <- confint(taylor_model, level = 0.95)
taylor_fit <- data.frame(
  n = nrow(taylor_data),
  intercept = unname(coef(taylor_model)[1]),
  coefficient_a = exp(unname(coef(taylor_model)[1])),
  beta = unname(coef(taylor_model)[2]),
  beta_se = taylor_summary$coefficients[2, 2],
  beta_p_value = taylor_summary$coefficients[2, 4],
  beta_ci95_low = taylor_ci[2, 1],
  beta_ci95_high = taylor_ci[2, 2],
  intercept_ci95_low = taylor_ci[1, 1],
  intercept_ci95_high = taylor_ci[1, 2],
  r2 = taylor_summary$r.squared,
  stringsAsFactors = FALSE
)
write.csv(taylor_fit, file.path(TABLE_DIR, "taylor_power_law_fit.csv"), row.names = FALSE)

predict_expected <- function(mean_value) {
  mean_log <- log(mean_value)
  prediction_fit <- unname(coef(taylor_model)[1] + coef(taylor_model)[2] * mean_log)
  prediction_low <- unname(taylor_ci[1, 1] + taylor_ci[2, 1] * mean_log)
  prediction_high <- unname(taylor_ci[1, 2] + taylor_ci[2, 2] * mean_log)
  data.frame(
    expected_variance = exp(prediction_fit),
    expected_variance_ci95_low = exp(prediction_low),
    expected_variance_ci95_high = exp(prediction_high)
  )
}

cmask <- raster::raster(file.path(DATA_DIR, "raw", "la_morphology", "ChangeMask2010_2020.tif"))
mask_values <- raster::getValues(cmask) > 0
mask_values[is.na(mask_values)] <- FALSE
if (sum(mask_values) != 150166) {
  stop("Expected 150166 newly developed mask pixels, found: ", sum(mask_values))
}
mask_matrix <- matrix(mask_values, nrow = nrow(cmask), ncol = ncol(cmask), byrow = TRUE)

window_metrics_from_values <- function(values_for_mask, prefix, window_size = 256) {
  if (length(values_for_mask) != sum(mask_values)) {
    stop(prefix, " value length does not match the change-mask pixel count.")
  }
  full_values <- rep(NA_real_, length(mask_values))
  full_values[mask_values] <- values_for_mask
  value_matrix <- matrix(full_values, nrow = nrow(cmask), ncol = ncol(cmask), byrow = TRUE)

  n_window_rows <- nrow(value_matrix) %/% window_size
  n_window_cols <- ncol(value_matrix) %/% window_size
  rows <- vector("list", n_window_rows * n_window_cols)
  k <- 0
  for (window_row in seq_len(n_window_rows)) {
    row_index <- ((window_row - 1) * window_size + 1):(window_row * window_size)
    for (window_col in seq_len(n_window_cols)) {
      col_index <- ((window_col - 1) * window_size + 1):(window_col * window_size)
      local_mask <- as.vector(mask_matrix[row_index, col_index])
      if (!any(local_mask)) {
        next
      }
      local_values <- as.vector(value_matrix[row_index, col_index])[local_mask]
      local_values <- local_values[is.finite(local_values)]
      if (length(local_values) == 0) {
        next
      }
      k <- k + 1
      mean_value <- mean(local_values)
      variance_value <- population_variance(local_values)
      expected <- predict_expected(mean_value)
      log10_variance_ratio <- if (finite_positive(variance_value) &&
                                  finite_positive(expected$expected_variance)) {
        log10(variance_value / expected$expected_variance)
      } else {
        NA_real_
      }
      rows[[k]] <- data.frame(
        window_id = k,
        window_row = window_row,
        window_col = window_col,
        n_pixels = length(local_values),
        mean = mean_value,
        variance = variance_value,
        expected_variance = expected$expected_variance,
        expected_variance_ci95_low = expected$expected_variance_ci95_low,
        expected_variance_ci95_high = expected$expected_variance_ci95_high,
        log10_variance_ratio = log10_variance_ratio,
        stringsAsFactors = FALSE
      )
    }
  }
  out <- do.call(rbind, rows[seq_len(k)])
  names(out)[names(out) %in% c("mean", "variance", "expected_variance",
                               "expected_variance_ci95_low", "expected_variance_ci95_high",
                               "log10_variance_ratio")] <-
    paste0(prefix, "_", names(out)[names(out) %in% c("mean", "variance", "expected_variance",
                                                     "expected_variance_ci95_low", "expected_variance_ci95_high",
                                                     "log10_variance_ratio")])
  out
}

first_evaldf_path <- file.path(DATA_DIR, "model_evaldfs", model_specs$model_key[1], "Evaldf_0.0001.csv")
first_evaldf <- read.csv(first_evaldf_path)
reference_volume_values <- first_evaldf$origbf * first_evaldf$origbh * 900
reference_windows <- window_metrics_from_values(reference_volume_values, "reference")
if (nrow(reference_windows) != 107) {
  stop("Expected 107 non-empty reference windows, found: ", nrow(reference_windows))
}

growth_table_path <- file.path(TABLE_DIR, "building_volume_growth_epoch1000_unet_single_latent_no_bias_correction.csv")
growth_table <- if (file.exists(growth_table_path)) read.csv(growth_table_path, stringsAsFactors = FALSE) else NULL

window_rows <- lapply(seq_len(nrow(model_specs)), function(i) {
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
    generated_volume_values <- frame$predbf * frame$predbh * 900
    generated_windows <- window_metrics_from_values(generated_volume_values, "generated")
    if (nrow(generated_windows) != 107) {
      stop("Expected 107 non-empty generated windows in ", path, ", found: ", nrow(generated_windows))
    }
    if (!all(generated_windows$window_row == reference_windows$window_row) ||
        !all(generated_windows$window_col == reference_windows$window_col)) {
      stop("Generated and reference window grids do not align for ", path)
    }
    merged <- cbind(
      generated_windows,
      reference_windows[, setdiff(names(reference_windows), c("window_id", "window_row", "window_col", "n_pixels"))]
    )
    merged$model_key <- spec$model_key
    merged$model_class <- spec$model_class
    merged$model_class_label <- spec$model_class_label
    merged$training_regime <- spec$training_regime
    merged$training_regime_label <- spec$training_regime_label
    merged$learning_rate <- learning_rate
    merged$evaldf_path <- normalizePath(path, mustWork = TRUE)
    if (!is.null(growth_table)) {
      growth_match <- growth_table[
        growth_table$model_key == spec$model_key &
          abs(growth_table$learning_rate - learning_rate) < 1e-12,
      ]
      if (nrow(growth_match) == 1) {
        merged$generated_growth_pct <- growth_match$generated_growth_pct
        merged$reference_growth_pct <- growth_match$reference_growth_pct
      }
    }
    merged
  }))
})
window_metrics <- do.call(rbind, window_rows)
ordered_columns <- c(
  "model_key", "model_class", "model_class_label", "training_regime", "training_regime_label",
  "learning_rate", "window_id", "window_row", "window_col", "n_pixels",
  setdiff(names(window_metrics), c("model_key", "model_class", "model_class_label",
                                   "training_regime", "training_regime_label", "learning_rate",
                                   "window_id", "window_row", "window_col", "n_pixels"))
)
window_metrics <- window_metrics[, ordered_columns]

write.csv(
  window_metrics,
  file.path(TABLE_DIR, "taylor_power_law_window_metrics_epoch1000_unet_single_latent_no_bias_correction.csv"),
  row.names = FALSE
)

summary_rows <- do.call(rbind, lapply(split(window_metrics, interaction(
  window_metrics$model_key, window_metrics$learning_rate, drop = TRUE
)), function(frame) {
  log_frame <- frame[
    finite_positive(frame$generated_variance) &
      finite_positive(frame$generated_expected_variance) &
      is.finite(frame$generated_log10_variance_ratio),
  ]
  expected_frame <- frame[
    finite_positive(frame$generated_expected_variance) &
      finite_positive(frame$generated_expected_variance_ci95_low) &
      finite_positive(frame$generated_expected_variance_ci95_high),
  ]
  if (nrow(log_frame) < 2) {
    stop("Fewer than two positive finite generated-variance windows for ", frame$model_key[1],
         " lr=", frame$learning_rate[1])
  }
  if (nrow(expected_frame) < 2) {
    stop("Fewer than two positive finite expected-variance windows for ", frame$model_key[1],
         " lr=", frame$learning_rate[1])
  }
  fit <- lm(log(generated_expected_variance) ~ log(generated_variance), data = log_frame)
  fit_summary <- summary(fit)
  fit_ci <- confint(fit, level = 0.95)
  data.frame(
    model_key = frame$model_key[1],
    model_class = frame$model_class[1],
    model_class_label = frame$model_class_label[1],
    training_regime = frame$training_regime[1],
    training_regime_label = frame$training_regime_label[1],
    learning_rate = frame$learning_rate[1],
    n_windows = nrow(frame),
    n_windows_with_expected_variance = nrow(expected_frame),
    n_windows_for_log_metrics = nrow(log_frame),
    nonpositive_generated_expected_variance_windows = sum(!finite_positive(frame$generated_expected_variance)),
    nonpositive_generated_variance_windows = sum(!finite_positive(frame$generated_variance)),
    median_log10_variance_ratio = median(log_frame$generated_log10_variance_ratio),
    q1_log10_variance_ratio = unname(quantile(log_frame$generated_log10_variance_ratio, 0.25)),
    q3_log10_variance_ratio = unname(quantile(log_frame$generated_log10_variance_ratio, 0.75)),
    median_generated_variance = median(frame$generated_variance),
    median_expected_variance = median(expected_frame$generated_expected_variance),
    pct_windows_below_expected = 100 * mean(expected_frame$generated_variance < expected_frame$generated_expected_variance),
    pct_windows_within_expected_ci = 100 * mean(
      expected_frame$generated_variance >= expected_frame$generated_expected_variance_ci95_low &
        expected_frame$generated_variance <= expected_frame$generated_expected_variance_ci95_high
    ),
    expected_vs_generated_log_slope = unname(coef(fit)[2]),
    expected_vs_generated_log_slope_ci95_low = fit_ci[2, 1],
    expected_vs_generated_log_slope_ci95_high = fit_ci[2, 2],
    expected_vs_generated_log_r2 = fit_summary$r.squared,
    stringsAsFactors = FALSE
  )
}))
summary_rows <- summary_rows[order(
  summary_rows$model_class_label,
  summary_rows$training_regime_label,
  summary_rows$learning_rate
), ]
write.csv(
  summary_rows,
  file.path(TABLE_DIR, "taylor_power_law_model_summary_epoch1000_unet_single_latent_no_bias_correction.csv"),
  row.names = FALSE
)

reference_log_frame <- reference_windows[
  finite_positive(reference_windows$reference_variance) &
    finite_positive(reference_windows$reference_expected_variance) &
    is.finite(reference_windows$reference_log10_variance_ratio),
]
reference_expected_frame <- reference_windows[
  finite_positive(reference_windows$reference_expected_variance) &
    finite_positive(reference_windows$reference_expected_variance_ci95_low) &
    finite_positive(reference_windows$reference_expected_variance_ci95_high),
]
if (nrow(reference_log_frame) != nrow(reference_windows)) {
  warning(
    "Reference Taylor windows include ",
    nrow(reference_windows) - nrow(reference_log_frame),
    " non-positive or non-finite observed variance value(s), excluded from log metrics."
  )
}
reference_summary <- data.frame(
  n_windows = nrow(reference_windows),
  n_windows_with_expected_variance = nrow(reference_expected_frame),
  n_windows_for_log_metrics = nrow(reference_log_frame),
  nonpositive_reference_expected_variance_windows = sum(!finite_positive(reference_windows$reference_expected_variance)),
  nonpositive_reference_variance_windows = sum(!finite_positive(reference_windows$reference_variance)),
  median_log10_variance_ratio = median(reference_log_frame$reference_log10_variance_ratio),
  q1_log10_variance_ratio = unname(quantile(reference_log_frame$reference_log10_variance_ratio, 0.25)),
  q3_log10_variance_ratio = unname(quantile(reference_log_frame$reference_log10_variance_ratio, 0.75)),
  median_reference_variance = median(reference_windows$reference_variance),
  median_reference_expected_variance = median(reference_expected_frame$reference_expected_variance),
  pct_windows_below_expected = 100 * mean(reference_expected_frame$reference_variance < reference_expected_frame$reference_expected_variance),
  pct_windows_within_expected_ci = 100 * mean(
    reference_expected_frame$reference_variance >= reference_expected_frame$reference_expected_variance_ci95_low &
      reference_expected_frame$reference_variance <= reference_expected_frame$reference_expected_variance_ci95_high
  ),
  stringsAsFactors = FALSE
)
write.csv(reference_summary, file.path(TABLE_DIR, "taylor_power_law_reference_summary.csv"), row.names = FALSE)

line_for <- function(model_label, regime_label) {
  vals <- summary_rows$median_log10_variance_ratio[
    summary_rows$model_class_label == model_label &
      summary_rows$training_regime_label == regime_label
  ]
  format_range(vals, digits = 2)
}

excluded_variance_window_range <- range(summary_rows$nonpositive_generated_variance_windows, na.rm = TRUE)
excluded_expected_window_range <- range(summary_rows$nonpositive_generated_expected_variance_windows, na.rm = TRUE)

text_lines <- c(
  "# Taylor's Power Law Scaling",
  "",
  paste0(
    "The empirical Los Angeles building-volume mean-variance scaling fit used ",
    taylor_fit$n, " grid-resolution observations and estimated ",
    "sigma^2 = ", sprintf("%.2f", taylor_fit$coefficient_a),
    " * mu^", sprintf("%.2f", taylor_fit$beta),
    " (95% CI for beta: ", sprintf("%.2f", taylor_fit$beta_ci95_low),
    "-", sprintf("%.2f", taylor_fit$beta_ci95_high),
    "; R^2 = ", sprintf("%.3f", taylor_fit$r2), ")."
  ),
  "",
  paste0(
    "Across ", reference_summary$n_windows, " non-empty 256 x 256 windows in NLCD-based newly developed land ",
    "(", reference_summary$n_windows_for_log_metrics, " finite-positive windows for log metrics), ",
    "the reference morphology had a median log10 observed-to-expected variance ratio of ",
    sprintf("%.2f", reference_summary$median_log10_variance_ratio),
    " (IQR: ", sprintf("%.2f", reference_summary$q1_log10_variance_ratio),
    " to ", sprintf("%.2f", reference_summary$q3_log10_variance_ratio), ")."
  ),
  "",
  paste0(
    "For uncorrected epoch-1000 generated outputs, median log10 generated-to-expected variance ratios ranged from ",
    line_for("U-Net baseline", "Los Angeles Model"), " for the U-Net Los Angeles Model, ",
    line_for("U-Net baseline", "CONUS Model"), " for the U-Net CONUS Model, ",
    line_for("single-latent cGAN", "Los Angeles Model"), " for the single-latent cGAN Los Angeles Model, and ",
    line_for("single-latent cGAN", "CONUS Model"), " for the single-latent cGAN CONUS Model. ",
    "Values below zero indicate lower generated variance than expected from Taylor's Power Law at the same window-mean building volume."
  ),
  "",
  paste0(
    "All configurations produced 107 non-empty windows. Log-ratio summaries and log-log regressions used only finite positive ",
    "generated and expected variance values; the number of excluded zero/non-positive generated-variance windows ranged from ",
    excluded_variance_window_range[1], " to ", excluded_variance_window_range[2],
    " and the number of windows with undefined/non-positive Taylor-expected variance ranged from ",
    excluded_expected_window_range[1], " to ", excluded_expected_window_range[2],
    " across configurations."
  ),
  "",
  paste0(
    "Figure caption: Taylor's Power Law scaling of building-volume heterogeneity and comparison with uncorrected generated morphology. ",
    "The left panel shows the empirical relationship between mean building volume (\u03bc) and variance (\u03c3\u00b2) across grid resolutions ",
    "for the 2015 Los Angeles urban area; the solid line is the fitted Taylor relationship and the dashed lines show coefficient-bound ",
    "95% confidence limits used to derive expected-variance intervals. The middle panel compares Taylor-expected variance with generated ",
    "building-volume variance in NLCD-based newly developed pixels aggregated to 256 \u00d7 256-pixel GAN inference windows for the representative ",
    "single-latent cGAN Los Angeles Model at learning rate 0.0002; vertical bars denote expected-variance intervals and the diagonal line marks ",
    "equality. The bottom panel summarizes all epoch-1000 U-Net and single-latent cGAN configurations without bias correction, plotting the ",
    "median window-level log\u2081\u2080(generated variance / expected variance) by learning rate. Color indicates training regime and shape indicates ",
    "model class; the dashed horizontal line denotes generated = expected variance, the red horizontal line denotes the reference median, ",
    "and values below zero indicate lower generated variance than expected from the Taylor relationship."
  )
)
writeLines(text_lines, file.path(TEXT_DIR, "taylor_power_law_scaling_results.md"))

cat("Wrote Taylor fit to", file.path(TABLE_DIR, "taylor_power_law_fit.csv"), "\n")
cat("Wrote Taylor window metrics to", file.path(TABLE_DIR, "taylor_power_law_window_metrics_epoch1000_unet_single_latent_no_bias_correction.csv"), "\n")
cat("Wrote Taylor model summary to", file.path(TABLE_DIR, "taylor_power_law_model_summary_epoch1000_unet_single_latent_no_bias_correction.csv"), "\n")
cat("Wrote Taylor reference summary to", file.path(TABLE_DIR, "taylor_power_law_reference_summary.csv"), "\n")
cat("Provenance guard passed for corrected MSASample / 2A / BH / lr=0.001 source.\n")
