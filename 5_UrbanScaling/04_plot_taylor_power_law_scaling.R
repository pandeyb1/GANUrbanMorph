rm(list = ls())
gc()

font_cache_dir <- file.path(tempdir(), "r-fontconfig-cache")
dir.create(font_cache_dir, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(XDG_CACHE_HOME = font_cache_dir)

required_packages <- c("ggplot2", "scales", "gridExtra")
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
  normalizePath("04_plot_taylor_power_law_scaling.R", mustWork = FALSE)
}

SCRIPT_DIR <- dirname(script_path())
DATA_DIR <- file.path(SCRIPT_DIR, "data")
TABLE_DIR <- file.path(SCRIPT_DIR, "tables")
FIGURE_DIR <- file.path(SCRIPT_DIR, "figures")
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)

required_files <- c(
  file.path(DATA_DIR, "derived", "taylor", "HeteroScalingLA.csv"),
  file.path(TABLE_DIR, "taylor_power_law_fit.csv"),
  file.path(TABLE_DIR, "taylor_power_law_window_metrics_epoch1000_unet_single_latent_no_bias_correction.csv"),
  file.path(TABLE_DIR, "taylor_power_law_model_summary_epoch1000_unet_single_latent_no_bias_correction.csv"),
  file.path(TABLE_DIR, "taylor_power_law_reference_summary.csv")
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0) {
  stop("Missing Taylor plotting input(s). Re-run 03_compute_taylor_power_law_scaling.R.\n",
       paste(missing_files, collapse = "\n"))
}

taylor_data <- read.csv(file.path(DATA_DIR, "derived", "taylor", "HeteroScalingLA.csv"), stringsAsFactors = FALSE)
taylor_data <- taylor_data[
  is.finite(taylor_data$Mean) & taylor_data$Mean > 0 &
    is.finite(taylor_data$Variance) & taylor_data$Variance > 0 &
    is.finite(taylor_data$Resolution),
]
taylor_data$resolution_km <- taylor_data$Resolution / 1000
taylor_fit <- read.csv(file.path(TABLE_DIR, "taylor_power_law_fit.csv"), stringsAsFactors = FALSE)
window_metrics <- read.csv(
  file.path(TABLE_DIR, "taylor_power_law_window_metrics_epoch1000_unet_single_latent_no_bias_correction.csv"),
  stringsAsFactors = FALSE
)
model_summary <- read.csv(
  file.path(TABLE_DIR, "taylor_power_law_model_summary_epoch1000_unet_single_latent_no_bias_correction.csv"),
  stringsAsFactors = FALSE
)
reference_summary <- read.csv(file.path(TABLE_DIR, "taylor_power_law_reference_summary.csv"), stringsAsFactors = FALSE)

if (nrow(taylor_data) != 301) {
  stop("Expected 301 Taylor scaling rows, found: ", nrow(taylor_data))
}
if (length(unique(paste(window_metrics$model_key, window_metrics$learning_rate))) != 16) {
  stop("Expected 16 model/learning-rate Taylor configurations.")
}
if (!all(table(paste(window_metrics$model_key, window_metrics$learning_rate)) == 107)) {
  stop("Expected 107 non-empty windows for every model/learning-rate Taylor configuration.")
}

taylor_model <- lm(log(Variance) ~ log(Mean), data = taylor_data)
line_x <- exp(seq(log(min(taylor_data$Mean)), log(max(taylor_data$Mean)), length.out = 250))
taylor_ci <- confint(taylor_model, level = 0.95)
line_log_x <- log(line_x)
taylor_line <- data.frame(
  mean = line_x,
  fit = exp(unname(coef(taylor_model)[1] + coef(taylor_model)[2] * line_log_x)),
  ci95_low = exp(unname(taylor_ci[1, 1] + taylor_ci[2, 1] * line_log_x)),
  ci95_high = exp(unname(taylor_ci[1, 2] + taylor_ci[2, 2] * line_log_x))
)
equation_label <- paste0(
  "atop(sigma^2 == ", sprintf("%.2f", taylor_fit$coefficient_a),
  " %.% mu^{", sprintf("%.2f", taylor_fit$beta),
  "}, R^2 == ", sprintf("%.3f", taylor_fit$r2), ")"
)

panel_a <- ggplot2::ggplot(taylor_data, ggplot2::aes(x = Mean, y = Variance)) +
  ggplot2::geom_point(ggplot2::aes(color = resolution_km), alpha = 0.7, size = 2.1) +
  ggplot2::geom_line(
    data = taylor_line,
    ggplot2::aes(x = mean, y = fit),
    inherit.aes = FALSE,
    color = "grey10",
    linewidth = 0.9
  ) +
  ggplot2::geom_line(
    data = taylor_line,
    ggplot2::aes(x = mean, y = ci95_low),
    inherit.aes = FALSE,
    color = "grey45",
    linewidth = 0.7,
    linetype = "dashed"
  ) +
  ggplot2::geom_line(
    data = taylor_line,
    ggplot2::aes(x = mean, y = ci95_high),
    inherit.aes = FALSE,
    color = "grey45",
    linewidth = 0.7,
    linetype = "dashed"
  ) +
  ggplot2::annotate(
    "text",
    x = min(taylor_data$Mean) * 1.55,
    y = max(taylor_data$Variance) / 2.2,
    hjust = 0,
    vjust = 1,
    size = 3.1,
    label = equation_label,
    parse = TRUE
  ) +
  ggplot2::scale_x_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  ) +
  ggplot2::scale_y_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  ) +
  ggplot2::scale_color_gradient(
    low = "#56B4E9",
    high = "#D55E00",
    name = "Grid\nresolution\n(km)"
  ) +
  ggplot2::labs(
    x = expression(mu),
    y = expression(sigma^2),
    title = "Empirical Taylor's Power Law"
  ) +
  ggplot2::theme_bw(base_size = 10) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", hjust = 0, size = 11),
    axis.title = ggplot2::element_text(size = 13),
    panel.grid.minor = ggplot2::element_blank(),
    legend.title = ggplot2::element_text(size = 7),
    legend.text = ggplot2::element_text(size = 7),
    legend.key.height = grid::unit(0.45, "cm"),
    legend.key.width = grid::unit(0.28, "cm"),
    legend.background = ggplot2::element_rect(fill = "white", color = NA)
  )

example_windows <- window_metrics[
  window_metrics$model_key == "single_latent_los_angeles_model" &
    abs(window_metrics$learning_rate - 0.0002) < 1e-12,
]
if (nrow(example_windows) != 107) {
  stop("Expected 107 windows for representative single-latent Los Angeles Model lr=0.0002.")
}
example_summary <- model_summary[
  model_summary$model_key == "single_latent_los_angeles_model" &
    abs(model_summary$learning_rate - 0.0002) < 1e-12,
]
if (nrow(example_summary) != 1) {
  stop("Could not identify the representative summary row.")
}
example_limits <- c(1e4, 1e7)

panel_b <- ggplot2::ggplot(
  example_windows,
  ggplot2::aes(x = generated_variance, y = generated_expected_variance)
) +
  ggplot2::geom_abline(slope = 1, intercept = 0, color = "grey50", linewidth = 0.7) +
  ggplot2::geom_errorbar(
    ggplot2::aes(ymin = generated_expected_variance_ci95_low, ymax = generated_expected_variance_ci95_high),
    width = 0,
    color = "grey30",
    alpha = 0.42,
    linewidth = 0.45
  ) +
  ggplot2::geom_point(color = "grey10", alpha = 0.38, size = 1.8) +
  ggplot2::annotate(
    "text",
    x = example_limits[1] * 1.25,
    y = example_limits[2] / 1.8,
    hjust = 0,
    vjust = 1,
    size = 3.0,
    label = paste0(
      "\u03b2 = ", sprintf("%.2f", example_summary$expected_vs_generated_log_slope),
      "\n95% CI: [", sprintf("%.2f", example_summary$expected_vs_generated_log_slope_ci95_low),
      ", ", sprintf("%.2f", example_summary$expected_vs_generated_log_slope_ci95_high), "]",
      "\nR\u00b2 = ", sprintf("%.3f", example_summary$expected_vs_generated_log_r2)
    )
  ) +
  ggplot2::scale_x_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  ) +
  ggplot2::scale_y_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  ) +
  ggplot2::labs(
    x = "Generated Variance",
    y = "Expected Variance",
    title = "Single-Latent LA Model (LR = 0.0002)"
  ) +
  ggplot2::theme_bw(base_size = 10) +
  ggplot2::coord_cartesian(xlim = example_limits, ylim = example_limits) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", hjust = 0, size = 11),
    axis.title = ggplot2::element_text(size = 13),
    panel.grid.minor = ggplot2::element_blank(),
    legend.position = "none"
  )

learning_rate_levels <- c(0.0001, 0.0002, 0.0005, 0.001)
learning_rate_labels <- c("1e-4", "2e-4", "5e-4", "1e-3")
model_summary$learning_rate_label <- factor(
  learning_rate_labels[match(model_summary$learning_rate, learning_rate_levels)],
  levels = learning_rate_labels
)
model_summary$model_class_label <- factor(
  model_summary$model_class_label,
  levels = c("U-Net baseline", "single-latent cGAN")
)
model_summary$training_regime_label <- factor(
  model_summary$training_regime_label,
  levels = c("Los Angeles Model", "CONUS Model")
)
model_summary$line_group <- interaction(
  model_summary$model_class_label,
  model_summary$training_regime_label,
  drop = TRUE
)

reference_band <- data.frame(
  xmin = 0.55,
  xmax = length(learning_rate_labels) + 0.45,
  y = reference_summary$median_log10_variance_ratio,
  ymin = reference_summary$q1_log10_variance_ratio,
  ymax = reference_summary$q3_log10_variance_ratio
)
model_summary_y_min <- min(model_summary$median_log10_variance_ratio, reference_band$y, 0, na.rm = TRUE) - 0.06
model_summary_y_max <- max(model_summary$median_log10_variance_ratio, reference_band$y, 0, na.rm = TRUE) + 0.08
point_dodge <- ggplot2::position_dodge(width = 0.44)

panel_c <- ggplot2::ggplot(model_summary) +
  ggplot2::geom_hline(yintercept = 0, color = "grey35", linewidth = 0.55, linetype = "dashed") +
  ggplot2::geom_hline(
    data = reference_band,
    ggplot2::aes(yintercept = y),
    color = "#CC3311",
    linewidth = 0.75
  ) +
  ggplot2::geom_point(
    ggplot2::aes(
      x = learning_rate_label,
      y = median_log10_variance_ratio,
      color = training_regime_label,
      shape = model_class_label,
      group = interaction(training_regime_label, model_class_label)
    ),
    size = 3.2,
    stroke = 0.9,
    alpha = 0.72,
    position = point_dodge
  ) +
  ggplot2::annotate(
    "text",
    x = "1e-3",
    y = 0.025,
    label = "Generated = Expected",
    hjust = 1,
    vjust = 0,
    color = "grey25",
    size = 2.8
  ) +
  ggplot2::annotate(
    "text",
    x = "1e-3",
    y = reference_band$y - 0.015,
    label = "Reference median",
    hjust = 1,
    vjust = 1,
    color = "#CC3311",
    size = 2.8
  ) +
  ggplot2::scale_color_manual(
    values = c("Los Angeles Model" = "#0072B2", "CONUS Model" = "#D55E00"),
    labels = c("Los Angeles Model" = "LA Model", "CONUS Model" = "CONUS Model"),
    name = NULL
  ) +
  ggplot2::scale_shape_manual(
    values = c("U-Net baseline" = 16, "single-latent cGAN" = 17),
    labels = c("U-Net baseline" = "U-Net", "single-latent cGAN" = "single-latent cGAN"),
    name = NULL
  ) +
  ggplot2::scale_x_discrete(drop = FALSE) +
  ggplot2::labs(
    x = "Learning Rate",
    y = expression(atop("Median log"[10] * " ratio", "(generated / expected)")),
    title = "Generated v/s Expected Variance: Across Model Configurations"
  ) +
  ggplot2::coord_cartesian(ylim = c(model_summary_y_min, model_summary_y_max), clip = "off") +
  ggplot2::guides(
    color = ggplot2::guide_legend(nrow = 1, order = 1, title.position = "left"),
    shape = ggplot2::guide_legend(nrow = 1, order = 2, title.position = "left")
  ) +
  ggplot2::theme_bw(base_size = 10) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", hjust = 0, size = 11),
    axis.title = ggplot2::element_text(size = 13),
    panel.grid.minor.x = ggplot2::element_blank(),
    panel.grid.minor.y = ggplot2::element_blank(),
    legend.position = "bottom",
    legend.box = "horizontal",
    legend.direction = "horizontal",
    legend.title = ggplot2::element_text(size = 7),
    legend.text = ggplot2::element_text(size = 7),
    legend.key.width = grid::unit(0.34, "cm"),
    legend.key.height = grid::unit(0.24, "cm"),
    legend.spacing.x = grid::unit(0.12, "cm"),
    legend.spacing.y = grid::unit(0, "cm"),
    legend.box.spacing = grid::unit(0.02, "cm"),
    legend.margin = ggplot2::margin(0, 0, 0, 0),
    legend.box.margin = ggplot2::margin(0, 0, 0, 0),
    legend.background = ggplot2::element_rect(fill = "white", color = NA),
    legend.box.background = ggplot2::element_rect(fill = "white", color = NA)
  )

top_panels <- gridExtra::arrangeGrob(panel_a, panel_b, ncol = 2, widths = c(1.05, 1.05))
combined <- gridExtra::arrangeGrob(top_panels, panel_c, ncol = 1, heights = c(1, 0.72))

png_path <- file.path(FIGURE_DIR, "taylor_power_law_scaling_epoch1000_unet_single_latent_no_bias_correction.png")
pdf_path <- file.path(FIGURE_DIR, "taylor_power_law_scaling_epoch1000_unet_single_latent_no_bias_correction.pdf")
ggplot2::ggsave(png_path, combined, width = 9.2, height = 7.2, units = "in", dpi = 450, bg = "white")
ggplot2::ggsave(pdf_path, combined, width = 9.2, height = 7.2, units = "in", device = grDevices::cairo_pdf, bg = "white")

cat("Wrote Taylor figure PNG to", png_path, "\n")
cat("Wrote Taylor figure PDF to", pdf_path, "\n")
