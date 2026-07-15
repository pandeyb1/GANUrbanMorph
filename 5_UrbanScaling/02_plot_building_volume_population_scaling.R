rm(list = ls())
gc()

font_cache_dir <- file.path(tempdir(), "r-fontconfig-cache")
dir.create(font_cache_dir, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(XDG_CACHE_HOME = font_cache_dir)

required_packages <- c("sf", "ggplot2", "scales", "gridExtra")
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
  normalizePath("02_plot_building_volume_population_scaling.R", mustWork = FALSE)
}

SCRIPT_DIR <- dirname(script_path())
DATA_DIR <- file.path(SCRIPT_DIR, "data")
TABLE_DIR <- file.path(SCRIPT_DIR, "tables")
FIGURE_DIR <- file.path(SCRIPT_DIR, "figures")
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)

msa_path <- file.path(DATA_DIR, "derived", "msa", "BuildingVolume2015MSA.gpkg")
msa_volume <- sf::st_read(msa_path, quiet = TRUE)
msa_volume <- as.data.frame(sf::st_drop_geometry(msa_volume))
msa_volume$ACK2E001 <- as.numeric(msa_volume$ACK2E001)
msa_volume$vols <- as.numeric(msa_volume$vols)
msa_volume <- msa_volume[is.finite(msa_volume$ACK2E001) & is.finite(msa_volume$vols) & msa_volume$vols > 0, ]

scaling_model <- lm(log(vols) ~ log(ACK2E001), data = msa_volume)
population_scaling_fit <- read.csv(file.path(TABLE_DIR, "population_scaling_fit.csv"))
volume_growth <- read.csv(file.path(TABLE_DIR, "building_volume_growth_epoch1000_unet_single_latent_no_bias_correction.csv"))
la_population <- read.csv(file.path(TABLE_DIR, "la_population_volume_summary.csv"))
la_msa_point <- msa_volume[grepl("^Los Angeles-Long Beach-Anaheim", msa_volume$NAME), ]
if (nrow(la_msa_point) != 1) {
  stop("Could not identify the Los Angeles MSA record in BuildingVolume2015MSA.gpkg.")
}
required_la_columns <- c("population_2010", "legacy_reference_total_2020_raw_units")
if (!all(required_la_columns %in% names(la_population))) {
  stop("Missing LA urban-area columns in la_population_volume_summary.csv. Re-run 01_compute_building_volume_population_scaling.R.")
}

get_legend <- function(plot) {
  grob <- ggplot2::ggplotGrob(plot)
  legend_index <- which(vapply(grob$grobs, function(x) x$name, character(1)) == "guide-box")
  if (length(legend_index) != 1) {
    stop("Expected exactly one legend grob.")
  }
  grob$grobs[[legend_index]]
}

line_x <- 10 ^ seq(4.5, 8, length.out = 200)
standard_ci <- confint(scaling_model, level = 0.95)
slope_mid <- as.numeric(coef(scaling_model)[2])
intercept_mid <- exp(as.numeric(coef(scaling_model)[1]))
slope_low <- standard_ci[2, 1]
slope_high <- standard_ci[2, 2]
intercept_low <- exp(standard_ci[1, 1])
intercept_high <- exp(standard_ci[1, 2])
line_data <- data.frame(
  population = line_x,
  fit = line_x ^ slope_mid * intercept_mid,
  lwr = line_x ^ slope_low * intercept_low,
  upr = line_x ^ slope_high * intercept_high
)

left_panel <- ggplot2::ggplot(msa_volume, ggplot2::aes(x = ACK2E001, y = vols)) +
  ggplot2::geom_point(color = "grey35", alpha = 0.22, size = 1.7) +
  ggplot2::geom_line(
    data = line_data,
    ggplot2::aes(x = population, y = fit),
    inherit.aes = FALSE,
    color = "#CC3311",
    linewidth = 0.9
  ) +
  ggplot2::geom_line(
    data = line_data,
    ggplot2::aes(x = population, y = lwr),
    inherit.aes = FALSE,
    color = "grey10",
    linewidth = 0.65,
    linetype = "dashed"
  ) +
  ggplot2::geom_line(
    data = line_data,
    ggplot2::aes(x = population, y = upr),
    inherit.aes = FALSE,
    color = "grey10",
    linewidth = 0.65,
    linetype = "dashed"
  ) +
  ggplot2::geom_point(
    data = la_population,
    ggplot2::aes(x = population_2010, y = legacy_reference_total_2020_raw_units),
    inherit.aes = FALSE,
    color = "#CC3311",
    fill = "#CC3311",
    shape = 21,
    size = 2.9
  ) +
  ggplot2::annotate(
    "text",
    x = min(msa_volume$ACK2E001) * 1.35,
    y = max(msa_volume$vols) / 1.9,
    hjust = 0,
    size = 3.1,
    label = paste0(
      "\u03b2 = ", sprintf("%.2f", population_scaling_fit$beta),
      "\n95% CI: ", sprintf("%.2f", population_scaling_fit$beta_ci95_low),
      "-", sprintf("%.2f", population_scaling_fit$beta_ci95_high),
      "\nR² = ", sprintf("%.3f", population_scaling_fit$r2)
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
    x = "Population",
    y = expression("Building Volume " * (m^3)),
    title = "2015 MSA Scaling"
  ) +
  ggplot2::theme_bw(base_size = 10) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", hjust = 0, size = 11),
    axis.title = ggplot2::element_text(size = 13),
    panel.grid.minor = ggplot2::element_blank(),
    legend.position = "none"
  )

volume_growth$learning_rate_label <- factor(
  c("1e-4", "2e-4", "5e-4", "1e-3")[
    match(volume_growth$learning_rate, c(0.0001, 0.0002, 0.0005, 0.001))
  ],
  levels = c("1e-4", "2e-4", "5e-4", "1e-3")
)
volume_growth$model_class_label <- factor(
  volume_growth$model_class_label,
  levels = c("U-Net baseline", "single-latent cGAN")
)
volume_growth$training_regime_label <- factor(
  volume_growth$training_regime_label,
  levels = c("Los Angeles Model", "CONUS Model")
)
volume_growth$line_group <- interaction(
  volume_growth$model_class_label,
  volume_growth$training_regime_label,
  volume_growth$learning_rate_label,
  drop = TRUE
)
reference_growth_pct <- unique(round(volume_growth$reference_growth_pct, 10))
if (length(reference_growth_pct) != 1) {
  stop("Reference growth percent should be identical across evaldfs.")
}

baseline_point <- data.frame(
  population = la_population$population_2010,
  volume = la_population$baseline_volume_2010_m3
)
reference_arrow <- data.frame(
  population_start = la_population$population_2010,
  volume_start = la_population$baseline_volume_2010_m3,
  population_end = la_population$population_2020,
  volume_end = la_population$reference_total_2020_m3
)
model_arrows <- transform(
  volume_growth,
  population_start = la_population$population_2010,
  volume_start = la_population$baseline_volume_2010_m3,
  population_end = la_population$population_2020,
  volume_end = generated_total_2020_m3
)
y_values <- c(
  baseline_point$volume,
  reference_arrow$volume_end,
  model_arrows$volume_end
)
y_padding <- diff(range(y_values)) * 0.18
if (!is.finite(y_padding) || y_padding == 0) {
  y_padding <- max(y_values) * 0.02
}
x_padding <- (la_population$population_2020 - la_population$population_2010) * 0.35

right_panel <- ggplot2::ggplot(
  model_arrows
) +
  ggplot2::geom_segment(
    data = model_arrows,
    ggplot2::aes(
      x = population_start,
      y = volume_start,
      xend = population_end,
      yend = volume_end,
      color = training_regime_label,
      linetype = learning_rate_label,
      group = line_group
    ),
    alpha = 0.34,
    linewidth = 0.5,
    arrow = grid::arrow(length = grid::unit(0.08, "in")),
    show.legend = c(color = TRUE, linetype = TRUE)
  ) +
  ggplot2::geom_segment(
    data = reference_arrow,
    ggplot2::aes(x = population_start, y = volume_start, xend = population_end, yend = volume_end),
    inherit.aes = FALSE,
    color = "#CC3311",
    linewidth = 1.05,
    arrow = grid::arrow(length = grid::unit(0.1, "in"))
  ) +
  ggplot2::geom_point(
    data = baseline_point,
    ggplot2::aes(x = population, y = volume),
    inherit.aes = FALSE,
    color = "#CC3311",
    fill = "#CC3311",
    shape = 21,
    size = 2.7
  ) +
  ggplot2::geom_point(
    data = reference_arrow,
    ggplot2::aes(x = population_end, y = volume_end),
    inherit.aes = FALSE,
    color = "#CC3311",
    fill = "#CC3311",
    shape = 21,
    size = 3.1
  ) +
  ggplot2::geom_point(
    data = model_arrows,
    ggplot2::aes(
      x = population_end,
      y = volume_end,
      color = training_regime_label,
      shape = model_class_label
    ),
    size = 2.7,
    stroke = 0.9
  ) +
  ggplot2::annotate(
    "text",
    x = la_population$population_2010,
    y = baseline_point$volume - y_padding * 0.55,
    label = "2010",
    hjust = 0,
    size = 2.9
  ) +
  ggplot2::annotate(
    "text",
    x = la_population$population_2020,
    y = reference_arrow$volume_end + y_padding * 0.55,
    label = paste0("Reference (", sprintf("%.2f", reference_growth_pct), "%)"),
    hjust = 1,
    color = "#CC3311",
    size = 2.9
  ) +
  ggplot2::scale_color_manual(
    values = c("Los Angeles Model" = "#0072B2", "CONUS Model" = "#D55E00"),
    labels = c("Los Angeles Model" = "LA Model", "CONUS Model" = "CONUS Model"),
    name = "Training"
  ) +
  ggplot2::scale_shape_manual(
    values = c("U-Net baseline" = 16, "single-latent cGAN" = 17),
    labels = c("U-Net baseline" = "U-Net", "single-latent cGAN" = "single-latent cGAN"),
    name = "Model"
  ) +
  ggplot2::scale_linetype_manual(
    values = c("1e-4" = "solid", "2e-4" = "dashed", "5e-4" = "dotdash", "1e-3" = "dotted"),
    name = "LR"
  ) +
  ggplot2::scale_x_continuous(
    labels = scales::label_number(scale = 1e-6, suffix = "M", accuracy = 0.01),
    limits = c(la_population$population_2010 - x_padding, la_population$population_2020 + x_padding * 1.55)
  ) +
  ggplot2::scale_y_continuous(
    labels = scales::label_number(scale = 1e-9, suffix = "B", accuracy = 0.01),
    limits = range(y_values) + c(-y_padding, y_padding)
  ) +
  ggplot2::labs(
    x = "Population",
    y = expression("Building Volume " * (10^9 ~ m^3)),
    title = "Los Angeles Urban Area 2010-2020 Growth"
  ) +
  ggplot2::theme_bw(base_size = 10) +
  ggplot2::guides(
    color = ggplot2::guide_legend(nrow = 1, order = 1, title.position = "left"),
    shape = ggplot2::guide_legend(nrow = 1, order = 2, title.position = "left"),
    linetype = ggplot2::guide_legend(
      nrow = 1,
      order = 3,
      title.position = "left",
      override.aes = list(color = "grey45", alpha = 1, linewidth = 0.7)
    )
  ) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", hjust = 0, size = 11),
    axis.title = ggplot2::element_text(size = 13),
    panel.grid.minor = ggplot2::element_blank(),
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

legend_grob <- get_legend(right_panel)
legend_grob <- grid::grobTree(
  grid::rectGrob(gp = grid::gpar(fill = "white", col = NA)),
  legend_grob
)
right_panel <- right_panel + ggplot2::theme(legend.position = "none")
panel_grob <- gridExtra::arrangeGrob(left_panel, right_panel, ncol = 2, widths = c(1.05, 1.15))
combined <- gridExtra::arrangeGrob(panel_grob, legend_grob, ncol = 1, heights = c(1, 0.075))

png_path <- file.path(FIGURE_DIR, "building_volume_population_scaling_epoch1000_unet_single_latent_no_bias_correction.png")
pdf_path <- file.path(FIGURE_DIR, "building_volume_population_scaling_epoch1000_unet_single_latent_no_bias_correction.pdf")
ggplot2::ggsave(png_path, combined, width = 8.8, height = 4.8, units = "in", dpi = 450, bg = "white")
ggplot2::ggsave(pdf_path, combined, width = 8.8, height = 4.8, units = "in", device = grDevices::cairo_pdf, bg = "white")

cat("Wrote figure PNG to", png_path, "\n")
cat("Wrote figure PDF to", pdf_path, "\n")
