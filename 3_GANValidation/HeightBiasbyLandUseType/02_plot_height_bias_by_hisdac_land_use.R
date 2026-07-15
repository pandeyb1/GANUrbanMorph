#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
})

font_cache_dir <- file.path(tempdir(), "fontconfig")
dir.create(font_cache_dir, showWarnings = FALSE, recursive = TRUE)
Sys.setenv(XDG_CACHE_HOME = tempdir())

script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- sub(file_arg, "", script_args[grep(file_arg, script_args)])
if (length(script_path) == 0) {
  script_dir <- getwd()
} else {
  script_dir <- dirname(normalizePath(script_path))
}

base_dir <- script_dir
tables_dir <- file.path(base_dir, "tables")
figures_dir <- file.path(base_dir, "figures")
text_dir <- file.path(base_dir, "text")
dir.create(tables_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(text_dir, showWarnings = FALSE, recursive = TRUE)

tile_stats_path <- file.path(
  tables_dir,
  "height_bias_by_hisdac_land_use_tile_stats_epoch1000_conus_unet_single_latent_oracle_bf.csv"
)
summary_path <- file.path(
  tables_dir,
  "height_bias_by_hisdac_land_use_summary_epoch1000_conus_unet_single_latent_oracle_bf.csv"
)
png_path <- file.path(
  figures_dir,
  "height_bias_by_hisdac_land_use_epoch1000_conus_unet_single_latent_oracle_bf.png"
)
pdf_path <- file.path(
  figures_dir,
  "height_bias_by_hisdac_land_use_epoch1000_conus_unet_single_latent_oracle_bf.pdf"
)
text_path <- file.path(text_dir, "height_bias_by_hisdac_land_use_results.md")

if (!file.exists(tile_stats_path)) {
  stop("Missing tile stats table. Run 01_aggregate_height_bias_by_hisdac_land_use.py first: ", tile_stats_path)
}

land_use_order <- c(
  "Commercial",
  "Industrial",
  "Residential-Income",
  "Residential-Owned",
  "Governmental",
  "Recreational",
  "Agriculture",
  "Vacant Land"
)
model_order <- c("U-Net", "Single-latent cGAN")
lr_order <- c("lr_0p0001", "lr_0p0002", "lr_0p0005", "lr_0p001")
wmean <- function(x, w) {
  valid <- is.finite(x) & is.finite(w) & w > 0
  if (!any(valid)) {
    return(NA_real_)
  }
  sum(x[valid] * w[valid]) / sum(w[valid])
}

tile_stats <- read_csv(tile_stats_path, show_col_types = FALSE) %>%
  mutate(
    land_use_type = factor(land_use_type, levels = land_use_order),
    model_class = factor(model_class, levels = model_order),
    learning_rate_label = factor(learning_rate_label, levels = lr_order),
    learning_rate_display = recode(
      as.character(learning_rate_label),
      "lr_0p0001" = "0.0001",
      "lr_0p0002" = "0.0002",
      "lr_0p0005" = "0.0005",
      "lr_0p001" = "0.001"
    )
  ) %>%
  filter(!is.na(land_use_type))

summary_tbl <- tile_stats %>%
  group_by(
    training_regime,
    training_regime_label,
    dataset_name,
    family,
    model_class,
    learning_rate,
    learning_rate_label,
    learning_rate_display,
    checkpoint,
    target,
    evaluation_mode,
    mask_type,
    hisdac_code,
    land_use_type
  ) %>%
  summarise(
    n_tiles = n_distinct(filename),
    median_tile_bias_m = median(mean_bias_m, na.rm = TRUE),
    tile_bias_iqr_low_m = quantile(mean_bias_m, 0.25, na.rm = TRUE, names = FALSE),
    tile_bias_iqr_high_m = quantile(mean_bias_m, 0.75, na.rm = TRUE, names = FALSE),
    reference_mean_bh_m = wmean(reference_mean_bh_m, n_pixels),
    generated_mean_bh_m = wmean(generated_mean_bh_m, n_pixels),
    mean_bias_m = wmean(mean_bias_m, n_pixels),
    mae_m = wmean(mae_m, n_pixels),
    n_pixels = sum(n_pixels, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    n_reference_built_pixels = if_else(mask_type == "reference_built", n_pixels, NA_real_)
  ) %>%
  arrange(mask_type, land_use_type, model_class, learning_rate)

write_csv(summary_tbl, summary_path)

plot_summary <- summary_tbl %>%
  filter(mask_type == "reference_built")

reference_panel <- plot_summary %>%
  group_by(land_use_type) %>%
  summarise(
    panel = "Mean Building Height",
    series = "Reference",
    model_class = factor("Reference", levels = c("Reference", model_order)),
    learning_rate_display = "Reference",
    y = mean(reference_mean_bh_m, na.rm = TRUE),
    ymin = NA_real_,
    ymax = NA_real_,
    .groups = "drop"
  )

generated_panel <- plot_summary %>%
  transmute(
    panel = "Mean Building Height",
    series = as.character(model_class),
    model_class = factor(as.character(model_class), levels = c("Reference", model_order)),
    learning_rate_display,
    land_use_type,
    y = generated_mean_bh_m,
    ymin = NA_real_,
    ymax = NA_real_
  )

bias_panel <- plot_summary %>%
  transmute(
    panel = "Generated - Reference Bias",
    series = as.character(model_class),
    model_class = factor(as.character(model_class), levels = c("Reference", model_order)),
    learning_rate_display,
    land_use_type,
    y = mean_bias_m,
    ymin = tile_bias_iqr_low_m,
    ymax = tile_bias_iqr_high_m
  )

average_bias_panel <- plot_summary %>%
  group_by(land_use_type, model_class) %>%
  summarise(
    panel = factor("Generated - Reference Bias", levels = c("Mean Building Height", "Generated - Reference Bias")),
    y = mean(mean_bias_m, na.rm = TRUE),
    .groups = "drop"
  )

plot_data <- bind_rows(reference_panel, generated_panel, bias_panel) %>%
  mutate(
    panel = factor(panel, levels = c("Mean Building Height", "Generated - Reference Bias")),
    learning_rate_display = factor(
      learning_rate_display,
      levels = c("Reference", "0.0001", "0.0002", "0.0005", "0.001")
    )
  )

pd <- position_dodge(width = 0.55)
color_values <- c(
  "Reference" = "black",
  "U-Net" = "#2C7FB8",
  "Single-latent cGAN" = "#D95F02"
)
shape_values <- c(
  "Reference" = 16,
  "0.0001" = 21,
  "0.0002" = 22,
  "0.0005" = 24,
  "0.001" = 23
)

figure <- ggplot(plot_data, aes(x = land_use_type, y = y)) +
  geom_hline(
    data = data.frame(panel = factor("Generated - Reference Bias", levels = levels(plot_data$panel))),
    aes(yintercept = 0),
    linewidth = 0.35,
    color = "grey35"
  ) +
  geom_line(
    data = filter(plot_data, series == "Reference"),
    aes(group = 1, color = model_class),
    linewidth = 0.55
  ) +
  geom_errorbar(
    data = filter(plot_data, panel == "Generated - Reference Bias"),
    aes(ymin = ymin, ymax = ymax, color = model_class, group = interaction(model_class, learning_rate_display)),
    position = pd,
    width = 0.18,
    alpha = 0.65,
    linewidth = 0.35
  ) +
  geom_errorbar(
    data = average_bias_panel,
    aes(
      x = land_use_type,
      ymin = y,
      ymax = y,
      color = model_class,
      group = model_class
    ),
    position = pd,
    width = 0.42,
    linewidth = 1.15,
    alpha = 0.95
  ) +
  geom_point(
    aes(
      color = model_class,
      shape = learning_rate_display,
      group = interaction(model_class, learning_rate_display)
    ),
    position = pd,
    size = 2.7,
    stroke = 0.75,
    alpha = 0.82
  ) +
  facet_grid(panel ~ ., scales = "free_y") +
  scale_color_manual(values = color_values, name = "Series", drop = FALSE) +
  scale_shape_manual(values = shape_values, name = "Learning rate", drop = FALSE) +
  labs(
    x = "2015 Land-use Type",
    y = "Building Height (m)"
  ) +
  theme_bw(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1),
    axis.title = element_text(size = 13),
    strip.background = element_rect(fill = "grey92", color = "grey70"),
    strip.text = element_text(size = 12, face = "bold"),
    legend.position = "bottom",
    legend.box = "vertical",
    panel.grid.minor = element_blank(),
    plot.margin = margin(8, 10, 8, 8)
  )

ggsave(png_path, figure, width = 8.3, height = 7.6, dpi = 450)
ggsave(pdf_path, figure, width = 8.3, height = 7.6, device = cairo_pdf)

main_summary <- plot_summary %>%
  group_by(model_class) %>%
  summarise(
    bias_min = min(mean_bias_m, na.rm = TRUE),
    bias_max = max(mean_bias_m, na.rm = TRUE),
    mae_min = min(mae_m, na.rm = TRUE),
    mae_max = max(mae_m, na.rm = TRUE),
    .groups = "drop"
  )

strongest <- plot_summary %>%
  arrange(mean_bias_m) %>%
  slice(1)

land_use_summary <- plot_summary %>%
  group_by(land_use_type) %>%
  summarise(
    reference_mean = mean(reference_mean_bh_m, na.rm = TRUE),
    generated_mean = mean(generated_mean_bh_m, na.rm = TRUE),
    bias_mean = mean(mean_bias_m, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(bias_mean)

format_range <- function(x, y) {
  paste0(sprintf("%.2f", x), " to ", sprintf("%.2f", y))
}

unet <- filter(main_summary, model_class == "U-Net")
single <- filter(main_summary, model_class == "Single-latent cGAN")

text_lines <- c(
  "# Height Bias by HISDAC Land-Use Type",
  "",
  paste0(
    "Using the CONUS Test Dataset and reference-BF-conditioned BH inference, ",
    "we evaluated building-height bias across HISDAC 2015 land-use classes for ",
    "the CONUS-trained U-Net baseline and single-latent cGAN at epoch 1000."
  ),
  "",
  paste0(
    "Across land-use classes and learning rates, U-Net mean BH bias ranged from ",
    format_range(unet$bias_min, unet$bias_max),
    " m, whereas single-latent cGAN mean BH bias ranged from ",
    format_range(single$bias_min, single$bias_max),
    " m. The strongest underprediction occurred for ",
    as.character(strongest$land_use_type),
    " in the ",
    as.character(strongest$model_class),
    " at learning rate ",
    strongest$learning_rate_display,
    ", with mean bias of ",
    sprintf("%.2f", strongest$mean_bias_m),
    " m. These results show that the BH underprediction noted in the main validation results is not uniform across land-use contexts, and that land-use classes with taller reference buildings exhibit stronger negative height bias."
  ),
  "",
  "Land-use classes ordered by average generated-minus-reference bias:",
  paste0(
    "- ",
    as.character(land_use_summary$land_use_type),
    ": ",
    sprintf("%.2f", land_use_summary$bias_mean),
    " m"
  )
)

writeLines(text_lines, text_path)

message("Wrote summary table: ", summary_path)
message("Wrote figure: ", png_path)
message("Wrote figure: ", pdf_path)
message("Wrote manuscript text: ", text_path)
