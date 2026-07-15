rm(list = ls())
gc()

script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
  }
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(normalizePath(sys.frames()[[1]]$ofile, mustWork = TRUE))
  }
  normalizePath("00_prepare_data.R", mustWork = FALSE)
}

SCRIPT_DIR <- dirname(script_path())
PROJECT_ROOT <- normalizePath(file.path(SCRIPT_DIR, "..", "..", ".."), mustWork = TRUE)
DATA_DIR <- file.path(SCRIPT_DIR, "data")
RAW_DIR <- file.path(DATA_DIR, "raw")
DERIVED_DIR <- file.path(DATA_DIR, "derived")
MANIFEST_DIR <- file.path(DATA_DIR, "manifests")
TABLE_DIR <- file.path(SCRIPT_DIR, "tables")
FIGURE_DIR <- file.path(SCRIPT_DIR, "figures")
TEXT_DIR <- file.path(SCRIPT_DIR, "text")

dir.create(RAW_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(DERIVED_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(MANIFEST_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TEXT_DIR, recursive = TRUE, showWarnings = FALSE)

copy_if_needed <- function(source, destination) {
  if (!file.exists(source)) {
    stop("Missing required source file: ", source)
  }
  dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
  source_info <- file.info(source)
  destination_info <- if (file.exists(destination)) file.info(destination) else NULL
  needs_copy <- !file.exists(destination) || is.na(destination_info$size) ||
    !identical(as.numeric(source_info$size), as.numeric(destination_info$size))
  if (needs_copy) {
    ok <- file.copy(source, destination, overwrite = TRUE, copy.date = TRUE)
    if (!ok) {
      stop("Could not copy ", source, " to ", destination)
    }
  }
  needs_copy
}

add_manifest_row <- function(source, destination, role, notes, copied) {
  destination_exists <- file.exists(destination)
  data.frame(
    role = role,
    source = normalizePath(source, mustWork = TRUE),
    destination = if (destination_exists) normalizePath(destination, mustWork = TRUE) else destination,
    copied_this_run = copied,
    bytes = if (destination_exists) file.info(destination)$size else NA_real_,
    md5 = if (destination_exists) unname(tools::md5sum(destination)) else NA_character_,
    notes = notes,
    stringsAsFactors = FALSE
  )
}

copy_one <- function(source_rel, destination_rel, role, notes = "") {
  source <- file.path(PROJECT_ROOT, source_rel)
  destination <- file.path(SCRIPT_DIR, destination_rel)
  copied <- copy_if_needed(source, destination)
  add_manifest_row(source, destination, role, notes, copied)
}

copy_matching <- function(source_dir_rel, destination_dir_rel, pattern, role, notes = "") {
  source_dir <- file.path(PROJECT_ROOT, source_dir_rel)
  if (!dir.exists(source_dir)) {
    stop("Missing required source directory: ", source_dir)
  }
  sources <- list.files(source_dir, pattern = pattern, full.names = TRUE)
  if (length(sources) == 0) {
    stop("No files matched pattern ", pattern, " in ", source_dir)
  }
  rows <- lapply(sources, function(source) {
    destination <- file.path(SCRIPT_DIR, destination_dir_rel, basename(source))
    copied <- copy_if_needed(source, destination)
    add_manifest_row(source, destination, role, notes, copied)
  })
  do.call(rbind, rows)
}

manifest_rows <- list()

manifest_rows[[length(manifest_rows) + 1]] <- copy_one(
  "Script/5_UrbanScaling/Data/BuildingVolume2015MSA.gpkg",
  "data/derived/msa/BuildingVolume2015MSA.gpkg",
  "derived_msa_building_volume",
  "Derived 2015 MSA building-volume table used for the national scaling fit."
)
manifest_rows[[length(manifest_rows) + 1]] <- copy_one(
  "Script/5_UrbanScaling/Data/MSA_notPR.gpkg",
  "data/derived/msa/MSA_notPR.gpkg",
  "derived_msa_boundary",
  "2015 metropolitan statistical areas excluding Puerto Rico."
)
manifest_rows[[length(manifest_rows) + 1]] <- copy_one(
  "Script/5_UrbanScaling/HeteroScalingLA.csv",
  "data/derived/taylor/HeteroScalingLA.csv",
  "derived_taylor_power_law_scaling",
  "Precomputed Los Angeles building-volume mean-variance scaling table used for Taylor's Power Law fit."
)

manifest_rows[[length(manifest_rows) + 1]] <- copy_matching(
  "Script/5_UrbanScaling/Data/nhgis0039_csv",
  "data/raw/nhgis/nhgis0039_csv",
  ".*",
  "raw_nhgis_2015_cbsa_population",
  "NHGIS 2015 CBSA population CSV and codebook."
)
manifest_rows[[length(manifest_rows) + 1]] <- copy_matching(
  "Script/5_UrbanScaling/Data/nhgis0039_shape/nhgis0039_shapefile_tl2015_us_cbsa_2015",
  "data/raw/nhgis/nhgis0039_shapefile_tl2015_us_cbsa_2015",
  "US_cbsa_2015\\.(shp|shx|dbf|prj|sbn|sbx|xml)$",
  "raw_nhgis_2015_cbsa_shape",
  "NHGIS 2015 CBSA shape files used to define MSA geography."
)
manifest_rows[[length(manifest_rows) + 1]] <- copy_matching(
  "Script/5_UrbanScaling/Data/nhgis0040_csv",
  "data/raw/nhgis/nhgis0040_csv",
  "nhgis0040_ds(172|258)_.*\\.(csv|txt)$",
  "raw_nhgis_urban_area_population",
  "NHGIS 2010 and 2020 urban-area population CSVs and codebooks."
)
manifest_rows[[length(manifest_rows) + 1]] <- copy_matching(
  "Script/5_UrbanScaling/Data/nhgis0040_shape/nhgis0040_shapefile_tl2010_us_urb_area_2010",
  "data/raw/nhgis/nhgis0040_shapefile_tl2010_us_urb_area_2010",
  "US_urb_area_2010\\.(shp|shx|dbf|prj|xml)$",
  "raw_nhgis_2010_urban_area_shape",
  "NHGIS 2010 urban-area shape files retained for provenance."
)

manifest_rows[[length(manifest_rows) + 1]] <- copy_one(
  "Script/5_UrbanScaling/Data/Landscan/landscan-global-2010-assets/landscan-global-2010.tif",
  "data/raw/landscan/landscan-global-2010.tif",
  "raw_landscan_population",
  "LandScan 2010 population raster retained for provenance."
)
manifest_rows[[length(manifest_rows) + 1]] <- copy_one(
  "Script/5_UrbanScaling/Data/Landscan/landscan-global-2020-assets/landscan-global-2020.tif",
  "data/raw/landscan/landscan-global-2020.tif",
  "raw_landscan_population",
  "LandScan 2020 population raster retained for provenance."
)

la_files <- c(
  "BF2010.tif", "BH2010.tif", "BF_2020.tif", "BH_2020.tif",
  "nlcd2010LA.tif", "nlcd2020LA.tif", "ChangeMask2010_2020.tif"
)
for (file_name in la_files) {
  manifest_rows[[length(manifest_rows) + 1]] <- copy_one(
    file.path("Data/TrainingData/tmp", file_name),
    file.path("data/raw/la_morphology", file_name),
    "raw_la_morphology",
    "Los Angeles 2010/2020 morphology and NLCD raster input."
  )
}

model_specs <- data.frame(
  model_key = c(
    "unet_los_angeles_model",
    "unet_conus_model",
    "single_latent_los_angeles_model",
    "single_latent_conus_model"
  ),
  model_class = c("1", "1", "2A", "2A"),
  model_class_label = c("U-Net baseline", "U-Net baseline", "single-latent cGAN", "single-latent cGAN"),
  training_regime = c("LALegacy", "MSASample", "LALegacy", "MSASample"),
  training_regime_label = c("Los Angeles Model", "CONUS Model", "Los Angeles Model", "CONUS Model"),
  source_output_dir = c(
    "Script/4_Inference/Outputs/MorphologyGrowthEpoch1000UNetLALegacy",
    "Script/4_Inference/Outputs/MorphologyGrowthEpoch1000UNetCONUS",
    "Script/4_Inference/Outputs/MorphologyGrowthEpoch1000SingleLatent",
    "Script/4_Inference/Outputs/MorphologyGrowthEpoch1000SingleLatentCONUS"
  ),
  stringsAsFactors = FALSE
)
write.csv(model_specs, file.path(MANIFEST_DIR, "model_input_specs.csv"), row.names = FALSE)

for (i in seq_len(nrow(model_specs))) {
  spec <- model_specs[i, ]
  evaldf_source_dir <- file.path(spec$source_output_dir, "evaldf")
  manifest_rows[[length(manifest_rows) + 1]] <- copy_matching(
    evaldf_source_dir,
    file.path("data/model_evaldfs", spec$model_key),
    "Evaldf_.*\\.csv$",
    "model_evaldf",
    paste(spec$model_class_label, spec$training_regime_label, "paired BF/BH predictions.")
  )
  manifest_rows[[length(manifest_rows) + 1]] <- copy_one(
    file.path(spec$source_output_dir, "morphology_growth_epoch1000_metrics.csv"),
    file.path("data/model_metrics", spec$model_key, "morphology_growth_epoch1000_metrics.csv"),
    "model_metrics",
    paste(spec$model_class_label, spec$training_regime_label, "morphology metrics and model source paths.")
  )
}

input_manifest <- do.call(rbind, manifest_rows)
write.csv(input_manifest, file.path(MANIFEST_DIR, "input_file_manifest.csv"), row.names = FALSE)

directory_size <- function(path) {
  files <- list.files(path, recursive = TRUE, full.names = TRUE, all.files = TRUE, no.. = TRUE)
  sum(file.info(files)$size, na.rm = TRUE)
}

metro_buildings_path <- normalizePath(
  file.path(PROJECT_ROOT, "Script/5_UrbanScaling/Data/MetroBuildings"),
  mustWork = TRUE
)
external_manifest <- data.frame(
  role = "external_raw_building_geometry_not_copied",
  source = metro_buildings_path,
  destination = NA_character_,
  bytes = directory_size(metro_buildings_path),
  notes = paste(
    "Per revision decision, the 19 GB per-MSA building GeoPackage directory is not copied.",
    "The derived BuildingVolume2015MSA.gpkg table copied into data/derived/msa is used for figure reproduction."
  ),
  stringsAsFactors = FALSE
)
write.csv(external_manifest, file.path(MANIFEST_DIR, "external_data_manifest.csv"), row.names = FALSE)

cat("Wrote input manifest to", file.path(MANIFEST_DIR, "input_file_manifest.csv"), "\n")
cat("Wrote external data manifest to", file.path(MANIFEST_DIR, "external_data_manifest.csv"), "\n")
