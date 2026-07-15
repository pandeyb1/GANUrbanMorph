#!/usr/bin/env python3
"""Aggregate CONUS BH prediction bias by HISDAC 2015 land-use type.

This script evaluates epoch-1000 CONUS-trained U-Net and single-latent cGAN
building-height generators on the CONUS stratified test archive. BH inference is
reference-BF-conditioned (oracle_bf), matching the user's "original BF" request.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
OVERALL_VALIDATION_DIR = SCRIPT_DIR.parent / "OverallValidation"
if str(OVERALL_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(OVERALL_VALIDATION_DIR))

from validation_core import (  # noqa: E402
    DatasetConfig,
    DatasetBundle,
    decode_prediction_batch,
    encode_bf_unit,
    encode_condition_batch,
    height_feet_to_meters,
    infer_batch,
    load_generator_from_run,
    model_folder_name,
    read_json,
    run_dir_name_for_model,
    choose_device,
    set_torch_determinism,
)


DEFAULT_TEST_ROOT = (
    REPO_ROOT
    / "Report/GIScRSSubmission/Revision/codev3/Stratified_Archive/output_test"
)
DEFAULT_HISDAC_RASTER = (
    REPO_ROOT / "Data/HISDAC/Majority/Majority/Majority_2015.tif"
)
DEFAULT_MODEL_ROOT = Path("/Volumes/HDD/Models/MSASampleModels")

LEARNING_RATES = [0.0001, 0.0002, 0.0005, 0.001]
FAMILIES = ["1", "2A"]
CHECKPOINT = 1000
TRAINING_REGIME = "MSASample"
DATASET_NAME = "CONUSStratifiedTest"
TARGET = "BH"
EVALUATION_MODE = "oracle_bf"
PRIMARY_LATENT_SEED = 17
REFERENCE_BH_THRESHOLD_M = 0.5
EXPECTED_CORRECTED_2A_BH_LR001 = (
    DEFAULT_MODEL_ROOT
    / "2A_BH_cGANRandomVecFixed_MSASample"
    / "lr_0p001_seed_2026"
    / "generator_epoch_1000.pth"
)

HISDAC_LABELS = {
    1: "Agriculture",
    2: "Commercial",
    3: "Industrial",
    4: "Recreational",
    5: "Residential-Income",
    6: "Residential-Owned",
    7: "Governmental",
    8: "Vacant Land",
}
MODEL_CLASS_LABELS = {
    "1": "U-Net",
    "2A": "Single-latent cGAN",
}


@dataclass(frozen=True)
class ModelSpec:
    family: str
    model_class: str
    learning_rate: float
    run_dir: Path
    checkpoint_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate CONUS test BH bias by HISDAC 2015 land-use type."
    )
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--hisdac-raster", type=Path, default=DEFAULT_HISDAC_RASTER)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-seed", type=int, default=PRIMARY_LATENT_SEED)
    parser.add_argument("--cache-mode", choices=["lazy", "all"], default="lazy")
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "text": output_dir / "text",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.test_root.exists():
        raise FileNotFoundError(f"Missing CONUS test root: {args.test_root}")
    for subdir in ["central", "BFrac", "BHeight"]:
        path = args.test_root / subdir
        if not path.exists():
            raise FileNotFoundError(f"Missing CONUS test subdirectory: {path}")
    manifest = args.test_root / "tile_manifest_rendered.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing CONUS rendered tile manifest: {manifest}")
    manifest_df = pd.read_csv(manifest)
    if manifest_df.shape[0] != 2000:
        raise ValueError(
            f"Expected 2,000 CONUS test tiles in tile_manifest_rendered.csv; "
            f"found {manifest_df.shape[0]}"
        )
    if not args.hisdac_raster.exists():
        raise FileNotFoundError(f"Missing HISDAC raster: {args.hisdac_raster}")
    if not args.model_root.exists():
        raise FileNotFoundError(
            f"Missing model root: {args.model_root}. Mount /Volumes/HDD before running."
        )


def lr_token(learning_rate: float) -> str:
    return f"lr_{learning_rate:g}".replace(".", "p")


def model_specs(model_root: Path) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for family in FAMILIES:
        folder = model_folder_name(TRAINING_REGIME, family, TARGET)
        for learning_rate in LEARNING_RATES:
            run_dir = (
                model_root
                / folder
                / run_dir_name_for_model(TRAINING_REGIME, family, TARGET, learning_rate)
            )
            checkpoint_path = run_dir / f"generator_epoch_{CHECKPOINT:04d}.pth"
            metadata_path = run_dir / "metadata.json"
            if family == "2A" and round(learning_rate, 4) == 0.001:
                expected = (
                    model_root
                    / "2A_BH_cGANRandomVecFixed_MSASample"
                    / "lr_0p001_seed_2026"
                    / "generator_epoch_1000.pth"
                )
                if checkpoint_path != expected:
                    raise ValueError(
                        "CONUS single-latent BH lr=0.001 must use seed_2026: "
                        f"{checkpoint_path}"
                    )
                if "lr_0p001_seed_5026" in str(checkpoint_path):
                    raise ValueError(
                        "Invalid source selected for CONUS single-latent BH lr=0.001: "
                        f"{checkpoint_path}"
                    )
            if "lr_0p001_seed_5026" in str(checkpoint_path) and family == "2A":
                raise ValueError(f"Forbidden seed_5026 source encountered: {checkpoint_path}")
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
            if not metadata_path.exists():
                raise FileNotFoundError(f"Missing metadata: {metadata_path}")
            metadata = read_json(metadata_path)
            specs.append(
                ModelSpec(
                    family=family,
                    model_class=MODEL_CLASS_LABELS[family],
                    learning_rate=learning_rate,
                    run_dir=run_dir,
                    checkpoint_path=checkpoint_path,
                    metadata_path=metadata_path,
                    metadata=metadata,
                )
            )
    if len(specs) != 8:
        raise ValueError(f"Expected 8 model configurations; found {len(specs)}")
    return specs


def write_model_manifest(specs: list[ModelSpec], path: Path) -> None:
    rows = []
    for spec in specs:
        rows.append(
            {
                "training_regime": TRAINING_REGIME,
                "training_regime_label": "CONUS Model",
                "family": spec.family,
                "model_class": spec.model_class,
                "target": TARGET,
                "evaluation_mode": EVALUATION_MODE,
                "learning_rate": spec.learning_rate,
                "checkpoint": CHECKPOINT,
                "run_dir": str(spec.run_dir),
                "checkpoint_path": str(spec.checkpoint_path),
                "metadata_path": str(spec.metadata_path),
                "condition_kind": spec.metadata.get("condition_kind"),
                "condition_channels": spec.metadata.get("condition_channels"),
                "latent_dim": spec.metadata.get("latent_dim", ""),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def overlay_hisdac_to_tile(
    hisdac_src: rasterio.io.DatasetReader,
    tile_path: Path,
) -> np.ndarray:
    with rasterio.open(tile_path) as tile_src:
        # The HISDAC raster and test tiles use the same Albers projection
        # parameters, but differ in datum metadata. Calling GDAL reprojection can
        # trigger an unavailable online NADCON grid. Reading by projected tile
        # bounds preserves the intended nearest-neighbor overlay.
        window = from_bounds(*tile_src.bounds, transform=hisdac_src.transform)
        destination = hisdac_src.read(
            1,
            window=window,
            out_shape=tile_src.shape,
            boundless=True,
            fill_value=0,
            resampling=Resampling.nearest,
        )
    destination = np.nan_to_num(destination, nan=0.0, posinf=0.0, neginf=0.0)
    rounded = np.rint(destination).astype(np.uint8)
    rounded[~np.isin(rounded, list(HISDAC_LABELS))] = 0
    return rounded


def build_hisdac_cache(
    filenames: list[str],
    test_root: Path,
    hisdac_raster: Path,
) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    with rasterio.open(hisdac_raster) as hisdac_src:
        for index, filename in enumerate(filenames, start=1):
            cache[filename] = overlay_hisdac_to_tile(
                hisdac_src,
                test_root / "central" / filename,
            )
            if index % 250 == 0:
                print(f"  HISDAC overlay: {index}/{len(filenames)} tiles")
    return cache


def summarize_tile_by_land_use(
    *,
    filename: str,
    manifest_lookup: dict[str, dict[str, Any]],
    spec: ModelSpec,
    land_use: np.ndarray,
    reference_bh_m: np.ndarray,
    generated_bh_m: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest = manifest_lookup.get(filename, {})
    masks = {
        "reference_built": reference_bh_m > REFERENCE_BH_THRESHOLD_M,
        "active_union": (reference_bh_m > REFERENCE_BH_THRESHOLD_M)
        | (generated_bh_m > REFERENCE_BH_THRESHOLD_M),
    }
    error = generated_bh_m - reference_bh_m
    abs_error = np.abs(error)
    for mask_type, base_mask in masks.items():
        for code, label in HISDAC_LABELS.items():
            mask = base_mask & (land_use == code)
            count = int(mask.sum())
            if count == 0:
                continue
            rows.append(
                {
                    "training_regime": TRAINING_REGIME,
                    "training_regime_label": "CONUS Model",
                    "dataset_name": DATASET_NAME,
                    "family": spec.family,
                    "model_class": spec.model_class,
                    "learning_rate": spec.learning_rate,
                    "learning_rate_label": lr_token(spec.learning_rate),
                    "checkpoint": CHECKPOINT,
                    "target": TARGET,
                    "evaluation_mode": EVALUATION_MODE,
                    "mask_type": mask_type,
                    "filename": filename,
                    "geoid": manifest.get("geoid", ""),
                    "msa_name": manifest.get("msa_name", ""),
                    "source_states": manifest.get("source_states", ""),
                    "hisdac_code": code,
                    "land_use_type": label,
                    "n_pixels": count,
                    "reference_mean_bh_m": float(reference_bh_m[mask].mean()),
                    "generated_mean_bh_m": float(generated_bh_m[mask].mean()),
                    "reference_median_bh_m": float(np.median(reference_bh_m[mask])),
                    "generated_median_bh_m": float(np.median(generated_bh_m[mask])),
                    "reference_p90_bh_m": float(np.percentile(reference_bh_m[mask], 90)),
                    "generated_p90_bh_m": float(np.percentile(generated_bh_m[mask], 90)),
                    "mean_bias_m": float(error[mask].mean()),
                    "median_bias_m": float(np.median(error[mask])),
                    "mae_m": float(abs_error[mask].mean()),
                }
            )
    return rows


def write_overlay_coverage(
    filenames: list[str],
    dataset: DatasetBundle,
    land_use_cache: dict[str, np.ndarray],
    path: Path,
) -> None:
    rows = []
    total_ref_built = 0
    total_ref_built_valid = 0
    for filename in filenames:
        _central, _bf, bh = dataset.get_tile(filename)
        reference_bh_m = height_feet_to_meters(bh)
        reference_mask = reference_bh_m > REFERENCE_BH_THRESHOLD_M
        valid_mask = np.isin(land_use_cache[filename], list(HISDAC_LABELS))
        ref_count = int(reference_mask.sum())
        valid_count = int((reference_mask & valid_mask).sum())
        total_ref_built += ref_count
        total_ref_built_valid += valid_count
        rows.append(
            {
                "filename": filename,
                "reference_built_pixels": ref_count,
                "reference_built_pixels_with_valid_hisdac": valid_count,
                "valid_hisdac_coverage": valid_count / ref_count if ref_count else np.nan,
            }
        )
    coverage = total_ref_built_valid / total_ref_built if total_ref_built else np.nan
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    message = f"HISDAC valid coverage in reference-built pixels: {coverage:.2%}"
    if np.isfinite(coverage) and coverage < 0.95:
        print(f"WARNING: {message}; below 95% validation threshold.")
    else:
        print(message)


def main() -> None:
    args = parse_args()
    dirs = ensure_dirs(args.output_dir)
    validate_inputs(args)
    set_torch_determinism()
    device = choose_device(args.device)
    print(f"Using device: {device}")

    specs = model_specs(args.model_root)
    write_model_manifest(
        specs,
        dirs["tables"] / "height_bias_by_hisdac_land_use_model_source_manifest.csv",
    )

    dataset = DatasetBundle(
        DatasetConfig(
            name=DATASET_NAME,
            root=args.test_root,
            bf_condition_dir="central",
            bf_target_dir="BFrac",
            bh_condition_dir="BFrac",
            bh_target_dir="BHeight",
            manifest_path=None,
        ),
        cache_mode=args.cache_mode,
    )
    filenames = dataset.filenames
    if len(dataset.filenames) != 2000:
        raise ValueError(f"Expected 2,000 matched CONUS test tiles; found {len(dataset.filenames)}")
    print(f"Matched CONUS test tiles: {len(dataset.filenames)}")

    manifest_df = pd.read_csv(args.test_root / "tile_manifest_rendered.csv")
    manifest_lookup = {
        str(row["filename"]): row.to_dict()
        for _, row in manifest_df.iterrows()
    }

    print("Building HISDAC 2015 land-use overlay cache...")
    land_use_cache = build_hisdac_cache(filenames, args.test_root, args.hisdac_raster)
    write_overlay_coverage(
        filenames,
        dataset,
        land_use_cache,
        dirs["tables"] / "height_bias_by_hisdac_land_use_overlay_coverage.csv",
    )

    all_rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(specs, start=1):
        print(
            f"Running {spec_index}/{len(specs)}: {spec.model_class}, "
            f"lr={spec.learning_rate:g}, epoch={CHECKPOINT}"
        )
        model = load_generator_from_run(
            spec.family,
            spec.checkpoint_path,
            spec.metadata,
            device,
        )
        for batch_index in range(0, len(filenames), args.batch_size):
            batch_filenames = filenames[batch_index: batch_index + args.batch_size]
            bf_reference_batch = np.stack(
                [encode_bf_unit(dataset.get_tile(filename)[1]) for filename in batch_filenames],
                axis=0,
            )
            bh_reference_batch = np.stack(
                [height_feet_to_meters(dataset.get_tile(filename)[2]) for filename in batch_filenames],
                axis=0,
            )
            condition_np = encode_condition_batch(
                bf_reference_batch,
                condition_kind=str(spec.metadata["condition_kind"]),
                condition_channels=int(spec.metadata["condition_channels"]),
            )
            predictions_scaled = infer_batch(
                model,
                spec.family,
                torch.from_numpy(condition_np),
                batch_filenames,
                int(args.latent_seed),
                device,
            )
            predictions_m = decode_prediction_batch(predictions_scaled, "height")
            for index, filename in enumerate(batch_filenames):
                all_rows.extend(
                    summarize_tile_by_land_use(
                        filename=filename,
                        manifest_lookup=manifest_lookup,
                        spec=spec,
                        land_use=land_use_cache[filename],
                        reference_bh_m=bh_reference_batch[index],
                        generated_bh_m=predictions_m[index],
                    )
                )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tile_stats = pd.DataFrame(all_rows)
    out_path = (
        dirs["tables"]
        / "height_bias_by_hisdac_land_use_tile_stats_epoch1000_conus_unet_single_latent_oracle_bf.csv"
    )
    tile_stats.to_csv(out_path, index=False)
    print(f"Wrote {len(tile_stats):,} tile-land-use rows: {out_path}")


if __name__ == "__main__":
    main()
