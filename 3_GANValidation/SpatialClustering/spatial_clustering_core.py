from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-spatial-clustering"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "spatial-clustering-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STANDARD_MODE = "standard"
ORACLE_MODE = "oracle_bf"
PIPELINE_MODE = "pipeline_bf"
SPATIAL_STAGE = "moran"
SUPPORTED_MASK_MODES = {"full_tile", "active_union", "reference_active"}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_overall_validation_module(overall_dir: Path) -> Any:
    module_path = overall_dir / "validation_core.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find OverallValidation validation_core.py at {module_path}")
    spec = importlib.util.spec_from_file_location("overall_validation_core", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import OverallValidation validation_core.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["overall_validation_core"] = module
    spec.loader.exec_module(module)
    return module


def ensure_results_dirs(results_root: Path) -> dict[str, Path]:
    dirs = {
        "root": results_root,
        "manifests": results_root / "manifests",
        "metrics": results_root / "metrics",
        "tables": results_root / "tables",
        "figures": results_root / "figures",
        "logs": results_root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_spatial_config(config_path: Path) -> dict[str, Any]:
    raw = read_json(config_path)
    config_dir = config_path.parent
    overall_dir = resolve_path(config_dir, raw["overall_validation_dir"])
    overall_module = load_overall_validation_module(overall_dir)
    overall_config_path = resolve_path(config_dir, raw["overall_validation_config"])
    overall_config = overall_module.load_config(overall_config_path)

    checkpoint = int(raw.get("checkpoint", 1000))
    learning_rates = [float(item) for item in raw.get("learning_rates", overall_config["learning_rates"])]
    training_regimes = [str(item) for item in raw.get("training_regimes", overall_config["model_roots"].keys())]
    families = [str(item) for item in raw.get("families", overall_config["active_families"])]
    datasets = [str(item) for item in raw.get("datasets", overall_config["dataset_specs"].keys())]
    targets = [str(item).upper() for item in raw.get("targets", ["BF", "BH"])]
    bh_modes = [str(item) for item in raw.get("bh_modes", [ORACLE_MODE])]
    mask_mode = str(raw.get("mask_mode", "full_tile"))
    if mask_mode not in SUPPORTED_MASK_MODES:
        raise ValueError(f"Unsupported mask_mode={mask_mode!r}; choose one of {sorted(SUPPORTED_MASK_MODES)}")
    neighbor_mode = str(raw.get("neighbor_mode", "rook"))
    if neighbor_mode != "rook":
        raise ValueError("Only rook contiguity is currently implemented for Moran's I")

    missing_regimes = sorted(set(training_regimes) - set(overall_config["model_roots"]))
    missing_datasets = sorted(set(datasets) - set(overall_config["dataset_specs"]))
    if missing_regimes:
        raise KeyError(f"Training regimes not found in overall config: {missing_regimes}")
    if missing_datasets:
        raise KeyError(f"Datasets not found in overall config: {missing_datasets}")

    overall_config["model_roots"] = {
        key: value for key, value in overall_config["model_roots"].items() if key in training_regimes
    }
    overall_config["dataset_specs"] = {
        key: value for key, value in overall_config["dataset_specs"].items() if key in datasets
    }
    overall_config["active_families"] = families
    overall_config["learning_rates"] = learning_rates
    overall_config["checkpoints"] = [checkpoint]
    overall_config["batch_size"] = int(raw.get("batch_size", overall_config.get("batch_size", 16)))
    overall_config["cache_mode"] = str(raw.get("cache_mode", overall_config.get("cache_mode", "lazy")))
    overall_config["device"] = str(raw.get("device", overall_config.get("device", "auto")))
    overall_config["primary_latent_seed"] = int(raw.get("primary_latent_seed", overall_config["primary_latent_seed"]))

    return {
        "config_path": config_path,
        "config_dir": config_dir,
        "overall_validation_dir": overall_dir,
        "overall_validation_config": overall_config_path,
        "overall_module": overall_module,
        "overall_config": overall_config,
        "results_root": resolve_path(config_dir, raw.get("results_root", "results")),
        "device": overall_config["device"],
        "batch_size": overall_config["batch_size"],
        "cache_mode": overall_config["cache_mode"],
        "checkpoint": checkpoint,
        "training_regimes": training_regimes,
        "families": families,
        "datasets": datasets,
        "learning_rates": learning_rates,
        "targets": targets,
        "bh_modes": bh_modes,
        "primary_latent_seed": overall_config["primary_latent_seed"],
        "mask_mode": mask_mode,
        "neighbor_mode": neighbor_mode,
    }


def snapshot_inputs(config: dict[str, Any], results_dirs: dict[str, Path]) -> None:
    ov = config["overall_module"]
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": getattr(ov.torch, "__version__", None),
        "rasterio": getattr(ov.rasterio, "__version__", None),
        "spatial_config_path": str(config["config_path"]),
        "overall_validation_dir": str(config["overall_validation_dir"]),
        "overall_validation_config": str(config["overall_validation_config"]),
        "results_root": str(results_dirs["root"]),
        "timestamp_utc": now_utc_iso(),
        "checkpoint": config["checkpoint"],
        "learning_rates": config["learning_rates"],
        "targets": config["targets"],
        "bh_modes": config["bh_modes"],
        "mask_mode": config["mask_mode"],
        "neighbor_mode": config["neighbor_mode"],
    }
    write_json(results_dirs["manifests"] / "runtime_environment.json", payload)

    dataset_snapshot: dict[str, Any] = {}
    for name, spec in config["overall_config"]["dataset_specs"].items():
        if spec.manifest_path and spec.manifest_path.exists():
            dataset_snapshot[name] = read_json(spec.manifest_path)
        else:
            dataset_snapshot[name] = {"root": str(spec.root), "manifest_found": False}
    write_json(results_dirs["manifests"] / "dataset_manifests_snapshot.json", dataset_snapshot)


def target_modes(config: dict[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    targets = set(config["targets"])
    if "BF" in targets:
        pairs.append(("BF", STANDARD_MODE))
    if "BH" in targets:
        for mode in config["bh_modes"]:
            pairs.append(("BH", mode))
    return pairs


def make_spatial_row_id(group: Any, target: str, evaluation_mode: str) -> str:
    return "::".join(
        [
            SPATIAL_STAGE,
            target,
            evaluation_mode,
            group.training_regime,
            group.family,
            group.dataset_name,
            f"lr_{group.learning_rate:g}",
            f"ckpt_{group.checkpoint:04d}",
        ]
    )


def spatial_evaluation_matrix_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    ov = config["overall_module"]
    rows: list[dict[str, Any]] = []
    for group in ov.make_group_specs(config["overall_config"]):
        for target, evaluation_mode in target_modes(config):
            rows.append(
                {
                    "row_id": make_spatial_row_id(group, target, evaluation_mode),
                    "stage": SPATIAL_STAGE,
                    "target": target,
                    "evaluation_mode": evaluation_mode,
                    "group_id": group.group_id,
                    "training_regime": group.training_regime,
                    "family": group.family,
                    "dataset_name": group.dataset_name,
                    "learning_rate": group.learning_rate,
                    "checkpoint": group.checkpoint,
                    "mask_mode": config["mask_mode"],
                    "neighbor_mode": config["neighbor_mode"],
                }
            )
    return rows


def append_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def write_spatial_evaluation_matrix(config: dict[str, Any], results_dirs: dict[str, Path]) -> None:
    path = results_dirs["manifests"] / "spatial_evaluation_matrix.csv"
    rows = spatial_evaluation_matrix_rows(config)
    pd.DataFrame(rows).to_csv(path, index=False)


def reset_metric_outputs(results_dirs: dict[str, Path]) -> None:
    for path in [
        results_dirs["metrics"] / "tile_morans_i.csv",
        results_dirs["metrics"] / "overall_morans_i.csv",
        results_dirs["logs"] / "completed_rows.csv",
    ]:
        if path.exists():
            path.unlink()


def reset_summary_outputs(results_dirs: dict[str, Path]) -> None:
    for path in [
        results_dirs["metrics"] / "overall_morans_i.csv",
        results_dirs["tables"] / "table_morans_i_summary_epoch1000.csv",
        results_dirs["figures"] / "morans_i_model_comparison_epoch1000.png",
    ]:
        if path.exists():
            path.unlink()


def parse_filter_values(value: str | None) -> set[str]:
    if value is None or not str(value).strip():
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


def row_matches_cli_filters(row: dict[str, Any], filters: dict[str, str | None]) -> bool:
    for column, raw_values in filters.items():
        values = parse_filter_values(raw_values)
        if not values:
            continue
        actual = str(row[column])
        if column == "target":
            actual = actual.upper()
            values = {value.upper() for value in values}
        if column == "learning_rate":
            actual_float = float(row[column])
            if not any(math.isclose(actual_float, float(value), rel_tol=0.0, abs_tol=1e-12) for value in values):
                return False
            continue
        if actual not in values:
            return False
    return True


def load_completed_row_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["row_id"] for row in reader if row.get("status") == "completed"}


def append_log_rows(results_dirs: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    append_csv_rows(results_dirs["logs"] / "completed_rows.csv", rows)


def morans_i_rook(values: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, int, int]:
    """Compute Moran's I using first-order rook contiguity on a 2D raster tile."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Moran's I expects a 2D tile, received shape={arr.shape}")
    finite = np.isfinite(arr)
    if mask is None:
        valid = finite
    else:
        valid = np.asarray(mask, dtype=bool) & finite
    n_pixels = int(valid.sum())
    if n_pixels < 2:
        return math.nan, n_pixels, 0
    mean_value = float(arr[valid].mean())
    centered = np.zeros_like(arr, dtype=np.float64)
    centered[valid] = arr[valid] - mean_value
    denominator = float(np.sum(centered[valid] * centered[valid]))
    if denominator <= 0.0:
        return math.nan, n_pixels, 0

    horizontal_valid = valid[:, :-1] & valid[:, 1:]
    vertical_valid = valid[:-1, :] & valid[1:, :]
    n_pairs = int(horizontal_valid.sum() + vertical_valid.sum())
    if n_pairs == 0:
        return math.nan, n_pixels, 0
    numerator = float(np.sum((centered[:, :-1] * centered[:, 1:])[horizontal_valid]))
    numerator += float(np.sum((centered[:-1, :] * centered[1:, :])[vertical_valid]))
    return float((n_pixels / n_pairs) * (numerator / denominator)), n_pixels, n_pairs


def make_mask(target: str, reference: np.ndarray, prediction: np.ndarray, config: dict[str, Any]) -> np.ndarray | None:
    mode = config["mask_mode"]
    if mode == "full_tile":
        return None
    if mode == "active_union":
        return config["overall_module"].active_pixel_mask(target, reference, prediction, config["overall_config"])
    if target == "BF":
        threshold = float(config["overall_config"]["bf_active_threshold"])
        return reference >= threshold
    threshold = float(config["overall_config"]["bh_active_threshold_m"])
    return reference > threshold


def tile_moran_row(
    *,
    row_id: str,
    group: Any,
    target: str,
    evaluation_mode: str,
    filename: str,
    reference: np.ndarray,
    prediction: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    mask = make_mask(target, reference, prediction, config)
    reference_i, reference_n, reference_pairs = morans_i_rook(reference, mask)
    prediction_i, prediction_n, prediction_pairs = morans_i_rook(prediction, mask)
    difference = prediction_i - reference_i if np.isfinite(reference_i) and np.isfinite(prediction_i) else math.nan
    return {
        "row_id": row_id,
        "group_id": group.group_id,
        "target": target.lower(),
        "evaluation_mode": evaluation_mode,
        "training_regime": group.training_regime,
        "family": group.family,
        "dataset_name": group.dataset_name,
        "learning_rate": group.learning_rate,
        "checkpoint": group.checkpoint,
        "filename": filename,
        "mask_mode": config["mask_mode"],
        "neighbor_mode": config["neighbor_mode"],
        "reference_moran_i": reference_i,
        "generated_moran_i": prediction_i,
        "moran_difference": difference,
        "moran_abs_error": abs(difference) if np.isfinite(difference) else math.nan,
        "reference_mean": float(np.nanmean(reference)),
        "generated_mean": float(np.nanmean(prediction)),
        "reference_sd": float(np.nanstd(reference)),
        "generated_sd": float(np.nanstd(prediction)),
        "reference_n_pixels": reference_n,
        "generated_n_pixels": prediction_n,
        "reference_neighbor_pairs": reference_pairs,
        "generated_neighbor_pairs": prediction_pairs,
    }


def evaluate_spatial_group(
    *,
    group: Any,
    dataset: Any,
    filenames: list[str],
    requested_modes: list[tuple[str, str, str]],
    config: dict[str, Any],
    device: Any,
) -> list[dict[str, Any]]:
    ov = config["overall_module"]
    requested_set = {(target, mode) for target, mode, _row_id in requested_modes}
    need_bf_prediction = ("BF", STANDARD_MODE) in requested_set or ("BH", PIPELINE_MODE) in requested_set
    need_bh_prediction = any(target == "BH" for target, _mode in requested_set)

    bf_model = (
        ov.load_generator_from_run(group.family, group.bf_checkpoint_path, group.bf_metadata, device)
        if need_bf_prediction
        else None
    )
    bh_model = (
        ov.load_generator_from_run(group.family, group.bh_checkpoint_path, group.bh_metadata, device)
        if need_bh_prediction
        else None
    )
    batch_size = int(config["batch_size"])
    primary_seed = int(config["primary_latent_seed"])
    rows: list[dict[str, Any]] = []

    for batch_filenames in ov.batched(filenames, batch_size):
        tiles = [dataset.get_tile(filename) for filename in batch_filenames]
        central_batch = np.stack([tile[0] for tile in tiles], axis=0)
        bf_reference_batch = np.stack([ov.encode_bf_unit(tile[1]) for tile in tiles], axis=0)
        bh_reference_batch = ov.height_feet_to_meters(np.stack([tile[2] for tile in tiles], axis=0))

        bf_predictions: np.ndarray | None = None
        if need_bf_prediction:
            bf_condition_np = ov.encode_condition_batch(
                central_batch,
                condition_kind=str(group.bf_metadata["condition_kind"]),
                condition_channels=int(group.bf_metadata["condition_channels"]),
            )
            bf_predictions_scaled = ov.infer_batch(
                bf_model,
                group.family,
                ov.torch.from_numpy(bf_condition_np),
                batch_filenames,
                primary_seed,
                device,
            )
            bf_predictions = ov.decode_prediction_batch(bf_predictions_scaled, "bf")

        bh_predictions_by_mode: dict[str, np.ndarray] = {}
        if need_bh_prediction:
            if ("BH", ORACLE_MODE) in requested_set:
                bh_condition_np = ov.encode_condition_batch(
                    bf_reference_batch,
                    condition_kind=str(group.bh_metadata["condition_kind"]),
                    condition_channels=int(group.bh_metadata["condition_channels"]),
                )
                bh_predictions_scaled = ov.infer_batch(
                    bh_model,
                    group.family,
                    ov.torch.from_numpy(bh_condition_np),
                    batch_filenames,
                    primary_seed,
                    device,
                )
                bh_predictions_by_mode[ORACLE_MODE] = ov.decode_prediction_batch(bh_predictions_scaled, "height")
            if ("BH", PIPELINE_MODE) in requested_set:
                if bf_predictions is None:
                    raise RuntimeError("Pipeline BH requires BF predictions, but BF inference was not run")
                bh_condition_np = ov.encode_condition_batch(
                    bf_predictions,
                    condition_kind=str(group.bh_metadata["condition_kind"]),
                    condition_channels=int(group.bh_metadata["condition_channels"]),
                )
                bh_predictions_scaled = ov.infer_batch(
                    bh_model,
                    group.family,
                    ov.torch.from_numpy(bh_condition_np),
                    batch_filenames,
                    primary_seed,
                    device,
                )
                bh_predictions_by_mode[PIPELINE_MODE] = ov.decode_prediction_batch(bh_predictions_scaled, "height")

        row_id_lookup = {(target, mode): row_id for target, mode, row_id in requested_modes}
        for index, filename in enumerate(batch_filenames):
            if ("BF", STANDARD_MODE) in requested_set:
                if bf_predictions is None:
                    raise RuntimeError("BF row requested, but BF predictions are unavailable")
                rows.append(
                    tile_moran_row(
                        row_id=row_id_lookup[("BF", STANDARD_MODE)],
                        group=group,
                        target="BF",
                        evaluation_mode=STANDARD_MODE,
                        filename=filename,
                        reference=bf_reference_batch[index],
                        prediction=bf_predictions[index],
                        config=config,
                    )
                )
            for mode, bh_predictions in bh_predictions_by_mode.items():
                rows.append(
                    tile_moran_row(
                        row_id=row_id_lookup[("BH", mode)],
                        group=group,
                        target="BH",
                        evaluation_mode=mode,
                        filename=filename,
                        reference=bh_reference_batch[index],
                        prediction=bh_predictions[index],
                        config=config,
                    )
                )

    del bf_model
    del bh_model
    if getattr(ov.torch, "cuda", None) is not None and ov.torch.cuda.is_available():
        ov.torch.cuda.empty_cache()
    return rows


def run_metrics_stage(
    *,
    config: dict[str, Any],
    results_dirs: dict[str, Path],
    quick_tiles: int | None,
    resume: bool,
    cli_filters: dict[str, str | None],
) -> None:
    ov = config["overall_module"]
    ov.set_torch_determinism()
    device = ov.choose_device(str(config["device"]))
    groups = ov.make_group_specs(config["overall_config"])
    datasets = {
        name: ov.DatasetBundle(spec, cache_mode=str(config["cache_mode"]))
        for name, spec in config["overall_config"]["dataset_specs"].items()
    }
    filenames_by_dataset = {
        name: dataset.limited_filenames(quick_tiles)
        for name, dataset in datasets.items()
    }

    matrix_rows = spatial_evaluation_matrix_rows(config)
    row_lookup = {row["row_id"]: row for row in matrix_rows}
    completed = load_completed_row_ids(results_dirs["logs"] / "completed_rows.csv") if resume else set()
    group_lookup = {group.group_id: group for group in groups}
    eligible_rows = [
        row for row in matrix_rows
        if row["row_id"] not in completed and row_matches_cli_filters(row, cli_filters)
    ]
    eligible_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in eligible_rows:
        eligible_by_group.setdefault(str(row["group_id"]), []).append(row)

    total_rows = len(eligible_rows)
    completed_this_run = 0
    for group_id, row_specs in eligible_by_group.items():
        group = group_lookup[group_id]
        requested_modes = [
            (str(row["target"]).upper(), str(row["evaluation_mode"]), str(row["row_id"]))
            for row in row_specs
        ]
        tile_rows = evaluate_spatial_group(
            group=group,
            dataset=datasets[group.dataset_name],
            filenames=filenames_by_dataset[group.dataset_name],
            requested_modes=requested_modes,
            config=config,
            device=device,
        )
        append_csv_rows(results_dirs["metrics"] / "tile_morans_i.csv", tile_rows)
        append_log_rows(
            results_dirs,
            [
                {
                    "timestamp_utc": now_utc_iso(),
                    "stage": SPATIAL_STAGE,
                    "row_id": row_id,
                    "group_id": group.group_id,
                    "status": "completed",
                }
                for _target, _mode, row_id in requested_modes
            ],
        )
        completed_this_run += len(requested_modes)
        last_label = row_lookup[requested_modes[-1][2]]["row_id"]
        print(f"{now_utc_iso()} moran {completed_this_run}/{total_rows} completed; last={last_label}", flush=True)


def safe_regression(reference: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    valid = np.isfinite(reference) & np.isfinite(generated)
    reference = reference[valid]
    generated = generated[valid]
    if reference.size < 2 or np.allclose(reference, reference[0]):
        return {"slope": math.nan, "intercept": math.nan, "r2": math.nan, "pearson_r": math.nan}
    slope, intercept = np.polyfit(reference, generated, deg=1)
    corr = float(np.corrcoef(reference, generated)[0, 1])
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(corr * corr),
        "pearson_r": corr,
    }


def summarize_moran_tiles(tile_df: pd.DataFrame) -> pd.DataFrame:
    if tile_df.empty:
        return pd.DataFrame()
    frame = tile_df.copy()
    frame["family"] = frame["family"].astype(str)
    frame = frame.drop_duplicates(["row_id", "filename"], keep="last")
    group_columns = [
        "row_id",
        "group_id",
        "target",
        "evaluation_mode",
        "training_regime",
        "family",
        "dataset_name",
        "learning_rate",
        "checkpoint",
        "mask_mode",
        "neighbor_mode",
    ]
    rows: list[dict[str, Any]] = []
    for keys, subset in frame.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,), strict=False))
        valid = subset[np.isfinite(subset["reference_moran_i"]) & np.isfinite(subset["generated_moran_i"])]
        diff = valid["generated_moran_i"].to_numpy(dtype=float) - valid["reference_moran_i"].to_numpy(dtype=float)
        regression = safe_regression(
            valid["reference_moran_i"].to_numpy(dtype=float),
            valid["generated_moran_i"].to_numpy(dtype=float),
        )
        row.update(
            {
                "n_tiles": int(subset.shape[0]),
                "n_valid_tiles": int(valid.shape[0]),
                "reference_moran_mean": float(valid["reference_moran_i"].mean()) if not valid.empty else math.nan,
                "generated_moran_mean": float(valid["generated_moran_i"].mean()) if not valid.empty else math.nan,
                "reference_moran_median": float(valid["reference_moran_i"].median()) if not valid.empty else math.nan,
                "generated_moran_median": float(valid["generated_moran_i"].median()) if not valid.empty else math.nan,
                "moran_mae": float(np.mean(np.abs(diff))) if diff.size else math.nan,
                "moran_rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else math.nan,
                "moran_mbe": float(np.mean(diff)) if diff.size else math.nan,
                "moran_slope": regression["slope"],
                "moran_intercept": regression["intercept"],
                "moran_r2": regression["r2"],
                "moran_pearson_r": regression["pearson_r"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["training_regime", "family", "dataset_name", "learning_rate", "target", "evaluation_mode"]
    )


def model_class_plot_constants() -> tuple[
    list[tuple[str, str, str]],
    list[str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[float, str],
    dict[str, float],
]:
    panel_specs = [
        ("LALegacy", "BF", "Building Footprint | Los Angeles Model"),
        ("MSASample", "BF", "Building Footprint | CONUS Model"),
        ("LALegacy", "BH", "Building Heights (m) | Los Angeles Model"),
        ("MSASample", "BH", "Building Heights (m) | CONUS Model"),
    ]
    family_order = ["1", "2A", "3"]
    family_labels = {"1": "U-Net", "2A": "Single-latent\ncGAN", "3": "Multi-latent\ncGAN"}
    dataset_colors = {"CONUSStratifiedTest": "#315f72", "SanDiegoTestNoOverlap": "#c87533"}
    dataset_labels = {"CONUSStratifiedTest": "CONUS Test Dataset", "SanDiegoTestNoOverlap": "San Diego Test Dataset"}
    lr_markers = {0.0001: "o", 0.0002: "s", 0.0005: "^", 0.001: "D"}
    offsets = {"CONUSStratifiedTest": -0.16, "SanDiegoTestNoOverlap": 0.16}
    return panel_specs, family_order, family_labels, dataset_colors, dataset_labels, lr_markers, offsets


def save_moran_model_comparison_figure(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    (
        panel_specs,
        family_order,
        family_labels,
        dataset_colors,
        dataset_labels,
        lr_markers,
        offsets,
    ) = model_class_plot_constants()
    metric_specs = [
        ("moran_mae", "Moran's I Mean Absolute Error"),
        ("moran_mbe", "Moran's I Mean Bias"),
    ]
    y_limits: dict[tuple[str, str], tuple[float, float]] = {}
    for metric, _ylabel in metric_specs:
        for target_label in ("BF", "BH"):
            values = summary_df[summary_df["target_label"] == target_label][metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            lower = float(np.min(values))
            upper = float(np.max(values))
            if metric == "moran_mbe":
                lower = min(lower, 0.0)
                upper = max(upper, 0.0)
            span = upper - lower
            pad = 0.08 * span if span > 0.0 else max(abs(upper), 1.0) * 0.08
            y_limits[(target_label, metric)] = (lower - pad, upper + pad)

    fig, axes = plt.subplots(2, 4, figsize=(16.5, 6.2), constrained_layout=False)
    for row_index, (metric, ylabel) in enumerate(metric_specs):
        for col_index, (training_regime, target_label, title) in enumerate(panel_specs):
            ax = axes[row_index, col_index]
            panel = summary_df[
                (summary_df["training_regime"] == training_regime)
                & (summary_df["target_label"] == target_label)
            ]
            for dataset_name, dataset_panel in panel.groupby("dataset_name"):
                color = dataset_colors.get(dataset_name, "#666666")
                offset = offsets.get(dataset_name, 0.0)
                for learning_rate, lr_panel in dataset_panel.groupby("learning_rate"):
                    lr_panel = lr_panel.set_index("family").reindex(family_order)
                    x_values = np.arange(len(family_order), dtype=float) + offset
                    ax.scatter(
                        x_values,
                        lr_panel[metric].to_numpy(dtype=float),
                        color=color,
                        marker=lr_markers.get(round(float(learning_rate), 4), "o"),
                        s=42,
                        edgecolor="black",
                        linewidth=0.25,
                        alpha=0.88,
                    )
                medians = dataset_panel.groupby("family")[metric].median().reindex(family_order)
                ax.scatter(
                    np.arange(len(family_order), dtype=float) + offset,
                    medians.to_numpy(dtype=float),
                    color=color,
                    marker="_",
                    s=430,
                    linewidth=2.0,
                    zorder=4,
                )
            if metric == "moran_mbe":
                ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            if (target_label, metric) in y_limits:
                ax.set_ylim(*y_limits[(target_label, metric)])
            ax.set_xticks(np.arange(len(family_order)))
            if row_index == len(metric_specs) - 1:
                ax.set_xticklabels([family_labels[item] for item in family_order], fontsize=11)
            else:
                ax.set_xticklabels([])
            if col_index == 0:
                ax.set_ylabel(ylabel, fontsize=13, labelpad=28)
                ax.yaxis.set_label_coords(-0.19, 0.5)
            ax.set_title(title if row_index == 0 else "", fontsize=12)
            ax.grid(axis="y", alpha=0.25)

    dataset_handles = [
        plt.Line2D([0], [0], color=color, marker="o", linestyle="", label=dataset_labels.get(dataset, dataset))
        for dataset, color in dataset_colors.items()
    ]
    lr_handles = [
        plt.Line2D([0], [0], color="black", marker=marker, linestyle="", label=f"LR {learning_rate:g}")
        for learning_rate, marker in lr_markers.items()
    ]
    median_handle = plt.Line2D([0], [0], color="black", marker="_", linestyle="", markersize=18, label="Median across LRs")
    fig.subplots_adjust(left=0.095, right=0.995, top=0.88, bottom=0.16, hspace=0.20, wspace=0.20)
    fig.legend(
        handles=[*dataset_handles, *lr_handles, median_handle],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_moran_summary_table(overall_df: pd.DataFrame) -> pd.DataFrame:
    if overall_df.empty:
        return overall_df
    frame = overall_df[
        (overall_df["checkpoint"] == 1000)
        & (
            ((overall_df["target"] == "bf") & (overall_df["evaluation_mode"] == STANDARD_MODE))
            | ((overall_df["target"] == "bh") & (overall_df["evaluation_mode"] == ORACLE_MODE))
        )
    ].copy()
    frame["target_label"] = frame["target"].map({"bf": "BF", "bh": "BH"})
    frame["training_regime_label"] = frame["training_regime"].map(
        {"LALegacy": "Los Angeles Model", "MSASample": "CONUS Model"}
    )
    frame["dataset_label"] = frame["dataset_name"].map(
        {"CONUSStratifiedTest": "CONUS Test Dataset", "SanDiegoTestNoOverlap": "San Diego Test Dataset"}
    )
    frame["family_label"] = frame["family"].map(
        {"1": "U-Net", "2A": "Single-latent cGAN", "3": "Multi-latent cGAN"}
    )
    columns = [
        "training_regime",
        "training_regime_label",
        "target",
        "target_label",
        "dataset_name",
        "dataset_label",
        "family",
        "family_label",
        "learning_rate",
        "checkpoint",
        "evaluation_mode",
        "mask_mode",
        "n_tiles",
        "n_valid_tiles",
        "reference_moran_mean",
        "generated_moran_mean",
        "moran_mae",
        "moran_rmse",
        "moran_mbe",
        "moran_slope",
        "moran_r2",
    ]
    return frame[columns].sort_values(
        ["training_regime", "target", "dataset_name", "family", "learning_rate"]
    )


def build_summary_outputs(config: dict[str, Any], results_dirs: dict[str, Path]) -> None:
    tile_path = results_dirs["metrics"] / "tile_morans_i.csv"
    if not tile_path.exists():
        raise FileNotFoundError(f"Missing tile Moran's I metrics: {tile_path}")
    tile_df = pd.read_csv(tile_path, dtype={"family": str}, low_memory=False)
    if not tile_df.empty:
        tile_df = tile_df.drop_duplicates(["row_id", "filename"], keep="last")
        tile_df.to_csv(tile_path, index=False)
    overall_df = summarize_moran_tiles(tile_df)
    overall_path = results_dirs["metrics"] / "overall_morans_i.csv"
    overall_df.to_csv(overall_path, index=False)
    table = build_moran_summary_table(overall_df)
    table_path = results_dirs["tables"] / "table_morans_i_summary_epoch1000.csv"
    table.to_csv(table_path, index=False)
    if not table.empty:
        save_moran_model_comparison_figure(
            table,
            results_dirs["figures"] / "morans_i_model_comparison_epoch1000.png",
        )
