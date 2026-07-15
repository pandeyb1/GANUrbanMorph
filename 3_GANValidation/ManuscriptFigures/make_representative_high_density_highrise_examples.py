from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-representative-examples"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "representative-examples-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds


THIS_DIR = Path(__file__).resolve().parent
GAN_VALIDATION_DIR = THIS_DIR.parent
OVERALL_DIR = GAN_VALIDATION_DIR / "OverallValidation"
DEFAULT_CONFIG = OVERALL_DIR / "validation_config.json"
DEFAULT_OUT = THIS_DIR / "figures" / "representative_high_density_highrise_epoch1000.png"
DEFAULT_TILE_TABLE = THIS_DIR / "tables" / "representative_high_density_highrise_tiles.csv"

DATASET_LABELS = {
    "CONUSStratifiedTest": "CONUS Test Dataset",
    "SanDiegoTestNoOverlap": "San Diego Test Dataset",
}
FAMILY_LABELS = {"1": "U-Net", "2A": "Single-latent cGAN"}
SELECTION_LABELS = {
    "high_density": "High footprint density",
    "high_rise": "High-rise setting",
}
MSA_TILE_RE = re.compile(r"^MSA_(?P<msa_id>\d+)_X(?P<x>-?\d+)_Y(?P<y>-?\d+)")


def load_overall_validation_module(overall_dir: Path) -> Any:
    module_path = overall_dir / "validation_core.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find validation_core.py at {module_path}")
    spec = importlib.util.spec_from_file_location("overall_validation_core_for_examples", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["overall_validation_core_for_examples"] = module
    spec.loader.exec_module(module)
    return module


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate representative high-density/high-rise test-tile examples for manuscript response figures."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to OverallValidation validation_config.json")
    parser.add_argument("--overall-validation-dir", default=str(OVERALL_DIR), help="Path to OverallValidation code folder")
    parser.add_argument("--training-regime", default="MSASample", choices=["LALegacy", "MSASample"])
    parser.add_argument("--families", default="1,2A", help="Comma-separated model families to plot, default: 1,2A")
    parser.add_argument("--datasets", default="CONUSStratifiedTest")
    parser.add_argument("--examples-per-type", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--checkpoint", type=int, default=1000)
    parser.add_argument("--latent-seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-mode", default="lazy", choices=["lazy", "all"])
    parser.add_argument("--min-active-fraction", type=float, default=0.01)
    parser.add_argument(
        "--unique-msas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer at most one selected tile from each MSA to avoid near-duplicate neighborhood examples.",
    )
    parser.add_argument(
        "--min-grid-distance",
        type=float,
        default=512.0,
        help="Minimum X/Y grid distance between selected examples from the same dataset when coordinates are parseable.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--tile-table", default=str(DEFAULT_TILE_TABLE))
    parser.add_argument("--copy-to-overall-results", action="store_true", default=True)
    parser.add_argument(
        "--context-source",
        default="lulc",
        choices=["lulc", "osm", "satellite", "none"],
        help="Leftmost context column source. Use osm/satellite only with a local directory of licensed images.",
    )
    parser.add_argument(
        "--osm-dir",
        default=None,
        help="Optional directory with pre-rendered OpenStreetMap context images named by tile stem or filename.",
    )
    parser.add_argument(
        "--osm-label",
        default="OpenStreetMap",
        help="Column label used when --context-source osm is selected.",
    )
    parser.add_argument(
        "--satellite-dir",
        default=None,
        help="Optional directory with licensed satellite/context images named by tile stem or filename.",
    )
    parser.add_argument(
        "--satellite-label",
        default="Satellite image",
        help="Column label used when --context-source satellite is selected.",
    )
    parser.add_argument(
        "--coordinate-axes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show longitude/latitude ticks on the reference BF and BH panels.",
    )
    parser.add_argument(
        "--rescale-generated-bh-to-reference-max",
        action="store_true",
        help="Min-max rescale generated BH panels to [0, 1] and multiply by the reference BH maximum for that tile.",
    )
    parser.add_argument(
        "--context-panel-scale",
        type=float,
        default=0.90,
        help="Scale factor for the leftmost context panel axes relative to the raster tile axes; 0.90 makes OSM panels 90%% as large.",
    )
    return parser.parse_args()


def configure_overall(args: argparse.Namespace, ov: Any) -> dict[str, Any]:
    config = ov.load_config(Path(args.config).resolve())
    families = parse_csv_list(args.families)
    datasets = parse_csv_list(args.datasets)
    config["active_families"] = families
    config["learning_rates"] = [float(args.learning_rate)]
    config["checkpoints"] = [int(args.checkpoint)]
    config["model_roots"] = {
        key: value
        for key, value in config["model_roots"].items()
        if key == args.training_regime
    }
    config["dataset_specs"] = {
        key: value
        for key, value in config["dataset_specs"].items()
        if key in datasets
    }
    config["device"] = args.device
    config["cache_mode"] = args.cache_mode
    config["primary_latent_seed"] = int(args.latent_seed)
    return config


def parse_tile_identity(filename: str) -> tuple[str | None, float | None, float | None]:
    match = MSA_TILE_RE.match(Path(filename).stem)
    if match is None:
        return None, None, None
    return (
        str(match.group("msa_id")),
        float(match.group("x")),
        float(match.group("y")),
    )


def tile_scores(dataset: Any, dataset_name: str, ov: Any, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bf_threshold = float(config["bf_active_threshold"])
    bh_threshold = float(config["bh_active_threshold_m"])
    for filename in dataset.filenames:
        msa_id, tile_x, tile_y = parse_tile_identity(filename)
        _central, bf_raw, bh_raw = dataset.get_tile(filename)
        bf = ov.encode_bf_unit(bf_raw)
        bh = ov.height_feet_to_meters(bh_raw)
        active_bf = bf >= bf_threshold
        active_bh = bh > bh_threshold
        active_union = active_bf | active_bh
        active_heights = bh[active_union]
        if active_heights.size:
            bh_active_mean = float(np.nanmean(active_heights))
            bh_p95 = float(np.nanpercentile(active_heights, 95))
            bh_max = float(np.nanmax(active_heights))
        else:
            bh_active_mean = np.nan
            bh_p95 = np.nan
            bh_max = np.nan
        rows.append(
            {
                "dataset_name": dataset_name,
                "filename": filename,
                "msa_id": msa_id,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "bf_mean": float(np.nanmean(bf)),
                "bf_active_fraction": float(np.mean(active_bf)),
                "bh_mean_m": float(np.nanmean(bh)),
                "bh_active_mean_m": bh_active_mean,
                "bh_p95_m": bh_p95,
                "bh_max_m": bh_max,
                "active_union_fraction": float(np.mean(active_union)),
            }
        )
    return pd.DataFrame(rows)


def candidate_is_diverse(
    candidate: pd.Series,
    selected_rows: list[pd.Series],
    *,
    dataset_name: str,
    unique_msas: bool,
    min_grid_distance: float,
) -> bool:
    candidate_msa = candidate.get("msa_id")
    candidate_x = candidate.get("tile_x")
    candidate_y = candidate.get("tile_y")
    for selected in selected_rows:
        if str(selected.get("dataset_name")) != dataset_name:
            continue
        selected_msa = selected.get("msa_id")
        if unique_msas and pd.notna(candidate_msa) and pd.notna(selected_msa):
            if str(candidate_msa) == str(selected_msa):
                return False
        selected_x = selected.get("tile_x")
        selected_y = selected.get("tile_y")
        if (
            pd.notna(candidate_x)
            and pd.notna(candidate_y)
            and pd.notna(selected_x)
            and pd.notna(selected_y)
        ):
            distance = float(np.hypot(float(candidate_x) - float(selected_x), float(candidate_y) - float(selected_y)))
            if distance < min_grid_distance:
                return False
    return True


def append_top_diverse_candidates(
    *,
    selected_rows: list[pd.Series],
    candidates: pd.DataFrame,
    dataset_name: str,
    selection_type: str,
    examples_per_type: int,
    used: set[tuple[str, str]],
    unique_msas: bool,
    min_grid_distance: float,
) -> None:
    rank = 1
    deferred_rows: list[pd.Series] = []
    for _index, candidate in candidates.iterrows():
        if rank > examples_per_type:
            break
        key = (dataset_name, str(candidate["filename"]))
        if key in used:
            continue
        if not candidate_is_diverse(
            candidate,
            selected_rows,
            dataset_name=dataset_name,
            unique_msas=unique_msas,
            min_grid_distance=min_grid_distance,
        ):
            deferred_rows.append(candidate)
            continue
        row = candidate.copy()
        row["selection_type"] = selection_type
        row["selection_rank"] = rank
        selected_rows.append(row)
        used.add(key)
        rank += 1

    # If the dataset cannot satisfy the diversity constraints, fall back to top-ranked rows
    # rather than silently returning fewer examples.
    for candidate in deferred_rows:
        if rank > examples_per_type:
            break
        key = (dataset_name, str(candidate["filename"]))
        if key in used:
            continue
        row = candidate.copy()
        row["selection_type"] = selection_type
        row["selection_rank"] = rank
        selected_rows.append(row)
        used.add(key)
        rank += 1


def select_representative_tiles(
    scores: pd.DataFrame,
    *,
    min_active_fraction: float,
    examples_per_type: int,
    unique_msas: bool = True,
    min_grid_distance: float = 512.0,
) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    used: set[tuple[str, str]] = set()
    for dataset_name, dataset_scores in scores.groupby("dataset_name"):
        high_density_candidates = dataset_scores.sort_values(
            ["bf_mean", "active_union_fraction", "bh_p95_m"],
            ascending=[False, False, False],
        )
        append_top_diverse_candidates(
            selected_rows=selected_rows,
            candidates=high_density_candidates,
            dataset_name=dataset_name,
            selection_type="high_density",
            examples_per_type=examples_per_type,
            used=used,
            unique_msas=unique_msas,
            min_grid_distance=min_grid_distance,
        )

        highrise_candidates = dataset_scores[
            dataset_scores["active_union_fraction"] >= min_active_fraction
        ].sort_values(
            ["bh_p95_m", "bh_active_mean_m", "bf_mean"],
            ascending=[False, False, False],
        )
        append_top_diverse_candidates(
            selected_rows=selected_rows,
            candidates=highrise_candidates,
            dataset_name=dataset_name,
            selection_type="high_rise",
            examples_per_type=examples_per_type,
            used=used,
            unique_msas=unique_msas,
            min_grid_distance=min_grid_distance,
        )
    selected = pd.DataFrame(selected_rows)
    selected["selection_order"] = selected["selection_type"].map({"high_density": 0, "high_rise": 1})
    return selected[
        [
            "dataset_name",
            "selection_type",
            "selection_rank",
            "filename",
            "msa_id",
            "tile_x",
            "tile_y",
            "bf_mean",
            "bf_active_fraction",
            "bh_mean_m",
            "bh_active_mean_m",
            "bh_p95_m",
            "bh_max_m",
            "active_union_fraction",
            "selection_order",
        ]
    ].sort_values(["dataset_name", "selection_rank", "selection_order"]).drop(columns=["selection_order"])


def group_lookup(config: dict[str, Any], ov: Any) -> dict[tuple[str, str, str], Any]:
    lookup: dict[tuple[str, str, str], Any] = {}
    for group in ov.make_group_specs(config):
        key = (group.training_regime, group.family, group.dataset_name)
        lookup[key] = group
    return lookup


def predict_tile(
    *,
    ov: Any,
    group: Any,
    dataset: Any,
    filename: str,
    family: str,
    device: Any,
    latent_seed: int,
) -> dict[str, np.ndarray]:
    central, bf_raw, bh_raw = dataset.get_tile(filename)
    bf_ref = ov.encode_bf_unit(bf_raw)
    bh_ref = ov.height_feet_to_meters(bh_raw)

    bf_model = ov.load_generator_from_run(family, group.bf_checkpoint_path, group.bf_metadata, device)
    bh_model = ov.load_generator_from_run(family, group.bh_checkpoint_path, group.bh_metadata, device)

    central_batch = np.stack([central], axis=0)
    bf_ref_batch = np.stack([bf_ref], axis=0)
    bf_condition_np = ov.encode_condition_batch(
        central_batch,
        condition_kind=str(group.bf_metadata["condition_kind"]),
        condition_channels=int(group.bf_metadata["condition_channels"]),
    )
    bf_pred_scaled = ov.infer_batch(
        bf_model,
        family,
        ov.torch.from_numpy(bf_condition_np),
        [filename],
        latent_seed,
        device,
    )
    bf_pred = ov.decode_prediction_batch(bf_pred_scaled, "bf")[0]

    # BH examples use reference-BF conditioning to match the main manuscript-facing BH evaluation.
    bh_condition_np = ov.encode_condition_batch(
        bf_ref_batch,
        condition_kind=str(group.bh_metadata["condition_kind"]),
        condition_channels=int(group.bh_metadata["condition_channels"]),
    )
    bh_pred_scaled = ov.infer_batch(
        bh_model,
        family,
        ov.torch.from_numpy(bh_condition_np),
        [filename],
        latent_seed,
        device,
    )
    bh_pred = ov.decode_prediction_batch(bh_pred_scaled, "height")[0]

    del bf_model
    del bh_model
    if getattr(ov.torch, "cuda", None) is not None and ov.torch.cuda.is_available():
        ov.torch.cuda.empty_cache()
    return {"bf": bf_pred, "bh": bh_pred}


def geographic_extent(dataset: Any, filename: str) -> tuple[float, float, float, float] | None:
    path = dataset.central_cache.folder / filename
    with rasterio.open(path) as src:
        if src.crs is None:
            return None
        left, bottom, right, top = transform_bounds(
            src.crs,
            "EPSG:4326",
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top,
            densify_pts=21,
        )
    return float(left), float(right), float(bottom), float(top)


def find_context_image(context_dir: Path, filename: str) -> Path | None:
    candidates = []
    path = Path(filename)
    stems = [path.stem, path.name]
    extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    for stem in stems:
        for extension in extensions:
            candidates.append(context_dir / f"{stem}{extension}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_local_context_image(context_dir: str, filename: str, *, context_name: str) -> np.ndarray:
    image_path = find_context_image(Path(context_dir).expanduser().resolve(), filename)
    if image_path is None:
        raise FileNotFoundError(f"No {context_name} image found for {filename} in {context_dir}")
    if image_path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(image_path) as src:
            image = src.read()
            if image.shape[0] >= 3:
                image = np.moveaxis(image[:3], 0, -1)
            else:
                image = image[0]
            image = image.astype(np.float32)
            finite = np.isfinite(image)
            if finite.any():
                p2, p98 = np.nanpercentile(image[finite], [2, 98])
                image = np.clip((image - p2) / max(p98 - p2, 1e-6), 0, 1)
    else:
        image = mpimg.imread(image_path)
    return image


def load_context_panel(
    *,
    dataset: Any,
    filename: str,
    central: np.ndarray,
    context_source: str,
    osm_dir: str | None,
    osm_label: str,
    satellite_dir: str | None,
    satellite_label: str,
) -> tuple[np.ndarray | None, str, str, float | None, float | None, str | None]:
    if context_source == "none":
        return None, "", "gray", None, None, None
    if context_source == "osm":
        if not osm_dir:
            raise ValueError("--context-source osm requires --osm-dir with pre-rendered OpenStreetMap images")
        image = load_local_context_image(osm_dir, filename, context_name="OpenStreetMap context")
        return image, osm_label, "gray", None, None, "© OpenStreetMap contributors"
    if context_source == "satellite":
        if not satellite_dir:
            raise ValueError("--context-source satellite requires --satellite-dir")
        image = load_local_context_image(satellite_dir, filename, context_name="satellite/context")
        return image, satellite_label, "gray", None, None, None
    lulc = central.astype(np.float32)
    lulc[(lulc < 0) | (lulc > 95)] = np.nan
    return lulc, "LULC input", "tab20", 0.0, 95.0, None


def collect_predictions(
    selected_tiles: pd.DataFrame,
    datasets: dict[str, Any],
    groups: dict[tuple[str, str, str], Any],
    ov: Any,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[str]]:
    device = ov.choose_device(str(args.device))
    family_order = parse_csv_list(args.families)
    rows: list[dict[str, Any]] = []
    for _index, selected in selected_tiles.iterrows():
        dataset_name = str(selected["dataset_name"])
        filename = str(selected["filename"])
        central, bf_raw, bh_raw = datasets[dataset_name].get_tile(filename)
        context_array, context_title, context_cmap, context_vmin, context_vmax, context_attribution = load_context_panel(
            dataset=datasets[dataset_name],
            filename=filename,
            central=central,
            context_source=str(args.context_source),
            osm_dir=args.osm_dir,
            osm_label=str(args.osm_label),
            satellite_dir=args.satellite_dir,
            satellite_label=str(args.satellite_label),
        )
        row: dict[str, Any] = {
            "dataset_name": dataset_name,
            "selection_type": str(selected["selection_type"]),
            "filename": filename,
            "extent": geographic_extent(datasets[dataset_name], filename),
            "context": context_array,
            "context_title": context_title,
            "context_cmap": context_cmap,
            "context_vmin": context_vmin,
            "context_vmax": context_vmax,
            "context_attribution": context_attribution,
            "bf_reference": ov.encode_bf_unit(bf_raw),
            "bh_reference": ov.height_feet_to_meters(bh_raw),
            "predictions": {},
            "scores": selected.to_dict(),
        }
        for family in family_order:
            group = groups[(args.training_regime, family, dataset_name)]
            row["predictions"][family] = predict_tile(
                ov=ov,
                group=group,
                dataset=datasets[dataset_name],
                filename=filename,
                family=family,
                device=device,
                latent_seed=int(config["primary_latent_seed"]),
            )
        rows.append(row)
    return rows, family_order


def robust_height_vmax(rows: list[dict[str, Any]], family_order: list[str]) -> float:
    arrays: list[np.ndarray] = []
    for row in rows:
        arrays.append(row["bh_reference"])
        for family in family_order:
            arrays.append(row["predictions"][family]["bh"])
    values = np.concatenate([arr[np.isfinite(arr)].ravel() for arr in arrays if np.isfinite(arr).any()])
    if values.size == 0:
        return 10.0
    return float(max(10.0, np.nanpercentile(values, 99.5)))


def row_label(row: dict[str, Any]) -> str:
    scores = row["scores"]
    selection_label = SELECTION_LABELS.get(row["selection_type"], row["selection_type"])
    return (
        f"{selection_label}\n"
        f"BF mean={scores['bf_mean']:.3f}; "
        f"BH p95={scores['bh_p95_m']:.1f} m"
    )


def degree_formatter(value: float, _position: int) -> str:
    return f"{value:.2f}°"


def apply_coordinate_ticks(
    ax: Any,
    extent: tuple[float, float, float, float] | None,
    array_shape: tuple[int, ...],
    *,
    show: bool,
) -> None:
    height = int(array_shape[0])
    width = int(array_shape[1])
    ax.set_box_aspect(1)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    if extent is None or not show:
        ax.set_xticks([])
        ax.set_yticks([])
        return
    left, right, bottom, top = extent
    x_positions = np.array([0, (width - 1) / 2, width - 1], dtype=float)
    y_positions = np.array([0, (height - 1) / 2, height - 1], dtype=float)
    x_labels = [degree_formatter(value, 0) for value in np.linspace(left, right, len(x_positions))]
    y_labels = [degree_formatter(value, 0) for value in np.linspace(top, bottom, len(y_positions))]
    ax.set_xticks(x_positions, x_labels)
    ax.set_yticks(y_positions, y_labels)
    ax.yaxis.tick_right()
    ax.tick_params(axis="x", labelsize=6, length=2, pad=1)
    ax.tick_params(axis="y", labelsize=6, length=2, pad=1, labelleft=False, labelright=True)
    # Keep coordinate text compact; explicit axis labels crowd multi-panel examples.


def rescale_generated_bh_to_reference_max(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    pred = prediction.astype(np.float32)
    finite_pred = np.isfinite(pred)
    finite_ref = np.isfinite(reference)
    if not finite_pred.any() or not finite_ref.any():
        return pred
    pred_min = float(np.nanmin(pred[finite_pred]))
    pred_max = float(np.nanmax(pred[finite_pred]))
    ref_max = float(np.nanmax(reference[finite_ref]))
    if pred_max <= pred_min or ref_max <= 0:
        return np.zeros_like(pred, dtype=np.float32)
    scaled = (pred - pred_min) / (pred_max - pred_min)
    return np.clip(scaled, 0.0, 1.0) * ref_max


def resize_context_axes_to_reference(
    axes: np.ndarray,
    *,
    scale: float,
    context_col: int,
    reference_col: int,
) -> None:
    scale = float(np.clip(scale, 0.05, 1.0))
    for row_index in range(axes.shape[0]):
        context_ax = axes[row_index, context_col]
        reference_pos = axes[row_index, reference_col].get_position()
        context_pos = context_ax.get_position()
        center_x = 0.5 * (context_pos.x0 + context_pos.x1)
        center_y = 0.5 * (reference_pos.y0 + reference_pos.y1)
        width = reference_pos.width * scale
        height = reference_pos.height * scale
        context_ax.set_position(
            [
                center_x - width / 2.0,
                center_y - height / 2.0,
                width,
                height,
            ]
        )


def save_figure(
    rows: list[dict[str, Any]],
    family_order: list[str],
    out_path: Path,
    *,
    coordinate_axes: bool,
    rescale_generated_bh: bool,
    context_panel_scale: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(rows)
    has_context = any(row.get("context") is not None for row in rows)
    n_cols = 2 + (2 * len(family_order)) + (1 if has_context else 0)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.55 * n_cols, 2.55 * n_rows),
        constrained_layout=False,
        squeeze=False,
    )
    bh_vmax = robust_height_vmax(rows, family_order)
    column_titles = []
    if has_context:
        first_context_title = next((row.get("context_title") for row in rows if row.get("context") is not None), "Context")
        column_titles.append(str(first_context_title))
    column_titles.append("Reference BF")
    column_titles.extend([f"{FAMILY_LABELS.get(family, family)} BF" for family in family_order])
    column_titles.append("Reference BH")
    bh_suffix = " BH*" if rescale_generated_bh else " BH"
    column_titles.extend([f"{FAMILY_LABELS.get(family, family)}{bh_suffix}" for family in family_order])
    reference_column_indexes = {0 if not has_context else 1}

    last_bf_im = None
    last_bh_im = None
    for row_index, row in enumerate(rows):
        extent = row.get("extent")
        plot_items: list[tuple[str, np.ndarray, str, float | None, float | None]] = []
        if has_context:
            plot_items.append(
                (
                    "context",
                    row["context"],
                    str(row["context_cmap"]),
                    row["context_vmin"],
                    row["context_vmax"],
                )
            )
        plot_items.extend([
            ("bf", row["bf_reference"], "viridis", 0.0, 1.0)
        ])
        plot_items.extend(
            [
                ("bf", row["predictions"][family]["bf"], "viridis", 0.0, 1.0)
                for family in family_order
            ]
        )
        plot_items.append(("bh", row["bh_reference"], "magma", 0.0, bh_vmax))
        bh_predictions = []
        for family in family_order:
            bh_pred = row["predictions"][family]["bh"]
            if rescale_generated_bh:
                bh_pred = rescale_generated_bh_to_reference_max(bh_pred, row["bh_reference"])
            bh_predictions.append(("bh", bh_pred, "magma", 0.0, bh_vmax))
        plot_items.extend(bh_predictions)
        for col_index, (_kind, array, cmap, vmin, vmax) in enumerate(plot_items):
            ax = axes[row_index, col_index]
            im = ax.imshow(
                array,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                origin="upper",
                aspect="auto",
            )
            if _kind == "context":
                ax.set_frame_on(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            if _kind == "bf":
                last_bf_im = im
            elif _kind == "bh":
                last_bh_im = im
            apply_coordinate_ticks(
                ax,
                extent,
                array.shape,
                show=coordinate_axes and col_index in reference_column_indexes,
            )
            if row_index == 0:
                ax.set_title(column_titles[col_index], fontsize=10)
    fig.subplots_adjust(left=0.18, right=0.995, top=0.90, bottom=0.10, hspace=0.10, wspace=0.08)
    if last_bf_im is not None:
        bf_start = 1 if has_context else 0
        bf_end = bf_start + 1 + len(family_order)
        cbar_bf = fig.colorbar(last_bf_im, ax=axes[:, bf_start:bf_end], orientation="horizontal", fraction=0.025, pad=0.03)
        cbar_bf.set_label("Building footprint fraction", fontsize=9)
    if last_bh_im is not None:
        bh_start = (1 if has_context else 0) + 1 + len(family_order)
        cbar_bh = fig.colorbar(last_bh_im, ax=axes[:, bh_start:], orientation="horizontal", fraction=0.025, pad=0.03)
        cbar_bh.set_label("Building height (m)", fontsize=9)
    if has_context:
        resize_context_axes_to_reference(
            axes,
            scale=context_panel_scale,
            context_col=0,
            reference_col=1,
        )
    for row_index, row in enumerate(rows):
        pos = axes[row_index, 0].get_position()
        fig.text(
            0.165,
            0.5 * (pos.y0 + pos.y1),
            row_label(row),
            fontsize=9,
            ha="right",
            va="center",
        )
    attributions = sorted({str(row["context_attribution"]) for row in rows if row.get("context_attribution")})
    if attributions:
        fig.text(
            0.995,
            0.012,
            "; ".join(attributions),
            fontsize=7,
            ha="right",
            va="bottom",
        )
    if rescale_generated_bh:
        fig.text(
            0.18,
            0.012,
            "BH* = generated BH min-max rescaled to the reference tile maximum.",
            fontsize=7,
            ha="left",
            va="bottom",
        )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    ov = load_overall_validation_module(Path(args.overall_validation_dir).resolve())
    ov.set_torch_determinism()
    config = configure_overall(args, ov)
    datasets = {
        name: ov.DatasetBundle(spec, cache_mode=str(config["cache_mode"]))
        for name, spec in config["dataset_specs"].items()
    }
    score_frames = [
        tile_scores(dataset, dataset_name, ov, config)
        for dataset_name, dataset in datasets.items()
    ]
    scores = pd.concat(score_frames, ignore_index=True)
    selected = select_representative_tiles(
        scores,
        min_active_fraction=float(args.min_active_fraction),
        examples_per_type=int(args.examples_per_type),
        unique_msas=bool(args.unique_msas),
        min_grid_distance=float(args.min_grid_distance),
    )
    tile_table = Path(args.tile_table).resolve()
    tile_table.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(tile_table, index=False)

    groups = group_lookup(config, ov)
    prediction_rows, family_order = collect_predictions(selected, datasets, groups, ov, config, args)
    out_path = Path(args.out).resolve()
    save_figure(
        prediction_rows,
        family_order,
        out_path,
        coordinate_axes=bool(args.coordinate_axes),
        rescale_generated_bh=bool(args.rescale_generated_bh_to_reference_max),
        context_panel_scale=float(args.context_panel_scale),
    )
    if args.copy_to_overall_results:
        copy_path = OVERALL_DIR / "results" / "figures" / out_path.name
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, copy_path)
    print(out_path)
    print(tile_table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
