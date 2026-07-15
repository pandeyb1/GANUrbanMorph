#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds, transform as window_transform
from shapely import wkb as shapely_wkb
from shapely.geometry import box

from ma_exact_footprints import (
    DEFAULT_STATE_CACHE_DIR,
    NLCD_2015_PATH,
    aggregate_exact_overlap,
    get_nlcd_target_crs,
    load_eligible_msas,
    read_state_cache_subset,
    required_state_codes,
    run_crs_preflight,
)


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_OUTPUT_DIR = THIS_DIR / "output"
DEFAULT_TEST_OUTPUT_DIR = THIS_DIR / "output_test"
DEFAULT_SPLIT_PATH = THIS_DIR / "msa_role_split.csv"

MASTER_SEED = 2026
DEFAULT_TEST_FRACTION = 0.25
DEFAULT_MIN_TEST_MSAS = 80
DEFAULT_WORKERS = max(1, min(6, (os.cpu_count() or 4) - 1))

TILE_SIZE = 256
PIXEL_SIZE_M = 30.0
TILE_AREA_M2 = (PIXEL_SIZE_M * TILE_SIZE) ** 2
BUFFER_M = 3840.0
PRIMARY_STRIDE = 128
FALLBACK_STRIDE = 64
MIN_POLYGON_COVERAGE = 0.80
MIN_BF_MEAN = 0.01
TARGET_POOL_MULTIPLIER = 2
METERS_TO_FEET = 1.0 / 0.3048

TRAIN_ARCHIVE_NAME = "Archive_CONUS_M1_random_stratified_2015.zip"
TEST_ARCHIVE_NAME = "Archive_CONUS_M1_random_stratified_2015_test.zip"

SCAN_STAGE_PRIMARY = "primary"
SCAN_STAGE_FALLBACK = "fallback"

BF_BINS = [
    ("0.01-0.05", "bf_0p01_0p05"),
    ("0.05-0.15", "bf_0p05_0p15"),
    ("0.15+", "bf_0p15_plus"),
]
HEIGHT_BINS = [
    ("<6", "h_lt6"),
    ("6-10", "h_6_10"),
    ("10-15", "h_10_15"),
    ("15+", "h_15_plus"),
]
BF_BIN_ID = {label: ident for label, ident in BF_BINS}
HEIGHT_BIN_ID = {label: ident for label, ident in HEIGHT_BINS}
STRATUM_LABEL_TO_ID = {
    (bf_label, height_label): f"{BF_BIN_ID[bf_label]}__{HEIGHT_BIN_ID[height_label]}"
    for bf_label, _ in BF_BINS
    for height_label, _ in HEIGHT_BINS
}
STRATUM_ID_TO_LABELS = {value: key for key, value in STRATUM_LABEL_TO_ID.items()}
STRATUM_IDS = [STRATUM_LABEL_TO_ID[(bf_label, height_label)] for bf_label, _ in BF_BINS for height_label, _ in HEIGHT_BINS]

STAGE_RANK = {
    SCAN_STAGE_PRIMARY: 0,
    SCAN_STAGE_FALLBACK: 1,
}

SCAN_COLUMNS = [
    "dataset_role",
    "stage",
    "order",
    "geoid",
    "msa_name",
    "source_states",
    "row",
    "col",
    "stride",
    "filename",
    "bf_mean",
    "bh_built_mean_m",
    "bh_density_m",
    "overlap_area_m2",
    "weighted_height_area_m3",
    "n_buildings",
    "bf_bin",
    "height_bin",
    "stratum_id",
]

RENDER_COLUMNS = [
    "dataset_role",
    "geoid",
    "msa_name",
    "source_states",
    "row",
    "col",
    "filename",
    "scan_stage",
    "scan_stride",
    "scan_bf_mean",
    "scan_bh_built_mean_m",
    "scan_bh_density_m",
    "scan_bf_bin",
    "scan_height_bin",
    "scan_stratum_id",
    "render_bf_mean",
    "render_bh_mean_m",
    "render_bh_built_mean_m",
    "render_bh_density_m",
    "render_bh_nonzero_frac",
    "render_bh_max_m",
    "render_bf_bin",
    "render_height_bin",
    "render_stratum_id",
]


@dataclass(frozen=True)
class DatasetSpec:
    role: str
    archive_name: str
    target_total: int
    stratum_floor: int
    per_msa_total_cap: int


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def dataset_spec_for(role: str) -> DatasetSpec:
    if role == "train":
        return DatasetSpec(
            role="train",
            archive_name=TRAIN_ARCHIVE_NAME,
            target_total=8000,
            stratum_floor=80,
            per_msa_total_cap=640,
        )
    if role == "test":
        return DatasetSpec(
            role="test",
            archive_name=TEST_ARCHIVE_NAME,
            target_total=2000,
            stratum_floor=20,
            per_msa_total_cap=160,
        )
    raise ValueError(f"Unsupported dataset role: {role}")


def zero_stratum_counts() -> dict[str, int]:
    return {sid: 0 for sid in STRATUM_IDS}


def normalize_counts(counts: Counter | dict[str, int]) -> dict[str, int]:
    out = zero_stratum_counts()
    for sid, value in counts.items():
        if sid in out:
            out[sid] = int(value)
    return out


def stratum_labels(stratum_id: str) -> tuple[str, str]:
    return STRATUM_ID_TO_LABELS[stratum_id]


def assign_bf_bin(bf_mean: float) -> Optional[str]:
    if bf_mean < MIN_BF_MEAN:
        return None
    if bf_mean < 0.05:
        return "0.01-0.05"
    if bf_mean < 0.15:
        return "0.05-0.15"
    return "0.15+"


def assign_height_bin(bh_built_mean_m: float) -> str:
    if bh_built_mean_m < 6.0:
        return "<6"
    if bh_built_mean_m < 10.0:
        return "6-10"
    if bh_built_mean_m < 15.0:
        return "10-15"
    return "15+"


def make_stratum_id(bf_bin: str, height_bin: str) -> str:
    return STRATUM_LABEL_TO_ID[(bf_bin, height_bin)]


def compute_balanced_hybrid_targets(
    counts: Counter | dict[str, int],
    spec: DatasetSpec,
    *,
    allow_partial: bool,
) -> dict[str, int]:
    available = normalize_counts(counts)
    total_available = sum(available.values())
    if total_available <= 0:
        return zero_stratum_counts()

    if allow_partial:
        target_total = min(spec.target_total, total_available)
    else:
        if total_available < spec.target_total:
            raise RuntimeError(
                f"{spec.role} candidate pool is too small for selection: "
                f"{total_available} candidates for {spec.target_total} requested tiles."
            )
        target_total = spec.target_total

    populated = [sid for sid in STRATUM_IDS if available[sid] > 0]
    targets = zero_stratum_counts()
    for sid in populated:
        targets[sid] = min(spec.stratum_floor, available[sid])

    assigned = sum(targets.values())
    if assigned > target_total:
        raise RuntimeError(
            f"{spec.role} stratum floors exceed total target: floor sum {assigned}, "
            f"target total {target_total}."
        )

    remaining = target_total - assigned
    if remaining <= 0:
        return targets

    residual = {sid: available[sid] - targets[sid] for sid in populated}
    residual_total = sum(residual.values())
    if residual_total <= 0:
        return targets

    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    used = 0
    for sid in populated:
        raw = remaining * residual[sid] / residual_total
        whole = min(residual[sid], int(math.floor(raw)))
        allocations[sid] = whole
        used += whole
        remainders.append((raw - whole, sid))

    remainder = remaining - used
    for _, sid in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remainder <= 0:
            break
        spare = residual[sid] - allocations[sid]
        if spare <= 0:
            continue
        allocations[sid] += 1
        remainder -= 1

    if remainder > 0:
        for sid in sorted(populated, key=lambda key: (-residual[key], key)):
            if remainder <= 0:
                break
            spare = residual[sid] - allocations[sid]
            if spare <= 0:
                continue
            take = min(spare, remainder)
            allocations[sid] += take
            remainder -= take

    for sid, add in allocations.items():
        targets[sid] += add
    return targets


def compute_pool_targets(counts: Counter | dict[str, int], spec: DatasetSpec) -> dict[str, int]:
    provisional = compute_balanced_hybrid_targets(counts, spec, allow_partial=True)
    return {sid: TARGET_POOL_MULTIPLIER * provisional[sid] for sid in STRATUM_IDS}


def compute_pool_deficits(
    counts: Counter | dict[str, int],
    pool_targets: dict[str, int],
) -> dict[str, int]:
    current = normalize_counts(counts)
    return {sid: max(0, int(pool_targets[sid]) - current[sid]) for sid in STRATUM_IDS}


def compute_per_stratum_caps(stratum_targets: dict[str, int]) -> dict[str, int]:
    return {
        sid: (max(1, int(math.floor(target * 0.20))) if target > 0 else 0)
        for sid, target in stratum_targets.items()
    }


def minimum_feasible_cap(stratum_counts_by_msa: list[int], target: int, base_cap: int) -> int:
    if target <= 0:
        return 0
    if not stratum_counts_by_msa:
        return base_cap

    lower = max(1, base_cap)
    upper = max(max(stratum_counts_by_msa), lower)
    if sum(min(count, lower) for count in stratum_counts_by_msa) >= target:
        return lower

    while lower < upper:
        middle = (lower + upper) // 2
        capacity = sum(min(count, middle) for count in stratum_counts_by_msa)
        if capacity >= target:
            upper = middle
        else:
            lower = middle + 1
    return lower


def compute_effective_per_stratum_caps(
    candidate_df: pd.DataFrame,
    stratum_targets: dict[str, int],
) -> dict[str, int]:
    base_caps = compute_per_stratum_caps(stratum_targets)
    effective_caps = dict(base_caps)
    for sid, target in stratum_targets.items():
        if target <= 0:
            continue
        counts = (
            candidate_df.loc[candidate_df["stratum_id"] == sid]
            .groupby("geoid")
            .size()
            .sort_values(ascending=False)
            .tolist()
        )
        if not counts:
            continue
        feasible_cap = minimum_feasible_cap(counts, int(target), int(base_caps[sid]))
        effective_caps[sid] = feasible_cap
    return effective_caps


def target_crs_file(output_dir: Path) -> Path:
    return output_dir / "target_crs_proj4.txt"


def candidate_tiles_path(output_dir: Path) -> Path:
    return output_dir / "candidate_tiles.csv"


def candidate_counts_path(output_dir: Path) -> Path:
    return output_dir / "candidate_bin_counts.csv"


def msa_selection_path(output_dir: Path) -> Path:
    return output_dir / "msa_selection.csv"


def tile_manifest_path(output_dir: Path) -> Path:
    return output_dir / "tile_manifest.csv"


def tile_manifest_rendered_path(output_dir: Path) -> Path:
    return output_dir / "tile_manifest_rendered.csv"


def collection_manifest_path(output_dir: Path) -> Path:
    return output_dir / "collection_manifest.json"


def scan_stage_dir(output_dir: Path, stage: str) -> Path:
    return output_dir / "scan_chunks" / stage


def render_metrics_dir(output_dir: Path) -> Path:
    return output_dir / "render_metrics_by_msa"


def default_args_for_role(role: str, default_output_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Build the {role} CONUS stratified archive.")
    parser.add_argument("--stage", choices=["scan", "select", "render", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--state-cache-dir", type=Path, default=DEFAULT_STATE_CACHE_DIR)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--min-test-msas", type=int, default=DEFAULT_MIN_TEST_MSAS)
    parser.add_argument("--max-msas", type=int, default=None)
    parser.add_argument("--max-windows-per-msa", type=int, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--force-rescan", action="store_true")
    parser.add_argument("--skip-zip", action="store_true")
    return parser


def tile_bounds(transform, row: int, col: int, tile_size: int = TILE_SIZE) -> tuple[float, float, float, float]:
    x0 = float(transform.c + col * transform.a)
    x1 = float(transform.c + (col + tile_size) * transform.a)
    y0 = float(transform.f + row * transform.e)
    y1 = float(transform.f + (row + tile_size) * transform.e)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def tile_geometry(transform, row: int, col: int, tile_size: int = TILE_SIZE):
    return box(*tile_bounds(transform, row, col, tile_size))


def snapped_window_ranges(
    bounds: tuple[float, float, float, float],
    transform,
    raster_width: int,
    raster_height: int,
    stride: int,
    tile_size: int = TILE_SIZE,
) -> tuple[range, range]:
    raw = from_bounds(*bounds, transform=transform)
    row_start = int(math.floor(raw.row_off / stride) * stride)
    col_start = int(math.floor(raw.col_off / stride) * stride)
    row_stop = int(math.ceil((raw.row_off + raw.height - tile_size) / stride) * stride)
    col_stop = int(math.ceil((raw.col_off + raw.width - tile_size) / stride) * stride)

    max_row = max(0, raster_height - tile_size)
    max_col = max(0, raster_width - tile_size)
    row_start = min(max(row_start, 0), max_row)
    col_start = min(max(col_start, 0), max_col)
    row_stop = min(max(row_stop, 0), max_row)
    col_stop = min(max(col_stop, 0), max_col)
    if row_stop < row_start:
        row_stop = row_start
    if col_stop < col_start:
        col_stop = col_start
    return range(row_start, row_stop + 1, stride), range(col_start, col_stop + 1, stride)


def scan_tile_fast_metrics(buildings: gpd.GeoDataFrame, tile_geom) -> tuple[float, float, int]:
    if buildings.empty:
        return 0.0, 0.0, 0
    overlaps = buildings.geometry.intersection(tile_geom)
    areas = overlaps.area.to_numpy(dtype=float)
    valid = areas > 0.0
    if not valid.any():
        return 0.0, 0.0, 0
    heights = buildings["height_m"].to_numpy(dtype=float)[valid]
    overlap_area = float(areas[valid].sum())
    weighted_height_area = float(np.sum(areas[valid] * heights))
    return overlap_area, weighted_height_area, int(valid.sum())


def ensure_split_file(
    target_crs,
    split_path: Path,
    *,
    master_seed: int,
    test_fraction: float,
    min_test_msas: int,
) -> pd.DataFrame:
    if split_path.exists():
        split = pd.read_csv(split_path, dtype={"GEOID": str})
        expected = {"GEOID", "NAME", "global_order", "dataset_role"}
        if not expected.issubset(split.columns):
            raise RuntimeError(f"Existing split file is missing required columns: {split_path}")
        return split

    all_msas = load_eligible_msas(target_crs, master_seed, None)
    total = len(all_msas)
    n_test = max(min_test_msas, int(round(total * test_fraction)))
    n_test = min(max(1, n_test), max(1, total - 1))
    dataset_role = np.array(["train"] * total, dtype=object)
    dataset_role[:n_test] = "test"
    split = pd.DataFrame(
        {
            "GEOID": all_msas["GEOID"].astype(str),
            "NAME": all_msas["NAME"].astype(str),
            "global_order": np.arange(1, total + 1),
            "dataset_role": dataset_role,
        }
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(split_path, index=False)
    return split


def load_role_msas(role: str, target_crs, args) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    split = ensure_split_file(
        target_crs,
        args.split_path,
        master_seed=args.master_seed,
        test_fraction=args.test_fraction,
        min_test_msas=args.min_test_msas,
    )
    all_msas = load_eligible_msas(target_crs, args.master_seed, None)
    merged = all_msas.merge(split[["GEOID", "dataset_role", "global_order"]], on="GEOID", how="inner")
    merged = merged.loc[merged["dataset_role"] == role].copy()
    merged = merged.sort_values("global_order").reset_index(drop=True)
    if args.max_msas is not None:
        merged = merged.iloc[: args.max_msas].copy().reset_index(drop=True)
    merged["order"] = np.arange(1, len(merged) + 1)
    return merged, split


def clear_existing_outputs(output_dir: Path) -> None:
    paths = [
        output_dir / "central",
        output_dir / "BFrac",
        output_dir / "BHeight",
        output_dir / "scan_chunks",
        output_dir / "render_metrics_by_msa",
        candidate_tiles_path(output_dir),
        candidate_counts_path(output_dir),
        msa_selection_path(output_dir),
        tile_manifest_path(output_dir),
        tile_manifest_rendered_path(output_dir),
        collection_manifest_path(output_dir),
        target_crs_file(output_dir),
        output_dir / "msa_role_split.csv",
        output_dir / TRAIN_ARCHIVE_NAME,
        output_dir / TEST_ARCHIVE_NAME,
    ]
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def create_empty_scan_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SCAN_COLUMNS)


def read_scan_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["stratum_counts"] = normalize_counts(payload.get("stratum_counts", {}))
    return payload


def summary_paths_for(stage_dir: Path, order: int, geoid: str) -> tuple[Path, Path]:
    stem = f"{int(order):03d}_{geoid}"
    return stage_dir / f"{stem}_candidates.csv", stage_dir / f"{stem}_summary.json"


def build_scan_task(
    role: str,
    stage: str,
    stage_dir: Path,
    msa_row,
    args,
    target_proj4: str,
    *,
    stride: int,
    allowed_strata: Optional[set[str]],
) -> dict:
    candidate_csv, summary_json = summary_paths_for(stage_dir, int(msa_row["order"]), str(msa_row["GEOID"]))
    return {
        "dataset_role": role,
        "stage": stage,
        "order": int(msa_row["order"]),
        "geoid": str(msa_row["GEOID"]),
        "msa_name": str(msa_row["NAME"]),
        "msa_wkb": bytes(msa_row.geometry.wkb),
        "state_cache_dir": str(args.state_cache_dir),
        "nlcd_path": str(NLCD_2015_PATH),
        "target_proj4": target_proj4,
        "stride": int(stride),
        "tile_size": TILE_SIZE,
        "buffer_m": BUFFER_M,
        "min_polygon_coverage": MIN_POLYGON_COVERAGE,
        "min_bf_mean": MIN_BF_MEAN,
        "max_windows_per_msa": args.max_windows_per_msa,
        "allowed_strata": sorted(allowed_strata) if allowed_strata else None,
        "candidate_csv": str(candidate_csv),
        "summary_json": str(summary_json),
    }


def scan_msa_worker(task: dict) -> dict:
    stage_dir = Path(task["candidate_csv"]).parent
    stage_dir.mkdir(parents=True, exist_ok=True)

    msa_geom = shapely_wkb.loads(task["msa_wkb"])
    buffered = msa_geom.buffer(float(task["buffer_m"]))
    allowed_strata = set(task["allowed_strata"]) if task["allowed_strata"] is not None else None
    state_codes = required_state_codes(task["msa_name"])
    source_states = ";".join(state_codes)

    buildings = read_state_cache_subset(
        Path(task["state_cache_dir"]),
        state_codes,
        buffered.bounds,
    )
    if len(buildings):
        buildings = buildings.loc[buildings.geometry.intersects(buffered)].copy()
        buildings = buildings.reset_index(drop=True)
        try:
            spatial_index = buildings.sindex
        except Exception:
            spatial_index = None
    else:
        spatial_index = None

    with rasterio.open(task["nlcd_path"]) as src:
        nlcd_transform = src.transform
        raster_width = src.width
        raster_height = src.height

    row_range, col_range = snapped_window_ranges(
        buffered.bounds,
        nlcd_transform,
        raster_width,
        raster_height,
        int(task["stride"]),
        int(task["tile_size"]),
    )

    records: list[dict] = []
    stratum_counts = Counter()
    windows_considered = 0
    stop_early = False
    started = time.time()

    for row in row_range:
        for col in col_range:
            tile = tile_geometry(nlcd_transform, row, col, int(task["tile_size"]))
            coverage = float(tile.intersection(msa_geom).area / TILE_AREA_M2)
            if coverage < float(task["min_polygon_coverage"]):
                continue
            windows_considered += 1
            if task["max_windows_per_msa"] is not None and windows_considered > int(task["max_windows_per_msa"]):
                stop_early = True
                break

            if buildings.empty:
                continue
            if spatial_index is not None:
                subset_idx = list(spatial_index.intersection(tile.bounds))
                if not subset_idx:
                    continue
                subset = buildings.iloc[subset_idx].copy()
                subset = subset.loc[subset.geometry.intersects(tile)].copy()
            else:
                subset = buildings.loc[buildings.geometry.intersects(tile)].copy()
            if subset.empty:
                continue

            overlap_area_m2, weighted_height_area, n_buildings = scan_tile_fast_metrics(subset, tile)
            if overlap_area_m2 <= 0.0:
                continue
            bf_mean = overlap_area_m2 / TILE_AREA_M2
            if bf_mean < float(task["min_bf_mean"]):
                continue
            bh_built_mean_m = weighted_height_area / overlap_area_m2
            bh_density_m = weighted_height_area / TILE_AREA_M2

            bf_bin = assign_bf_bin(bf_mean)
            if bf_bin is None:
                continue
            height_bin = assign_height_bin(bh_built_mean_m)
            stratum_id = make_stratum_id(bf_bin, height_bin)
            if allowed_strata is not None and stratum_id not in allowed_strata:
                continue

            filename = f"MSA_{task['geoid']}_X{col}_Y{row}.tif"
            records.append(
                {
                    "dataset_role": task["dataset_role"],
                    "stage": task["stage"],
                    "order": task["order"],
                    "geoid": task["geoid"],
                    "msa_name": task["msa_name"],
                    "source_states": source_states,
                    "row": int(row),
                    "col": int(col),
                    "stride": int(task["stride"]),
                    "filename": filename,
                    "bf_mean": float(bf_mean),
                    "bh_built_mean_m": float(bh_built_mean_m),
                    "bh_density_m": float(bh_density_m),
                    "overlap_area_m2": float(overlap_area_m2),
                    "weighted_height_area_m3": float(weighted_height_area),
                    "n_buildings": int(n_buildings),
                    "bf_bin": bf_bin,
                    "height_bin": height_bin,
                    "stratum_id": stratum_id,
                }
            )
            stratum_counts[stratum_id] += 1
        if stop_early:
            break

    frame = pd.DataFrame.from_records(records, columns=SCAN_COLUMNS) if records else create_empty_scan_frame()
    frame.to_csv(task["candidate_csv"], index=False)
    summary = {
        "dataset_role": task["dataset_role"],
        "stage": task["stage"],
        "order": task["order"],
        "geoid": task["geoid"],
        "msa_name": task["msa_name"],
        "source_states": source_states,
        "stride": int(task["stride"]),
        "windows_considered": int(windows_considered),
        "candidates_added": int(len(frame)),
        "elapsed_sec": round(time.time() - started, 3),
        "candidate_csv": str(task["candidate_csv"]),
        "summary_json": str(task["summary_json"]),
        "stratum_counts": normalize_counts(stratum_counts),
    }
    Path(task["summary_json"]).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def iter_parallel_results(worker_fn, tasks: list[dict], max_workers: int):
    if not tasks:
        return
    if max_workers <= 1:
        for task in tasks:
            yield worker_fn(task)
        return
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(worker_fn, task): task for task in tasks}
            for future in as_completed(futures):
                yield future.result()
        return
    except (OSError, PermissionError) as exc:
        log(f"ProcessPool unavailable ({exc}); falling back to threads.")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker_fn, task): task for task in tasks}
        for future in as_completed(futures):
            yield future.result()


def run_parallel_scan_stage(
    role: str,
    msas: gpd.GeoDataFrame,
    output_dir: Path,
    args,
    target_proj4: str,
    *,
    stage: str,
    stride: int,
    allowed_strata: Optional[set[str]] = None,
) -> list[dict]:
    stage_dir = scan_stage_dir(output_dir, stage)
    stage_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    pending_tasks: list[dict] = []
    allowed_strata = set(allowed_strata) if allowed_strata else None

    for _, msa_row in msas.iterrows():
        task = build_scan_task(
            role,
            stage,
            stage_dir,
            msa_row,
            args,
            target_proj4,
            stride=stride,
            allowed_strata=allowed_strata,
        )
        candidate_csv = Path(task["candidate_csv"])
        summary_json = Path(task["summary_json"])
        if candidate_csv.exists() and summary_json.exists():
            summaries.append(read_scan_summary(summary_json))
        else:
            pending_tasks.append(task)

    if pending_tasks:
        for summary in iter_parallel_results(scan_msa_worker, pending_tasks, int(args.workers)):
            summaries.append(summary)
            deficits = {
                sid: summary["stratum_counts"][sid]
                for sid in STRATUM_IDS
                if summary["stratum_counts"][sid] > 0
            }
            log(
                f"{stage.title()} MSA {int(summary['order']):03d} {summary['msa_name']}: "
                f"{summary['candidates_added']} candidates"
                + (f", counts {deficits}" if deficits else "")
            )

    summaries.sort(key=lambda payload: (int(payload["order"]), STAGE_RANK[payload["stage"]], payload["geoid"]))
    return summaries


def read_stage_candidates(stage_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(stage_dir.glob("*_candidates.csv")):
        frame = pd.read_csv(path, dtype={"geoid": str})
        if len(frame):
            frames.append(frame)
    if not frames:
        return create_empty_scan_frame()
    return pd.concat(frames, ignore_index=True)


def aggregate_scan_outputs(
    role: str,
    output_dir: Path,
    split: pd.DataFrame,
    target_proj4: str,
    summaries: list[dict],
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_frames = []
    for stage in [SCAN_STAGE_PRIMARY, SCAN_STAGE_FALLBACK]:
        frame = read_stage_candidates(scan_stage_dir(output_dir, stage))
        if len(frame):
            stage_frames.append(frame)
    if stage_frames:
        candidate_df = pd.concat(stage_frames, ignore_index=True)
        candidate_df["stage_rank"] = candidate_df["stage"].map(STAGE_RANK)
        candidate_df = candidate_df.sort_values(["stage_rank", "order", "geoid", "row", "col"]).drop_duplicates(
            subset=["geoid", "row", "col"],
            keep="first",
        )
        candidate_df = candidate_df.drop(columns=["stage_rank"]).reset_index(drop=True)
    else:
        candidate_df = create_empty_scan_frame()
    candidate_df.to_csv(candidate_tiles_path(output_dir), index=False)

    selection_rows = []
    for summary in summaries:
        row = {
            "dataset_role": role,
            "stage": summary["stage"],
            "order": summary["order"],
            "geoid": summary["geoid"],
            "msa_name": summary["msa_name"],
            "source_states": summary["source_states"],
            "stride": summary["stride"],
            "windows_considered": summary["windows_considered"],
            "candidates_added": summary["candidates_added"],
            "elapsed_sec": summary["elapsed_sec"],
        }
        for sid in STRATUM_IDS:
            row[f"count_{sid}"] = int(summary["stratum_counts"][sid])
        selection_rows.append(row)
    pd.DataFrame(selection_rows).to_csv(msa_selection_path(output_dir), index=False)

    cumulative = Counter()
    count_rows = []
    for summary in summaries:
        for sid in STRATUM_IDS:
            count = int(summary["stratum_counts"][sid])
            if count <= 0:
                continue
            cumulative[sid] += count
            bf_label, height_label = stratum_labels(sid)
            count_rows.append(
                {
                    "dataset_role": role,
                    "stage": summary["stage"],
                    "order": summary["order"],
                    "geoid": summary["geoid"],
                    "msa_name": summary["msa_name"],
                    "stratum_id": sid,
                    "bf_bin": bf_label,
                    "height_bin": height_label,
                    "count": count,
                    "cumulative_count": cumulative[sid],
                }
            )
    pd.DataFrame(count_rows).to_csv(candidate_counts_path(output_dir), index=False)

    target_crs_file(output_dir).write_text(target_proj4, encoding="utf-8")
    split.to_csv(output_dir / "msa_role_split.csv", index=False)
    return candidate_df


def run_scan_stage(role: str, spec: DatasetSpec, args) -> tuple[pd.DataFrame, dict]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.force_rescan:
        clear_existing_outputs(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Output directory: {output_dir}")
    log(f"State cache directory: {args.state_cache_dir}")
    log(f"Split path: {args.split_path}")
    log(f"Master seed: {args.master_seed}")

    target_crs, target_proj4 = get_nlcd_target_crs(NLCD_2015_PATH)
    x, y = run_crs_preflight(target_crs)
    log(f"Target CRS preflight passed: ({x}, {y})")

    msas, split = load_role_msas(role, target_crs, args)
    log(
        f"Loaded {len(msas)} {role} MSAs from deterministic split at {args.split_path.name}; "
        "San Diego excluded."
    )

    primary_summaries = run_parallel_scan_stage(
        role,
        msas,
        output_dir,
        args,
        target_proj4,
        stage=SCAN_STAGE_PRIMARY,
        stride=PRIMARY_STRIDE,
        allowed_strata=None,
    )
    primary_df = aggregate_scan_outputs(role, output_dir, split, target_proj4, primary_summaries)
    primary_counts = normalize_counts(Counter(primary_df["stratum_id"])) if len(primary_df) else zero_stratum_counts()
    provisional_targets = compute_balanced_hybrid_targets(primary_counts, spec, allow_partial=True)
    pool_targets = compute_pool_targets(primary_counts, spec)
    primary_deficits = compute_pool_deficits(primary_counts, pool_targets)

    fallback_summaries: list[dict] = []
    deficit_strata = {sid for sid, deficit in primary_deficits.items() if deficit > 0}
    if deficit_strata:
        log(
            "Primary scan fallback deficits by stratum: "
            + ", ".join(f"{sid}={primary_deficits[sid]}" for sid in STRATUM_IDS if primary_deficits[sid] > 0)
        )
        fallback_summaries = run_parallel_scan_stage(
            role,
            msas,
            output_dir,
            args,
            target_proj4,
            stage=SCAN_STAGE_FALLBACK,
            stride=FALLBACK_STRIDE,
            allowed_strata=deficit_strata,
        )

    all_summaries = primary_summaries + fallback_summaries
    candidate_df = aggregate_scan_outputs(role, output_dir, split, target_proj4, all_summaries)
    final_counts = normalize_counts(Counter(candidate_df["stratum_id"])) if len(candidate_df) else zero_stratum_counts()
    final_targets = compute_balanced_hybrid_targets(final_counts, spec, allow_partial=True)
    remaining_deficits = compute_pool_deficits(final_counts, pool_targets)

    scan_context = {
        "primary_candidate_counts": primary_counts,
        "primary_provisional_targets": provisional_targets,
        "primary_pool_targets": pool_targets,
        "primary_deficits": primary_deficits,
        "fallback_deficit_strata": sorted(deficit_strata),
        "final_candidate_counts": final_counts,
        "final_partial_targets": final_targets,
        "remaining_pool_deficits": remaining_deficits,
    }
    write_collection_manifest(output_dir, args, spec, scan_context, candidate_df)
    return candidate_df, scan_context


def load_candidate_tiles(output_dir: Path) -> pd.DataFrame:
    path = candidate_tiles_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate tile table: {path}")
    frame = pd.read_csv(path, dtype={"geoid": str})
    if len(frame):
        frame["row"] = frame["row"].astype(int)
        frame["col"] = frame["col"].astype(int)
        frame["order"] = frame["order"].astype(int)
        frame["stride"] = frame["stride"].astype(int)
    return frame


def select_tiles(candidate_df: pd.DataFrame, spec: DatasetSpec, master_seed: int) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    counts = normalize_counts(Counter(candidate_df["stratum_id"])) if len(candidate_df) else zero_stratum_counts()
    targets = compute_balanced_hybrid_targets(counts, spec, allow_partial=False)
    caps = compute_effective_per_stratum_caps(candidate_df, targets)

    rng = np.random.default_rng(master_seed + (0 if spec.role == "train" else 1000))
    working = candidate_df.copy().reset_index(drop=True)
    working["shuffle_key"] = rng.permutation(len(working))

    scarcity = []
    for sid in STRATUM_IDS:
        target = targets[sid]
        if target <= 0:
            continue
        availability = counts[sid]
        scarcity.append((availability / target, availability, sid))
    scarcity.sort(key=lambda item: (item[0], item[1], item[2]))

    selected_indices: list[int] = []
    msa_total = Counter()
    msa_stratum = Counter()
    deficits: dict[str, int] = {}

    for _, _, sid in scarcity:
        subset = working.loc[working["stratum_id"] == sid].sort_values(
            ["shuffle_key", "order", "geoid", "row", "col"]
        )
        taken = 0
        cap = caps[sid]
        for idx, row in subset.iterrows():
            geoid = str(row["geoid"])
            if msa_total[geoid] >= spec.per_msa_total_cap:
                continue
            if msa_stratum[(geoid, sid)] >= cap:
                continue
            selected_indices.append(idx)
            msa_total[geoid] += 1
            msa_stratum[(geoid, sid)] += 1
            taken += 1
            if taken >= targets[sid]:
                break
        if taken < targets[sid]:
            deficits[sid] = int(targets[sid] - taken)

    if deficits:
        raise RuntimeError(
            "Selection could not satisfy the per-stratum targets under the MSA caps: "
            + ", ".join(f"{sid}={value}" for sid, value in deficits.items())
        )

    selected_df = working.loc[selected_indices].drop(columns=["shuffle_key"]).copy()
    selected_df = selected_df.sort_values(["stratum_id", "order", "geoid", "row", "col"]).reset_index(drop=True)
    return selected_df, targets, caps


def run_select_stage(spec: DatasetSpec, args, scan_context: Optional[dict] = None) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    candidate_df = load_candidate_tiles(args.output_dir)
    selected_df, stratum_targets, per_stratum_caps = select_tiles(candidate_df, spec, args.master_seed)
    selected_df.to_csv(tile_manifest_path(args.output_dir), index=False)
    write_collection_manifest(
        args.output_dir,
        args,
        spec,
        scan_context or {},
        candidate_df,
        selected_df=selected_df,
        stratum_targets=stratum_targets,
        per_stratum_caps=per_stratum_caps,
    )
    return selected_df, stratum_targets, per_stratum_caps


def load_selected_tiles(output_dir: Path) -> pd.DataFrame:
    path = tile_manifest_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing tile manifest: {path}")
    frame = pd.read_csv(path, dtype={"geoid": str})
    if len(frame):
        frame["row"] = frame["row"].astype(int)
        frame["col"] = frame["col"].astype(int)
        frame["order"] = frame["order"].astype(int)
        frame["stride"] = frame["stride"].astype(int)
    return frame


def build_render_task(output_dir: Path, msa_row, selected_rows: pd.DataFrame) -> dict:
    metrics_csv = render_metrics_dir(output_dir) / f"{int(msa_row['order']):03d}_{msa_row['GEOID']}.csv"
    return {
        "geoid": str(msa_row["GEOID"]),
        "msa_name": str(msa_row["NAME"]),
        "msa_wkb": bytes(msa_row.geometry.wkb),
        "state_cache_dir": str(output_dir.parent / "state_cache"),
        "nlcd_path": str(NLCD_2015_PATH),
        "output_dir": str(output_dir),
        "metrics_csv": str(metrics_csv),
        "rows": selected_rows.to_dict(orient="records"),
    }


def render_msa_worker(task: dict) -> dict:
    output_dir = Path(task["output_dir"])
    metrics_csv = Path(task["metrics_csv"])
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "central").mkdir(parents=True, exist_ok=True)
    (output_dir / "BFrac").mkdir(parents=True, exist_ok=True)
    (output_dir / "BHeight").mkdir(parents=True, exist_ok=True)

    msa_geom = shapely_wkb.loads(task["msa_wkb"])
    buffered = msa_geom.buffer(BUFFER_M)
    state_codes = required_state_codes(task["msa_name"])
    buildings = read_state_cache_subset(Path(task["state_cache_dir"]), state_codes, buffered.bounds)
    if len(buildings):
        buildings = buildings.loc[buildings.geometry.intersects(buffered)].copy().reset_index(drop=True)
        try:
            spatial_index = buildings.sindex
        except Exception:
            spatial_index = None
    else:
        spatial_index = None

    render_rows: list[dict] = []
    with rasterio.open(task["nlcd_path"]) as src:
        for row in task["rows"]:
            row_off = int(row["row"])
            col_off = int(row["col"])
            window = Window(col_off=col_off, row_off=row_off, width=TILE_SIZE, height=TILE_SIZE)
            transform = window_transform(window, src.transform)
            tile = tile_geometry(src.transform, row_off, col_off, TILE_SIZE)

            central = src.read(1, window=window, boundless=False)
            base_profile = src.profile.copy()
            base_profile.update(
                height=TILE_SIZE,
                width=TILE_SIZE,
                transform=transform,
                count=1,
                compress="lzw",
            )

            if buildings.empty:
                subset = gpd.GeoDataFrame({"height_m": []}, geometry="geometry", crs=src.crs)
            elif spatial_index is not None:
                subset_idx = list(spatial_index.intersection(tile.bounds))
                subset = buildings.iloc[subset_idx].copy() if subset_idx else buildings.iloc[0:0].copy()
                if len(subset):
                    subset = subset.loc[subset.geometry.intersects(tile)].copy()
            else:
                subset = buildings.loc[buildings.geometry.intersects(tile)].copy()

            bf, bh_m = aggregate_exact_overlap(subset, transform, tile_size=TILE_SIZE)
            bh_ft = (bh_m * METERS_TO_FEET).astype(np.float32)

            central_path = output_dir / "central" / str(row["filename"])
            bf_path = output_dir / "BFrac" / str(row["filename"])
            bh_path = output_dir / "BHeight" / str(row["filename"])

            with rasterio.open(central_path, "w", **base_profile) as dst:
                dst.write(central, 1)
            bf_profile = base_profile.copy()
            bf_profile.update(dtype="float32", nodata=0.0)
            with rasterio.open(bf_path, "w", **bf_profile) as dst:
                dst.write(bf.astype(np.float32), 1)
            bh_profile = base_profile.copy()
            bh_profile.update(dtype="float32", nodata=0.0)
            with rasterio.open(bh_path, "w", **bh_profile) as dst:
                dst.write(bh_ft, 1)

            overlap_area_m2 = float(np.sum(bf, dtype=np.float64) * (PIXEL_SIZE_M ** 2))
            weighted_height_area = float(np.sum((bh_m * bf), dtype=np.float64) * (PIXEL_SIZE_M ** 2))
            render_bf_mean = float(np.mean(bf, dtype=np.float64))
            render_bh_mean_m = float(np.mean(bh_m, dtype=np.float64))
            render_bh_built_mean_m = float(weighted_height_area / overlap_area_m2) if overlap_area_m2 > 0 else 0.0
            render_bh_density_m = float(weighted_height_area / TILE_AREA_M2)
            render_bh_nonzero_frac = float(np.mean(bh_m > 0.0))
            render_bh_max_m = float(np.max(bh_m)) if bh_m.size else 0.0
            render_bf_bin = assign_bf_bin(render_bf_mean)
            render_height_bin = assign_height_bin(render_bh_built_mean_m) if overlap_area_m2 > 0 else None
            render_stratum_id = (
                make_stratum_id(render_bf_bin, render_height_bin)
                if render_bf_bin is not None and render_height_bin is not None
                else None
            )

            render_rows.append(
                {
                    "dataset_role": row["dataset_role"],
                    "geoid": row["geoid"],
                    "msa_name": row["msa_name"],
                    "source_states": row["source_states"],
                    "row": row_off,
                    "col": col_off,
                    "filename": row["filename"],
                    "scan_stage": row["stage"],
                    "scan_stride": int(row["stride"]),
                    "scan_bf_mean": float(row["bf_mean"]),
                    "scan_bh_built_mean_m": float(row["bh_built_mean_m"]),
                    "scan_bh_density_m": float(row["bh_density_m"]),
                    "scan_bf_bin": row["bf_bin"],
                    "scan_height_bin": row["height_bin"],
                    "scan_stratum_id": row["stratum_id"],
                    "render_bf_mean": render_bf_mean,
                    "render_bh_mean_m": render_bh_mean_m,
                    "render_bh_built_mean_m": render_bh_built_mean_m,
                    "render_bh_density_m": render_bh_density_m,
                    "render_bh_nonzero_frac": render_bh_nonzero_frac,
                    "render_bh_max_m": render_bh_max_m,
                    "render_bf_bin": render_bf_bin,
                    "render_height_bin": render_height_bin,
                    "render_stratum_id": render_stratum_id,
                }
            )

    render_df = pd.DataFrame(render_rows, columns=RENDER_COLUMNS)
    render_df.to_csv(metrics_csv, index=False)
    return {
        "geoid": task["geoid"],
        "msa_name": task["msa_name"],
        "rows_rendered": int(len(render_rows)),
        "metrics_csv": str(metrics_csv),
    }


def render_selected_tiles(role: str, selected_df: pd.DataFrame, args) -> pd.DataFrame:
    output_dir = args.output_dir
    metrics_dir = render_metrics_dir(output_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    target_crs, _ = get_nlcd_target_crs(NLCD_2015_PATH)
    msas, _ = load_role_msas(role, target_crs, args)
    msas = msas.set_index("GEOID", drop=False)

    tasks = []
    summaries: list[dict] = []
    for geoid, group in selected_df.groupby("geoid", sort=False):
        msa_row = msas.loc[str(geoid)]
        metrics_csv = metrics_dir / f"{int(msa_row['order']):03d}_{geoid}.csv"
        if metrics_csv.exists():
            summaries.append({"geoid": geoid, "msa_name": msa_row["NAME"], "rows_rendered": int(len(group))})
            continue
        task = build_render_task(output_dir, msa_row, group)
        task["state_cache_dir"] = str(args.state_cache_dir)
        tasks.append(task)

    if tasks:
        for summary in iter_parallel_results(render_msa_worker, tasks, int(args.workers)):
            summaries.append(summary)
            log(f"Rendered {summary['rows_rendered']} tiles for {summary['msa_name']}")

    frames = []
    for path in sorted(metrics_dir.glob("*.csv")):
        frame = pd.read_csv(path, dtype={"geoid": str})
        if len(frame):
            frames.append(frame)
    rendered_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RENDER_COLUMNS)
    rendered_df.to_csv(tile_manifest_rendered_path(output_dir), index=False)
    return rendered_df


def validate_rendered_archive(output_dir: Path, selected_df: pd.DataFrame) -> dict:
    expected = set(selected_df["filename"].tolist()) if len(selected_df) else set()
    central = {path.name for path in (output_dir / "central").glob("*.tif")} if (output_dir / "central").exists() else set()
    bfrac = {path.name for path in (output_dir / "BFrac").glob("*.tif")} if (output_dir / "BFrac").exists() else set()
    bheight = {path.name for path in (output_dir / "BHeight").glob("*.tif")} if (output_dir / "BHeight").exists() else set()
    return {
        "expected_tiles": len(expected),
        "central_tiles": len(central),
        "bfrac_tiles": len(bfrac),
        "bheight_tiles": len(bheight),
        "aligned_with_manifest": central == bfrac == bheight == expected,
    }


def load_existing_manifest(output_dir: Path) -> dict:
    path = collection_manifest_path(output_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def write_collection_manifest(
    output_dir: Path,
    args,
    spec: DatasetSpec,
    scan_context: dict,
    candidate_df: pd.DataFrame,
    *,
    selected_df: Optional[pd.DataFrame] = None,
    rendered_df: Optional[pd.DataFrame] = None,
    stratum_targets: Optional[dict[str, int]] = None,
    per_stratum_caps: Optional[dict[str, int]] = None,
    validation: Optional[dict] = None,
) -> None:
    manifest = load_existing_manifest(output_dir)
    candidate_counts = normalize_counts(Counter(candidate_df["stratum_id"])) if len(candidate_df) else zero_stratum_counts()

    if stratum_targets is None:
        try:
            stratum_targets = compute_balanced_hybrid_targets(candidate_counts, spec, allow_partial=True)
        except RuntimeError:
            stratum_targets = zero_stratum_counts()
    if per_stratum_caps is None:
        per_stratum_caps = compute_per_stratum_caps(stratum_targets)

    manifest.update(
        {
            "created_utc": now_utc(),
            "dataset_role": spec.role,
            "output_dir": str(output_dir),
            "state_cache_dir": str(args.state_cache_dir),
            "split_path": str(args.split_path),
            "master_seed": int(args.master_seed),
            "test_fraction": float(args.test_fraction),
            "min_test_msas": int(args.min_test_msas),
            "target_total": int(spec.target_total),
            "per_msa_total_cap": int(spec.per_msa_total_cap),
            "stratification": {
                "type": "2d_bf_x_height_balanced_hybrid",
                "bf_bins": [label for label, _ in BF_BINS],
                "height_bins_m": [label for label, _ in HEIGHT_BINS],
                "strata": [
                    {
                        "stratum_id": sid,
                        "bf_bin": stratum_labels(sid)[0],
                        "height_bin": stratum_labels(sid)[1],
                    }
                    for sid in STRATUM_IDS
                ],
                "minimum_bf_mean": MIN_BF_MEAN,
                "height_metric": "bh_built_mean_m",
                "quota_style": "balanced_hybrid",
                "stratum_floor": int(spec.stratum_floor),
            },
            "candidate_pool_counts": candidate_counts,
            "computed_stratum_targets": stratum_targets,
            "dynamic_per_stratum_caps": per_stratum_caps,
        }
    )

    scan_keys = {
        "primary_candidate_counts": "primary_candidate_counts",
        "primary_provisional_targets": "primary_provisional_targets",
        "primary_pool_targets": "primary_pool_targets",
        "primary_fallback_deficits": "primary_deficits",
        "fallback_deficit_strata": "fallback_deficit_strata",
        "remaining_pool_deficits": "remaining_pool_deficits",
    }
    for manifest_key, context_key in scan_keys.items():
        if context_key in scan_context:
            manifest[manifest_key] = scan_context[context_key]

    if selected_df is not None:
        manifest["selected_tiles"] = int(len(selected_df))
        manifest["selected_stratum_counts"] = normalize_counts(Counter(selected_df["stratum_id"])) if len(selected_df) else zero_stratum_counts()
    if rendered_df is not None:
        manifest["rendered_tiles"] = int(len(rendered_df))
        manifest["rendered_scan_stratum_counts"] = normalize_counts(Counter(rendered_df["scan_stratum_id"])) if len(rendered_df) else zero_stratum_counts()
        manifest["rendered_exact_stratum_counts"] = normalize_counts(Counter(rendered_df["render_stratum_id"].dropna())) if len(rendered_df) else zero_stratum_counts()
    if validation is not None:
        manifest["validation"] = validation

    collection_manifest_path(output_dir).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def package_archive(output_dir: Path, archive_name: str) -> Path:
    archive_path = output_dir / archive_name
    include_paths = [
        "central",
        "BFrac",
        "BHeight",
        "candidate_tiles.csv",
        "candidate_bin_counts.csv",
        "msa_selection.csv",
        "tile_manifest.csv",
        "tile_manifest_rendered.csv",
        "collection_manifest.json",
        "msa_role_split.csv",
        "target_crs_proj4.txt",
    ]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in include_paths:
            path = output_dir / rel
            if not path.exists():
                continue
            if path.is_dir():
                for subpath in sorted(path.rglob("*")):
                    if subpath.is_file():
                        zf.write(subpath, subpath.relative_to(output_dir))
            else:
                zf.write(path, path.relative_to(output_dir))
    return archive_path


def run_render_stage(role: str, spec: DatasetSpec, args, scan_context: Optional[dict] = None) -> tuple[pd.DataFrame, dict]:
    candidate_df = load_candidate_tiles(args.output_dir)
    selected_df = load_selected_tiles(args.output_dir)
    rendered_df = render_selected_tiles(role, selected_df, args)
    validation = validate_rendered_archive(args.output_dir, selected_df)
    write_collection_manifest(
        args.output_dir,
        args,
        spec,
        scan_context or {},
        candidate_df,
        selected_df=selected_df,
        rendered_df=rendered_df,
        validation=validation,
    )
    if not args.skip_zip:
        archive_path = package_archive(args.output_dir, spec.archive_name)
        log(f"Wrote archive: {archive_path}")
    return rendered_df, validation


def main_for_role(role: str, default_output_dir: Path) -> None:
    spec = dataset_spec_for(role)
    parser = default_args_for_role(role, default_output_dir)
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.state_cache_dir = Path(args.state_cache_dir)
    args.split_path = Path(args.split_path)

    scan_context: dict = {}
    if args.stage in {"scan", "all"}:
        _, scan_context = run_scan_stage(role, spec, args)
        if args.stage == "scan":
            return

    if args.stage in {"select", "all"}:
        _, stratum_targets, per_stratum_caps = run_select_stage(spec, args, scan_context)
        scan_context = dict(scan_context)
        scan_context["selected_targets"] = stratum_targets
        scan_context["selected_per_stratum_caps"] = per_stratum_caps
        if args.stage == "select":
            return

    if args.stage in {"render", "all"}:
        run_render_stage(role, spec, args, scan_context)
