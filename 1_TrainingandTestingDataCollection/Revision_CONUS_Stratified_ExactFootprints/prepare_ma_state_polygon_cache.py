#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(THIS_DIR))

from ma_exact_footprints import (  # noqa: E402
    CACHE_COLUMNS,
    DEFAULT_STATE_CACHE_DIR,
    LONLAT_CRS,
    MA_CSV_DIR,
    MA_USECOLS,
    NLCD_2015_PATH,
    drop_leading_unnamed_column,
    get_nlcd_target_crs,
    normalize_to_multipolygon,
    parse_footprint_polygon,
    reproject_geodataframe,
    run_crs_preflight,
    state_cache_path,
)


DEFAULT_CHUNK_SIZE = 50_000


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare per-state exact-footprint caches from raw Model America v1 CSV files."
    )
    parser.add_argument("--input-dir", type=Path, default=MA_CSV_DIR)
    parser.add_argument("--state-cache-dir", type=Path, default=DEFAULT_STATE_CACHE_DIR)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--states",
        type=str,
        default=None,
        help="Optional comma-separated list of state codes to process.",
    )
    parser.add_argument(
        "--max-chunks-per-state",
        type=int,
        default=None,
        help="Optional chunk cap for partial cache construction.",
    )
    return parser.parse_args()


def selected_state_files(input_dir: Path, states_arg: str | None) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")
    requested = None
    if states_arg:
        requested = {token.strip().upper() for token in states_arg.split(",") if token.strip()}
    files = sorted(path for path in input_dir.glob("*.csv") if path.is_file())
    if requested is not None:
        files = [path for path in files if path.stem.upper() in requested]
    if not files:
        raise FileNotFoundError(f"No CSV files selected under {input_dir}")
    return files


def build_usecols(column_name: str) -> bool:
    return column_name in MA_USECOLS or column_name.startswith("Unnamed:")


def summarize_errors(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(arr):
        return {"mean": None, "median": None, "p95": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def process_chunk(chunk: pd.DataFrame, target_crs) -> tuple[gpd.GeoDataFrame, dict[str, float | int]]:
    chunk = drop_leading_unnamed_column(chunk)
    source_rows = len(chunk)
    chunk["geometry"] = chunk["Footprint2D"].map(parse_footprint_polygon)
    malformed_count = int(chunk["geometry"].isna().sum())

    chunk = chunk.loc[chunk["geometry"].notna()].copy()
    if not len(chunk):
        empty = gpd.GeoDataFrame(columns=CACHE_COLUMNS, geometry="geometry", crs=target_crs)
        stats = {
            "source_rows": source_rows,
            "malformed_count": malformed_count,
            "invalid_before": 0,
            "invalid_after": 0,
            "empty_after_repair": 0,
            "nonfinite_after_project": 0,
            "written_rows": 0,
        }
        return empty, stats

    gdf = gpd.GeoDataFrame(chunk, geometry="geometry", crs=LONLAT_CRS)
    invalid_before = int((~gdf.geometry.is_valid).sum())
    if invalid_before:
        invalid_mask = ~gdf.geometry.is_valid
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)
    gdf["geometry"] = gdf["geometry"].map(normalize_to_multipolygon)

    invalid_after_mask = (~gdf.geometry.is_valid) | gdf.geometry.is_empty | gdf.geometry.isna()
    invalid_after = int((~gdf.geometry.is_valid).sum())
    empty_after_repair = int((gdf.geometry.is_empty | gdf.geometry.isna()).sum())
    gdf = gdf.loc[~invalid_after_mask].copy()
    if not len(gdf):
        empty = gpd.GeoDataFrame(columns=CACHE_COLUMNS, geometry="geometry", crs=target_crs)
        stats = {
            "source_rows": source_rows,
            "malformed_count": malformed_count,
            "invalid_before": invalid_before,
            "invalid_after": invalid_after,
            "empty_after_repair": empty_after_repair,
            "nonfinite_after_project": 0,
            "written_rows": 0,
        }
        return empty, stats

    projected = reproject_geodataframe(gdf, LONLAT_CRS, target_crs)
    projected["geometry"] = projected["geometry"].map(normalize_to_multipolygon)
    projected_bounds = projected.geometry.bounds
    finite_mask = np.isfinite(projected_bounds.to_numpy(dtype=float)).all(axis=1)
    projected = projected.loc[finite_mask].copy()
    projected_bounds = projected.geometry.bounds
    nonfinite_after_project = int((~finite_mask).sum())

    if not len(projected):
        empty = gpd.GeoDataFrame(columns=CACHE_COLUMNS, geometry="geometry", crs=target_crs)
        stats = {
            "source_rows": source_rows,
            "malformed_count": malformed_count,
            "invalid_before": invalid_before,
            "invalid_after": invalid_after,
            "empty_after_repair": empty_after_repair,
            "nonfinite_after_project": nonfinite_after_project,
            "written_rows": 0,
        }
        return empty, stats

    projected["height_m"] = np.clip(pd.to_numeric(projected["Height"], errors="coerce") * 0.3048, 0.0, 75.0)
    projected["geom_area_m2_exact"] = projected.geometry.area.astype(float)
    area2d = pd.to_numeric(projected["Area2D"], errors="coerce")
    area2d_m2 = area2d * 0.09290304
    projected["area2d_rel_err"] = np.where(
        area2d_m2 > 0,
        np.abs(projected["geom_area_m2_exact"] - area2d_m2) / area2d_m2,
        np.nan,
    )
    projected["minx"] = projected_bounds.minx.astype(float)
    projected["miny"] = projected_bounds.miny.astype(float)
    projected["maxx"] = projected_bounds.maxx.astype(float)
    projected["maxy"] = projected_bounds.maxy.astype(float)

    projected = projected.loc[np.isfinite(projected["geom_area_m2_exact"]) & (projected["geom_area_m2_exact"] > 0)].copy()
    projected = projected.loc[np.isfinite(projected["height_m"])].copy()
    projected = projected[[column for column in CACHE_COLUMNS if column in projected.columns]].copy()

    stats = {
        "source_rows": source_rows,
        "malformed_count": malformed_count,
        "invalid_before": invalid_before,
        "invalid_after": invalid_after,
        "empty_after_repair": empty_after_repair,
        "nonfinite_after_project": nonfinite_after_project,
        "written_rows": int(len(projected)),
    }
    return projected, stats


def write_state_frame(frame: gpd.GeoDataFrame, cache_path: Path, state_code: str, append: bool) -> None:
    if not len(frame):
        return
    pyogrio.write_dataframe(
        frame,
        cache_path,
        layer=state_code,
        driver="GPKG",
        append=append,
    )


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.state_cache_dir = args.state_cache_dir.resolve()
    args.state_cache_dir.mkdir(parents=True, exist_ok=True)

    target_crs, target_proj4 = get_nlcd_target_crs(NLCD_2015_PATH)
    precheck = run_crs_preflight(target_crs)
    log(f"Target CRS preflight passed: {precheck}")

    state_files = selected_state_files(args.input_dir, args.states)
    manifest_rows = []
    summary = {
        "created_utc": now_utc(),
        "input_dir": str(args.input_dir),
        "state_cache_dir": str(args.state_cache_dir),
        "nlcd_2015": str(NLCD_2015_PATH),
        "target_proj4": target_proj4,
        "states": {},
    }

    for state_file in state_files:
        state_code = state_file.stem.upper()
        cache_path = state_cache_path(args.state_cache_dir, state_code)
        if cache_path.exists():
            cache_path.unlink()

        log(f"Processing {state_code} from {state_file.name}")
        source_row_count = 0
        cached_feature_count = 0
        dropped_malformed_empty_features = 0
        invalid_before_repair = 0
        invalid_after_repair = 0
        area_errors: list[float] = []

        append = False
        for chunk_index, chunk in enumerate(
            pd.read_csv(
                state_file,
                chunksize=args.chunk_size,
                usecols=build_usecols,
                low_memory=False,
            ),
            start=1,
        ):
            frame, stats = process_chunk(chunk, target_crs)
            write_state_frame(frame, cache_path, state_code, append=append)
            append = append or bool(len(frame))

            source_row_count += int(stats["source_rows"])
            cached_feature_count += int(stats["written_rows"])
            invalid_before_repair += int(stats["invalid_before"])
            invalid_after_repair += int(stats["invalid_after"])
            dropped_malformed_empty_features += int(stats["source_rows"]) - int(stats["written_rows"])
            if len(frame):
                finite_errors = frame["area2d_rel_err"].to_numpy(dtype=float)
                area_errors.extend(finite_errors[np.isfinite(finite_errors)].tolist())

            log(
                f"  chunk {chunk_index:04d}: source_rows={stats['source_rows']} "
                f"written={stats['written_rows']}"
            )
            if args.max_chunks_per_state is not None and chunk_index >= args.max_chunks_per_state:
                log(f"  stopping {state_code} after {chunk_index} chunks due to --max-chunks-per-state.")
                break

        error_summary = summarize_errors(area_errors)
        manifest_row = {
            "state_code": state_code,
            "source_csv": state_file.name,
            "cache_path": str(cache_path),
            "source_row_count": int(source_row_count),
            "cached_feature_count": int(cached_feature_count),
            "dropped_malformed_empty_features": int(dropped_malformed_empty_features),
            "invalid_before_repair": int(invalid_before_repair),
            "invalid_after_repair": int(invalid_after_repair),
            "area2d_rel_err_mean": error_summary["mean"],
            "area2d_rel_err_median": error_summary["median"],
            "area2d_rel_err_p95": error_summary["p95"],
        }
        manifest_rows.append(manifest_row)
        summary["states"][state_code] = manifest_row

    manifest = pd.DataFrame(manifest_rows).sort_values("state_code").reset_index(drop=True)
    manifest.to_csv(args.state_cache_dir / "state_cache_manifest.csv", index=False)
    with (args.state_cache_dir / "state_cache_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    log(f"Wrote {len(manifest_rows)} state cache summaries to {args.state_cache_dir}")


if __name__ == "__main__":
    main()
