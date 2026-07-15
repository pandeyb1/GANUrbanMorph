#!/usr/bin/env python
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from pyproj import CRS, Transformer
from rasterio import features
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import transform as shapely_transform


PROJECT_ROOT = Path("/Users/9oy/Documents/Projects/IM3/EvaluationP")
MA_CSV_DIR = Path("/Users/9oy/Documents/Data/IM3/Buildings/MAv1")
MSA_FRAME_PATH = PROJECT_ROOT / "Script/5_UrbanScaling/Data/MSA_notPR.gpkg"
NLCD_2015_PATH = Path(
    "/Users/9oy/Documents/Data/US/Environmental/NLCD/Annual/Annual_NLCD_LndCov_2015_CU_C1V0.tif"
)
DEFAULT_STATE_CACHE_DIR = (
    PROJECT_ROOT / "Report/GIScRSSubmission/Revision/codev3/Stratified_Archive/state_cache"
)

SAN_DIEGO_GEOID = "41740"
SAN_DIEGO_NAME = "San Diego-Carlsbad, CA"
LONLAT_CRS = CRS.from_epsg(4326)

# MSA_notPR.gpkg coordinates do not reproject cleanly under the stored CRS metadata.
# This override matches the data numerically and produces correct metro locations.
MSA_SOURCE_CRS = CRS.from_proj4(
    "+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 "
    "+datum=WGS84 +units=m +y_0=-1606786.260576 +no_defs"
)

PRECHECK_LON = -118.240979
PRECHECK_LAT = 34.895239

CACHE_COLUMNS = [
    "ID",
    "State_Abbr",
    "Height",
    "NumFloors",
    "BuildingType",
    "Standard",
    "Area",
    "Area2D",
    "height_m",
    "geom_area_m2_exact",
    "area2d_rel_err",
    "minx",
    "miny",
    "maxx",
    "maxy",
    "geometry",
]
CACHE_READ_COLUMNS = [
    "ID",
    "State_Abbr",
    "height_m",
    "geom_area_m2_exact",
    "minx",
    "miny",
    "maxx",
    "maxy",
    "geometry",
]
MA_USECOLS = {
    "ID",
    "State_Abbr",
    "Height",
    "NumFloors",
    "BuildingType",
    "Standard",
    "Area",
    "Area2D",
    "Footprint2D",
}


def get_nlcd_target_crs(nlcd_path: Path = NLCD_2015_PATH) -> tuple[CRS, str]:
    with rasterio.open(nlcd_path) as src:
        proj4 = src.crs.to_proj4()
    if not proj4:
        raise RuntimeError(f"Could not derive a proj4 target CRS from {nlcd_path}.")
    return CRS.from_proj4(proj4), proj4


def run_crs_preflight(target_crs: CRS) -> tuple[float, float]:
    transformer = Transformer.from_crs(LONLAT_CRS, target_crs, always_xy=True)
    x, y = transformer.transform(PRECHECK_LON, PRECHECK_LAT)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise RuntimeError(
            "CRS preflight failed: EPSG:4326 -> NLCD target CRS produced non-finite coordinates. "
            "Check the local PROJ/GDAL installation before building caches or archives."
        )
    return x, y


def drop_leading_unnamed_column(frame: pd.DataFrame) -> pd.DataFrame:
    if not len(frame.columns):
        return frame
    first = str(frame.columns[0])
    if first.startswith("Unnamed:"):
        return frame.iloc[:, 1:].copy()
    return frame


def parse_footprint_polygon(footprint_str: object) -> Optional[Polygon]:
    if pd.isna(footprint_str) or not isinstance(footprint_str, str):
        return None
    coords = []
    try:
        for pair in footprint_str.split("_"):
            lat_str, lon_str = pair.split("/")
            coords.append((float(lon_str), float(lat_str)))
    except Exception:
        return None
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        return Polygon(coords)
    except Exception:
        return None


def normalize_to_multipolygon(geometry):
    if geometry is None or geometry.is_empty:
        return geometry
    if geometry.geom_type == "Polygon":
        return MultiPolygon([geometry])
    if geometry.geom_type == "MultiPolygon":
        return geometry
    if geometry.geom_type == "GeometryCollection":
        polys = [geom for geom in geometry.geoms if geom.geom_type in {"Polygon", "MultiPolygon"} and not geom.is_empty]
        if not polys:
            return None
        pieces = []
        for geom in polys:
            if geom.geom_type == "Polygon":
                pieces.append(geom)
            else:
                pieces.extend(list(geom.geoms))
        return MultiPolygon(pieces) if pieces else None
    return geometry


def transform_geometries(
    geometries: Iterable,
    source_crs: CRS | str,
    target_crs: CRS,
) -> list:
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    out = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            out.append(None)
            continue
        out.append(shapely_transform(transformer.transform, geom))
    return out


def reproject_geodataframe(
    frame: gpd.GeoDataFrame,
    source_crs: CRS | str,
    target_crs: CRS,
) -> gpd.GeoDataFrame:
    projected = transform_geometries(frame.geometry.tolist(), source_crs, target_crs)
    out = frame.copy()
    out["geometry"] = projected
    return gpd.GeoDataFrame(out, geometry="geometry", crs=target_crs)


def required_state_codes(msa_name: str) -> list[str]:
    suffix = msa_name.split(",")[-1].strip()
    return [state.strip() for state in suffix.split("-") if state.strip()]


def state_cache_path(state_cache_dir: Path, state_code: str) -> Path:
    return state_cache_dir / f"{state_code}.gpkg"


def load_eligible_msas(target_crs: CRS, master_seed: int, max_msas: Optional[int]) -> gpd.GeoDataFrame:
    msa = gpd.read_file(MSA_FRAME_PATH)
    msa["GEOID"] = msa["GEOID"].astype(str)
    mask = (msa["GEOID"] != SAN_DIEGO_GEOID) & (msa["NAME"] != SAN_DIEGO_NAME)
    msa = msa.loc[mask, ["GEOID", "NAME", "geometry"]].copy()
    msa = gpd.GeoDataFrame(msa, geometry="geometry", crs=MSA_SOURCE_CRS)
    msa = reproject_geodataframe(msa, MSA_SOURCE_CRS, target_crs)
    rng = np.random.default_rng(master_seed)
    order = rng.permutation(len(msa))
    msa = msa.iloc[order].reset_index(drop=True)
    msa["order"] = np.arange(1, len(msa) + 1)
    if max_msas is not None:
        msa = msa.iloc[:max_msas].copy().reset_index(drop=True)
        msa["order"] = np.arange(1, len(msa) + 1)
    return msa


def read_state_cache_subset(
    state_cache_dir: Path,
    state_codes: Iterable[str],
    bbox: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for state_code in state_codes:
        cache_path = state_cache_path(state_cache_dir, state_code)
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing state cache for {state_code}: {cache_path}")
        frame = pyogrio.read_dataframe(
            cache_path,
            layer=state_code,
            bbox=bbox,
            columns=CACHE_READ_COLUMNS,
        )
        if len(frame):
            frames.append(frame)
    if not frames:
        return gpd.GeoDataFrame(columns=CACHE_READ_COLUMNS, geometry="geometry")
    merged = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=frames[0].crs)


def pixel_polygon(transform, row: int, col: int):
    x_left = float(transform.c + col * transform.a)
    x_right = float(x_left + transform.a)
    y_top = float(transform.f + row * transform.e)
    y_bottom = float(y_top + transform.e)
    return box(min(x_left, x_right), min(y_bottom, y_top), max(x_left, x_right), max(y_bottom, y_top))


def aggregate_exact_overlap(
    buildings: gpd.GeoDataFrame,
    transform,
    tile_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    area_sum_m2 = np.zeros((tile_size, tile_size), dtype=np.float32)
    weighted_height_sum = np.zeros((tile_size, tile_size), dtype=np.float32)
    if buildings.empty:
        return area_sum_m2, weighted_height_sum

    pixel_cache: dict[tuple[int, int], object] = {}
    pixel_area = abs(float(transform.a) * float(transform.e))

    for geometry, height_m in zip(buildings.geometry, buildings["height_m"].to_numpy(dtype=float)):
        if geometry is None or geometry.is_empty:
            continue
        mask = features.rasterize(
            [(geometry, 1)],
            out_shape=(tile_size, tile_size),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )
        rows, cols = np.nonzero(mask)
        for row, col in zip(rows.tolist(), cols.tolist()):
            key = (row, col)
            pix = pixel_cache.get(key)
            if pix is None:
                pix = pixel_polygon(transform, row, col)
                pixel_cache[key] = pix
            overlap_m2 = float(geometry.intersection(pix).area)
            if overlap_m2 <= 0.0:
                continue
            area_sum_m2[row, col] += overlap_m2
            weighted_height_sum[row, col] += overlap_m2 * float(height_m)

    bf = np.clip(area_sum_m2 / pixel_area, 0.0, 1.0).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        bh_m = np.where(area_sum_m2 > 0.0, weighted_height_sum / area_sum_m2, 0.0).astype(np.float32)
    return bf, bh_m
