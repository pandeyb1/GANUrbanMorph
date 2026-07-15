from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-osm-context"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "osm-context-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds


THIS_DIR = Path(__file__).resolve().parent
GAN_VALIDATION_DIR = THIS_DIR.parent
OVERALL_DIR = GAN_VALIDATION_DIR / "OverallValidation"
DEFAULT_CONFIG = OVERALL_DIR / "validation_config.json"
DEFAULT_TILE_TABLE = THIS_DIR / "tables" / "representative_high_density_highrise_tiles.csv"
DEFAULT_OUT_DIR = THIS_DIR / "context_images" / "osm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-render OpenStreetMap context panels for representative GeoTIFF tiles. "
            "This is the network-dependent step; the manuscript figure script consumes the saved PNGs offline."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to OverallValidation validation_config.json")
    parser.add_argument("--overall-validation-dir", default=str(OVERALL_DIR), help="Path to OverallValidation code folder")
    parser.add_argument("--tile-table", default=str(DEFAULT_TILE_TABLE), help="Representative-tile CSV to read or update")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory where OSM PNGs will be written")
    parser.add_argument("--dataset", default="CONUSStratifiedTest", help="Dataset name in validation_config.json")
    parser.add_argument("--dataset-root", default=None, help="Optional override for the dataset root containing central/BFrac/BHeight")
    parser.add_argument("--central-dir", default=None, help="Optional override for the folder containing input GeoTIFFs")
    parser.add_argument("--refresh-selection", action="store_true", help="Recompute the representative CONUS tile table before rendering OSM panels")
    parser.add_argument("--examples-per-type", type=int, default=2, help="Used only with --refresh-selection")
    parser.add_argument("--min-active-fraction", type=float, default=0.01, help="Used only with --refresh-selection")
    parser.add_argument(
        "--unique-msas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Used only with --refresh-selection; prefer at most one selected tile from each MSA.",
    )
    parser.add_argument(
        "--min-grid-distance",
        type=float,
        default=512.0,
        help="Used only with --refresh-selection; minimum X/Y grid distance between selected examples.",
    )
    parser.add_argument("--selection-only", action="store_true", help="Refresh/write the tile table and exit without rendering OSM PNGs")
    parser.add_argument("--cache-mode", default="lazy", choices=["lazy", "all"], help="Used only with --refresh-selection")
    parser.add_argument("--zoom", default="auto", help="OSM tile zoom level, or 'auto'")
    parser.add_argument("--zoom-adjust", type=int, default=0, help="Adjustment applied when --zoom auto is used")
    parser.add_argument("--source", default="OpenStreetMap.Mapnik", help="contextily provider path, default OpenStreetMap.Mapnik")
    parser.add_argument("--wait", type=float, default=1.0, help="Seconds to wait between tile requests")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum tile-download retries")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path).resolve()


def dataset_paths(args: argparse.Namespace) -> tuple[Path, str]:
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    if args.dataset not in config["dataset_specs"]:
        raise KeyError(f"{args.dataset} is not defined in {config_path}")
    spec = config["dataset_specs"][args.dataset]
    root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else resolve_path(config_path.parent, spec["root"])
    central_dir = args.central_dir if args.central_dir else spec["bf_condition_dir"]
    return root, central_dir


def refresh_selection_table(args: argparse.Namespace) -> pd.DataFrame:
    overall_dir = Path(args.overall_validation_dir).resolve()
    ov = load_module(overall_dir / "validation_core.py", "overall_validation_core_for_osm_context")
    rep = load_module(THIS_DIR / "make_representative_high_density_highrise_examples.py", "representative_examples_for_osm_context")
    config = ov.load_config(Path(args.config).resolve())
    if args.dataset_root:
        spec = config["dataset_specs"][args.dataset]
        config["dataset_specs"][args.dataset] = ov.DatasetConfig(
            name=spec.name,
            root=Path(args.dataset_root).expanduser().resolve(),
            bf_condition_dir=spec.bf_condition_dir,
            bf_target_dir=spec.bf_target_dir,
            bh_condition_dir=spec.bh_condition_dir,
            bh_target_dir=spec.bh_target_dir,
            manifest_path=spec.manifest_path,
        )
    config["dataset_specs"] = {args.dataset: config["dataset_specs"][args.dataset]}
    config["cache_mode"] = args.cache_mode
    dataset = ov.DatasetBundle(config["dataset_specs"][args.dataset], cache_mode=str(config["cache_mode"]))
    scores = rep.tile_scores(dataset, args.dataset, ov, config)
    selected = rep.select_representative_tiles(
        scores,
        min_active_fraction=float(args.min_active_fraction),
        examples_per_type=int(args.examples_per_type),
        unique_msas=bool(args.unique_msas),
        min_grid_distance=float(args.min_grid_distance),
    )
    tile_table = Path(args.tile_table).resolve()
    tile_table.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(tile_table, index=False)
    return selected


def load_tile_table(args: argparse.Namespace) -> pd.DataFrame:
    if args.refresh_selection:
        return refresh_selection_table(args)
    tile_table = Path(args.tile_table).resolve()
    if not tile_table.exists():
        raise FileNotFoundError(f"Tile table does not exist: {tile_table}. Use --refresh-selection to create it.")
    selected = pd.read_csv(tile_table)
    if "dataset_name" in selected.columns:
        selected = selected[selected["dataset_name"] == args.dataset].copy()
    if selected.empty:
        raise ValueError(f"No rows for {args.dataset} in {tile_table}")
    return selected


def contextily_provider(provider_path: str) -> Any:
    try:
        import contextily as cx
    except ImportError as exc:
        raise ImportError(
            "This script requires contextily. Install it in the active environment with "
            "`conda install -c conda-forge contextily` or `pip install contextily`."
        ) from exc
    provider: Any = cx.providers
    for part in provider_path.split("."):
        provider = getattr(provider, part)
    return cx, provider


def geodetic_bounds(tif_path: Path) -> tuple[float, float, float, float]:
    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"{tif_path} has no CRS; cannot request an OSM basemap")
        left, bottom, right, top = transform_bounds(
            src.crs,
            "EPSG:4326",
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top,
            densify_pts=21,
        )
    return float(left), float(bottom), float(right), float(top)


def mercator_bounds(lonlat_bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    left, bottom, right, top = lonlat_bounds
    wm_left, wm_bottom, wm_right, wm_top = transform_bounds(
        "EPSG:4326",
        "EPSG:3857",
        left,
        bottom,
        right,
        top,
        densify_pts=21,
    )
    return float(wm_left), float(wm_bottom), float(wm_right), float(wm_top)


def render_osm_png(
    *,
    cx: Any,
    provider: Any,
    tif_path: Path,
    out_path: Path,
    zoom: int | str,
    zoom_adjust: int,
    wait: float,
    max_retries: int,
    dpi: int,
) -> None:
    lonlat = geodetic_bounds(tif_path)
    mercator = mercator_bounds(lonlat)
    img, extent = cx.bounds2img(
        lonlat[0],
        lonlat[1],
        lonlat[2],
        lonlat[3],
        zoom=zoom,
        source=provider,
        ll=True,
        wait=wait,
        max_retries=max_retries,
        use_cache=True,
        zoom_adjust=zoom_adjust if zoom == "auto" else None,
    )
    fig, ax = plt.subplots(figsize=(2.8, 2.8), dpi=dpi)
    ax.imshow(img, extent=extent)
    ax.set_xlim(mercator[0], mercator[2])
    ax.set_ylim(mercator[1], mercator[3])
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    selected = load_tile_table(args)
    if args.selection_only:
        print(f"Wrote/loaded {len(selected)} selected tile row(s).")
        print(Path(args.tile_table).resolve())
        print(selected.to_string(index=False))
        return 0
    root, central_dir = dataset_paths(args)
    cx, provider = contextily_provider(str(args.source))
    zoom: int | str = "auto" if str(args.zoom).lower() == "auto" else int(args.zoom)
    out_dir = Path(args.out_dir).expanduser().resolve()
    rendered = 0
    skipped = 0
    for filename in selected["filename"].astype(str):
        tif_path = root / central_dir / filename
        if not tif_path.exists():
            raise FileNotFoundError(f"Cannot find input GeoTIFF for OSM bounds: {tif_path}")
        out_path = out_dir / f"{Path(filename).stem}.png"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        render_osm_png(
            cx=cx,
            provider=provider,
            tif_path=tif_path,
            out_path=out_path,
            zoom=zoom,
            zoom_adjust=int(args.zoom_adjust),
            wait=float(args.wait),
            max_retries=int(args.max_retries),
            dpi=int(args.dpi),
        )
        rendered += 1
    print(f"Rendered {rendered} OSM context image(s); skipped {skipped} existing image(s).")
    print(out_dir)
    print("Use with: --context-source osm --osm-dir", out_dir)
    print("Figure attribution is handled by make_representative_high_density_highrise_examples.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
