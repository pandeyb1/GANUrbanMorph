from __future__ import annotations

import argparse
from pathlib import Path

from spatial_clustering_core import (
    build_summary_outputs,
    ensure_results_dirs,
    load_spatial_config,
    reset_metric_outputs,
    reset_summary_outputs,
    run_metrics_stage,
    snapshot_inputs,
    write_spatial_evaluation_matrix,
)


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to spatial_clustering_config.json")
    parser.add_argument("--resume", action="store_true", help="Skip rows already logged as completed")
    parser.add_argument("--training-regime", default=None, help="Comma-separated subset, e.g. LALegacy,MSASample")
    parser.add_argument("--family", default=None, help="Comma-separated subset, e.g. 1,2A,3")
    parser.add_argument("--dataset", default=None, help="Comma-separated subset of dataset names")
    parser.add_argument("--learning-rate", default=None, help="Comma-separated subset of learning rates")
    parser.add_argument("--target", default=None, help="Comma-separated subset: BF,BH")
    parser.add_argument("--evaluation-mode", default=None, help="Comma-separated subset: standard,oracle_bf,pipeline_bf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tile-level Moran's I validation for generated urban morphology rasters"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("metrics", "all"):
        add_shared_args(subparsers.add_parser(command))
    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("--config", required=True, help="Path to spatial_clustering_config.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_spatial_config(Path(args.config).resolve())
    results_dirs = ensure_results_dirs(config["results_root"])
    snapshot_inputs(config, results_dirs)
    write_spatial_evaluation_matrix(config, results_dirs)

    if args.command == "summary":
        reset_summary_outputs(results_dirs)
        build_summary_outputs(config, results_dirs)
        return 0

    if not args.resume:
        reset_metric_outputs(results_dirs)
        reset_summary_outputs(results_dirs)
        write_spatial_evaluation_matrix(config, results_dirs)

    run_metrics_stage(
        config=config,
        results_dirs=results_dirs,
        quick_tiles=None,
        resume=args.resume,
        cli_filters={
            "training_regime": args.training_regime,
            "family": args.family,
            "dataset_name": args.dataset,
            "learning_rate": args.learning_rate,
            "target": args.target,
            "evaluation_mode": args.evaluation_mode,
        },
    )

    if args.command == "all":
        build_summary_outputs(config, results_dirs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
