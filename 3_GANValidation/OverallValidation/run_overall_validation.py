from __future__ import annotations

import argparse
from pathlib import Path

from validation_core import (
    DIVERSITY_STAGE,
    POINT_STAGE,
    build_filters,
    build_summary_outputs,
    ensure_results_dirs,
    load_config,
    make_group_specs,
    reset_stage_outputs,
    reset_summary_outputs,
    run_diversity_stage,
    run_point_stage,
    snapshot_dataset_manifests,
    snapshot_runtime_environment,
    write_evaluation_matrix,
)


def add_shared_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to validation_config.json")
    parser.add_argument("--resume", action="store_true", help="Skip completed rows recorded in prior outputs")
    parser.add_argument("--training-regime", default=None, help="Comma-separated subset of training regimes")
    parser.add_argument("--family", default=None, help="Comma-separated subset of model families")
    parser.add_argument("--dataset", default=None, help="Comma-separated subset of dataset names")
    parser.add_argument("--learning-rate", default=None, help="Comma-separated subset of learning rates")
    parser.add_argument("--checkpoint", default=None, help="Comma-separated subset of checkpoints")
    parser.add_argument("--target", default=None, help="Comma-separated subset of targets: BF,BH")
    parser.add_argument("--evaluation-mode", default=None, help="Comma-separated subset of evaluation modes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible overall validation bundle for urban morphology GAN models")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("point", "diversity", "all"):
        add_shared_run_args(subparsers.add_parser(command))
    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("--config", required=True, help="Path to validation_config.json")
    return parser.parse_args()


def prepare_context(config_path: Path) -> tuple[dict, dict[str, Path], list]:
    config = load_config(config_path)
    results_dirs = ensure_results_dirs(config["results_root"])
    snapshot_runtime_environment(results_dirs, config)
    snapshot_dataset_manifests(results_dirs, config)
    groups = make_group_specs(config)
    write_evaluation_matrix(results_dirs, groups)
    return config, results_dirs, groups


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config, results_dirs, groups = prepare_context(config_path)
    if args.command == "summary":
        reset_summary_outputs(results_dirs)
        build_summary_outputs(config, results_dirs)
        return 0
    filters = build_filters(args)
    if args.command == "point":
        if not args.resume:
            reset_stage_outputs(results_dirs, POINT_STAGE)
        run_point_stage(
            config=config,
            results_dirs=results_dirs,
            groups=groups,
            quick_tiles=None,
            resume=args.resume,
            filters=filters,
        )
        return 0
    if args.command == "diversity":
        if not args.resume:
            reset_stage_outputs(results_dirs, DIVERSITY_STAGE)
        run_diversity_stage(
            config=config,
            results_dirs=results_dirs,
            groups=groups,
            quick_tiles=None,
            resume=args.resume,
            filters=filters,
        )
        return 0
    if args.command == "all":
        if not args.resume:
            reset_stage_outputs(results_dirs, POINT_STAGE)
            reset_stage_outputs(results_dirs, DIVERSITY_STAGE)
            reset_summary_outputs(results_dirs)
        run_point_stage(
            config=config,
            results_dirs=results_dirs,
            groups=groups,
            quick_tiles=None,
            resume=args.resume,
            filters=filters,
        )
        run_diversity_stage(
            config=config,
            results_dirs=results_dirs,
            groups=groups,
            quick_tiles=None,
            resume=args.resume,
            filters=filters,
        )
        build_summary_outputs(config, results_dirs)
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
