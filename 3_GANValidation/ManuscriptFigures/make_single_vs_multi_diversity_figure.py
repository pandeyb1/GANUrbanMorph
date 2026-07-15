from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cgan-diversity"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "cgan-diversity-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
GAN_VALIDATION_DIR = THIS_DIR.parent
OVERALL_DIR = GAN_VALIDATION_DIR / "OverallValidation"
DEFAULT_TABLE = OVERALL_DIR / "results" / "diversity" / "overall_diversity_metrics.csv"
DEFAULT_OUT = THIS_DIR / "figures" / "single_vs_multi_cgan_diversity_epoch1000.png"
OVERALL_FIGURE_DIR = OVERALL_DIR / "results" / "figures"

TRAINING_COLORS = {
    "LALegacy": "#0072B2",
    "MSASample": "#009E73",
}
TRAINING_LABELS = {
    "LALegacy": "Los Angeles Model",
    "MSASample": "CONUS Model",
}
DATASET_LABELS = {
    "CONUSStratifiedTest": "CONUS Test Dataset",
    "SanDiegoTestNoOverlap": "San Diego Test Dataset",
}
TARGET_LABELS = {
    "BF": "Building Footprint",
    "BH": "Building Height",
}
LR_MARKERS = {
    0.0001: "o",
    0.0002: "s",
    0.0005: "^",
    0.001: "D",
}
FAMILY_ORDER = ["2A", "3"]
FAMILY_LABELS = {
    "2A": "Single-latent\ncGAN",
    "3": "Multi-latent\ncGAN",
}
DATASET_OFFSETS = {
    "LALegacy": -0.08,
    "MSASample": 0.08,
}
METRIC_SPECS = [
    ("div_pairwise_mae", "Pairwise sample diversity"),
    ("div_tile_mean_sd", "Tile-mean diversity"),
    ("div_tile_heterogeneity_sd", "Tile-heterogeneity diversity"),
]
PANEL_SPECS = [
    ("BF", "CONUSStratifiedTest"),
    ("BF", "SanDiegoTestNoOverlap"),
    ("BH", "CONUSStratifiedTest"),
    ("BH", "SanDiegoTestNoOverlap"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare single-latent and multi-latent cGAN output diversity at epoch 1000."
    )
    parser.add_argument("--table", default=str(DEFAULT_TABLE), help="Path to overall_diversity_metrics.csv")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output PNG path")
    parser.add_argument("--checkpoint", type=int, default=1000)
    parser.add_argument("--copy-to-overall-results", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_frame(table_path: Path, checkpoint: int) -> pd.DataFrame:
    if not table_path.exists():
        raise FileNotFoundError(f"Missing diversity table: {table_path}")
    frame = pd.read_csv(table_path, dtype={"family": str}, low_memory=False)
    frame["family"] = frame["family"].astype(str)
    frame["target_label"] = frame["target"].astype(str).str.upper()
    frame = frame[
        frame["family"].isin(FAMILY_ORDER)
        & (frame["checkpoint"].astype(int) == int(checkpoint))
        & (
            ((frame["target_label"] == "BF") & (frame["evaluation_mode"] == "standard"))
            | ((frame["target_label"] == "BH") & (frame["evaluation_mode"] == "oracle_bf"))
        )
    ].copy()
    if frame.empty:
        raise ValueError(f"No diversity rows found for checkpoint={checkpoint}")
    return frame


def compute_y_limits(frame: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    limits: dict[tuple[str, str], tuple[float, float]] = {}
    for metric, _label in METRIC_SPECS:
        for target_label in ("BF", "BH"):
            values = frame[frame["target_label"] == target_label][metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            lower = 0.0
            upper = float(np.max(values))
            pad = 0.10 * (upper - lower) if upper > lower else 0.01
            limits[(target_label, metric)] = (lower, upper + pad)
    return limits


def scatter_kwargs(row: pd.Series) -> dict[str, Any]:
    color = TRAINING_COLORS.get(str(row["training_regime"]), "#777777")
    marker = LR_MARKERS.get(round(float(row["learning_rate"]), 4), "o")
    return {
        "marker": marker,
        "s": 48,
        "facecolor": color,
        "edgecolor": "black",
        "linewidth": 0.35,
        "alpha": 0.82,
    }


def save_figure(frame: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_limits = compute_y_limits(frame)
    fig, axes = plt.subplots(
        len(METRIC_SPECS),
        len(PANEL_SPECS),
        figsize=(15.5, 9.6),
        constrained_layout=False,
        squeeze=False,
    )
    for row_index, (metric, ylabel) in enumerate(METRIC_SPECS):
        for col_index, (target_label, dataset_name) in enumerate(PANEL_SPECS):
            ax = axes[row_index, col_index]
            panel = frame[
                (frame["target_label"] == target_label)
                & (frame["dataset_name"] == dataset_name)
            ]
            for _idx, row in panel.iterrows():
                family = str(row["family"])
                x_value = FAMILY_ORDER.index(family) + DATASET_OFFSETS.get(str(row["training_regime"]), 0.0)
                ax.scatter(x_value, float(row[metric]), **scatter_kwargs(row))
            for training_regime, training_panel in panel.groupby("training_regime"):
                medians = training_panel.groupby("family")[metric].median().reindex(FAMILY_ORDER)
                x_values = [
                    index + DATASET_OFFSETS.get(str(training_regime), 0.0)
                    for index in range(len(FAMILY_ORDER))
                ]
                ax.scatter(
                    x_values,
                    medians.to_numpy(dtype=float),
                    color=TRAINING_COLORS.get(str(training_regime), "#777777"),
                    marker="_",
                    s=390,
                    linewidth=2.0,
                    zorder=4,
                )
            if (target_label, metric) in y_limits:
                ax.set_ylim(*y_limits[(target_label, metric)])
            ax.set_xticks(np.arange(len(FAMILY_ORDER)))
            if row_index == len(METRIC_SPECS) - 1:
                ax.set_xticklabels([FAMILY_LABELS[item] for item in FAMILY_ORDER], fontsize=11)
            else:
                ax.set_xticklabels([])
            if col_index == 0:
                ax.set_ylabel(ylabel, fontsize=12)
            title = f"{TARGET_LABELS[target_label]}\n{DATASET_LABELS[dataset_name]}"
            ax.set_title(title if row_index == 0 else "", fontsize=12)
            ax.grid(axis="y", alpha=0.25)

    training_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="black",
            label=TRAINING_LABELS.get(training_regime, training_regime),
            markersize=7,
        )
        for training_regime, color in TRAINING_COLORS.items()
    ]
    lr_handles = [
        plt.Line2D([0], [0], color="black", marker=marker, linestyle="", label=f"LR {learning_rate:g}", markersize=6)
        for learning_rate, marker in LR_MARKERS.items()
    ]
    median_handle = plt.Line2D([0], [0], color="black", marker="_", linestyle="", markersize=18, label="Median across LRs")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.91, bottom=0.14, hspace=0.22, wspace=0.20)
    fig.legend(
        handles=[*training_handles, *lr_handles, median_handle],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    frame = load_frame(Path(args.table).resolve(), int(args.checkpoint))
    out_path = Path(args.out).resolve()
    save_figure(frame, out_path)
    if args.copy_to_overall_results:
        copy_path = OVERALL_FIGURE_DIR / out_path.name
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, copy_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
