from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-maintext-error-bias"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "maintext-error-bias-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
GAN_VALIDATION_DIR = THIS_DIR.parent
OVERALL_DIR = GAN_VALIDATION_DIR / "OverallValidation"
DEFAULT_TABLE = OVERALL_DIR / "results" / "tables" / "table_model_class_metric_summary_epoch1000.csv"
DEFAULT_OUT = THIS_DIR / "figures" / "model_class_error_bias_scatter_epoch1000_unet_single_latent.png"
OVERALL_FIGURE_DIR = OVERALL_DIR / "results" / "figures"

COMBINATION_COLORS = {
    ("LALegacy", "CONUSStratifiedTest"): "#0072B2",
    ("LALegacy", "SanDiegoTestNoOverlap"): "#D55E00",
    ("MSASample", "CONUSStratifiedTest"): "#009E73",
    ("MSASample", "SanDiegoTestNoOverlap"): "#CC79A7",
}
REGIME_LABELS = {
    "LALegacy": "Los Angeles Model",
    "MSASample": "CONUS Model",
}
DATASET_LABELS = {
    "CONUSStratifiedTest": "CONUS Test Dataset",
    "SanDiegoTestNoOverlap": "San Diego Test Dataset",
}
LR_MARKERS = {
    0.0001: "o",
    0.0002: "s",
    0.0005: "^",
    0.001: "D",
}
FAMILY_LABELS = {
    "1": "U-Net",
    "2A": "Single-latent cGAN",
}

PANEL_SPECS = [
    {
        "target_label": "BF",
        "quantity": "Average",
        "x_metric": "average_absolute_error",
        "y_metric": "average_bias",
        "title": "Building Footprint: Average",
        "xlabel": "MAE in average BF",
        "ylabel": "MB in average BF",
    },
    {
        "target_label": "BF",
        "quantity": "Heterogeneity",
        "x_metric": "heterogeneity_absolute_error",
        "y_metric": "heterogeneity_bias",
        "title": "Building Footprint: Heterogeneity",
        "xlabel": "MAE in BF heterogeneity",
        "ylabel": "MB in BF heterogeneity",
    },
    {
        "target_label": "BH",
        "quantity": "Average",
        "x_metric": "average_absolute_error",
        "y_metric": "average_bias",
        "title": "Building Height: Average",
        "xlabel": "MAE in average BH (m)",
        "ylabel": "MB in average BH (m)",
    },
    {
        "target_label": "BH",
        "quantity": "Heterogeneity",
        "x_metric": "heterogeneity_absolute_error",
        "y_metric": "heterogeneity_bias",
        "title": "Building Height: Heterogeneity",
        "xlabel": "MAE in BH heterogeneity (m)",
        "ylabel": "MB in BH heterogeneity (m)",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compact four-panel MAE-vs-MB figure for U-Net and single-latent cGAN at epoch 1000."
    )
    parser.add_argument("--table", default=str(DEFAULT_TABLE), help="Path to table_model_class_metric_summary_epoch1000.csv")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output PNG path")
    parser.add_argument("--families", default="1,2A", help="Comma-separated families to include")
    parser.add_argument("--checkpoint", type=int, default=1000)
    parser.add_argument("--copy-to-overall-results", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_frame(table_path: Path, families: list[str], checkpoint: int) -> pd.DataFrame:
    if not table_path.exists():
        raise FileNotFoundError(f"Missing metric table: {table_path}")
    frame = pd.read_csv(table_path, dtype={"family": str}, low_memory=False)
    frame["family"] = frame["family"].astype(str)
    frame = frame[
        frame["family"].isin(families)
        & (frame["checkpoint"].astype(int) == int(checkpoint))
    ].copy()
    if frame.empty:
        raise ValueError(f"No rows found for families={families} checkpoint={checkpoint}")
    return frame


def padded_limits(values: np.ndarray, *, include_zero: bool) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    span = upper - lower
    pad = 0.08 * span if span > 0 else max(abs(upper), abs(lower), 1.0) * 0.08
    return lower - pad, upper + pad


def marker_style(*, family: str, dataset_name: str, learning_rate: float, training_regime: str) -> dict[str, Any]:
    color = COMBINATION_COLORS.get((training_regime, dataset_name), "#777777")
    marker = LR_MARKERS.get(round(float(learning_rate), 4), "o")
    is_cgan = family == "2A"
    return {
        "marker": marker,
        "s": 54,
        "facecolor": color if is_cgan else "white",
        "edgecolor": color,
        "linewidth": 1.25,
        "alpha": 0.72 if is_cgan else 0.98,
    }


def add_panel(ax: Any, frame: pd.DataFrame, spec: dict[str, str]) -> None:
    panel = frame[frame["target_label"] == spec["target_label"]].copy()
    x_metric = spec["x_metric"]
    y_metric = spec["y_metric"]
    for _index, row in panel.iterrows():
        style = marker_style(
            family=str(row["family"]),
            dataset_name=str(row["dataset_name"]),
            learning_rate=float(row["learning_rate"]),
            training_regime=str(row["training_regime"]),
        )
        ax.scatter(
            float(row[x_metric]),
            float(row[y_metric]),
            **style,
        )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.75)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.75)
    ax.set_xlim(*padded_limits(panel[x_metric].to_numpy(dtype=float), include_zero=True))
    ax.set_ylim(*padded_limits(panel[y_metric].to_numpy(dtype=float), include_zero=True))
    ax.set_title(spec["title"], fontsize=12, loc="left")
    ax.set_xlabel(spec["xlabel"], fontsize=11)
    ax.set_ylabel(spec["ylabel"], fontsize=11)
    ax.grid(alpha=0.22)


def add_legend(fig: Any) -> None:
    combination_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor=color,
            label=f"{REGIME_LABELS.get(regime, regime)} | {DATASET_LABELS.get(dataset, dataset)}",
            markersize=7,
        )
        for (regime, dataset), color in COMBINATION_COLORS.items()
    ]
    family_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#444444",
            label="U-Net",
            markersize=7,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#777777",
            markeredgecolor="#777777",
            alpha=0.72,
            label="Single-latent cGAN",
            markersize=7,
        ),
    ]
    lr_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            color="black",
            label=f"LR {learning_rate:g}",
            markersize=6,
        )
        for learning_rate, marker in LR_MARKERS.items()
    ]
    fig.legend(
        handles=[*combination_handles, *family_handles, *lr_handles],
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=7.6,
        handletextpad=0.5,
        columnspacing=0.95,
    )


def save_figure(frame: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 8.0), constrained_layout=False)
    for ax, spec in zip(axes.flat, PANEL_SPECS, strict=True):
        add_panel(ax, frame, spec)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.94, bottom=0.17, hspace=0.36, wspace=0.30)
    add_legend(fig)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    families = parse_csv_list(str(args.families))
    frame = load_frame(Path(args.table).resolve(), families, int(args.checkpoint))
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
