from __future__ import annotations

import shutil
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-manuscript-figures"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "manuscript-figures-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
GAN_VALIDATION_DIR = THIS_DIR.parent
OVERALL_DIR = GAN_VALIDATION_DIR / "OverallValidation"
SPATIAL_DIR = GAN_VALIDATION_DIR / "SpatialClustering" / "spatial_clustering_results_A100"
MANUSCRIPT_FIGURE_DIR = THIS_DIR / "figures"
OVERALL_FIGURE_DIR = OVERALL_DIR / "results" / "figures"
SPATIAL_FIGURE_DIR = SPATIAL_DIR / "figures"

MODEL_CLASS_TABLE = OVERALL_DIR / "results" / "tables" / "table_model_class_metric_summary_epoch1000.csv"
MORAN_TABLE = SPATIAL_DIR / "tables" / "table_morans_i_summary_epoch1000.csv"


PANEL_SPECS = [
    ("LALegacy", "BF", "Building Footprint | Los Angeles Model"),
    ("MSASample", "BF", "Building Footprint | CONUS Model"),
    ("LALegacy", "BH", "Building Heights (m) | Los Angeles Model"),
    ("MSASample", "BH", "Building Heights (m) | CONUS Model"),
]
DATASET_COLORS = {"CONUSStratifiedTest": "#315f72", "SanDiegoTestNoOverlap": "#c87533"}
DATASET_LABELS = {
    "CONUSStratifiedTest": "CONUS Test Dataset",
    "SanDiegoTestNoOverlap": "San Diego Test Dataset",
}
LR_MARKERS = {0.0001: "o", 0.0002: "s", 0.0005: "^", 0.001: "D"}
DATASET_OFFSETS = {"CONUSStratifiedTest": -0.13, "SanDiegoTestNoOverlap": 0.13}


def copy_to(path: Path, destinations: list[Path]) -> None:
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MODEL_CLASS_TABLE.exists():
        raise FileNotFoundError(f"Missing model-class summary table: {MODEL_CLASS_TABLE}")
    if not MORAN_TABLE.exists():
        raise FileNotFoundError(f"Missing Moran's I summary table: {MORAN_TABLE}")
    metric_df = pd.read_csv(MODEL_CLASS_TABLE, dtype={"family": str}, low_memory=False)
    moran_df = pd.read_csv(MORAN_TABLE, dtype={"family": str}, low_memory=False)
    for frame in (metric_df, moran_df):
        frame["family"] = frame["family"].astype(str)
        frame["target_label"] = frame["target_label"].astype(str)
    return metric_df, moran_df


def metric_needs_zero_line(metric: str) -> bool:
    lowered = metric.lower()
    return "bias" in lowered or lowered.endswith("_mbe") or lowered.endswith("_mb")


def compute_y_limits(
    frame: pd.DataFrame,
    metric_specs: list[tuple[str, str]],
    zero_line_metrics: set[str],
) -> dict[tuple[str, str], tuple[float, float]]:
    y_limits: dict[tuple[str, str], tuple[float, float]] = {}
    for metric, _ylabel in metric_specs:
        for target_label in ("BF", "BH"):
            values = frame[frame["target_label"] == target_label][metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            lower = float(np.min(values))
            upper = float(np.max(values))
            if metric in zero_line_metrics:
                lower = min(lower, 0.0)
                upper = max(upper, 0.0)
            span = upper - lower
            pad = 0.08 * span if span > 0.0 else max(abs(upper), 1.0) * 0.08
            y_limits[(target_label, metric)] = (lower - pad, upper + pad)
    return y_limits


def save_grid_figure(
    frame: pd.DataFrame,
    *,
    metric_specs: list[tuple[str, str]],
    family_order: list[str],
    family_labels: dict[str, str],
    group_labels: tuple[str, ...],
    group_label_rows: tuple[tuple[int, int], ...],
    out_path: Path,
    figsize: tuple[float, float],
) -> None:
    if frame.empty:
        raise ValueError(f"No data available for {out_path.name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    zero_line_metrics = {metric for metric, _ylabel in metric_specs if metric_needs_zero_line(metric)}
    y_limits = compute_y_limits(frame, metric_specs, zero_line_metrics)
    fig, axes = plt.subplots(
        len(metric_specs),
        len(PANEL_SPECS),
        figsize=figsize,
        constrained_layout=False,
        squeeze=False,
    )
    for row_index, (metric, ylabel) in enumerate(metric_specs):
        for col_index, (training_regime, target_label, title) in enumerate(PANEL_SPECS):
            ax = axes[row_index, col_index]
            panel = frame[
                (frame["training_regime"] == training_regime)
                & (frame["target_label"] == target_label)
            ]
            for dataset_name, dataset_panel in panel.groupby("dataset_name"):
                color = DATASET_COLORS.get(dataset_name, "#666666")
                offset = DATASET_OFFSETS.get(dataset_name, 0.0)
                for learning_rate, lr_panel in dataset_panel.groupby("learning_rate"):
                    lr_panel = lr_panel.set_index("family").reindex(family_order)
                    x_values = np.arange(len(family_order), dtype=float) + offset
                    y_values = lr_panel[metric].to_numpy(dtype=float)
                    ax.scatter(
                        x_values,
                        y_values,
                        color=color,
                        marker=LR_MARKERS.get(round(float(learning_rate), 4), "o"),
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
            if metric in zero_line_metrics:
                ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            if (target_label, metric) in y_limits:
                ax.set_ylim(*y_limits[(target_label, metric)])
            ax.set_xticks(np.arange(len(family_order)))
            if row_index == len(metric_specs) - 1:
                ax.set_xticklabels([family_labels[item] for item in family_order], fontsize=11)
            else:
                ax.set_xticklabels([])
            if col_index == 0:
                ax.set_ylabel(ylabel, fontsize=12, labelpad=28)
                ax.yaxis.set_label_coords(-0.20, 0.5)
            ax.set_title(title if row_index == 0 else "", fontsize=12)
            ax.grid(axis="y", alpha=0.25)

    dataset_handles = [
        plt.Line2D([0], [0], color=color, marker="o", linestyle="", label=DATASET_LABELS.get(dataset, dataset))
        for dataset, color in DATASET_COLORS.items()
    ]
    lr_handles = [
        plt.Line2D([0], [0], color="black", marker=marker, linestyle="", label=f"LR {learning_rate:g}")
        for learning_rate, marker in LR_MARKERS.items()
    ]
    median_handle = plt.Line2D([0], [0], color="black", marker="_", linestyle="", markersize=18, label="Median across LRs")
    bottom = 0.22 if len(metric_specs) <= 2 else 0.12 if len(metric_specs) <= 4 else 0.075
    top = 0.90 if len(metric_specs) <= 4 else 0.94
    fig.subplots_adjust(left=0.095, right=0.995, top=top, bottom=bottom, hspace=0.22, wspace=0.20)
    for group_label, (start_row, end_row) in zip(group_labels, group_label_rows, strict=False):
        group_y = 0.5 * (axes[start_row, 0].get_position().y1 + axes[end_row, 0].get_position().y0)
        fig.text(0.018, group_y, group_label, rotation=90, ha="center", va="center", fontsize=13, fontweight="bold")
    fig.legend(
        handles=[*dataset_handles, *lr_handles, median_handle],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_unet_single_metric_figure(metric_df: pd.DataFrame) -> Path:
    frame = metric_df[metric_df["family"].isin(["1", "2A"])].copy()
    out_path = MANUSCRIPT_FIGURE_DIR / "model_class_comparison_epoch1000_unet_single_latent.png"
    save_grid_figure(
        frame,
        metric_specs=[
            ("average_absolute_error", "Mean Absolute Error"),
            ("average_bias", "Mean Bias"),
            ("heterogeneity_absolute_error", "Mean Absolute Error"),
            ("heterogeneity_bias", "Mean Bias"),
        ],
        family_order=["1", "2A"],
        family_labels={"1": "U-Net", "2A": "Single-latent\ncGAN"},
        group_labels=("Average", "Heterogeneity"),
        group_label_rows=((0, 1), (2, 3)),
        out_path=out_path,
        figsize=(15.5, 10.8),
    )
    copy_to(out_path, [OVERALL_FIGURE_DIR / out_path.name])
    return out_path


def save_unet_single_moran_figure(moran_df: pd.DataFrame) -> Path:
    frame = moran_df[moran_df["family"].isin(["1", "2A"])].copy()
    out_path = MANUSCRIPT_FIGURE_DIR / "morans_i_model_comparison_epoch1000_unet_single_latent.png"
    save_grid_figure(
        frame,
        metric_specs=[
            ("moran_mae", "Moran's I Mean Absolute Error"),
            ("moran_mbe", "Moran's I Mean Bias"),
        ],
        family_order=["1", "2A"],
        family_labels={"1": "U-Net", "2A": "Single-latent\ncGAN"},
        group_labels=(),
        group_label_rows=(),
        out_path=out_path,
        figsize=(15.5, 6.2),
    )
    copy_to(out_path, [SPATIAL_FIGURE_DIR / out_path.name])
    return out_path


def save_single_multi_combined_figure(metric_df: pd.DataFrame, moran_df: pd.DataFrame) -> Path:
    keys = ["training_regime", "target_label", "dataset_name", "family", "learning_rate", "checkpoint"]
    moran_subset = moran_df[keys + ["moran_mae", "moran_mbe"]].copy()
    frame = metric_df.merge(moran_subset, on=keys, how="left")
    frame = frame[frame["family"].isin(["2A", "3"])].copy()
    out_path = MANUSCRIPT_FIGURE_DIR / "single_vs_multi_cgan_error_bias_epoch1000.png"
    save_grid_figure(
        frame,
        metric_specs=[
            ("average_absolute_error", "Mean Absolute Error"),
            ("average_bias", "Mean Bias"),
            ("heterogeneity_absolute_error", "Mean Absolute Error"),
            ("heterogeneity_bias", "Mean Bias"),
            ("moran_mae", "Moran's I Mean Absolute Error"),
            ("moran_mbe", "Moran's I Mean Bias"),
        ],
        family_order=["2A", "3"],
        family_labels={"2A": "Single-latent\ncGAN", "3": "Multi-latent\ncGAN"},
        group_labels=("Average", "Heterogeneity", "Spatial Clustering"),
        group_label_rows=((0, 1), (2, 3), (4, 5)),
        out_path=out_path,
        figsize=(15.5, 15.2),
    )
    copy_to(out_path, [OVERALL_FIGURE_DIR / out_path.name, SPATIAL_FIGURE_DIR / out_path.name])
    return out_path


def main() -> int:
    MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metric_df, moran_df = load_inputs()
    outputs = [
        save_unet_single_metric_figure(metric_df),
        save_unet_single_moran_figure(moran_df),
        save_single_multi_combined_figure(metric_df, moran_df),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
