from __future__ import annotations

try:
    import common as _common
except ImportError:
    from analysis import common as _common

_DEFAULT_REAL_RESULTS_DIR = _common.DEFAULT_REAL_RESULTS_DIR
_default_output_dir = _common.default_output_dir
_output_dir_for_run = _common.output_dir_for_run
_list_analysis_run_folders = _common.list_analysis_run_folders
_parse_run_folder_metadata = _common.parse_run_folder_metadata
_run_folder_output_id = _common.run_folder_output_id
_common_find_examples_xlsx = _common.find_examples_xlsx
_common_find_step_level_xlsx = _common.find_step_level_xlsx
_resolve_repo_root = _common.resolve_repo_root
_resolve_real_results_root = _common.resolve_real_results_root
"""
    python analyze_quantitative2_project.py --repo-root .
    python analyze_quantitative2_project.py --repo-root . --model ds --prompt b
"""

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress, spearmanr

try:
    from minepy import MINE
    HAVE_MINEPY = True
except Exception:
    HAVE_MINEPY = False


DATASET_LABELS = _common.DATASET_LABELS

MODEL_LABELS = _common.MODEL_FULL_LABELS

PROMPT_LABELS = _common.PROMPT_LABELS

METHODS = _common.METHODS_FIXED_COLS_TITLE

METHOD_ORDER = list(METHODS.keys())
METHOD_MARKERS = {
    "RCC": "o",
    "Sampling": "s",
    "DeepConf": "^",
}

sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.85,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PLOT_DPI = 220


# ============================================================
# Args
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir("quantitative2_project_plots"))
    parser.add_argument("--model", type=str, default=None, choices=["ds_8b", "qwq_32b", "ds", "qwq"])
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--faith-threshold", type=float, default=0.5)
    return parser.parse_args()


# ============================================================
# Basic helpers
# ============================================================

def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


TRUE_STRINGS = _common.TRUE_STRINGS
FALSE_STRINGS = _common.FALSE_STRINGS


def parse_correct(x: Any) -> Optional[float]:
    if x is None:
        return None

    if isinstance(x, bool):
        return float(x)

    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x >= 0.5)

    s = str(x).strip().lower()

    if s in TRUE_STRINGS:
        return 1.0

    if s in FALSE_STRINGS:
        return 0.0

    return None


def list_run_folders(
    repo_root: Path,
    model_filter: Optional[str],
    prompt_filter: Optional[str],
) -> List[Path]:
    return _list_analysis_run_folders(
        repo_root=repo_root,
        real_results_dir=_DEFAULT_REAL_RESULTS_DIR,
        model_filter=model_filter,
        prompt_filter=prompt_filter,
        require_examples=True,
    )


def parse_run_metadata(folder: Path | str) -> Tuple[str, str, str, str, str, str]:
    meta = _parse_run_folder_metadata(folder)
    return (
        meta["dataset_key"],
        meta["model_key"],
        meta["prompt_key"],
        meta["dataset"],
        meta["model_full"],
        meta["prompt"],
    )


def find_examples_xlsx(folder: Path) -> Path:
    candidates = sorted(folder.glob("*.xlsx"))
    candidates = [p for p in candidates if "examples" in p.name.lower()]

    if not candidates:
        raise FileNotFoundError(f"No *examples*.xlsx found in {folder}")

    return candidates[0]


# ============================================================
# Association metrics
# ============================================================

def compute_assoc(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)

    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    if len(x_arr) < 5:
        return np.nan

    if np.allclose(np.std(x_arr), 0) or np.allclose(np.std(y_arr), 0):
        return np.nan

    if HAVE_MINEPY:
        try:
            mine = MINE(alpha=0.6, c=15)
            mine.compute_score(x_arr, y_arr)
            return float(mine.mic())
        except Exception:
            pass

    try:
        rho = spearmanr(x_arr, y_arr, nan_policy="omit").correlation
        return float(abs(rho)) if rho is not None and np.isfinite(rho) else np.nan
    except Exception:
        return np.nan


def load_run_dataframe(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df = df.copy()

    df["correct_num"] = df["correct"].apply(parse_correct)

    faith_cols = [
        "faithfulness_rcc",
        "faithfulness_sampling",
        "faithfulness_deepconf",
    ]

    existing_faith_cols = [c for c in faith_cols if c in df.columns]

    if existing_faith_cols:
        df["faithfulness_mean"] = df[existing_faith_cols].mean(axis=1, skipna=True)
    else:
        df["faithfulness_mean"] = np.nan

    return df


# ============================================================
# Data collection
# ============================================================

def collect_combo_metrics(
    run_folders: Sequence[Path],
    faith_threshold: float,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Returns mapping:
        (model_key, prompt_key) -> dict with per-dataset metric rows.
    """

    combos: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for folder in run_folders:
        (
            dataset_key,
            model_key,
            prompt_key,
            dataset_label,
            model_label,
            prompt_label,
        ) = parse_run_metadata(folder)

        xlsx_path = find_examples_xlsx(folder)
        df = load_run_dataframe(xlsx_path)

        combo_key = (model_key, prompt_key)

        if combo_key not in combos:
            combos[combo_key] = {
                "model_key": model_key,
                "prompt_key": prompt_key,
                "model_label": model_label,
                "prompt_label": prompt_label,
                "rows": [],
            }

        row: Dict[str, Any] = {
            "dataset_key": dataset_key,
            "dataset_label": dataset_label,
            "folder": folder.name,
            "accuracy": (
                float(df["correct_num"].dropna().mean())
                if df["correct_num"].notna().any()
                else np.nan
            ),
            "mean_faithfulness": (
                float(df["faithfulness_mean"].dropna().mean())
                if df["faithfulness_mean"].notna().any()
                else np.nan
            ),
        }

        for method_name, meta in METHODS.items():
            if meta["conf_col"] not in df.columns or meta["faith_col"] not in df.columns:
                row[f"assoc_shared__{method_name}"] = np.nan
                row[f"assoc_correct__{method_name}"] = np.nan
                row[f"assoc_incorrect__{method_name}"] = np.nan
                row[f"assoc_faithful__{method_name}"] = np.nan
                row[f"assoc_unfaithful__{method_name}"] = np.nan
                row[f"mean_signal_faith__{method_name}"] = np.nan
                row[f"n_shared__{method_name}"] = 0
                row[f"n_correct__{method_name}"] = 0
                row[f"n_incorrect__{method_name}"] = 0
                row[f"n_faithful__{method_name}"] = 0
                row[f"n_unfaithful__{method_name}"] = 0
                continue

            conf = pd.to_numeric(df[meta["conf_col"]], errors="coerce")
            faith = pd.to_numeric(df[meta["faith_col"]], errors="coerce")
            corr = pd.to_numeric(df["correct_num"], errors="coerce")

            valid_shared = conf.notna() & faith.notna()
            valid_correct = valid_shared & (corr == 1)
            valid_incorrect = valid_shared & (corr == 0)
            faithful_mask = valid_shared & (faith >= faith_threshold) & corr.notna()
            unfaithful_mask = valid_shared & (faith < faith_threshold) & corr.notna()

            row[f"assoc_shared__{method_name}"] = compute_assoc(conf[valid_shared], faith[valid_shared])
            row[f"assoc_correct__{method_name}"] = compute_assoc(conf[valid_correct], faith[valid_correct])
            row[f"assoc_incorrect__{method_name}"] = compute_assoc(conf[valid_incorrect], faith[valid_incorrect])
            row[f"assoc_faithful__{method_name}"] = compute_assoc(conf[faithful_mask], corr[faithful_mask])
            row[f"assoc_unfaithful__{method_name}"] = compute_assoc(conf[unfaithful_mask], corr[unfaithful_mask])

            row[f"mean_signal_faith__{method_name}"] = (
                float(faith[valid_shared].mean()) if valid_shared.any() else np.nan
            )

            row[f"n_shared__{method_name}"] = int(valid_shared.sum())
            row[f"n_correct__{method_name}"] = int(valid_correct.sum())
            row[f"n_incorrect__{method_name}"] = int(valid_incorrect.sum())
            row[f"n_faithful__{method_name}"] = int(faithful_mask.sum())
            row[f"n_unfaithful__{method_name}"] = int(unfaithful_mask.sum())

        combos[combo_key]["rows"].append(row)

    return combos


# ============================================================
# Table helpers
# ============================================================

def rows_to_matrix(rows_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [f"{prefix}__{m}" for m in METHODS]
    matrix = rows_df.set_index("dataset_label")[cols].copy()
    matrix.columns = list(METHODS.keys())
    return matrix


def mean_signal_faith_df(rows_df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"mean_signal_faith__{m}" for m in METHODS]
    matrix = rows_df.set_index("dataset_label")[cols].copy()
    matrix.columns = list(METHODS.keys())
    return matrix


# ============================================================
# Plot helpers
# ============================================================

def save_plot(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")

    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")

    plt.close(fig)


def finite_min_max(values: Sequence[float]) -> Optional[Tuple[float, float]]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return None

    return float(arr.min()), float(arr.max())


def axis_limits_for_points(
    x_values: Sequence[float],
    y_values: Sequence[float],
    diagonal: bool = False,
    force_unit_if_wide: bool = True,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Dynamic limits with padding.

    This fixes the visual issue where labels/points close to 0 or 1
    can be clipped, while still keeping the plot interpretable.
    """

    x_mm = finite_min_max(x_values)
    y_mm = finite_min_max(y_values)

    if x_mm is None or y_mm is None:
        return (0.0, 1.0), (0.0, 1.0)

    xmin, xmax = x_mm
    ymin, ymax = y_mm

    if diagonal:
        lo = min(xmin, ymin)
        hi = max(xmax, ymax)
    else:
        lo_x, hi_x = xmin, xmax
        lo_y, hi_y = ymin, ymax

        def pad_interval(lo: float, hi: float) -> Tuple[float, float]:
            if math.isclose(lo, hi):
                pad = 0.08
            else:
                pad = max(0.06, 0.18 * (hi - lo))

            return max(0.0, lo - pad), min(1.0, hi + pad)

        xlim = pad_interval(lo_x, hi_x)
        ylim = pad_interval(lo_y, hi_y)

        if force_unit_if_wide:
            if xlim[1] - xlim[0] > 0.75:
                xlim = (0.0, 1.0)
            if ylim[1] - ylim[0] > 0.75:
                ylim = (0.0, 1.0)

        return xlim, ylim

    if math.isclose(lo, hi):
        pad = 0.08
    else:
        pad = max(0.06, 0.18 * (hi - lo))

    lim = (max(0.0, lo - pad), min(1.0, hi + pad))

    if force_unit_if_wide and lim[1] - lim[0] > 0.75:
        lim = (0.0, 1.0)

    return lim, lim


def annotate_point_safely(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    """
    Adds labels while reducing clipping near the plot boundaries.
    """

    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]

    dx = 0.018 * x_span
    dy = 0.018 * y_span

    ha = "left"
    va = "bottom"

    x_text = x + dx
    y_text = y + dy

    if x > xlim[1] - 0.12 * x_span:
        x_text = x - dx
        ha = "right"

    if y > ylim[1] - 0.12 * y_span:
        y_text = y - dy
        va = "top"

    ax.text(
        x_text,
        y_text,
        text,
        fontsize=9,
        ha=ha,
        va=va,
        color="#333333",
        clip_on=True,
    )


def add_missing_methods_note(
    ax: plt.Axes,
    missing_methods: Sequence[str],
    reason: str = "not enough valid data",
) -> None:
    if not missing_methods:
        return

    missing_text = "Missing: " + ", ".join(missing_methods)
    ax.text(
        0.02,
        0.03,
        missing_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#666666",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.85,
        ),
    )


def add_regression_annotation(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    loc: str = "tl",
) -> None:
    valid = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    x_valid = x.to_numpy(dtype=float)[valid]
    y_valid = y.to_numpy(dtype=float)[valid]

    if len(x_valid) < 2:
        return

    if np.allclose(np.std(x_valid), 0) or np.allclose(np.std(y_valid), 0):
        return

    try:
        slope, intercept, r_value, p_value, std_err = linregress(x_valid, y_valid)
    except Exception:
        return

    if loc == "tr":
        xy = (0.95, 0.95)
        ha = "right"
    else:
        xy = (0.05, 0.95)
        ha = "left"

    ax.text(
        xy[0],
        xy[1],
        f"R²={r_value**2:.2f}\np={p_value:.3f}",
        transform=ax.transAxes,
        va="top",
        ha=ha,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.8),
    )


# ============================================================
# Plots
# ============================================================

def plot_heatmap_shared(
    shared_df: pd.DataFrame,
    title_suffix: str,
    metric_name: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    cmap = sns.color_palette("crest", as_cmap=True)

    sns.heatmap(
        shared_df,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": metric_name},
    )

    ax.set_title(f"{metric_name} between confidence and faithfulness\n{title_suffix}")
    ax.set_xlabel("Method")
    ax.set_ylabel("Dataset")

    fig.tight_layout()
    save_plot(fig, out_path)


def plot_violin_by_dataset(
    values_df: pd.DataFrame,
    accuracy_map: pd.Series,
    mean_f_map: pd.Series,
    title: str,
    out_path: Path,
) -> None:
    long_df = (
        values_df
        .reset_index()
        .melt(id_vars="dataset_label", var_name="method", value_name="score")
        .dropna()
    )

    if long_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5))

    sns.violinplot(
        data=long_df,
        x="dataset_label",
        y="score",
        ax=ax,
        inner="quartile",
        color="#86B6C6",
        cut=0,
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Association score")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)

    ax2 = ax.twinx()
    xs = np.arange(len(values_df.index))

    ax2.set_ylim(0, 1)
    ax2.plot(
        xs,
        accuracy_map.reindex(values_df.index).values,
        "o",
        color="black",
        label="Accuracy",
        zorder=5,
    )
    ax2.plot(
        xs,
        mean_f_map.reindex(values_df.index).values,
        "*",
        color="black",
        markersize=10,
        label="Mean faithfulness",
        zorder=5,
    )
    ax2.set_ylabel("Accuracy / Mean faithfulness")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    save_plot(fig, out_path)


def make_assoc_scatter_df(
    rows_df: pd.DataFrame,
    prefix_a: str,
    prefix_b: str,
    keep_missing: bool = True,
) -> pd.DataFrame:
    """
    Creates scatter records.

    keep_missing=True keeps rows with NaN x/y so per-dataset plots can report
    which methods were unavailable instead of silently dropping them.
    """

    records = []

    for _, row in rows_df.iterrows():
        for method_name in METHODS:
            a = row.get(f"{prefix_a}__{method_name}", np.nan)
            b = row.get(f"{prefix_b}__{method_name}", np.nan)

            if not keep_missing and (pd.isna(a) or pd.isna(b)):
                continue

            records.append({
                "dataset": row["dataset_label"],
                "method": method_name,
                "x": float(a) if pd.notna(a) else np.nan,
                "y": float(b) if pd.notna(b) else np.nan,
            })

    return pd.DataFrame(records)


def plot_scatter_global(
    scatter_df: pd.DataFrame,
    xlab: str,
    ylab: str,
    title: str,
    out_path: Path,
    diagonal: bool = False,
) -> None:
    if scatter_df.empty:
        return

    valid_df = scatter_df.dropna(subset=["x", "y"]).copy()

    if valid_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    sns.scatterplot(
        data=valid_df,
        x="x",
        y="y",
        hue="dataset",
        style="method",
        s=90,
        alpha=0.85,
        ax=ax,
    )

    xlim, ylim = axis_limits_for_points(valid_df["x"], valid_df["y"], diagonal=diagonal)

    if diagonal:
        lo = min(xlim[0], ylim[0])
        hi = max(xlim[1], ylim[1])
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        xlim = (lo, hi)
        ylim = (lo, hi)
    else:
        if len(valid_df) >= 2:
            sns.regplot(
                data=valid_df,
                x="x",
                y="y",
                scatter=False,
                ax=ax,
                color="black",
                line_kws={"linewidth": 1.5},
            )
            add_regression_annotation(ax, valid_df["x"], valid_df["y"], loc="tl")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)

    fig.tight_layout()
    save_plot(fig, out_path)


def plot_scatter_per_dataset(
    scatter_df: pd.DataFrame,
    xlab: str,
    ylab: str,
    title: str,
    out_path: Path,
    diagonal: bool = False,
) -> None:
    """
    Fixed version.

    Improvements:
      - Uses dynamic axis limits per dataset with padding.
      - Does not clip labels close to boundaries.
      - Keeps a note for methods that cannot be plotted because x/y is NaN.
      - Uses method-specific colors consistently.
    """

    if scatter_df.empty:
        return

    datasets = list(scatter_df["dataset"].drop_duplicates())

    if not datasets:
        return

    n = len(datasets)
    n_cols = 2 if n > 1 else 1
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.1 * n_cols, 5.8 * n_rows),
        sharex=False,
        sharey=False,
    )

    axes = np.atleast_1d(axes).ravel()

    for ax, dataset in zip(axes, datasets):
        ds_all = scatter_df[scatter_df["dataset"] == dataset].copy()
        ds_valid = ds_all.dropna(subset=["x", "y"]).copy()

        missing_methods = [
            method
            for method in METHOD_ORDER
            if method in ds_all["method"].values
            and ds_all.loc[ds_all["method"] == method, ["x", "y"]].isna().any(axis=None)
        ]

        if ds_valid.empty:
            ax.set_title(dataset, fontsize=15)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            if diagonal:
                ax.plot([0, 1], [0, 1], "k--", linewidth=1)

            ax.text(
                0.5,
                0.5,
                "No valid method points",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="#666666",
            )
            add_missing_methods_note(ax, missing_methods)
            continue

        xlim, ylim = axis_limits_for_points(ds_valid["x"], ds_valid["y"], diagonal=diagonal)

        if diagonal:
            lo = min(xlim[0], ylim[0])
            hi = max(xlim[1], ylim[1])
            xlim = (lo, hi)
            ylim = (lo, hi)

        for method_name in METHOD_ORDER:
            row = ds_valid[ds_valid["method"] == method_name]

            if row.empty:
                continue

            x = float(row.iloc[0]["x"])
            y = float(row.iloc[0]["y"])

            ax.scatter(
                x,
                y,
                s=120,
                color=METHODS[method_name]["color"],
                marker=METHOD_MARKERS.get(method_name, "o"),
                edgecolor="white",
                linewidth=0.8,
                alpha=0.95,
                label=method_name,
                zorder=4,
            )

            annotate_point_safely(ax, x, y, method_name, xlim, ylim)

        if diagonal:
            ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], "k--", linewidth=1, alpha=0.8)
        else:
            if len(ds_valid) >= 2:
                x_vals = ds_valid["x"].to_numpy(dtype=float)
                y_vals = ds_valid["y"].to_numpy(dtype=float)

                if not np.allclose(np.std(x_vals), 0) and not np.allclose(np.std(y_vals), 0):
                    sns.regplot(
                        data=ds_valid,
                        x="x",
                        y="y",
                        scatter=False,
                        ax=ax,
                        color="black",
                        line_kws={"linewidth": 1.2, "alpha": 0.75},
                    )
                    add_regression_annotation(ax, ds_valid["x"], ds_valid["y"], loc="tl")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.set_title(dataset, fontsize=15)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        add_missing_methods_note(ax, missing_methods)

    for ax in axes[n:]:
        ax.set_visible(False)

    handles = []
    labels = []

    for method_name in METHOD_ORDER:
        handle = plt.Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS.get(method_name, "o"),
            color="none",
            markerfacecolor=METHODS[method_name]["color"],
            markeredgecolor="white",
            markersize=9,
            label=method_name,
        )
        handles.append(handle)
        labels.append(method_name)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(METHOD_ORDER),
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(title, fontsize=18, y=0.995)

    fig.tight_layout(rect=[0, 0.055, 1, 0.965])
    save_plot(fig, out_path)


def plot_split_violin(
    rows_df: pd.DataFrame,
    prefix_left: str,
    prefix_right: str,
    left_label: str,
    right_label: str,
    overlay_series: pd.Series,
    overlay_label: str,
    title: str,
    out_path: Path,
) -> None:
    records = []

    for _, row in rows_df.iterrows():
        for method_name in METHODS:
            a = row.get(f"{prefix_left}__{method_name}")
            b = row.get(f"{prefix_right}__{method_name}")

            if pd.notna(a):
                records.append({
                    "dataset": row["dataset_label"],
                    "value": float(a),
                    "condition": left_label,
                })

            if pd.notna(b):
                records.append({
                    "dataset": row["dataset_label"],
                    "value": float(b),
                    "condition": right_label,
                })

    long_df = pd.DataFrame(records)

    if long_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5))

    palette = {
        left_label: "cornflowerblue",
        right_label: "lightcoral",
    }

    sns.violinplot(
        data=long_df,
        x="dataset",
        y="value",
        hue="condition",
        split=True,
        palette=palette,
        ax=ax,
        cut=0,
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Association score")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ds_order = [t.get_text() for t in ax.get_xticklabels()]
    xs = np.arange(len(ds_order))

    ax2.set_ylim(0, 1)
    ax2.plot(
        xs,
        overlay_series.reindex(ds_order).values,
        ".",
        color="black",
        markersize=10,
        label=overlay_label,
        zorder=5,
    )
    ax2.set_ylabel(overlay_label)
    ax2.legend(loc="lower right")

    fig.tight_layout()
    save_plot(fig, out_path)


def plot_shared_vs_mf(
    rows_df: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    records = []

    for _, row in rows_df.iterrows():
        for method_name in METHODS:
            x = row.get(f"assoc_shared__{method_name}")
            y = row.get(f"mean_signal_faith__{method_name}")

            if pd.isna(x) or pd.isna(y):
                continue

            records.append({
                "dataset": row["dataset_label"],
                "method": method_name,
                "assoc": float(x),
                "mf": float(y),
            })

    df = pd.DataFrame(records)

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    sns.scatterplot(
        data=df,
        x="assoc",
        y="mf",
        hue="dataset",
        style="method",
        s=90,
        alpha=0.85,
        ax=ax,
    )

    if len(df) >= 2:
        sns.regplot(
            data=df,
            x="assoc",
            y="mf",
            scatter=False,
            ax=ax,
            color="black",
            line_kws={"linewidth": 1.5},
        )
        add_regression_annotation(ax, df["assoc"], df["mf"], loc="tr")

    xlim, ylim = axis_limits_for_points(df["assoc"], df["mf"], diagonal=False)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Association score (confidence vs faithfulness)")
    ax.set_ylabel("Mean faithfulness")
    ax.set_title(title)

    fig.tight_layout()
    save_plot(fig, out_path)


# ============================================================
# Saving
# ============================================================

def save_metric_tables(rows_df: pd.DataFrame, out_dir: Path) -> None:
    rows_df.to_csv(out_dir / "per_dataset_metrics.csv", index=False)

    rows_to_matrix(rows_df, "assoc_shared").to_csv(out_dir / "assoc_shared_matrix.csv")
    rows_to_matrix(rows_df, "assoc_correct").to_csv(out_dir / "assoc_correct_matrix.csv")
    rows_to_matrix(rows_df, "assoc_incorrect").to_csv(out_dir / "assoc_incorrect_matrix.csv")
    rows_to_matrix(rows_df, "assoc_faithful").to_csv(out_dir / "assoc_faithful_matrix.csv")
    rows_to_matrix(rows_df, "assoc_unfaithful").to_csv(out_dir / "assoc_unfaithful_matrix.csv")
    mean_signal_faith_df(rows_df).to_csv(out_dir / "mean_signal_faith_matrix.csv")


# ============================================================
# Main per-combo runner
# ============================================================

def run_one_combo(
    combo: Dict[str, Any],
    output_root: Path,
    faith_threshold: float,
) -> None:
    rows_df = pd.DataFrame(combo["rows"]).sort_values("dataset_label")

    out_dir = output_root / f"{combo['model_key']}__{combo['prompt_key']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_name = "MIC" if HAVE_MINEPY else "|Spearman|"
    title_suffix = f"{combo['model_label']} | {combo['prompt_label']}"

    save_metric_tables(rows_df, out_dir)

    shared_df = rows_to_matrix(rows_df, "assoc_shared")

    accuracy_map = rows_df.set_index("dataset_label")["accuracy"]
    mean_f_map = rows_df.set_index("dataset_label")["mean_faithfulness"]

    plot_heatmap_shared(
        shared_df,
        title_suffix,
        metric_name,
        out_dir / "2_assoc_per_dataset_heatmap.png",
    )

    plot_violin_by_dataset(
        shared_df,
        accuracy_map,
        mean_f_map,
        title=f"{metric_name} distribution across methods by dataset\n{title_suffix}",
        out_path=out_dir / "3_assoc_distribution_by_dataset_violin.png",
    )

    scatter_ci = make_assoc_scatter_df(
        rows_df,
        "assoc_correct",
        "assoc_incorrect",
        keep_missing=True,
    )

    plot_scatter_global(
        scatter_ci,
        xlab=f"{metric_name} on incorrect examples",
        ylab=f"{metric_name} on correct examples",
        title=f"Correct vs Incorrect association\n{title_suffix}",
        out_path=out_dir / "4_assoc_correct_vs_incorrect_scatter.png",
        diagonal=False,
    )

    plot_scatter_per_dataset(
        scatter_ci,
        xlab=f"{metric_name} incorrect",
        ylab=f"{metric_name} correct",
        title=f"Correct vs Incorrect association per dataset\n{title_suffix}",
        out_path=out_dir / "4b_assoc_correct_vs_incorrect_per_dataset_reg.png",
        diagonal=False,
    )

    plot_scatter_per_dataset(
        scatter_ci,
        xlab=f"{metric_name} incorrect",
        ylab=f"{metric_name} correct",
        title=f"Correct vs Incorrect association per dataset\n{title_suffix}",
        out_path=out_dir / "4c_assoc_correct_vs_incorrect_per_dataset_diag.png",
        diagonal=True,
    )

    plot_split_violin(
        rows_df,
        "assoc_correct",
        "assoc_incorrect",
        "correct",
        "incorrect",
        accuracy_map,
        "Accuracy",
        title=f"Association by correctness split\n{title_suffix}",
        out_path=out_dir / "4d_assoc_by_correctness_violin.png",
    )

    scatter_fu = make_assoc_scatter_df(
        rows_df,
        "assoc_faithful",
        "assoc_unfaithful",
        keep_missing=True,
    )

    plot_scatter_global(
        scatter_fu,
        xlab=f"{metric_name} on unfaithful examples",
        ylab=f"{metric_name} on faithful examples",
        title=f"Faithful vs Unfaithful association\n{title_suffix}",
        out_path=out_dir / "5_assoc_faithful_vs_unfaithful_scatter.png",
        diagonal=False,
    )

    plot_scatter_per_dataset(
        scatter_fu,
        xlab=f"{metric_name} unfaithful",
        ylab=f"{metric_name} faithful",
        title=f"Faithful vs Unfaithful association per dataset\n{title_suffix}",
        out_path=out_dir / "5b_assoc_faithful_vs_unfaithful_per_dataset_reg.png",
        diagonal=False,
    )

    plot_scatter_per_dataset(
        scatter_fu,
        xlab=f"{metric_name} unfaithful",
        ylab=f"{metric_name} faithful",
        title=f"Faithful vs Unfaithful association per dataset\n{title_suffix}",
        out_path=out_dir / "5c_assoc_faithful_vs_unfaithful_per_dataset_diag.png",
        diagonal=True,
    )

    plot_split_violin(
        rows_df,
        "assoc_faithful",
        "assoc_unfaithful",
        "faithful",
        "unfaithful",
        mean_f_map,
        "Mean faithfulness",
        title=f"Association by faithfulness split\n{title_suffix}",
        out_path=out_dir / "5d_assoc_by_faithfulness_violin.png",
    )

    plot_shared_vs_mf(
        rows_df,
        title=f"Shared association vs mean faithfulness\n{title_suffix}",
        out_path=out_dir / "7_assoc_vs_mean_faithfulness_scatter.png",
    )

    info = {
        "model": combo["model_label"],
        "prompt": combo["prompt_label"],
        "faith_threshold": faith_threshold,
        "association_metric": (
            "MIC"
            if HAVE_MINEPY
            else "absolute Spearman correlation (minepy unavailable)"
        ),
        "note": (
            "Per-dataset scatter plots keep missing methods in the data and show them "
            "as notes when an association score cannot be computed because of too few "
            "examples or constant values after splitting."
        ),
    }

    pd.Series(info).to_json(out_dir / "metadata.json", indent=2)


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    args = parse_args()

    run_folders = list_run_folders(args.repo_root, args.model, args.prompt)

    if not run_folders:
        raise RuntimeError("No matching run folders found.")

    combos = collect_combo_metrics(run_folders, faith_threshold=args.faith_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for combo_key in sorted(combos):
        run_one_combo(combos[combo_key], args.output_dir, args.faith_threshold)

    print(f"Saved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
