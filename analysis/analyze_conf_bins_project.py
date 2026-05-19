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
Typical usage:
    python analyze_conf_bins_project.py --repo-root .

Single run:
    python analyze_conf_bins_project.py --repo-root . --run-folder ds_8b/aime_b
    python analyze_conf_bins_project.py --repo-root . --run-folder aime_b
"""

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Configuration
# ============================================================

REAL_RESULTS_DIR = _DEFAULT_REAL_RESULTS_DIR

MODEL_LABELS = _common.MODEL_FULL_LABELS

DATASET_LABELS = _common.DATASET_LABELS

PROMPT_SUFFIXES = {"_msh_perc": "msh+perception", "_perc": "perception", "_b": "baseline"}

METHODS = _common.METHODS_CONF_BINS

DECISIVENESS_CANDIDATES = _common.DECISIVENESS_CANDIDATES

CORRECT_CANDIDATES = _common.CORRECT_CANDIDATES

TRUE_STRINGS = _common.TRUE_STRINGS
FALSE_STRINGS = _common.FALSE_STRINGS

CMFG_STAR_BINS = 10
PLOT_DPI = 300


# ============================================================
# Plot style
# ============================================================

sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.8,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--real-results-dir",
        type=Path,
        default=REAL_RESULTS_DIR,
        help="Path to real_results directory, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--run-folder",
        type=str,
        default=None,
        help=(
            "Optional single run folder. Examples: ds_8b/aime_b, "
            "qwq_32b/sgpqa_msh_perc, or just aime_b."
        ),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="conf_bin_project_real",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=CMFG_STAR_BINS,
        help="Number of equal-mass bins used for cMFG*.",
    )
    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def clean_text(text: Any) -> str:
    return " ".join(str(text).replace("\xa0", " ").split()).strip()


def normalize_col(c: str) -> str:
    return (
        str(c)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col(c) for c in df.columns]
    return df


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        c_norm = normalize_col(c)
        if c_norm in cols:
            return c_norm
    return None


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


def safe_nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)

    if arr.size == 0:
        return np.nan

    valid = np.isfinite(arr)

    if valid.sum() == 0:
        return np.nan

    return float(np.mean(arr[valid]))


def prettify_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.7)
    ax.grid(False, axis="x")
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def find_examples_xlsx(folder: Path) -> Path:
    candidates = sorted(folder.glob("results_*_examples.xlsx"))
    if not candidates:
        candidates = sorted(p for p in folder.glob("*.xlsx") if "examples" in p.name.lower())

    if not candidates:
        raise FileNotFoundError(f"No *examples*.xlsx found in {folder}")

    return candidates[0]


def get_output_dir(run_folder: Path, output_name: str) -> Path:
    return _output_dir_for_run(output_name, run_folder)


def get_plot_dir(out_dir: Path) -> Path:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def parse_run_metadata(run_folder: Path) -> Dict[str, str]:
    model_key = run_folder.parent.name
    run_name = run_folder.name

    model = MODEL_LABELS.get(model_key, model_key)

    dataset_key = run_name
    prompt = "unknown"

    for suffix, prompt_label in PROMPT_SUFFIXES.items():
        if run_name.endswith(suffix):
            dataset_key = run_name[:-len(suffix)]
            prompt = prompt_label
            break

    dataset = DATASET_LABELS.get(dataset_key.lower(), dataset_key)

    return {
        "model_key": model_key,
        "model": model,
        "dataset_key": dataset_key,
        "dataset": dataset,
        "prompt": prompt,
        "run_name": run_name,
    }


def list_run_folders(
    repo_root: Path,
    real_results_dir: Path,
    run_folder: Optional[str],
) -> List[Path]:
    real_root = real_results_dir if real_results_dir.is_absolute() else repo_root / real_results_dir

    if run_folder is not None:
        raw = Path(run_folder)

        candidates = []

        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(repo_root / raw)
            candidates.append(real_root / raw)

            if len(raw.parts) == 1:
                candidates.extend(sorted(real_root.glob(f"*/{raw.name}")))

        for p in candidates:
            if p.is_dir():
                return [p]

        raise FileNotFoundError(
            f"Could not find run folder '{run_folder}'. Tried:\n"
            + "\n".join(f"  - {p}" for p in candidates)
        )

    if not real_root.is_dir():
        raise FileNotFoundError(f"Could not find real_results directory: {real_root.resolve()}")

    folders: List[Path] = []

    for model_dir in sorted(real_root.iterdir()):
        if not model_dir.is_dir():
            continue

        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            try:
                _ = find_examples_xlsx(run_dir)
            except FileNotFoundError:
                continue

            folders.append(run_dir)

    return folders


def get_plot_bins_df(stats_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = stats_df[
        ~stats_df["Bin"].isin(["Overall", "Bin-Avg Faithfulness", "cMFG*"])
    ].copy()

    plot_df = plot_df.reset_index(drop=True)
    plot_df["Bin Index"] = np.arange(len(plot_df))
    plot_df["Bin Center"] = (plot_df["Bin Index"] + 0.5) / 10.0

    return plot_df


# ============================================================
# cMFG* computation
# ============================================================

def compute_cmfg_star(
    confidence: pd.Series,
    faithfulness: pd.Series,
    n_bins: int = CMFG_STAR_BINS,
) -> float:
    df = pd.DataFrame({
        "confidence": pd.to_numeric(confidence, errors="coerce"),
        "faithfulness": pd.to_numeric(faithfulness, errors="coerce"),
    }).dropna()

    if df.empty:
        return np.nan

    df = df.sort_values("confidence").reset_index(drop=True)
    n = len(df)

    if n < 2:
        return float(df["faithfulness"].mean())

    conf_min = float(df["confidence"].min())
    conf_max = float(df["confidence"].max())

    if np.isclose(conf_min, conf_max):
        return float(df["faithfulness"].mean())

    k = min(n_bins, n)
    index_bins = np.array_split(np.arange(n), k)

    bin_faithfulness = []
    bin_widths = []

    for b_idx, inds in enumerate(index_bins):
        if len(inds) == 0:
            continue

        left_i = int(inds[0])
        right_i = int(inds[-1])

        bin_conf_left = float(df.loc[left_i, "confidence"])
        bin_conf_right = float(df.loc[right_i, "confidence"])
        bin_faith = float(df.loc[inds, "faithfulness"].mean())

        if b_idx == 0:
            lower = conf_min
        else:
            prev_right_i = int(index_bins[b_idx - 1][-1])
            prev_right_conf = float(df.loc[prev_right_i, "confidence"])
            lower = 0.5 * (prev_right_conf + bin_conf_left)

        if b_idx == len(index_bins) - 1:
            upper = conf_max
        else:
            next_left_i = int(index_bins[b_idx + 1][0])
            next_left_conf = float(df.loc[next_left_i, "confidence"])
            upper = 0.5 * (bin_conf_right + next_left_conf)

        width = max(upper - lower, 0.0)

        bin_faithfulness.append(bin_faith)
        bin_widths.append(width)

    values = np.asarray(bin_faithfulness, dtype=float)
    weights = np.asarray(bin_widths, dtype=float)

    if weights.sum() <= 0:
        return float(df["faithfulness"].mean())

    return float(np.average(values, weights=weights))


# ============================================================
# Statistics
# ============================================================

def get_stats_df(
    df_raw: pd.DataFrame,
    method_key: str,
    n_cmfg_bins: int,
) -> pd.DataFrame:
    df = normalize_columns(df_raw)
    meta = METHODS[method_key]

    conf_col = find_col(df, meta["conf_candidates"])
    faith_col = find_col(df, meta["faith_candidates"])
    dec_col = find_col(df, DECISIVENESS_CANDIDATES)
    correct_col = find_col(df, CORRECT_CANDIDATES)

    missing = []
    if conf_col is None:
        missing.append(f"confidence column from {meta['conf_candidates']}")
    if faith_col is None:
        missing.append(f"faithfulness column from {meta['faith_candidates']}")
    if dec_col is None:
        missing.append(f"decisiveness column from {DECISIVENESS_CANDIDATES}")

    if missing:
        raise KeyError(f"Missing columns for {method_key}: {missing}")

    confidence_bins = [i / 10 for i in range(1, 11)]

    confidences: List[float] = []
    decisivenesses: List[float] = []
    faithfulnesses: List[float] = []
    accuracies: List[float] = []
    bin_ids: List[int] = []

    conf_series = pd.to_numeric(df[conf_col], errors="coerce")
    faith_series = pd.to_numeric(df[faith_col], errors="coerce")
    dec_series = pd.to_numeric(df[dec_col], errors="coerce")

    if correct_col is not None:
        acc_series = df[correct_col].apply(parse_correct)
    else:
        acc_series = pd.Series([np.nan] * len(df), index=df.index)

    for conf, dec, faith, acc in zip(conf_series, dec_series, faith_series, acc_series):
        if pd.isna(conf) or pd.isna(dec) or pd.isna(faith):
            continue

        conf_float = float(conf)
        conf_clipped = min(max(conf_float, 0.0), 1.0)

        bin_idx = int(np.searchsorted(confidence_bins, conf_clipped, side="left"))
        bin_idx = min(max(bin_idx, 0), len(confidence_bins) - 1)

        bin_ids.append(bin_idx)
        confidences.append(conf_float)
        decisivenesses.append(float(dec))
        faithfulnesses.append(float(faith))
        accuracies.append(acc if acc is not None else np.nan)

    bin_counts = [bin_ids.count(i) for i in range(len(confidence_bins))]

    bin_idx_to_name = {
        i: f"({confidence_bins[i] - 0.1:.1f}, {confidence_bins[i]:.1f}]"
        for i in range(1, len(confidence_bins))
    }
    bin_idx_to_name[0] = f"[0.0, {confidence_bins[0]:.1f}]"

    mean_f_per_bin: List[float] = []
    mean_acc_per_bin: List[float] = []

    for i in range(len(confidence_bins)):
        bin_f = [
            faithfulnesses[j]
            for j in range(len(faithfulnesses))
            if bin_ids[j] == i
        ]

        bin_a = [
            accuracies[j]
            for j in range(len(accuracies))
            if bin_ids[j] == i and not pd.isna(accuracies[j])
        ]

        mean_f_per_bin.append(safe_nanmean(bin_f))
        mean_acc_per_bin.append(safe_nanmean(bin_a))

    bin_avg_faithfulness = safe_nanmean(mean_f_per_bin)

    df_data = []

    for i in range(len(confidence_bins)):
        bin_confidences = [
            confidences[j]
            for j in range(len(confidences))
            if bin_ids[j] == i
        ]

        bin_decisivenesses = [
            decisivenesses[j]
            for j in range(len(decisivenesses))
            if bin_ids[j] == i
        ]

        mean_confidence = safe_nanmean(bin_confidences)
        mean_decisiveness = safe_nanmean(bin_decisivenesses)

        df_data.append({
            "Bin": bin_idx_to_name[i],
            "Bin Counts": int(bin_counts[i]),
            "Mean Bin Faithfulness": mean_f_per_bin[i],
            "Mean Bin Confidence": mean_confidence,
            "Mean Bin Decisiveness": mean_decisiveness,
            "Mean Bin Accuracy": mean_acc_per_bin[i],
        })

    overall_acc = safe_nanmean(accuracies)
    overall_conf = safe_nanmean(confidences)
    overall_dec = safe_nanmean(decisivenesses)
    overall_f = safe_nanmean(faithfulnesses)

    cmfg_star = compute_cmfg_star(
        confidence=pd.Series(confidences),
        faithfulness=pd.Series(faithfulnesses),
        n_bins=n_cmfg_bins,
    )

    df_data.append({
        "Bin": "Overall",
        "Bin Counts": int(sum(bin_counts)),
        "Mean Bin Faithfulness": overall_f,
        "Mean Bin Confidence": overall_conf,
        "Mean Bin Decisiveness": overall_dec,
        "Mean Bin Accuracy": overall_acc,
    })

    df_data.append({
        "Bin": "Bin-Avg Faithfulness",
        "Bin Counts": None,
        "Mean Bin Faithfulness": bin_avg_faithfulness,
        "Mean Bin Confidence": None,
        "Mean Bin Decisiveness": None,
        "Mean Bin Accuracy": None,
    })

    df_data.append({
        "Bin": "cMFG*",
        "Bin Counts": None,
        "Mean Bin Faithfulness": cmfg_star,
        "Mean Bin Confidence": None,
        "Mean Bin Decisiveness": None,
        "Mean Bin Accuracy": None,
    })

    return pd.DataFrame(df_data)


# ============================================================
# Per-method plots
# ============================================================

def plot_method_conf_bin_curves(
    stats_df: pd.DataFrame,
    plot_dir: Path,
    method_key: str,
) -> None:
    meta = METHODS[method_key]
    plot_df = get_plot_bins_df(stats_df)

    x = plot_df["Bin Center"].to_numpy(dtype=float)
    x_labels = plot_df["Bin"].tolist()

    metric_specs = [
        ("Mean Bin Faithfulness", "Bin Faithfulness", meta["color"], "-", "o"),
        ("Mean Bin Accuracy", "Accuracy", "#222222", "--", "s"),
        ("Mean Bin Decisiveness", "Decisiveness", "#E15759", "-.", "^"),
        ("Mean Bin Confidence", "Confidence", "#777777", ":", "D"),
    ]

    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    for col, label, color, linestyle, marker in metric_specs:
        y = pd.to_numeric(plot_df[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y)

        if valid.sum() == 0:
            continue

        ax.plot(
            x[valid],
            y[valid],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=5,
            linewidth=2.0,
        )

    ax.set_xlabel("Intrinsic Confidence Bin")
    ax.set_ylabel("Mean Value")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0.0, 1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right")

    ax.set_title(f"{meta['label']}: Confidence-Bin Statistics")
    ax.legend(frameon=True, loc="best")

    prettify_axis(ax)
    fig.tight_layout()

    save_figure(fig, plot_dir / f"{method_key}_confidence_bin_curves")


def plot_method_bin_counts(
    stats_df: pd.DataFrame,
    plot_dir: Path,
    method_key: str,
) -> None:
    meta = METHODS[method_key]
    plot_df = get_plot_bins_df(stats_df)

    x = np.arange(len(plot_df))
    counts = pd.to_numeric(plot_df["Bin Counts"], errors="coerce").fillna(0).to_numpy(dtype=int)
    x_labels = plot_df["Bin"].tolist()

    fig, ax = plt.subplots(figsize=(8.8, 4.5))

    bars = ax.bar(
        x,
        counts,
        color=meta["color"],
        alpha=0.85,
        edgecolor="#222222",
        linewidth=0.4,
    )

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(count),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Intrinsic Confidence Bin")
    ax.set_ylabel("Number of Samples")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    # ax.set_title(f"{meta['label']}: Samples per Confidence Bin")

    prettify_axis(ax)
    fig.tight_layout()

    save_figure(fig, plot_dir / f"{method_key}_confidence_bin_counts")


def plot_method_faithfulness_with_counts(
    stats_df: pd.DataFrame,
    plot_dir: Path,
    method_key: str,
) -> None:
    meta = METHODS[method_key]
    plot_df = get_plot_bins_df(stats_df)

    x = np.arange(len(plot_df))
    x_labels = plot_df["Bin"].tolist()

    faith = pd.to_numeric(plot_df["Mean Bin Faithfulness"], errors="coerce").to_numpy(dtype=float)
    counts = pd.to_numeric(plot_df["Bin Counts"], errors="coerce").fillna(0).to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(8.8, 5.0))

    valid_faith = np.isfinite(faith)

    ax1.plot(
        x[valid_faith],
        faith[valid_faith],
        color=meta["color"],
        marker="o",
        linewidth=2.4,
        markersize=5.5,
        label="Bin Faithfulness",
    )

    ax1.set_xlabel("Intrinsic Confidence Bin")
    ax1.set_ylabel("Mean Bin Faithfulness")
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=35, ha="right")

    ax2 = ax1.twinx()
    ax2.bar(
        x,
        counts,
        color="#bdbdbd",
        alpha=0.28,
        edgecolor="none",
        label="Sample Count",
    )
    ax2.set_ylabel("Sample Count")
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=True, loc="best")
    ax1.set_title(f"{meta['label']}: Faithfulness Across Confidence Bins")

    prettify_axis(ax1)
    fig.tight_layout()

    save_figure(fig, plot_dir / f"{method_key}_faithfulness_with_counts")


# ============================================================
# All-method comparison plots
# ============================================================

def plot_all_methods_metric_comparison(
    stats_by_method: Dict[str, pd.DataFrame],
    plot_dir: Path,
    metric_col: str,
    metric_label: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    reference_df = None

    for method_key, stats_df in stats_by_method.items():
        meta = METHODS[method_key]
        plot_df = get_plot_bins_df(stats_df)

        if reference_df is None:
            reference_df = plot_df

        x = plot_df["Bin Center"].to_numpy(dtype=float)
        y = pd.to_numeric(plot_df[metric_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y)

        if valid.sum() == 0:
            continue

        ax.plot(
            x[valid],
            y[valid],
            label=meta["label"],
            color=meta["color"],
            marker="o",
            markersize=5,
            linewidth=2.2,
        )

    if reference_df is not None:
        x_ticks = reference_df["Bin Center"].to_numpy(dtype=float)
        x_labels = reference_df["Bin"].tolist()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=35, ha="right")

    ax.set_xlabel("Intrinsic Confidence Bin")
    ax.set_ylabel(metric_label)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0.0, 1.0)
    # ax.set_title(f"All Methods: {metric_label} by Confidence Bin")
    ax.legend(frameon=True)

    prettify_axis(ax)
    fig.tight_layout()

    save_figure(fig, plot_dir / filename)


def plot_all_methods_counts(
    stats_by_method: Dict[str, pd.DataFrame],
    plot_dir: Path,
) -> None:
    method_keys = list(stats_by_method.keys())

    if not method_keys:
        return

    reference_df = get_plot_bins_df(stats_by_method[method_keys[0]])
    x = np.arange(len(reference_df))
    x_labels = reference_df["Bin"].tolist()

    width = 0.24

    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    offsets = np.linspace(
        -width * (len(method_keys) - 1) / 2,
        width * (len(method_keys) - 1) / 2,
        len(method_keys),
    )

    for offset, method_key in zip(offsets, method_keys):
        meta = METHODS[method_key]
        plot_df = get_plot_bins_df(stats_by_method[method_key])
        counts = pd.to_numeric(plot_df["Bin Counts"], errors="coerce").fillna(0).to_numpy(dtype=float)

        ax.bar(
            x + offset,
            counts,
            width=width,
            label=meta["label"],
            color=meta["color"],
            alpha=0.85,
            edgecolor="#222222",
            linewidth=0.3,
        )

    ax.set_xlabel("Intrinsic Confidence Bin")
    ax.set_ylabel("Number of Samples")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    # ax.set_title("All Methods: Samples per Confidence Bin")
    ax.legend(frameon=True)

    prettify_axis(ax)
    fig.tight_layout()

    save_figure(fig, plot_dir / "all_methods_confidence_bin_counts")


def plot_summary_all_methods(
    summary_df: pd.DataFrame,
    plot_dir: Path,
) -> None:
    if summary_df.empty:
        return

    metric_specs = [
        ("cmfg_star", "cMFG$^*$"),
        ("overall_confidence", "Confidence"),
        ("overall_decisiveness", "Decisiveness"),
        ("overall_accuracy", "Accuracy"),
    ]

    methods = summary_df["method"].tolist()
    x = np.arange(len(metric_specs))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    offsets = np.linspace(
        -width * (len(methods) - 1) / 2,
        width * (len(methods) - 1) / 2,
        len(methods),
    )

    for offset, (_, row) in zip(offsets, summary_df.iterrows()):
        method_label = row["method"]

        method_key = None
        for k, meta in METHODS.items():
            if meta["label"] == method_label:
                method_key = k
                break

        color = METHODS[method_key]["color"] if method_key is not None else "#777777"

        values = [
            row[col] if pd.notna(row[col]) else np.nan
            for col, _ in metric_specs
        ]

        ax.bar(
            x + offset,
            values,
            width=width,
            label=method_label,
            color=color,
            alpha=0.88,
            edgecolor="#222222",
            linewidth=0.3,
        )

    ax.set_ylabel("Value")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_specs], rotation=25, ha="right")
    ax.set_title("Overall Summary Across Methods")
    ax.legend(frameon=True)

    prettify_axis(ax)
    fig.tight_layout()

    save_figure(fig, plot_dir / "all_methods_summary_metrics")


# ============================================================
# Analysis
# ============================================================

def analyze_run(
    run_folder: Path,
    output_name: str,
    n_cmfg_bins: int,
) -> None:
    meta = parse_run_metadata(run_folder)
    examples_path = find_examples_xlsx(run_folder)
    df = pd.read_excel(examples_path)

    out_dir = get_output_dir(run_folder, output_name)
    plot_dir = get_plot_dir(out_dir)

    summary_rows = []
    stats_by_method: Dict[str, pd.DataFrame] = {}

    print(
        f"\n[RUN] {meta['model']} | {meta['dataset']} | {meta['prompt']} "
        f"| file={examples_path.name}"
    )

    for method_key, method_meta in METHODS.items():
        stats_df = get_stats_df(df, method_key, n_cmfg_bins=n_cmfg_bins)
        stats_by_method[method_key] = stats_df

        csv_path = out_dir / f"conf_bin_analysis_{method_key}.csv"
        stats_df.to_csv(csv_path, index=False)

        plot_method_conf_bin_curves(stats_df, plot_dir, method_key)
        plot_method_bin_counts(stats_df, plot_dir, method_key)
        plot_method_faithfulness_with_counts(stats_df, plot_dir, method_key)

        overall = stats_df[stats_df["Bin"] == "Overall"].iloc[0]
        binavg = stats_df[stats_df["Bin"] == "Bin-Avg Faithfulness"].iloc[0]
        cmfg_star = stats_df[stats_df["Bin"] == "cMFG*"].iloc[0]

        summary_rows.append({
            "model": meta["model"],
            "dataset": meta["dataset"],
            "prompt": meta["prompt"],
            "run_folder": str(run_folder),
            "method": method_meta["label"],
            "n_samples": overall["Bin Counts"],
            "cmfg_star": cmfg_star["Mean Bin Faithfulness"],
            "overall_faithfulness_diagnostic": overall["Mean Bin Faithfulness"],
            "bin_avg_faithfulness_diagnostic": binavg["Mean Bin Faithfulness"],
            "overall_confidence": overall["Mean Bin Confidence"],
            "overall_decisiveness": overall["Mean Bin Decisiveness"],
            "overall_accuracy": overall["Mean Bin Accuracy"],
        })

        print(f"  Saved {method_key} confidence-bin analysis to {csv_path}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = out_dir / "conf_bin_summary_all_methods.csv"
        summary_df.to_csv(summary_path, index=False)

        plot_summary_all_methods(summary_df, plot_dir)

    plot_all_methods_metric_comparison(
        stats_by_method,
        plot_dir,
        metric_col="Mean Bin Faithfulness",
        metric_label="Mean Bin Faithfulness",
        filename="all_methods_mean_bin_faithfulness_by_confidence_bin",
    )

    plot_all_methods_metric_comparison(
        stats_by_method,
        plot_dir,
        metric_col="Mean Bin Accuracy",
        metric_label="Mean Accuracy",
        filename="all_methods_mean_accuracy_by_confidence_bin",
    )

    plot_all_methods_metric_comparison(
        stats_by_method,
        plot_dir,
        metric_col="Mean Bin Decisiveness",
        metric_label="Mean Decisiveness",
        filename="all_methods_mean_decisiveness_by_confidence_bin",
    )

    plot_all_methods_metric_comparison(
        stats_by_method,
        plot_dir,
        metric_col="Mean Bin Confidence",
        metric_label="Mean Confidence",
        filename="all_methods_mean_confidence_by_confidence_bin",
    )

    plot_all_methods_counts(stats_by_method, plot_dir)

    print(f"Saved all confidence-bin outputs to {out_dir}")
    print(f"Saved plots to {plot_dir}")


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    run_folders = list_run_folders(
        repo_root=repo_root,
        real_results_dir=args.real_results_dir,
        run_folder=args.run_folder,
    )

    if not run_folders:
        raise RuntimeError("No matching run folders found.")

    print(f"Found {len(run_folders)} run folder(s).")

    for run_folder in run_folders:
        try:
            analyze_run(
                run_folder=run_folder,
                output_name=args.output_name,
                n_cmfg_bins=int(args.bins),
            )
        except Exception as e:
            print(f"[SKIP] {run_folder}: {e}")


if __name__ == "__main__":
    main()