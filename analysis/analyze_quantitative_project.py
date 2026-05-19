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
    python analyze_quantitative_project.py --repo-root .
    python analyze_quantitative_project.py --repo-root . --run-folder ds_8b/aime_b
"""

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde

METHODS = _common.METHODS_FIXED_COLS

sns.set_theme(style="whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--run-folder",
        type=str,
        default=None,
        help="Optional single run folder, e.g. ds_8b/aime_b or aime_b",
    )
    parser.add_argument("--output-name", type=str, default="quantitative_project")
    return parser.parse_args()


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


def list_run_folders(repo_root: Path, run_folder: Optional[str]) -> List[Path]:
    return _list_analysis_run_folders(
        repo_root=repo_root,
        real_results_dir=_DEFAULT_REAL_RESULTS_DIR,
        run_folder=run_folder,
        require_examples=True,
    )


def find_examples_xlsx(folder: Path) -> Path:
    candidates = sorted(folder.glob("*.xlsx"))
    candidates = [p for p in candidates if "examples" in p.name.lower()]
    if not candidates:
        raise FileNotFoundError(f"No *examples*.xlsx found in {folder}")
    return candidates[0]


def get_output_dir(run_folder: Path, output_name: str) -> Path:
    return _output_dir_for_run(output_name, run_folder)


def get_method_samples(df: pd.DataFrame, method_key: str) -> List[Dict[str, Any]]:
    meta = METHODS[method_key]
    samples: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        confidence = row.get(meta["conf_col"])
        faithfulness = row.get(meta["faith_col"])
        decisiveness = row.get("avg_decisiveness")
        accuracy = parse_correct(row.get("correct"))

        if pd.isna(confidence) or pd.isna(faithfulness) or pd.isna(decisiveness):
            continue

        samples.append({
            "idx": idx,
            "input": row.get("question", ""),
            "output": row.get("generated_text", ""),
            "faithfulness": float(faithfulness),
            "intrinsic_confidence": float(confidence),
            "expressed_confidence": float(decisiveness),
            "accuracy": accuracy,
            "token_count": row.get("deepconf_num_tokens", None),
        })

    return samples


def get_mask(samples: Sequence[Dict[str, Any]], *keys: str) -> np.ndarray:
    return np.array([
        all(s.get(k) is not None and not pd.isna(s.get(k)) for k in keys)
        for s in samples
    ])


def plot_scatter_colored(
    samples: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str,
    method_label: str,
) -> None:
    mask = get_mask(samples, "intrinsic_confidence", "expressed_confidence", "faithfulness")
    s = [x for x, m in zip(samples, mask) if m]

    if not s:
        return

    ic = np.array([x["intrinsic_confidence"] for x in s])
    ec = np.array([x["expressed_confidence"] for x in s])
    fc = np.array([x["faithfulness"] for x in s])

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(ic, ec, c=fc, cmap="RdYlGn", alpha=0.65, s=18, vmin=0, vmax=1)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="y=x")
    plt.colorbar(sc, ax=ax, label="Faithfulness")

    ax.set_xlabel("Intrinsic confidence")
    ax.set_ylabel("Expressed confidence (decisiveness)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{method_label}: intrinsic vs expressed confidence\n(colored by faithfulness)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_scatter_colored.png", dpi=150)
    plt.close(fig)


def plot_residual(
    samples: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str,
    method_label: str,
) -> None:
    mask = get_mask(samples, "intrinsic_confidence", "expressed_confidence", "faithfulness")
    s = [x for x, m in zip(samples, mask) if m]

    if not s:
        return

    ic = np.array([x["intrinsic_confidence"] for x in s])
    ec = np.array([x["expressed_confidence"] for x in s])
    fc = np.array([x["faithfulness"] for x in s])

    residual = ec - ic

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(ic, residual, c=fc, cmap="RdYlGn", alpha=0.65, s=18, vmin=0, vmax=1)

    ax.axhline(0, color="k", lw=1, ls="--", alpha=0.4)
    plt.colorbar(sc, ax=ax, label="Faithfulness")

    ax.set_xlabel("Intrinsic confidence")
    ax.set_ylabel("Expressed - Intrinsic")
    ax.set_xlim(0, 1)
    ax.set_title(f"{method_label}: confidence residual vs intrinsic confidence")

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_residual.png", dpi=150)
    plt.close(fig)


def plot_2x2_quadrant(
    samples: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str,
    method_label: str,
) -> None:
    mask = get_mask(samples, "intrinsic_confidence", "expressed_confidence", "faithfulness")
    s = [x for x, m in zip(samples, mask) if m]

    if not s:
        return

    ic = np.array([x["intrinsic_confidence"] for x in s])
    ec = np.array([x["expressed_confidence"] for x in s])
    fc = np.array([x["faithfulness"] for x in s])

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    quadrant_defs = [
        ("High Intrinsic, Low Expressed", ic >= 0.5, ec < 0.5, axes[0][0]),
        ("High Intrinsic, High Expressed", ic >= 0.5, ec >= 0.5, axes[0][1]),
        ("Low Intrinsic, Low Expressed", ic < 0.5, ec < 0.5, axes[1][0]),
        ("Low Intrinsic, High Expressed", ic < 0.5, ec >= 0.5, axes[1][1]),
    ]

    norm = Normalize(0, 1)

    for title, imask, emask, ax in quadrant_defs:
        qmask = imask & emask

        ax.axvline(0.5, color="gray", lw=0.7, ls=":")
        ax.axhline(0.5, color="gray", lw=0.7, ls=":")
        ax.set_xlabel("Intrinsic confidence")
        ax.set_ylabel("Expressed confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if qmask.sum() == 0:
            ax.set_title(f"{title}\n(n=0)", fontsize=9)
            continue

        ax.scatter(
            ic[qmask],
            ec[qmask],
            c=fc[qmask],
            cmap="RdYlGn",
            alpha=0.65,
            s=16,
            vmin=0,
            vmax=1,
        )
        ax.set_title(
            f"{title}\n(n={qmask.sum()}, mean faith={fc[qmask].mean():.2f})",
            fontsize=9,
        )

    sm = ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])

    fig.subplots_adjust(hspace=0.4)
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="Faithfulness", shrink=0.6, pad=0.04)
    fig.suptitle(f"{method_label}: confidence quadrants", fontsize=12)

    fig.savefig(out_dir / f"{prefix}_quadrant_2x2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_faithfulness_dist(
    samples: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str,
    method_label: str,
) -> None:
    fc = np.array([s["faithfulness"] for s in samples if s["faithfulness"] is not None], dtype=float)

    if len(fc) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fc, bins=30, density=True, alpha=0.4, color="steelblue", label="histogram")

    if len(fc) >= 2 and np.std(fc) > 0:
        kde = gaussian_kde(fc)
        xs = np.linspace(0, 1, 300)
        ax.plot(xs, kde(xs), color="steelblue", lw=2, label="KDE")

    ax.set_xlabel("Faithfulness")
    ax.set_ylabel("Density")
    ax.set_title(f"{method_label}: faithfulness distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_faithfulness_distribution.png", dpi=150)
    plt.close(fig)


def plot_faithfulness_by_accuracy(
    samples: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str,
    method_label: str,
) -> None:
    s = [x for x in samples if x["accuracy"] is not None]

    if not s:
        return

    correct = [x["faithfulness"] for x in s if x["accuracy"] >= 0.5]
    wrong = [x["faithfulness"] for x in s if x["accuracy"] < 0.5]

    fig, ax = plt.subplots(figsize=(7, 4))

    for data, label, color in [
        (correct, "Correct", "steelblue"),
        (wrong, "Wrong", "tomato"),
    ]:
        if len(data) < 2:
            continue

        ax.hist(
            data,
            bins=20,
            density=True,
            alpha=0.45,
            color=color,
            label=f"{label} (n={len(data)})",
        )

        if np.std(data) > 0:
            kde = gaussian_kde(data)
            xs = np.linspace(0, 1, 300)
            ax.plot(xs, kde(xs), color=color, lw=2)

    ax.set_xlabel("Faithfulness")
    ax.set_ylabel("Density")
    ax.set_title(f"{method_label}: faithfulness by accuracy")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_faithfulness_by_accuracy.png", dpi=150)
    plt.close(fig)


def plot_accuracy_confidence_heatmap(
    samples: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str,
    method_label: str,
) -> None:
    """
    Paper-friendly 2x2 heatmap.

    Empty cells are masked and shown as light grey with no 'N/A' text.
    This avoids inserting fake values while keeping the visualization clean.
    """

    s = [
        x for x in samples
        if x["accuracy"] is not None
        and x["faithfulness"] is not None
        and x["intrinsic_confidence"] is not None
    ]

    if not s:
        return

    cells = {
        "Correct\nHigh Conf": [],
        "Correct\nLow Conf": [],
        "Wrong\nHigh Conf": [],
        "Wrong\nLow Conf": [],
    }

    for x in s:
        acc_key = "Correct" if x["accuracy"] >= 0.5 else "Wrong"
        conf_key = "High Conf" if x["intrinsic_confidence"] >= 0.5 else "Low Conf"
        cells[f"{acc_key}\n{conf_key}"].append(x["faithfulness"])

    means = [np.mean(v) if len(v) > 0 else np.nan for v in cells.values()]
    counts = [len(v) for v in cells.values()]

    data = np.array(means, dtype=float).reshape(2, 2)
    count_data = np.array(counts, dtype=int).reshape(2, 2)

    masked_data = np.ma.masked_invalid(data)

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#f2f2f2")

    fig, ax = plt.subplots(figsize=(6.5, 5.4))

    im = ax.imshow(masked_data, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["High intrinsic", "Low intrinsic"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Correct", "Wrong"])

    for i in range(2):
        for j in range(2):
            v = data[i, j]
            n = count_data[i, j]

            if n > 0 and np.isfinite(v):
                txt = f"{v:.2f}\n(n={n})"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="black",
                    fontweight="bold",
                )
            else:
                # Empty cell: clean paper-style annotation.
                # No "N/A"; no fake mean value.
                ax.text(
                    j,
                    i,
                    "n=0",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#777777",
                )

    cbar = plt.colorbar(im, ax=ax, label="Mean faithfulness", fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(
        f"{method_label}: mean faithfulness\naccuracy × intrinsic confidence",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_accuracy_confidence_heatmap.png", dpi=150)
    plt.close(fig)


def plot_reliability(
    samples: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str,
    method_label: str,
) -> None:
    s = [
        x for x in samples
        if x["intrinsic_confidence"] is not None
        and x["expressed_confidence"] is not None
        and x["faithfulness"] is not None
    ]

    if not s:
        return

    ic = np.array([x["intrinsic_confidence"] for x in s])
    ec = np.array([x["expressed_confidence"] for x in s])
    fc = np.array([x["faithfulness"] for x in s])

    bins = np.linspace(0, 1, 11)
    bin_idx = np.digitize(ic, bins) - 1
    bin_idx = np.clip(bin_idx, 0, 9)

    bin_centers, bin_means, bin_counts, faith_means = [], [], [], []

    for b in range(10):
        idxs = np.where(bin_idx == b)[0]
        if len(idxs) > 0:
            bin_centers.append(bins[b] + 0.05)
            bin_means.append(ec[idxs].mean())
            bin_counts.append(len(idxs))
            faith_means.append(fc[idxs].mean())

    fig, ax = plt.subplots(figsize=(7, 6))

    sc = ax.scatter(bin_centers, bin_means, c=bin_counts, cmap="Blues", s=80, zorder=5)
    ax.plot(bin_centers, bin_means, color="steelblue", lw=1.5, label="Mean expressed confidence")

    cbar = plt.colorbar(sc, ax=ax, label="Count in bin", pad=0.15)
    cbar.set_label("Count in bin", fontsize=12)

    ax.set_xlabel("Intrinsic confidence (binned)")
    ax.set_ylabel("Expressed confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.07)

    ax2 = ax.twinx()
    #faith_means = [x + 0.03 for x in faith_means]
    ax2.plot(
        bin_centers,
        faith_means,
        color="purple",
        lw=1.5,
        ls="--",
        marker="x",
        ms=6,
        label="Mean faithfulness",
    )

    ax2.set_ylabel("Faithfulness", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")
    ax2.set_ylim(0, 1.07)

    ax.plot([0, 1], [0, 1], color="steelblue", lw=1, alpha=0.6, ls="--")
    ax2.axhline(y=1.0, color="purple", lw=1, alpha=0.6, ls="--")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right")
    ax.set_title(f"{method_label}: reliability diagram")

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_reliability_diagram.png", dpi=150)
    plt.close(fig)


def plot_joint_heatmap(
    samples: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str,
    method_label: str,
) -> None:
    s = [
        x for x in samples
        if x["intrinsic_confidence"] is not None
        and x["expressed_confidence"] is not None
    ]

    if not s:
        return

    ic = np.array([x["intrinsic_confidence"] for x in s])
    ec = np.array([x["expressed_confidence"] for x in s])

    fig, ax = plt.subplots(figsize=(6, 5))

    h = ax.hist2d(ic, ec, bins=20, range=[[0, 1], [0, 1]], cmap="YlOrRd")
    plt.colorbar(h[3], ax=ax, label="Count")

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.6)

    ax.set_xlabel("Intrinsic confidence")
    ax.set_ylabel("Expressed confidence")
    ax.set_title(f"{method_label}: joint confidence heatmap")

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_joint_heatmap.png", dpi=150)
    plt.close(fig)


def plot_faithfulness_by_value_bins(
    samples: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str,
    method_label: str,
    value_key: str,
    label_name: str,
) -> None:
    s = [x for x in samples if x["faithfulness"] is not None and x[value_key] is not None]

    if not s:
        return

    bins = [
        (0.0, 0.25, "[0, 0.25)"),
        (0.25, 0.5, "[0.25, 0.5)"),
        (0.5, 0.75, "[0.5, 0.75)"),
        (0.75, 1.01, "[0.75, 1]"),
    ]

    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]

    groups = [
        ([x["faithfulness"] for x in s if lo <= x[value_key] < hi], label)
        for lo, hi, label in bins
    ]

    data_to_plot = [g[0] for g in groups if len(g[0]) >= 2]
    labels_to_plot = [g[1] for g in groups if len(g[0]) >= 2]

    if data_to_plot:
        parts = ax.violinplot(
            data_to_plot,
            positions=range(len(data_to_plot)),
            showmedians=True,
            showextrema=True,
        )

        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        parts["cmedians"].set_color("black")
        parts["cmedians"].set_lw(2)

        ax.set_xticks(range(len(labels_to_plot)))
        ax.set_xticklabels(labels_to_plot)

        for j, data in enumerate(data_to_plot):
            ax.text(j, 0.02, f"n={len(data)}", ha="center", fontsize=8, color="gray")

    ax.set_xlabel(f"{label_name} bin")
    ax.set_ylabel("Faithfulness")
    ax.set_title(f"{method_label}: faithfulness by {label_name} bin (violin)")
    ax.set_ylim(0, 1)

    ax = axes[1]

    for (lo, hi, label), color in zip(bins, colors):
        data = [x["faithfulness"] for x in s if lo <= x[value_key] < hi]

        if len(data) < 2:
            continue

        ax.hist(
            data,
            bins=20,
            density=True,
            alpha=0.45,
            color=color,
            label=f"{label} (n={len(data)})",
        )

        if np.std(data) > 0:
            kde = gaussian_kde(data)
            xs = np.linspace(0, 1, 300)
            ax.plot(xs, kde(xs), color=color, lw=2)

    ax.set_xlabel("Faithfulness")
    ax.set_ylabel("Density")
    ax.set_title(f"{method_label}: faithfulness by {label_name} bin (hist)")
    ax.legend(title=f"{label_name} bin")

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_faithfulness_by_{value_key}_bins.png", dpi=150)
    plt.close(fig)


def plot_faithfulness_by_residual_bins(
    samples: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str,
    method_label: str,
) -> None:
    s = [
        x for x in samples
        if x["faithfulness"] is not None
        and x["intrinsic_confidence"] is not None
        and x["expressed_confidence"] is not None
    ]

    if not s:
        return

    for x in s:
        x["residual"] = x["expressed_confidence"] - x["intrinsic_confidence"]

    residual_bins = [
        (-1.0, -0.5, "[-1, -0.5)"),
        (-0.5, 0.0, "[-0.5, 0)"),
        (0.0, 0.5, "[0, 0.5)"),
        (0.5, 1.01, "[0.5, 1]"),
    ]

    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]

    groups = [
        ([x["faithfulness"] for x in s if lo <= x["residual"] < hi], label)
        for lo, hi, label in residual_bins
    ]

    data_to_plot = [g[0] for g in groups if len(g[0]) >= 2]
    labels_to_plot = [g[1] for g in groups if len(g[0]) >= 2]

    if data_to_plot:
        parts = ax.violinplot(
            data_to_plot,
            positions=range(len(data_to_plot)),
            showmedians=True,
            showextrema=True,
        )

        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        parts["cmedians"].set_color("black")
        parts["cmedians"].set_lw(2)

        ax.set_xticks(range(len(labels_to_plot)))
        ax.set_xticklabels(labels_to_plot)

        for j, data in enumerate(data_to_plot):
            ax.text(j, 0.02, f"n={len(data)}", ha="center", fontsize=8, color="gray")

    ax.set_xlabel("Residual bin")
    ax.set_ylabel("Faithfulness")
    ax.set_title(f"{method_label}: faithfulness by residual bin (violin)")
    ax.set_ylim(0, 1)

    ax = axes[1]

    for (lo, hi, label), color in zip(residual_bins, colors):
        data = [x["faithfulness"] for x in s if lo <= x["residual"] < hi]

        if len(data) < 2:
            continue

        ax.hist(
            data,
            bins=20,
            density=True,
            alpha=0.45,
            color=color,
            label=f"{label} (n={len(data)})",
        )

        if np.std(data) > 0:
            kde = gaussian_kde(data)
            xs = np.linspace(0, 1, 300)
            ax.plot(xs, kde(xs), color=color, lw=2)

    ax.set_xlabel("Faithfulness")
    ax.set_ylabel("Density")
    ax.set_title(f"{method_label}: faithfulness by residual bin (hist)")
    ax.legend(title="Residual bin")

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_faithfulness_by_residual_bins.png", dpi=150)
    plt.close(fig)


def save_faithfulness_bins(
    samples: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str,
    strict: bool = False,
) -> None:
    fc = np.array([
        s["faithfulness"] if s["faithfulness"] is not None else float("nan")
        for s in samples
    ])

    valid = ~np.isnan(fc)

    if valid.sum() == 0:
        return

    if strict:
        bin_edges = [0.0, 0.25000001, 0.500001, 0.7500001, 1.001]
        bin_labels = ["0.0_to_0.25", "0.25_to_0.50", "0.50_to_0.75", "0.75_to_1.0"]
    else:
        quartiles = np.nanpercentile(fc[valid], [25, 50, 75])
        bin_edges = [0.0] + list(quartiles) + [1.001]
        bin_labels = ["Q1_lowest", "Q2_low_mid", "Q3_high_mid", "Q4_highest"]

    for i, label in enumerate(bin_labels):
        lo, hi = bin_edges[i], bin_edges[i + 1]

        bin_samples = [
            s for s, fv in zip(samples, fc)
            if not math.isnan(fv) and lo <= fv < hi
        ]

        if not bin_samples:
            continue

        bin_samples = sorted(
            bin_samples,
            key=lambda x: (x["expressed_confidence"] - x["intrinsic_confidence"]),
        )

        path = out_dir / f"{prefix}_faithfulness_bin_{label}.csv"

        with open(path, "w", newline="", encoding="utf-8") as f:
            import csv

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "input",
                    "output",
                    "faithfulness",
                    "residual",
                    "intrinsic_confidence",
                    "expressed_confidence",
                    "accuracy",
                ],
            )

            writer.writeheader()

            writer.writerows([
                {
                    "input": s["input"],
                    "output": s["output"],
                    "faithfulness": s["faithfulness"],
                    "residual": round(
                        (s["expressed_confidence"] or 0)
                        - (s["intrinsic_confidence"] or 0),
                        4,
                    ),
                    "intrinsic_confidence": s["intrinsic_confidence"],
                    "expressed_confidence": s["expressed_confidence"],
                    "accuracy": s["accuracy"],
                }
                for s in bin_samples
            ])


def plot_one_method(samples: List[Dict[str, Any]], out_dir: Path, method_key: str) -> None:
    method_label = METHODS[method_key]["label"]
    prefix = method_key

    plot_scatter_colored(samples, out_dir, prefix, method_label)
    plot_residual(samples, out_dir, prefix, method_label)
    plot_2x2_quadrant(samples, out_dir, prefix, method_label)
    plot_faithfulness_dist(samples, out_dir, prefix, method_label)
    plot_faithfulness_by_accuracy(samples, out_dir, prefix, method_label)
    plot_accuracy_confidence_heatmap(samples, out_dir, prefix, method_label)
    plot_reliability(samples, out_dir, prefix, method_label)
    plot_joint_heatmap(samples, out_dir, prefix, method_label)
    plot_faithfulness_by_value_bins(
        samples,
        out_dir,
        prefix,
        method_label,
        "intrinsic_confidence",
        "intrinsic confidence",
    )
    plot_faithfulness_by_value_bins(
        samples,
        out_dir,
        prefix,
        method_label,
        "expressed_confidence",
        "expressed confidence",
    )
    plot_faithfulness_by_residual_bins(samples, out_dir, prefix, method_label)
    save_faithfulness_bins(samples, out_dir, prefix, strict=False)
    save_faithfulness_bins(samples, out_dir, prefix, strict=True)


def save_summary(
    samples_by_method: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
) -> None:
    rows = []

    for method_key, samples in samples_by_method.items():
        if not samples:
            continue

        accuracy_values = [s["accuracy"] for s in samples if s["accuracy"] is not None]

        rows.append({
            "method": METHODS[method_key]["label"],
            "n_samples": len(samples),
            "mean_confidence": np.mean([s["intrinsic_confidence"] for s in samples]),
            "mean_expressed_confidence": np.mean([s["expressed_confidence"] for s in samples]),
            "mean_faithfulness": np.mean([s["faithfulness"] for s in samples]),
            "accuracy": np.mean(accuracy_values) if accuracy_values else np.nan,
        })

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "summary_by_method.csv", index=False)


def analyze_run(run_folder: Path, output_name: str) -> None:
    xlsx_path = find_examples_xlsx(run_folder)
    df = pd.read_excel(xlsx_path)

    out_dir = get_output_dir(run_folder, output_name)

    samples_by_method: Dict[str, List[Dict[str, Any]]] = {}

    for method_key in METHODS:
        samples = get_method_samples(df, method_key)
        samples_by_method[method_key] = samples
        plot_one_method(samples, out_dir, method_key)

    save_summary(samples_by_method, out_dir)

    print(f"Saved quantitative plots to {out_dir}")


def main() -> None:
    args = parse_args()

    run_folders = list_run_folders(args.repo_root, args.run_folder)

    if not run_folders:
        raise RuntimeError("No matching run folders found.")

    for run_folder in run_folders:
        try:
            analyze_run(run_folder, args.output_name)
        except Exception as e:
            print(f"[SKIP] {run_folder.name}: {e}")


if __name__ == "__main__":
    main()
