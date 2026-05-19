#!/usr/bin/env python3
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
import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Repo format
# ============================================================

DATASET_LABELS = _common.DATASET_LABELS

MODEL_LABELS = _common.MODEL_FULL_LABELS

MODEL_SHORT_LABELS = _common.MODEL_SHORT_LABELS

PROMPT_LABELS = _common.PROMPT_LABELS

DATASET_ORDER = _common.DATASET_ORDER_WITH_LEGACY

MODEL_ORDER = _common.MODEL_ORDER_SHORT

PROMPT_ORDER = _common.PROMPT_ORDER


# ============================================================
# Methods
# ============================================================

METHODS = {k: {"label": _common.METHOD_LABELS[k], "color": _common.METHOD_COLORS[k]} for k in _common.METHOD_ORDER}

METHOD_ORDER = _common.METHOD_ORDER
METHOD_LABEL_ORDER = [METHODS[k]["label"] for k in METHOD_ORDER]


# ============================================================
# Existing step-level score columns
# ============================================================

SCORES = {
    "faithfulness": {
        "label": "Faithfulness",
        "columns": {
            "rcc": [
                "faith_rcc",
                "step_faith_rcc",
                "faithfulness_rcc",
                "step_faithfulness_rcc",
            ],
            "sampling": [
                "faith_sampling",
                "step_faith_sampling",
                "faithfulness_sampling",
                "step_faithfulness_sampling",
            ],
            "deepconf": [
                "faith_deepconf",
                "step_faith_deepconf",
                "faithfulness_deepconf",
                "step_faithfulness_deepconf",
            ],
        },
    },
    "confidence": {
        "label": "Confidence",
        "columns": {
            "rcc": [
                "rcc_p",
                "rcc_q",
                "step_rcc_p",
                "step_rcc_q",
                "rcc_confidence",
                "step_rcc_confidence",
            ],
            "sampling": [
                "sampling_conf",
                "step_sampling_conf",
                "sampling_confidence",
                "step_sampling_confidence",
            ],
            "deepconf": [
                "deepconf",
                "step_deepconf",
                "deepconf_confidence",
                "step_deepconf_confidence",
            ],
        },
    },
}

SCORE_ORDER = ["faithfulness", "confidence"]


# ============================================================
# Styling
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
    "grid.color": "#e3e3e3",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.85,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PLOT_DPI = 300


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir("step_prompt_faith_conf_curves"),
    )

    parser.add_argument("--dataset", type=str, default=None, choices=["aime", "hle", "legal", "musr", "sgpqa"])
    parser.add_argument("--model", type=str, default=None, choices=["ds_8b", "qwq_32b", "ds", "qwq"])
    parser.add_argument("--prompt", type=str, default=None)

    parser.add_argument(
        "--n-bins",
        type=int,
        default=20,
        help="Number of normalized step-position bins.",
    )

    parser.add_argument(
        "--raw-max-step",
        type=int,
        default=120,
        help="Maximum raw step index to plot.",
    )

    parser.add_argument(
        "--rcc-conf-col",
        type=str,
        default="rcc_p",
        choices=["rcc_p", "rcc_q"],
        help="Preferred RCC step confidence column.",
    )

    parser.add_argument(
        "--min-count-per-point",
        type=int,
        default=1,
        help="Minimum number of observations needed to keep a trajectory point.",
    )

    parser.add_argument(
        "--no-ci",
        action="store_true",
        help="Disable standard-error bands.",
    )

    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="Also save PDF versions.",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Ignored unknown Colab/Jupyter arguments: {unknown}")

    if args.n_bins < 2:
        raise ValueError("--n-bins must be at least 2")

    if args.raw_max_step < 1:
        raise ValueError("--raw-max-step must be positive")

    return args


# ============================================================
# Helpers
# ============================================================

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


def safe_filename(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)


def parse_run_metadata(folder: Path | str) -> Dict[str, str]:
    return _parse_run_folder_metadata(folder)


def list_run_folders(
    repo_root: Path,
    dataset_filter: Optional[str],
    model_filter: Optional[str],
    prompt_filter: Optional[str],
) -> List[Path]:
    return _list_analysis_run_folders(
        repo_root=repo_root,
        real_results_dir=_DEFAULT_REAL_RESULTS_DIR,
        dataset_filter=dataset_filter,
        model_filter=model_filter,
        prompt_filter=prompt_filter,
        require_examples=False,
        require_steps=True,
    )


def find_step_level_xlsx(folder: Path) -> Optional[Path]:
    candidates = sorted(folder.glob("*.xlsx"))

    preferred = [
        p for p in candidates
        if "step" in p.name.lower() and "level" in p.name.lower()
    ]

    if preferred:
        return preferred[0]

    fallback = [
        p for p in candidates
        if "step" in p.name.lower()
    ]

    if fallback:
        return fallback[0]

    return None


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col

    lower_map = {str(c).lower(): c for c in df.columns}

    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]

    return None


def clip01_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(lower=0.0, upper=1.0)


def save_figure(fig: plt.Figure, out_path: Path, save_pdf: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path.with_suffix(".png"), dpi=PLOT_DPI, bbox_inches="tight")

    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")

    plt.close(fig)


def order_existing(values: Sequence[str], preferred_order: Sequence[str]) -> List[str]:
    values_unique = list(dict.fromkeys([v for v in values if pd.notna(v)]))
    ordered = [v for v in preferred_order if v in values_unique]
    ordered += [v for v in values_unique if v not in ordered]
    return ordered


def make_prompt_color_map(prompts: Sequence[str]) -> Dict[str, Any]:
    ordered = order_existing(prompts, PROMPT_ORDER)
    palette = sns.color_palette("tab10", n_colors=max(10, len(ordered)))
    return {prompt: palette[i % len(palette)] for i, prompt in enumerate(ordered)}


# ============================================================
# Loading existing step-level faithfulness and confidence
# ============================================================

def load_step_file(folder: Path) -> Optional[pd.DataFrame]:
    meta = parse_run_metadata(folder)
    step_path = find_step_level_xlsx(folder)

    if step_path is None:
        print(f"[SKIP] {folder.name}: no step-level xlsx found")
        return None

    try:
        df = pd.read_excel(step_path)
    except Exception as e:
        print(f"[SKIP] {folder.name}: failed to read {step_path}: {e}")
        return None

    df = df.copy()

    if "idx" not in df.columns:
        print(f"[SKIP] {folder.name}: missing idx column")
        return None

    if "step_idx" not in df.columns:
        df["step_idx"] = df.groupby("idx").cumcount()

    df["step_idx"] = pd.to_numeric(df["step_idx"], errors="coerce")

    if "correct" in df.columns:
        df["correct_num"] = df["correct"].apply(parse_correct)
    else:
        df["correct_num"] = np.nan

    for k, v in meta.items():
        df[k] = v

    df["folder"] = folder.name
    df["step_file"] = str(step_path)

    max_step = df.groupby("idx")["step_idx"].transform("max")
    df["normalized_step"] = df["step_idx"] / max_step.replace(0, np.nan)
    df.loc[df["normalized_step"].isna(), "normalized_step"] = 0.0
    df["normalized_step"] = df["normalized_step"].clip(lower=0.0, upper=1.0)

    return df


def build_existing_score_long(
    step_df: pd.DataFrame,
    folder_name: str,
    rcc_conf_col: str,
) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    rows: List[pd.DataFrame] = []
    selected_rows: List[Dict[str, Any]] = []

    id_cols = [
        "folder",
        "step_file",
        "dataset_key",
        "dataset",
        "model_key",
        "model",
        "model_full",
        "prompt_key",
        "prompt",
        "idx",
        "step_idx",
        "normalized_step",
        "correct_num",
    ]

    for method_key in METHOD_ORDER:
        method_label = METHODS[method_key]["label"]

        for score_key in SCORE_ORDER:
            score_label = SCORES[score_key]["label"]

            candidates = list(SCORES[score_key]["columns"][method_key])

            if score_key == "confidence" and method_key == "rcc":
                candidates = [rcc_conf_col] + [c for c in candidates if c != rcc_conf_col]

            selected_col = first_existing_col(step_df, candidates)

            selected_rows.append({
                "folder": folder_name,
                "method_key": method_key,
                "method": method_label,
                "score_key": score_key,
                "score_label": score_label,
                "selected_column": selected_col,
                "found": selected_col is not None,
            })

            if selected_col is None:
                print(
                    f"[WARN] {folder_name}: no existing column found for "
                    f"{method_label} / {score_label}; skipped."
                )
                continue

            temp = step_df[id_cols].copy()
            temp["method_key"] = method_key
            temp["method"] = method_label
            temp["score_key"] = score_key
            temp["score_label"] = score_label
            temp["score_column"] = selected_col
            temp["score_value"] = clip01_series(step_df[selected_col])

            temp = temp.dropna(subset=["score_value"])

            if temp.empty:
                continue

            rows.append(temp)

    if not rows:
        return pd.DataFrame(), selected_rows

    return pd.concat(rows, ignore_index=True), selected_rows


def load_all_existing_step_scores(
    run_folders: Sequence[Path],
    rcc_conf_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    selected_all: List[Dict[str, Any]] = []

    for folder in run_folders:
        step_df = load_step_file(folder)

        if step_df is None or step_df.empty:
            continue

        score_df, selected_rows = build_existing_score_long(
            step_df=step_df,
            folder_name=folder.name,
            rcc_conf_col=rcc_conf_col,
        )

        selected_all.extend(selected_rows)

        if score_df.empty:
            print(f"[SKIP] {folder.name}: no usable step faithfulness/confidence columns")
            continue

        frames.append(score_df)
        print(f"[OK] {folder.name}: loaded {len(score_df)} step faithfulness/confidence rows")

    if not frames:
        raise RuntimeError("No usable step faithfulness/confidence columns were loaded.")

    selected_df = pd.DataFrame(selected_all)

    return pd.concat(frames, ignore_index=True), selected_df


# ============================================================
# Aggregation
# ============================================================

def make_normalized_summary(
    score_df: pd.DataFrame,
    n_bins: int,
    min_count: int,
) -> pd.DataFrame:
    df = score_df.copy()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    df["norm_bin"] = pd.cut(
        df["normalized_step"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )

    df = df.dropna(subset=["norm_bin"]).copy()
    df["norm_bin"] = df["norm_bin"].astype(int)
    df["norm_center"] = df["norm_bin"].map(lambda i: centers[i])

    group_cols = [
        "dataset_key",
        "dataset",
        "model_key",
        "model",
        "prompt_key",
        "prompt",
        "method_key",
        "method",
        "score_key",
        "score_label",
        "norm_bin",
        "norm_center",
    ]

    summary = (
        df
        .groupby(group_cols, dropna=False, observed=False)
        .agg(
            mean=("score_value", "mean"),
            std=("score_value", "std"),
            count=("score_value", "size"),
            median=("score_value", "median"),
            q25=("score_value", lambda x: np.nanquantile(x, 0.25)),
            q75=("score_value", lambda x: np.nanquantile(x, 0.75)),
        )
        .reset_index()
    )

    summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    summary = summary[summary["count"] >= min_count].copy()

    return summary


def make_raw_step_summary(
    score_df: pd.DataFrame,
    raw_max_step: int,
    min_count: int,
) -> pd.DataFrame:
    df = score_df.copy()

    df = df[pd.to_numeric(df["step_idx"], errors="coerce").notna()].copy()
    df["step_idx"] = df["step_idx"].astype(int)
    df = df[(df["step_idx"] >= 0) & (df["step_idx"] <= raw_max_step)].copy()

    group_cols = [
        "dataset_key",
        "dataset",
        "model_key",
        "model",
        "prompt_key",
        "prompt",
        "method_key",
        "method",
        "score_key",
        "score_label",
        "step_idx",
    ]

    summary = (
        df
        .groupby(group_cols, dropna=False, observed=False)
        .agg(
            mean=("score_value", "mean"),
            std=("score_value", "std"),
            count=("score_value", "size"),
            median=("score_value", "median"),
            q25=("score_value", lambda x: np.nanquantile(x, 0.25)),
            q75=("score_value", lambda x: np.nanquantile(x, 0.75)),
        )
        .reset_index()
    )

    summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    summary = summary[summary["count"] >= min_count].copy()

    return summary


def compute_normalized_auc(norm_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    group_cols = [
        "dataset_key",
        "dataset",
        "model_key",
        "model",
        "prompt_key",
        "prompt",
        "method_key",
        "method",
        "score_key",
        "score_label",
    ]

    for keys, group in norm_summary.groupby(group_cols, dropna=False, observed=False):
        key_dict = dict(zip(group_cols, keys))
        g = group.sort_values("norm_center").copy()

        x = g["norm_center"].to_numpy(dtype=float)
        y = g["mean"].to_numpy(dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        if len(x) == 0:
            auc = np.nan
        elif len(x) == 1:
            auc = float(y[0])
        else:
            span = float(x.max() - x.min())

            if span <= 0:
                auc = float(np.nanmean(y))
            else:
                auc = float(np.trapezoid(y, x) / span)

        rows.append({
            **key_dict,
            "trajectory_auc": auc,
            "n_curve_points": int(len(x)),
            "mean_curve_value": float(np.nanmean(y)) if len(y) else np.nan,
        })

    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================

def plot_normalized_prompt_curves(
    norm_summary: pd.DataFrame,
    out_dir: Path,
    save_pdf: bool,
    show_ci: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = order_existing(norm_summary["prompt"].tolist(), PROMPT_ORDER)
    prompt_colors = make_prompt_color_map(prompts)

    combo_cols = [
        "dataset",
        "model",
        "method",
        "score_key",
        "score_label",
    ]

    for keys, sub in norm_summary.groupby(combo_cols, dropna=False, observed=False):
        dataset, model, method, score_key, score_label = keys

        fig, ax = plt.subplots(figsize=(8.4, 5.4))

        for prompt in prompts:
            g = sub[sub["prompt"] == prompt].sort_values("norm_center")

            if g.empty:
                continue

            x = g["norm_center"].to_numpy(dtype=float)
            y = g["mean"].to_numpy(dtype=float)
            sem = g["sem"].fillna(0.0).to_numpy(dtype=float)

            color = prompt_colors[prompt]

            ax.plot(
                x,
                y,
                label=prompt,
                color=color,
                linewidth=2.0,
                alpha=0.95,
            )

            if show_ci:
                lo = np.clip(y - sem, 0.0, 1.0)
                hi = np.clip(y + sem, 0.0, 1.0)
                ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Normalized step position")
        ax.set_ylabel(score_label)
        ax.set_title(f"{dataset} — {model} — {method}: {score_label} trajectory by prompt")
        ax.legend(title="Prompt", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
        ax.grid(True, axis="both", alpha=0.55)

        fig.tight_layout()

        filename = (
            f"{safe_filename(dataset)}__{safe_filename(model)}__"
            f"{safe_filename(method)}__{safe_filename(score_key)}__normalized_prompt_curves"
        )

        save_figure(fig, out_dir / filename, save_pdf=save_pdf)


def plot_raw_step_prompt_curves(
    raw_summary: pd.DataFrame,
    out_dir: Path,
    save_pdf: bool,
    show_ci: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = order_existing(raw_summary["prompt"].tolist(), PROMPT_ORDER)
    prompt_colors = make_prompt_color_map(prompts)

    combo_cols = [
        "dataset",
        "model",
        "method",
        "score_key",
        "score_label",
    ]

    for keys, sub in raw_summary.groupby(combo_cols, dropna=False, observed=False):
        dataset, model, method, score_key, score_label = keys

        fig, ax = plt.subplots(figsize=(8.4, 5.4))

        for prompt in prompts:
            g = sub[sub["prompt"] == prompt].sort_values("step_idx")

            if g.empty:
                continue

            x = g["step_idx"].to_numpy(dtype=float)
            y = g["mean"].to_numpy(dtype=float)
            sem = g["sem"].fillna(0.0).to_numpy(dtype=float)

            color = prompt_colors[prompt]

            ax.plot(
                x,
                y,
                label=prompt,
                color=color,
                linewidth=2.0,
                alpha=0.95,
            )

            if show_ci:
                lo = np.clip(y - sem, 0.0, 1.0)
                hi = np.clip(y + sem, 0.0, 1.0)
                ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)

        ax.set_ylim(0, 1)
        ax.set_xlabel("Raw step index")
        ax.set_ylabel(score_label)
        ax.set_title(f"{dataset} — {model} — {method}: raw-step {score_label} by prompt")
        ax.legend(title="Prompt", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
        ax.grid(True, axis="both", alpha=0.55)

        fig.tight_layout()

        filename = (
            f"{safe_filename(dataset)}__{safe_filename(model)}__"
            f"{safe_filename(method)}__{safe_filename(score_key)}__raw_step_prompt_curves"
        )

        save_figure(fig, out_dir / filename, save_pdf=save_pdf)


def plot_normalized_overview_grids(
    norm_summary: pd.DataFrame,
    out_dir: Path,
    save_pdf: bool,
    show_ci: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = order_existing(norm_summary["prompt"].tolist(), PROMPT_ORDER)
    prompt_colors = make_prompt_color_map(prompts)

    combo_cols = ["dataset", "model", "score_key", "score_label"]

    for keys, sub_all in norm_summary.groupby(combo_cols, dropna=False, observed=False):
        dataset, model, score_key, score_label = keys

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18.0, 5.2),
            sharex=True,
            sharey=True,
        )

        for ax, method_key in zip(axes, METHOD_ORDER):
            method_label = METHODS[method_key]["label"]
            sub = sub_all[sub_all["method_key"] == method_key]

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(method_label)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                continue

            for prompt in prompts:
                g = sub[sub["prompt"] == prompt].sort_values("norm_center")

                if g.empty:
                    continue

                x = g["norm_center"].to_numpy(dtype=float)
                y = g["mean"].to_numpy(dtype=float)
                sem = g["sem"].fillna(0.0).to_numpy(dtype=float)

                color = prompt_colors[prompt]

                ax.plot(
                    x,
                    y,
                    label=prompt,
                    color=color,
                    linewidth=1.9,
                    alpha=0.95,
                )

                if show_ci:
                    lo = np.clip(y - sem, 0.0, 1.0)
                    hi = np.clip(y + sem, 0.0, 1.0)
                    ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0)

            ax.set_title(method_label)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Normalized step position")
            ax.grid(True, alpha=0.55)

        axes[0].set_ylabel(score_label)

        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                title="Prompt",
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=True,
            )

        fig.suptitle(f"{dataset} — {model}: {score_label} step trajectories by prompt", fontsize=16)
        fig.tight_layout(rect=[0, 0, 0.88, 0.94])

        filename = (
            f"{safe_filename(dataset)}__{safe_filename(model)}__"
            f"{safe_filename(score_key)}__normalized_prompt_overview_grid"
        )

        save_figure(fig, out_dir / filename, save_pdf=save_pdf)


def plot_auc_heatmaps(
    auc_df: pd.DataFrame,
    out_dir: Path,
    save_pdf: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = order_existing(auc_df["dataset"].tolist(), DATASET_ORDER)
    prompts = order_existing(auc_df["prompt"].tolist(), PROMPT_ORDER)

    combo_cols = ["model", "method", "score_key", "score_label"]

    for keys, sub in auc_df.groupby(combo_cols, dropna=False, observed=False):
        model, method, score_key, score_label = keys

        pivot = (
            sub
            .pivot_table(
                index="dataset",
                columns="prompt",
                values="trajectory_auc",
                aggfunc="mean",
            )
            .reindex(index=datasets, columns=prompts)
        )

        if pivot.empty:
            continue

        fig_width = max(10.0, 0.75 * len(prompts) + 3.0)
        fig, ax = plt.subplots(figsize=(fig_width, 4.6))

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": f"{score_label} trajectory AUC"},
        )

        ax.set_title(f"{model} — {method}: normalized trajectory AUC for {score_label}")
        ax.set_xlabel("Prompt")
        ax.set_ylabel("Dataset")
        ax.tick_params(axis="x", rotation=35)

        fig.tight_layout()

        filename = (
            f"{safe_filename(model)}__{safe_filename(method)}__"
            f"{safe_filename(score_key)}__auc_heatmap"
        )

        save_figure(fig, out_dir / filename, save_pdf=save_pdf)


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    run_folders = list_run_folders(
        repo_root=args.repo_root,
        dataset_filter=args.dataset,
        model_filter=args.model,
        prompt_filter=args.prompt,
    )

    if not run_folders:
        raise RuntimeError("No matching run folders found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(run_folders)} matching run folders.")
    print("Loading step-level faithfulness and confidence only...")

    score_df, selected_columns = load_all_existing_step_scores(
        run_folders=run_folders,
        rcc_conf_col=args.rcc_conf_col,
    )

    selected_path = args.output_dir / "selected_score_columns.csv"
    selected_columns.to_csv(selected_path, index=False)
    print(f"Saved selected score-column report to {selected_path}")

    print("Building normalized trajectory summary...")
    norm_summary = make_normalized_summary(
        score_df=score_df,
        n_bins=args.n_bins,
        min_count=args.min_count_per_point,
    )

    norm_path = args.output_dir / "normalized_trajectory_summary.csv"
    norm_summary.to_csv(norm_path, index=False)
    print(f"Saved normalized trajectory summary to {norm_path}")

    print("Building raw-step trajectory summary...")
    raw_summary = make_raw_step_summary(
        score_df=score_df,
        raw_max_step=args.raw_max_step,
        min_count=args.min_count_per_point,
    )

    raw_path = args.output_dir / "raw_step_trajectory_summary.csv"
    raw_summary.to_csv(raw_path, index=False)
    print(f"Saved raw-step trajectory summary to {raw_path}")

    print("Computing normalized AUC summary...")
    auc_df = compute_normalized_auc(norm_summary)

    auc_path = args.output_dir / "normalized_auc_summary.csv"
    auc_df.to_csv(auc_path, index=False)
    print(f"Saved normalized AUC summary to {auc_path}")

    show_ci = not args.no_ci

    print("Generating normalized prompt-curve plots...")
    plot_normalized_prompt_curves(
        norm_summary=norm_summary,
        out_dir=plots_dir / "normalized_prompt_curves",
        save_pdf=args.save_pdf,
        show_ci=show_ci,
    )

    print("Generating normalized overview-grid plots...")
    plot_normalized_overview_grids(
        norm_summary=norm_summary,
        out_dir=plots_dir / "normalized_prompt_overview_grids",
        save_pdf=args.save_pdf,
        show_ci=show_ci,
    )

    print("Generating raw-step prompt-curve plots...")
    plot_raw_step_prompt_curves(
        raw_summary=raw_summary,
        out_dir=plots_dir / "raw_step_prompt_curves",
        save_pdf=args.save_pdf,
        show_ci=show_ci,
    )

    print("Generating AUC heatmaps...")
    plot_auc_heatmaps(
        auc_df=auc_df,
        out_dir=plots_dir / "auc_heatmaps",
        save_pdf=args.save_pdf,
    )

    print("")
    print("Done.")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Plots directory:  {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
