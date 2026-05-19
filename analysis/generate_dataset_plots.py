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
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

PLOT_DPI = 300


# ============================================================
# Constants
# ============================================================

DATASET_LABELS = _common.DATASET_LABELS

DATASET_ORDER = _common.DATASET_ORDER

MODEL_LABELS_FROM_RUN = _common.MODEL_FULL_LABELS

MODEL_LABELS_FROM_MODEL_LINE = _common.MODEL_LABELS_FROM_MODEL_LINE

MODEL_ORDER = _common.MODEL_ORDER_FULL
METHOD_ORDER = ["RCC", "DeepConf", "Sampling"]

COLORS = {
    "DeepSeek-R1-8B": "#4C78A8",
    "QwQ-32B": "#E15759",
    "RCC": "#4C78A8",
    "DeepConf": "#9C6ADE",
    "Sampling": "#59A14F",
}


# ============================================================
# Helpers
# ============================================================

def prettify_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.7)
    ax.grid(False, axis="x")
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")


def sem(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def parse_run_field(run: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Expected examples:
        main_benchmarks/aime_ds_b
        main_benchmarks/aime_ds_perc
        main_benchmarks/aime_ds_msh_perc
        main_benchmarks/hle_qwq_b
        main_benchmarks/hle_qwq_32b_perc
    """
    run_tail = run.strip().split("/")[-1]

    prompt_key = None
    for suffix in ["_msh_perc", "_perc", "_b"]:
        if run_tail.endswith(suffix):
            prompt_key = suffix[1:]
            stem = run_tail[:-len(suffix)]
            break
    else:
        stem = run_tail

    model_key = None
    dataset_key = None

    possible_model_keys = sorted(MODEL_LABELS_FROM_RUN.keys(), key=len, reverse=True)

    for mk in possible_model_keys:
        suffix = "_" + mk
        if stem.endswith(suffix):
            model_key = mk
            dataset_key = stem[:-len(suffix)]
            break

    return model_key, dataset_key, prompt_key


def normalize_dataset(dataset: Optional[str]) -> Optional[str]:
    if dataset is None:
        return None
    key = str(dataset).strip().lower()
    return DATASET_LABELS.get(key, dataset)


def normalize_model_from_run_key(model_key: Optional[str]) -> Optional[str]:
    if model_key is None:
        return None
    return MODEL_LABELS_FROM_RUN.get(model_key)


def normalize_model_from_model_line(model_line: Optional[str]) -> Optional[str]:
    if model_line is None:
        return None

    key = str(model_line).strip().lower()

    if key in MODEL_LABELS_FROM_MODEL_LINE:
        return MODEL_LABELS_FROM_MODEL_LINE[key]

    if "deepseek-r1" in key and "8b" in key:
        return "DeepSeek-R1-8B"

    if "qwq" in key and "32b" in key:
        return "QwQ-32B"

    return None


def normalize_prompt(prompt_key: Optional[str]) -> Optional[str]:
    prompt_map = {
        "b": "baseline",
        "perc": "perception",
        "msh_perc": "msh+perception",
    }
    if prompt_key is None:
        return None
    return prompt_map.get(prompt_key, prompt_key)


def parse_method_block(lines: List[str], start_idx: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Parses:
        Faithfulness (RCC): 0.790
          MFG=0.788, cMFG=0.788, cMFG*=0.788
    """
    line = lines[start_idx].strip()

    mean_match = re.search(r":\s*([0-9]*\.?[0-9]+)", line)
    mean_faith = float(mean_match.group(1)) if mean_match else None

    cmfg_star = None
    if start_idx + 1 < len(lines):
        next_line = lines[start_idx + 1].strip()
        cmfg_star_match = re.search(r"cMFG\*\s*=\s*([0-9]*\.?[0-9]+)", next_line)
        if cmfg_star_match:
            cmfg_star = float(cmfg_star_match.group(1))

    return mean_faith, cmfg_star


def parse_summary_txt(path: Path) -> Optional[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    data = {
        "summary_path": str(path),
        "run": None,
        "model_raw": None,
        "dataset_raw": None,
        "decisiveness": None,
        "faith_rcc": None,
        "faith_dc": None,
        "faith_samp": None,
        "cmfg_star_rcc": None,
        "cmfg_star_dc": None,
        "cmfg_star_samp": None,
    }

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()

        if line.startswith("Run:"):
            data["run"] = line.split("Run:", 1)[1].strip()

        elif line.startswith("Model:"):
            data["model_raw"] = line.split("Model:", 1)[1].strip()

        elif line.startswith("Dataset:"):
            data["dataset_raw"] = line.split("Dataset:", 1)[1].strip()

        elif line.startswith("Decisiveness:"):
            m = re.search(r"Decisiveness:\s*([0-9]*\.?[0-9]+)", line)
            if m:
                data["decisiveness"] = float(m.group(1))

        elif line.startswith("Faithfulness (RCC):"):
            mean_faith, cmfg_star = parse_method_block(lines, i)
            data["faith_rcc"] = mean_faith
            data["cmfg_star_rcc"] = cmfg_star

        elif line.startswith("Faithfulness (DeepConf):"):
            mean_faith, cmfg_star = parse_method_block(lines, i)
            data["faith_dc"] = mean_faith
            data["cmfg_star_dc"] = cmfg_star

        elif line.startswith("Faithfulness (Sampling):"):
            mean_faith, cmfg_star = parse_method_block(lines, i)
            data["faith_samp"] = mean_faith
            data["cmfg_star_samp"] = cmfg_star

    if data["run"] is None and data["dataset_raw"] is None and data["model_raw"] is None:
        return None

    model_key, dataset_key, prompt_key = parse_run_field(data["run"] or "")

    data["model"] = (
        normalize_model_from_run_key(model_key)
        or normalize_model_from_model_line(data["model_raw"])
    )
    data["dataset"] = normalize_dataset(dataset_key) or normalize_dataset(data["dataset_raw"])
    data["prompt"] = normalize_prompt(prompt_key)

    return data


def load_from_real_results(real_results_dir: Path) -> pd.DataFrame:
    summary_paths = sorted(real_results_dir.rglob("summary.txt"))

    if not summary_paths:
        raise FileNotFoundError(
            f"No summary.txt files found under {real_results_dir.resolve()}"
        )

    rows = []
    for path in summary_paths:
        parsed = parse_summary_txt(path)
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        raise RuntimeError("summary.txt files were found, but none could be parsed.")

    df = pd.DataFrame(rows)

    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df = df[df["dataset"].isin(DATASET_ORDER)].copy()

    numeric_cols = [
        "decisiveness",
        "faith_rcc",
        "faith_dc",
        "faith_samp",
        "cmfg_star_rcc",
        "cmfg_star_dc",
        "cmfg_star_samp",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "dataset",
            "model",
            "decisiveness",
            "cmfg_star_rcc",
            "cmfg_star_dc",
            "cmfg_star_samp",
        ]
    )

    return df


def make_summaries(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dec_summary = (
        df.groupby(["dataset", "model"], as_index=False)
        .agg(
            mean_decisiveness=("decisiveness", "mean"),
            sem_decisiveness=("decisiveness", sem),
            n_runs=("decisiveness", "count"),
        )
    )

    cmfg_long = df.melt(
        id_vars=["dataset", "model", "prompt"],
        value_vars=["cmfg_star_rcc", "cmfg_star_dc", "cmfg_star_samp"],
        var_name="method_col",
        value_name="cmfg_star",
    )

    method_map = {
        "cmfg_star_rcc": "RCC",
        "cmfg_star_dc": "DeepConf",
        "cmfg_star_samp": "Sampling",
    }

    cmfg_long["method"] = cmfg_long["method_col"].map(method_map)

    cmfg_summary = (
        cmfg_long.groupby(["dataset", "method"], as_index=False)
        .agg(
            mean_cmfg_star=("cmfg_star", "mean"),
            sem_cmfg_star=("cmfg_star", sem),
            n_runs=("cmfg_star", "count"),
        )
    )

    return dec_summary, cmfg_summary


def plot_dataset_dec_cmfg(
    dec_summary: pd.DataFrame,
    cmfg_summary: pd.DataFrame,
    output_prefix: str,
) -> None:
    datasets = [d for d in DATASET_ORDER if d in set(dec_summary["dataset"])]
    x = np.arange(len(datasets))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.25), sharey=False)

    # =======================================================
    # Panel A: Decisiveness
    # =======================================================
    ax = axes[0]
    width = 0.34
    offsets = np.linspace(-width / 2, width / 2, len(MODEL_ORDER))

    for offset, model in zip(offsets, MODEL_ORDER):
        sub = dec_summary[dec_summary["model"] == model].set_index("dataset")
        vals = [sub.loc[d, "mean_decisiveness"] if d in sub.index else np.nan for d in datasets]
        errs = [sub.loc[d, "sem_decisiveness"] if d in sub.index else 0.0 for d in datasets]

        ax.bar(
            x + offset,
            vals,
            yerr=errs,
            width=width,
            label=model,
            color=COLORS.get(model, "#777777"),
            alpha=0.88,
            edgecolor="#222222",
            linewidth=0.3,
            capsize=3,
        )

    ax.set_title("(a) Linguistic Decisiveness")
    ax.set_ylabel("Mean Decisiveness")
    ax.set_xlabel("Dataset")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0, ha="center")
    ax.legend(frameon=True, loc="best", title="Model")
    prettify_axis(ax)

    # =======================================================
    # Panel B: cMFG*
    # =======================================================
    ax = axes[1]

    for method in METHOD_ORDER:
        sub = cmfg_summary[cmfg_summary["method"] == method].set_index("dataset")
        vals = [sub.loc[d, "mean_cmfg_star"] if d in sub.index else np.nan for d in datasets]
        errs = [sub.loc[d, "sem_cmfg_star"] if d in sub.index else 0.0 for d in datasets]

        ax.errorbar(
            x,
            vals,
            yerr=errs,
            label=method,
            color=COLORS.get(method, "#777777"),
            marker="o",
            markersize=5,
            linewidth=2.1,
            capsize=3,
        )

    ax.set_title("(b) cMFG$^*$ by Estimator")
    ax.set_ylabel("Mean cMFG$^*$")
    ax.set_xlabel("Dataset")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0, ha="center")
    ax.legend(frameon=True, loc="best", title="Estimator")
    prettify_axis(ax)

    # No overall figure title. Subplot titles are kept.
    fig.tight_layout()

    fig.savefig(f"{output_prefix}.png", dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--real-results-dir", type=Path, default=_DEFAULT_REAL_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir("dataset_plots"))
    args, _ = parser.parse_known_args()

    real_results_dir = args.real_results_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not real_results_dir.is_dir():
        raise FileNotFoundError(f"Could not find directory: {real_results_dir.resolve()}")

    df = load_from_real_results(real_results_dir)
    dec_summary, cmfg_summary = make_summaries(df)

    df.to_csv(output_dir / "figure1_parsed_summary_from_txt.csv", index=False)
    dec_summary.to_csv(output_dir / "dataset_decisiveness_summary.csv", index=False)
    cmfg_summary.to_csv(output_dir / "dataset_cmfg_star_summary.csv", index=False)

    plot_dataset_dec_cmfg(
        dec_summary=dec_summary,
        cmfg_summary=cmfg_summary,
        output_prefix=str(output_dir / "dataset_decisiveness_faithfulness"),
    )

    print(f"Parsed {len(df)} usable summary.txt runs from {real_results_dir}")
    print(f"Saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()