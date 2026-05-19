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
# Plot style: matched to your existing scripts
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
# Labels / ordering
# ============================================================

DATASET_LABELS = _common.DATASET_LABELS

MODEL_LABELS_FROM_RUN = _common.MODEL_FULL_LABELS

MODEL_LABELS_FROM_MODEL_LINE = _common.MODEL_LABELS_FROM_MODEL_LINE

PROMPT_LABELS = {"b": "baseline", "blank": "baseline", "perc": "perception", "msh_perc": "msh+perception"}

PROMPT_ORDER = ["perception", "msh+perception"]

PROMPT_COLORS = {
    "perception": "#59A14F",
    "msh+perception": "#9C6ADE",
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


def normalize_dataset(dataset_key: Optional[str]) -> Optional[str]:
    if dataset_key is None:
        return None
    key = str(dataset_key).strip().lower()
    return DATASET_LABELS.get(key, dataset_key)


def normalize_prompt(prompt_key: Optional[str]) -> Optional[str]:
    if prompt_key is None:
        return None
    key = str(prompt_key).strip().lower()
    return PROMPT_LABELS.get(key, prompt_key)


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


def parse_run_field(run: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Supports run formats such as:
        main_benchmarks/aime_ds_b
        main_benchmarks/aime_ds_perc
        main_benchmarks/aime_ds_msh_perc
        main_benchmarks/sgpqa_qwq_b
        ds_8b/aime_b
        qwq_32b/sgpqa_msh_perc
    """
    run = str(run).strip()

    if "/" in run:
        first, tail = run.split("/", 1)
        if first in MODEL_LABELS_FROM_RUN:
            model_key = first
            tail = tail.split("/")[-1]

            for prompt_key in ["msh_perc", "perc", "b"]:
                suffix = f"_{prompt_key}"
                if tail.endswith(suffix):
                    dataset_key = tail[:-len(suffix)]
                    return model_key, dataset_key, prompt_key

    tail = run.split("/")[-1]

    prompt_key = None
    for suffix in ["_msh_perc", "_perc", "_b"]:
        if tail.endswith(suffix):
            prompt_key = suffix[1:]
            stem = tail[:-len(suffix)]
            break
    else:
        stem = tail

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


def parse_metric_line(lines: List[str], i: int) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Parses:
        Faithfulness (RCC): 0.790
          MFG=0.788, cMFG=0.788, cMFG*=0.788
    """
    mean_faith = None
    mfg = None
    cmfg = None
    cmfg_star = None

    mean_match = re.search(r":\s*([0-9]*\.?[0-9]+)", lines[i])
    if mean_match:
        mean_faith = float(mean_match.group(1))

    if i + 1 < len(lines):
        next_line = lines[i + 1].strip()

        mfg_match = re.search(r"\bMFG\s*=\s*([0-9]*\.?[0-9]+)", next_line)
        cmfg_match = re.search(r"\bcMFG\s*=\s*([0-9]*\.?[0-9]+)", next_line)
        cmfg_star_match = re.search(r"\bcMFG\*\s*=\s*([0-9]*\.?[0-9]+)", next_line)

        if mfg_match:
            mfg = float(mfg_match.group(1))
        if cmfg_match:
            cmfg = float(cmfg_match.group(1))
        if cmfg_star_match:
            cmfg_star = float(cmfg_star_match.group(1))

    return mean_faith, mfg, cmfg, cmfg_star


def parse_summary_txt(path: Path) -> Optional[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    data = {
        "summary_path": str(path),
        "run": None,
        "model_raw": None,
        "dataset_raw": None,
        "prompt_raw": None,
        "accuracy": None,
        "decisiveness": None,
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

        elif line.startswith("Hedge:"):
            data["prompt_raw"] = line.split("Hedge:", 1)[1].strip()

        elif line.startswith("Accuracy:"):
            m = re.search(r"Accuracy:\s*([0-9]*\.?[0-9]+)", line)
            if m:
                data["accuracy"] = float(m.group(1))

        elif line.startswith("Decisiveness:"):
            m = re.search(r"Decisiveness:\s*([0-9]*\.?[0-9]+)", line)
            if m:
                data["decisiveness"] = float(m.group(1))

        elif line.startswith("Faithfulness (RCC):"):
            _, _, _, cmfg_star = parse_metric_line(lines, i)
            data["cmfg_star_rcc"] = cmfg_star

        elif line.startswith("Faithfulness (DeepConf):"):
            _, _, _, cmfg_star = parse_metric_line(lines, i)
            data["cmfg_star_dc"] = cmfg_star

        elif line.startswith("Faithfulness (Sampling):"):
            _, _, _, cmfg_star = parse_metric_line(lines, i)
            data["cmfg_star_samp"] = cmfg_star

    if data["run"] is None:
        data["run"] = path.parent.name

    model_key, dataset_key, prompt_key = parse_run_field(data["run"])

    model = (
        MODEL_LABELS_FROM_RUN.get(model_key)
        or normalize_model_from_model_line(data["model_raw"])
    )
    dataset = normalize_dataset(dataset_key) or normalize_dataset(data["dataset_raw"])
    prompt = normalize_prompt(prompt_key) or normalize_prompt(data["prompt_raw"])

    data["model"] = model
    data["dataset"] = dataset
    data["prompt"] = prompt

    if model is None or dataset is None or prompt is None:
        print(f"[WARN] Could not fully parse metadata for {path}")
        return None

    return data


def load_from_real_results(real_results_dir: Path) -> pd.DataFrame:
    summary_paths = sorted(real_results_dir.rglob("summary.txt"))

    if not summary_paths:
        raise FileNotFoundError(f"No summary.txt files found under {real_results_dir.resolve()}")

    rows = []
    for path in summary_paths:
        parsed = parse_summary_txt(path)
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        raise RuntimeError("summary.txt files were found, but none could be parsed.")

    df = pd.DataFrame(rows)

    numeric_cols = [
        "accuracy",
        "decisiveness",
        "cmfg_star_rcc",
        "cmfg_star_dc",
        "cmfg_star_samp",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "model",
            "dataset",
            "prompt",
            "accuracy",
            "decisiveness",
            "cmfg_star_rcc",
            "cmfg_star_dc",
            "cmfg_star_samp",
        ]
    ).copy()

    df["mean_cmfg_star"] = df[
        ["cmfg_star_rcc", "cmfg_star_dc", "cmfg_star_samp"]
    ].mean(axis=1)

    return df


def compute_prompt_deltas(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "decisiveness", "mean_cmfg_star"]

    base = (
        df[df["prompt"] == "baseline"]
        .set_index(["model", "dataset"])[metric_cols]
    )

    rows = []

    for prompt in PROMPT_ORDER:
        sub = df[df["prompt"] == prompt].set_index(["model", "dataset"])
        common_index = sub.index.intersection(base.index)

        if len(common_index) == 0:
            print(f"[WARN] No baseline-matched cells found for prompt: {prompt}")
            continue

        sub = sub.loc[common_index]
        b = base.loc[common_index]

        delta = sub[metric_cols] - b[metric_cols]
        delta = delta.reset_index()
        delta["prompt"] = prompt
        rows.append(delta)

    if not rows:
        raise RuntimeError("No prompt deltas could be computed.")

    out = pd.concat(rows, ignore_index=True)

    out = out.rename(columns={
        "accuracy": "delta_accuracy",
        "decisiveness": "delta_decisiveness",
        "mean_cmfg_star": "delta_cmfg_star",
    })

    return out


def summarize_deltas(delta_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        delta_df.groupby("prompt", as_index=False)
        .agg(
            delta_accuracy_mean=("delta_accuracy", "mean"),
            delta_accuracy_sem=("delta_accuracy", sem),
            delta_decisiveness_mean=("delta_decisiveness", "mean"),
            delta_decisiveness_sem=("delta_decisiveness", sem),
            delta_cmfg_star_mean=("delta_cmfg_star", "mean"),
            delta_cmfg_star_sem=("delta_cmfg_star", sem),
            n_cells=("delta_accuracy", "count"),
        )
    )

    return summary


def compute_shared_ylim(panel_data: List[Tuple[List[float], List[float]]]) -> Tuple[float, float]:
    lowers = []
    uppers = []

    for vals, errs in panel_data:
        vals = np.asarray(vals, dtype=float)
        errs = np.asarray(errs, dtype=float)
        valid = np.isfinite(vals) & np.isfinite(errs)
        if valid.any():
            lowers.extend((vals[valid] - errs[valid]).tolist())
            uppers.extend((vals[valid] + errs[valid]).tolist())

    if not lowers or not uppers:
        return -0.01, 0.05

    y_min = min(lowers)
    y_max = max(uppers)

    # Keep zero visible.
    y_min = min(y_min, 0.0)
    y_max = max(y_max, 0.0)

    span = y_max - y_min
    pad = max(0.003, 0.08 * span)

    return y_min - pad, y_max + pad


def plot_prompt_effect(summary: pd.DataFrame, output_prefix: str) -> None:
    metric_specs = [
        ("delta_accuracy_mean", "delta_accuracy_sem", r"$\Delta$ Accuracy"),
        ("delta_decisiveness_mean", "delta_decisiveness_sem", r"$\Delta$ Decisiveness"),
        ("delta_cmfg_star_mean", "delta_cmfg_star_sem", r"$\Delta$ Mean cMFG$^*$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharey=True)

    x = np.arange(len(PROMPT_ORDER))

    panel_values = []
    for mean_col, sem_col, _ in metric_specs:
        vals = []
        errs = []
        for prompt in PROMPT_ORDER:
            row = summary[summary["prompt"] == prompt]
            if row.empty:
                vals.append(np.nan)
                errs.append(0.0)
            else:
                vals.append(float(row.iloc[0][mean_col]))
                errs.append(float(row.iloc[0][sem_col]))
        panel_values.append((vals, errs))

    shared_ymin, shared_ymax = compute_shared_ylim(panel_values)

    for idx, (ax, (mean_col, sem_col, title), (vals, errs)) in enumerate(
        zip(axes, metric_specs, panel_values)
    ):
        ax.bar(
            x,
            vals,
            yerr=errs,
            color=[PROMPT_COLORS[p] for p in PROMPT_ORDER],
            alpha=0.88,
            edgecolor="#222222",
            linewidth=0.3,
            capsize=3,
        )

        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(PROMPT_ORDER, rotation=0, ha="center")
        ax.set_xlabel("Prompt Intervention")
        ax.set_ylim(shared_ymin, shared_ymax)

        if idx == 0:
            ax.set_ylabel("Change Relative to Baseline")
        else:
            ax.set_ylabel("")

        prettify_axis(ax)

    fig.tight_layout()

    fig.savefig(f"{output_prefix}.png", dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--real-results-dir", type=Path, default=_DEFAULT_REAL_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir("prompt_plots"))
    args, _ = parser.parse_known_args()

    real_results_dir = args.real_results_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not real_results_dir.is_dir():
        raise FileNotFoundError(f"Could not find directory: {real_results_dir.resolve()}")

    df = load_from_real_results(real_results_dir)
    delta_df = compute_prompt_deltas(df)
    summary = summarize_deltas(delta_df)

    df.to_csv(output_dir / "prompt_effect_parsed_summary_from_txt.csv", index=False)
    delta_df.to_csv(output_dir / "prompt_effect_delta_by_cell.csv", index=False)
    summary.to_csv(output_dir / "prompt_effect_delta_summary.csv", index=False)

    plot_prompt_effect(
        summary=summary,
        output_prefix=str(output_dir / "prompt_effect_relative_to_baseline"),
    )

    print(f"Parsed {len(df)} usable summary.txt runs from {real_results_dir}")
    print(f"Saved outputs to {output_dir.resolve()}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()