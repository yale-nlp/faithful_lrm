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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns


# ============================================================
# Configuration
# ============================================================

DEFAULT_REAL_RESULTS_ROOT = _DEFAULT_REAL_RESULTS_DIR
DEFAULT_OUTPUT_DIR = _default_output_dir("faithfulness_length_plots_fast")

MODEL_LABELS = _common.MODEL_FULL_LABELS

DATASET_LABELS = _common.DATASET_LABELS

PROMPT_SUFFIXES = _common.PROMPT_SUFFIX_TO_LABEL

PROMPT_LABELS = _common.PROMPT_DISPLAY_LABELS

METHODS = _common.METHOD_TUPLES

TRACE_FAITH_COLS = _common.TRACE_FAITH_COLS

STEP_FAITH_COLS = _common.STEP_FAITH_COLS

TRACE_TOKEN_COLS = _common.TRACE_TOKEN_COLS

TRACE_TEXT_COLS = _common.TRACE_TEXT_COLS

STEP_TEXT_COLS = _common.STEP_TEXT_COLS

STEP_TOKEN_COLS = _common.STEP_TOKEN_COLS

METHOD_COLORS = _common.METHOD_COLORS

DEFAULT_FIGSIZE = (20, 5.5)

GRIDSIZE_TRACE = 28
GRIDSIZE_STEP = 34
SCATTER_TRACE_SIZE = 18
SCATTER_STEP_SIZE = 9
SCATTER_TRACE_ALPHA = 0.25
SCATTER_STEP_ALPHA = 0.12

DEFAULT_MAX_DENSITY_POINTS = 300_000
DEFAULT_MAX_SCATTER_POINTS = 80_000

RANDOM_STATE = 0


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


def simple_token_count(text: Any) -> int:
    if text is None:
        return 0
    if isinstance(text, float) and math.isnan(text):
        return 0
    return len(re.findall(r"\S+", str(text)))


def vectorized_token_count(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna("")
        .astype(str)
        .str.count(r"\S+")
        .astype(float)
    )


def safe_filename(s: str) -> str:
    s = clean_text(s).replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)


def make_soft_cmap(color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f"soft_{color}",
        ["#ffffff", "#f1f3f9", color],
        N=256,
    )


def prettify_axis(ax: plt.Axes) -> None:
    ax.grid(True)
    ax.set_facecolor("white")


def drop_nan_pairs(
    x: Sequence[float],
    y: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def downsample_xy(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
    seed: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_points is None or max_points <= 0:
        return x, y

    n = len(x)
    if n <= max_points:
        return x, y

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], y[idx]


def read_excel_header(path: Path) -> List[str]:
    return list(pd.read_excel(path, nrows=0).columns)


def find_original_col(
    original_cols: Sequence[str],
    candidates: Sequence[str],
) -> Optional[str]:
    norm_to_original: Dict[str, str] = {}

    for col in original_cols:
        norm_to_original[normalize_col(col)] = col

    for c in candidates:
        c_norm = normalize_col(c)
        if c_norm in norm_to_original:
            return norm_to_original[c_norm]

    return None


def find_examples_xlsx(folder: Path) -> Optional[Path]:
    candidates = sorted(folder.glob("results_*_examples.xlsx"))
    if candidates:
        return candidates[0]

    candidates = sorted(folder.glob("*examples*.xlsx"))
    return candidates[0] if candidates else None


def find_step_level_xlsx(folder: Path) -> Optional[Path]:
    candidates = sorted(folder.glob("results_*_step_level.xlsx"))
    if candidates:
        return candidates[0]

    candidates = sorted(folder.glob("*step_level*.xlsx"))
    return candidates[0] if candidates else None


def parse_dataset_prompt_from_folder(folder_name: str) -> Tuple[str, str, str]:
    folder_name = folder_name.strip()

    prompt = "unknown"
    dataset_key = folder_name

    for suffix, prompt_name in PROMPT_SUFFIXES.items():
        if folder_name.endswith(suffix):
            prompt = prompt_name
            dataset_key = folder_name[: -len(suffix)]
            break

    dataset_label = DATASET_LABELS.get(dataset_key, dataset_key)
    prompt_label = PROMPT_LABELS.get(prompt, prompt)

    return dataset_key, dataset_label, prompt_label


def get_model_label_from_run_folder(run_folder: Path) -> str:
    model_key = run_folder.parent.name
    return MODEL_LABELS.get(model_key, model_key)


def find_run_folders(root: Path, baseline_only: bool = False) -> List[Path]:
    folders: List[Path] = []

    if not root.exists():
        return folders

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue

        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            if baseline_only and not run_dir.name.endswith("_b"):
                continue

            if find_examples_xlsx(run_dir) is None:
                continue

            folders.append(run_dir)

    return folders


def output_exists(base_path: Path, save_png: bool, save_pdf: bool) -> bool:
    required = []

    if save_png:
        required.append(base_path.with_suffix(".png"))

    if save_pdf:
        required.append(base_path.with_suffix(".pdf"))

    return all(p.exists() for p in required)


def save_figure(
    fig: plt.Figure,
    outpath_base: Path,
    save_png: bool,
    save_pdf: bool,
) -> None:
    outpath_base.parent.mkdir(parents=True, exist_ok=True)

    if save_png:
        fig.savefig(
            outpath_base.with_suffix(".png"),
            dpi=220,
            bbox_inches="tight",
        )

    if save_pdf:
        fig.savefig(
            outpath_base.with_suffix(".pdf"),
            bbox_inches="tight",
        )


# ============================================================
# Fast Excel readers
# ============================================================

def read_examples_needed_columns(path: Path) -> pd.DataFrame:
    original_cols = read_excel_header(path)

    needed_cols = set()

    for method_key in TRACE_FAITH_COLS:
        col = find_original_col(original_cols, TRACE_FAITH_COLS[method_key])
        if col is not None:
            needed_cols.add(col)

    token_col = find_original_col(original_cols, TRACE_TOKEN_COLS)
    text_col = find_original_col(original_cols, TRACE_TEXT_COLS)

    if token_col is not None:
        needed_cols.add(token_col)
    elif text_col is not None:
        needed_cols.add(text_col)

    if not needed_cols:
        return pd.DataFrame()

    df = pd.read_excel(path, usecols=list(needed_cols))
    return normalize_columns(df)


def read_steps_needed_columns(path: Path) -> pd.DataFrame:
    original_cols = read_excel_header(path)

    needed_cols = set()

    for method_key in STEP_FAITH_COLS:
        col = find_original_col(original_cols, STEP_FAITH_COLS[method_key])
        if col is not None:
            needed_cols.add(col)

    token_col = find_original_col(original_cols, STEP_TOKEN_COLS)
    text_col = find_original_col(original_cols, STEP_TEXT_COLS)

    if token_col is not None:
        needed_cols.add(token_col)
    elif text_col is not None:
        needed_cols.add(text_col)

    if not needed_cols:
        return pd.DataFrame()

    df = pd.read_excel(path, usecols=list(needed_cols))
    return normalize_columns(df)


# ============================================================
# Fast data extraction
# ============================================================

def get_trace_tokens_fast(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in TRACE_TOKEN_COLS:
        col_norm = normalize_col(col)
        if col_norm in df.columns:
            return pd.to_numeric(df[col_norm], errors="coerce").astype(float)

    for col in TRACE_TEXT_COLS:
        col_norm = normalize_col(col)
        if col_norm in df.columns:
            return vectorized_token_count(df[col_norm])

    return None


def get_step_tokens_fast(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in STEP_TOKEN_COLS:
        col_norm = normalize_col(col)
        if col_norm in df.columns:
            return pd.to_numeric(df[col_norm], errors="coerce").astype(float)

    for col in STEP_TEXT_COLS:
        col_norm = normalize_col(col)
        if col_norm in df.columns:
            return vectorized_token_count(df[col_norm])

    return None


def find_normalized_col(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        c_norm = normalize_col(c)
        if c_norm in cols:
            return c_norm
    return None


def extract_trace_data_all_methods(
    examples_df: pd.DataFrame,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if examples_df.empty:
        return [(np.array([]), np.array([])) for _ in METHODS]

    tokens = get_trace_tokens_fast(examples_df)
    if tokens is None:
        return [(np.array([]), np.array([])) for _ in METHODS]

    out: List[Tuple[np.ndarray, np.ndarray]] = []

    for method_key, _ in METHODS:
        faith_col = find_normalized_col(examples_df, TRACE_FAITH_COLS[method_key])

        if faith_col is None:
            out.append((np.array([]), np.array([])))
            continue

        faith = pd.to_numeric(examples_df[faith_col], errors="coerce").astype(float)

        mask = np.isfinite(tokens.to_numpy(dtype=float)) & np.isfinite(faith.to_numpy(dtype=float))

        out.append((
            tokens.to_numpy(dtype=float)[mask],
            faith.to_numpy(dtype=float)[mask],
        ))

    return out


def extract_step_data_all_methods(
    step_df: pd.DataFrame,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if step_df.empty:
        return [(np.array([]), np.array([])) for _ in METHODS]

    tokens = get_step_tokens_fast(step_df)
    if tokens is None:
        return [(np.array([]), np.array([])) for _ in METHODS]

    out: List[Tuple[np.ndarray, np.ndarray]] = []

    for method_key, _ in METHODS:
        faith_col = find_normalized_col(step_df, STEP_FAITH_COLS[method_key])

        if faith_col is None:
            out.append((np.array([]), np.array([])))
            continue

        faith = pd.to_numeric(step_df[faith_col], errors="coerce").astype(float)

        mask = np.isfinite(tokens.to_numpy(dtype=float)) & np.isfinite(faith.to_numpy(dtype=float))

        out.append((
            tokens.to_numpy(dtype=float)[mask],
            faith.to_numpy(dtype=float)[mask],
        ))

    return out


# ============================================================
# Plotting
# ============================================================

def plot_density_row(
    axes: np.ndarray,
    data_list: List[Tuple[np.ndarray, np.ndarray]],
    xlabel: str,
    ylabel_prefix: str,
    gridsize: int,
    max_points: int,
) -> None:
    for j, (method_key, method_label) in enumerate(METHODS):
        ax = axes[j]
        x, y = drop_nan_pairs(data_list[j][0], data_list[j][1])

        if len(x) == 0:
            ax.text(
                0.5,
                0.5,
                "No Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(method_label, pad=12)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"{ylabel_prefix} Faithfulness")
            ax.set_ylim(-0.02, 1.02)
            prettify_axis(ax)
            continue

        x, y = downsample_xy(x, y, max_points=max_points)

        hb = ax.hexbin(
            x,
            y,
            gridsize=gridsize,
            mincnt=1,
            cmap=make_soft_cmap(METHOD_COLORS[method_key]),
            linewidths=0.2,
            bins="log",
        )

        xmax = np.percentile(x, 99.5) * 1.05
        xmax = max(xmax, 1.0)

        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(method_label, pad=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{ylabel_prefix} Faithfulness")
        prettify_axis(ax)

        cb = plt.colorbar(
            hb,
            ax=ax,
            location="right",
            shrink=0.8,
            pad=0.01,
            aspect=25,
        )
        cb.set_label("Density (Log10 Counts)")


def plot_scatter_row(
    axes: np.ndarray,
    data_list: List[Tuple[np.ndarray, np.ndarray]],
    xlabel: str,
    ylabel_prefix: str,
    size: int,
    alpha: float,
    max_points: int,
) -> None:
    for j, (method_key, method_label) in enumerate(METHODS):
        ax = axes[j]
        x, y = drop_nan_pairs(data_list[j][0], data_list[j][1])

        if len(x) == 0:
            ax.text(
                0.5,
                0.5,
                "No Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(method_label, pad=12)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"{ylabel_prefix} Faithfulness")
            ax.set_ylim(-0.02, 1.02)
            prettify_axis(ax)
            continue

        x, y = downsample_xy(x, y, max_points=max_points)

        ax.scatter(
            x,
            y,
            s=size,
            alpha=alpha,
            color=METHOD_COLORS[method_key],
            edgecolors="none",
            rasterized=True,
        )

        xmax = np.percentile(x, 99.5) * 1.05
        xmax = max(xmax, 1.0)

        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(method_label, pad=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{ylabel_prefix} Faithfulness")
        prettify_axis(ax)


# ============================================================
# Main processing
# ============================================================

def process_run(
    run_folder: Path,
    examples_path: Path,
    step_path: Optional[Path],
    density_dir: Path,
    scatter_dir: Path,
    save_png: bool,
    save_pdf: bool,
    overwrite: bool,
    max_density_points: int,
    max_scatter_points: int,
    skip_scatter: bool,
    skip_density: bool,
) -> Dict[str, Any]:
    model_label = get_model_label_from_run_folder(run_folder)
    _, dataset_label, prompt_label = parse_dataset_prompt_from_folder(run_folder.name)

    paper_title = f"{dataset_label} | {model_label} | {prompt_label}"
    filename_base = safe_filename(f"{run_folder.parent.name}_{run_folder.name}")

    trace_density_base = density_dir / f"{filename_base}_trace_density"
    step_density_base = density_dir / f"{filename_base}_step_density"
    trace_scatter_base = scatter_dir / f"{filename_base}_trace_scatter"
    step_scatter_base = scatter_dir / f"{filename_base}_step_scatter"

    needed_outputs = []

    if not skip_density:
        needed_outputs.extend([trace_density_base, step_density_base])

    if not skip_scatter:
        needed_outputs.extend([trace_scatter_base, step_scatter_base])

    if not overwrite and all(output_exists(p, save_png, save_pdf) for p in needed_outputs):
        return {
            "run_folder": str(run_folder),
            "model": model_label,
            "dataset": dataset_label,
            "prompt": prompt_label,
            "examples_file": str(examples_path),
            "step_file": str(step_path) if step_path is not None else "",
            "trace_points_total": "skipped_existing",
            "step_points_total": "skipped_existing",
        }

    examples_df = read_examples_needed_columns(examples_path)
    trace_data = extract_trace_data_all_methods(examples_df)

    if step_path is not None:
        step_df = read_steps_needed_columns(step_path)
        step_data = extract_step_data_all_methods(step_df)
    else:
        step_data = [(np.array([]), np.array([])) for _ in METHODS]

    n_trace_points = int(sum(len(x) for x, _ in trace_data))
    n_step_points = int(sum(len(x) for x, _ in step_data))

    if not skip_density:
        if overwrite or not output_exists(trace_density_base, save_png, save_pdf):
            fig, axes = plt.subplots(
                1,
                3,
                figsize=DEFAULT_FIGSIZE,
                sharey=True,
                constrained_layout=True,
            )
            plot_density_row(
                axes=axes,
                data_list=trace_data,
                xlabel="Trace Length (Tokens)",
                ylabel_prefix="Trace",
                gridsize=GRIDSIZE_TRACE,
                max_points=max_density_points,
            )
            fig.suptitle(paper_title, y=1.05)
            save_figure(fig, trace_density_base, save_png=save_png, save_pdf=save_pdf)
            plt.close(fig)

        if overwrite or not output_exists(step_density_base, save_png, save_pdf):
            fig, axes = plt.subplots(
                1,
                3,
                figsize=DEFAULT_FIGSIZE,
                sharey=True,
                constrained_layout=True,
            )
            plot_density_row(
                axes=axes,
                data_list=step_data,
                xlabel="Step Length (Tokens)",
                ylabel_prefix="Step",
                gridsize=GRIDSIZE_STEP,
                max_points=max_density_points,
            )
            fig.suptitle(paper_title, y=1.05)
            save_figure(fig, step_density_base, save_png=save_png, save_pdf=save_pdf)
            plt.close(fig)

    if not skip_scatter:
        if overwrite or not output_exists(trace_scatter_base, save_png, save_pdf):
            fig, axes = plt.subplots(
                1,
                3,
                figsize=DEFAULT_FIGSIZE,
                sharey=True,
                constrained_layout=True,
            )
            plot_scatter_row(
                axes=axes,
                data_list=trace_data,
                xlabel="Trace Length (Tokens)",
                ylabel_prefix="Trace",
                size=SCATTER_TRACE_SIZE,
                alpha=SCATTER_TRACE_ALPHA,
                max_points=max_scatter_points,
            )
            fig.suptitle(paper_title, y=1.05)
            save_figure(fig, trace_scatter_base, save_png=save_png, save_pdf=save_pdf)
            plt.close(fig)

        if overwrite or not output_exists(step_scatter_base, save_png, save_pdf):
            fig, axes = plt.subplots(
                1,
                3,
                figsize=DEFAULT_FIGSIZE,
                sharey=True,
                constrained_layout=True,
            )
            plot_scatter_row(
                axes=axes,
                data_list=step_data,
                xlabel="Step Length (Tokens)",
                ylabel_prefix="Step",
                size=SCATTER_STEP_SIZE,
                alpha=SCATTER_STEP_ALPHA,
                max_points=max_scatter_points,
            )
            fig.suptitle(paper_title, y=1.05)
            save_figure(fig, step_scatter_base, save_png=save_png, save_pdf=save_pdf)
            plt.close(fig)

    return {
        "run_folder": str(run_folder),
        "model": model_label,
        "dataset": dataset_label,
        "prompt": prompt_label,
        "examples_file": str(examples_path),
        "step_file": str(step_path) if step_path is not None else "",
        "trace_points_total": n_trace_points,
        "step_points_total": n_step_points,
    }


def write_summary(summary_rows: List[Dict[str, Any]], outpath: Path) -> None:
    lines: List[str] = []

    lines.append("FAITHFULNESS VS LENGTH PLOT SUMMARY")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Number of processed runs: {len(summary_rows)}")
    lines.append("")
    lines.append("Each run can produce four figures:")
    lines.append("  - Trace Density")
    lines.append("  - Step Density")
    lines.append("  - Trace Scatter")
    lines.append("  - Step Scatter")
    lines.append("")
    lines.append("Each figure has three panels:")
    lines.append("  - RCC")
    lines.append("  - Sampling")
    lines.append("  - DeepConf")
    lines.append("")
    lines.append("Processed runs:")

    for row in summary_rows:
        lines.append(
            f"  - {row['model']} | {row['dataset']} | {row['prompt']} "
            f"| trace points={row['trace_points_total']} "
            f"| step points={row['step_points_total']}"
        )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_REAL_RESULTS_ROOT),
        help="Path to real_results directory.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only process folders ending in _b.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate plots even if output files already exist.",
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Save only PNG files. Much faster than saving PDFs too.",
    )
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Save only PDF files.",
    )
    parser.add_argument(
        "--skip-scatter",
        action="store_true",
        help="Skip scatter plots. This is the biggest speedup for large step-level files.",
    )
    parser.add_argument(
        "--skip-density",
        action="store_true",
        help="Skip density plots.",
    )
    parser.add_argument(
        "--max-density-points",
        type=int,
        default=DEFAULT_MAX_DENSITY_POINTS,
        help="Maximum points used per density panel. Set 0 for no downsampling.",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=DEFAULT_MAX_SCATTER_POINTS,
        help="Maximum points used per scatter panel. Set 0 for no downsampling.",
    )

    args, _ = parser.parse_known_args()

    root = Path(args.root)
    output_dir = Path(args.outdir)
    density_dir = output_dir / "density"
    scatter_dir = output_dir / "scatter"

    save_png = True
    save_pdf = True

    if args.png_only:
        save_pdf = False

    if args.pdf_only:
        save_png = False

    density_dir.mkdir(parents=True, exist_ok=True)
    scatter_dir.mkdir(parents=True, exist_ok=True)

    folders = find_run_folders(
        root=root,
        baseline_only=bool(args.baseline_only),
    )

    if not folders:
        print(f"No matching run folders found under: {root.resolve()}")
        return

    summary_rows: List[Dict[str, Any]] = []

    for i, run_folder in enumerate(folders, start=1):
        examples_path = find_examples_xlsx(run_folder)
        step_path = find_step_level_xlsx(run_folder)

        if examples_path is None:
            print(f"[SKIP] No examples file: {run_folder}")
            continue

        print(f"[{i}/{len(folders)}] {run_folder}")

        row = process_run(
            run_folder=run_folder,
            examples_path=examples_path,
            step_path=step_path,
            density_dir=density_dir,
            scatter_dir=scatter_dir,
            save_png=save_png,
            save_pdf=save_pdf,
            overwrite=bool(args.overwrite),
            max_density_points=int(args.max_density_points),
            max_scatter_points=int(args.max_scatter_points),
            skip_scatter=bool(args.skip_scatter),
            skip_density=bool(args.skip_density),
        )

        summary_rows.append(row)

    write_summary(
        summary_rows=summary_rows,
        outpath=output_dir / "summary.txt",
    )

    print("")
    print("Done.")
    print(f"Input:  {root.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print("")
    print("Saved folders:")
    print(f"  {density_dir}")
    print(f"  {scatter_dir}")
    print("")
    print("Summary:")
    print(f"  {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()