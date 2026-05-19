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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Folder / label configuration
# ============================================================

REAL_RESULTS_DIR = _DEFAULT_REAL_RESULTS_DIR


DATASET_LABELS = _common.DATASET_LABELS

MODEL_LABELS = _common.MODEL_FULL_LABELS

MODEL_ORDER = _common.MODEL_ORDER_FULL

PROMPT_LABELS = _common.PROMPT_LABELS

# Longest suffixes first.
PROMPT_SUFFIXES = _common.PROMPT_SUFFIXES

DATASET_ORDER = _common.DATASET_ORDER_WITH_LEGACY

PROMPT_ORDER = _common.PROMPT_ORDER


# ============================================================
# Methods and columns
# ============================================================

METHODS = _common.METHODS_CANDIDATES

METHOD_ORDER = _common.METHOD_ORDER
METHOD_LABEL_ORDER = [METHODS[k]["label"] for k in METHOD_ORDER]

DECISIVENESS_CANDIDATES = _common.DECISIVENESS_CANDIDATES

CORRECT_CANDIDATES = _common.CORRECT_CANDIDATES

GAP_BIN_ORDER = ["Aligned", "Moderate Mismatch", "Strong Mismatch"]

GAP_BIN_COLORS = {
    "Aligned": "#A6CEE3",
    "Moderate Mismatch": "#FDBF6F",
    "Strong Mismatch": "#E31A1C",
}

DIRECTION_ORDER = [
    "Decisiveness > Confidence",
    "Confidence > Decisiveness",
    "Near Tie",
]

DIRECTION_COLORS = {
    "Decisiveness > Confidence": "#E15759",
    "Confidence > Decisiveness": "#4C78A8",
    "Near Tie": "#9D9D9D",
}


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
    "grid.color": "#e3e3e3",
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

PLOT_DPI = 300


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
        help="Path to real_results. Relative paths are resolved under --repo-root.",
    )
    parser.add_argument(
        "--run-folder",
        type=str,
        default=None,
        help=(
            "Optional single run folder, e.g. ds_8b/aime_b, qwq_32b/sgpqa_msh_perc, "
            "or just aime_b."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir("confidence_decisiveness_gap_analysis"),
    )

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)

    parser.add_argument(
        "--aligned-quantile",
        type=float,
        default=0.25,
        help="Bottom quantile of absolute gap used for the Aligned bin.",
    )
    parser.add_argument(
        "--strong-quantile",
        type=float,
        default=0.75,
        help="Upper quantile of absolute gap used for the Strong Mismatch bin.",
    )
    parser.add_argument(
        "--direction-tolerance",
        type=float,
        default=0.02,
        help="Tolerance around signed-gap zero used for the Near Tie direction.",
    )
    parser.add_argument(
        "--cmfg-bins",
        type=int,
        default=10,
        help="Number of equal-mass confidence bins used to compute cMFG*.",
    )
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="Also save PDF versions of all plots.",
    )

    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"[INFO] Ignored unknown Colab/Jupyter arguments: {unknown}")

    if not (0 < args.aligned_quantile < 1):
        raise ValueError("--aligned-quantile must be in (0, 1)")

    if not (0 < args.strong_quantile < 1):
        raise ValueError("--strong-quantile must be in (0, 1)")

    if args.aligned_quantile >= args.strong_quantile:
        raise ValueError("--aligned-quantile must be smaller than --strong-quantile")

    if args.direction_tolerance < 0:
        raise ValueError("--direction-tolerance must be non-negative")

    if args.cmfg_bins < 1:
        raise ValueError("--cmfg-bins must be at least 1")

    return args


# ============================================================
# Basic helpers
# ============================================================

TRUE_STRINGS = _common.TRUE_STRINGS
FALSE_STRINGS = _common.FALSE_STRINGS


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


def safe_filename(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)


def normalize_dataset_key(dataset_key: Optional[str]) -> Optional[str]:
    if dataset_key is None:
        return None
    key = clean_text(dataset_key).lower()
    return DATASET_LABELS.get(key, dataset_key)


def normalize_model_key(model_key: Optional[str]) -> Optional[str]:
    if model_key is None:
        return None
    key = clean_text(model_key).lower()
    return MODEL_LABELS.get(key, model_key)


def normalize_prompt_key(prompt_key: Optional[str]) -> Optional[str]:
    if prompt_key is None:
        return None
    key = clean_text(prompt_key).lower().replace("-", "_").replace(" ", "_")
    return PROMPT_LABELS.get(key, prompt_key)


def parse_run_name(run_name: str) -> Tuple[str, str]:
    """Return (dataset_key, prompt_key) from names such as aime_b or sgpqa_msh_perc."""
    run_name = clean_text(run_name)

    for suffix in PROMPT_SUFFIXES:
        if run_name.endswith(suffix):
            dataset_key = run_name[: -len(suffix)]
            prompt_key = suffix[1:]
            return dataset_key, prompt_key

    return run_name, "unknown"


def parse_run_metadata(run_folder: Path) -> Dict[str, str]:
    """Parse metadata from corrected nested format, for the nested run-folder format."""
    folder_name = run_folder.name
    parent_name = run_folder.parent.name

    model_key = parent_name
    dataset_key, prompt_key = parse_run_name(folder_name)

    dataset = normalize_dataset_key(dataset_key) or dataset_key
    model = normalize_model_key(model_key) or model_key
    prompt = normalize_prompt_key(prompt_key) or prompt_key

    return {
        "dataset_key": dataset_key,
        "dataset": dataset,
        "model_key": model_key,
        "model": model,
        "model_full": model,
        "prompt_key": prompt_key,
        "prompt": prompt,
        "run_name": folder_name,
    }


def filters_match(
    meta: Dict[str, str],
    dataset_filter: Optional[str],
    model_filter: Optional[str],
    prompt_filter: Optional[str],
) -> bool:
    if dataset_filter is not None:
        d = dataset_filter.lower()
        if d not in {meta["dataset_key"].lower(), meta["dataset"].lower()}:
            return False

    if model_filter is not None:
        m = model_filter.lower()
        if m not in {meta["model_key"].lower(), meta["model"].lower()}:
            return False

    if prompt_filter is not None:
        p = prompt_filter.lower()
        if p not in {meta["prompt_key"].lower(), meta["prompt"].lower()}:
            return False

    return True


def find_examples_xlsx(folder: Path) -> Optional[Path]:
    candidates = sorted(folder.glob("results_*_examples.xlsx"))
    if candidates:
        return candidates[0]

    candidates = sorted(p for p in folder.glob("*.xlsx") if "examples" in p.name.lower())
    return candidates[0] if candidates else None


def list_run_folders(
    repo_root: Path,
    real_results_dir: Path,
    run_folder: Optional[str],
    dataset_filter: Optional[str],
    model_filter: Optional[str],
    prompt_filter: Optional[str],
) -> List[Path]:
    repo_root = repo_root.resolve()
    real_root = real_results_dir if real_results_dir.is_absolute() else repo_root / real_results_dir

    if run_folder is not None:
        raw = Path(run_folder)
        candidates: List[Path] = []

        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(repo_root / raw)
            candidates.append(real_root / raw)
            if len(raw.parts) == 1:
                candidates.extend(sorted(real_root.glob(f"*/{raw.name}")))

        for p in candidates:
            if p.is_dir() and find_examples_xlsx(p) is not None:
                meta = parse_run_metadata(p)
                if filters_match(meta, dataset_filter, model_filter, prompt_filter):
                    return [p]

        raise FileNotFoundError(
            f"Could not find run folder '{run_folder}' with an examples file. Tried:\n"
            + "\n".join(f"  - {p}" for p in candidates)
        )

    folders: List[Path] = []

    if real_root.is_dir():
        for model_dir in sorted(real_root.iterdir()):
            if not model_dir.is_dir():
                continue

            for run_dir in sorted(model_dir.iterdir()):
                if not run_dir.is_dir():
                    continue

                if find_examples_xlsx(run_dir) is None:
                    continue

                meta = parse_run_metadata(run_dir)
                if filters_match(meta, dataset_filter, model_filter, prompt_filter):
                    folders.append(run_dir)


    seen = set()
    out = []
    for p in folders:
        key = str(p.resolve())
        if key not in seen:
            out.append(p)
            seen.add(key)

    return out


def clip01(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None

    if not np.isfinite(v):
        return None

    return float(np.clip(v, 0.0, 1.0))


def percentile_binning_text(aligned_quantile: float, strong_quantile: float) -> str:
    aligned_pct = int(round(100 * aligned_quantile))
    moderate_pct = int(round(100 * (strong_quantile - aligned_quantile)))
    strong_pct = int(round(100 * (1.0 - strong_quantile)))

    return (
        "Relative Absolute-Gap Bins Per Method: "
        f"Aligned Bottom {aligned_pct}%, "
        f"Moderate Middle {moderate_pct}%, "
        f"Strong Top {strong_pct}%"
    )


def gap_bin_title(
    bin_name: str,
    aligned_quantile: float,
    strong_quantile: float,
) -> str:
    aligned_pct = int(round(100 * aligned_quantile))
    strong_pct = int(round(100 * (1.0 - strong_quantile)))

    if bin_name == "Aligned":
        return f"Aligned Samples, Bottom {aligned_pct}% Absolute Confidence–Decisiveness Gap"

    if bin_name == "Moderate Mismatch":
        return "Moderate Confidence–Decisiveness Mismatch"

    return f"Strong Mismatch, Top {strong_pct}% Absolute Confidence–Decisiveness Gap"


def direction_label(signed_gap: float, tolerance: float) -> str:
    if abs(signed_gap) <= tolerance:
        return "Near Tie"

    if signed_gap > 0:
        return "Decisiveness > Confidence"

    return "Confidence > Decisiveness"


# ============================================================
# cMFG* computation
# ============================================================

def compute_cmfg_star(
    confidence: Sequence[Any],
    faithfulness: Sequence[Any],
    n_bins: int = 10,
) -> float:
    """
    Compute width-weighted conditional MFG, cMFG*.

    Inputs are example-level confidence C(T) and example-level faithfulness F_C(T).
    This is not a step-level metric.
    """
    conf = pd.to_numeric(pd.Series(confidence), errors="coerce").to_numpy(dtype=float)
    faith = pd.to_numeric(pd.Series(faithfulness), errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(conf) & np.isfinite(faith)
    conf = conf[mask]
    faith = faith[mask]

    if len(conf) == 0:
        return np.nan

    conf = np.clip(conf, 0.0, 1.0)
    faith = np.clip(faith, 0.0, 1.0)

    if len(conf) == 1:
        return float(faith[0])

    order = np.argsort(conf, kind="mergesort")
    conf = conf[order]
    faith = faith[order]

    k = max(1, min(int(n_bins), len(conf)))
    index_bins = np.array_split(np.arange(len(conf)), k)

    bin_records: List[Dict[str, float]] = []

    for idxs in index_bins:
        if len(idxs) == 0:
            continue

        bin_records.append({
            "c_min": float(conf[idxs[0]]),
            "c_max": float(conf[idxs[-1]]),
            "f_mean": float(np.mean(faith[idxs])),
        })

    if not bin_records:
        return np.nan

    weighted_sum = 0.0
    width_sum = 0.0

    for j, rec in enumerate(bin_records):
        if j == 0:
            left = rec["c_min"]
        else:
            left = 0.5 * (bin_records[j - 1]["c_max"] + rec["c_min"])

        if j == len(bin_records) - 1:
            right = rec["c_max"]
        else:
            right = 0.5 * (rec["c_max"] + bin_records[j + 1]["c_min"])

        width = max(0.0, float(right - left))

        weighted_sum += width * rec["f_mean"]
        width_sum += width

    if width_sum <= 0:
        return float(np.mean(faith))

    return float(weighted_sum / width_sum)


# ============================================================
# Data collection
# ============================================================

def collect_gap_rows(
    run_folders: Sequence[Path],
    direction_tolerance: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: List[Dict[str, Any]] = []
    run_summary_rows: List[Dict[str, Any]] = []

    for folder in run_folders:
        meta = parse_run_metadata(folder)
        xlsx_path = find_examples_xlsx(folder)

        if xlsx_path is None:
            print(f"[SKIP] {folder}: no *examples*.xlsx found")
            continue

        try:
            df_raw = pd.read_excel(xlsx_path)
        except Exception as e:
            print(f"[SKIP] {folder}: failed to read {xlsx_path}: {e}")
            continue

        df = normalize_columns(df_raw)

        if "idx" not in df.columns:
            df = df.copy()
            df["idx"] = np.arange(len(df))

        correct_col = find_col(df, CORRECT_CANDIDATES)
        if correct_col is not None:
            correct_num = df[correct_col].apply(parse_correct)
        else:
            correct_num = pd.Series(np.nan, index=df.index)

        total_examples = int(correct_num.notna().sum()) if correct_col is not None else int(len(df))
        accuracy = float(correct_num.dropna().mean()) if correct_num.notna().any() else np.nan

        run_summary_rows.append({
            "folder": folder.name,
            "run_path": str(folder),
            "examples_path": str(xlsx_path),
            **meta,
            "n_examples": int(len(df)),
            "n_with_correct_label": total_examples,
            "accuracy": accuracy,
        })

        dec_col = find_col(df, DECISIVENESS_CANDIDATES)
        if dec_col is None:
            print(f"[SKIP] {folder}: missing decisiveness column from {DECISIVENESS_CANDIDATES}")
            continue

        decisiveness_series_raw = pd.to_numeric(df[dec_col], errors="coerce")

        for method_key, method_meta in METHODS.items():
            conf_col = find_col(df, method_meta["confidence_candidates"])
            faith_col = find_col(df, method_meta["faithfulness_candidates"])

            if conf_col is None:
                print(f"[WARN] {folder}: missing confidence column for {method_meta['label']}")
                continue

            if faith_col is None:
                print(f"[WARN] {folder}: missing faithfulness column for {method_meta['label']}")
                continue

            confidence_raw = pd.to_numeric(df[conf_col], errors="coerce")
            faithfulness_raw = pd.to_numeric(df[faith_col], errors="coerce")

            for row_idx, row in df.iterrows():
                conf = confidence_raw.loc[row_idx]
                dec = decisiveness_series_raw.loc[row_idx]
                faith = faithfulness_raw.loc[row_idx]

                if pd.isna(conf) or pd.isna(dec) or pd.isna(faith):
                    continue

                conf_clipped = clip01(conf)
                dec_clipped = clip01(dec)
                faith_clipped = clip01(faith)

                if conf_clipped is None or dec_clipped is None or faith_clipped is None:
                    continue

                signed_gap = dec_clipped - conf_clipped
                abs_gap = abs(signed_gap)

                long_rows.append({
                    "folder": folder.name,
                    "run_path": str(folder),
                    "examples_path": str(xlsx_path),
                    **meta,
                    "idx": row.get("idx"),
                    "method_key": method_key,
                    "method": method_meta["label"],
                    "confidence_col": conf_col,
                    "faithfulness_col": faith_col,
                    "decisiveness_col": dec_col,
                    "confidence_raw": float(conf),
                    "decisiveness_raw": float(dec),
                    "faithfulness_raw": float(faith),
                    "confidence": conf_clipped,
                    "decisiveness": dec_clipped,
                    "faithfulness": faith_clipped,
                    "signed_gap": signed_gap,
                    "abs_gap": abs_gap,
                    "direction": direction_label(signed_gap, direction_tolerance),
                    "correct_num": correct_num.loc[row_idx] if row_idx in correct_num.index else np.nan,
                    "question": row.get("question", ""),
                    "gold": row.get("gold", ""),
                    "final_answer_extracted": row.get("final_answer_extracted", ""),
                })

    long_df = pd.DataFrame(long_rows)
    run_summary_df = pd.DataFrame(run_summary_rows)

    return long_df, run_summary_df


def assign_gap_percentile_bins(
    long_df: pd.DataFrame,
    aligned_quantile: float,
    strong_quantile: float,
    cmfg_bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if long_df.empty:
        long_df = long_df.copy()
        long_df["abs_gap_percentile_rank"] = np.nan
        long_df["gap_bin"] = pd.Series(dtype=str)
        return long_df, pd.DataFrame()

    long_df = long_df.copy()
    long_df["abs_gap_percentile_rank"] = np.nan
    long_df["gap_bin"] = pd.Series(index=long_df.index, dtype="object")

    metadata_rows: List[Dict[str, Any]] = []

    for method_key in METHOD_ORDER:
        method_label = METHODS[method_key]["label"]

        method_mask = (
            (long_df["method_key"] == method_key)
            & long_df["abs_gap"].notna()
        )

        method_df = long_df[method_mask].copy()

        if method_df.empty:
            metadata_rows.append({
                "method_key": method_key,
                "method": method_label,
                "n_valid": 0,
                "aligned_quantile": aligned_quantile,
                "strong_quantile": strong_quantile,
                "q_aligned_abs_gap": np.nan,
                "q_strong_abs_gap": np.nan,
                "aligned_count": 0,
                "moderate_mismatch_count": 0,
                "strong_mismatch_count": 0,
                "aligned_cmfg_star": np.nan,
                "moderate_mismatch_cmfg_star": np.nan,
                "strong_mismatch_cmfg_star": np.nan,
            })
            continue

        method_df = method_df.sort_values(
            ["abs_gap", "folder", "idx"],
            ascending=[True, True, True],
        )

        n = len(method_df)

        if n == 1:
            percentile_rank = np.array([0.5], dtype=float)
        else:
            percentile_rank = np.arange(n, dtype=float) / float(n - 1)

        bins = np.where(
            percentile_rank <= aligned_quantile,
            "Aligned",
            np.where(
                percentile_rank >= strong_quantile,
                "Strong Mismatch",
                "Moderate Mismatch",
            ),
        )

        long_df.loc[method_df.index, "abs_gap_percentile_rank"] = percentile_rank
        long_df.loc[method_df.index, "gap_bin"] = bins

        values = method_df["abs_gap"].to_numpy(dtype=float)

        meta = {
            "method_key": method_key,
            "method": method_label,
            "n_valid": n,
            "aligned_quantile": aligned_quantile,
            "strong_quantile": strong_quantile,
            "q_aligned_abs_gap": float(np.nanquantile(values, aligned_quantile)),
            "q_strong_abs_gap": float(np.nanquantile(values, strong_quantile)),
        }

        temp = method_df.copy()
        temp["gap_bin"] = bins

        for bin_name, prefix in [
            ("Aligned", "aligned"),
            ("Moderate Mismatch", "moderate_mismatch"),
            ("Strong Mismatch", "strong_mismatch"),
        ]:
            sub = temp[temp["gap_bin"] == bin_name]
            meta[f"{prefix}_count"] = int(len(sub))
            meta[f"{prefix}_abs_gap_min"] = float(sub["abs_gap"].min()) if len(sub) else np.nan
            meta[f"{prefix}_abs_gap_max"] = float(sub["abs_gap"].max()) if len(sub) else np.nan
            meta[f"{prefix}_cmfg_star"] = (
                compute_cmfg_star(sub["confidence"], sub["faithfulness"], n_bins=cmfg_bins)
                if len(sub)
                else np.nan
            )

        metadata_rows.append(meta)

    binning_metadata = pd.DataFrame(metadata_rows)

    long_df["gap_bin"] = pd.Categorical(
        long_df["gap_bin"],
        categories=GAP_BIN_ORDER,
        ordered=True,
    )

    long_df["direction"] = pd.Categorical(
        long_df["direction"],
        categories=DIRECTION_ORDER,
        ordered=True,
    )

    return long_df, binning_metadata


# ============================================================
# Complete-grid helpers
# ============================================================

def method_info_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "method_key": method_key,
            "method": METHODS[method_key]["label"],
        }
        for method_key in METHOD_ORDER
    ])


def gap_bins_df() -> pd.DataFrame:
    return pd.DataFrame({"gap_bin": GAP_BIN_ORDER})


def direction_df() -> pd.DataFrame:
    return pd.DataFrame({"direction": DIRECTION_ORDER})


def build_complete_grid(
    group_cols: Sequence[str],
    run_summary_df: pd.DataFrame,
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = list(group_cols)

    no_bin_cols = [
        c for c in group_cols
        if c not in {"gap_bin", "direction"}
    ]

    has_method_cols = "method_key" in no_bin_cols and "method" in no_bin_cols
    non_method_cols = [c for c in no_bin_cols if c not in {"method_key", "method"}]

    if non_method_cols:
        if all(c in run_summary_df.columns for c in non_method_cols):
            base = run_summary_df[non_method_cols].drop_duplicates().copy()
        else:
            base = long_df[non_method_cols].drop_duplicates().copy()
    else:
        base = pd.DataFrame({"__dummy__": [0]})

    if has_method_cols:
        base = base.merge(method_info_df(), how="cross")

    if "gap_bin" in group_cols:
        base = base.merge(gap_bins_df(), how="cross")

    if "direction" in group_cols:
        base = base.merge(direction_df(), how="cross")

    if "__dummy__" in base.columns:
        base = base.drop(columns=["__dummy__"])

    return base[group_cols].copy()


def safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")

    out = numer / denom
    out = out.replace([np.inf, -np.inf], np.nan)

    return out


# ============================================================
# Summary tables
# ============================================================

def build_summary_tables(
    long_df: pd.DataFrame,
    run_summary_df: pd.DataFrame,
    cmfg_bins: int,
) -> Dict[str, pd.DataFrame]:
    if run_summary_df.empty:
        return {
            "summary": pd.DataFrame(),
            "dataset_prompt_summary": pd.DataFrame(),
            "dataset_prompt_method_summary": pd.DataFrame(),
            "dataset_summary": pd.DataFrame(),
            "prompt_summary": pd.DataFrame(),
            "model_summary": pd.DataFrame(),
            "method_summary": pd.DataFrame(),
            "direction_summary": pd.DataFrame(),
            "strong_direction_summary": pd.DataFrame(),
        }

    if long_df.empty:
        long_df = pd.DataFrame(columns=[
            "dataset_key", "dataset",
            "model_key", "model",
            "prompt_key", "prompt",
            "method_key", "method",
            "gap_bin",
            "direction",
            "confidence",
            "decisiveness",
            "faithfulness",
            "signed_gap",
            "abs_gap",
            "correct_num",
        ])

    summary = aggregate_gap_summary(
        long_df,
        run_summary_df,
        group_cols=[
            "dataset_key",
            "dataset",
            "model_key",
            "model",
            "prompt_key",
            "prompt",
            "method_key",
            "method",
            "gap_bin",
        ],
        denom_cols=[
            "dataset_key",
            "dataset",
            "model_key",
            "model",
            "prompt_key",
            "prompt",
        ],
        cmfg_bins=cmfg_bins,
    )

    dataset_prompt_summary = aggregate_gap_summary(
        long_df,
        run_summary_df,
        group_cols=[
            "dataset_key",
            "dataset",
            "prompt_key",
            "prompt",
            "method_key",
            "method",
            "gap_bin",
        ],
        denom_cols=["dataset_key", "dataset", "prompt_key", "prompt"],
        cmfg_bins=cmfg_bins,
    )

    dataset_prompt_method_summary = aggregate_gap_summary(
        long_df,
        run_summary_df,
        group_cols=[
            "dataset_key",
            "dataset",
            "prompt_key",
            "prompt",
            "method_key",
            "method",
        ],
        denom_cols=["dataset_key", "dataset", "prompt_key", "prompt"],
        cmfg_bins=cmfg_bins,
    )

    dataset_summary = aggregate_gap_summary(
        long_df,
        run_summary_df,
        group_cols=[
            "dataset_key",
            "dataset",
            "method_key",
            "method",
            "gap_bin",
        ],
        denom_cols=["dataset_key", "dataset"],
        cmfg_bins=cmfg_bins,
    )

    prompt_summary = aggregate_gap_summary(
        long_df,
        run_summary_df,
        group_cols=[
            "prompt_key",
            "prompt",
            "method_key",
            "method",
            "gap_bin",
        ],
        denom_cols=["prompt_key", "prompt"],
        cmfg_bins=cmfg_bins,
    )

    model_summary = aggregate_gap_summary(
        long_df,
        run_summary_df,
        group_cols=[
            "model_key",
            "model",
            "method_key",
            "method",
            "gap_bin",
        ],
        denom_cols=["model_key", "model"],
        cmfg_bins=cmfg_bins,
    )

    method_summary = aggregate_gap_summary(
        long_df,
        run_summary_df,
        group_cols=[
            "method_key",
            "method",
            "gap_bin",
        ],
        denom_cols=[],
        cmfg_bins=cmfg_bins,
    )

    direction_summary = aggregate_direction_summary(
        long_df,
        run_summary_df,
        group_cols=[
            "method_key",
            "method",
            "direction",
        ],
        strong_only=False,
        cmfg_bins=cmfg_bins,
    )

    strong_direction_summary = aggregate_direction_summary(
        long_df,
        run_summary_df,
        group_cols=[
            "dataset_key",
            "dataset",
            "prompt_key",
            "prompt",
            "method_key",
            "method",
            "direction",
        ],
        strong_only=True,
        cmfg_bins=cmfg_bins,
    )

    return {
        "summary": summary,
        "dataset_prompt_summary": dataset_prompt_summary,
        "dataset_prompt_method_summary": dataset_prompt_method_summary,
        "dataset_summary": dataset_summary,
        "prompt_summary": prompt_summary,
        "model_summary": model_summary,
        "method_summary": method_summary,
        "direction_summary": direction_summary,
        "strong_direction_summary": strong_direction_summary,
    }


def _cmfg_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    cmfg_bins: int,
) -> pd.DataFrame:
    group_cols = list(group_cols)

    if df.empty:
        return pd.DataFrame(columns=[*group_cols, "cmfg_star"])

    rows: List[Dict[str, Any]] = []

    grouped = df.groupby(group_cols, dropna=False, observed=True)

    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["cmfg_star"] = compute_cmfg_star(
            g["confidence"],
            g["faithfulness"],
            n_bins=cmfg_bins,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_gap_summary(
    long_df: pd.DataFrame,
    run_summary_df: pd.DataFrame,
    group_cols: Sequence[str],
    denom_cols: Sequence[str],
    cmfg_bins: int,
) -> pd.DataFrame:
    group_cols = list(group_cols)
    denom_cols = list(denom_cols)

    if not long_df.empty:
        observed = (
            long_df
            .groupby(group_cols, dropna=False, observed=True)
            .agg(
                n_samples_bin=("abs_gap", "size"),
                mean_confidence=("confidence", "mean"),
                mean_decisiveness=("decisiveness", "mean"),
                mean_signed_gap=("signed_gap", "mean"),
                mean_abs_gap=("abs_gap", "mean"),
                median_abs_gap=("abs_gap", "median"),
                accuracy=("correct_num", "mean"),
            )
            .reset_index()
        )

        cmfg_df = _cmfg_by_group(
            long_df,
            group_cols=group_cols,
            cmfg_bins=cmfg_bins,
        )

        observed = observed.merge(
            cmfg_df,
            on=group_cols,
            how="left",
        )
    else:
        observed = pd.DataFrame(columns=[
            *group_cols,
            "n_samples_bin",
            "cmfg_star",
            "mean_confidence",
            "mean_decisiveness",
            "mean_signed_gap",
            "mean_abs_gap",
            "median_abs_gap",
            "accuracy",
        ])

    full_grid = build_complete_grid(group_cols, run_summary_df, long_df)

    summary = full_grid.merge(
        observed,
        on=group_cols,
        how="left",
    )

    summary["n_samples_bin"] = summary["n_samples_bin"].fillna(0).astype(int)

    for col in [
        "cmfg_star",
        "mean_confidence",
        "mean_decisiveness",
        "mean_signed_gap",
        "mean_abs_gap",
        "median_abs_gap",
        "accuracy",
    ]:
        if col not in summary.columns:
            summary[col] = np.nan

    method_total_cols = [c for c in group_cols if c != "gap_bin"]

    if not long_df.empty:
        method_totals = (
            long_df
            .groupby(method_total_cols, dropna=False, observed=True)
            .size()
            .reset_index(name="n_samples_method_total")
        )
    else:
        method_totals = pd.DataFrame(columns=[*method_total_cols, "n_samples_method_total"])

    method_grid = full_grid[method_total_cols].drop_duplicates()

    method_totals = method_grid.merge(
        method_totals,
        on=method_total_cols,
        how="left",
    )

    method_totals["n_samples_method_total"] = (
        method_totals["n_samples_method_total"]
        .fillna(0)
        .astype(int)
    )

    summary = summary.merge(
        method_totals,
        on=method_total_cols,
        how="left",
    )

    summary["n_samples_method_total"] = summary["n_samples_method_total"].fillna(0).astype(int)

    summary["frac_within_method"] = safe_divide(
        summary["n_samples_bin"],
        summary["n_samples_method_total"],
    )

    summary.loc[
        (summary["n_samples_bin"] == 0) & (summary["n_samples_method_total"] > 0),
        "frac_within_method",
    ] = 0.0

    if denom_cols:
        denom_df = (
            run_summary_df
            .groupby(denom_cols, dropna=False)
            .agg(n_examples=("n_examples", "sum"))
            .reset_index()
        )

        summary = summary.merge(
            denom_df,
            on=denom_cols,
            how="left",
        )
    else:
        summary["n_examples"] = int(run_summary_df["n_examples"].sum())

    summary["rate_among_examples"] = safe_divide(
        summary["n_samples_bin"],
        summary["n_examples"],
    )

    summary.loc[summary["n_samples_bin"] == 0, "rate_among_examples"] = 0.0

    return summary


def aggregate_direction_summary(
    long_df: pd.DataFrame,
    run_summary_df: pd.DataFrame,
    group_cols: Sequence[str],
    strong_only: bool,
    cmfg_bins: int,
) -> pd.DataFrame:
    group_cols = list(group_cols)

    df = long_df.copy()

    if strong_only:
        df = df[df["gap_bin"] == "Strong Mismatch"].copy()

    if not df.empty:
        observed = (
            df
            .groupby(group_cols, dropna=False, observed=True)
            .agg(
                n_samples_direction=("signed_gap", "size"),
                mean_confidence=("confidence", "mean"),
                mean_decisiveness=("decisiveness", "mean"),
                mean_signed_gap=("signed_gap", "mean"),
                mean_abs_gap=("abs_gap", "mean"),
                accuracy=("correct_num", "mean"),
            )
            .reset_index()
        )

        cmfg_df = _cmfg_by_group(
            df,
            group_cols=group_cols,
            cmfg_bins=cmfg_bins,
        )

        observed = observed.merge(
            cmfg_df,
            on=group_cols,
            how="left",
        )
    else:
        observed = pd.DataFrame(columns=[
            *group_cols,
            "n_samples_direction",
            "cmfg_star",
            "mean_confidence",
            "mean_decisiveness",
            "mean_signed_gap",
            "mean_abs_gap",
            "accuracy",
        ])

    full_grid = build_complete_grid(group_cols, run_summary_df, long_df)

    summary = full_grid.merge(
        observed,
        on=group_cols,
        how="left",
    )

    summary["n_samples_direction"] = summary["n_samples_direction"].fillna(0).astype(int)

    for col in [
        "cmfg_star",
        "mean_confidence",
        "mean_decisiveness",
        "mean_signed_gap",
        "mean_abs_gap",
        "accuracy",
    ]:
        if col not in summary.columns:
            summary[col] = np.nan

    total_cols = [c for c in group_cols if c != "direction"]

    if not df.empty:
        totals = (
            df
            .groupby(total_cols, dropna=False, observed=True)
            .size()
            .reset_index(name="n_samples_direction_total")
        )
    else:
        totals = pd.DataFrame(columns=[*total_cols, "n_samples_direction_total"])

    total_grid = full_grid[total_cols].drop_duplicates()

    totals = total_grid.merge(
        totals,
        on=total_cols,
        how="left",
    )

    totals["n_samples_direction_total"] = (
        totals["n_samples_direction_total"]
        .fillna(0)
        .astype(int)
    )

    summary = summary.merge(
        totals,
        on=total_cols,
        how="left",
    )

    summary["frac_direction"] = safe_divide(
        summary["n_samples_direction"],
        summary["n_samples_direction_total"],
    )

    summary.loc[
        (summary["n_samples_direction"] == 0) & (summary["n_samples_direction_total"] > 0),
        "frac_direction",
    ] = 0.0

    return summary


# ============================================================
# Plot helpers
# ============================================================

def save_figure(fig: plt.Figure, out_path: Path, save_pdf: bool = False) -> None:
    fig.savefig(out_path.with_suffix(".png"), dpi=PLOT_DPI, bbox_inches="tight")

    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")

    plt.close(fig)


def order_existing(values: Sequence[str], preferred_order: Sequence[str]) -> List[str]:
    values_unique = list(dict.fromkeys([v for v in values if pd.notna(v)]))
    ordered = [v for v in preferred_order if v in values_unique]
    ordered += [v for v in values_unique if v not in ordered]
    return ordered


def make_stacked_bar(
    ax: plt.Axes,
    pivot_df: pd.DataFrame,
    columns_order: Sequence[str],
    colors: Dict[str, str],
    title: str,
    ylabel: str,
    ylim_unit: bool = True,
) -> None:
    bottom = np.zeros(len(pivot_df))
    x = np.arange(len(pivot_df.index))

    for col in columns_order:
        vals = (
            pivot_df[col].to_numpy(dtype=float)
            if col in pivot_df.columns
            else np.zeros(len(pivot_df))
        )

        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=col,
            color=colors[col],
            edgecolor="white",
            linewidth=0.5,
        )

        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if ylim_unit:
        ax.set_ylim(0, 1.0)

    ax.grid(True, axis="y", alpha=0.6)
    ax.grid(False, axis="x")


# ============================================================
# Plots
# ============================================================

def plot_gap_distribution(
    long_df: pd.DataFrame,
    binning_metadata: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    aligned_quantile: float,
    strong_quantile: float,
) -> None:
    if long_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9.2, 5.1))

    try:
        sns.kdeplot(
            data=long_df,
            x="abs_gap",
            hue="method",
            hue_order=METHOD_LABEL_ORDER,
            common_norm=False,
            fill=True,
            alpha=0.25,
            linewidth=2.0,
            palette={v["label"]: v["color"] for v in METHODS.values()},
            warn_singular=False,
            ax=ax,
        )
    except Exception:
        sns.histplot(
            data=long_df,
            x="abs_gap",
            hue="method",
            hue_order=METHOD_LABEL_ORDER,
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            palette={v["label"]: v["color"] for v in METHODS.values()},
            ax=ax,
        )

    for method_key in METHOD_ORDER:
        color = METHODS[method_key]["color"]
        row = binning_metadata[binning_metadata["method_key"] == method_key]

        if row.empty:
            continue

        q_aligned = row.iloc[0]["q_aligned_abs_gap"]
        q_strong = row.iloc[0]["q_strong_abs_gap"]

        if pd.notna(q_aligned):
            ax.axvline(q_aligned, color=color, linestyle="--", linewidth=1.0, alpha=0.65)

        if pd.notna(q_strong):
            ax.axvline(q_strong, color=color, linestyle=":", linewidth=1.4, alpha=0.85)

    ax.set_xlim(0, 1)
    ax.set_xlabel("|Decisiveness − Confidence|")
    ax.set_ylabel("Density")
    ax.set_title(
        "Distribution of Confidence–Decisiveness Absolute Gap\n"
        f"{percentile_binning_text(aligned_quantile, strong_quantile)}"
    )

    fig.tight_layout()

    save_figure(
        fig,
        plot_dir / "abs_gap_distribution_by_method",
        save_pdf=save_pdf,
    )


def plot_all_methods_cmfg_star_by_gap_bin(
    method_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    aligned_quantile: float,
    strong_quantile: float,
) -> None:
    if method_summary.empty:
        return

    df = method_summary.copy()
    df["gap_bin"] = pd.Categorical(df["gap_bin"], categories=GAP_BIN_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_LABEL_ORDER, ordered=True)
    df = df.sort_values(["gap_bin", "method"])

    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    sns.barplot(
        data=df,
        x="gap_bin",
        y="cmfg_star",
        hue="method",
        order=GAP_BIN_ORDER,
        hue_order=METHOD_LABEL_ORDER,
        palette={v["label"]: v["color"] for v in METHODS.values()},
        ax=ax,
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence–Decisiveness Gap Bin")
    ax.set_ylabel(r"cMFG$^*$")
    ax.set_title(
        r"cMFG$^*$ by Confidence–Decisiveness Discordance"
        "\n"
        f"{percentile_binning_text(aligned_quantile, strong_quantile)}"
    )
    ax.legend(title="Method", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(True, axis="y", alpha=0.6)
    ax.grid(False, axis="x")

    fig.tight_layout()

    save_figure(
        fig,
        plot_dir / "all_methods_cmfg_star_by_gap_bin",
        save_pdf=save_pdf,
    )


def plot_all_methods_gap_values_by_gap_bin(
    method_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    aligned_quantile: float,
    strong_quantile: float,
) -> None:
    if method_summary.empty:
        return

    df = method_summary.copy()
    df["gap_bin"] = pd.Categorical(df["gap_bin"], categories=GAP_BIN_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_LABEL_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    sns.lineplot(
        data=df,
        x="gap_bin",
        y="mean_abs_gap",
        hue="method",
        style="method",
        markers=True,
        dashes=False,
        hue_order=METHOD_LABEL_ORDER,
        palette={v["label"]: v["color"] for v in METHODS.values()},
        linewidth=2.2,
        markersize=8,
        ax=ax,
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence–Decisiveness Gap Bin")
    ax.set_ylabel("Mean Absolute Gap")
    ax.set_title(
        "Mean |Decisiveness − Confidence| by Discordance Bin\n"
        f"{percentile_binning_text(aligned_quantile, strong_quantile)}"
    )
    ax.legend(title="Method", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(True, axis="y", alpha=0.6)
    ax.grid(False, axis="x")

    fig.tight_layout()

    save_figure(
        fig,
        plot_dir / "all_methods_mean_abs_gap_by_gap_bin",
        save_pdf=save_pdf,
    )


def plot_stacked_bins_by_dataset(
    dataset_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    aligned_quantile: float,
    strong_quantile: float,
) -> None:
    if dataset_summary.empty:
        return

    datasets = order_existing(dataset_summary["dataset"].tolist(), DATASET_ORDER)
    subtitle = percentile_binning_text(aligned_quantile, strong_quantile)

    for method_key, method_meta in METHODS.items():
        method_label = method_meta["label"]
        mdf = dataset_summary[dataset_summary["method_key"] == method_key].copy()

        if mdf.empty:
            continue

        pivot = (
            mdf
            .pivot_table(
                index="dataset",
                columns="gap_bin",
                values="frac_within_method",
                aggfunc="mean",
            )
            .reindex(index=datasets, columns=GAP_BIN_ORDER)
            .fillna(0.0)
        )

        fig, ax = plt.subplots(figsize=(7.4, 4.9))

        make_stacked_bar(
            ax,
            pivot,
            columns_order=GAP_BIN_ORDER,
            colors=GAP_BIN_COLORS,
            title=f"{method_label}: Discordance-Bin Composition Across Datasets\n{subtitle}",
            ylabel="Fraction of Samples",
            ylim_unit=True,
        )

        ax.legend(title="Gap Bin", loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()

        save_figure(
            fig,
            plot_dir / f"{method_key}_gap_bin_composition_by_dataset",
            save_pdf=save_pdf,
        )


def plot_stacked_bins_by_prompt(
    prompt_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    aligned_quantile: float,
    strong_quantile: float,
) -> None:
    if prompt_summary.empty:
        return

    prompts = order_existing(prompt_summary["prompt"].tolist(), PROMPT_ORDER)
    subtitle = percentile_binning_text(aligned_quantile, strong_quantile)

    for method_key, method_meta in METHODS.items():
        method_label = method_meta["label"]
        mdf = prompt_summary[prompt_summary["method_key"] == method_key].copy()

        if mdf.empty:
            continue

        pivot = (
            mdf
            .pivot_table(
                index="prompt",
                columns="gap_bin",
                values="frac_within_method",
                aggfunc="mean",
            )
            .reindex(index=prompts, columns=GAP_BIN_ORDER)
            .fillna(0.0)
        )

        fig_height = max(5.0, 0.38 * len(pivot) + 2.0)
        fig, ax = plt.subplots(figsize=(9.5, fig_height))

        make_stacked_bar(
            ax,
            pivot,
            columns_order=GAP_BIN_ORDER,
            colors=GAP_BIN_COLORS,
            title=f"{method_label}: Discordance-Bin Composition Across Prompts\n{subtitle}",
            ylabel="Fraction of Samples",
            ylim_unit=True,
        )

        ax.legend(title="Gap Bin", loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()

        save_figure(
            fig,
            plot_dir / f"{method_key}_gap_bin_composition_by_prompt",
            save_pdf=save_pdf,
        )


def plot_stacked_bins_by_model(
    model_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    aligned_quantile: float,
    strong_quantile: float,
) -> None:
    if model_summary.empty:
        return

    models = order_existing(model_summary["model"].tolist(), MODEL_ORDER)
    subtitle = percentile_binning_text(aligned_quantile, strong_quantile)

    for method_key, method_meta in METHODS.items():
        method_label = method_meta["label"]
        mdf = model_summary[model_summary["method_key"] == method_key].copy()

        if mdf.empty:
            continue

        pivot = (
            mdf
            .pivot_table(
                index="model",
                columns="gap_bin",
                values="frac_within_method",
                aggfunc="mean",
            )
            .reindex(index=models, columns=GAP_BIN_ORDER)
            .fillna(0.0)
        )

        fig, ax = plt.subplots(figsize=(7.4, 4.9))

        make_stacked_bar(
            ax,
            pivot,
            columns_order=GAP_BIN_ORDER,
            colors=GAP_BIN_COLORS,
            title=f"{method_label}: Discordance-Bin Composition Across Models\n{subtitle}",
            ylabel="Fraction of Samples",
            ylim_unit=True,
        )

        ax.legend(title="Gap Bin", loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()

        save_figure(
            fig,
            plot_dir / f"{method_key}_gap_bin_composition_by_model",
            save_pdf=save_pdf,
        )


def plot_dataset_prompt_cmfg_star_heatmaps(
    dataset_prompt_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    aligned_quantile: float,
    strong_quantile: float,
) -> None:
    if dataset_prompt_summary.empty:
        return

    datasets = order_existing(dataset_prompt_summary["dataset"].tolist(), DATASET_ORDER)
    prompts = order_existing(dataset_prompt_summary["prompt"].tolist(), PROMPT_ORDER)

    for method_key, method_meta in METHODS.items():
        method_label = method_meta["label"]

        for bin_name in GAP_BIN_ORDER:
            sub = dataset_prompt_summary[
                (dataset_prompt_summary["method_key"] == method_key)
                & (dataset_prompt_summary["gap_bin"] == bin_name)
            ].copy()

            if sub.empty:
                continue

            pivot = (
                sub
                .pivot_table(
                    index="dataset",
                    columns="prompt",
                    values="cmfg_star",
                    aggfunc="mean",
                )
                .reindex(index=datasets, columns=prompts)
            )

            fig_width = max(9.8, 0.75 * len(prompts) + 3.0)
            fig, ax = plt.subplots(figsize=(fig_width, 4.5))

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
                cbar_kws={"label": r"cMFG$^*$"},
            )

            ax.set_title(
                rf"{method_label}: cMFG$^*$ for {gap_bin_title(bin_name, aligned_quantile, strong_quantile)}"
            )
            ax.set_xlabel("Prompt")
            ax.set_ylabel("Dataset")
            ax.tick_params(axis="x", rotation=35)

            fig.tight_layout()

            save_figure(
                fig,
                plot_dir / f"{method_key}_{safe_filename(bin_name.lower())}_cmfg_star_heatmap_dataset_prompt",
                save_pdf=save_pdf,
            )


def plot_dataset_prompt_strong_fraction_heatmaps(
    dataset_prompt_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    aligned_quantile: float,
    strong_quantile: float,
) -> None:
    if dataset_prompt_summary.empty:
        return

    datasets = order_existing(dataset_prompt_summary["dataset"].tolist(), DATASET_ORDER)
    prompts = order_existing(dataset_prompt_summary["prompt"].tolist(), PROMPT_ORDER)
    strong_pct = int(round(100 * (1.0 - strong_quantile)))

    for method_key, method_meta in METHODS.items():
        method_label = method_meta["label"]

        sub = dataset_prompt_summary[
            (dataset_prompt_summary["method_key"] == method_key)
            & (dataset_prompt_summary["gap_bin"] == "Strong Mismatch")
        ].copy()

        if sub.empty:
            continue

        pivot = (
            sub
            .pivot_table(
                index="dataset",
                columns="prompt",
                values="frac_within_method",
                aggfunc="mean",
            )
            .reindex(index=datasets, columns=prompts)
            .fillna(0.0)
        )

        fig_width = max(9.8, 0.75 * len(prompts) + 3.0)
        fig, ax = plt.subplots(figsize=(fig_width, 4.5))

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="Reds",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Fraction of Samples"},
        )

        ax.set_title(
            f"{method_label}: Fraction of Strong Confidence–Decisiveness Mismatches\n"
            f"Top {strong_pct}% Absolute Gap Per Method"
        )
        ax.set_xlabel("Prompt")
        ax.set_ylabel("Dataset")
        ax.tick_params(axis="x", rotation=35)

        fig.tight_layout()

        save_figure(
            fig,
            plot_dir / f"{method_key}_strong_mismatch_fraction_heatmap_dataset_prompt",
            save_pdf=save_pdf,
        )


def plot_gap_direction_overall(
    direction_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
) -> None:
    if direction_summary.empty:
        return

    df = direction_summary.copy()
    df["method"] = pd.Categorical(df["method"], categories=METHOD_LABEL_ORDER, ordered=True)
    df["direction"] = pd.Categorical(df["direction"], categories=DIRECTION_ORDER, ordered=True)

    pivot = (
        df
        .pivot_table(
            index="method",
            columns="direction",
            values="frac_direction",
            aggfunc="mean",
        )
        .reindex(index=METHOD_LABEL_ORDER, columns=DIRECTION_ORDER)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    make_stacked_bar(
        ax,
        pivot,
        columns_order=DIRECTION_ORDER,
        colors=DIRECTION_COLORS,
        title="Direction of Confidence–Decisiveness Gap Across All Samples",
        ylabel="Fraction of Samples",
        ylim_unit=True,
    )

    ax.legend(title="Direction", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()

    save_figure(
        fig,
        plot_dir / "all_methods_gap_direction_fraction_overall",
        save_pdf=save_pdf,
    )


def plot_strong_mismatch_direction_by_dataset_prompt(
    strong_direction_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
) -> None:
    if strong_direction_summary.empty:
        return

    datasets = order_existing(strong_direction_summary["dataset"].tolist(), DATASET_ORDER)
    prompts = order_existing(strong_direction_summary["prompt"].tolist(), PROMPT_ORDER)

    for method_key, method_meta in METHODS.items():
        method_label = method_meta["label"]

        for direction in [
            "Decisiveness > Confidence",
            "Confidence > Decisiveness",
        ]:
            sub = strong_direction_summary[
                (strong_direction_summary["method_key"] == method_key)
                & (strong_direction_summary["direction"] == direction)
            ].copy()

            if sub.empty:
                continue

            pivot = (
                sub
                .pivot_table(
                    index="dataset",
                    columns="prompt",
                    values="frac_direction",
                    aggfunc="mean",
                )
                .reindex(index=datasets, columns=prompts)
                .fillna(0.0)
            )

            fig_width = max(9.8, 0.75 * len(prompts) + 3.0)
            fig, ax = plt.subplots(figsize=(fig_width, 4.5))

            cmap = "Reds" if direction == "Decisiveness > Confidence" else "Blues"

            sns.heatmap(
                pivot,
                ax=ax,
                cmap=cmap,
                vmin=0,
                vmax=1,
                linewidths=0.5,
                linecolor="white",
                annot=True,
                fmt=".2f",
                cbar_kws={"label": "Fraction Among Strong Mismatches"},
            )

            ax.set_title(
                f"{method_label}: {direction} Among Strong Mismatches"
            )
            ax.set_xlabel("Prompt")
            ax.set_ylabel("Dataset")
            ax.tick_params(axis="x", rotation=35)

            fig.tight_layout()

            save_figure(
                fig,
                plot_dir / f"{method_key}_{safe_filename(direction.lower())}_among_strong_mismatch_heatmap",
                save_pdf=save_pdf,
            )


def plot_strong_fraction_vs_cmfg_star(
    dataset_prompt_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
) -> None:
    if dataset_prompt_summary.empty:
        return

    strong = dataset_prompt_summary[
        dataset_prompt_summary["gap_bin"] == "Strong Mismatch"
    ].copy()

    if strong.empty:
        return

    strong = strong.dropna(subset=["frac_within_method", "cmfg_star"])

    if strong.empty:
        return

    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    sns.scatterplot(
        data=strong,
        x="frac_within_method",
        y="cmfg_star",
        hue="method",
        style="dataset",
        hue_order=METHOD_LABEL_ORDER,
        palette={v["label"]: v["color"] for v in METHODS.values()},
        s=95,
        alpha=0.85,
        ax=ax,
    )

    if len(strong) >= 3:
        try:
            sns.regplot(
                data=strong,
                x="frac_within_method",
                y="cmfg_star",
                scatter=False,
                color="black",
                line_kws={"linewidth": 1.3, "alpha": 0.75},
                ax=ax,
            )
        except Exception:
            pass

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fraction of Strong Confidence–Decisiveness Mismatches")
    ax.set_ylabel(r"cMFG$^*$")
    ax.set_title(r"Do Strong Confidence–Decisiveness Mismatches Correspond to Lower cMFG$^*$?")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)

    fig.tight_layout()

    save_figure(
        fig,
        plot_dir / "strong_mismatch_fraction_vs_cmfg_star",
        save_pdf=save_pdf,
    )


def make_all_plots(
    long_df: pd.DataFrame,
    binning_metadata: pd.DataFrame,
    summaries: Dict[str, pd.DataFrame],
    plot_dir: Path,
    save_pdf: bool,
    aligned_quantile: float,
    strong_quantile: float,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_gap_distribution(
        long_df,
        binning_metadata,
        plot_dir,
        save_pdf,
        aligned_quantile=aligned_quantile,
        strong_quantile=strong_quantile,
    )

    plot_all_methods_cmfg_star_by_gap_bin(
        summaries["method_summary"],
        plot_dir,
        save_pdf,
        aligned_quantile=aligned_quantile,
        strong_quantile=strong_quantile,
    )

    plot_all_methods_gap_values_by_gap_bin(
        summaries["method_summary"],
        plot_dir,
        save_pdf,
        aligned_quantile=aligned_quantile,
        strong_quantile=strong_quantile,
    )

    plot_stacked_bins_by_dataset(
        summaries["dataset_summary"],
        plot_dir,
        save_pdf,
        aligned_quantile=aligned_quantile,
        strong_quantile=strong_quantile,
    )

    plot_stacked_bins_by_prompt(
        summaries["prompt_summary"],
        plot_dir,
        save_pdf,
        aligned_quantile=aligned_quantile,
        strong_quantile=strong_quantile,
    )

    plot_stacked_bins_by_model(
        summaries["model_summary"],
        plot_dir,
        save_pdf,
        aligned_quantile=aligned_quantile,
        strong_quantile=strong_quantile,
    )

    plot_dataset_prompt_cmfg_star_heatmaps(
        summaries["dataset_prompt_summary"],
        plot_dir,
        save_pdf,
        aligned_quantile=aligned_quantile,
        strong_quantile=strong_quantile,
    )

    plot_dataset_prompt_strong_fraction_heatmaps(
        summaries["dataset_prompt_summary"],
        plot_dir,
        save_pdf,
        aligned_quantile=aligned_quantile,
        strong_quantile=strong_quantile,
    )

    plot_gap_direction_overall(
        summaries["direction_summary"],
        plot_dir,
        save_pdf,
    )

    plot_strong_mismatch_direction_by_dataset_prompt(
        summaries["strong_direction_summary"],
        plot_dir,
        save_pdf,
    )

    plot_strong_fraction_vs_cmfg_star(
        summaries["dataset_prompt_summary"],
        plot_dir,
        save_pdf,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    run_folders = list_run_folders(
        repo_root=args.repo_root,
        real_results_dir=args.real_results_dir,
        run_folder=args.run_folder,
        dataset_filter=args.dataset,
        model_filter=args.model,
        prompt_filter=args.prompt,
    )

    if not run_folders:
        raise RuntimeError("No matching run folders found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(run_folders)} matching run folders.")
    print("Collecting confidence–decisiveness gap rows...")

    long_df, run_summary_df = collect_gap_rows(
        run_folders=run_folders,
        direction_tolerance=args.direction_tolerance,
    )

    print("Assigning percentile gap bins per method...")

    long_df, binning_metadata = assign_gap_percentile_bins(
        long_df=long_df,
        aligned_quantile=args.aligned_quantile,
        strong_quantile=args.strong_quantile,
        cmfg_bins=args.cmfg_bins,
    )

    long_rows_path = args.output_dir / "gap_long_rows.csv"
    run_summary_path = args.output_dir / "run_level_summary.csv"
    binning_path = args.output_dir / "gap_binning_metadata.csv"

    long_df.to_csv(long_rows_path, index=False)
    run_summary_df.to_csv(run_summary_path, index=False)
    binning_metadata.to_csv(binning_path, index=False)

    print(f"Saved long gap rows to {long_rows_path}")
    print(f"Saved run-level summary to {run_summary_path}")
    print(f"Saved gap binning metadata to {binning_path}")

    summaries = build_summary_tables(
        long_df=long_df,
        run_summary_df=run_summary_df,
        cmfg_bins=args.cmfg_bins,
    )

    output_name_map = {
        "summary": "gap_summary.csv",
        "dataset_prompt_summary": "gap_dataset_prompt_summary.csv",
        "dataset_prompt_method_summary": "gap_dataset_prompt_method_summary.csv",
        "dataset_summary": "gap_dataset_summary.csv",
        "prompt_summary": "gap_prompt_summary.csv",
        "model_summary": "gap_model_summary.csv",
        "method_summary": "gap_method_summary.csv",
        "direction_summary": "gap_direction_summary.csv",
        "strong_direction_summary": "strong_mismatch_direction_summary.csv",
    }

    for key, filename in output_name_map.items():
        path = args.output_dir / filename
        summaries[key].to_csv(path, index=False)
        print(f"Saved {key} to {path}")

    print("Generating plots...")

    make_all_plots(
        long_df=long_df,
        binning_metadata=binning_metadata,
        summaries=summaries,
        plot_dir=plot_dir,
        save_pdf=args.save_pdf,
        aligned_quantile=args.aligned_quantile,
        strong_quantile=args.strong_quantile,
    )

    print("")
    print("Done.")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Plots directory:  {plot_dir.resolve()}")


if __name__ == "__main__":
    main()