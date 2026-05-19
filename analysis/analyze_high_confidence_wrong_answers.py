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
# Repository format
# ============================================================

REAL_RESULTS_DIR = _DEFAULT_REAL_RESULTS_DIR


DATASET_LABELS = _common.DATASET_LABELS

MODEL_FULL_LABELS = _common.MODEL_FULL_LABELS

MODEL_SHORT_LABELS = _common.MODEL_SHORT_LABELS

PROMPT_LABELS = _common.PROMPT_LABELS

# Longest suffixes first.
PROMPT_SUFFIXES = _common.PROMPT_SUFFIXES

DATASET_ORDER = _common.DATASET_ORDER_WITH_LEGACY

PROMPT_ORDER = _common.PROMPT_ORDER

MODEL_ORDER = _common.MODEL_ORDER_SHORT


# ============================================================
# Methods and columns
# ============================================================

METHODS = _common.METHODS_CANDIDATES

METHOD_ORDER = _common.METHOD_ORDER
METHOD_LABEL_ORDER = [METHODS[k]["label"] for k in METHOD_ORDER]

CONF_BIN_ORDER = ["Low", "High", "Very High"]

CONF_BIN_COLORS = {
    "Low": "#A6CEE3",
    "High": "#FDBF6F",
    "Very High": "#E31A1C",
}

CORRECT_CANDIDATES = _common.CORRECT_CANDIDATES


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
        default=_default_output_dir("wrong_confidence_percentile_bin_analysis"),
    )

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)

    parser.add_argument(
        "--low-quantile",
        type=float,
        default=0.25,
        help="Bottom quantile used for Low confidence.",
    )
    parser.add_argument(
        "--very-high-quantile",
        type=float,
        default=0.75,
        help="Upper quantile used for Very High confidence.",
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

    # Colab/Jupyter safe.
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"[INFO] Ignored unknown Colab/Jupyter arguments: {unknown}")

    if not (0 < args.low_quantile < 1):
        raise ValueError("--low-quantile must be in (0, 1)")

    if not (0 < args.very_high_quantile < 1):
        raise ValueError("--very-high-quantile must be in (0, 1)")

    if args.low_quantile >= args.very_high_quantile:
        raise ValueError("--low-quantile must be smaller than --very-high-quantile")

    if args.cmfg_bins < 1:
        raise ValueError("--cmfg-bins must be at least 1")

    return args


# ============================================================
# Parsing helpers
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

    for candidate in candidates:
        c_norm = normalize_col(candidate)
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

    key = clean_text(dataset_key).lower().replace("-", "_").replace(" ", "_")
    return DATASET_LABELS.get(key, dataset_key)


def normalize_model_key(model_key: Optional[str]) -> Optional[str]:
    if model_key is None:
        return None

    key = clean_text(model_key).lower().replace("-", "_").replace(" ", "_")
    return MODEL_SHORT_LABELS.get(key, model_key)


def normalize_model_full_key(model_key: Optional[str]) -> Optional[str]:
    if model_key is None:
        return None

    key = clean_text(model_key).lower().replace("-", "_").replace(" ", "_")
    return MODEL_FULL_LABELS.get(key, model_key)


def normalize_prompt_key(prompt_key: Optional[str]) -> Optional[str]:
    if prompt_key is None:
        return None

    key = clean_text(prompt_key).lower().replace("-", "_").replace(" ", "_")
    return PROMPT_LABELS.get(key, prompt_key)


def parse_run_name(run_name: str) -> Tuple[str, str]:
    """
    Return (dataset_key, prompt_key) from corrected nested run names:
      aime_b
      hle_perc
      sgpqa_msh_perc
      legal_self_monitoring
    """
    run_name = clean_text(run_name)

    for suffix in PROMPT_SUFFIXES:
        if run_name.endswith(suffix):
            dataset_key = run_name[: -len(suffix)]
            prompt_key = suffix[1:]
            return dataset_key, prompt_key

    return run_name, "unknown"


def parse_run_metadata(run_folder: Path) -> Dict[str, str]:
    """
    Parse metadata from corrected nested format:

        real_results/ds_8b/aime_b

    """
    folder_name = run_folder.name
    parent_name = run_folder.parent.name

    model_key = parent_name
    dataset_key, prompt_key = parse_run_name(folder_name)

    dataset = normalize_dataset_key(dataset_key) or dataset_key
    model = normalize_model_key(model_key) or model_key
    model_full = normalize_model_full_key(model_key) or model_key
    prompt = normalize_prompt_key(prompt_key) or prompt_key

    return {
        "dataset_key": dataset_key,
        "dataset": dataset,
        "model_key": model_key,
        "model": model,
        "model_full": model_full,
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
        d = dataset_filter.lower().replace("-", "_").replace(" ", "_")
        if d not in {
            meta["dataset_key"].lower(),
            meta["dataset"].lower().replace("-", "_").replace(" ", "_"),
        }:
            return False

    if model_filter is not None:
        m = model_filter.lower().replace("-", "_").replace(" ", "_")
        if m not in {
            meta["model_key"].lower(),
            meta["model"].lower().replace("-", "_").replace(" ", "_"),
            meta["model_full"].lower().replace("-", "_").replace(" ", "_"),
        }:
            return False

    if prompt_filter is not None:
        p = prompt_filter.lower().replace("-", "_").replace(" ", "_")
        if p not in {
            meta["prompt_key"].lower(),
            meta["prompt"].lower().replace("-", "_").replace(" ", "_"),
        }:
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

    # Correct nested format.
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


    # Deduplicate while preserving order.
    seen = set()
    out = []

    for p in folders:
        key = str(p.resolve())

        if key not in seen:
            out.append(p)
            seen.add(key)

    return out


def percentile_binning_text(low_quantile: float, very_high_quantile: float) -> str:
    low_pct = int(round(100 * low_quantile))
    high_pct = int(round(100 * very_high_quantile))
    top_pct = int(round(100 * (1.0 - very_high_quantile)))

    return (
        "Percentile Bins Per Method: "
        f"Low ≤ P{low_pct}, High P{low_pct}–P{high_pct}, "
        f"Very High ≥ P{high_pct} / Top {top_pct}%"
    )


def bin_title_text(bin_name: str, low_quantile: float, very_high_quantile: float) -> str:
    low_pct = int(round(100 * low_quantile))
    high_pct = int(round(100 * very_high_quantile))
    top_pct = int(round(100 * (1.0 - very_high_quantile)))

    if bin_name == "Low":
        return f"Low-Confidence Wrong Answers, Bottom {low_pct}% Per Method"

    if bin_name == "High":
        return f"High-Confidence Wrong Answers, P{low_pct}–P{high_pct} Per Method"

    return f"Very-High-Confidence Wrong Answers, Top {top_pct}% Per Method"


# ============================================================
# cMFG* computation
# ============================================================

def compute_cmfg_star(
    confidence: Sequence[Any],
    faithfulness: Sequence[Any],
    n_bins: int = 10,
) -> float:
    """
    Compute cMFG* from example-level confidence and example-level faithfulness.

    This is used for wrong-answer subsets here, but it is still computed from
    example-level rows, not step-level rows.
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

def collect_wrong_confidence_rows(
    run_folders: Sequence[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        long_df:
            One row per wrong answer per method.
            The percentile bin is assigned later globally per method.

        run_summary_df:
            One row per run folder.
    """
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

        correct_col = find_col(df, CORRECT_CANDIDATES)

        if correct_col is None:
            print(f"[SKIP] {folder}: missing correct column from {CORRECT_CANDIDATES}")
            continue

        df = df.copy()

        if "idx" not in df.columns:
            df["idx"] = np.arange(len(df))

        df["correct_num"] = df[correct_col].apply(parse_correct)

        total_examples = int(df["correct_num"].notna().sum())
        wrong_mask = df["correct_num"] == 0
        wrong_examples = int(wrong_mask.sum())

        accuracy = float(df["correct_num"].dropna().mean()) if total_examples > 0 else np.nan
        wrong_rate = float(wrong_examples / total_examples) if total_examples > 0 else np.nan

        run_summary_rows.append({
            "folder": folder.name,
            "run_path": str(folder),
            "examples_path": str(xlsx_path),
            **meta,
            "n_total_examples": total_examples,
            "n_wrong_examples": wrong_examples,
            "accuracy": accuracy,
            "wrong_rate": wrong_rate,
        })

        wrong_df = df[wrong_mask].copy()

        if wrong_df.empty:
            continue

        for method_key, method_meta in METHODS.items():
            conf_col = find_col(wrong_df, method_meta["confidence_candidates"])
            faith_col = find_col(wrong_df, method_meta["faithfulness_candidates"])

            if conf_col is None:
                print(f"[WARN] {folder}: missing confidence column for {method_meta['label']}")
                continue

            if faith_col is None:
                print(f"[WARN] {folder}: missing faithfulness column for {method_meta['label']}")
                continue

            confidence_series = pd.to_numeric(wrong_df[conf_col], errors="coerce")
            faith_series = pd.to_numeric(wrong_df[faith_col], errors="coerce")

            for row_idx, row in wrong_df.iterrows():
                conf = confidence_series.loc[row_idx]
                faith_value = faith_series.loc[row_idx]

                if pd.isna(conf) or pd.isna(faith_value):
                    continue

                conf = float(conf)
                conf_clipped = float(np.clip(conf, 0.0, 1.0))

                faith_value = float(faith_value)
                faith_clipped = float(np.clip(faith_value, 0.0, 1.0))

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
                    "confidence": conf,
                    "confidence_clipped": conf_clipped,
                    "faithfulness": faith_clipped,
                    "correct_num": 0.0,
                    "question": row.get("question", ""),
                    "gold": row.get("gold", ""),
                    "final_answer_extracted": row.get("final_answer_extracted", ""),
                })

    long_df = pd.DataFrame(long_rows)
    run_summary_df = pd.DataFrame(run_summary_rows)

    return long_df, run_summary_df


def assign_percentile_bins(
    long_df: pd.DataFrame,
    low_quantile: float,
    very_high_quantile: float,
    cmfg_bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assigns Low / High / Very High by percentile rank within each method.

    This avoids fixed absolute thresholds such as confidence >= 0.8.
    """
    if long_df.empty:
        long_df = long_df.copy()
        long_df["confidence_percentile_rank"] = np.nan
        long_df["confidence_bin"] = pd.Series(dtype=str)
        return long_df, pd.DataFrame()

    long_df = long_df.copy()
    long_df["confidence_percentile_rank"] = np.nan
    long_df["confidence_bin"] = pd.Series(index=long_df.index, dtype="object")

    metadata_rows: List[Dict[str, Any]] = []

    for method_key in METHOD_ORDER:
        method_label = METHODS[method_key]["label"]

        method_mask = (
            (long_df["method_key"] == method_key)
            & long_df["confidence_clipped"].notna()
        )

        method_df = long_df[method_mask].copy()

        if method_df.empty:
            metadata_rows.append({
                "method_key": method_key,
                "method": method_label,
                "n_wrong_with_confidence": 0,
                "low_quantile": low_quantile,
                "very_high_quantile": very_high_quantile,
                "q_low_value": np.nan,
                "q_very_high_value": np.nan,
                "low_count": 0,
                "high_count": 0,
                "very_high_count": 0,
                "low_conf_min": np.nan,
                "low_conf_max": np.nan,
                "high_conf_min": np.nan,
                "high_conf_max": np.nan,
                "very_high_conf_min": np.nan,
                "very_high_conf_max": np.nan,
                "low_cmfg_star": np.nan,
                "high_cmfg_star": np.nan,
                "very_high_cmfg_star": np.nan,
            })
            continue

        method_df = method_df.sort_values(
            ["confidence_clipped", "folder", "idx"],
            ascending=[True, True, True],
        )

        n = len(method_df)

        if n == 1:
            percentile_rank = np.array([0.5], dtype=float)
        else:
            percentile_rank = np.arange(n, dtype=float) / float(n - 1)

        bins = np.where(
            percentile_rank <= low_quantile,
            "Low",
            np.where(
                percentile_rank >= very_high_quantile,
                "Very High",
                "High",
            ),
        )

        long_df.loc[method_df.index, "confidence_percentile_rank"] = percentile_rank
        long_df.loc[method_df.index, "confidence_bin"] = bins

        values = method_df["confidence_clipped"].to_numpy(dtype=float)

        q_low_value = float(np.nanquantile(values, low_quantile))
        q_very_high_value = float(np.nanquantile(values, very_high_quantile))

        meta: Dict[str, Any] = {
            "method_key": method_key,
            "method": method_label,
            "n_wrong_with_confidence": n,
            "low_quantile": low_quantile,
            "very_high_quantile": very_high_quantile,
            "q_low_value": q_low_value,
            "q_very_high_value": q_very_high_value,
        }

        temp = method_df.copy()
        temp["confidence_bin"] = bins

        for bin_name, prefix in [
            ("Low", "low"),
            ("High", "high"),
            ("Very High", "very_high"),
        ]:
            sub = temp[temp["confidence_bin"] == bin_name]

            meta[f"{prefix}_count"] = int(len(sub))
            meta[f"{prefix}_conf_min"] = (
                float(sub["confidence_clipped"].min()) if len(sub) else np.nan
            )
            meta[f"{prefix}_conf_max"] = (
                float(sub["confidence_clipped"].max()) if len(sub) else np.nan
            )
            meta[f"{prefix}_cmfg_star"] = (
                compute_cmfg_star(sub["confidence_clipped"], sub["faithfulness"], n_bins=cmfg_bins)
                if len(sub)
                else np.nan
            )

        metadata_rows.append(meta)

    binning_metadata = pd.DataFrame(metadata_rows)

    long_df["confidence_bin"] = pd.Categorical(
        long_df["confidence_bin"],
        categories=CONF_BIN_ORDER,
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


def confidence_bins_df() -> pd.DataFrame:
    return pd.DataFrame({"confidence_bin": CONF_BIN_ORDER})


def build_complete_grid(
    group_cols: Sequence[str],
    run_summary_df: pd.DataFrame,
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = list(group_cols)
    no_bin_cols = [c for c in group_cols if c != "confidence_bin"]

    has_method_cols = "method_key" in no_bin_cols and "method" in no_bin_cols
    non_method_cols = [c for c in no_bin_cols if c not in {"method_key", "method"}]

    if non_method_cols:
        source = (
            run_summary_df
            if all(c in run_summary_df.columns for c in non_method_cols)
            else long_df
        )
        base = source[non_method_cols].drop_duplicates().copy()
    else:
        base = pd.DataFrame({"__dummy__": [0]})

    if has_method_cols:
        base = base.merge(method_info_df(), how="cross")

    if "confidence_bin" in group_cols:
        base = base.merge(confidence_bins_df(), how="cross")

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

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["cmfg_star"] = compute_cmfg_star(
            group["confidence_clipped"],
            group["faithfulness"],
            n_bins=cmfg_bins,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_summary_tables(
    long_df: pd.DataFrame,
    run_summary_df: pd.DataFrame,
    cmfg_bins: int,
) -> Dict[str, pd.DataFrame]:
    if run_summary_df.empty:
        return {
            "summary": pd.DataFrame(),
            "dataset_prompt_summary": pd.DataFrame(),
            "dataset_summary": pd.DataFrame(),
            "prompt_summary": pd.DataFrame(),
            "model_summary": pd.DataFrame(),
            "method_summary": pd.DataFrame(),
        }

    if long_df.empty:
        long_df = pd.DataFrame(columns=[
            "dataset_key", "dataset",
            "model_key", "model",
            "prompt_key", "prompt",
            "method_key", "method",
            "confidence_bin",
            "confidence",
            "confidence_clipped",
            "faithfulness",
        ])

    summary = aggregate_summary(
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
            "confidence_bin",
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

    dataset_prompt_summary = aggregate_summary(
        long_df,
        run_summary_df,
        group_cols=[
            "dataset_key",
            "dataset",
            "prompt_key",
            "prompt",
            "method_key",
            "method",
            "confidence_bin",
        ],
        denom_cols=["dataset_key", "dataset", "prompt_key", "prompt"],
        cmfg_bins=cmfg_bins,
    )

    dataset_summary = aggregate_summary(
        long_df,
        run_summary_df,
        group_cols=[
            "dataset_key",
            "dataset",
            "method_key",
            "method",
            "confidence_bin",
        ],
        denom_cols=["dataset_key", "dataset"],
        cmfg_bins=cmfg_bins,
    )

    prompt_summary = aggregate_summary(
        long_df,
        run_summary_df,
        group_cols=[
            "prompt_key",
            "prompt",
            "method_key",
            "method",
            "confidence_bin",
        ],
        denom_cols=["prompt_key", "prompt"],
        cmfg_bins=cmfg_bins,
    )

    model_summary = aggregate_summary(
        long_df,
        run_summary_df,
        group_cols=[
            "model_key",
            "model",
            "method_key",
            "method",
            "confidence_bin",
        ],
        denom_cols=["model_key", "model"],
        cmfg_bins=cmfg_bins,
    )

    method_summary = aggregate_summary(
        long_df,
        run_summary_df,
        group_cols=[
            "method_key",
            "method",
            "confidence_bin",
        ],
        denom_cols=[],
        cmfg_bins=cmfg_bins,
    )

    return {
        "summary": summary,
        "dataset_prompt_summary": dataset_prompt_summary,
        "dataset_summary": dataset_summary,
        "prompt_summary": prompt_summary,
        "model_summary": model_summary,
        "method_summary": method_summary,
    }


def aggregate_summary(
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
                n_wrong_method_bin=("confidence_clipped", "size"),
                mean_wrong_confidence=("confidence_clipped", "mean"),
                median_wrong_confidence=("confidence_clipped", "median"),
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
            "n_wrong_method_bin",
            "mean_wrong_confidence",
            "median_wrong_confidence",
            "cmfg_star",
        ])

    full_grid = build_complete_grid(
        group_cols=group_cols,
        run_summary_df=run_summary_df,
        long_df=long_df,
    )

    summary = full_grid.merge(
        observed,
        on=group_cols,
        how="left",
    )

    summary["n_wrong_method_bin"] = summary["n_wrong_method_bin"].fillna(0).astype(int)

    for col in [
        "mean_wrong_confidence",
        "median_wrong_confidence",
        "cmfg_star",
    ]:
        if col not in summary.columns:
            summary[col] = np.nan

    method_total_cols = [c for c in group_cols if c != "confidence_bin"]

    if not long_df.empty:
        method_totals = (
            long_df
            .groupby(method_total_cols, dropna=False, observed=True)
            .size()
            .reset_index(name="n_wrong_method_total")
        )
    else:
        method_totals = pd.DataFrame(columns=[*method_total_cols, "n_wrong_method_total"])

    method_grid = full_grid[method_total_cols].drop_duplicates()

    method_totals = method_grid.merge(
        method_totals,
        on=method_total_cols,
        how="left",
    )

    method_totals["n_wrong_method_total"] = (
        method_totals["n_wrong_method_total"]
        .fillna(0)
        .astype(int)
    )

    summary = summary.merge(
        method_totals,
        on=method_total_cols,
        how="left",
    )

    summary["n_wrong_method_total"] = summary["n_wrong_method_total"].fillna(0).astype(int)

    summary["frac_wrong_within_method"] = safe_divide(
        summary["n_wrong_method_bin"],
        summary["n_wrong_method_total"],
    )

    summary.loc[
        (summary["n_wrong_method_bin"] == 0) & (summary["n_wrong_method_total"] > 0),
        "frac_wrong_within_method",
    ] = 0.0

    if denom_cols:
        denom_df = (
            run_summary_df
            .groupby(denom_cols, dropna=False)
            .agg(
                n_total_examples=("n_total_examples", "sum"),
                n_wrong_examples=("n_wrong_examples", "sum"),
            )
            .reset_index()
        )

        summary = summary.merge(
            denom_df,
            on=denom_cols,
            how="left",
        )
    else:
        summary["n_total_examples"] = int(run_summary_df["n_total_examples"].sum())
        summary["n_wrong_examples"] = int(run_summary_df["n_wrong_examples"].sum())

    summary["rate_bin_among_all_examples"] = safe_divide(
        summary["n_wrong_method_bin"],
        summary["n_total_examples"],
    )

    summary["rate_bin_among_wrong_examples"] = safe_divide(
        summary["n_wrong_method_bin"],
        summary["n_wrong_examples"],
    )

    summary.loc[summary["n_wrong_method_bin"] == 0, "rate_bin_among_all_examples"] = 0.0
    summary.loc[summary["n_wrong_method_bin"] == 0, "rate_bin_among_wrong_examples"] = 0.0

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
    title: str,
    ylabel: str,
    y_is_fraction: bool = True,
) -> None:
    bottom = np.zeros(len(pivot_df))
    x = np.arange(len(pivot_df.index))

    for bin_name in CONF_BIN_ORDER:
        vals = (
            pivot_df[bin_name].to_numpy(dtype=float)
            if bin_name in pivot_df.columns
            else np.zeros(len(pivot_df))
        )

        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=bin_name,
            color=CONF_BIN_COLORS[bin_name],
            edgecolor="white",
            linewidth=0.5,
        )

        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if y_is_fraction:
        ax.set_ylim(0, 1.0)

    ax.grid(True, axis="y", alpha=0.6)
    ax.grid(False, axis="x")


# ============================================================
# Plots
# ============================================================

def plot_confidence_distribution_wrong(
    long_df: pd.DataFrame,
    binning_metadata: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    low_quantile: float,
    very_high_quantile: float,
) -> None:
    if long_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    try:
        sns.kdeplot(
            data=long_df,
            x="confidence_clipped",
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
            x="confidence_clipped",
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

        q_low = row.iloc[0]["q_low_value"]
        q_high = row.iloc[0]["q_very_high_value"]

        if pd.notna(q_low):
            ax.axvline(q_low, color=color, linestyle="--", linewidth=1.0, alpha=0.65)

        if pd.notna(q_high):
            ax.axvline(q_high, color=color, linestyle=":", linewidth=1.2, alpha=0.85)

    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence on Wrong Answers")
    ax.set_ylabel("Density")
    ax.set_title(
        "Confidence Distribution on Wrong Answers\n"
        f"{percentile_binning_text(low_quantile, very_high_quantile)}"
    )

    fig.tight_layout()

    save_figure(
        fig,
        plot_dir / "wrong_answer_confidence_distribution_by_method",
        save_pdf=save_pdf,
    )


def plot_dataset_stacked_by_method(
    dataset_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    low_quantile: float,
    very_high_quantile: float,
) -> None:
    if dataset_summary.empty:
        return

    datasets = order_existing(dataset_summary["dataset"].tolist(), DATASET_ORDER)
    subtitle = percentile_binning_text(low_quantile, very_high_quantile)

    for method_key, method_meta in METHODS.items():
        method_label = method_meta["label"]
        mdf = dataset_summary[dataset_summary["method_key"] == method_key].copy()

        if mdf.empty:
            continue

        pivot = (
            mdf
            .pivot_table(
                index="dataset",
                columns="confidence_bin",
                values="frac_wrong_within_method",
                aggfunc="mean",
            )
            .reindex(index=datasets, columns=CONF_BIN_ORDER)
            .fillna(0.0)
        )

        fig, ax = plt.subplots(figsize=(7.4, 4.9))

        make_stacked_bar(
            ax,
            pivot,
            title=f"{method_label}: Wrong Answers by Relative Confidence Bin Across Datasets\n{subtitle}",
            ylabel="Fraction of Wrong Answers",
            y_is_fraction=True,
        )

        ax.legend(title="Confidence Bin", loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()

        save_figure(
            fig,
            plot_dir / f"{method_key}_wrong_confidence_percentile_bins_by_dataset",
            save_pdf=save_pdf,
        )


def plot_prompt_stacked_by_method(
    prompt_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    low_quantile: float,
    very_high_quantile: float,
) -> None:
    if prompt_summary.empty:
        return

    prompts = order_existing(prompt_summary["prompt"].tolist(), PROMPT_ORDER)
    subtitle = percentile_binning_text(low_quantile, very_high_quantile)

    for method_key, method_meta in METHODS.items():
        method_label = method_meta["label"]
        mdf = prompt_summary[prompt_summary["method_key"] == method_key].copy()

        if mdf.empty:
            continue

        pivot = (
            mdf
            .pivot_table(
                index="prompt",
                columns="confidence_bin",
                values="frac_wrong_within_method",
                aggfunc="mean",
            )
            .reindex(index=prompts, columns=CONF_BIN_ORDER)
            .fillna(0.0)
        )

        fig_height = max(5.0, 0.38 * len(pivot) + 2.0)
        fig, ax = plt.subplots(figsize=(9.4, fig_height))

        make_stacked_bar(
            ax,
            pivot,
            title=f"{method_label}: Wrong Answers by Relative Confidence Bin Across Prompts\n{subtitle}",
            ylabel="Fraction of Wrong Answers",
            y_is_fraction=True,
        )

        ax.legend(title="Confidence Bin", loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()

        save_figure(
            fig,
            plot_dir / f"{method_key}_wrong_confidence_percentile_bins_by_prompt",
            save_pdf=save_pdf,
        )


def plot_model_stacked_by_method(
    model_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    low_quantile: float,
    very_high_quantile: float,
) -> None:
    if model_summary.empty:
        return

    models = order_existing(model_summary["model"].tolist(), MODEL_ORDER)
    subtitle = percentile_binning_text(low_quantile, very_high_quantile)

    for method_key, method_meta in METHODS.items():
        method_label = method_meta["label"]
        mdf = model_summary[model_summary["method_key"] == method_key].copy()

        if mdf.empty:
            continue

        pivot = (
            mdf
            .pivot_table(
                index="model",
                columns="confidence_bin",
                values="frac_wrong_within_method",
                aggfunc="mean",
            )
            .reindex(index=models, columns=CONF_BIN_ORDER)
            .fillna(0.0)
        )

        fig, ax = plt.subplots(figsize=(7.4, 4.9))

        make_stacked_bar(
            ax,
            pivot,
            title=f"{method_label}: Wrong Answers by Relative Confidence Bin Across Models\n{subtitle}",
            ylabel="Fraction of Wrong Answers",
            y_is_fraction=True,
        )

        ax.legend(title="Confidence Bin", loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()

        save_figure(
            fig,
            plot_dir / f"{method_key}_wrong_confidence_percentile_bins_by_model",
            save_pdf=save_pdf,
        )


def plot_method_comparison_very_high(
    method_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    prompt_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    low_quantile: float,
    very_high_quantile: float,
) -> None:
    if method_summary.empty:
        return

    top_pct = int(round(100 * (1.0 - very_high_quantile)))
    title_label = f"Very-High Relative-Confidence Wrong Answers, Top {top_pct}% Per Method"

    # Overall method-level comparison.
    vh = method_summary[method_summary["confidence_bin"] == "Very High"].copy()

    if not vh.empty:
        all_methods = pd.DataFrame({
            "method_key": METHOD_ORDER,
            "method": METHOD_LABEL_ORDER,
        })

        vh = all_methods.merge(
            vh,
            on=["method_key", "method"],
            how="left",
        )

        vh["frac_wrong_within_method"] = vh["frac_wrong_within_method"].fillna(0.0)
        vh["n_wrong_method_bin"] = vh["n_wrong_method_bin"].fillna(0).astype(int)

        color_map = {METHODS[k]["label"]: METHODS[k]["color"] for k in METHOD_ORDER}
        colors = [color_map[m] for m in vh["method"]]

        fig, ax = plt.subplots(figsize=(7.2, 4.8))

        bars = ax.bar(
            vh["method"].astype(str),
            vh["frac_wrong_within_method"],
            color=colors,
            edgecolor="#222222",
            linewidth=0.4,
        )

        for bar, value, count in zip(
            bars,
            vh["frac_wrong_within_method"],
            vh["n_wrong_method_bin"],
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.2f}\n(n={count})",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Fraction of Wrong Answers")
        ax.set_title(title_label)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.6)
        ax.grid(False, axis="x")

        fig.tight_layout()

        save_figure(
            fig,
            plot_dir / "all_methods_very_high_relative_wrong_fraction_overall",
            save_pdf=save_pdf,
        )

    # Dataset comparison.
    vh_dataset = dataset_summary[dataset_summary["confidence_bin"] == "Very High"].copy()

    if not vh_dataset.empty:
        datasets = order_existing(vh_dataset["dataset"].tolist(), DATASET_ORDER)

        fig, ax = plt.subplots(figsize=(8.4, 4.9))

        sns.barplot(
            data=vh_dataset,
            x="dataset",
            y="frac_wrong_within_method",
            hue="method",
            order=datasets,
            hue_order=METHOD_LABEL_ORDER,
            palette={v["label"]: v["color"] for v in METHODS.values()},
            ax=ax,
        )

        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Fraction of Wrong Answers")
        ax.set_title(f"{title_label} Across Datasets")
        ax.legend(title="Method", loc="upper left", bbox_to_anchor=(1.02, 1.0))
        ax.grid(True, axis="y", alpha=0.6)
        ax.grid(False, axis="x")

        fig.tight_layout()

        save_figure(
            fig,
            plot_dir / "all_methods_very_high_relative_wrong_fraction_by_dataset",
            save_pdf=save_pdf,
        )

    # Prompt comparison.
    vh_prompt = prompt_summary[prompt_summary["confidence_bin"] == "Very High"].copy()

    if not vh_prompt.empty:
        prompts = order_existing(vh_prompt["prompt"].tolist(), PROMPT_ORDER)

        fig_height = max(5.2, 0.35 * len(prompts) + 2.2)
        fig, ax = plt.subplots(figsize=(10.8, fig_height))

        sns.barplot(
            data=vh_prompt,
            x="prompt",
            y="frac_wrong_within_method",
            hue="method",
            order=prompts,
            hue_order=METHOD_LABEL_ORDER,
            palette={v["label"]: v["color"] for v in METHODS.values()},
            ax=ax,
        )

        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Prompt")
        ax.set_ylabel("Fraction of Wrong Answers")
        ax.set_title(f"{title_label} Across Prompts")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(title="Method", loc="upper left", bbox_to_anchor=(1.02, 1.0))
        ax.grid(True, axis="y", alpha=0.6)
        ax.grid(False, axis="x")

        fig.tight_layout()

        save_figure(
            fig,
            plot_dir / "all_methods_very_high_relative_wrong_fraction_by_prompt",
            save_pdf=save_pdf,
        )


def plot_cmfg_star_by_confidence_bin(
    method_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    low_quantile: float,
    very_high_quantile: float,
) -> None:
    if method_summary.empty:
        return

    df = method_summary.copy()
    df["confidence_bin"] = pd.Categorical(df["confidence_bin"], categories=CONF_BIN_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_LABEL_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(8.4, 4.9))

    sns.barplot(
        data=df,
        x="confidence_bin",
        y="cmfg_star",
        hue="method",
        order=CONF_BIN_ORDER,
        hue_order=METHOD_LABEL_ORDER,
        palette={v["label"]: v["color"] for v in METHODS.values()},
        ax=ax,
    )

    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Relative Confidence Bin on Wrong Answers")
    ax.set_ylabel(r"cMFG$^*$")
    ax.set_title(
        r"cMFG$^*$ by Relative Confidence Bin on Wrong Answers"
        "\n"
        f"{percentile_binning_text(low_quantile, very_high_quantile)}"
    )
    ax.legend(title="Method", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(True, axis="y", alpha=0.6)
    ax.grid(False, axis="x")

    fig.tight_layout()

    save_figure(
        fig,
        plot_dir / "all_methods_cmfg_star_by_wrong_confidence_bin",
        save_pdf=save_pdf,
    )


def plot_dataset_prompt_heatmaps(
    dataset_prompt_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    low_quantile: float,
    very_high_quantile: float,
) -> None:
    if dataset_prompt_summary.empty:
        return

    datasets = order_existing(dataset_prompt_summary["dataset"].tolist(), DATASET_ORDER)
    prompts = order_existing(dataset_prompt_summary["prompt"].tolist(), PROMPT_ORDER)

    for method_key, method_meta in METHODS.items():
        method_label = method_meta["label"]

        for bin_name in CONF_BIN_ORDER:
            sub = dataset_prompt_summary[
                (dataset_prompt_summary["method_key"] == method_key)
                & (dataset_prompt_summary["confidence_bin"] == bin_name)
            ].copy()

            if sub.empty:
                continue

            pivot = (
                sub
                .pivot_table(
                    index="dataset",
                    columns="prompt",
                    values="frac_wrong_within_method",
                    aggfunc="mean",
                )
                .reindex(index=datasets, columns=prompts)
                .fillna(0.0)
            )

            fig_width = max(9.8, 0.75 * len(prompts) + 3.0)
            fig, ax = plt.subplots(figsize=(fig_width, 4.4))

            cmap = "Reds" if bin_name == "Very High" else "YlOrBr"

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
                cbar_kws={"label": "Fraction of Wrong Answers"},
            )

            ax.set_title(
                f"{method_label}: {bin_title_text(bin_name, low_quantile, very_high_quantile)}"
            )
            ax.set_xlabel("Prompt")
            ax.set_ylabel("Dataset")
            ax.tick_params(axis="x", rotation=35)

            fig.tight_layout()

            save_figure(
                fig,
                plot_dir / f"{method_key}_{safe_filename(bin_name.lower())}_relative_wrong_fraction_heatmap_dataset_prompt",
                save_pdf=save_pdf,
            )


def plot_dataset_prompt_cmfg_star_heatmaps(
    dataset_prompt_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    low_quantile: float,
    very_high_quantile: float,
) -> None:
    if dataset_prompt_summary.empty:
        return

    datasets = order_existing(dataset_prompt_summary["dataset"].tolist(), DATASET_ORDER)
    prompts = order_existing(dataset_prompt_summary["prompt"].tolist(), PROMPT_ORDER)

    for method_key, method_meta in METHODS.items():
        method_label = method_meta["label"]

        for bin_name in CONF_BIN_ORDER:
            sub = dataset_prompt_summary[
                (dataset_prompt_summary["method_key"] == method_key)
                & (dataset_prompt_summary["confidence_bin"] == bin_name)
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
            fig, ax = plt.subplots(figsize=(fig_width, 4.4))

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
                rf"{method_label}: cMFG$^*$ for {bin_title_text(bin_name, low_quantile, very_high_quantile)}"
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


def plot_dataset_prompt_count_heatmaps(
    dataset_prompt_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
    low_quantile: float,
    very_high_quantile: float,
) -> None:
    if dataset_prompt_summary.empty:
        return

    datasets = order_existing(dataset_prompt_summary["dataset"].tolist(), DATASET_ORDER)
    prompts = order_existing(dataset_prompt_summary["prompt"].tolist(), PROMPT_ORDER)

    for method_key, method_meta in METHODS.items():
        method_label = method_meta["label"]

        for bin_name in CONF_BIN_ORDER:
            sub = dataset_prompt_summary[
                (dataset_prompt_summary["method_key"] == method_key)
                & (dataset_prompt_summary["confidence_bin"] == bin_name)
            ].copy()

            if sub.empty:
                continue

            pivot = (
                sub
                .pivot_table(
                    index="dataset",
                    columns="prompt",
                    values="n_wrong_method_bin",
                    aggfunc="sum",
                )
                .reindex(index=datasets, columns=prompts)
                .fillna(0)
            )

            fig_width = max(9.8, 0.75 * len(prompts) + 3.0)
            fig, ax = plt.subplots(figsize=(fig_width, 4.4))

            cmap = "Reds" if bin_name == "Very High" else "YlOrBr"

            sns.heatmap(
                pivot,
                ax=ax,
                cmap=cmap,
                linewidths=0.5,
                linecolor="white",
                annot=True,
                fmt=".0f",
                cbar_kws={"label": "Count"},
            )

            ax.set_title(
                f"{method_label}: Count of {bin_title_text(bin_name, low_quantile, very_high_quantile)}"
            )
            ax.set_xlabel("Prompt")
            ax.set_ylabel("Dataset")
            ax.tick_params(axis="x", rotation=35)

            fig.tight_layout()

            save_figure(
                fig,
                plot_dir / f"{method_key}_{safe_filename(bin_name.lower())}_relative_wrong_count_heatmap_dataset_prompt",
                save_pdf=save_pdf,
            )


def make_all_plots(
    long_df: pd.DataFrame,
    binning_metadata: pd.DataFrame,
    summaries: Dict[str, pd.DataFrame],
    plot_dir: Path,
    save_pdf: bool,
    low_quantile: float,
    very_high_quantile: float,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_confidence_distribution_wrong(
        long_df,
        binning_metadata,
        plot_dir,
        save_pdf,
        low_quantile=low_quantile,
        very_high_quantile=very_high_quantile,
    )

    plot_dataset_stacked_by_method(
        summaries["dataset_summary"],
        plot_dir,
        save_pdf,
        low_quantile=low_quantile,
        very_high_quantile=very_high_quantile,
    )

    plot_prompt_stacked_by_method(
        summaries["prompt_summary"],
        plot_dir,
        save_pdf,
        low_quantile=low_quantile,
        very_high_quantile=very_high_quantile,
    )

    plot_model_stacked_by_method(
        summaries["model_summary"],
        plot_dir,
        save_pdf,
        low_quantile=low_quantile,
        very_high_quantile=very_high_quantile,
    )

    plot_method_comparison_very_high(
        summaries["method_summary"],
        summaries["dataset_summary"],
        summaries["prompt_summary"],
        plot_dir,
        save_pdf,
        low_quantile=low_quantile,
        very_high_quantile=very_high_quantile,
    )

    plot_cmfg_star_by_confidence_bin(
        summaries["method_summary"],
        plot_dir,
        save_pdf,
        low_quantile=low_quantile,
        very_high_quantile=very_high_quantile,
    )

    plot_dataset_prompt_heatmaps(
        summaries["dataset_prompt_summary"],
        plot_dir,
        save_pdf,
        low_quantile=low_quantile,
        very_high_quantile=very_high_quantile,
    )

    plot_dataset_prompt_cmfg_star_heatmaps(
        summaries["dataset_prompt_summary"],
        plot_dir,
        save_pdf,
        low_quantile=low_quantile,
        very_high_quantile=very_high_quantile,
    )

    plot_dataset_prompt_count_heatmaps(
        summaries["dataset_prompt_summary"],
        plot_dir,
        save_pdf,
        low_quantile=low_quantile,
        very_high_quantile=very_high_quantile,
    )


# ============================================================
# README
# ============================================================

def write_readme(
    out_dir: Path,
    n_runs: int,
    long_df: pd.DataFrame,
    run_summary_df: pd.DataFrame,
    low_quantile: float,
    very_high_quantile: float,
    cmfg_bins: int,
) -> None:
    low_pct = int(round(100 * low_quantile))
    high_pct = int(round(100 * very_high_quantile))
    top_pct = int(round(100 * (1.0 - very_high_quantile)))

    lines: List[str] = []

    lines.append("WRONG-ANSWER RELATIVE CONFIDENCE-BIN ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Folder format:")
    lines.append("  - Preferred: real_results/{model_key}/{dataset_prompt}/results_*_examples.xlsx")
    lines.append("  - Backward-compatible: {dataset}_{model}_{prompt}/results_*_examples.xlsx")
    lines.append("")
    lines.append("This analysis splits wrong answers by percentile rank within each method:")
    lines.append(f"  - Low: bottom {low_pct}% confidence among wrong answers for that method")
    lines.append(f"  - High: P{low_pct} to P{high_pct}")
    lines.append(f"  - Very High: top {top_pct}% confidence among wrong answers for that method")
    lines.append("")
    lines.append("This avoids fixed absolute thresholds such as confidence >= 0.8.")
    lines.append("Therefore DeepConf will still appear even if its absolute confidence scale is lower.")
    lines.append("")
    lines.append("Reported faithfulness-style aggregate metric:")
    lines.append(f"  - cMFG* computed with {cmfg_bins} equal-mass bins")
    lines.append("  - cMFG* is computed from example-level confidence and example-level faithfulness.")
    lines.append("  - Step-level cMFG* is not used because cMFG* is not defined for steps.")
    lines.append("")
    lines.append(f"Number of run folders scanned: {n_runs}")
    lines.append(f"Number of run folders with readable examples: {len(run_summary_df)}")
    lines.append(f"Number of wrong-answer method rows: {len(long_df)}")
    lines.append("")

    if not run_summary_df.empty:
        lines.append("Overall Run-Level Totals:")
        lines.append(f"  - Total examples: {int(run_summary_df['n_total_examples'].sum())}")
        lines.append(f"  - Total wrong examples: {int(run_summary_df['n_wrong_examples'].sum())}")

        denom = run_summary_df["n_total_examples"].sum()
        numer = run_summary_df["n_wrong_examples"].sum()

        if denom > 0:
            lines.append(f"  - Overall wrong rate: {numer / denom:.4f}")

        lines.append("")

    lines.append("Main CSV Files:")
    lines.append("  - wrong_confidence_long.csv")
    lines.append("  - binning_metadata.csv")
    lines.append("  - run_level_wrong_confidence_summary.csv")
    lines.append("  - wrong_confidence_summary.csv")
    lines.append("  - wrong_confidence_dataset_prompt_summary.csv")
    lines.append("  - wrong_confidence_dataset_summary.csv")
    lines.append("  - wrong_confidence_prompt_summary.csv")
    lines.append("  - wrong_confidence_model_summary.csv")
    lines.append("  - wrong_confidence_method_summary.csv")
    lines.append("")
    lines.append("Main Plots:")
    lines.append("  - plots/wrong_answer_confidence_distribution_by_method.png")
    lines.append("  - plots/all_methods_very_high_relative_wrong_fraction_overall.png")
    lines.append("  - plots/all_methods_very_high_relative_wrong_fraction_by_dataset.png")
    lines.append("  - plots/all_methods_very_high_relative_wrong_fraction_by_prompt.png")
    lines.append("  - plots/all_methods_cmfg_star_by_wrong_confidence_bin.png")
    lines.append("  - plots/*_relative_wrong_fraction_heatmap_dataset_prompt.png")
    lines.append("  - plots/*_cmfg_star_heatmap_dataset_prompt.png")
    lines.append("  - plots/*_relative_wrong_count_heatmap_dataset_prompt.png")

    out_dir.joinpath("README_results.txt").write_text("\n".join(lines), encoding="utf-8")


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
    print("Collecting wrong-answer confidence rows...")

    long_df, run_summary_df = collect_wrong_confidence_rows(
        run_folders=run_folders,
    )

    print("Assigning percentile bins per method...")

    long_df, binning_metadata = assign_percentile_bins(
        long_df=long_df,
        low_quantile=args.low_quantile,
        very_high_quantile=args.very_high_quantile,
        cmfg_bins=args.cmfg_bins,
    )

    long_path = args.output_dir / "wrong_confidence_long.csv"
    run_summary_path = args.output_dir / "run_level_wrong_confidence_summary.csv"
    binning_path = args.output_dir / "binning_metadata.csv"

    long_df.to_csv(long_path, index=False)
    run_summary_df.to_csv(run_summary_path, index=False)
    binning_metadata.to_csv(binning_path, index=False)

    print(f"Saved long wrong-answer table to {long_path}")
    print(f"Saved run-level summary to {run_summary_path}")
    print(f"Saved binning metadata to {binning_path}")

    summaries = build_summary_tables(
        long_df=long_df,
        run_summary_df=run_summary_df,
        cmfg_bins=args.cmfg_bins,
    )

    output_name_map = {
        "summary": "wrong_confidence_summary.csv",
        "dataset_prompt_summary": "wrong_confidence_dataset_prompt_summary.csv",
        "dataset_summary": "wrong_confidence_dataset_summary.csv",
        "prompt_summary": "wrong_confidence_prompt_summary.csv",
        "model_summary": "wrong_confidence_model_summary.csv",
        "method_summary": "wrong_confidence_method_summary.csv",
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
        low_quantile=args.low_quantile,
        very_high_quantile=args.very_high_quantile,
    )

    write_readme(
        out_dir=args.output_dir,
        n_runs=len(run_folders),
        long_df=long_df,
        run_summary_df=run_summary_df,
        low_quantile=args.low_quantile,
        very_high_quantile=args.very_high_quantile,
        cmfg_bins=args.cmfg_bins,
    )

    print("")
    print("Done.")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Plots directory:  {plot_dir.resolve()}")


if __name__ == "__main__":
    main()