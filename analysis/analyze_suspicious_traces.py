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
"""
Typical usage:
  python analyze_trace_signals.py --debug
  python analyze_trace_signals.py --run-folder ds_8b/aime_b --debug
  python analyze_trace_signals.py --dataset aime --model ds_8b --prompt b --debug
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

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# ============================================================
# Directory format
# ============================================================

REAL_RESULTS_DIR = _DEFAULT_REAL_RESULTS_DIR


DATASET_LABELS = _common.DATASET_LABELS

MODEL_FULL_LABELS = _common.MODEL_FULL_LABELS

MODEL_SHORT_LABELS = _common.MODEL_SHORT_LABELS

PROMPT_LABELS = _common.PROMPT_LABELS

PROMPT_SUFFIXES = _common.PROMPT_SUFFIXES

DATASET_ORDER = _common.DATASET_ORDER_WITH_LEGACY
MODEL_ORDER = _common.MODEL_ORDER_SHORT

PROMPT_ORDER = _common.PROMPT_ORDER


# ============================================================
# Methods
# ============================================================

METHODS = _common.METHODS_TRACE_SIGNALS

METHOD_ORDER = _common.METHOD_ORDER
METHOD_LABEL_ORDER = [METHODS[k]["label"] for k in METHOD_ORDER]
METHOD_PALETTE = {METHODS[k]["label"]: METHODS[k]["color"] for k in METHOD_ORDER}

CORRECT_CANDIDATES = _common.CORRECT_CANDIDATES

DECISIVENESS_CANDIDATES = _common.DECISIVENESS_CANDIDATES


# ============================================================
# Signals
# ============================================================

SIGNAL_LABELS = {
    "signal_largest_conf_drop": "Largest Confidence Drop",
    "signal_min_step_conf": "Minimum Step Confidence",
    "signal_high_final_conf_low_faith": "High Final Confidence + Low Faithfulness",
    "signal_high_decisiveness_low_faith": "High Decisiveness + Low Faithfulness",
    "signal_confidence_exceeds_faith": "Confidence Exceeds Faithfulness",
    "signal_decisiveness_exceeds_faith": "Decisiveness Exceeds Faithfulness",
}

SIGNAL_COLUMNS = list(SIGNAL_LABELS.keys())

SIGNAL_CMAPS = {
    "signal_largest_conf_drop": "Reds",
    "signal_min_step_conf": "YlGnBu",
    "signal_high_final_conf_low_faith": "Reds",
    "signal_high_decisiveness_low_faith": "Reds",
    "signal_confidence_exceeds_faith": "Oranges",
    "signal_decisiveness_exceeds_faith": "Oranges",
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
    "legend.title_fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PLOT_DPI = 300


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repository root. If omitted, the script searches cwd and script parents.",
    )
    parser.add_argument(
        "--real-results-dir",
        type=Path,
        default=REAL_RESULTS_DIR,
        help="Path to real_results. Relative paths are searched under cwd, repo root, and script parents.",
    )
    parser.add_argument(
        "--run-folder",
        type=str,
        default=None,
        help="Optional run folder, e.g. ds_8b/aime_b, qwq_32b/sgpqa_msh_perc, or just aime_b.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir("trace_signal_analysis"),
    )

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)

    parser.add_argument(
        "--rcc-step-conf-col",
        type=str,
        default="rcc_p",
        choices=["rcc_p", "rcc_q"],
        help="Preferred RCC step-level confidence column.",
    )

    parser.add_argument(
        "--cmfg-bins",
        type=int,
        default=10,
        help="Number of equal-mass confidence bins used to compute cMFG*.",
    )

    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Disable PDF export. PNG files are always saved.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print folder-discovery diagnostics.",
    )

    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"[INFO] Ignored unknown Colab/Jupyter arguments: {unknown}")

    if args.cmfg_bins < 1:
        raise ValueError("--cmfg-bins must be at least 1")

    return args


# ============================================================
# Helpers
# ============================================================

TRUE_STRINGS = _common.TRUE_STRINGS
FALSE_STRINGS = _common.FALSE_STRINGS


def script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def clean_text(text: Any) -> str:
    return " ".join(str(text).replace("\xa0", " ").split()).strip()


def normalize_token(text: Any) -> str:
    return clean_text(text).lower().replace("-", "_").replace(" ", "_")


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


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = set(df.columns)

    for col in candidates:
        col_norm = normalize_col(col)
        if col_norm in cols:
            return col_norm

    lower_map = {str(c).lower(): c for c in df.columns}

    for col in candidates:
        col_lower = str(col).lower()
        if col_lower in lower_map:
            return lower_map[col_lower]

    return None


def parse_correct(x: Any) -> Optional[float]:
    if x is None:
        return None

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, bool):
        return float(x)

    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x >= 0.5)

    s = str(x).strip().lower()

    if s in TRUE_STRINGS:
        return 1.0

    if s in FALSE_STRINGS:
        return 0.0

    return None


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return np.nan

    if not np.isfinite(v):
        return np.nan

    return v


def clip01_scalar(x: Any) -> float:
    v = safe_float(x)

    if not np.isfinite(v):
        return np.nan

    return float(np.clip(v, 0.0, 1.0))


def clip01_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(lower=0.0, upper=1.0)


def safe_nanquantile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan

    return float(np.quantile(arr, q))


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")

    mask = x_num.notna() & y_num.notna()
    x_valid = x_num[mask]
    y_valid = y_num[mask]

    if len(x_valid) < 5:
        return np.nan

    if x_valid.nunique(dropna=True) < 2 or y_valid.nunique(dropna=True) < 2:
        return np.nan

    return float(x_valid.corr(y_valid, method="spearman"))


def safe_filename(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)


def order_existing(values: Sequence[str], preferred_order: Sequence[str]) -> List[str]:
    values_unique = list(dict.fromkeys([v for v in values if pd.notna(v)]))
    ordered = [v for v in preferred_order if v in values_unique]
    ordered += [v for v in values_unique if v not in ordered]
    return ordered


def normalize_dataset_key(dataset_key: Optional[str]) -> Optional[str]:
    if dataset_key is None:
        return None
    return DATASET_LABELS.get(normalize_token(dataset_key), dataset_key)


def normalize_model_key(model_key: Optional[str]) -> Optional[str]:
    if model_key is None:
        return None
    return MODEL_SHORT_LABELS.get(normalize_token(model_key), model_key)


def normalize_model_full_key(model_key: Optional[str]) -> Optional[str]:
    if model_key is None:
        return None
    return MODEL_FULL_LABELS.get(normalize_token(model_key), model_key)


def normalize_prompt_key(prompt_key: Optional[str]) -> Optional[str]:
    if prompt_key is None:
        return None
    return PROMPT_LABELS.get(normalize_token(prompt_key), prompt_key)


def parse_run_name(run_name: str) -> Tuple[str, str]:
    run_name = clean_text(run_name)

    for suffix in PROMPT_SUFFIXES:
        if run_name.endswith(suffix):
            dataset_key = run_name[: -len(suffix)]
            prompt_key = suffix[1:]
            return dataset_key, prompt_key

    return run_name, "unknown"


def parse_run_metadata(run_folder: Path) -> Dict[str, str]:
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
        d = normalize_token(dataset_filter)
        allowed = {
            normalize_token(meta["dataset_key"]),
            normalize_token(meta["dataset"]),
        }
        if d not in allowed:
            return False

    if model_filter is not None:
        m = normalize_token(model_filter)
        allowed = {
            normalize_token(meta["model_key"]),
            normalize_token(meta["model"]),
            normalize_token(meta["model_full"]),
        }
        if m not in allowed:
            return False

    if prompt_filter is not None:
        p = normalize_token(prompt_filter)
        allowed = {
            normalize_token(meta["prompt_key"]),
            normalize_token(meta["prompt"]),
        }
        if p not in allowed:
            return False

    return True


def candidate_repo_roots(repo_root_arg: Optional[Path]) -> List[Path]:
    roots: List[Path] = []

    if repo_root_arg is not None:
        roots.append(repo_root_arg.expanduser().resolve())

    roots.append(Path.cwd().resolve())
    roots.append(script_dir())
    roots.extend(script_dir().parents)

    out: List[Path] = []
    seen = set()

    for root in roots:
        key = str(root)

        if key not in seen:
            out.append(root)
            seen.add(key)

    return out


def resolve_real_results_root(
    repo_root_arg: Optional[Path],
    real_results_dir: Path,
    debug: bool = False,
) -> Path:
    tried: List[Path] = []

    if real_results_dir.is_absolute():
        tried.append(real_results_dir)
    else:
        for root in candidate_repo_roots(repo_root_arg):
            tried.append(root / real_results_dir)

    for path in tried:
        if path.is_dir():
            if debug:
                print(f"[DEBUG] Using real_results root: {path.resolve()}")
            return path.resolve()

    raise FileNotFoundError(
        "Could not find real_results directory. Tried:\n"
        + "\n".join(f"  - {p}" for p in tried)
    )


def find_examples_xlsx(folder: Path) -> Optional[Path]:
    candidates = sorted(folder.glob("results_*_examples.xlsx"))

    if candidates:
        return candidates[0]

    candidates = sorted(
        p for p in folder.glob("*.xlsx")
        if "examples" in p.name.lower()
    )

    return candidates[0] if candidates else None


def find_step_level_xlsx(folder: Path) -> Optional[Path]:
    candidates = sorted(folder.glob("results_*_step_level.xlsx"))

    if candidates:
        return candidates[0]

    candidates = sorted(folder.glob("*.xlsx"))

    preferred = [
        p for p in candidates
        if "step" in p.name.lower() and "level" in p.name.lower()
    ]

    if preferred:
        return preferred[0]

    fallback = [
        p for p in candidates
        if "step" in p.name.lower() and "examples" not in p.name.lower()
    ]

    if fallback:
        return fallback[0]

    return None


def list_run_folders(
    repo_root: Optional[Path],
    real_results_dir: Path,
    run_folder: Optional[str],
    dataset_filter: Optional[str],
    model_filter: Optional[str],
    prompt_filter: Optional[str],
    debug: bool = False,
) -> List[Path]:
    real_root = resolve_real_results_root(
        repo_root_arg=repo_root,
        real_results_dir=real_results_dir,
        debug=debug,
    )

    def is_valid_run_dir(p: Path) -> bool:
        if not p.is_dir():
            return False

        if find_examples_xlsx(p) is None:
            return False

        meta = parse_run_metadata(p)

        return filters_match(
            meta=meta,
            dataset_filter=dataset_filter,
            model_filter=model_filter,
            prompt_filter=prompt_filter,
        )

    if run_folder is not None:
        raw = Path(run_folder)
        candidates: List[Path] = []

        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(real_root / raw)

            if len(raw.parts) == 1:
                candidates.extend(sorted(real_root.glob(f"*/{raw.name}")))

            for root in candidate_repo_roots(repo_root):
                candidates.append(root / raw)
                candidates.append(root / "real_results" / raw)

        for candidate in candidates:
            if is_valid_run_dir(candidate):
                return [candidate.resolve()]

        raise FileNotFoundError(
            f"Could not find run folder '{run_folder}' with a results_*_examples.xlsx file. Tried:\n"
            + "\n".join(f"  - {p}" for p in candidates)
        )

    folders: List[Path] = []

    for examples_path in sorted(real_root.glob("*/*/results_*_examples.xlsx")):
        run_dir = examples_path.parent

        if is_valid_run_dir(run_dir):
            folders.append(run_dir.resolve())

    for examples_path in sorted(real_root.rglob("*examples*.xlsx")):
        run_dir = examples_path.parent

        if is_valid_run_dir(run_dir):
            folders.append(run_dir.resolve())


    out: List[Path] = []
    seen = set()

    for folder in folders:
        key = str(folder)

        if key not in seen:
            out.append(folder)
            seen.add(key)

    if debug:
        print(f"[DEBUG] Found {len(out)} matching run folders.")
        for folder in out:
            print(f"[DEBUG]   {folder}")

    return out


def save_figure(fig: plt.Figure, out_path: Path, save_pdf: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=PLOT_DPI, bbox_inches="tight")

    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")

    plt.close(fig)


# ============================================================
# cMFG* computation
# ============================================================

def compute_cmfg_star(
    confidence: Sequence[Any],
    faithfulness: Sequence[Any],
    n_bins: int = 10,
) -> float:
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


def cmfg_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    cmfg_bins: int,
) -> pd.DataFrame:
    group_cols = list(group_cols)

    if df.empty:
        return pd.DataFrame(columns=[*group_cols, "cmfg_star"])

    rows: List[Dict[str, Any]] = []

    for keys, sub in df.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["cmfg_star"] = compute_cmfg_star(
            sub["final_confidence"],
            sub["faithfulness"],
            n_bins=cmfg_bins,
        )
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Step-level aggregation
# ============================================================

def summarize_step_confidence(group: pd.DataFrame, conf_col: str) -> Dict[str, float]:
    g = group.sort_values("step_idx").copy()

    conf = clip01_series(g[conf_col]).to_numpy(dtype=float)
    conf = conf[np.isfinite(conf)]

    out = {
        "n_valid_step_conf": int(len(conf)),
        "step_conf_min": np.nan,
        "step_conf_first": np.nan,
        "step_conf_last": np.nan,
        "step_conf_mean": np.nan,
        "largest_conf_drop": np.nan,
    }

    if len(conf) == 0:
        return out

    out["step_conf_min"] = float(np.nanmin(conf))
    out["step_conf_first"] = float(conf[0])
    out["step_conf_last"] = float(conf[-1])
    out["step_conf_mean"] = float(np.nanmean(conf))

    if len(conf) >= 2:
        adjacent_drops = conf[:-1] - conf[1:]
        adjacent_drops = np.maximum(adjacent_drops, 0.0)
        out["largest_conf_drop"] = float(np.nanmax(adjacent_drops))

    return out


def build_step_aggregates(
    step_df: pd.DataFrame,
    rcc_step_conf_col: str,
) -> pd.DataFrame:
    if step_df.empty:
        return pd.DataFrame()

    df = normalize_columns(step_df)

    if "idx" not in df.columns:
        raise KeyError("Step-level file is missing required column: idx")

    if "step_idx" not in df.columns:
        df["step_idx"] = df.groupby("idx").cumcount()

    df["idx_key"] = df["idx"].astype(str)

    rows: List[Dict[str, Any]] = []

    for method_key in METHOD_ORDER:
        method_meta = METHODS[method_key]
        conf_candidates = list(method_meta["step_conf_cols"])

        if method_key == "rcc":
            conf_candidates = [
                rcc_step_conf_col,
                *[c for c in conf_candidates if c != rcc_step_conf_col],
            ]

        conf_col = first_existing_col(df, conf_candidates)

        if conf_col is None:
            continue

        for idx_key, group in df.groupby("idx_key", sort=False):
            summary = summarize_step_confidence(group, conf_col)

            rows.append({
                "idx_key": idx_key,
                "method_key": method_key,
                "method": method_meta["label"],
                "step_conf_col": conf_col,
                **summary,
            })

    return pd.DataFrame(rows)


# ============================================================
# Signal computation
# ============================================================

def compute_signals(
    final_confidence: float,
    faithfulness: float,
    avg_decisiveness: float,
    step_conf_min: float,
    largest_conf_drop: float,
) -> Dict[str, float]:
    high_final_conf_low_faith = (
        math.sqrt(final_confidence * (1.0 - faithfulness))
        if np.isfinite(final_confidence) and np.isfinite(faithfulness)
        else np.nan
    )

    high_decisiveness_low_faith = (
        math.sqrt(avg_decisiveness * (1.0 - faithfulness))
        if np.isfinite(avg_decisiveness) and np.isfinite(faithfulness)
        else np.nan
    )

    confidence_exceeds_faith = (
        max(final_confidence - faithfulness, 0.0)
        if np.isfinite(final_confidence) and np.isfinite(faithfulness)
        else np.nan
    )

    decisiveness_exceeds_faith = (
        max(avg_decisiveness - faithfulness, 0.0)
        if np.isfinite(avg_decisiveness) and np.isfinite(faithfulness)
        else np.nan
    )

    return {
        "signal_largest_conf_drop": largest_conf_drop if np.isfinite(largest_conf_drop) else np.nan,
        "signal_min_step_conf": step_conf_min if np.isfinite(step_conf_min) else np.nan,
        "signal_high_final_conf_low_faith": high_final_conf_low_faith,
        "signal_high_decisiveness_low_faith": high_decisiveness_low_faith,
        "signal_confidence_exceeds_faith": confidence_exceeds_faith,
        "signal_decisiveness_exceeds_faith": decisiveness_exceeds_faith,
    }


# ============================================================
# Data collection
# ============================================================

def collect_signal_rows(
    run_folders: Sequence[Path],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    run_summary_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    selected_col_rows: List[Dict[str, Any]] = []

    for folder in tqdm(run_folders, desc="Collecting Traces", unit="run"):
        meta = parse_run_metadata(folder)

        examples_path = find_examples_xlsx(folder)
        step_path = find_step_level_xlsx(folder)

        if examples_path is None:
            missing_rows.append({
                "folder": folder.name,
                "run_path": str(folder),
                "issue": "missing_examples_xlsx",
            })
            continue

        try:
            examples_df_raw = pd.read_excel(examples_path)
        except Exception as e:
            missing_rows.append({
                "folder": folder.name,
                "run_path": str(folder),
                "issue": f"failed_read_examples: {e}",
            })
            continue

        examples_df = normalize_columns(examples_df_raw)

        if "idx" not in examples_df.columns:
            examples_df["idx"] = np.arange(len(examples_df))

        examples_df["idx_key"] = examples_df["idx"].astype(str)

        correct_col = first_existing_col(examples_df, CORRECT_CANDIDATES)

        if correct_col is not None:
            examples_df["correct_num"] = examples_df[correct_col].apply(parse_correct)
        else:
            examples_df["correct_num"] = np.nan
            missing_rows.append({
                "folder": folder.name,
                "run_path": str(folder),
                "issue": "missing_correct_column",
            })

        step_agg = pd.DataFrame()

        if step_path is not None:
            try:
                step_df = pd.read_excel(step_path)
                step_agg = build_step_aggregates(
                    step_df=step_df,
                    rcc_step_conf_col=args.rcc_step_conf_col,
                )
            except Exception as e:
                missing_rows.append({
                    "folder": folder.name,
                    "run_path": str(folder),
                    "issue": f"failed_step_aggregation: {e}",
                })
                step_agg = pd.DataFrame()
        else:
            missing_rows.append({
                "folder": folder.name,
                "run_path": str(folder),
                "issue": "missing_step_level_xlsx",
            })

        step_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}

        if not step_agg.empty:
            for rec in step_agg.to_dict(orient="records"):
                step_lookup[(str(rec["method_key"]), str(rec["idx_key"]))] = rec

        n_examples = int(len(examples_df))
        n_correct_labeled = int(examples_df["correct_num"].notna().sum())
        accuracy = (
            float(examples_df["correct_num"].dropna().mean())
            if examples_df["correct_num"].notna().any()
            else np.nan
        )

        run_summary_rows.append({
            "folder": folder.name,
            "run_path": str(folder),
            "examples_path": str(examples_path),
            "step_path": str(step_path) if step_path is not None else "",
            **meta,
            "n_examples": n_examples,
            "n_correct_labeled": n_correct_labeled,
            "accuracy": accuracy,
        })

        decisiveness_col = first_existing_col(examples_df, DECISIVENESS_CANDIDATES)

        if decisiveness_col is None:
            missing_rows.append({
                "folder": folder.name,
                "run_path": str(folder),
                "issue": "missing_decisiveness_column",
            })

        for method_key in METHOD_ORDER:
            method_meta = METHODS[method_key]
            method_label = method_meta["label"]

            conf_col = first_existing_col(examples_df, method_meta["example_conf_cols"])
            faith_col = first_existing_col(examples_df, method_meta["example_faith_cols"])

            selected_col_rows.append({
                "folder": folder.name,
                "run_path": str(folder),
                **meta,
                "method_key": method_key,
                "method": method_label,
                "selected_confidence_col": conf_col,
                "selected_faithfulness_col": faith_col,
                "selected_decisiveness_col": decisiveness_col,
            })

            if conf_col is None:
                missing_rows.append({
                    "folder": folder.name,
                    "run_path": str(folder),
                    "method_key": method_key,
                    "method": method_label,
                    "issue": "missing_example_confidence_column",
                })

            if faith_col is None:
                missing_rows.append({
                    "folder": folder.name,
                    "run_path": str(folder),
                    "method_key": method_key,
                    "method": method_label,
                    "issue": "missing_example_faithfulness_column",
                })

            for _, row in examples_df.iterrows():
                idx_key = str(row["idx_key"])
                step_rec = step_lookup.get((method_key, idx_key), {})

                final_confidence = (
                    clip01_scalar(row.get(conf_col))
                    if conf_col is not None
                    else np.nan
                )

                faithfulness = (
                    clip01_scalar(row.get(faith_col))
                    if faith_col is not None
                    else np.nan
                )

                avg_decisiveness = (
                    clip01_scalar(row.get(decisiveness_col))
                    if decisiveness_col is not None
                    else np.nan
                )

                step_conf_min = clip01_scalar(step_rec.get("step_conf_min", np.nan))
                largest_conf_drop = clip01_scalar(step_rec.get("largest_conf_drop", np.nan))

                correct_num = row.get("correct_num", np.nan)
                wrong_num = 1.0 - float(correct_num) if pd.notna(correct_num) else np.nan

                signals = compute_signals(
                    final_confidence=final_confidence,
                    faithfulness=faithfulness,
                    avg_decisiveness=avg_decisiveness,
                    step_conf_min=step_conf_min,
                    largest_conf_drop=largest_conf_drop,
                )

                rows.append({
                    "folder": folder.name,
                    "run_path": str(folder),
                    **meta,
                    "idx": row.get("idx"),
                    "idx_key": idx_key,
                    "method_key": method_key,
                    "method": method_label,
                    "correct_num": correct_num,
                    "wrong_num": wrong_num,
                    "final_confidence": final_confidence,
                    "faithfulness": faithfulness,
                    "avg_decisiveness": avg_decisiveness,
                    "n_valid_step_conf": step_rec.get("n_valid_step_conf", np.nan),
                    "step_conf_col": step_rec.get("step_conf_col", np.nan),
                    "step_conf_min": step_conf_min,
                    "step_conf_first": step_rec.get("step_conf_first", np.nan),
                    "step_conf_last": step_rec.get("step_conf_last", np.nan),
                    "step_conf_mean": step_rec.get("step_conf_mean", np.nan),
                    "largest_conf_drop": largest_conf_drop,
                    **signals,
                })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(run_summary_rows),
        pd.DataFrame(missing_rows),
        pd.DataFrame(selected_col_rows),
    )


# ============================================================
# Summary tables
# ============================================================

def safe_wrong_rate(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce").dropna()

    if vals.empty:
        return np.nan

    return float(1.0 - vals.mean())


def build_group_summary(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    cmfg_bins: int,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = list(group_cols)

    agg_spec = {
        "n_method_rows": ("method", "size"),
        "wrong_rate": ("correct_num", safe_wrong_rate),
        "mean_correctness": ("correct_num", "mean"),
        "mean_final_confidence": ("final_confidence", "mean"),
        "mean_avg_decisiveness": ("avg_decisiveness", "mean"),
        "mean_largest_conf_drop_raw": ("largest_conf_drop", "mean"),
        "mean_min_step_conf_raw": ("step_conf_min", "mean"),
    }

    for col in SIGNAL_COLUMNS:
        agg_spec[f"mean_{col}"] = (col, "mean")
        agg_spec[f"median_{col}"] = (col, "median")
        agg_spec[f"p75_{col}"] = (col, lambda x: safe_nanquantile(x, 0.75))
        agg_spec[f"p90_{col}"] = (col, lambda x: safe_nanquantile(x, 0.90))

    summary = (
        df
        .groupby(group_cols, dropna=False, observed=False)
        .agg(**agg_spec)
        .reset_index()
    )

    cmfg_df = cmfg_by_group(
        df=df,
        group_cols=group_cols,
        cmfg_bins=cmfg_bins,
    )

    summary = summary.merge(
        cmfg_df,
        on=group_cols,
        how="left",
    )

    return summary


def build_signal_mean_table(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    group_cols = list(group_cols)

    for keys, sub in df.groupby(group_cols, dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = dict(zip(group_cols, keys))

        for signal_col in SIGNAL_COLUMNS:
            vals = pd.to_numeric(sub[signal_col], errors="coerce")

            records.append({
                **base,
                "signal_col": signal_col,
                "signal": SIGNAL_LABELS[signal_col],
                "n": int(vals.notna().sum()),
                "mean_signal": float(vals.mean()) if vals.notna().any() else np.nan,
                "median_signal": float(vals.median()) if vals.notna().any() else np.nan,
                "p75_signal": safe_nanquantile(vals, 0.75),
                "p90_signal": safe_nanquantile(vals, 0.90),
            })

    return pd.DataFrame(records)


def build_signal_outcome_correlations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []

    group_cols = ["method_key", "method"]

    for keys, sub in df.groupby(group_cols, dropna=False, observed=False):
        method_key, method = keys

        for signal_col in SIGNAL_COLUMNS:
            records.append({
                "method_key": method_key,
                "method": method,
                "signal_col": signal_col,
                "signal": SIGNAL_LABELS[signal_col],
                "spearman_with_wrongness": safe_spearman(sub[signal_col], sub["wrong_num"]),
                "spearman_with_trace_faithfulness": safe_spearman(sub[signal_col], sub["faithfulness"]),
                "spearman_with_final_confidence": safe_spearman(sub[signal_col], sub["final_confidence"]),
                "spearman_with_decisiveness": safe_spearman(sub[signal_col], sub["avg_decisiveness"]),
            })

    return pd.DataFrame(records)


def save_summary_tables(
    df: pd.DataFrame,
    out_dir: Path,
    cmfg_bins: int,
) -> Dict[str, pd.DataFrame]:
    summaries = {
        "method_summary": build_group_summary(
            df,
            ["method_key", "method"],
            cmfg_bins=cmfg_bins,
        ),
        "dataset_summary": build_group_summary(
            df,
            ["dataset_key", "dataset", "method_key", "method"],
            cmfg_bins=cmfg_bins,
        ),
        "model_summary": build_group_summary(
            df,
            ["model_key", "model", "method_key", "method"],
            cmfg_bins=cmfg_bins,
        ),
        "prompt_summary": build_group_summary(
            df,
            ["prompt_key", "prompt", "method_key", "method"],
            cmfg_bins=cmfg_bins,
        ),
        "dataset_model_summary": build_group_summary(
            df,
            ["dataset_key", "dataset", "model_key", "model", "method_key", "method"],
            cmfg_bins=cmfg_bins,
        ),
        "dataset_prompt_summary": build_group_summary(
            df,
            ["dataset_key", "dataset", "prompt_key", "prompt", "method_key", "method"],
            cmfg_bins=cmfg_bins,
        ),
        "dataset_model_prompt_summary": build_group_summary(
            df,
            [
                "dataset_key",
                "dataset",
                "model_key",
                "model",
                "prompt_key",
                "prompt",
                "method_key",
                "method",
            ],
            cmfg_bins=cmfg_bins,
        ),
        "signal_means_method": build_signal_mean_table(
            df,
            ["method_key", "method"],
        ),
        "signal_means_dataset": build_signal_mean_table(
            df,
            ["dataset_key", "dataset", "method_key", "method"],
        ),
        "signal_means_model": build_signal_mean_table(
            df,
            ["model_key", "model", "method_key", "method"],
        ),
        "signal_means_dataset_model": build_signal_mean_table(
            df,
            ["dataset_key", "dataset", "model_key", "model", "method_key", "method"],
        ),
        "signal_means_dataset_prompt": build_signal_mean_table(
            df,
            ["dataset_key", "dataset", "prompt_key", "prompt", "method_key", "method"],
        ),
        "signal_outcome_correlations": build_signal_outcome_correlations(df),
    }

    for name, table in summaries.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)

    return summaries


# ============================================================
# Plotting
# ============================================================

def plot_signal_means_by_method(
    signal_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
) -> None:
    if signal_summary.empty:
        return

    df = signal_summary.copy()
    signal_order = [SIGNAL_LABELS[c] for c in SIGNAL_COLUMNS]

    df["signal"] = pd.Categorical(df["signal"], categories=signal_order, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_LABEL_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(10.6, 5.4), constrained_layout=True)

    sns.barplot(
        data=df,
        y="signal",
        x="mean_signal",
        hue="method",
        order=signal_order,
        hue_order=METHOD_LABEL_ORDER,
        palette=METHOD_PALETTE,
        ax=ax,
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Average Per-Trace Signal Value")
    ax.set_ylabel("")
    ax.set_title("Average Continuous Trace Signals by Method")
    ax.legend(title="Method", loc="lower right", frameon=True)
    ax.grid(True, axis="x", alpha=0.6)
    ax.grid(False, axis="y")

    save_figure(fig, plot_dir / "average_trace_signals_by_method", save_pdf)


def plot_cmfg_star_by_method(
    method_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
) -> None:
    if method_summary.empty or "cmfg_star" not in method_summary.columns:
        return

    df = method_summary.copy()
    df["method"] = pd.Categorical(df["method"], categories=METHOD_LABEL_ORDER, ordered=True)
    df = df.sort_values("method")

    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)

    sns.barplot(
        data=df,
        x="method",
        y="cmfg_star",
        order=METHOD_LABEL_ORDER,
        palette=METHOD_PALETTE,
        ax=ax,
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Method")
    ax.set_ylabel(r"cMFG$^*$")
    ax.set_title(r"Trace-Diagnostic cMFG$^*$ by Method")
    ax.grid(True, axis="y", alpha=0.6)
    ax.grid(False, axis="x")

    save_figure(fig, plot_dir / "cmfg_star_by_method", save_pdf)


def plot_cmfg_star_by_dataset(
    dataset_summary: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
) -> None:
    if dataset_summary.empty or "cmfg_star" not in dataset_summary.columns:
        return

    datasets = order_existing(dataset_summary["dataset"].tolist(), DATASET_ORDER)
    df = dataset_summary.copy()

    fig, ax = plt.subplots(figsize=(8.9, 4.8), constrained_layout=True)

    sns.barplot(
        data=df,
        x="dataset",
        y="cmfg_star",
        hue="method",
        order=datasets,
        hue_order=METHOD_LABEL_ORDER,
        palette=METHOD_PALETTE,
        ax=ax,
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(r"cMFG$^*$")
    ax.set_title(r"Dataset-Level cMFG$^*$ by Method")
    ax.legend(title="Method", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    ax.grid(True, axis="y", alpha=0.6)
    ax.grid(False, axis="x")

    save_figure(fig, plot_dir / "cmfg_star_by_dataset", save_pdf)


def plot_signal_by_dataset(
    signal_dataset: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
) -> None:
    if signal_dataset.empty:
        return

    datasets = order_existing(signal_dataset["dataset"].tolist(), DATASET_ORDER)

    for signal_col in SIGNAL_COLUMNS:
        sub = signal_dataset[signal_dataset["signal_col"] == signal_col].copy()

        if sub.empty:
            continue

        sub["dataset"] = pd.Categorical(sub["dataset"], categories=datasets, ordered=True)
        sub["method"] = pd.Categorical(sub["method"], categories=METHOD_LABEL_ORDER, ordered=True)

        fig, ax = plt.subplots(figsize=(8.9, 4.8), constrained_layout=True)

        sns.barplot(
            data=sub,
            x="dataset",
            y="mean_signal",
            hue="method",
            order=datasets,
            hue_order=METHOD_LABEL_ORDER,
            palette=METHOD_PALETTE,
            ax=ax,
        )

        ax.set_ylim(0, 1)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Average Per-Trace Signal Value")
        ax.set_title(f"{SIGNAL_LABELS[signal_col]} Across Datasets")
        ax.legend(title="Method", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
        ax.grid(True, axis="y", alpha=0.6)
        ax.grid(False, axis="x")

        save_figure(fig, plot_dir / f"{signal_col}_by_dataset", save_pdf)


def plot_signal_by_model(
    signal_model: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
) -> None:
    if signal_model.empty:
        return

    models = order_existing(signal_model["model"].tolist(), MODEL_ORDER)

    for signal_col in SIGNAL_COLUMNS:
        sub = signal_model[signal_model["signal_col"] == signal_col].copy()

        if sub.empty:
            continue

        sub["model"] = pd.Categorical(sub["model"], categories=models, ordered=True)
        sub["method"] = pd.Categorical(sub["method"], categories=METHOD_LABEL_ORDER, ordered=True)

        fig, ax = plt.subplots(figsize=(8.9, 4.8), constrained_layout=True)

        sns.barplot(
            data=sub,
            x="model",
            y="mean_signal",
            hue="method",
            order=models,
            hue_order=METHOD_LABEL_ORDER,
            palette=METHOD_PALETTE,
            ax=ax,
        )

        ax.set_ylim(0, 1)
        ax.set_xlabel("Model")
        ax.set_ylabel("Average Per-Trace Signal Value")
        ax.set_title(f"{SIGNAL_LABELS[signal_col]} Across Models")
        ax.legend(title="Method", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
        ax.grid(True, axis="y", alpha=0.6)
        ax.grid(False, axis="x")

        save_figure(fig, plot_dir / f"{signal_col}_by_model", save_pdf)


def plot_dataset_model_heatmaps(
    signal_dataset_model: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
) -> None:
    if signal_dataset_model.empty:
        return

    datasets = order_existing(signal_dataset_model["dataset"].tolist(), DATASET_ORDER)
    models = order_existing(signal_dataset_model["model"].tolist(), MODEL_ORDER)

    for method_key in METHOD_ORDER:
        method_label = METHODS[method_key]["label"]
        method_df = signal_dataset_model[signal_dataset_model["method_key"] == method_key].copy()

        if method_df.empty:
            continue

        for signal_col in SIGNAL_COLUMNS:
            sub = method_df[method_df["signal_col"] == signal_col].copy()

            if sub.empty:
                continue

            pivot = (
                sub
                .pivot_table(
                    index="dataset",
                    columns="model",
                    values="mean_signal",
                    aggfunc="mean",
                )
                .reindex(index=datasets, columns=models)
            )

            fig, ax = plt.subplots(figsize=(7.9, 4.8), constrained_layout=True)

            sns.heatmap(
                pivot,
                ax=ax,
                cmap=SIGNAL_CMAPS.get(signal_col, "Reds"),
                vmin=0,
                vmax=1,
                linewidths=0.5,
                linecolor="white",
                annot=True,
                fmt=".2f",
                cbar_kws={"label": "Average Per-Trace Signal Value"},
            )

            ax.set_title(f"{method_label}: {SIGNAL_LABELS[signal_col]}")
            ax.set_xlabel("Model")
            ax.set_ylabel("Dataset")
            ax.tick_params(axis="x", rotation=25)

            save_figure(
                fig,
                plot_dir / f"{method_key}_{signal_col}_heatmap_dataset_model",
                save_pdf,
            )


def plot_dataset_prompt_heatmaps(
    signal_dataset_prompt: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
) -> None:
    if signal_dataset_prompt.empty:
        return

    datasets = order_existing(signal_dataset_prompt["dataset"].tolist(), DATASET_ORDER)
    prompts = order_existing(signal_dataset_prompt["prompt"].tolist(), PROMPT_ORDER)

    for method_key in METHOD_ORDER:
        method_label = METHODS[method_key]["label"]
        method_df = signal_dataset_prompt[signal_dataset_prompt["method_key"] == method_key].copy()

        if method_df.empty:
            continue

        for signal_col in SIGNAL_COLUMNS:
            sub = method_df[method_df["signal_col"] == signal_col].copy()

            if sub.empty:
                continue

            pivot = (
                sub
                .pivot_table(
                    index="dataset",
                    columns="prompt",
                    values="mean_signal",
                    aggfunc="mean",
                )
                .reindex(index=datasets, columns=prompts)
            )

            fig_width = max(10.0, 0.75 * len(prompts) + 3.0)
            fig, ax = plt.subplots(figsize=(fig_width, 4.8), constrained_layout=True)

            sns.heatmap(
                pivot,
                ax=ax,
                cmap=SIGNAL_CMAPS.get(signal_col, "Reds"),
                vmin=0,
                vmax=1,
                linewidths=0.5,
                linecolor="white",
                annot=True,
                fmt=".2f",
                cbar_kws={"label": "Average Per-Trace Signal Value"},
            )

            ax.set_title(f"{method_label}: {SIGNAL_LABELS[signal_col]} by Dataset and Prompt")
            ax.set_xlabel("Prompt")
            ax.set_ylabel("Dataset")
            ax.tick_params(axis="x", rotation=35)

            save_figure(
                fig,
                plot_dir / f"{method_key}_{signal_col}_heatmap_dataset_prompt",
                save_pdf,
            )


def plot_signal_outcome_correlation_heatmaps(
    corr_df: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
) -> None:
    if corr_df.empty:
        return

    signal_order = [SIGNAL_LABELS[c] for c in SIGNAL_COLUMNS]

    metrics = [
        (
            "spearman_with_wrongness",
            "Spearman Correlation with Wrongness",
            "signal_spearman_with_wrongness_heatmap",
        ),
        (
            "spearman_with_trace_faithfulness",
            "Spearman Correlation with Trace Faithfulness",
            "signal_spearman_with_trace_faithfulness_heatmap",
        ),
    ]

    for value_col, title, filename in metrics:
        pivot = (
            corr_df
            .pivot_table(
                index="signal",
                columns="method",
                values=value_col,
                aggfunc="mean",
            )
            .reindex(index=signal_order, columns=METHOD_LABEL_ORDER)
        )

        fig, ax = plt.subplots(figsize=(7.8, 5.1), constrained_layout=True)

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=0.5,
            linecolor="white",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Spearman ρ"},
        )

        ax.set_title(title)
        ax.set_xlabel("Method")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)
        ax.tick_params(axis="y", rotation=0)

        save_figure(fig, plot_dir / filename, save_pdf)


def plot_step_signal_scatter(
    df: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
) -> None:
    if df.empty:
        return

    plot_df = df.dropna(subset=["signal_min_step_conf", "signal_largest_conf_drop"]).copy()

    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)

    for method_key in METHOD_ORDER:
        method_label = METHODS[method_key]["label"]
        color = METHODS[method_key]["color"]
        sub = plot_df[plot_df["method_key"] == method_key].copy()

        if sub.empty:
            continue

        ax.scatter(
            sub["signal_min_step_conf"],
            sub["signal_largest_conf_drop"],
            color=color,
            alpha=0.58,
            s=38,
            edgecolor="white",
            linewidth=0.35,
            label=method_label,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Minimum Step Confidence Per Trace")
    ax.set_ylabel("Largest Adjacent Confidence Drop Per Trace")
    ax.set_title("Step-Level Temporal Confidence Signals")
    ax.legend(title="Method", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    ax.grid(True, alpha=0.6)

    save_figure(fig, plot_dir / "min_step_confidence_vs_largest_confidence_drop", save_pdf)


def plot_final_conf_vs_faith_scatter(
    df: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
) -> None:
    if df.empty:
        return

    plot_df = df.dropna(subset=["final_confidence", "faithfulness"]).copy()

    if plot_df.empty:
        return

    fig, axes = plt.subplots(
        1,
        len(METHOD_ORDER),
        figsize=(5.2 * len(METHOD_ORDER), 4.9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes)
    mappable = None

    for ax, method_key in zip(axes, METHOD_ORDER):
        method_label = METHODS[method_key]["label"]
        sub = plot_df[plot_df["method_key"] == method_key].copy()

        if not sub.empty:
            sizes = 24.0 + 120.0 * pd.to_numeric(
                sub["signal_high_final_conf_low_faith"],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)

            colors = pd.to_numeric(sub["avg_decisiveness"], errors="coerce").to_numpy(dtype=float)

            mappable = ax.scatter(
                sub["final_confidence"],
                sub["faithfulness"],
                c=colors,
                s=sizes,
                cmap="viridis",
                vmin=0,
                vmax=1,
                alpha=0.62,
                edgecolor="white",
                linewidth=0.35,
            )

        ax.set_title(method_label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Final Confidence")
        ax.grid(True, alpha=0.55)

    axes[0].set_ylabel("Trace Faithfulness")

    if mappable is not None:
        cbar = fig.colorbar(
            mappable,
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.02,
        )
        cbar.set_label("Average Decisiveness")

    fig.suptitle(
        "Final Confidence vs. Trace Faithfulness\n"
        "Point Size = High Final Confidence + Low Faithfulness Signal",
        fontsize=15,
    )

    save_figure(fig, plot_dir / "final_confidence_vs_trace_faithfulness_by_method", save_pdf)


def plot_decisiveness_vs_faith_scatter(
    df: pd.DataFrame,
    plot_dir: Path,
    save_pdf: bool,
) -> None:
    if df.empty:
        return

    plot_df = df.dropna(subset=["avg_decisiveness", "faithfulness"]).copy()

    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.1, 6.0), constrained_layout=True)

    for method_key in METHOD_ORDER:
        method_label = METHODS[method_key]["label"]
        color = METHODS[method_key]["color"]
        sub = plot_df[plot_df["method_key"] == method_key].copy()

        if sub.empty:
            continue

        sizes = 24.0 + 120.0 * pd.to_numeric(
            sub["signal_high_decisiveness_low_faith"],
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)

        ax.scatter(
            sub["avg_decisiveness"],
            sub["faithfulness"],
            s=sizes,
            color=color,
            alpha=0.58,
            edgecolor="white",
            linewidth=0.35,
            label=method_label,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Average Decisiveness")
    ax.set_ylabel("Trace Faithfulness")
    ax.set_title(
        "Decisiveness vs. Trace Faithfulness\n"
        "Point Size = High Decisiveness + Low Faithfulness Signal"
    )
    ax.legend(title="Method", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    ax.grid(True, alpha=0.6)

    save_figure(fig, plot_dir / "decisiveness_vs_trace_faithfulness_by_method", save_pdf)


def make_all_plots(
    df: pd.DataFrame,
    summaries: Dict[str, pd.DataFrame],
    plot_dir: Path,
    save_pdf: bool,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_signal_means_by_method(
        summaries["signal_means_method"],
        plot_dir,
        save_pdf,
    )

    plot_cmfg_star_by_method(
        summaries["method_summary"],
        plot_dir,
        save_pdf,
    )

    plot_cmfg_star_by_dataset(
        summaries["dataset_summary"],
        plot_dir,
        save_pdf,
    )

    plot_signal_by_dataset(
        summaries["signal_means_dataset"],
        plot_dir,
        save_pdf,
    )

    plot_signal_by_model(
        summaries["signal_means_model"],
        plot_dir,
        save_pdf,
    )

    plot_dataset_model_heatmaps(
        summaries["signal_means_dataset_model"],
        plot_dir,
        save_pdf,
    )

    plot_dataset_prompt_heatmaps(
        summaries["signal_means_dataset_prompt"],
        plot_dir,
        save_pdf,
    )

    plot_signal_outcome_correlation_heatmaps(
        summaries["signal_outcome_correlations"],
        plot_dir,
        save_pdf,
    )

    plot_step_signal_scatter(
        df,
        plot_dir,
        save_pdf,
    )

    plot_final_conf_vs_faith_scatter(
        df,
        plot_dir,
        save_pdf,
    )

    plot_decisiveness_vs_faith_scatter(
        df,
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
        debug=args.debug,
    )

    if not run_folders:
        raise RuntimeError(
            "No matching run folders found. Expected files like:\n"
            "  real_results/ds_8b/aime_b/results_*_examples.xlsx\n"
            "  real_results/ds_8b/aime_b/results_*_step_level.xlsx\n"
            "  real_results/qwq_32b/sgpqa_msh_perc/results_*_examples.xlsx\n"
            "Run with --debug to see searched locations."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    save_pdf = not args.no_pdf

    print(f"Found {len(run_folders)} matching run folders.")
    print("Collecting continuous trace signals...")

    signal_df, run_summary_df, missing_df, selected_cols_df = collect_signal_rows(
        run_folders=run_folders,
        args=args,
    )

    if signal_df.empty:
        raise RuntimeError("No method-specific trace rows were collected.")

    print(f"Collected {len(signal_df)} method-specific rows in memory.")
    print("Saving summary-only CSV outputs...")

    run_summary_df.to_csv(args.output_dir / "run_level_summary.csv", index=False)
    missing_df.to_csv(args.output_dir / "missing_or_skipped_report.csv", index=False)
    selected_cols_df.to_csv(args.output_dir / "selected_columns_report.csv", index=False)

    summaries = save_summary_tables(
        df=signal_df,
        out_dir=args.output_dir,
        cmfg_bins=args.cmfg_bins,
    )

    print("Generating paper-style plots...")

    make_all_plots(
        df=signal_df,
        summaries=summaries,
        plot_dir=plot_dir,
        save_pdf=save_pdf,
    )

    print("")
    print("Done.")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Plots directory:  {plot_dir.resolve()}")
    print("")
    print("Key outputs:")
    print(f"  {args.output_dir / 'method_summary.csv'}")
    print(f"  {args.output_dir / 'dataset_summary.csv'}")
    print(f"  {args.output_dir / 'model_summary.csv'}")
    print(f"  {plot_dir / 'average_trace_signals_by_method.png'}")
    print(f"  {plot_dir / 'cmfg_star_by_method.png'}")
    print(f"  {plot_dir / 'min_step_confidence_vs_largest_confidence_drop.png'}")


if __name__ == "__main__":
    main()