from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent
DEFAULT_REAL_RESULTS_DIR = Path("real_results")
DEFAULT_OUTPUTS_DIR = ANALYSIS_DIR / "outputs"

DATASET_LABELS: Dict[str, str] = {
    "aime": "AIME",
    "hle": "HLE",
    "legal": "LegalBench",
    "legalbench": "LegalBench",
    "musr": "MuSR",
    "sgpqa": "SuperGPQA",
    "supergpqa": "SuperGPQA",
}

DATASET_ORDER = ["HLE", "SuperGPQA", "AIME", "MuSR", "LegalBench"]
DATASET_ORDER_WITH_LEGACY = DATASET_ORDER

MODEL_FULL_LABELS: Dict[str, str] = {
    "ds_8b": "DeepSeek-R1-8B",
    "deepseek": "DeepSeek-R1-8B",
    "ds": "DeepSeek-R1-8B",
    "qwq_32b": "QwQ-32B",
    "qwq": "QwQ-32B",
}

MODEL_SHORT_LABELS: Dict[str, str] = {
    "ds_8b": "DeepSeek",
    "deepseek": "DeepSeek",
    "ds": "DeepSeek",
    "qwq_32b": "QwQ",
    "qwq": "QwQ",
}

MODEL_LABELS_FROM_MODEL_LINE: Dict[str, str] = {
    "deepseek-ai/deepseek-r1-0528-qwen3-8b": "DeepSeek-R1-8B",
    "qwen/qwq-32b": "QwQ-32B",
    "qwq-32b": "QwQ-32B",
}

MODEL_ORDER_FULL = ["DeepSeek-R1-8B", "QwQ-32B"]
MODEL_ORDER_SHORT = ["DeepSeek", "QwQ"]

PROMPT_LABELS: Dict[str, str] = {
    "b": "baseline",
    "blank": "baseline",
    "base": "baseline",
    "baseline": "baseline",
    "basic": "basic",
    "gen": "genuine",
    "genuine": "genuine",
    "hum": "human",
    "human": "human",
    "perc": "perception",
    "perception": "perception",
    "sm": "self_monitoring",
    "self_monitoring": "self_monitoring",
    "msh": "ms_hedge",
    "ms_hedge": "ms_hedge",
    "msh_human": "sys:ms_hedge+human",
    "msh_perc": "msh+perception",
    "msh_perception": "msh+perception",
    "sys_msh_perc": "sys:ms_hedge+perception",
    "sys_msh_perception": "sys:ms_hedge+perception",
}

PROMPT_DISPLAY_LABELS: Dict[str, str] = {
    "baseline": "Baseline",
    "perception": "Perception",
    "msh_perception": "MetSens+Hedge + Perception",
    "msh+perception": "MetSens+Hedge + Perception",
    "ms_hedge": "MetSens+Hedge",
    "sys:ms_hedge+human": "sys:ms_hedge+human",
    "sys:ms_hedge+perception": "sys:ms_hedge+perception",
}

PROMPT_SUFFIXES = [
    "_sys_msh_perception",
    "_sys_msh_perc",
    "_msh_perception",
    "_msh_perc",
    "_msh_human",
    "_self_monitoring",
    "_perception",
    "_baseline",
    "_genuine",
    "_basic",
    "_human",
    "_perc",
    "_msh",
    "_gen",
    "_hum",
    "_sm",
    "_b",
]

PROMPT_SUFFIX_TO_LABEL = {
    "_msh_perc": "msh_perception",
    "_msh_perception": "msh_perception",
    "_perc": "perception",
    "_perception": "perception",
    "_b": "baseline",
    "_baseline": "baseline",
}

PROMPT_ORDER = [
    "baseline",
    "basic",
    "genuine",
    "human",
    "perception",
    "self_monitoring",
    "ms_hedge",
    "msh+perception",
    "sys:ms_hedge+human",
    "sys:ms_hedge+perception",
]
PROMPT_ORDER_PUBLIC = ["perception", "msh+perception"]

METHOD_ORDER = ["rcc", "sampling", "deepconf"]
METHOD_LABELS = {"rcc": "RCC", "sampling": "Sampling", "deepconf": "DeepConf"}
METHOD_COLORS = {"rcc": "#4C78A8", "sampling": "#59A14F", "deepconf": "#9C6ADE"}
METHOD_TUPLES = [(key, METHOD_LABELS[key]) for key in METHOD_ORDER]
METHOD_LABEL_ORDER = [METHOD_LABELS[key] for key in METHOD_ORDER]

METHODS_CONF_BINS = {
    key: {
        "label": METHOD_LABELS[key],
        "conf_candidates": {
            "rcc": ["rcc_confidence", "rcc_conf", "rcc_p", "cr"],
            "sampling": ["sampling_conf", "sampling_confidence", "cs"],
            "deepconf": ["deepconf_confidence", "deepconf_conf", "deepconf", "cd"],
        }[key],
        "faith_candidates": {
            "rcc": ["faithfulness_rcc", "faith_rcc", "fr"],
            "sampling": ["faithfulness_sampling", "faith_sampling", "fs"],
            "deepconf": ["faithfulness_deepconf", "faith_deepconf", "fd"],
        }[key],
        "color": METHOD_COLORS[key],
    }
    for key in METHOD_ORDER
}

METHODS_CANDIDATES = {
    key: {
        "label": METHOD_LABELS[key],
        "confidence_candidates": METHODS_CONF_BINS[key]["conf_candidates"],
        "faithfulness_candidates": METHODS_CONF_BINS[key]["faith_candidates"],
        "color": METHOD_COLORS[key],
    }
    for key in METHOD_ORDER
}

METHODS_FIXED_COLS = {
    "rcc": {"label": "RCC", "conf_col": "rcc_confidence", "faith_col": "faithfulness_rcc", "color": METHOD_COLORS["rcc"]},
    "sampling": {"label": "Sampling", "conf_col": "sampling_conf", "faith_col": "faithfulness_sampling", "color": METHOD_COLORS["sampling"]},
    "deepconf": {"label": "DeepConf", "conf_col": "deepconf_confidence", "faith_col": "faithfulness_deepconf", "color": METHOD_COLORS["deepconf"]},
}

METHODS_FIXED_COLS_TITLE = {
    "RCC": {"conf_col": "rcc_confidence", "faith_col": "faithfulness_rcc", "color": METHOD_COLORS["rcc"]},
    "Sampling": {"conf_col": "sampling_conf", "faith_col": "faithfulness_sampling", "color": METHOD_COLORS["sampling"]},
    "DeepConf": {"conf_col": "deepconf_confidence", "faith_col": "faithfulness_deepconf", "color": METHOD_COLORS["deepconf"]},
}

METHODS_TRACE_SIGNALS = {
    "rcc": {
        "label": "RCC",
        "color": METHOD_COLORS["rcc"],
        "example_conf_cols": ["rcc_confidence", "rcc_conf", "rcc_p", "cr"],
        "example_faith_cols": ["faithfulness_rcc", "faith_rcc", "fr"],
        "step_conf_cols": ["rcc_p", "rcc_q", "step_rcc_p", "step_rcc_q"],
    },
    "sampling": {
        "label": "Sampling",
        "color": METHOD_COLORS["sampling"],
        "example_conf_cols": ["sampling_conf", "sampling_confidence", "cs"],
        "example_faith_cols": ["faithfulness_sampling", "faith_sampling", "fs"],
        "step_conf_cols": ["sampling_conf", "step_sampling_conf", "sampling_confidence"],
    },
    "deepconf": {
        "label": "DeepConf",
        "color": METHOD_COLORS["deepconf"],
        "example_conf_cols": ["deepconf_confidence", "deepconf_conf", "deepconf", "cd"],
        "example_faith_cols": ["faithfulness_deepconf", "faith_deepconf", "fd"],
        "step_conf_cols": ["deepconf", "step_deepconf", "deepconf_confidence"],
    },
}

METHOD_COLS_CLUSTERING = {
    "RCC": {"confidence": ["rcc_confidence", "rcc_conf", "rcc_p", "cr"], "faithfulness": ["faithfulness_rcc", "faith_rcc", "fr"]},
    "DeepConf": {"confidence": ["deepconf_confidence", "deepconf", "deepconf_conf", "cd"], "faithfulness": ["faithfulness_deepconf", "faith_deepconf", "fd"]},
    "Sampling": {"confidence": ["sampling_conf", "sampling_confidence", "cs"], "faithfulness": ["faithfulness_sampling", "faith_sampling", "fs"]},
}

DECISIVENESS_CANDIDATES = ["avg_decisiveness", "mean_decisiveness", "decisiveness", "avg_dec", "mean_dec", "dec"]
CORRECT_CANDIDATES = ["correct", "is_correct", "final_correct", "answer_correct", "accuracy", "acc"]
TRUE_STRINGS = {"true", "1", "yes", "correct"}
FALSE_STRINGS = {"false", "0", "no", "incorrect"}

TRACE_FAITH_COLS = {
    "rcc": ["faithfulness_rcc", "faith_rcc", "fr"],
    "sampling": ["faithfulness_sampling", "faith_sampling", "fs"],
    "deepconf": ["faithfulness_deepconf", "faith_deepconf", "fd"],
}
STEP_FAITH_COLS = {
    "rcc": ["faith_rcc", "step_faith_rcc", "faithfulness_rcc", "fr"],
    "sampling": ["faith_sampling", "step_faith_sampling", "faithfulness_sampling", "fs"],
    "deepconf": ["faith_deepconf", "step_faith_deepconf", "faithfulness_deepconf", "fd"],
}
TRACE_TOKEN_COLS = ["deepconf_num_tokens", "num_tokens", "trace_tokens", "generated_tokens", "output_tokens", "completion_tokens"]
TRACE_TEXT_COLS = ["generated_text", "generation", "response", "full_response", "output", "completion", "text"]
STEP_TOKEN_COLS = ["step_num_tokens", "num_tokens", "tokens", "step_tokens"]
STEP_TEXT_COLS = ["step_text", "step", "reasoning_step", "text"]


def clean_text(text: Any) -> str:
    return " ".join(str(text).replace("\xa0", " ").split()).strip()


def safe_filename(value: Any) -> str:
    value = clean_text(value).replace(" ", "_")
    value = value.replace("/", "__").replace("\\", "__")
    return re.sub(r"[^A-Za-z0-9_.+=@-]+", "_", value).strip("_") or "unnamed"


def normalize_col(column: str) -> str:
    return str(column).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col(c) for c in df.columns]
    return df


def default_output_dir(name: str) -> Path:
    override = os.environ.get("ANALYSIS_OUTPUT_ROOT")
    root = Path(override).expanduser().resolve() if override else DEFAULT_OUTPUTS_DIR
    return root / safe_filename(name)


def output_dir_for_run(output_name: str, run_folder: Path) -> Path:
    out_dir = default_output_dir(output_name) / run_folder_output_id(run_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_folder_output_id(run_folder: Path) -> str:
    run_folder = Path(run_folder)
    return safe_filename(f"{run_folder.parent.name}__{run_folder.name}")


def resolve_repo_root(repo_root: Optional[Path] = None) -> Path:
    return REPO_ROOT if repo_root is None else Path(repo_root).expanduser().resolve()


def resolve_real_results_root(repo_root: Optional[Path] = None, real_results_dir: Path = DEFAULT_REAL_RESULTS_DIR) -> Path:
    real_results_dir = Path(real_results_dir).expanduser()
    if real_results_dir.is_absolute():
        return real_results_dir.resolve()
    return (resolve_repo_root(repo_root) / real_results_dir).resolve()


def find_examples_xlsx(folder: Path) -> Optional[Path]:
    candidates = sorted(folder.glob("results_*_examples.xlsx"))
    if candidates:
        return candidates[0]
    candidates = sorted(p for p in folder.glob("*.xlsx") if "examples" in p.name.lower())
    return candidates[0] if candidates else None


def find_step_level_xlsx(folder: Path) -> Optional[Path]:
    candidates = sorted(folder.glob("results_*_step_level.xlsx"))
    if candidates:
        return candidates[0]
    candidates = sorted(folder.glob("results_*_steps.xlsx"))
    if candidates:
        return candidates[0]
    candidates = sorted(folder.glob("*.xlsx"))
    preferred = [p for p in candidates if "step" in p.name.lower() and "level" in p.name.lower()]
    if preferred:
        return preferred[0]
    fallback = [p for p in candidates if "step" in p.name.lower() and "examples" not in p.name.lower()]
    return fallback[0] if fallback else None


def parse_prompt_from_run_name(run_name: str) -> Tuple[str, str, str]:
    run_name = clean_text(run_name)
    for suffix in PROMPT_SUFFIXES:
        if run_name.endswith(suffix):
            dataset_key = run_name[:-len(suffix)]
            prompt_key = suffix[1:]
            return dataset_key, prompt_key, PROMPT_LABELS.get(prompt_key, prompt_key)
    return run_name, "unknown", "unknown"


def parse_run_folder_metadata(folder: Path | str) -> Dict[str, str]:
    folder_path = Path(folder)
    dataset_key, prompt_key, prompt = parse_prompt_from_run_name(folder_path.name)
    model_key = folder_path.parent.name.lower()
    return {
        "dataset_key": dataset_key,
        "dataset": DATASET_LABELS.get(dataset_key.lower(), dataset_key),
        "model_key": model_key,
        "model": MODEL_SHORT_LABELS.get(model_key, model_key),
        "model_full": MODEL_FULL_LABELS.get(model_key, model_key),
        "prompt_key": prompt_key,
        "prompt": prompt,
        "run_name": folder_path.name,
    }


def _matches_filter(value: str, label: str, user_filter: Optional[str]) -> bool:
    if user_filter is None:
        return True
    f = str(user_filter).lower()
    return f in {str(value).lower(), str(label).lower()}


def metadata_matches(meta: Dict[str, str], dataset_filter: Optional[str] = None, model_filter: Optional[str] = None, prompt_filter: Optional[str] = None) -> bool:
    return (
        _matches_filter(meta.get("dataset_key", ""), meta.get("dataset", ""), dataset_filter)
        and _matches_filter(meta.get("model_key", ""), meta.get("model", ""), model_filter)
        and _matches_filter(meta.get("prompt_key", ""), meta.get("prompt", ""), prompt_filter)
    )


def list_analysis_run_folders(
    repo_root: Optional[Path] = None,
    real_results_dir: Path = DEFAULT_REAL_RESULTS_DIR,
    run_folder: Optional[str] = None,
    dataset_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    prompt_filter: Optional[str] = None,
    require_examples: bool = True,
    require_steps: bool = False,
) -> List[Path]:
    repo = resolve_repo_root(repo_root)
    real_root = resolve_real_results_root(repo, real_results_dir)

    def is_valid(path: Path) -> bool:
        if not path.is_dir():
            return False
        if require_examples and find_examples_xlsx(path) is None:
            return False
        if require_steps and find_step_level_xlsx(path) is None:
            return False
        return metadata_matches(parse_run_folder_metadata(path), dataset_filter, model_filter, prompt_filter)

    if run_folder is not None:
        raw = Path(run_folder).expanduser()
        candidates: List[Path] = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend([repo / raw, real_root / raw])
            if len(raw.parts) == 1:
                candidates.extend(sorted(real_root.glob(f"*/{raw.name}")))
        for candidate in candidates:
            if is_valid(candidate):
                return [candidate.resolve()]
        raise FileNotFoundError(
            f"Could not find run folder '{run_folder}'. Tried:\n" + "\n".join(f"  - {p}" for p in candidates)
        )

    folders: List[Path] = []
    if real_root.is_dir():
        for model_dir in sorted(real_root.iterdir()):
            if not model_dir.is_dir():
                continue
            for run_dir in sorted(model_dir.iterdir()):
                if is_valid(run_dir):
                    folders.append(run_dir.resolve())

    seen = set()
    out: List[Path] = []
    for folder in folders:
        key = str(folder)
        if key not in seen:
            seen.add(key)
            out.append(folder)
    return out


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = set(df.columns)
    for candidate in candidates:
        normalized = normalize_col(candidate)
        if normalized in cols:
            return normalized
    return None


def parse_correct(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value)):
        return float(value >= 0.5)
    normalized = str(value).strip().lower()
    if normalized in TRUE_STRINGS:
        return 1.0
    if normalized in FALSE_STRINGS:
        return 0.0
    return None


def apply_plot_style(plt_module: Any, sns_module: Any, grid_color: str = "#e6e6e6", title_size: int = 13) -> None:
    sns_module.set_theme(style="whitegrid")
    plt_module.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": grid_color,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.85,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": title_size,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
