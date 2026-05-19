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
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# ============================================================
# Fixed configuration
# ============================================================

REAL_RESULTS_ROOT = _common.DEFAULT_REAL_RESULTS_DIR
OUTDIR = _default_output_dir("clustering_output_dataset_points")

RANDOM_STATE = 0
MAX_AUTO_K = 8
STANDARDIZE_FEATURES = True

FIGSIZE = (11, 8)
POINT_SIZE = 180
LABEL_POINTS = True

CMFG_STAR_BINS = 10

# Column names in the parsed run-level cache.
CMFG_METRICS = {
    "cMFG*_RCC": "RCC",
    "cMFG*_DeepConf": "DeepConf",
    "cMFG*_Sampling": "Sampling",
}

MODEL_MAP = _common.MODEL_FULL_LABELS

DATASET_MAP = _common.DATASET_LABELS

PROMPT_SUFFIXES = _common.PROMPT_SUFFIX_TO_LABEL

METHOD_COLS = _common.METHOD_COLS_CLUSTERING

CACHE_RUN_METRICS = "parsed_run_cmfg_star_metrics.csv"
CACHE_DATASET_VECTORS = "dataset_point_vectors.csv"
CACHE_POINT_META = "dataset_point_metadata.csv"


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

def clean_text(text: str) -> str:
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


def normalize_condition(condition: str) -> str:
    s = clean_text(condition).lower()
    s = s.replace("sys:", "")
    s = s.replace(" ", "")
    s = s.replace("-", "_")

    aliases = {
        "b": "baseline",
        "base": "baseline",
        "blank": "baseline",
        "baseline": "baseline",
        "perc": "perception",
        "perception": "perception",
        "mshperc": "msh_perception",
        "msh_perc": "msh_perception",
        "msh+perc": "msh_perception",
        "msh+perception": "msh_perception",
        "mshedge+perception": "msh_perception",
        "metsens+hedge+perception": "msh_perception",
    }

    return aliases.get(s, s)


def condition_display_name(condition: str) -> str:
    labels = {
        "baseline": "Baseline",
        "perception": "Perception",
        "msh_perception": "MetSens+Hedge + Perception",
    }
    return labels.get(condition, condition)


def extract_family(model_name: str) -> str:
    model_name = clean_text(model_name)

    if "deepseek" in model_name.lower():
        return "DeepSeek"
    if "qwq" in model_name.lower() or "qwen" in model_name.lower():
        return "Qwen"

    return model_name.split("-")[0]


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        c_norm = normalize_col(c)
        if c_norm in cols:
            return c_norm
    return None


def find_original_col(original_cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    norm_to_original = {normalize_col(c): c for c in original_cols}

    for c in candidates:
        c_norm = normalize_col(c)
        if c_norm in norm_to_original:
            return norm_to_original[c_norm]

    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_dataset_prompt_from_folder(folder_name: str) -> Tuple[str, str]:
    folder_name = folder_name.strip()

    prompt = "unknown"
    dataset_key = folder_name

    for suffix, prompt_name in PROMPT_SUFFIXES.items():
        if folder_name.endswith(suffix):
            prompt = prompt_name
            dataset_key = folder_name[: -len(suffix)]
            break

    dataset = DATASET_MAP.get(dataset_key, dataset_key)
    return dataset, normalize_condition(prompt)


def find_examples_file(run_dir: Path) -> Optional[Path]:
    files = sorted(run_dir.glob("results_*_examples.xlsx"))
    return files[0] if files else None


def read_examples_needed_columns(path: Path) -> pd.DataFrame:
    """
    Fast reader: loads only confidence/faithfulness columns needed for cMFG*.
    """
    original_cols = list(pd.read_excel(path, nrows=0).columns)
    needed_cols = set()

    for specs in METHOD_COLS.values():
        conf_col = find_original_col(original_cols, specs["confidence"])
        faith_col = find_original_col(original_cols, specs["faithfulness"])

        if conf_col is not None:
            needed_cols.add(conf_col)
        if faith_col is not None:
            needed_cols.add(faith_col)

    if not needed_cols:
        return pd.DataFrame()

    df = pd.read_excel(path, usecols=list(needed_cols))
    return normalize_columns(df)


# ============================================================
# cMFG* computation
# ============================================================

def compute_cmfg_star(
    confidence: pd.Series,
    faithfulness: pd.Series,
    n_bins: int = CMFG_STAR_BINS,
) -> float:
    """
    Computes width-weighted conditional MFG, cMFG*.

    Steps:
      1. Sort examples by confidence.
      2. Partition into equal-mass bins.
      3. Average faithfulness inside each bin.
      4. Weight each bin by confidence-axis width.
      5. Integrate over empirical confidence support.
    """
    df = pd.DataFrame({
        "confidence": safe_numeric(confidence),
        "faithfulness": safe_numeric(faithfulness),
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

    values = np.array(bin_faithfulness, dtype=float)
    weights = np.array(bin_widths, dtype=float)

    if weights.sum() <= 0:
        return float(df["faithfulness"].mean())

    return float(np.average(values, weights=weights))


def compute_per_run_metrics_from_examples(
    examples_df: pd.DataFrame,
    n_bins: int,
) -> Dict[str, float]:
    examples_df = normalize_columns(examples_df)
    out: Dict[str, float] = {}

    for method_name, specs in METHOD_COLS.items():
        conf_col = find_col(examples_df, specs["confidence"])
        faith_col = find_col(examples_df, specs["faithfulness"])

        out_key = f"cMFG*_{method_name}"

        if conf_col is None or faith_col is None:
            out[out_key] = np.nan
            continue

        out[out_key] = compute_cmfg_star(
            confidence=examples_df[conf_col],
            faithfulness=examples_df[faith_col],
            n_bins=n_bins,
        )

    return out


# ============================================================
# Load current real_results format
# ============================================================

def parse_real_results(root: Path, n_bins: int) -> pd.DataFrame:
    if not root.exists():
        raise FileNotFoundError(f"Could not find real_results root: {root.resolve()}")

    records: List[Dict] = []

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue

        model_key = model_dir.name
        model_name = MODEL_MAP.get(model_key, model_key)
        family = extract_family(model_name)

        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            dataset, condition = parse_dataset_prompt_from_folder(run_dir.name)
            examples_file = find_examples_file(run_dir)

            if examples_file is None:
                print(f"[WARN] No results_*_examples.xlsx found in {run_dir}")
                continue

            try:
                examples_df = read_examples_needed_columns(examples_file)
            except Exception as e:
                print(f"[WARN] Failed to read {examples_file}: {e}")
                continue

            if examples_df.empty:
                print(f"[WARN] No usable confidence/faithfulness columns in {examples_file}")
                continue

            metrics = compute_per_run_metrics_from_examples(
                examples_df=examples_df,
                n_bins=n_bins,
            )

            record = {
                "model": model_name,
                "model_key": model_key,
                "family": family,
                "dataset": dataset,
                "condition_raw": condition,
                "condition": normalize_condition(condition),
                "run_folder": run_dir.name,
                "n_examples": len(examples_df),
                "cMFG*_RCC": metrics.get("cMFG*_RCC", np.nan),
                "cMFG*_DeepConf": metrics.get("cMFG*_DeepConf", np.nan),
                "cMFG*_Sampling": metrics.get("cMFG*_Sampling", np.nan),
            }

            records.append(record)

            print(
                f"[LOAD] {model_name:18s} | {dataset:12s} | {condition:15s} | "
                f"n={len(examples_df):4d} | "
                f"RCC={record['cMFG*_RCC']:.4f} "
                f"DeepConf={record['cMFG*_DeepConf']:.4f} "
                f"Sampling={record['cMFG*_Sampling']:.4f}"
            )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(f"Parsed zero runs from {root.resolve()}")

    return df


# ============================================================
# Caching
# ============================================================

def cache_is_usable(outdir: Path, bins: int) -> bool:
    run_cache = outdir / CACHE_RUN_METRICS
    vector_cache = outdir / CACHE_DATASET_VECTORS
    meta_cache = outdir / CACHE_POINT_META

    if not (run_cache.is_file() and vector_cache.is_file() and meta_cache.is_file()):
        return False

    try:
        raw_df = pd.read_csv(run_cache)
    except Exception:
        return False

    if "cmfg_star_bins" not in raw_df.columns:
        return False

    cached_bins = pd.to_numeric(raw_df["cmfg_star_bins"], errors="coerce").dropna().unique()
    return len(cached_bins) == 1 and int(cached_bins[0]) == int(bins)


def load_cached_data(outdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = pd.read_csv(outdir / CACHE_RUN_METRICS)
    X_df = pd.read_csv(outdir / CACHE_DATASET_VECTORS, index_col=0)
    point_meta = pd.read_csv(outdir / CACHE_POINT_META)
    return raw_df, X_df, point_meta


def save_cached_data(
    outdir: Path,
    raw_df: pd.DataFrame,
    X_df: pd.DataFrame,
    point_meta: pd.DataFrame,
    bins: int,
) -> None:
    raw_out = raw_df.copy()
    raw_out["cmfg_star_bins"] = int(bins)

    raw_out.to_csv(outdir / CACHE_RUN_METRICS, index=False)
    X_df.to_csv(outdir / CACHE_DATASET_VECTORS)
    point_meta.to_csv(outdir / CACHE_POINT_META, index=False)


# ============================================================
# Build dataset-point vectors
# ============================================================

def build_dataset_point_matrix(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    rows   = point_id = model__dataset
    cols   = condition__method
    values = cMFG* values
    """
    preferred_condition_order = ["baseline", "perception", "msh_perception"]

    observed_conditions = df["condition"].dropna().drop_duplicates().tolist()

    condition_order = [
        c for c in preferred_condition_order if c in observed_conditions
    ] + [
        c for c in observed_conditions if c not in preferred_condition_order
    ]

    long_records = []

    point_meta = (
        df[["model", "family", "dataset"]]
        .drop_duplicates()
        .copy()
    )
    point_meta["point_id"] = point_meta["model"] + "__" + point_meta["dataset"]

    for _, row in df.iterrows():
        point_id = f"{row['model']}__{row['dataset']}"

        for metric_col, method_name in CMFG_METRICS.items():
            long_records.append({
                "point_id": point_id,
                "model": row["model"],
                "family": row["family"],
                "dataset": row["dataset"],
                "condition": row["condition"],
                "method": method_name,
                "feature": f"{row['condition']}__{method_name}",
                "value": row[metric_col],
            })

    long_df = pd.DataFrame(long_records)
    X_df = long_df.pivot(index="point_id", columns="feature", values="value")

    feature_order = []

    for condition in condition_order:
        for method_name in ["RCC", "DeepConf", "Sampling"]:
            feat = f"{condition}__{method_name}"
            if feat in X_df.columns:
                feature_order.append(feat)

    X_df = X_df.reindex(columns=feature_order).sort_index()
    X_df = X_df.dropna(axis=1, how="all")

    point_meta_df = (
        point_meta
        .set_index("point_id")
        .loc[X_df.index]
        .reset_index()
    )

    return X_df, point_meta_df


# ============================================================
# Analysis
# ============================================================

def prepare_matrix(X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if X_df.empty:
        raise ValueError("Empty feature matrix after parsing.")

    imputer = SimpleImputer(strategy="mean")
    X_raw = imputer.fit_transform(X_df)

    if STANDARDIZE_FEATURES:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X_raw)
    else:
        X_proc = X_raw.copy()

    return X_raw, X_proc


def choose_best_k(X_proc: np.ndarray) -> Tuple[int, Dict[int, float]]:
    n_points = X_proc.shape[0]

    if n_points <= 2:
        return 1, {}

    upper = min(MAX_AUTO_K, n_points - 1)

    if upper < 2:
        return 1, {}

    scores: Dict[int, float] = {}

    for k in range(2, upper + 1):
        km = KMeans(
            n_clusters=k,
            n_init=100,
            random_state=RANDOM_STATE,
        )
        labels = km.fit_predict(X_proc)

        if len(np.unique(labels)) < 2:
            continue

        try:
            scores[k] = silhouette_score(X_proc, labels)
        except Exception:
            pass

    if not scores:
        return 2, {}

    best_k = max(scores, key=scores.get)
    return best_k, scores


def fit_kmeans(X_proc: np.ndarray) -> Tuple[np.ndarray, int, Dict[int, float]]:
    n_points = X_proc.shape[0]

    if n_points <= 1:
        return np.zeros(n_points, dtype=int), 1, {}

    k_used, sil_scores = choose_best_k(X_proc)

    if k_used <= 1:
        return np.zeros(n_points, dtype=int), 1, sil_scores

    km = KMeans(
        n_clusters=k_used,
        n_init=100,
        random_state=RANDOM_STATE,
    )
    labels = km.fit_predict(X_proc)

    return labels, k_used, sil_scores


def run_pca(X_proc: np.ndarray) -> Tuple[np.ndarray, PCA]:
    n_components = min(2, X_proc.shape[0], X_proc.shape[1])

    if n_components < 1:
        raise ValueError("Cannot run PCA on empty matrix.")

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_proc)

    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])

    return coords, pca


# ============================================================
# Plotting
# ============================================================

def build_family_color_map(families: List[str]) -> Dict[str, str]:
    palette = [
        "#f26b6b",
        "#59c3c3",
        "#8a7dff",
        "#f4b942",
        "#4caf50",
        "#ff7eb6",
        "#7f8c8d",
        "#1abc9c",
        "#e67e22",
        "#3498db",
    ]
    uniq = list(dict.fromkeys(families))
    return {fam: palette[i % len(palette)] for i, fam in enumerate(uniq)}


def build_dataset_marker_map(datasets: List[str]) -> Dict[str, str]:
    marker_cycle = ["o", "^", "s", "D", "P", "X", "v", "<", ">"]
    uniq = list(dict.fromkeys(datasets))
    return {ds: marker_cycle[i % len(marker_cycle)] for i, ds in enumerate(uniq)}


def build_cluster_edge_map(clusters: List[int]) -> Dict[int, str]:
    palette = [
        "#ff2d96",
        "#00a8ff",
        "#222222",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#e74c3c",
        "#16a085",
        "#d35400",
    ]

    cluster_series = pd.Series(clusters).dropna().astype(int)
    uniq = sorted(cluster_series.unique().tolist())

    return {cid: palette[i % len(palette)] for i, cid in enumerate(uniq)}

def save_dataset_point_pca_plot(
    plot_df: pd.DataFrame,
    explained_ratio: np.ndarray,
    outpath_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    family_colors = build_family_color_map(plot_df["family"].tolist())
    dataset_markers = build_dataset_marker_map(plot_df["dataset"].tolist())
    cluster_edges = build_cluster_edge_map(plot_df["cluster"].tolist())

    for _, row in plot_df.iterrows():
        ax.scatter(
            row["PC1"],
            row["PC2"],
            s=POINT_SIZE,
            marker=dataset_markers[row["dataset"]],
            facecolor=family_colors[row["family"]],
            edgecolor=cluster_edges[int(row["cluster"])],
            linewidth=2.2,
            alpha=0.95,
            zorder=3,
        )

        if LABEL_POINTS:
            label = f"{row['model']} | {row['dataset']}"
            ax.annotate(
                label,
                (row["PC1"], row["PC2"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

    pc1_var = explained_ratio[0] * 100 if len(explained_ratio) >= 1 else 0.0
    pc2_var = explained_ratio[1] * 100 if len(explained_ratio) >= 2 else 0.0

    ax.set_title("PCA of Dataset-Level cMFG* Vectors", fontsize=15)
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}% Variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}% Variance)", fontsize=12)
    ax.grid(True, alpha=0.8, linestyle="-")
    ax.set_axisbelow(True)

    family_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=family_colors[fam],
            markeredgecolor="black",
            markersize=9,
            linewidth=0,
            label=fam,
        )
        for fam in family_colors
    ]

    dataset_handles = [
        Line2D(
            [0], [0],
            marker=dataset_markers[ds],
            color="gray",
            markerfacecolor="gray",
            markeredgecolor="gray",
            markersize=9,
            linewidth=0,
            label=ds,
        )
        for ds in dataset_markers
    ]

    cluster_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=cluster_edges[cid],
            markeredgewidth=2.2,
            markersize=9,
            linewidth=0,
            label=f"Cluster {cid}",
        )
        for cid in sorted(cluster_edges)
    ]

    leg1 = ax.legend(
        handles=family_handles,
        title="Family",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=dataset_handles,
        title="Dataset",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.63),
        frameon=True,
    )
    ax.add_artist(leg2)

    ax.legend(
        handles=cluster_handles,
        title="KMeans Cluster",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.27),
        frameon=True,
    )

    fig.tight_layout()

    outpath_pdf = outpath_png.with_suffix(".pdf")
    fig.savefig(outpath_png, dpi=220, bbox_inches="tight")
    fig.savefig(outpath_pdf, bbox_inches="tight")

    plt.close(fig)


def save_similarity_heatmap(
    sim_df: pd.DataFrame,
    outpath_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))

    sns.heatmap(
        sim_df,
        ax=ax,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        square=True,
        annot_kws={"fontsize": 7},
    )

    ax.set_title("Cosine Similarity Between Dataset-Level cMFG* Points")
    ax.set_xlabel("Dataset-Level Point")
    ax.set_ylabel("Dataset-Level Point")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()

    outpath_pdf = outpath_png.with_suffix(".pdf")
    fig.savefig(outpath_png, dpi=220, bbox_inches="tight")
    fig.savefig(outpath_pdf, bbox_inches="tight")

    plt.close(fig)


def save_cluster_metric_heatmap(
    X_df: pd.DataFrame,
    plot_df: pd.DataFrame,
    outpath_png: Path,
) -> None:
    order = (
        plot_df
        .sort_values(["cluster", "model", "dataset"])
        ["point_id"]
        .tolist()
    )

    X_ordered = X_df.loc[order].copy()

    display_index = []
    meta = plot_df.set_index("point_id")

    for pid in X_ordered.index:
        row = meta.loc[pid]
        display_index.append(f"C{int(row['cluster'])} | {row['model']} | {row['dataset']}")

    X_ordered.index = display_index

    fig_height = max(5, 0.45 * len(X_ordered))
    fig_width = max(9, 0.65 * X_ordered.shape[1])

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        X_ordered,
        ax=ax,
        cmap="mako",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        annot_kws={"fontsize": 8},
    )

    ax.set_title("Dataset-Level cMFG* Feature Vectors Ordered by Cluster")
    ax.set_xlabel("Prompt Condition × Confidence Estimator")
    ax.set_ylabel("Dataset-Level Point")

    fig.tight_layout()

    outpath_pdf = outpath_png.with_suffix(".pdf")
    fig.savefig(outpath_png, dpi=220, bbox_inches="tight")
    fig.savefig(outpath_pdf, bbox_inches="tight")

    plt.close(fig)


# ============================================================
# Summary
# ============================================================

def write_summary(
    outpath: Path,
    X_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    plot_df: pd.DataFrame,
    pca: PCA,
    loadings_df: pd.DataFrame,
    k_used: int,
    silhouette_scores: Dict[int, float],
    used_cache: bool,
) -> None:
    lines: List[str] = []

    lines.append("DATASET-LEVEL cMFG* CLUSTERING SUMMARY")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Input:")
    lines.append("Parsed current real_results folder structure." if not used_cache else "Loaded cached cMFG* vectors.")
    lines.append("")
    lines.append("Representation:")
    lines.append("Each point is one (model, dataset) pair.")
    lines.append("Features are cMFG* values across prompt condition × confidence estimator.")
    lines.append("")
    lines.append(f"Number of parsed runs: {raw_df.shape[0]}")
    lines.append(f"Number of dataset-level points: {X_df.shape[0]}")
    lines.append(f"Number of features per point: {X_df.shape[1]}")
    lines.append(f"cMFG* equal-mass bins: {CMFG_STAR_BINS}")
    lines.append("")

    lines.append("Feature columns:")
    for col in X_df.columns:
        lines.append(f"  - {col}")
    lines.append("")

    if {"model", "dataset", "condition"}.issubset(raw_df.columns):
        lines.append("Parsed run coverage:")
        coverage = (
            raw_df
            .groupby(["model", "dataset"])["condition"]
            .apply(lambda x: ", ".join(sorted(pd.unique(x))))
            .reset_index()
        )

        for _, row in coverage.iterrows():
            lines.append(f"  - {row['model']} | {row['dataset']}: {row['condition']}")
        lines.append("")

    lines.append("Cluster assignments:")
    for _, row in plot_df.sort_values(["cluster", "model", "dataset"]).iterrows():
        lines.append(f"  - {row['model']} | {row['dataset']}: Cluster {int(row['cluster'])}")
    lines.append("")

    if silhouette_scores:
        lines.append("Silhouette scores by k:")
        for k, val in sorted(silhouette_scores.items()):
            lines.append(f"  - k={k}: {val:.4f}")
        lines.append("")

    lines.append(f"K used: {k_used}")
    lines.append("")

    evr = pca.explained_variance_ratio_

    if len(evr) >= 1:
        lines.append(f"PC1 explained variance: {100 * evr[0]:.2f}%")
    if len(evr) >= 2:
        lines.append(f"PC2 explained variance: {100 * evr[1]:.2f}%")
    lines.append("")

    lines.append("Cluster counts by dataset:")
    lines.append(str(pd.crosstab(plot_df["cluster"], plot_df["dataset"])))
    lines.append("")

    lines.append("Cluster counts by family:")
    lines.append(str(pd.crosstab(plot_df["cluster"], plot_df["family"])))
    lines.append("")

    lines.append("Cluster counts by model:")
    lines.append(str(pd.crosstab(plot_df["cluster"], plot_df["model"])))
    lines.append("")

    if "PC1" in loadings_df.columns:
        s = loadings_df["PC1"].copy()
        s = s.reindex(s.abs().sort_values(ascending=False).index).head(12)

        lines.append("Top absolute PC1 loadings:")
        for feat, val in s.items():
            lines.append(f"  - {feat}: {val:.4f}")
        lines.append("")

    if "PC2" in loadings_df.columns:
        s = loadings_df["PC2"].copy()
        s = s.reindex(s.abs().sort_values(ascending=False).index).head(12)

        lines.append("Top absolute PC2 loadings:")
        for feat, val in s.items():
            lines.append(f"  - {feat}: {val:.4f}")
        lines.append("")

    lines.append("Notes:")
    lines.append("- cMFG* is recomputed directly from example-level confidence and faithfulness columns unless cached vectors are reused.")
    lines.append("- Missing values are mean-imputed before PCA/KMeans.")
    lines.append("- PCA and KMeans use standardized features by default.")
    lines.append("- This clustering is descriptive, not inferential.")

    outpath.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default=str(REAL_RESULTS_ROOT),
        help="Path to real_results directory.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(_default_output_dir("clustering_output_dataset_points")),
        help="Output directory.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=CMFG_STAR_BINS,
        help="Number of equal-mass bins used to compute cMFG*.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore cached CSV files and recompute cMFG* from Excel files.",
    )
    parser.add_argument(
        "--skip-extra-heatmaps",
        action="store_true",
        help="Only save PCA plot and summary. Skips cosine and feature heatmaps.",
    )

    args, _ = parser.parse_known_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    bins = int(args.bins)

    outdir.mkdir(parents=True, exist_ok=True)

    used_cache = False

    if not args.recompute and cache_is_usable(outdir, bins):
        print(f"[CACHE] Loading cached CSVs from {outdir.resolve()}")
        raw_df, X_df, point_meta = load_cached_data(outdir)
        used_cache = True
    else:
        print("[PARSE] Computing cMFG* from real_results Excel files.")
        raw_df = parse_real_results(root=root, n_bins=bins)
        X_df, point_meta = build_dataset_point_matrix(raw_df)
        save_cached_data(outdir, raw_df, X_df, point_meta, bins=bins)

    X_raw, X_proc = prepare_matrix(X_df)

    labels_zero, k_used, silhouette_scores = fit_kmeans(X_proc)
    labels = labels_zero + 1

    coords, pca = run_pca(X_proc)

    plot_df = point_meta.copy()
    plot_df["cluster"] = labels
    plot_df["PC1"] = coords[:, 0]
    plot_df["PC2"] = coords[:, 1]

    cosine_df = pd.DataFrame(
        cosine_similarity(X_raw),
        index=X_df.index,
        columns=X_df.index,
    )

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=X_df.columns,
        columns=[f"PC{i + 1}" for i in range(pca.components_.shape[0])],
    )

    pca_png = outdir / "dataset_points_pca.png"
    cosine_png = outdir / "cosine_similarity_heatmap.png"
    cluster_heatmap_png = outdir / "clustered_cmfg_star_feature_heatmap.png"
    summary_path = outdir / "summary.txt"

    save_dataset_point_pca_plot(
        plot_df=plot_df,
        explained_ratio=pca.explained_variance_ratio_,
        outpath_png=pca_png,
    )

    if not args.skip_extra_heatmaps:
        save_similarity_heatmap(
            sim_df=cosine_df,
            outpath_png=cosine_png,
        )

        save_cluster_metric_heatmap(
            X_df=X_df,
            plot_df=plot_df,
            outpath_png=cluster_heatmap_png,
        )

    write_summary(
        outpath=summary_path,
        X_df=X_df,
        raw_df=raw_df,
        plot_df=plot_df,
        pca=pca,
        loadings_df=loadings_df,
        k_used=k_used,
        silhouette_scores=silhouette_scores,
        used_cache=used_cache,
    )

    plot_df.to_csv(outdir / "dataset_points_pca_coordinates.csv", index=False)
    loadings_df.to_csv(outdir / "pca_loadings.csv")
    cosine_df.to_csv(outdir / "cosine_similarity_matrix.csv")

    print("")
    print("Done.")
    print(f"Input root: {root.resolve()}")
    print(f"Output:     {outdir.resolve()}")
    print(f"Used cache: {used_cache}")
    print("")
    print("Saved files:")
    print(f"  {outdir / 'dataset_points_pca.png'}")
    print(f"  {outdir / 'dataset_points_pca.pdf'}")
    if not args.skip_extra_heatmaps:
        print(f"  {outdir / 'cosine_similarity_heatmap.png'}")
        print(f"  {outdir / 'cosine_similarity_heatmap.pdf'}")
        print(f"  {outdir / 'clustered_cmfg_star_feature_heatmap.png'}")
        print(f"  {outdir / 'clustered_cmfg_star_feature_heatmap.pdf'}")
    print(f"  {outdir / CACHE_RUN_METRICS}")
    print(f"  {outdir / CACHE_DATASET_VECTORS}")
    print(f"  {outdir / CACHE_POINT_META}")
    print(f"  {outdir / 'dataset_points_pca_coordinates.csv'}")
    print(f"  {outdir / 'pca_loadings.csv'}")
    print(f"  {outdir / 'cosine_similarity_matrix.csv'}")
    print(f"  {outdir / 'summary.txt'}")


if __name__ == "__main__":
    main()