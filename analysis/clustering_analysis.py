
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
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

HTML_PATH = _resolve_repo_root() / "results_dashboard.html"
OUTDIR = _default_output_dir("clustering_output")

RANDOM_STATE = 0
MAX_AUTO_K = 6
STANDARDIZE_FEATURES = True

MODEL_FIGSIZE = (10, 7)
SETTING_FIGSIZE = (11, 8)
POINT_SIZE_MODEL = 240
POINT_SIZE_SETTING = 120

ANNOTATE_MODELS = True
ANNOTATE_SETTINGS_IF_N_LEQ = 30  # usually settings are many, so labels are skipped


# ============================================================
# Representations to compare
# ============================================================

REPRESENTATIONS = {
    "cmfg_star_only": {
        "metrics": ["cMFG*_RCC", "cMFG*_DC", "cMFG*_Sampling"],
        "aliases": {
            "cMFG*_RCC": "RCC",
            "cMFG*_DC": "DC",
            "cMFG*_Sampling": "Sampling",
        },
        "description": "Only cMFG* metrics",
    },
    "confidence_only": {
        "metrics": ["RCC", "DC", "Samp", "Dec"],
        "aliases": {
            "RCC": "RCC",
            "DC": "DC",
            "Samp": "Sampling",
            "Dec": "Decision",
        },
        "description": "Only raw confidence / average confidence metrics",
    },
    "faithfulness_only": {
        "metrics": ["F_RCC", "F_DC", "F_Sampling"],
        "aliases": {
            "F_RCC": "RCC",
            "F_DC": "DC",
            "F_Sampling": "Sampling",
        },
        "description": "Only faithfulness metrics F(*)",
    },
    "everything": {
        "metrics": [
            "Acc",
            "RCC", "DC", "Samp", "Dec",
            "F_RCC", "F_DC", "F_Sampling",
            "cMFG*_RCC", "cMFG*_DC", "cMFG*_Sampling",
        ],
        "aliases": {
            "Acc": "Acc",
            "RCC": "Conf_RCC",
            "DC": "Conf_DC",
            "Samp": "Conf_Sampling",
            "Dec": "Conf_Decision",
            "F_RCC": "Faith_RCC",
            "F_DC": "Faith_DC",
            "F_Sampling": "Faith_Sampling",
            "cMFG*_RCC": "cMFG_RCC",
            "cMFG*_DC": "cMFG_DC",
            "cMFG*_Sampling": "cMFG_Sampling",
        },
        "description": "All scalar metrics from the dashboard",
    },
    "cmfg_leave_out_RCC": {
        "metrics": ["cMFG*_DC", "cMFG*_Sampling"],
        "aliases": {
            "cMFG*_DC": "DC",
            "cMFG*_Sampling": "Sampling",
        },
        "description": "cMFG* with RCC left out",
    },
    "cmfg_leave_out_DC": {
        "metrics": ["cMFG*_RCC", "cMFG*_Sampling"],
        "aliases": {
            "cMFG*_RCC": "RCC",
            "cMFG*_Sampling": "Sampling",
        },
        "description": "cMFG* with DC left out",
    },
    "cmfg_leave_out_Sampling": {
        "metrics": ["cMFG*_RCC", "cMFG*_DC"],
        "aliases": {
            "cMFG*_RCC": "RCC",
            "cMFG*_DC": "DC",
        },
        "description": "cMFG* with Sampling left out",
    },
}


# ============================================================
# Helpers
# ============================================================

def clean_text(text: str) -> str:
    return " ".join(str(text).replace("\xa0", " ").split()).strip()


def parse_float(text: str) -> float:
    s = clean_text(text).replace(",", "")
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def normalize_condition(condition: str) -> str:
    s = clean_text(condition).lower()
    s = s.replace("sys:", "")
    s = s.replace(" ", "")
    return s


def normalize_model_name(model: str) -> str:
    return clean_text(model)


def extract_family(model_name: str) -> str:
    return clean_text(model_name).split("-")[0]


def safe_filename(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    return s


# ============================================================
# HTML parsing
# ============================================================

def parse_dashboard_html(html_path: Path) -> pd.DataFrame:
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    sections = soup.select("div.section")
    if not sections:
        raise ValueError("No <div class='section'> blocks found in results_dashboard.html")

    records: List[Dict] = []

    for section in sections:
        title_el = section.select_one(".section-title")
        table_el = section.find("table")
        if title_el is None or table_el is None:
            continue

        model_name = normalize_model_name(title_el.get_text(" ", strip=True))
        family = extract_family(model_name)

        tbody = table_el.find("tbody")
        if tbody is None:
            continue

        current_dataset = None

        for tr in tbody.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue

            # separator rows
            if len(tds) == 1 and tds[0].has_attr("colspan"):
                continue

            cells = [clean_text(td.get_text(" ", strip=True)) for td in tds]
            if len(cells) < 13:
                continue

            dataset = cells[0] if cells[0] else current_dataset
            condition_raw = cells[1]
            condition = normalize_condition(condition_raw)

            if not dataset or not condition:
                continue

            current_dataset = dataset

            records.append({
                "model": model_name,
                "family": family,
                "dataset": dataset,
                "condition_raw": condition_raw,
                "condition": condition,

                "Acc": parse_float(cells[2]),
                "RCC": parse_float(cells[3]),
                "DC": parse_float(cells[4]),
                "Samp": parse_float(cells[5]),
                "Dec": parse_float(cells[6]),
                "F_RCC": parse_float(cells[7]),
                "F_DC": parse_float(cells[8]),
                "F_Sampling": parse_float(cells[9]),
                "cMFG*_RCC": parse_float(cells[10]),
                "cMFG*_DC": parse_float(cells[11]),
                "cMFG*_Sampling": parse_float(cells[12]),
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Parsed zero rows from dashboard HTML.")

    return df


# ============================================================
# Feature matrix builders
# ============================================================

def build_model_feature_matrix(
    df: pd.DataFrame,
    metrics: List[str],
    aliases: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    rows   = models
    cols   = dataset__condition__alias(metric)
    values = metric values
    """
    dataset_order = df["dataset"].drop_duplicates().tolist()
    condition_order = df["condition"].drop_duplicates().tolist()

    records = []
    for _, row in df.iterrows():
        for metric in metrics:
            alias = aliases[metric]
            feature = f"{row['dataset']}__{row['condition']}__{alias}"
            records.append({
                "model": row["model"],
                "dataset": row["dataset"],
                "condition": row["condition"],
                "metric": metric,
                "metric_alias": alias,
                "feature": feature,
                "value": row[metric],
            })

    long_df = pd.DataFrame(records)
    X_df = long_df.pivot(index="model", columns="feature", values="value")

    feature_order = []
    feature_meta_rows = []

    # preserve a stable, readable feature order
    metric_alias_order = [aliases[m] for m in metrics]
    for dataset in dataset_order:
        for condition in condition_order:
            for alias in metric_alias_order:
                feat = f"{dataset}__{condition}__{alias}"
                if feat in X_df.columns:
                    feature_order.append(feat)
                    feature_meta_rows.append({
                        "feature": feat,
                        "dataset": dataset,
                        "condition": condition,
                        "metric_alias": alias,
                    })

    X_df = X_df.reindex(columns=feature_order).sort_index()
    feature_meta = pd.DataFrame(feature_meta_rows)
    return X_df, feature_meta


def build_cmfg_setting_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    rows   = settings = dataset__condition__method
    cols   = models
    values = cMFG* values

    This is useful when there are only a few models.
    """
    cmfg_map = {
        "cMFG*_RCC": "RCC",
        "cMFG*_DC": "DC",
        "cMFG*_Sampling": "Sampling",
    }

    records = []
    for _, row in df.iterrows():
        for metric_col, method in cmfg_map.items():
            setting = f"{row['dataset']}__{row['condition']}__{method}"
            records.append({
                "setting": setting,
                "dataset": row["dataset"],
                "condition": row["condition"],
                "method": method,
                "model": row["model"],
                "value": row[metric_col],
            })

    long_df = pd.DataFrame(records)
    X_df = long_df.pivot(index="setting", columns="model", values="value")

    setting_meta = long_df[["setting", "dataset", "condition", "method"]].drop_duplicates()
    setting_meta = setting_meta.set_index("setting").loc[X_df.index].reset_index()

    return X_df, setting_meta


# ============================================================
# Analysis utilities
# ============================================================

def prepare_matrix(X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
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
        km = KMeans(n_clusters=k, n_init=100, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_proc)

        if len(np.unique(labels)) < 2:
            continue

        # silhouette requires at least 2 clusters and not all singleton issues
        try:
            score = silhouette_score(X_proc, labels)
            scores[k] = score
        except Exception:
            pass

    if not scores:
        return 2 if n_points >= 3 else 1, {}

    best_k = max(scores, key=scores.get)
    return best_k, scores


def fit_kmeans(X_proc: np.ndarray) -> Tuple[np.ndarray, int, Dict[int, float]]:
    n_points = X_proc.shape[0]
    if n_points <= 1:
        return np.zeros(n_points, dtype=int), 1, {}

    k, sil_scores = choose_best_k(X_proc)
    if k <= 1:
        return np.zeros(n_points, dtype=int), 1, sil_scores

    km = KMeans(n_clusters=k, n_init=100, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_proc)
    return labels, k, sil_scores


def run_pca(X_proc: np.ndarray) -> Tuple[np.ndarray, PCA]:
    n_components = min(2, X_proc.shape[0], X_proc.shape[1])
    if n_components < 1:
        raise ValueError("Cannot run PCA on empty matrix.")

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_proc)

    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])

    return coords, pca


def analyze_matrix(X_df: pd.DataFrame) -> Dict:
    X_raw, X_proc = prepare_matrix(X_df)
    labels_zero, k_used, sil_scores = fit_kmeans(X_proc)
    coords, pca = run_pca(X_proc)

    labels = labels_zero + 1
    cosine_df = pd.DataFrame(
        cosine_similarity(X_raw),
        index=X_df.index,
        columns=X_df.index,
    )

    return {
        "X_raw": X_raw,
        "X_proc": X_proc,
        "labels": labels,
        "k_used": k_used,
        "silhouette_scores": sil_scores,
        "coords": coords,
        "pca": pca,
        "cosine_df": cosine_df,
    }


# ============================================================
# Plot helpers
# ============================================================

def color_map_from_categories(categories: List[str], palette: List[str]) -> Dict[str, str]:
    uniq = list(dict.fromkeys(categories))
    return {cat: palette[i % len(palette)] for i, cat in enumerate(uniq)}


def cluster_edge_map(cluster_ids: List[int]) -> Dict[int, str]:
    palette = [
        "#ff2d96",
        "#00a8ff",
        "#222222",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#e74c3c",
    ]
    uniq = sorted(pd.unique(cluster_ids))
    return {cid: palette[i % len(palette)] for i, cid in enumerate(uniq)}


def save_model_pca_plot(
    plot_df: pd.DataFrame,
    explained_ratio: np.ndarray,
    title: str,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=MODEL_FIGSIZE)

    family_colors = color_map_from_categories(
        plot_df["family"].tolist(),
        ["#f26b6b", "#59c3c3", "#8a7dff", "#f4b942", "#4caf50", "#ff7eb6", "#7f8c8d"]
    )
    cluster_edges = cluster_edge_map(plot_df["cluster"].tolist())

    for _, row in plot_df.iterrows():
        ax.scatter(
            row["PC1"],
            row["PC2"],
            s=POINT_SIZE_MODEL,
            marker="o",
            facecolor=family_colors[row["family"]],
            edgecolor=cluster_edges[row["cluster"]],
            linewidth=2.5,
            alpha=0.95,
            zorder=3,
        )

        if ANNOTATE_MODELS:
            ax.annotate(
                row["model"],
                (row["PC1"], row["PC2"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=10,
            )

    pc1_var = explained_ratio[0] * 100 if len(explained_ratio) >= 1 else 0.0
    pc2_var = explained_ratio[1] * 100 if len(explained_ratio) >= 2 else 0.0

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}% variance)", fontsize=12)
    ax.grid(True, alpha=0.25, linestyle="--")
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

    cluster_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=cluster_edges[cid],
            markeredgewidth=2.5,
            markersize=9,
            linewidth=0,
            label=f"Cluster {cid}",
        )
        for cid in sorted(cluster_edges)
    ]

    leg1 = ax.legend(
        handles=family_handles,
        title="Family (fill color)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=cluster_handles,
        title="KMeans cluster (edge color)",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_setting_pca_plot(
    plot_df: pd.DataFrame,
    explained_ratio: np.ndarray,
    title: str,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=SETTING_FIGSIZE)

    dataset_colors = color_map_from_categories(
        plot_df["dataset"].tolist(),
        ["#f26b6b", "#59c3c3", "#8a7dff", "#f4b942", "#4caf50", "#ff7eb6"]
    )
    method_markers = {"RCC": "o", "DC": "^", "Sampling": "s"}
    cluster_edges = cluster_edge_map(plot_df["cluster"].tolist())

    for _, row in plot_df.iterrows():
        ax.scatter(
            row["PC1"],
            row["PC2"],
            s=POINT_SIZE_SETTING,
            marker=method_markers.get(row["method"], "o"),
            facecolor=dataset_colors[row["dataset"]],
            edgecolor=cluster_edges[row["cluster"]],
            linewidth=1.8,
            alpha=0.95,
            zorder=3,
        )

    if len(plot_df) <= ANNOTATE_SETTINGS_IF_N_LEQ:
        for _, row in plot_df.iterrows():
            label = f"{row['condition']}"
            ax.annotate(
                label,
                (row["PC1"], row["PC2"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    pc1_var = explained_ratio[0] * 100 if len(explained_ratio) >= 1 else 0.0
    pc2_var = explained_ratio[1] * 100 if len(explained_ratio) >= 2 else 0.0

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}% variance)", fontsize=12)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    dataset_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=dataset_colors[d],
            markeredgecolor="black",
            markersize=8,
            linewidth=0,
            label=d,
        )
        for d in dataset_colors
    ]

    method_handles = [
        Line2D(
            [0], [0],
            marker=method_markers[m],
            color="gray",
            markerfacecolor="gray",
            markeredgecolor="gray",
            markersize=8,
            linewidth=0,
            label=m,
        )
        for m in ["RCC", "DC", "Sampling"]
    ]

    cluster_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=cluster_edges[cid],
            markeredgewidth=2.2,
            markersize=8,
            linewidth=0,
            label=f"Cluster {cid}",
        )
        for cid in sorted(cluster_edges)
    ]

    leg1 = ax.legend(
        handles=dataset_handles,
        title="Dataset (fill color)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=method_handles,
        title="Method (marker)",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.62),
        frameon=True,
    )
    ax.add_artist(leg2)

    ax.legend(
        handles=cluster_handles,
        title="KMeans cluster (edge color)",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.26),
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_similarity_heatmap(sim_df: pd.DataFrame, title: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim_df.values)

    ax.set_xticks(np.arange(sim_df.shape[1]))
    ax.set_yticks(np.arange(sim_df.shape[0]))
    ax.set_xticklabels(sim_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(sim_df.index)

    for i in range(sim_df.shape[0]):
        for j in range(sim_df.shape[1]):
            ax.text(j, i, f"{sim_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Summary writers
# ============================================================

def write_representation_summary(
    summary_rows: List[Dict],
    outpath_csv: Path,
    outpath_txt: Path,
) -> None:
    df = pd.DataFrame(summary_rows)
    df.to_csv(outpath_csv, index=False)

    lines = []
    lines.append("REPRESENTATION COMPARISON SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append("These runs treat each model as one point.")
    lines.append("Given the small number of models, these cluster results are exploratory.")
    lines.append("")

    for _, row in df.iterrows():
        lines.append(f"[{row['representation']}]")
        lines.append(f"  description: {row['description']}")
        lines.append(f"  n_models: {int(row['n_models'])}")
        lines.append(f"  n_features: {int(row['n_features'])}")
        lines.append(f"  k_used: {int(row['k_used'])}")
        lines.append(f"  silhouette_best: {row['best_silhouette']}")
        lines.append(f"  pc1_var: {row['pc1_var_pct']:.2f}%")
        lines.append(f"  pc2_var: {row['pc2_var_pct']:.2f}%")
        lines.append("")

    outpath_txt.write_text("\n".join(lines), encoding="utf-8")


def write_setting_summary(
    plot_df: pd.DataFrame,
    pca: PCA,
    k_used: int,
    silhouette_scores: Dict[int, float],
    outpath_txt: Path,
) -> None:
    lines = []
    lines.append("EXPERIMENTAL-SETTING ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Here each point is one (dataset, prompt condition, method) setting.")
    lines.append("Each point is represented by its cMFG* values across models.")
    lines.append("This is often more informative when there are only a few models.")
    lines.append("")
    lines.append(f"n_settings: {len(plot_df)}")
    lines.append(f"k_used: {k_used}")
    lines.append("")

    if silhouette_scores:
        lines.append("silhouette scores by k:")
        for k, s in sorted(silhouette_scores.items()):
            lines.append(f"  - k={k}: {s:.4f}")
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
    lines.append("Cluster counts by method:")
    lines.append(str(pd.crosstab(plot_df["cluster"], plot_df["method"])))
    lines.append("")
    lines.append("Cluster counts by condition:")
    lines.append(str(pd.crosstab(plot_df["cluster"], plot_df["condition"])))
    lines.append("")

    outpath_txt.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, default=_resolve_repo_root() / "results_dashboard.html")
    parser.add_argument("--outdir", type=Path, default=_default_output_dir("clustering_output"))
    args = parser.parse_args()

    global HTML_PATH, OUTDIR
    HTML_PATH = args.html
    OUTDIR = args.outdir

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------
    # 1) Parse HTML
    # --------------------------------------------
    raw_df = parse_dashboard_html(HTML_PATH)
    raw_df.to_csv(OUTDIR / "parsed_dashboard_long.csv", index=False)

    model_meta = raw_df[["model", "family"]].drop_duplicates().set_index("model")

    # --------------------------------------------
    # 2) Compare model-based representations
    # --------------------------------------------
    rep_summary_rows = []

    for rep_name, rep_cfg in REPRESENTATIONS.items():
        rep_dir = OUTDIR / safe_filename(rep_name)
        rep_dir.mkdir(parents=True, exist_ok=True)

        metrics = rep_cfg["metrics"]
        aliases = rep_cfg["aliases"]
        description = rep_cfg["description"]

        X_df, feature_meta = build_model_feature_matrix(raw_df, metrics, aliases)
        X_df.to_csv(rep_dir / "model_vectors.csv")
        feature_meta.to_csv(rep_dir / "feature_metadata.csv", index=False)

        result = analyze_matrix(X_df)

        plot_df = model_meta.loc[X_df.index].reset_index()
        plot_df["cluster"] = result["labels"]
        plot_df["PC1"] = result["coords"][:, 0]
        plot_df["PC2"] = result["coords"][:, 1]
        plot_df.to_csv(rep_dir / "pca_coordinates.csv", index=False)

        pd.DataFrame({
            "model": X_df.index,
            "cluster": result["labels"],
        }).to_csv(rep_dir / "cluster_assignments.csv", index=False)

        result["cosine_df"].to_csv(rep_dir / "pairwise_cosine_similarity.csv")

        loadings_df = pd.DataFrame(
            result["pca"].components_.T,
            index=X_df.columns,
            columns=[f"PC{i+1}" for i in range(result["pca"].components_.shape[0])]
        )
        loadings_df.to_csv(rep_dir / "pca_loadings.csv")

        save_model_pca_plot(
            plot_df=plot_df,
            explained_ratio=result["pca"].explained_variance_ratio_,
            title=f"Model clustering — {rep_name}",
            outpath=rep_dir / "models_pca.png",
        )

        save_similarity_heatmap(
            sim_df=result["cosine_df"],
            title=f"Model cosine similarity — {rep_name}",
            outpath=rep_dir / "cosine_similarity_heatmap.png",
        )

        evr = result["pca"].explained_variance_ratio_
        rep_summary_rows.append({
            "representation": rep_name,
            "description": description,
            "n_models": X_df.shape[0],
            "n_features": X_df.shape[1],
            "k_used": result["k_used"],
            "best_silhouette": (
                max(result["silhouette_scores"].values())
                if result["silhouette_scores"] else np.nan
            ),
            "pc1_var_pct": 100 * evr[0] if len(evr) >= 1 else np.nan,
            "pc2_var_pct": 100 * evr[1] if len(evr) >= 2 else np.nan,
        })

    write_representation_summary(
        summary_rows=rep_summary_rows,
        outpath_csv=OUTDIR / "representation_comparison.csv",
        outpath_txt=OUTDIR / "representation_comparison_summary.txt",
    )

    # --------------------------------------------
    # 3) Experimental-setting analysis using cMFG* only
    # --------------------------------------------
    setting_dir = OUTDIR / "experimental_setting_analysis"
    setting_dir.mkdir(parents=True, exist_ok=True)

    X_settings_df, setting_meta = build_cmfg_setting_matrix(raw_df)
    X_settings_df.to_csv(setting_dir / "setting_vectors.csv")
    setting_meta.to_csv(setting_dir / "setting_metadata.csv", index=False)

    setting_result = analyze_matrix(X_settings_df)

    setting_plot_df = setting_meta.copy()
    setting_plot_df["cluster"] = setting_result["labels"]
    setting_plot_df["PC1"] = setting_result["coords"][:, 0]
    setting_plot_df["PC2"] = setting_result["coords"][:, 1]
    setting_plot_df.to_csv(setting_dir / "setting_pca_coordinates.csv", index=False)

    pd.DataFrame({
        "setting": X_settings_df.index,
        "cluster": setting_result["labels"],
    }).to_csv(setting_dir / "setting_cluster_assignments.csv", index=False)

    setting_result["cosine_df"].to_csv(setting_dir / "setting_pairwise_cosine_similarity.csv")

    save_setting_pca_plot(
        plot_df=setting_plot_df,
        explained_ratio=setting_result["pca"].explained_variance_ratio_,
        title="Experimental-setting clustering from cMFG* vectors",
        outpath=setting_dir / "settings_pca.png",
    )

    save_similarity_heatmap(
        sim_df=setting_result["cosine_df"],
        title="Setting cosine similarity (cMFG* across models)",
        outpath=setting_dir / "setting_cosine_similarity_heatmap.png",
    )

    # Crosstabs to see whether clusters align with dataset / prompt / method
    pd.crosstab(setting_plot_df["cluster"], setting_plot_df["dataset"]).to_csv(
        setting_dir / "cluster_by_dataset.csv"
    )
    pd.crosstab(setting_plot_df["cluster"], setting_plot_df["condition"]).to_csv(
        setting_dir / "cluster_by_condition.csv"
    )
    pd.crosstab(setting_plot_df["cluster"], setting_plot_df["method"]).to_csv(
        setting_dir / "cluster_by_method.csv"
    )

    write_setting_summary(
        plot_df=setting_plot_df,
        pca=setting_result["pca"],
        k_used=setting_result["k_used"],
        silhouette_scores=setting_result["silhouette_scores"],
        outpath_txt=setting_dir / "setting_summary.txt",
    )
    
    master_lines = []
    master_lines.append("Done.")
    master_lines.append(f"Input HTML: {HTML_PATH.resolve()}")
    master_lines.append(f"Output dir: {OUTDIR.resolve()}")
    master_lines.append("")
    master_lines.append("Main things to check:")
    master_lines.append("1) representation_comparison.csv")
    master_lines.append("2) */models_pca.png for each representation")
    master_lines.append("3) experimental_setting_analysis/settings_pca.png")
    master_lines.append("4) experimental_setting_analysis/cluster_by_{dataset,condition,method}.csv")
    (OUTDIR / "README_results.txt").write_text("\n".join(master_lines), encoding="utf-8")

    print("\n".join(master_lines))


if __name__ == "__main__":
    main()