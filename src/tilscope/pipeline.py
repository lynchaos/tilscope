"""End-to-end single-cell analysis pipeline.

Runs entirely on the core scientific stack (numpy / scipy / scikit-learn) so
it works anywhere. If scanpy + leidenalg are installed it transparently uses
Leiden community detection; otherwise it falls back to silhouette-selected
K-means. Either way the output is the same :class:`Results` object, which the
report and narrative layers consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score

from .data import CellData
from .qc import compute_qc, QCResult
from . import annotate


@dataclass
class Results:
    qc: QCResult
    obs: pd.DataFrame                      # per-cell: cluster, cell_type, exhaustion, embedding
    cluster_labels: dict[int, str]
    composition: pd.Series                 # cell-type proportions
    exhaustion: dict[str, float]           # headline exhaustion stats
    marker_matrix: pd.DataFrame            # mean expression: cell_type x marker
    params: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def _normalise_log(X: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """CP10K normalisation followed by log1p."""
    totals = np.maximum(X.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(X / totals * target_sum)


def _select_hvg(lognorm: np.ndarray, n_top: int) -> np.ndarray:
    var = lognorm.var(axis=0)
    return np.argsort(var)[::-1][:n_top]


def _cluster(pcs: np.ndarray, seed: int) -> tuple[np.ndarray, str, float]:
    """Leiden if available, else silhouette-tuned K-means."""
    try:  # pragma: no cover - optional acceleration path
        import scanpy as sc
        import anndata as ad

        a = ad.AnnData(pcs.astype("float32"))
        sc.pp.neighbors(a, use_rep="X", random_state=seed)
        sc.tl.leiden(a, random_state=seed, flavor="igraph", n_iterations=2, directed=False)
        labels = a.obs["leiden"].astype(int).to_numpy()
        return labels, "leiden", float("nan")
    except Exception:
        best_k, best_labels, best_sil = None, None, -1.0
        for k in range(5, 9):
            km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(pcs)
            sil = silhouette_score(pcs, km.labels_)
            if sil > best_sil:
                best_k, best_labels, best_sil = k, km.labels_, sil
        return best_labels, f"kmeans(k={best_k})", float(best_sil)


def run_pipeline(
    cd: CellData,
    seed: int = 0,
    n_hvg: int = 80,
    n_pcs: int = 30,
    min_genes: int | None = None,
    max_pct_mito: float | None = None,
) -> Results:
    rng = np.random.default_rng(seed)

    # 1. Quality control --------------------------------------------------
    qc = compute_qc(cd, min_genes=min_genes, max_pct_mito=max_pct_mito)
    keep = qc.metrics["pass_qc"].to_numpy()
    X = cd.X[keep]
    obs = cd.obs.loc[keep].copy()

    # 2. Normalise + log --------------------------------------------------
    lognorm = _normalise_log(X)
    logdf = pd.DataFrame(lognorm, columns=cd.var_names, index=obs.index)

    # 3. Feature selection + scaling (on HVGs) ----------------------------
    hvg = _select_hvg(lognorm, min(n_hvg, lognorm.shape[1]))
    scaled = lognorm[:, hvg]
    scaled = (scaled - scaled.mean(axis=0)) / (scaled.std(axis=0) + 1e-9)
    scaled = np.clip(scaled, -10, 10)

    # 4. PCA --------------------------------------------------------------
    n_comp = min(n_pcs, scaled.shape[1] - 1, scaled.shape[0] - 1)
    pcs = PCA(n_components=n_comp, random_state=seed).fit_transform(scaled)

    # 5. Clustering -------------------------------------------------------
    clusters, method, silhouette = _cluster(pcs, seed)

    # 6. 2D embedding for visualisation -----------------------------------
    perplexity = float(min(30, max(5, (pcs.shape[0] - 1) // 3)))
    emb = TSNE(n_components=2, init="pca", perplexity=perplexity,
               random_state=seed).fit_transform(pcs)

    # 7. Annotation (z-scored full matrix, never sees ground truth) -------
    logz = (logdf - logdf.mean(axis=0)) / (logdf.std(axis=0) + 1e-9)
    sig_scores = annotate.score_signatures(logz)
    cluster_labels = annotate.annotate_clusters(clusters, sig_scores)
    cell_types = np.array([cluster_labels[int(c)] for c in clusters])
    exh = annotate.exhaustion_score(logz)

    obs = obs.assign(
        cluster=clusters,
        cell_type=cell_types,
        exhaustion=exh.to_numpy(),
        emb_x=emb[:, 0],
        emb_y=emb[:, 1],
    )

    # 8. Headline biology -------------------------------------------------
    composition = obs["cell_type"].value_counts(normalize=True).sort_values(ascending=False)
    cd8 = obs[obs["cell_type"].str.startswith("CD8")]
    exh_thr = float(np.quantile(exh, 0.66))
    cd8_exh_frac = float((cd8["exhaustion"] > exh_thr).mean()) if len(cd8) else 0.0
    exhaustion = {
        "cd8_high_exhaustion_fraction": round(cd8_exh_frac, 3),
        "exhaustion_threshold_z": round(exh_thr, 3),
        "mean_cd8_exhaustion": round(float(cd8["exhaustion"].mean()), 3) if len(cd8) else 0.0,
    }

    # Mean expression of each programme's markers, per recovered cell type.
    markers = sorted({g for genes in annotate.SIGNATURES.values() for g in genes
                      if g in logdf.columns})
    marker_matrix = (
        logdf[markers].assign(cell_type=cell_types)
        .groupby("cell_type").mean()
    )

    # 9. Diagnostics (only possible because this is simulated truth) ------
    diagnostics = {"clustering_method": method, "silhouette": round(silhouette, 3)}
    if "true_label" in obs.columns:
        diagnostics["adjusted_rand_index"] = round(
            float(adjusted_rand_score(obs["true_label"], obs["cluster"])), 3
        )

    params = {
        "seed": seed, "n_hvg": int(min(n_hvg, lognorm.shape[1])),
        "n_pcs": int(n_comp), "target_sum": 1e4,
    }
    return Results(
        qc=qc, obs=obs, cluster_labels=cluster_labels, composition=composition,
        exhaustion=exhaustion, marker_matrix=marker_matrix,
        params=params, diagnostics=diagnostics,
    )
