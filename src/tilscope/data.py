"""Synthetic tumour-microenvironment (TME) single-cell RNA-seq generator.

The point of the synthetic generator is *reproducibility*: the demo runs
identically on any machine, in CI, or in an air-gapped review environment,
with no external data download. Real data is supported via :func:`load_h5ad`.

The simulated tissue contains the cell populations that matter for a T-cell
engager / immuno-oncology programme, with canonical marker genes and a
deliberately planted exhaustion gradient in the CD8+ compartment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# --- Canonical marker programmes -------------------------------------------
# Genes that are biologically elevated in each population. These double as the
# annotation reference in annotate.py, so the pipeline must *recover* them.
CELL_PROGRAMMES: dict[str, list[str]] = {
    "CD8 T (effector)":  ["CD8A", "CD8B", "GZMB", "GZMK", "IFNG", "NKG7", "PRF1"],
    "CD8 T (exhausted)": ["CD8A", "PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1", "CTLA4"],
    "CD4 T (helper)":    ["CD4", "IL7R", "CCR7", "CD40LG"],
    "Treg":              ["CD4", "FOXP3", "IL2RA", "IKZF2", "CTLA4"],
    "NK":                ["NCAM1", "KLRD1", "GNLY", "NKG7", "FCGR3A"],
    "B":                 ["CD19", "MS4A1", "CD79A", "CD79B"],
    "Myeloid":           ["CD68", "LYZ", "CD14", "ITGAM", "C1QA"],
    "Tumour":            ["EPCAM", "KRT8", "KRT18", "CDH1", "MKI67"],
}

# Pan-T-cell genes shared across all T populations.
SHARED_T = ["CD3D", "CD3E", "CD3G"]
# Leukocyte-common + housekeeping.
HOUSEKEEPING = ["PTPRC", "ACTB", "GAPDH", "B2M", "TMSB4X", "MALAT1"]
# Mitochondrial genes -> drive the % mito QC metric.
MITO = ["MT-ND1", "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP6", "MT-CYB", "MT-ND4"]

# Realistic-ish compartment proportions for an infiltrated solid tumour.
DEFAULT_PROPORTIONS = {
    "CD8 T (effector)":  0.18,
    "CD8 T (exhausted)": 0.14,
    "CD4 T (helper)":    0.15,
    "Treg":              0.07,
    "NK":                0.08,
    "B":                 0.08,
    "Myeloid":           0.16,
    "Tumour":            0.14,
}


@dataclass
class CellData:
    """Minimal AnnData-compatible container (cells x genes, dense).

    Kept dependency-free on purpose so the pipeline runs without scanpy.
    ``to_anndata`` upgrades to a real AnnData when the library is present.
    """
    X: np.ndarray                      # (n_cells, n_genes) raw counts
    var_names: list[str]               # gene names
    obs: pd.DataFrame                  # per-cell metadata
    layers: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return self.X.shape[0]

    @property
    def n_vars(self) -> int:
        return self.X.shape[1]

    @property
    def obs_names(self) -> list[str]:
        return list(self.obs.index)

    def to_anndata(self):  # pragma: no cover - exercised only when anndata present
        import anndata as ad
        a = ad.AnnData(X=self.X.astype("float32"))
        a.var_names = self.var_names
        a.obs = self.obs.copy()
        return a


def _build_gene_panel() -> tuple[list[str], dict[str, list[str]]]:
    genes: list[str] = []
    for g in SHARED_T + HOUSEKEEPING + MITO:
        if g not in genes:
            genes.append(g)
    for programme in CELL_PROGRAMMES.values():
        for g in programme:
            if g not in genes:
                genes.append(g)
    # Pad with anonymous background genes so the matrix has realistic width.
    n_filler = max(0, 220 - len(genes))
    genes += [f"BG{ i:04d}" for i in range(n_filler)]
    return genes, CELL_PROGRAMMES


def simulate_tme(
    n_cells: int = 2000,
    seed: int = 0,
    proportions: dict[str, float] | None = None,
    lowq_fraction: float = 0.06,
) -> CellData:
    """Simulate a TME scRNA-seq experiment.

    A negative-binomial-like count model (Poisson on a log-normal rate) gives
    over-dispersed counts. Marker programmes are boosted per population, an
    exhaustion gradient is layered onto the CD8 compartment, and a small
    fraction of low-quality (high-mito, low-complexity) cells is injected so
    that the QC module has something real to flag.
    """
    rng = np.random.default_rng(seed)
    proportions = proportions or DEFAULT_PROPORTIONS
    genes, programmes = _build_gene_panel()
    gene_idx = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)

    # Assign each cell a ground-truth type.
    types = list(proportions)
    probs = np.array([proportions[t] for t in types], dtype=float)
    probs /= probs.sum()
    labels = rng.choice(types, size=n_cells, p=probs)

    # Per-gene baseline expression rate (some genes globally brighter).
    base_rate = rng.lognormal(mean=-1.2, sigma=0.8, size=n_genes)
    mito_mask = np.array([g in MITO for g in genes])
    base_rate[mito_mask] *= 2.0  # mito genes are reasonably expressed at baseline

    # Per-cell library size factor (sequencing depth heterogeneity).
    libsize = rng.lognormal(mean=0.0, sigma=0.35, size=n_cells)

    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    exhaustion_truth = np.zeros(n_cells, dtype=np.float32)

    EXH = programmes["CD8 T (exhausted)"]
    for c in range(n_cells):
        t = labels[c]
        rate = base_rate.copy()
        # Boost this population's programme.
        for g in programmes[t]:
            rate[gene_idx[g]] *= rng.uniform(10, 28)
        # Pan-T genes for all T/Treg populations.
        if "T" in t or t == "Treg":
            for g in SHARED_T:
                rate[gene_idx[g]] *= rng.uniform(6, 12)
        # Continuous exhaustion gradient across the CD8 compartment.
        if t.startswith("CD8"):
            e = rng.beta(2, 2) if t == "CD8 T (exhausted)" else rng.beta(1.4, 5)
            exhaustion_truth[c] = e
            for g in EXH:
                rate[gene_idx[g]] *= (1.0 + 3.5 * e)

        expected = rate * libsize[c]
        counts = rng.poisson(expected).astype(np.float32)
        # Dropout: zero a fraction of low-count entries (capture inefficiency).
        drop = rng.random(n_genes) < (0.45 * np.exp(-expected))
        counts[drop] = 0.0
        X[c] = counts

    # Inject low-quality cells: collapse complexity, inflate mitochondrial reads.
    n_lowq = int(lowq_fraction * n_cells)
    lowq = rng.choice(n_cells, size=n_lowq, replace=False)
    X[lowq] *= 0.25
    X[np.ix_(lowq, np.where(mito_mask)[0])] *= rng.uniform(6, 12, size=(n_lowq, mito_mask.sum()))
    is_lowq = np.zeros(n_cells, dtype=bool)
    is_lowq[lowq] = True

    obs = pd.DataFrame(
        {
            "true_label": labels,
            "true_exhaustion": exhaustion_truth,
            "injected_lowq": is_lowq,
        },
        index=[f"cell_{i:05d}" for i in range(n_cells)],
    )
    return CellData(X=X, var_names=genes, obs=obs)


def load_h5ad(path: str) -> CellData:
    """Load a real dataset from an .h5ad file (requires anndata)."""
    import anndata as ad  # optional dependency

    a = ad.read_h5ad(path)
    X = a.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    return CellData(X=X.astype(np.float32), var_names=list(a.var_names), obs=a.obs.copy())
