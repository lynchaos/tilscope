"""Download and convert the Tirosh et al. 2016 melanoma scRNA-seq dataset.

Source: GEO accession GSE72056
Paper:  Tirosh I et al. "Dissecting the multicellular ecosystem of metastatic
        melanoma by single-cell RNA-seq." Science 2016;352(6282):189-96.
        https://doi.org/10.1126/science.aad0501

The dataset contains 4,645 single cells from 19 metastatic melanoma tumours
(Smart-seq2), including malignant cells, T cells, B cells, NK cells,
macrophages, CAFs, and endothelial cells — all populations tilscope expects.

Output: data/tirosh2016_melanoma.h5ad  (~120 MB uncompressed)

Usage:
    pip install anndata pandas numpy requests
    python scripts/download_tirosh2016.py
    tilscope run --input data/tirosh2016_melanoma.h5ad --out report_real.html
"""
from __future__ import annotations

import gzip
import io
import pathlib
import sys
import urllib.request

import numpy as np
import pandas as pd

GEO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE72nnn/GSE72056/suppl/"
    "GSE72056_melanoma_single_cell_revised_v2.txt.gz"
)
OUT_DIR = pathlib.Path(__file__).parent.parent / "data"
TXT_CACHE = OUT_DIR / "GSE72056_melanoma_single_cell_revised_v2.txt.gz"
H5AD_OUT = OUT_DIR / "tirosh2016_melanoma.h5ad"

# The first three rows in the GEO file are metadata, not gene expression.
META_ROWS = {
    "tumor",
    "malignant(1=no,2=yes,0=unresolved)",
    "non-malignant cell type (1=T,2=B,3=Macro.4=Endo.,5=CAF;6=NK)",
}
CELL_TYPE_MAP = {
    0: "unresolved",
    1: "T cell",
    2: "B cell",
    3: "Macrophage",
    4: "Endothelial",
    5: "CAF",
    6: "NK",
}


def download(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _progress(block: int, block_size: int, total: int) -> None:
    done = block * block_size
    if total > 0:
        pct = min(100, done * 100 // total)
        mb = done / 1e6
        print(f"\r  {pct:3d}%  {mb:.1f} MB", end="", flush=True)


def build_anndata(gz_path: pathlib.Path):
    import anndata as ad

    print("Parsing text file ...")
    with gzip.open(gz_path, "rt") as fh:
        raw = pd.read_csv(fh, sep="\t", index_col=0)

    # GEO file is genes x cells; first three rows are metadata.
    meta = raw.loc[list(META_ROWS)].copy()
    expr = raw.drop(index=list(META_ROWS)).copy()

    # Transpose to cells x genes, convert to float32.
    X = expr.T.astype(np.float32).values
    cell_ids = expr.columns.tolist()
    gene_ids = expr.index.tolist()

    # Remove ERCC spike-ins.
    keep = [i for i, g in enumerate(gene_ids) if not g.startswith("ERCC")]
    X = X[:, keep]
    gene_ids = [gene_ids[i] for i in keep]

    # Build obs metadata.
    tumor = meta.loc["tumor"].values.astype(str)
    malignant_code = pd.to_numeric(
        meta.loc["malignant(1=no,2=yes,0=unresolved)"].values, errors="coerce"
    ).astype(float)
    nmtype_code = pd.to_numeric(
        meta.loc["non-malignant cell type (1=T,2=B,3=Macro.4=Endo.,5=CAF;6=NK)"].values,
        errors="coerce",
    ).fillna(0).astype(int)

    cell_type = []
    for mal, nm in zip(malignant_code, nmtype_code):
        if mal == 2:
            cell_type.append("Malignant")
        else:
            cell_type.append(CELL_TYPE_MAP.get(int(nm), "unresolved"))

    obs = pd.DataFrame(
        {"tumor": tumor, "cell_type": cell_type},
        index=cell_ids,
    )

    adata = ad.AnnData(X=X)
    adata.obs = obs
    adata.var_names = gene_ids
    return adata


def main() -> None:
    try:
        import anndata  # noqa: F401
    except ImportError:
        sys.exit("anndata not installed. Run: pip install anndata")

    if not TXT_CACHE.exists():
        download(GEO_URL, TXT_CACHE)
    else:
        print(f"Using cached file: {TXT_CACHE}")

    adata = build_anndata(TXT_CACHE)
    print(f"Built AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"Cell types:\n{adata.obs['cell_type'].value_counts().to_string()}")

    H5AD_OUT.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(H5AD_OUT)
    print(f"\nSaved to {H5AD_OUT}")
    print(f"\nRun tilscope on it:")
    print(f"  tilscope run --input {H5AD_OUT} --out report_real.html")


if __name__ == "__main__":
    main()
