"""Automated quality control for single-cell count matrices.

Thresholds are derived from the data (median-absolute-deviation outlier
detection) rather than hard-coded, then optionally overridden by the user.
This mirrors the "reproducible pipelines with automated validation and QC"
requirement: the same input always yields the same flags, with the rationale
recorded in the returned summary.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .data import CellData, MITO


@dataclass
class QCResult:
    metrics: pd.DataFrame          # per-cell QC metrics + pass/fail
    thresholds: dict[str, float]   # the cut-offs actually applied
    summary: dict[str, float]      # headline numbers for the report


def _mad_bounds(x: np.ndarray, n_mads: float = 5.0) -> tuple[float, float]:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) or 1e-9
    return med - n_mads * mad, med + n_mads * mad


def compute_qc(
    cd: CellData,
    min_genes: int | None = None,
    max_pct_mito: float | None = None,
    n_mads: float = 5.0,
) -> QCResult:
    X = cd.X
    mito_idx = [i for i, g in enumerate(cd.var_names) if g in MITO]

    total_counts = X.sum(axis=1)
    n_genes = (X > 0).sum(axis=1)
    mito_counts = X[:, mito_idx].sum(axis=1) if mito_idx else np.zeros(X.shape[0])
    pct_mito = 100.0 * mito_counts / np.maximum(total_counts, 1.0)

    # Data-driven thresholds (operate in log space for the heavy-tailed metrics).
    lo_genes, _ = _mad_bounds(np.log1p(n_genes), n_mads)
    auto_min_genes = max(1.0, np.expm1(lo_genes))
    _, hi_mito = _mad_bounds(pct_mito, n_mads)

    thr_min_genes = float(min_genes if min_genes is not None else auto_min_genes)
    thr_max_mito = float(max_pct_mito if max_pct_mito is not None else min(hi_mito, 20.0))

    pass_genes = n_genes >= thr_min_genes
    pass_mito = pct_mito <= thr_max_mito
    passed = pass_genes & pass_mito

    metrics = pd.DataFrame(
        {
            "total_counts": total_counts,
            "n_genes": n_genes,
            "pct_mito": pct_mito,
            "pass_qc": passed,
        },
        index=cd.obs_names,
    )

    summary = {
        "n_cells_input": int(X.shape[0]),
        "n_cells_pass": int(passed.sum()),
        "n_cells_fail": int((~passed).sum()),
        "pct_fail": round(100.0 * (~passed).mean(), 2),
        "median_genes_per_cell": float(np.median(n_genes)),
        "median_counts_per_cell": float(np.median(total_counts)),
        "median_pct_mito": round(float(np.median(pct_mito)), 2),
    }
    thresholds = {"min_genes": round(thr_min_genes, 1), "max_pct_mito": round(thr_max_mito, 2)}
    return QCResult(metrics=metrics, thresholds=thresholds, summary=summary)
