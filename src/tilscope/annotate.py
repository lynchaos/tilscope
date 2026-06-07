"""Marker-based cell-type annotation and T-cell exhaustion scoring.

Signature scoring uses the canonical programmes in :data:`SIGNATURES`. Each
cell receives a per-signature score (mean z-scored expression of the genes in
the set that are present), clusters are labelled by their dominant signature,
and a dedicated exhaustion score is computed for the CD8 compartment - the
biology a T-cell-engager programme cares about most.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CELL_PROGRAMMES

# Annotation reference = the canonical programmes (recovered, not peeked at:
# the pipeline never sees the simulator's ground-truth labels).
SIGNATURES: dict[str, list[str]] = dict(CELL_PROGRAMMES)

# The exhaustion programme used for the continuous CD8 score.
EXHAUSTION_GENES = ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1", "CTLA4"]


def score_signatures(logz: pd.DataFrame) -> pd.DataFrame:
    """Per-cell signature scores from a z-scored log-normalised frame.

    ``logz`` is (cells x genes). Returns (cells x signatures).
    """
    scores = {}
    for name, genes in SIGNATURES.items():
        present = [g for g in genes if g in logz.columns]
        scores[name] = logz[present].mean(axis=1) if present else 0.0
    return pd.DataFrame(scores, index=logz.index)


def exhaustion_score(logz: pd.DataFrame) -> pd.Series:
    present = [g for g in EXHAUSTION_GENES if g in logz.columns]
    return logz[present].mean(axis=1) if present else pd.Series(0.0, index=logz.index)


def annotate_clusters(clusters: np.ndarray, sig_scores: pd.DataFrame) -> dict[int, str]:
    """Label each cluster by its highest mean signature score."""
    df = sig_scores.copy()
    df["__cluster__"] = clusters
    means = df.groupby("__cluster__").mean()
    return {int(cl): means.loc[cl].idxmax() for cl in means.index}
