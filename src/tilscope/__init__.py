"""TIL-Scope: reproducible single-cell tooling for the tumour microenvironment.

A compact, end-to-end demonstration of AI-first computational biology tooling:
automated QC, unsupervised clustering, marker-based annotation, T-cell
exhaustion scoring, and an LLM-written report.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .data import simulate_tme, load_h5ad, CellData
from .pipeline import run_pipeline, Results
from .narrative import generate_narrative
from .report import write_report

__all__ = [
    "__version__",
    "simulate_tme", "load_h5ad", "CellData",
    "run_pipeline", "Results",
    "generate_narrative", "write_report",
]
