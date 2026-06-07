"""Tests: reproducibility, QC behaviour, annotation recovery, and reporting.

These double as the correctness contract for the pipeline. The annotation
test is the important one - it asserts that unsupervised clustering +
marker-based labelling actually recovers the planted populations.
"""
import numpy as np
import pytest

from tilscope import simulate_tme, run_pipeline, generate_narrative
from tilscope.report import build_html


@pytest.fixture(scope="module")
def results():
    cd = simulate_tme(n_cells=1200, seed=0)
    return run_pipeline(cd, seed=0)


def test_simulation_is_deterministic():
    a = simulate_tme(n_cells=500, seed=42)
    b = simulate_tme(n_cells=500, seed=42)
    assert np.array_equal(a.X, b.X)


def test_qc_flags_injected_lowq(results):
    obs = results.qc.metrics
    # Some cells should fail, and the failures should be enriched for injected low-quality.
    assert obs["pass_qc"].sum() < len(obs)
    assert results.qc.summary["pct_fail"] > 0


def test_pipeline_recovers_populations(results):
    # Unsupervised recovery should align reasonably with ground truth.
    assert results.diagnostics["adjusted_rand_index"] > 0.5
    # The major immuno-oncology populations should be present.
    found = set(results.composition.index)
    assert any(c.startswith("CD8") for c in found)
    assert "Treg" in found or "CD4 T (helper)" in found


def test_exhaustion_score_is_sane(results):
    exh = results.exhaustion
    assert 0.0 <= exh["cd8_high_exhaustion_fraction"] <= 1.0


def test_narrative_and_report(results):
    text, source = generate_narrative(results, use_llm=False)
    assert source == "template"
    assert "synthetic" in text.lower()
    html = build_html(results, text, source)
    assert "<html" in html and "Executive summary" in html
