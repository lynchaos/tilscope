"""LLM-integrated narrative layer.

This is the "LLM-integrated tool" piece: it turns a structured analysis into a
written executive summary an experimentalist can read. With an Anthropic API
key it calls Claude; with no key it produces a deterministic, fully offline
summary built from the same structured findings. The tool therefore *always*
works and is reproducible in CI, while still demonstrating the agentic /
LLM-integration pattern when credentials are available.
"""
from __future__ import annotations

import json
import os

from .pipeline import Results

# A current Claude model; override with TILSCOPE_MODEL.
DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are a computational biology assistant writing for an immuno-oncology "
    "discovery team. Given a JSON summary of a single-cell analysis of the "
    "tumour microenvironment, write a concise (180-260 word) executive summary: "
    "the composition of the immune infiltrate, the state of the CD8+ T-cell "
    "compartment with emphasis on exhaustion, and one or two cautious, "
    "testable hypotheses relevant to T-cell-engager therapy. Use only the "
    "numbers provided. Be precise, never overclaim, and flag that the data is "
    "simulated."
)


def build_findings(results: Results) -> dict:
    """The structured payload that both the LLM and the fallback consume."""
    return {
        "note": "Synthetic demonstration data, not a real patient sample.",
        "qc": results.qc.summary,
        "qc_thresholds": results.qc.thresholds,
        "clustering": results.diagnostics,
        "composition_pct": {k: round(v * 100, 1) for k, v in results.composition.items()},
        "exhaustion": results.exhaustion,
        "parameters": results.params,
    }


def _llm_narrative(findings: dict, model: str) -> str:
    import anthropic  # optional dependency

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    msg = client.messages.create(
        model=model,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(findings, indent=2)}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()


def _template_narrative(findings: dict) -> str:
    qc = findings["qc"]
    comp = findings["composition_pct"]
    exh = findings["exhaustion"]
    diag = findings["clustering"]

    top = sorted(comp.items(), key=lambda kv: kv[1], reverse=True)
    comp_str = "; ".join(f"{name} {pct}%" for name, pct in top)
    cd8_pct = round(sum(v for k, v in comp.items() if k.startswith("CD8")), 1)
    ari = diag.get("adjusted_rand_index")
    ari_str = (f" Recovered clusters agree well with the underlying populations "
               f"(adjusted Rand index {ari}).") if ari is not None else ""

    return (
        f"This is a synthetic demonstration of a tumour-microenvironment "
        f"single-cell analysis and does not describe a real sample. After "
        f"automated QC, {qc['n_cells_pass']} of {qc['n_cells_input']} cells "
        f"passed ({qc['pct_fail']}% removed for low complexity or elevated "
        f"mitochondrial content), leaving a median of "
        f"{int(qc['median_genes_per_cell'])} genes per cell.\n\n"
        f"Unsupervised clustering ({diag['clustering_method']}) resolved the "
        f"infiltrate into interpretable populations: {comp_str}.{ari_str} The "
        f"CD8+ T-cell compartment makes up roughly {cd8_pct}% of recovered "
        f"cells.\n\n"
        f"Within that compartment, {round(exh['cd8_high_exhaustion_fraction'] * 100)}% "
        f"of CD8+ T cells score high for a canonical exhaustion programme "
        f"(PD-1, LAG-3, TIM-3, TIGIT, TOX), with a mean exhaustion z-score of "
        f"{exh['mean_cd8_exhaustion']}. A substantial exhausted fraction "
        f"alongside a measurable regulatory (Treg) presence is the kind of "
        f"immunosuppressive signature that can blunt T-cell-engager activity. "
        f"Testable follow-ups would be to (i) confirm co-expression of "
        f"inhibitory receptors at single-cell resolution and (ii) ask whether "
        f"checkpoint co-targeting shifts the effector-to-exhausted CD8 ratio."
    )


def generate_narrative(results: Results, use_llm: bool = False,
                       model: str | None = None) -> tuple[str, str]:
    """Return (narrative_text, source) where source is 'llm' or 'template'."""
    findings = build_findings(results)
    model = model or os.environ.get("TILSCOPE_MODEL", DEFAULT_MODEL)
    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _llm_narrative(findings, model), "llm"
        except Exception as exc:  # graceful fallback keeps the tool reliable
            return (_template_narrative(findings)
                    + f"\n\n[LLM narrative unavailable: {exc}. Showing deterministic summary.]",
                    "template")
    return _template_narrative(findings), "template"
