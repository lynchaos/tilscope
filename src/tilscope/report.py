"""Figure generation and self-contained HTML report.

Produces a single portable .html file (figures embedded as base64) plus a
machine-readable results.json. The visual language is a warm "lab paper"
aesthetic - deliberately not a generic dashboard - because the report is the
artifact a reviewer remembers.
"""
from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .pipeline import Results
from .narrative import build_findings

PAPER = "#f7f4ed"
INK = "#1c1a17"
ACCENT = "#7a2230"   # oxblood
TEAL = "#0f6e6e"
MUTED = "#8a8275"

# Stable categorical palette for cell types.
PALETTE = ["#7a2230", "#0f6e6e", "#c87a2c", "#3b5b7a", "#6a7a3b",
           "#8a4a6a", "#4a8a7a", "#a89a5a", "#5a5a8a", "#9a5a3a"]


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=PAPER, edgecolor="none")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _style(ax):
    ax.set_facecolor(PAPER)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _qc_figure(results: Results) -> str:
    m = results.qc.metrics
    thr = results.qc.thresholds
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), facecolor=PAPER)
    passed = m["pass_qc"]

    axes[0].hist(np.log1p(m["n_genes"]), bins=40, color=TEAL, alpha=0.85)
    axes[0].axvline(np.log1p(thr["min_genes"]), color=ACCENT, ls="--", lw=1.4)
    axes[0].set_title("Genes per cell (log1p)")
    axes[1].hist(np.log1p(m["total_counts"]), bins=40, color=TEAL, alpha=0.85)
    axes[1].set_title("Total counts per cell (log1p)")
    axes[2].hist(m.loc[passed, "pct_mito"], bins=30, color=TEAL, alpha=0.85, label="pass")
    axes[2].hist(m.loc[~passed, "pct_mito"], bins=30, color=ACCENT, alpha=0.7, label="fail")
    axes[2].axvline(thr["max_pct_mito"], color=ACCENT, ls="--", lw=1.4)
    axes[2].set_title("Mitochondrial content (%)")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _embedding_figure(results: Results) -> str:
    obs = results.obs
    types = list(results.composition.index)
    cmap = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(sorted(types))}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=PAPER)

    for t in sorted(types):
        sub = obs[obs["cell_type"] == t]
        axes[0].scatter(sub["emb_x"], sub["emb_y"], s=6, color=cmap[t], label=t, alpha=0.8, lw=0)
    axes[0].set_title("Cell types (recovered)")
    axes[0].legend(frameon=False, fontsize=7, markerscale=2, loc="upper right",
                   bbox_to_anchor=(1.0, 1.0))

    sc = axes[1].scatter(obs["emb_x"], obs["emb_y"], s=6,
                         c=obs["exhaustion"], cmap="inferno", alpha=0.85, lw=0)
    axes[1].set_title("CD8 exhaustion score")
    cb = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7, colors=INK)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); _style(ax)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _marker_figure(results: Results) -> str:
    mm = results.marker_matrix
    # Column-normalise for readability (relative expression across cell types).
    z = (mm - mm.mean(axis=0)) / (mm.std(axis=0) + 1e-9)
    fig, ax = plt.subplots(figsize=(min(14, 0.32 * z.shape[1] + 2), 0.5 * z.shape[0] + 1.5),
                           facecolor=PAPER)
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(z.shape[1])); ax.set_xticklabels(z.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(z.shape[0])); ax.set_yticklabels(z.index, fontsize=8)
    ax.set_title("Mean marker expression by cell type (z-scored)")
    _style(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=INK)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _composition_figure(results: Results) -> str:
    comp = results.composition * 100
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=PAPER)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(comp))]
    ax.barh(list(comp.index)[::-1], list(comp.values)[::-1], color=colors[::-1])
    ax.set_xlabel("% of recovered cells")
    ax.set_title("Immune infiltrate composition")
    _style(ax)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="card"><div class="val">{value}</div>'
            f'<div class="lbl">{label}</div>{sub_html}</div>')


def build_html(results: Results, narrative: str, narrative_source: str) -> str:
    qc = results.qc.summary
    exh = results.exhaustion
    diag = results.diagnostics
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    src_badge = ("Claude (LLM-generated)" if narrative_source == "llm"
                 else "deterministic offline summary")

    cards = "".join([
        _metric_card("cells passing QC", f"{qc['n_cells_pass']:,}",
                     f"of {qc['n_cells_input']:,} ({qc['pct_fail']}% removed)"),
        _metric_card("CD8 exhaustion", f"{round(exh['cd8_high_exhaustion_fraction']*100)}%",
                     "high-scoring CD8+ T cells"),
        _metric_card("clustering", diag["clustering_method"],
                     (f"ARI {diag['adjusted_rand_index']}" if "adjusted_rand_index" in diag else "")),
        _metric_card("populations", str(len(results.composition)), "recovered cell types"),
    ])

    narrative_html = "".join(f"<p>{p.strip()}</p>" for p in narrative.split("\n\n") if p.strip())

    figs = {
        "qc": _qc_figure(results),
        "emb": _embedding_figure(results),
        "comp": _composition_figure(results),
        "markers": _marker_figure(results),
    }

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TIL-Scope &mdash; TME Single-Cell Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,900&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --paper:{PAPER}; --ink:{INK}; --accent:{ACCENT}; --teal:{TEAL}; --muted:{MUTED};
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--paper); color:var(--ink);
    font-family:'IBM Plex Sans',sans-serif; line-height:1.6;
    background-image:radial-gradient(rgba(0,0,0,0.025) 1px,transparent 1px);
    background-size:22px 22px; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:56px 28px 80px; }}
  header {{ border-bottom:3px solid var(--ink); padding-bottom:22px; margin-bottom:8px; }}
  .kicker {{ font-family:'IBM Plex Mono',monospace; font-size:12px; letter-spacing:.22em;
    text-transform:uppercase; color:var(--accent); }}
  h1 {{ font-family:'Fraunces',serif; font-weight:900; font-size:clamp(38px,6vw,64px);
    line-height:1.0; margin:.18em 0 .15em; letter-spacing:-.01em; }}
  .lede {{ font-family:'Fraunces',serif; font-size:20px; color:#3a352e; max-width:60ch; }}
  .meta {{ font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--muted);
    margin-top:14px; display:flex; gap:18px; flex-wrap:wrap; }}
  .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:30px 0; }}
  .card {{ background:#fff; border:1px solid #e4ddd0; border-radius:4px; padding:16px 16px 14px;
    box-shadow:0 1px 0 #e4ddd0; }}
  .card .val {{ font-family:'Fraunces',serif; font-weight:600; font-size:30px; color:var(--accent);
    line-height:1; }}
  .card .lbl {{ font-size:12px; color:var(--ink); margin-top:8px; }}
  .card .sub {{ font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--muted); margin-top:3px; }}
  section {{ margin-top:46px; }}
  h2 {{ font-family:'Fraunces',serif; font-size:26px; font-weight:600; margin:0 0 4px;
    border-left:5px solid var(--teal); padding-left:14px; }}
  .secnum {{ font-family:'IBM Plex Mono',monospace; color:var(--teal); font-size:13px; }}
  figure {{ margin:18px 0 0; }}
  figure img {{ width:100%; border:1px solid #e4ddd0; border-radius:4px; background:#fff; }}
  figcaption {{ font-size:12.5px; color:var(--muted); margin-top:8px;
    font-family:'IBM Plex Mono',monospace; }}
  .summary {{ background:#fff; border:1px solid #e4ddd0; border-left:5px solid var(--accent);
    border-radius:4px; padding:22px 26px; }}
  .summary p {{ margin:0 0 12px; }} .summary p:last-child {{ margin-bottom:0; }}
  .badge {{ display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:11px;
    background:var(--ink); color:var(--paper); padding:3px 9px; border-radius:3px;
    letter-spacing:.04em; }}
  .note {{ background:#fbf0d8; border:1px solid #e8d7a8; border-radius:4px; padding:12px 16px;
    font-size:13px; }}
  footer {{ margin-top:56px; padding-top:20px; border-top:1px solid #d8d0c0;
    font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--muted); }}
  code {{ font-family:'IBM Plex Mono',monospace; background:#efe9dc; padding:1px 5px; border-radius:3px; }}
  @media (max-width:680px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }} }}
</style></head>
<body><div class="wrap">
  <header>
    <div class="kicker">TIL-Scope &middot; Tumour Microenvironment Atlas</div>
    <h1>Single-cell report:<br>the T-cell infiltrate</h1>
    <p class="lede">An automated, reproducible scRNA-seq analysis of the tumour
      immune microenvironment, with an LLM-written interpretation aimed at a
      T-cell-engager discovery programme.</p>
    <div class="meta">
      <span>generated {ts}</span><span>seed {results.params['seed']}</span>
      <span>{results.params['n_hvg']} HVGs &middot; {results.params['n_pcs']} PCs</span>
      <span>narrative: {src_badge}</span>
    </div>
  </header>

  <div class="cards">{cards}</div>

  <div class="note"><strong>Demonstration data.</strong> This report is built
    from a synthetic TME dataset for portfolio and reproducibility purposes.
    It is <em>not</em> derived from a real patient sample. The analysis,
    annotation and reporting code is identical to what would run on real data.</div>

  <section>
    <div class="secnum">01 / interpretation</div>
    <h2>Executive summary</h2>
    <div class="summary"><span class="badge">{src_badge}</span>
      <div style="height:12px"></div>{narrative_html}</div>
  </section>

  <section>
    <div class="secnum">02 / quality control</div>
    <h2>Automated QC</h2>
    <p>Thresholds are derived from the data (median-absolute-deviation outlier
      detection), then applied uniformly. Cut-offs:
      <code>min_genes = {results.qc.thresholds['min_genes']}</code>,
      <code>max_pct_mito = {results.qc.thresholds['max_pct_mito']}</code>.</p>
    <figure><img src="data:image/png;base64,{figs['qc']}" alt="QC distributions">
      <figcaption>Per-cell QC distributions. Dashed lines mark applied thresholds;
        failed cells shown in oxblood.</figcaption></figure>
  </section>

  <section>
    <div class="secnum">03 / cell populations</div>
    <h2>The immune infiltrate</h2>
    <figure><img src="data:image/png;base64,{figs['emb']}" alt="t-SNE embeddings">
      <figcaption>t-SNE of the post-QC cells, coloured by recovered cell type
        (left) and by CD8 exhaustion score (right).</figcaption></figure>
    <figure><img src="data:image/png;base64,{figs['comp']}" alt="Composition">
      <figcaption>Composition of the recovered infiltrate.</figcaption></figure>
  </section>

  <section>
    <div class="secnum">04 / marker validation</div>
    <h2>Do the labels hold up?</h2>
    <p>Annotation is unsupervised: clusters are labelled by their dominant
      canonical signature, never by peeking at ground truth. The heatmap is the
      audit &mdash; each population should light up its own marker programme.</p>
    <figure><img src="data:image/png;base64,{figs['markers']}" alt="Marker heatmap">
      <figcaption>Mean expression of canonical markers across recovered cell
        types (z-scored per gene).</figcaption></figure>
  </section>

  <footer>
    TIL-Scope &mdash; reproducible TME single-cell tooling.
    Run <code>tilscope run --out report.html --seed {results.params['seed']}</code>
    to regenerate this exact report. Add <code>--llm</code> with an
    <code>ANTHROPIC_API_KEY</code> set for a Claude-written interpretation.
  </footer>
</div></body></html>"""


def write_report(results: Results, narrative: str, narrative_source: str,
                 out_html: str) -> dict:
    html = build_html(results, narrative, narrative_source)
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    json_path = out_html.rsplit(".", 1)[0] + ".results.json"
    payload = build_findings(results)
    payload["narrative"] = narrative
    payload["narrative_source"] = narrative_source
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return {"html": out_html, "json": json_path}
