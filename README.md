# TIL-Scope

**Reproducible single-cell tooling for the tumour microenvironment — with an LLM-written report.**

![CI](https://github.com/lynchaos/tilscope/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

TIL-Scope takes a single-cell RNA-seq matrix of a tumour immune infiltrate and, in one command, produces a self-contained HTML report: automated QC, unsupervised clustering, marker-based cell-type annotation, a CD8⁺ **T-cell exhaustion** score, and a written executive summary aimed at a T-cell-engager discovery programme. The narrative is written by Claude when an API key is available, and by a deterministic offline summariser otherwise — so the tool is reproducible in CI and informative on a laptop with no credentials.

It is a compact, end-to-end demonstration of *AI-first* computational biology tooling: the kind of thing a wet-lab scientist can run unattended and a reviewer can read in two minutes.

> **Note.** The bundled demo runs on **synthetic** TME data so it reproduces identically anywhere, with no external download. The analysis, annotation and reporting code is the same code that runs on real data — point it at an `.h5ad` to use your own.

---

## Quickstart

```bash
pip install -e .
tilscope run --out report.html --seed 0
# -> report.html  +  report.results.json
```

Open `examples/report.html` for a pre-generated example.

### With a Claude-written interpretation

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=sk-...
tilscope run --out report.html --llm
```

### On your own data

```bash
pip install -e ".[sc]"          # adds anndata/scanpy + Leiden clustering
tilscope run --input my_tme.h5ad --out report.html
```

### On the Tirosh 2016 melanoma dataset (public benchmark)

The [Tirosh et al. 2016](https://doi.org/10.1126/science.aad0501) melanoma dataset
(GEO: [GSE72056](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE72056))
contains 4,645 single cells from 19 metastatic tumours — malignant cells, T cells,
B cells, NK cells, macrophages, CAFs, and endothelial cells — making it an ideal
real-data benchmark for tilscope.

```bash
pip install -e ".[sc]"
python scripts/download_tirosh2016.py   # downloads ~72 MB, saves data/tirosh2016_melanoma.h5ad
tilscope run --input data/tirosh2016_melanoma.h5ad --out report_real.html
```

---

## What it does

| Stage | Method |
|-------|--------|
| **QC** | Per-cell counts, gene complexity, % mitochondrial. Thresholds are **data-driven** (MAD outlier detection), not hard-coded, and recorded in the output. |
| **Normalisation** | CP10K + `log1p`. |
| **Feature selection** | Top-variance highly variable genes. |
| **Dimensionality reduction** | PCA → t-SNE embedding for visualisation. |
| **Clustering** | Leiden (if `scanpy`/`leidenalg` installed) or silhouette-tuned K-means. |
| **Annotation** | Unsupervised: clusters labelled by their dominant canonical signature (CD8 effector/exhausted, CD4, Treg, NK, B, myeloid, tumour). |
| **Exhaustion** | Per-cell score over PD-1, LAG-3, TIM-3, TIGIT, TOX, ENTPD1, CTLA-4. |
| **Report** | Self-contained HTML (figures embedded) + machine-readable JSON. |

The annotation **never sees ground truth**. On the synthetic demo it nonetheless recovers the planted populations with an adjusted Rand index ≈ 0.96 — the marker heatmap in the report is the audit that the labels hold up.

## The LLM-integration pattern

`narrative.py` turns the structured analysis into a written summary. It is designed the way production LLM tooling should be:

- **Structured-input, bounded-output** — the model only ever sees a JSON `findings` payload and a system prompt that forbids inventing numbers.
- **Graceful degradation** — no key, or an API error, falls back to a deterministic summary built from the same payload. The tool never breaks.
- **Auditable** — the narrative source (`llm` / `template`) is stamped into the report and the JSON.

## Engineering

- **Reproducible** — fixed seeds; the same input yields byte-identical simulated data and the same report.
- **Tested** — `pytest` suite asserts determinism, QC behaviour, and that annotation recovers the planted biology.
- **CI** — GitHub Actions across Python 3.10–3.12, including a report smoke-test.
- **Installable** — `pip install` with a `tilscope` console entry point; optional extras keep the core dependency-light.

## Project layout

```
src/tilscope/
  data.py        synthetic TME generator + .h5ad loader
  qc.py          MAD-based automated QC
  pipeline.py    normalise → HVG → PCA → cluster → annotate → score
  annotate.py    canonical marker signatures + exhaustion scoring
  narrative.py   LLM-integrated summary with deterministic fallback
  report.py      matplotlib figures + self-contained HTML
  cli.py         `tilscope run`
tests/           correctness + reproducibility contract
examples/        pre-generated report
```

## Caveats

The demo data is simulated and is **not** a substitute for a real dataset; the exhaustion gradient is planted by construction. The value here is the *tooling and method*, demonstrated end-to-end on data anyone can regenerate.

## License

MIT © Kemal Yaylali

---

**Contact:** [support@yaylali.uk](mailto:support@yaylali.uk) · **ORCID:** [0000-0003-1190-7807](https://orcid.org/0000-0003-1190-7807) · **GitHub:** [@lynchaos](https://github.com/lynchaos)
