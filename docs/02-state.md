# Project State

**Last updated:** 2026-05-19 (1000-example semantic embedding cache built)

This document is the live snapshot. Update it after any substantive change.

## Active focus

The project is in the **bootstrap phase**. WP-01 is complete; WP-02 remains active.

- [WP-02: SIR identifiability study on synthetic data](work-packages/WP-02-sir-identifiability-study.md) — `status: active`

ADR 0006 and ADR 0010 are now accepted. The next embedding work should scale the accepted hybrid pipeline and audit semantic clustering quality with stronger text embeddings.

## What is decided

See `decisions/` for the full ADR set. Accepted at bootstrap:

- 0001 — Success metric: $R^2 \geq 0.3$ on log-spread-parameters on post-cutoff eval set.
- 0002 — SIR treated as empirical curve compression, not mechanistic.
- 0003 — Reddit-only as initial platform; r/memes as first subreddit.
- 0004 — Embedding contamination mitigated via time-cutoff evaluation; final cutoff window pending ADR 0011.
- 0005 — SIR initial conditions fixed: $S(0) = 1 - I(0)$, $R(0) = 0$.
- 0006 — Embedding pipeline: hybrid VLM image embedding + filtered EasyOCR text embedding.
- 0010 — SIR parameterization for prediction: primary targets are stable curve summaries; transformed SIR parameters are auxiliary diagnostic targets with identifiability metadata.
- 0013 — JAX as the primary numerical backend for new research code; predictor framework choice remains open.

## What is NOT decided

These ADRs exist as `proposed` placeholders and are gated by experiments:

- 0007 — Engagement signal definition: how to convert posts → $I(t)$.
- 0008 — Non-viral baseline data source.
- 0009 — Clustering algorithm and granularity.
- 0011 — Train/test/eval split strategy.
- 0012 — Baseline set (the canonical list is in `01-scientific-rigor.md` §1; this ADR will lock specifics for the first model run).

## Environment

- Package manager: **uv** (see `03-environment.md`).
- Python: pinned at **3.12** via `.python-version`.
- `pyproject.toml` and `uv.lock` now include WP-02 runtime dependencies: `jax`, `diffrax`, `numpyro`, `corner`, `matplotlib`, `scipy`; WP-01 runtime dependencies: `torch`, `transformers`, `open-clip-torch`, `pandas`, `pyarrow`, `opencv-python-headless`, `easyocr`, `pytesseract`, `paddleocr`, `paddlepaddle`, `sentence-transformers`, `openai`, `umap-learn`, `hdbscan`, `scikit-learn`, `pillow`, `requests`, `tqdm`; dev dependency: `pytest`.
- Local hardware: NVIDIA 3090 GPU available.

## Data inventory

| Asset                           | Location                                   | Coverage           | Notes                                                                                 |
| ------------------------------- | ------------------------------------------ | ------------------ | ------------------------------------------------------------------------------------- |
| r/memes post archive            | `reddit_data/` (existing)                  | up to ~early-2020s | Inherited from prior iteration; format and integrity should be verified before use    |
| Top-40k-subreddits archive      | external (described in `00-background.md`) | TBD                | Not yet ingested                                                                      |
| Previous-iteration meme dataset | `dataset_saves/` (existing)                | ~700 fitted curves | From the 2023 iteration; useful as a sanity-check reference, not as a training source |
| Hardware                        | local 3090 GPU                             | —                  | Sufficient for local embedding/inference at moderate scale                            |

The legacy code in this repo (`memes_base/`, `train_script.py`, `test_ocr.py`, etc.) predates this framework. **It should not be assumed correct or representative of current decisions.** Treat it as reference material; reuse code only after reading and understanding it.

## Models

| Model            | Status                | Version / weights                         | Notes                                                |
| ---------------- | --------------------- | ----------------------------------------- | ---------------------------------------------------- |
| Embedding model  | chosen for first pass | OpenCLIP ViT-B-16 + filtered EasyOCR      | ADR 0006 accepted hybrid image/OCR pipeline          |
| Text embeddings  | under comparison      | MiniLM L6 v2, BGE base en v1.5, E5 base v2 | Cached on 1000 examples for semantic clustering work |
| Spread predictor | not yet trained       | —                                         | Architecture deferred until embedding pipeline lands |

## Eval set

Not yet defined. Will be locked in ADR 0011. Default expectation: held-out memes originating after the chosen embedding model's training cutoff.

## Recent changes log

Add a one-line entry every time you change the state of the project. Newest at top.

- 2026-05-19 — Built a cached 1000-example r/memes semantic embedding set with downloaded images, filtered EasyOCR OCR text, OpenCLIP image embeddings, and local MiniLM/BGE/E5 sentence embeddings; OpenAI embedding skipped pending API key.
- 2026-05-19 — Ran a 90-image cached WP-01 clustering check; HDBSCAN now forms non-noise clusters, but proxy purity/silhouette remain weak, so hand-labeled semantic evaluation is still needed before relying on clustering metrics.
- 2026-05-19 — Completed WP-01 and accepted ADR 0006: use hybrid OpenCLIP image embeddings plus filtered EasyOCR (`easyocr_filtered:none`) transcript embeddings; larger OCR/cluster audit remains follow-up work.
- 2026-05-19 — Ran WP-01 cached real r/memes pilot on 24 manually transcribed images; OpenCLIP B/16 image→text top-1 was weak (16.7%) and clustering produced no non-noise clusters, so ADR 0006 remains proposed.
- 2026-05-19 — Implemented WP-01 text-fidelity benchmark harness and ran a synthetic Impact-caption smoke test; ADR 0006 remains proposed and blocked on real r/memes + manual-transcript evidence.
- 2026-05-19 — User accepted ADR 0010.
- 2026-05-19 — Drafted ADR 0010 recommendation: primary prediction/evaluation targets should be stable curve summaries; transformed SIR parameters remain auxiliary diagnostic targets with identifiability metadata.
- 2026-05-19 — Recorded WP-02 first-pass conclusion: SIR curve fitting works for posterior-predictive curves, but natural parameters are not always identifiable; near-threshold and low-count regimes should be flagged rather than trusted as clean labels.
- 2026-05-19 — Ran WP-02 81-case Hessian grid and six selected NUTS validations; Hessian works in well-identified regimes and fails in low-count/ridge regimes. Low-count uncertainty threshold study added as optional next WP-02 step.
- 2026-05-19 — Accepted ADR 0013 (JAX-first numerical backend). Reduced representative NUTS check default to 4 sequential chains with 500 warmup + 500 retained samples per chain.
- 2026-05-19 — Started WP-02. Added JAX/diffrax SIR forward model, Poisson binned-count likelihood, MLE/Hessian scaffold, optional NumPyro/NUTS posterior path, corner-plot generation, and smoke tests.
- 2026-06-10 — Moved the agent protocol from `.cursor/rules/project-protocol.mdc` to `/AGENTS.md` for tool-neutral compatibility (Cursor + Claude Code + others). Deleted the Cursor rule; `ocr-research.mdc` retained as a contextual rule.
- 2026-05-19 — Adopted uv for package management. Bootstrapped `pyproject.toml`, `.python-version` (3.12), added `docs/03-environment.md` and updated protocol rule.
- 2026-05-19 — Framework bootstrapped (`docs/`, ADRs 0001–0012, WP-01, WP-02).
