# Project State

**Last updated:** 2026-11-27 (project resumed after a four-month pause; rules and docs simplified)

The current snapshot. History is in [`changelog.md`](changelog.md).

## Waiting on the user

Decisions and reviews only the user can give. Each entry says what is being asked, what the
agent recommends, and what is blocked meanwhile. Remove an entry once it is answered.

- **What counts as "one meme" in this project?** The clustering audit (WP-04) cannot be
  finished without it, and the modelling depends on it. Three candidates, from the audit's
  evidence: (a) a *template*, the reused base image: you and the agent labels agreed on
  every one of 56 informative pairs, and an image-only embedding or a plain perceptual hash
  keeps about 80% of each template together at 90% purity; (b) a *specific joke or trend*
  (the protocol's definition): agents agree with each other, but you disagreed on 8 of 39
  hard cases, and one global clustering threshold keeps only about a fifth of same-meme
  pairs together at 90% purity; (c) a *broad subject* (all Endgame memes), which is closer
  to several of your answers and has not been measured. Recommendation: none yet; the
  agent's lean is (a) as the unit that is measurable now, with (b) pursued as a
  neighbour-graph or soft-membership problem. Blocks ADR 0009 (clustering method), the
  embedding composition question below, and everything downstream of clusters.
- **Is the accepted embedding (image plus extracted text, ADR 0006/0017) still right?** For
  templates it is clearly worse than image alone (0.56 against 0.77); for meaning it helps
  at looser purity. Depends on the answer above. A bug in it (text-free images share one
  text vector) needs fixing either way.
- **Where should WSL live?** It was moved to the portable hard disk (D:) on 2026-11-27
  after C: filled. That works but makes everything slower (small-file writes about 14 times
  slower than the NVMe) and voids the fast-tier/archive-tier split of ADR 0015. Moving it
  back needs roughly 150 GB freed on C:, which only the user can do, and a size cap on the
  WSL disk so it cannot fill C: again. Agent recommendation: move back when convenient;
  nothing is blocked meanwhile. Details in `03-environment.md`.
- **Review of the submissions ingest (WP-16).** 11,782,247 posts ingested with exact
  reconciliation; the outcome has not been reviewed by anyone but the agent that produced
  it. Report: `experiments/WP-16/report.md`. Blocks nothing right now.
- **ADR 0016 (use several subreddits instead of r/memes alone).** Proposed; three cheap
  measurements it depends on have not been run. Blocks the train/test split (ADR 0011).

## Active focus

- [WP-04: clustering gold set and audit](work-packages/WP-04-clustering-gold-set-audit.md)
  — in progress on draft PR #1 (branch `worktree-wp01-clustering-dedup`), not yet on `main`.
  A 50,000-post sample from 2019 is drawn, its images fetched into `/mnt/d/memes/`, and
  near-duplicate families computed (`experiments/WP-04/`). Next steps are the sample-size
  derivation and labeling.
- [WP-02: SIR identifiability on synthetic data](work-packages/WP-02-sir-identifiability-study.md)
  — first pass complete; optional low-count follow-up open.
- [WP-12: SIR-Hawkes replication](work-packages/WP-12-sir-hawkes-replication.md) — active.

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
- 0015 — Storage layout: tiered NVMe (working set) / portable drive (archive + shards), with the image corpus in uncompressed 1.5 GB tar shards plus a parquet index. Format validated by measurement.
- 0017 — Clustering/meme-identity embedding is image ⊕ OCR only; post title/body excluded (kept available as a predictor-side ablation).

## What is NOT decided

These ADRs exist as `proposed` placeholders and are gated by experiments:

- 0007 — Engagement signal definition: how to convert posts → $I(t)$.
- 0008 — Non-viral baseline data source.
- 0009 — Clustering algorithm and granularity.
- 0011 — Train/test/eval split strategy. **Would be amended by ADR 0016** from a single time cut to a 2×2 factorial (community × era).
- 0012 — Baseline set (the canonical list is in `01-scientific-rigor.md` §1; this ADR will lock specifics for the first model run).
- 0016 — Multi-subreddit corpus: backbone/extension composition, $N_{c,t}$ exposure normalization, hierarchical partial pooling, factorial split. Drafted 2026-07-26; **three preconditions unrun** (see the ADR).

## Environment

- Package manager: **uv** (see `03-environment.md`).
- Python: pinned at **3.12** via `.python-version`.
- `pyproject.toml` and `uv.lock` now include WP-02 runtime dependencies: `jax`, `diffrax`, `numpyro`, `corner`, `matplotlib`, `scipy`; WP-01 runtime dependencies: `torch`, `transformers`, `open-clip-torch`, `pandas`, `pyarrow`, `opencv-python-headless`, `easyocr`, `pytesseract`, `paddleocr`, `paddlepaddle`, `sentence-transformers`, `openai`, `umap-learn`, `hdbscan`, `scikit-learn`, `pillow`, `requests`, `tqdm`; dev dependency: `pytest`.
- Hardware: NVIDIA 3090 in a **separate Windows desktop**, not the laptop this repo is usually edited on. GPU JAX requires WSL2 there (native Windows has no JAX CUDA build) and the Linux-gated `gpu` dependency group: `uv sync --group gpu`. See `03-environment.md`.

## Data inventory

| Asset                           | Location                                   | Coverage           | Notes                                                                                 |
| ------------------------------- | ------------------------------------------ | ------------------ | ------------------------------------------------------------------------------------- |
| **r/memes archive (current)**   | `/mnt/d/reddit_archive/subreddits25/` (11.9 GB) | **2008-07-11 → 2025-12-31** | Academic Torrents infohash `3e3f64de…` (2005-06→2025-12 top-40k). Downloaded and checksum-verified 2026-07-26. Both files **verified complete** by full streaming decompress: `memes_submissions.zst` 2.82 G → 33.3 GB, **11,782,247 submissions**; `memes_comments.zst` 9.11 G → 137.7 GB, **106,464,277 comments**. **Submissions fully ingested 2026-07-26 (WP-16); comments not.** Fetcher: `experiments/wp16/fetch_archive.py` |
| r/memes archive (superseded)    | `reddit_data/reddit/subreddits23/` (11 GB) | 2008-07-11 → 2023-12-31 | The 2023-12 release (`56aa49f9…`), verified complete: 11,332,165 submissions / 100,181,097 comments. **Kept as the control** that proved the 2025-12 cut is a faithful superset (all years 2008–2023 match exactly). Safe to delete once WP-16 is done. The `.parts` file is a benign qBittorrent selective-download artifact. **Ingestion gotcha:** `created_utc` is a *string* in old records and a *float* in new ones — any ingest must handle both |
| **Ingested submissions (current)** | `dataset_saves/reddit/submissions-2025-12/submissions.parquet` (586 MB, 48 row groups) | **2008-07-11 → 2025-12-31**, **11,782,247 rows**, 0 dropped | Produced by `experiments/wp16/ingest.py` from the 2025-12 cut, 2026-07-26. Reconciles exactly, and all 188 months match an independent byte-scan of the `.zst`. Preserves the eight legacy columns and adds `post_id, subreddit, num_comments, removed_by_category, over_18` (additive only). `author` + `subreddit` make ADR 0016's $N_{c,t}$ computable by group-by. Report: `experiments/WP-16/report.md` |
| Ingested submissions (superseded) | `dataset_saves/reddit/test_reddit_dataset/` (10 parquet shards, 302 MB) | 2017-02-13 → 2020-03-11, 4,121,376 rows | Written 2025-01-03 by `memes_base/reddit_dataset.py`. **This is what WP-01 sampled from — retain until anything depending on it is re-baselined.** Covered only 36% of the archive: it processed input chunks 0–2 of 10 and then split that partial result into all ten files, leaving no trace of failure. Two further internal gaps (118,314 rows) came from a strict-inequality bug. Fully diagnosed in WP-16 §"Step 1 result"; the ingester was rewritten, not repaired |
| Meme images                     | `experiments/WP-01/**/images/` (2296 files) | pilot only         | **No image corpus exists.** Every row above has an `img_url`, but the images themselves must be fetched over the network — the real cost of scaling, and subject to link rot on older posts |
| Top-40k-subreddits archive      | external (described in `00-background.md`) | TBD                | Not yet ingested                                                                      |
| Previous-iteration artifacts    | `dataset_saves/` (rest of dir)             | ~700 fitted curves | 2023 iteration. Also holds `test_reddit_dataset_embeds.lmbd`, an **LMDB** embed store — relevant to ADR 0015, since LMDB must not live on `/mnt/d` |
| WP-01/02/12 run artifacts       | `experiments/WP-*/` (584 MB, on WSL2)      | —                  | Embedding cache (2296 images), pilot images, grid sweeps, NUTS runs, plots. Gitignored but present; past certifications are now re-inspectable on this machine |
| Hardware                        | 3090 GPU on the Windows desktop (via WSL2) | —                  | Verified working 2026-07-26; sufficient for local embedding/inference at moderate scale |
| Bulk storage                    | `/mnt/d` — Seagate Portable 4 TB           | 3.7 TB free        | Mounted 2026-07-26. Archive/shard tier only — see ADR 0015 and the measured penalties in `03-environment.md` |

The legacy code in this repo (`memes_base/`, `train_script.py`, `test_ocr.py`, etc.) predates this framework. **It should not be assumed correct or representative of current decisions.** Treat it as reference material; reuse code only after reading and understanding it.

## Models

| Model            | Status                | Version / weights                         | Notes                                                |
| ---------------- | --------------------- | ----------------------------------------- | ---------------------------------------------------- |
| Embedding model  | chosen for first pass | OpenCLIP ViT-B-16 + filtered EasyOCR      | ADR 0006 accepted hybrid image/OCR pipeline          |
| Text embeddings  | under comparison      | MiniLM L6 v2, BGE base en v1.5, E5 base v2 | Cached on 1000 examples for semantic clustering work |
| Spread predictor | not yet trained       | —                                         | Architecture deferred until embedding pipeline lands |

## Eval set

Not yet defined. Will be locked in ADR 0011. Default expectation: held-out memes originating after the chosen embedding model's training cutoff.
