# WP-01 — OCR vs. VLM-native embedding benchmark

**Status:** completed
**Date opened:** 2026-05-19
**Owner:** unassigned
**Gates:** —
**Unlocks:** [ADR 0006](../decisions/0006-ocr-vs-vlm.md)

## Goal

Determine whether modern vision-language models embed meme images well enough — including the text-in-image content — to make a dedicated OCR step unnecessary. Produce evidence sufficient to decide ADR 0006 between OCR-pipeline, VLM-native, and hybrid options.

## Definition of done

A markdown report at `experiments/WP-01/report.md` containing:

- Description of the sample set (size, sampling strategy, characteristics).
- For each of the three pipelines below: cluster-quality metrics on the sample, qualitative cluster inspection notes, throughput / cost numbers.
- A recommendation, with reasoning, for which option ADR 0006 should adopt.

And a corresponding update to `0006-ocr-vs-vlm.md` recording the chosen option.

## Inputs / prerequisites

- A representative sample of meme posts. Recommendation: **~500–1000 posts** drawn from r/memes, stratified across:
  - Text-heavy memes (lots of in-image text).
  - Image-heavy memes (little/no in-image text).
  - Mixed/typical memes.
  Stratification helps; exact strategy is at the agent's discretion within this budget.
- Ground truth or proxy ground truth for clustering quality. Hand-label ~20–30 memes as belonging to ~5–10 known meme templates from the sample. This is a small bootstrap of WP-04's gold set — coordinate naming so they merge later.
- Compute: a local GPU is sufficient; no large-scale runs needed at this stage.

## Method

1. **Sample selection.** Draw the sample. Save the sample's post IDs + metadata to `experiments/WP-01/sample.json` (or equivalent — pick a stable format and document it).

2. **Build three embedding pipelines:**
   - **P-OCR:** OCR the image text → embed (OCR text + post text) with a text embedder → embed the image with an image embedder → concatenate.
   - **P-VLM:** Embed the image directly with a modern VLM (e.g., SigLIP-large, Qwen2-VL used as encoder, Florence-2). Embed post text separately, concatenate.
   - **P-VLM-only:** Embed only the image, no post text, no OCR. Useful as an upper-bound on what visual-only embedding can do.

   Document the exact model versions used for each pipeline.

3. **Cluster each pipeline's embeddings** with UMAP → HDBSCAN at a common, reasonable hyperparameter setting. Document the settings.

4. **Score clustering:**
   - On hand-labeled templates: compute purity and recall.
   - Cluster-size distribution (how many singletons vs. nontrivial clusters?).
   - Eyeball 5–10 random clusters per pipeline for coherence.

5. **Record cost/throughput:** seconds per embedding, dollars per million if applicable, model size on disk.

6. **Write the report.** Make a clear recommendation. If results are ambiguous, say so explicitly and propose what additional evidence would resolve it.

## Risks and ambiguities

- "Modern VLM" is a moving target. Pick 1–2 strong candidates; don't try to survey all of them in this WP. A broader survey is WP-03.
- The contamination question (ADR 0004) is **deferred** in this WP — we're judging cluster quality only. The chosen model's training cutoff matters for ADR 0006, but cluster quality is the primary metric here.
- Hand-labeled ground truth is small and noisy at this scale; document the limitation in the report.
- If the OCR pipeline fails on a meaningful fraction of the sample, decide upfront whether to count those as "OCR failed" or to retry with different OCR settings. Either is fine — just be explicit.

## Out of scope

- Survey of all modern embedding models (WP-03).
- Decisions about how to combine modalities once the pipeline is picked (a follow-up ADR after 0006).
- Full-scale embedding of the entire corpus (WP-05).
- Contamination evaluation (orthogonal axis; addressed in 0004 + 0011).

## Log

<!-- Newest at top. -->

- 2026-07-26 — 90-image clustering check (`clustering-large-001`) found contaminated (24 orphan items with silently blanked titles; 40/90 byte-identical duplicates) after user inspection of the rendered clusters. Deduplicated 63-item re-run (`clustering-large-002`) graded by the user: **no significant semantic difference between clusters** — the earlier "coherent clusters" recollection did not survive re-inspection on clean data. Fix in PR #1; postmortem: `research-notes/wp01-clustering-001-postmortem.md`.
- 2026-07-26 — Reinterpreted the follow-up 90-image clustering check. Its weak proxy purity/silhouette was **uninformative, not negative**: an unstratified random draw of 90 posts from a corpus with hundreds of meme families is near-all singletons, so those metrics have almost no within-cluster support to measure — which is why this WP's definition of done asked for 500–1000 *stratified* posts. The coherent "vibes" within the clusters that did form is the informative signal, and is mildly promising for ADR 0006 (user recollection; the rendered cluster examples have not been re-inspected — they are in gitignored `experiments/WP-01/`, Mac-only). Does not change the ADR 0006 outcome. Open follow-up in `open-questions.md` §"Cluster evaluation".
- 2026-05-19 — Completed WP-01. Final run uses 24 cached r/memes images with manual transcripts as labels and `easyocr_filtered:none` OCR predictions as the hybrid embedding text signal. ADR 0006 accepted Option B: VLM image embedding + filtered EasyOCR text embedding.
- 2026-05-19 — Rebuilt top-level WP-01 artifacts from cached real r/memes images: 24-image pilot sample, manual visual transcripts, regenerated hard negatives, and reran OpenAI CLIP B/32 vs OpenCLIP B/16 text-fidelity plus three pipeline clustering. Pilot suggests VLM-native text fidelity is weak, but sample is too small for final ADR acceptance.
- 2026-05-19 — Implemented WP-01 text-fidelity benchmark harness and ran a synthetic Impact-caption smoke test. Results are in `experiments/WP-01/report.md`; real r/memes + manual-transcript run still required before ADR 0006 can be accepted.
- 2026-05-19 — Package opened.

## Outcome

Accepted **Option B — Hybrid** for ADR 0006: OpenCLIP image embeddings plus filtered EasyOCR (`easyocr_filtered:none`) transcript embeddings, with post text embedded alongside OCR text. The 24-image pilot is sufficient to choose the architecture for this project phase, but a larger audit remains needed before production-scale OCR and cluster-quality claims.
