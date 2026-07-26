# WP-13 — Temporal-only neural TPP machinery + single-meme stream

**Status:** open
**Date opened:** 2026-06-10
**Owner:** unassigned
**Gates:** [ADR 0014](../decisions/0014-hawkes-track-scope.md) (accepted), WP-12 (for the HawkesN comparison baseline)
**Unlocks:** WP-14, WP-15; produces the per-event cost measurement for the plan-note §5 feasibility gate

## Goal

Validate that our neural-TPP training loop (NLL + compensator) is correct on synthetic Hawkes, then fit a **temporal-only** neural Hawkes to a *single real meme's* post-time stream extracted from the existing archive. No marks. First contact with real, corrupted timestamps and the source of the §5 cost extrapolation.

## How a "single meme" is defined here — and why this does NOT need clustering

This is the crux flagged in the plan note (§1.4) and worth stating plainly, because it differs from the project's eventual individuation story:

- **This phase does not solve meme individuation; it sidesteps it for one chosen meme.** We pick a single, well-known meme and pull *its* posts out of the archive using a cheap, reliable external label — **not** an embedding cluster. Acceptable labels, in order of preference:
  1. **Exact/near-duplicate image identity** (MD5 of the image bytes for exact reposts; a perceptual hash — pHash/dHash — to catch re-encodes and minor crops). r/memes is repost-heavy, so a single template image yields a clean, high-precision event stream with no semantic model at all.
  2. **Distinctive catchphrase / title keyword** match for memes carried by text.
  3. A KnowYourMeme-dated template as a cross-check on the date range.
- **Why this is legitimate:** the goal is to validate *training machinery* on a real stream, which needs one clean labeled stream by *any* means — it does not need a method that generalizes to all memes. Using image-hash dedup is a high-precision, low-recall individuation that is fine precisely because we only need one meme.
- **Relationship to the end-state vision:** in the holistic STPP (WP-15), per-meme structure is meant to emerge from the spatial kernel $s_\theta$ slicing the continuous field — *replacing* HDBSCAN, not using it. Whether that slicing is actually well-posed is the open question in plan-note §1.4, tested in WP-15. WP-13 deliberately brackets that question by hand-labeling one stream; it must not be read as evidence that per-meme individuation is solved.

## Definition of done

A report at `experiments/WP-13/report.md` with:
1. **Synthetic validation.** Reproduce a small EasyTPP result (e.g. NHP or THP on its synthetic Hawkes set) to confirm environment/understanding. Then show our minimal **JAX** neural-Hawkes recovers the known branching ratio on synthetic Hawkes, matching EasyTPP within tolerance.
2. **Single-meme fit.** For one externally-labeled meme stream: temporal-only neural Hawkes vs. parametric HawkesN (WP-12) vs. rigor §1 baselines (B1 global mean, B2 date-only; B4 k-NN and B5 not applicable here). Held-out *future* log-likelihood and next-event-time error.
3. **Cost measurement.** Per-event training and eval cost (wall-clock and FLOP estimate) on the local 3090, with and without temporal windowing — the input to the §5 5M+ extrapolation.
4. **Archive format verification.** A one-paragraph note confirming the `memes_submissions.zst` schema (fields used: `id`, `created_utc`, `title`, `url`, `score`) and the dedup/label procedure used.

## Inputs / prerequisites

- ADR 0014 accepted; WP-12 (HawkesN baseline).
- `reddit_data/reddit/subreddits23/memes_submissions.zst` (Pushshift NDJSON; per-post `created_utc` present).
- EasyTPP (PyTorch, reference only) + JAX core (ours).

## Method

1. Stand up EasyTPP, reproduce one synthetic-Hawkes number.
2. Implement the minimal JAX neural Hawkes (continuous-time decay state à la Mei-Eisner, or a small causal attention encoder); validate branching-ratio recovery on synthetic data against EasyTPP.
3. Build the single-meme stream via image-hash (preferred) or keyword label; verify by eyeballing 10–20 matched posts (`method-discipline.md` §5: look before metrics).
4. Fit and compare models on a time-split (train on early stream, evaluate on held-out future), report future-NLL and next-event-time error with bootstrap CIs (rigor §5).
5. Measure per-event cost; tabulate windowed vs. full.

## Design constraint — keep the single-stream fitter reusable (Branch A)

Per ADR 0014, the neural and parametric fitters built here should keep the same standalone `(event_times[, marks]) → fitted params + uncertainty` interface as WP-12, so the committed WP-15 Branch-A fallback (redirect to a per-cluster fitter) is free. Don't couple the single-stream fit to any holistic/field code.

## Risks and ambiguities

- **Label noise:** image-hash misses heavily-edited variants (low recall) and may catch unrelated reuse (rare false positives). Report precision on the hand-checked sample; recall loss biases the stream toward "clean" reposts — note it.
- **One meme is anecdote, not evidence.** This validates machinery; do not generalize spread conclusions from it.
- Reposts inflate self-excitation (plan note §1.5) even in one stream — flag the fitted branching ratio as upward-biased.

## Out of scope

- Marks / embeddings (WP-15).
- Solving general meme individuation / clustering (that is the WP-05/WP-06 pipeline, or the implicit-kernel question in WP-15).
- Any multi-meme or full-corpus run (gated by §5).

## Log

- 2026-06-10 — Drafted from `research-notes/hawkes-replication-plan.md` Phase 2. Includes the explicit "single meme without clustering" rationale in response to the user's question.

## Outcome

<pending>
