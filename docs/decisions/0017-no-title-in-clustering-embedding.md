# 0017 — Exclude post title/body from the clustering embedding

**Status:** accepted
**Date:** 2026-07-26
**Accepted:** 2026-07-26 — user, citing r/memes posting norms ("titles don't belong in the embedding")
**Amends:** [0006-ocr-vs-vlm.md](0006-ocr-vs-vlm.md) (scopes its "post text signal" out of the clustering path)
**Informs:** WP-04 (audit pipeline), WP-05 (pipeline build), 0009 (clustering), WP-10 (predictor inputs)

## Context

ADR 0006's accepted hybrid embeds the post title/body alongside the OCR transcript and the
image. That composition was accepted for "the project embedding pipeline" without
distinguishing two different jobs the embedding does:

1. **Meme identity** (clustering): which posts are instances of the same meme?
2. **Content representation** (prediction): what is this meme about, for the
   embedding → spread-parameter model?

The `clustering-large-001` postmortem (`../research-notes/wp01-clustering-001-postmortem.md`)
verified a concrete failure of using title text for job 1: the same image with the same OCR
text lands in different clusters when the title differs. Reposts under new titles — the normal
mode of meme circulation on Reddit — therefore fragment across clusters, directly aggravating
risk R4 (cluster ≠ meme).

Posting norms on r/memes make the title a poor identity signal in principle, not just in this
sample: the title is the poster's aside, not part of the meme content. Measured on the WP-01
120-item manifest: median title length 13 characters, 41% are ≤10 characters (e.g. "Kek",
"Yeet", emoji), and **0 of 120 posts have any body text** — the "title/body channel" is in
practice a short-quip channel. Titles also carry author-side, virality-adjacent signal
(clickbait phrasing), which leaks non-content information into meme identity.

## Decision

The **clustering / meme-identity embedding is image ⊕ OCR-transcript only.** Post title and
body do not enter it. In WP-01 pipeline terms: the audited and productionized clustering
pipeline is *image + OCR text*, a composition that (as of `clustering-large-002`) does not
exist in `experiments/wp01/pipelines.py` and must be added; `p_ocr`/`p_vlm` (which embed the
title) are not to be used for clustering.

Title/body remain available as a **separate, optional feature channel for the predictor**
(WP-10), where "does the poster's framing add predictive signal beyond content?" is a
legitimate ablation — alongside the existing comments-as-signal-vs-content open question.
No decision on predictor inputs is made here.

## Alternatives considered

- **Keep title in the clustering embedding (status quo).** Rejected: verified repost
  fragmentation; identity should be invariant to reposting, and the title is exactly the part
  that varies across reposts of the same meme.
- **Downweight the title (scaled concatenation).** Adds a tuning knob with no principled
  setting; the failure mode shrinks but does not vanish.
- **Separate title embedding, cluster on joint space.** Same objection, more machinery.
- **Exclude title everywhere, including the predictor.** Premature — poster framing may carry
  real predictive signal; that is an ablation for WP-10, not a decision to make now.

## What changes if this is accepted

- WP-04's audit targets the image ⊕ OCR pipeline (plus image-only as comparison).
- WP-05 builds that composition; `pipelines.py` gains it.
- No recorded result is invalidated: the only clustering runs that embedded titles are the
  voided `clustering-large-001` and the graded-uninformative `clustering-large-002`.

## Evidence trail

- `../research-notes/wp01-clustering-001-postmortem.md` — verified title-side cluster splits.
- User verdict 2026-07-26: no significant semantic distinction between clusters in any
  pipeline of the clean 63-item run; user judgment that r/memes posting norms put titles
  outside the meme content.
- Title statistics above (WP-01 `sample.json`, n=120).
