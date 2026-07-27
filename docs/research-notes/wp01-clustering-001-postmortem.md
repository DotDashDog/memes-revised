# Postmortem: the `clustering-large-001` item-set contamination

**Date:** 2026-07-26
**Status:** findings verified at code/data level; corrected run graded by the user — see Outcome
**Artifacts:** `experiments/WP-01/clustering-large-001/` (invalid), `experiments/WP-01/clustering-large-002/` (corrected re-run)

## How it was caught

The user, re-inspecting the rendered 90-image clusters as grader (via the cluster-inspection
artifact page), noticed that posts whose image and OCR text were identical — differing only in
post title — landed in *different* clusters under the text-bearing pipelines (e.g. the
"minister of health belgium" pair), while staying together under image-only. That observation
was correct and led to everything below.

## Findings (all verified, not inferred)

1. **The run's item universe was the `images/` directory, not the manifest.**
   `run_large_clustering.py::load_cached_items` globbed `images/meme-*.jpg`, shuffled, and took
   the first 90. `images/` still held **24 orphan files from an older sample generation** with
   different id numbering. Orphans have no `sample.json` entry, so the metadata lookup
   defaulted their **title and post text to `""` silently**.

2. **40 of the 90 items were byte-identical duplicate pairs** (sha256 of the local file).
   Fifteen pairs are the same post drawn in both sample generations (one member an orphan);
   five pairs have both members in the manifest with different `post_id`s, different image
   hosts, and different strata — i.e. apparently **genuine r/memes reposts**. Byte-identity
   across different hosts is expected because `sample.py` decodes every download and re-encodes
   it with `cv2.imwrite`, so pixel-identical uploads produce identical local files.

3. **Pipeline compositions were being misreported in discussion.** From
   `experiments/wp01/pipelines.py`:
   - `p_ocr` = image embedding ⊕ text embedding of *title + body + OCR transcript* — **this is
     the ADR 0006 accepted hybrid** ("post text signal: embed post title/body alongside OCR text").
   - `p_vlm` = image ⊕ text(*title + body*), no OCR.
   - `p_vlm_only` = image only.

4. **The observed splits were title-side, and mostly a defect artifact.** A blank-titled orphan
   and its titled twin embed different text and can split. Control: across all 20
   identical-file pairs, `p_vlm_only` (image-only) never split a pair; splits occurred only in
   the pipelines that embed the title. The Belgium pair specifically: `meme-00107` embedded
   with title "Yo, that's…", its orphan twin `meme-00067` with a blanked title.

## Consequences

- **`clustering-large-001`'s cluster structure and metrics must not be cited** (its numeric
  metrics were already ruled uninformative at n=90; now the structure itself is contaminated).
- The earlier "clusters that formed were internally coherent" recollection is further
  weakened: exact-duplicate pairs trivially co-cluster and inflate apparent coherence.
- A real design question surfaced, distinct from the defect: **under ADR 0006 the same meme
  reposted under a different title embeds differently**, so clustering identity is
  title-sensitive. This feeds risk R4 (cluster ≠ meme) and the cluster-evaluation open
  question, and may warrant an ADR 0006 amendment (exclude post text from the *clustering*
  embedding while keeping it for the predictor). Not decided here.

## Fix and re-run

`load_cached_items` is now **manifest-driven** (items must exist in `sample.json`; orphans are
skipped and counted) and **deduplicated by file sha256** (first drawn member of each group
kept). The corrected run `clustering-large-002` (same encoder, OCR pipeline, clusterer, seed)
kept **63 items** — 68 manifest-backed images minus 5 repost twins — with 100% metadata
coverage and 0 duplicates. Resulting structure: `p_ocr` 2 clusters + 22% noise, `p_vlm` 3 + 33%,
`p_vlm_only` 4 + 30%. The re-run is rendered for inspection on the same artifact page as the
original; the qualitative verdict is the user's to render.

## Outcome: user verdict on the corrected run (2026-07-26)

The user inspected `clustering-large-002` and found **no significant semantic difference
between the clusters** in any pipeline. The earlier mildly-promising recollection ("clusters
that formed were internally coherent") therefore does **not** survive re-inspection on clean
data — it was at least partly an artifact of exact-duplicate pairs co-clustering. Standing
qualitative support for the ADR 0006 pipeline's *clustering* ability at this scale: none.

Note this does not by itself reopen ADR 0006, which was accepted on text-fidelity and
OCR-quality evidence with cluster-quality evidence explicitly excluded (n=24 clustering was
already inconclusive). What it does do is remove any temptation to cite small-n clustering as
corroboration, and it makes the properly-powered audit — 500–1000 posts, family-stratified,
hand-labeled (open-questions §"Cluster evaluation") — the only path to a cluster-quality claim.

## Method note

This is a second instance of the pattern behind the 2026-06-10 bridge-test postmortem: an
apparently-fine recorded result invalidated only when a human inspected the underlying
artifact rather than the summary. The grader-inspects-the-artifact rule caught it both times.
