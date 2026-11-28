# WP-04 — Gold-set construction and powered clustering audit

**Status:** active
**Date opened:** 2026-07-26
**Owner:** unassigned
**Gates:** ADR 0017 (accepted 2026-07-26); ADR 0015 storage protocol for the fetch; image access for the sampled posts (WP-17 shards if available, else the self-contained fetch in Method §1)
**Unlocks:** ADR 0009 (clustering algorithm/granularity), WP-06 validation; replaces the "larger OCR/cluster audit" follow-up left open by WP-01

This promotes the WP-04 stub ("~50 hand-labeled memes") to a design that can actually answer
the question. The 50-post scale was never powered: the `clustering-large-001/002` episode
showed that an unstratified draw yields near-all singleton families, whose metrics are
uninformative for any pipeline (see `../research-notes/wp01-clustering-001-postmortem.md` and
`../open-questions.md` §"Cluster evaluation").

## Goal

Produce a hand-labeled gold set of meme *families* with enough within-family support that
cluster-quality metrics mean something, and use it to render the first defensible verdict on
whether the embedding pipeline's clusters correspond to human meme families — and at what
granularity.

**The end goal is meaning-based clustering, not image-appearance clustering** (user emphasis,
2026-07-26). Appearance similarity (dhash) appears in this package only as scaffolding — for
the power analysis and for labeling efficiency — never as the identity criterion. The audit's
two-layer labels (Method §4) exist precisely to measure how much of the embedding's clustering
is mere template matching versus meaning.

## Definition of done

A report in `experiments/WP-04/` containing: (a) the measured family-size distribution and the
sample-size derivation; (b) the labeled gold set (post ids + family labels + labeling
protocol); (c) pre-registered metrics and thresholds, surfaced for approval *before* the
scored run; (d) per-pipeline results with the required plots; (e) a verdict rendered by the
user from the plots, recorded in this file's Outcome. The gold set itself is a durable
artifact for ADR 0009 and any future embedding change. Additionally: (f) a **throughput
engineering note** (`../research-notes/image-pipeline-throughput.md`) recording measured
download, OCR, and embedding rates, the settings that achieved them, and every rate-limit or
failure mode encountered — this is required input for WP-17's full-corpus fetch and WP-05's
full-corpus embedding, per user request 2026-07-26.

## Inputs / prerequisites

- `dataset_saves/reddit/submissions-2025-12/submissions.parquet` (WP-16; has `img_url`, `post_id`, timestamps).
- `experiments/wp01/dhash.py` (perceptual hash), the OCR pipeline of ADR 0006, OpenCLIP backend.
- Image fetching for ~30–60k posts (Method §1) — i.redd.it covers 88% of image rows, so link
  rot is a bounded nuisance; record `fetch_status` per ADR 0015's requirement.

## Method

**1. Pool fetch.** Draw a uniform random pool of N₀ ≈ 30–60k image posts from a fixed,
pre-registered time window (e.g. 2018–2020, where volume is high and links mostly live).
Fetch images, byte-dedup (sha256), record fetch failures. If WP-17 shards already cover the
window, use them instead of fetching.

*Fetch discipline (this doubles as WP-17's dry run):*
- **Storage per ADR 0015:** write uncompressed tar shards + parquet index with `fetch_status`
  directly under `/mnt/d/memes/` (≥16 MB write buffers, immutable time-partitioned shards);
  the working set (labeling thumbnails, embedding cache) lives on the NVMe. No loose-file
  corpus, no LMDB/sqlite on `/mnt/d`.
- **Politeness without slowness:** async fetcher with a per-host token bucket — start
  ~8–16 concurrent connections to `i.redd.it` (a CDN; 88% of image rows) with keep-alive
  reuse, jittered pacing, honest static User-Agent, exponential backoff with full stop on
  any 429/403 burst. Non-direct hosts (imgur page URLs etc.) get a separate low-concurrency
  lane or are resolved to direct-image URLs first. Ramp concurrency up gradually while
  watching error rates rather than starting at the ceiling.
- **Resumable by construction:** the shard index is the ledger; a restart re-reads it and
  continues. Every attempt (success, 404, timeout, 429) lands in `fetch_status` — dead links
  are data (R2 selection), not noise.
- **Measure everything:** achieved req/s and MB/s over time, error-rate curves vs.
  concurrency, backoff events. These numbers go into the deliverable-(f) throughput note.

**2. Family-size distribution, then sample-size derivation.** Build a dhash near-duplicate
graph over the pool (exact + small-Hamming-radius matches); connected components approximate
*template families*. This yields the empirical family-size distribution — the quantity the
open question says must drive the sample size. Derive n from it: choose n so that the
expected number of families with ≥3 labeled members supports the target confidence interval
on pairwise F1 (pre-register the CI width; the derivation, not a feel-picked n, is the
deliverable). The 500–1000-post figure from WP-01's definition of done is the prior guess to
be checked, not the answer.

**3. Two-stratum sample.**
- **Family stratum:** sample seed posts, then include their dhash siblings — every drawn meme
  brings its family, giving within-cluster support.
- **Background stratum:** a uniform random draw, to measure false-merge behavior and keep a
  window into the corpus's true (mostly-singleton) cluster-size distribution.
Report every metric per stratum as well as pooled; the family stratum's size-distribution
bias is thereby quarantined, answering open-questions §"Cluster evaluation" point (b).

**4. Blind hand-labeling — two label layers.** Each item gets:
- **`template_id`** (appearance lineage): same base image/format. Seeded by dhash components,
  confirmed by eye; exact byte-duplicates are auto-collapsed and labeled once.
- **`family_id`** (meaning): "these posts are instances of the same meme *as a spread
  phenomenon*" — the labeler's semantic judgment, the layer the project actually cares about.
  Same joke migrating across different templates may be one family; the same template
  carrying unrelated jokes may be several.

Scoring clusters against *both* layers separates "the embedding matches meaning" from "the
embedding matches pixels" — the difference between the two scores is the measured
appearance-vs-meaning gap, directly addressing the user's end-goal emphasis and R4's
granularity question.

Blindness protocol: labeling happens from the image + OCR text only (no titles — excluded
from the identity signal per ADR 0017; no cluster assignments or embedding-similarity
suggestions visible). Items arrive in random order via a local keyboard-driven labeling UI
(single-page app + tiny localhost server writing append-only JSONL on the NVMe — crash-safe,
resumable); the labeler assigns to an existing family via a searchable catalog (thumbnail +
OCR substring search — labeler-driven recall, not model suggestions) or creates a new one;
`unique` is an explicit label, not a default. The protocol text is frozen before labeling
starts; a second-pass self-agreement check on a relabeled 10% subsample bounds label noise,
reported per layer.

**5. Pipelines under audit.** Image ⊕ OCR (the ADR 0017 clustering composition — to be added
to `pipelines.py`) and image-only as the comparison arm. Sentence-embedding variants of the
OCR channel (the cached MiniLM/BGE/E5 work) may ride along as secondary arms. OCR and
embedding run on the 3090 with proper batching (batch EasyOCR's detector/recognizer and the
OpenCLIP encoders; measure images/s at several batch sizes) — throughput numbers and settings
go into the deliverable-(f) note alongside the fetch measurements.

**6. Metrics, baseline, and pre-registered thresholds.** Primary: B-cubed precision/recall/F1
against the **meaning layer (`family_id`)** — the project's actual target — with the same
metrics against `template_id` reported alongside to expose the appearance-vs-meaning gap; ARI
as secondary. **Baseline to
beat: dhash connected components alone.** That baseline is principled: if semantic embeddings
cannot outperform a perceptual hash at grouping meme families, the embedding adds nothing to
clustering and ADR 0009 should reconsider its options. Thresholds (how much better than
dhash, at what CI) are surfaced for approval before the scored run, per
`method-discipline.md` — this file's Log records them when set.

**7. Plots before verdicts.** Required: (a) cluster–family contingency heatmap per pipeline;
(b) UMAP of the gold set colored by gold family vs. colored by cluster, side by side; (c)
B-cubed P/R across HDBSCAN granularity settings (the R4 granularity question made visible).
Expected appearance under success and failure written down before results are seen. The user
grades from the plots; no self-certified pass (AGENTS.md).

## Risks and ambiguities

- **dhash-seeded families bias toward near-duplicate (template-identical) families**, the
  easiest cases for an image embedder. Mitigation: the background stratum, plus hard-case
  labeling in §4; report the two strata separately and treat family-stratum F1 as an upper
  bound.
- **Label circularity:** labeling from image+OCR uses the same modalities the pipeline embeds.
  Unavoidable without external meme-family metadata; record it as a known limit. (Titles being
  excluded from both is a consequence of ADR 0017, not an extra assumption.)
- **Link rot in the chosen window** shifts the family-size distribution (old families lose
  members). Record fetch-failure rates per family size; if failures correlate with family
  size, flag before deriving n.
- **Sample-size derivation may demand more labeling than is feasible.** If n exceeds ~1.5k
  posts, stop and bring the trade-off to the user rather than quietly shrinking the CI.

## Out of scope

- Choosing the production clustering algorithm/granularity (ADR 0009 decides, fed by this).
- The full image-corpus fetch (WP-17) and any predictor-side question (WP-10), including
  whether titles help prediction.
- Multi-subreddit generalization (ADR 0016's community-shortcut audit is separate).

## Log

<!-- Newest at top. -->

- 2026-11-28 — **Local description models tested (follow-up to the diagnostic; author-only,
  not independently reviewed).** Question: can a vision-language model that fits on the
  3090 write "what is this meme about" descriptions good enough, fast enough? Files:
  `experiments/WP-19/` (`vlm_report.md`, `descriptions_*.json`, `timing_*.json`,
  `description_eval.json`, `description_followup.json`, `setup.md`); code
  `experiments/wp19/`; pass/fail expectations written beforehand in the docstring of
  `evaluate_descriptions.py`. Models: Qwen3-VL 8B (full precision, and with weights
  quantised to 8-bit at load) and 4B, served with vLLM in an isolated environment.
  *Speed (lower bounds: about 15 of 16 CPU cores were busy with other jobs):* 8B 2.3
  images/s, 8B 8-bit 3.5, 4B 5.0 (5–9 across runs); 51, 33 and 23 GPU-days for ten million
  images on the one card. For comparison the current text-extraction step measured 1–2
  images/s twice on 2026-11-27. Usable descriptions after one retry: 790, 791 and 775 of
  797.
  *Quality, against the user's 99 pair answers (the only reference independent of any
  description; AUC, 1 = perfect):* accepted pipeline 0.68; extracted text with a sentence
  model 0.77; 4B 0.83; 8B 0.86; 8B plus its own transcription 0.88; Opus descriptions 0.92.
  Paired differences: 8B minus extracted text +0.09 (0.02 to 0.17); Opus minus 8B +0.06
  (−0.00 to 0.12).
  *Quality, against the agent labels (which were built from the Opus descriptions and so
  favour them):* share of an image's family kept in its cluster at 90% purity: pipeline
  0.21, extracted text with a sentence model 0.28, 8B 0.28, 8B plus transcription 0.30,
  Opus 0.49. Appending the image embedding to any description lowers every score.
  *Reading:* by the bar set beforehand this is not a clear pass. The 8B model is clearly
  better than extracted text on the user's answers and not clearly worse than Opus there,
  but ties extracted text on the agent labels. The two references disagree and the user
  reference has only 99 pairs. 8-bit quantisation costs nothing measurable. Even the Opus
  descriptions keep under half of each family together at 90% purity, so descriptions
  improve the representation without making a clean partition possible.
  *Why the local model trails (from reading outputs):* inconsistent names for the same
  referent ("Area 51", "raid" versus "Area 51 raid": named in 18 of 23 posts against 23 of
  23), missed implied jokes, stray referents; and the Opus agents were told the subreddit
  and year while the local models were not, which is a prompt difference, not a model
  limit. *Contamination screen:* popularity-related words in 13 of 797 descriptions (8B)
  and 12 of 797 (Opus). *Not tested:* memes newer than the models' training data; prompt
  variants; a second set of user pairs.

- 2026-11-27 — **Diagnostic: why clustering fails on meaning (exploratory, author-only).**
  Scripts `experiments/wp04/diagnose_failure.py`, `diagnose_vs_user.py`,
  `explore_temporal_shape.py`; outputs `sample-001/diagnose_failure.json`,
  `user_check/representation_vs_user.json`, plots `diagnose_failure.png`,
  `temporal_shape.png`. Against agent family labels (unvalidated on hard pairs):
  (1) *Picture outranks meaning.* In the image-only embedding a different-meme, same-template
  pair is more similar than a same-meme, different-template pair 89% of the time; 72% in the
  accepted image-plus-text embedding; 37–42% for text alone. 661 of 745 same-meme pairs are
  cross-template, so this is the main case.
  (2) *No absolute scale.* The similarity of an image to its closest family member is
  indistinguishable from the similarity of a unique image to its closest neighbour: a
  threshold separates them 52% of the time for image-only (medians 0.554 and 0.553), 63% for
  the pipeline, 68% at best. Ranking works (closest neighbour is same-family for 51–64% of
  images); a global cut-off does not.
  (3) *The text encoder is the weak part of the text channel, not text extraction.* Same
  extracted text: CLIP's text encoder gives pair AUC 0.77, a sentence-embedding model
  (bge-base) 0.86; clean agent transcriptions with the same sentence model 0.88. Adding the
  image half to sentence-model text lowers meaning AUC (0.86 to 0.85) and restores picture
  dominance.
  (4) *Against the user's own 99 answers* (independent of any description, but pairs were
  chosen in strata defined by agent labels and are enriched in hard cases): image-only AUC
  0.58 (0.46–0.69), pipeline 0.68 (0.57–0.78), sentence-model text 0.77 (0.67–0.86), agent
  description of what the joke refers to plus a one-line premise 0.94 (0.88–0.98).
  (5) *Timing:* templates are mostly in use all year (middle 80% of posts spans a median of
  168 days for the 52 largest hash groups); event-driven families are bursts (19–70 days);
  catchphrase families are year-round.
  Reading: mostly a geometry problem (the embedding ranks neighbours usefully but distance
  is dominated by the picture and has no common scale), partly an information problem (the
  CLIP text encoder, and caption wording is not the same as what a meme is about), and not
  an algorithm problem (five methods agreed, per the review).

- 2026-11-27 — **Independent review of the scored run (fresh agent, given artifacts and not
  the conclusion): numbers reproduce, two of my readings were wrong, one pipeline bug
  found.** Full text: `experiments/WP-04/sample-001/review/review.md`. Status of the scored
  run is now `agent-reviewed`, with the family layer still unvalidated by the user check.
  *Reproduced:* all 24 primary values to four decimals with separate code; row alignment
  correct; a finer sweep moves numbers by at most 0.012; other linkages, a mutual-neighbour
  graph and other image-to-text weightings all give 0.34–0.39, so the result is not an
  artefact of the clustering method.
  *Corrected reading 1:* "text adds nothing" is true only at the 0.90 operating point. At
  precision 0.70 image plus text beats image only by 0.06 (0.02–0.10) pooled and 0.07
  (0.02–0.14) in the background, and by more at 0.50; the 2-D map shows the same. At 0.90,
  78% of the precision loss falls on images labelled unique, so a few false merges set the
  number. The plot disagreed with my summary; the plot was right.
  *Corrected reading 2:* "as good as a perceptual hash on templates" holds for image only
  and is false for image plus text, the accepted composition (0.56 against 0.82). Plot title
  fixed.
  *Bug:* `embed_pipeline` in `experiments/wp01/pipelines.py` encodes an image with no
  extracted text as the literal string "[empty]", so the 29 such images share an identical
  text half and 17 unrelated ones form one cluster. Zeroing the text half for them lifts
  image plus text from 0.37 to 0.39 pooled and 0.28 to 0.32 in the background. At corpus
  scale every text-free image would fall into one cluster. Not yet fixed in the pipeline.
  *Metric:* B-cubed recall counts an image as together with itself, which is why leaving
  everything alone scores 0.24. Counting only pairs of different images, the share of
  same-family pairs kept together at 90% purity is 0.21 pooled and 0.11 in the background
  for both embedding arms, 0.09 and 0.04 for the hash, 0 for no clustering.
  *Sample:* 79 of the 103 grouped background images have a family member in the hash-seeded
  stratum; clustering the 500 background images alone leaves 23 multi-member families,
  which is the registered "not enough families to tell" outcome for comparing arms there.
  *Labels:* fill-in formats are treated inconsistently (two are split across families while
  "me and the boys" is one), which charges the clustering with false merges.
  *What stands:* at 90% purity one global threshold keeps few meaning-level families
  together; pooled, the embedding beats the hash (+0.08, 0.04–0.12); the embedding does
  carry meaning (nearest neighbour from the same family 43% of the time even excluding
  same-template neighbours, text alone 45%, chance 1%). *What does not:* that text adds
  nothing, that the embedding misses meaning, or any ranking of arms in the background.

- 2026-11-27 — **User pair check done (100 pairs): template labels validated, family
  (meaning) labels fail the pre-set criterion.** Files: `experiments/WP-04/sample-001/user_check/`
  (`user_agreement.json`, `pairs_scored.csv`). Against the primary label set:
  *Template layer:* the user agreed on 41 of 41 pairs the agents call same (95% interval
  91–100%) and 15 of 15 hard negatives (80–100%); passes. 26 of the 100 pairs are
  near-identical images (hash distance ≤ 4), which the user also pointed out; without them
  the same-pair agreement is 15 of 15.
  *Family layer:* the user agreed on 37 of 40 pairs the agents call same (93%, interval
  80–97%; passes part a) but on only 31 of 39 hard negatives the agents call different (79%,
  interval 65–89%; fails part b, which needed 85% with the interval above 75%). All 20 random
  pairs agreed. Per the protocol the family-layer scores are therefore **unvalidated**.
  *What the disagreements are* (contact sheets in `user_check/disagree/`): in 8 pairs the
  user says same meme where the agents say different, and these are mostly posts sharing a
  franchise or subject (four are Avengers: Endgame / Infinity War jokes with different
  premises, one is two "dead memes" posts, one is two cake-day posts, one is the same
  Doctor Strange template with unrelated captions); one is an exact repost the primary
  labeller missed (the second labeller caught it). In 3 pairs the user says different where
  the agents say same: two are fill-in-the-caption formats the agents treated as one meme
  ("ah yes, enslaved X"; TF2 "The Engineer"), and one is a pair of identical reposts, which
  may be a mis-click. So the user's sense of "same meme" is broader than the protocol's
  (shared subject counts) and does not accept the protocol's template-as-joke exception.
  The label noise is not random: it sits exactly on the definition of the unit. That
  definition is a project-level decision now queued for the user in `docs/02-state.md`.
- 2026-11-27 — **Scored run (author-only until reviewed).** Labels: two independent agent
  resolutions of all 797 images agree on 98% of same-template pairs and 92–94% of
  same-family pairs (`stage2/labeller_agreement.json`); 54 families of two or more cover 223
  images, 113 templates cover 376. Primary number (share of each family kept together at 90%
  cluster purity, family layer): image plus text 0.37 (95% interval 0.28–0.49), image only
  0.37 (0.28–0.51), perceptual hash 0.29 (0.22–0.40), every image alone 0.24 (0.19–0.31).
  Background stratum: image plus text minus hash = 0.04 (−0.01 to 0.10). Template layer:
  image plus text 0.56, image only 0.77, hash 0.82. Same pattern with the second label set
  and with uncertain labels removed (`scores*.json`). This matches the failure description
  registered above: within the interval of the hash in the background stratum, and text adds
  nothing at that operating point. Not registered in advance, so exploratory: an image's
  nearest neighbour is in its own family for 59% (image plus text) and 51% (image only) of
  images that have a family, against under 1% by chance, and at looser purity the
  image-plus-text curve is above image-only (`plots/precision_recall.png`,
  `neighbours.json`). Also measured: text extraction ran at about 2 images/s and embedding
  at 10–13 images/s while the GPU was shared, far too slow for a full corpus; to be
  re-measured.

- 2026-11-27 — **Scoring plan and expectations, written while labelling is in progress and
  before any label or cluster has been compared.**
  *Claim under test:* clustering the embedding recovers meme families as judged by meaning,
  beyond what matching pictures already gives.
  *Arms:* image plus extracted text (the accepted composition) and image only, each
  clustered across a sweep from fine to coarse (agglomerative, average linkage on cosine
  distance, threshold swept; HDBSCAN across `min_cluster_size` shown alongside). Reference
  points: every item alone, and perceptual-hash groups at radii 0 to 10.
  *Primary number, fixed now:* on the family (meaning) layer, the share of same-family pairs
  kept together (B-cubed recall over items whose gold family has two or more members) at the
  sweep setting where B-cubed precision is 0.90, with a 95% interval from resampling gold
  families. Reported pooled and for the background stratum alone. Precision 0.90 is an
  operating point, not a pass mark: each cluster will later be treated as one meme's spread
  curve, so roughly one foreign post in ten is the most contamination worth considering; the
  whole precision-recall curve is plotted so the reader is not tied to it.
  *What success looks like:* in the background stratum the embedding's curve sits above the
  hash curve by more than the interval; image plus text beats image only on the family layer;
  and the embedding does better against family labels than a pure template matcher would.
  In the contingency plot, gold families map onto single clusters.
  *What failure looks like:* the embedding's family-layer curve is within the interval of the
  hash curve; or its clusters line up with templates and split or mix families (high score
  against template labels, low against family labels); or adding text makes no difference.
  *What this cannot detect:* families the labellers missed entirely (guarded only by the
  user's random-pair check); performance on the 42% of 2019 posts whose images are gone;
  anything about other years. With an unknown and probably small number of multi-member
  meaning families in 797 items, the interval may be too wide to separate the arms; that
  would be reported as "not enough families to tell", not as a tie.

- 2026-11-27 — **Work resumed; labelling plan and sample design changed before any label
  existed.** (1) *Who labels:* the user decided agents label the whole gold set and the user
  validates a subsample as same/different pairs. The frozen protocol, including the
  agreement criterion fixed in advance, is `experiments/wp04/labeling_protocol.md`; it
  replaces Method §4's labeller and the catalogue UI, not its two label layers or its
  blindness rules. (2) *Pool:* the 50,000-post fetch finished in July: 26,134 images
  retrieved (26,096 distinct), 21,020 links dead (42%), 2,794 not direct images. (3) *Hash
  radius = 4,* chosen by eye from contact sheets (`experiments/WP-04/radius-check/`): the
  groups that appear at radius 4 are all the same base image with different captions; at
  radius 6 one of eight sampled new groups merged unrelated images and a junk group of
  near-blank images doubled. The groups are templates, not reposts, which is what the
  template layer needs. (4) *The July sample-size simulation answered the wrong question.*
  `derive_n.py` reported a ±0.03 interval at 300 items, but its simulated "poor" clusterer
  still scored 0.89: with 91% of posts in no group, the pooled score is carried by
  singletons that are trivially correct, the same failure as the 90-image check. The
  informative quantity is how well multi-member families are kept together, and its
  precision depends on the number of such families in the sample (uniform seeding yields
  about 9 per 100 seeds). (5) *Sample frozen* (`experiments/WP-04/sample-001`, seed 42): 100
  seeds drawn one per multi-member hash group with groups equally likely, each bringing up
  to 6 members (297 images), plus 500 uniform background images; 797 in total. How many
  meaning-level families the background holds cannot be known before labelling; the
  interval actually achieved will be reported, and if it is too wide the sample will be
  extended on the basis of family counts alone, before any clustering score is looked at.
  (6) *Scoring changes that follow:* precision and recall are reported separately, recall
  over items in gold families of two or more, each per stratum, next to two reference
  points: every item alone, and hash groups. Hash groups are a fair baseline only in the
  background stratum, since the family stratum was built from them.

- 2026-07-27 — **Pre-registered pool window (Method §1), before any fetching:** calendar year
  **2019 UTC** (`created_utc ∈ [1546300800, 1577836800)`), uniform draw of **N₀ = 50,000**
  image posts from all hosts (host recorded; restricting to i.redd.it would bake host
  selection into the pool definition), seed 42, dedup by `post_id`. Rationale from the WP-16
  parquet: 2019 has 2.96M image posts (2nd-highest year), 93% i.redd.it, and only 8% flagged
  `removed_by_category` — vs 58% in 2020 and 51% in 2021, whose removal-correlated link rot
  would distort the family-size distribution. The 2018–2020 range in Method §1 was an
  example, not a registration; this entry is the registration.
- 2026-07-26 — Package drafted, replacing the ~50-meme stub, from the `clustering-large-001`
  postmortem fallout. Gated on ADR 0017 acceptance.

## Outcome

<!-- Filled in when done. -->
