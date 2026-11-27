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
