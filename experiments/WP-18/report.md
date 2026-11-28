# WP-18 report: the three measurements that gate the multi-subreddit corpus decision (ADR 0016)

Status of everything below: **UNCERTIFIED, checked only by its author** (the third agent on this
task, 2026-11-27). No independent reviewer has looked at it yet. ADR 0016 is still `proposed`; this
report does not decide it.

## Summary

- **Unique posting accounts per community-month** (the proposed population denominator): never near
  zero in the long-running communities, but it fails two of three pre-registered tests. The share
  of posts with an uncountable ("[deleted]") author drops from 30-38% to about 1% in mid-2023 in all
  seven communities at once (an archive artefact), and there are moderation-driven steps (r/comedyheaven
  x3 in May 2024; r/196 closed July 2023). The denominator as specified would carry these artefacts.
- **Cross-posting**: 8.3% (95% CI 7.7-8.9%) of downloadable image posts have a hash match in another
  of the seven communities, against 0.4% by chance; about 6.5% (5.6-7.1%) are the same meme on
  inspection. Concentrated in r/dankmemes (27.6%, mostly shared with r/memes) and mostly one account
  posting the same file twice. Naive pooling would double-count; the "same image in two communities"
  idea has enough pairs only for r/memes-r/dankmemes.
- **Community shortcut**: a linear classifier names the community of a held-out meme from its
  embedding 50.7% of the time (49.1-52.4%) against 14.3% chance, beating file format (33.2%) and
  posting time (14.9%); 44.8% six-way without the GIF-heavy r/whenthe (chance 16.7%). Long-running vs
  newer communities is 69% separable. Community is readable from content, as the ADR feared; r/memes
  and r/dankmemes are the exception.
- The six extra archives were verified complete (single zstd frame, line counts equal row counts).
- Everything here is UNCERTIFIED (author only). The decision on ADR 0016 is the user's.

Background in one paragraph. The project predicts how memes spread on Reddit. Its data so far is
r/memes alone, which has shrunk about 98% since 2020. ADR 0016 proposes widening the corpus to
seven meme communities: a "long-running set" present since before 2020 (r/memes, r/dankmemes,
r/meirl, r/comedyheaven) and a "newer set" (r/whenthe, r/196, r/shitposting). It asks for three
measurements first: (1) is the count of unique posting accounts per community per month usable as
the population size that spread curves are divided by; (2) how often the same image is posted in
more than one of these communities; (3) can a classifier tell which community a meme came from
using the image embedding the project uses, which would make "community" a shortcut a spread
model could exploit. What was expected, and what would count against the ADR, was written down
before each measurement in `experiments/WP-18/expectations.md`.

---

## Measurement 1: unique posting accounts per community per month

**Finding.** The count never gets close to zero in the long-running set (lowest month: 334
accounts, r/comedyheaven, Dec 2023), so the denominator does not blow up there. But it fails two of
the three pre-registered tests:

1. **The share of posts whose author cannot be counted drops by about 30 percentage points in one
   step in mid-2023, in all seven communities at once.** Before June 2023, 30-38% of posts in each
   long-running community have author "[deleted]" (Jan-May 2023: r/memes 30.3%, r/dankmemes 38.2%,
   r/meirl 35.2%, r/comedyheaven 34.3%); after it, about 1% (Aug-Dec 2023: 0.7-1.4%). r/196 goes
   14.4% -> 1.3%. Pre-registered failure line: a swing of more than 10 points. The swing is
   simultaneous across communities, which points to a change in how the archive was collected (posts
   captured soon after posting instead of re-fetched later, after accounts had been deleted), not
   to user behaviour. Consequence: before mid-2023 the named-account count misses the authors of a
   third of posts, after it almost none, so the denominator has a built-in step of up to roughly
   1.4-2.2x (upper bound per community; the hidden authors may overlap with named ones) at exactly
   where the ADR's evaluation window begins. Plot: `m1_deleted_author_share.png`.
2. **There are month-on-month jumps of 2x or more that are not a community starting up.** Long-running
   set: r/comedyheaven triples in May 2024 (464 -> 1,393 accounts). Newer set: r/196 has 2 accounts
   in July 2023 (the community was closed that month; 6,100 in June, 9,429 in August). Below the
   2x line but visible: r/memes falls 44% in Feb 2025 (12,363 -> 6,937) and r/dankmemes rises 73%
   in June 2023.
3. Near-zero months (pre-registered line: under 100 accounts inside a community's active life):
   none in the long-running set; one in the newer set (r/196, July 2023).

**What the steps are.** They line up with moderation changes visible in the archive's "removed"
flag, which is the "mod-policy discontinuity" the ADR asked about:
- r/memes Feb 2025: the share of posts flagged removed drops from 52% to 28%. Counting only
  accounts with a non-removed post, the drop is 17% (5,875 -> 4,861), not 44%. The step is mostly
  authors of removed posts disappearing from the archive.
- r/comedyheaven 2020-2023: 96-99% of posts are flagged removed (it appears to have run as an
  approve-first community); in May 2024 that ends (14% removed) and the count triples. Counting only
  accounts with a non-removed post, r/comedyheaven has fewer than 100 per month for almost every
  month from 2020 to March 2024.

So whether "posting accounts" should include authors of removed posts changes the denominator by up
to 2x in some community-months, and the right answer differs by community and era.

**Also found (expected).** The count tracks post volume almost exactly: across months, log(accounts)
vs log(posts) has R^2 of 0.95-0.996 in six communities (r/meirl 0.80), slope 0.86-0.99. As a
denominator it carries very little that the plain post count does not.

**Compared to what.** All judged against lines fixed in advance: under 100 accounts (a count under
100 has over 10% noise of its own), a 2x month-on-month change (r/memes' typical change is 7.4% per
month), and a 10-point swing in the uncountable share (a potential 10% bias in the denominator).
There is no sampling uncertainty here: these are full counts of every submission in the archive.
The uncertainty is in what the archive recorded.

**What it implies for the decision.** The ADR's mitigation 2 ("divide counts by unique authors of
the community-month") does not work as written. It would need at least a correction or break term
at mid-2023 for the uncountable-author step, an explicit rule for removed posts, and special
handling for r/comedyheaven (2020-2024) and r/196 (July 2023). Since the count is nearly
proportional to post volume, the ADR's open question "unique authors or total posts?" matters
little on this evidence: both carry the same archive artefacts. This counts **against mitigation 2
as specified**, not against widening the corpus as such.

**Confidence.** High that the steps exist (full counts, plain in the plots; the mid-2023 step falls
in the same month in seven communities). Medium on the explanations (collection-method change;
approve-first moderation in r/comedyheaven): inferred from the archive's own flags, not confirmed
from outside documentation. What would undermine it: evidence that the archive treats pre-2023
posts differently from what is assumed here, which would change the size of the step.

Checked by: author only. The third agent re-read the code (`experiments/wp18/m1_authors.py`,
`m1_extra.py`), both plots and the monthly table at every flagged month; the counts themselves were
computed by the first agent.

Plots: `experiments/WP-18/m1_unique_authors.png`, `experiments/WP-18/m1_deleted_author_share.png`.
Tables: `m1_authors_monthly.csv`, `m1_extra_monthly.csv`, `m1_summary.json`, `m1_extra_summary.json`.

---

## Measurement 2: how often the same image is posted in more than one community

Window: every image post in the seven communities, 3-30 Nov 2025 (four whole weeks), 25,495
posts. An image counts as matching another if their 64-bit perceptual hashes (dHash, the
project's `experiments/wp01/dhash.py`) differ in at most 4 bits, a radius fixed in advance.

**Finding.** About 1 in 12 downloadable image posts has a hash match in at least one of the other
six communities, and about 1 in 15 is the same meme on inspection. Most of it is one pair of
communities and one behaviour: the same account posting the same file to r/memes and r/dankmemes.

| | value | 95% interval |
|---|---|---|
| Pre-registered primary: share of downloadable image posts with a match (<= 4 bits) in another community, all seven pooled (n = 16,226) | **8.3%** | 7.7-8.9% |
| Same, average of the seven communities with equal weight | 9.1% | not computed |
| Exact hash matches only (0 bits) | 4.9% | 4.5-5.3% |
| Corrected by eye to "same meme" (see audit below; not pre-registered) | 6.5% | 5.6-7.1% |
| Same meme or same template with a new caption (not pre-registered) | 7.9% | 6.9-8.4% |
| Control: same test against 26,134 r/memes images from 2019 (chance collisions) | 0.4% | 0.3-0.5% |
| For scale: posts with a match inside their own community (ordinary reposts) | 6.5% | 6.1-6.9% |

Intervals for measured shares come from resampling whole days (2,000 resamples, seed 18016),
because cross-posts arrive in bursts.

Per community (radius 4, % of that community's downloadable image posts; exact-hash share in
brackets): r/dankmemes 27.6% (25.3-29.9) [19.5%], r/memes 10.7% (9.7-11.6) [6.0%], r/shitposting
8.7% (7.7-9.7) [4.7%], r/meirl 6.9% (5.7-8.0) [3.7%], r/196 4.4% (3.5-5.2) [2.2%], r/comedyheaven
4.4% (3.3-5.5) [1.8%], r/whenthe 1.1% (0.7-1.6) [1.0%]. A quarter of all r/dankmemes image posts
(25.0%) match an r/memes post.

**Who is cross-posting.** 74% of exact-match pairs are the same account posting the same file to
two communities, usually within minutes (median gap between the two posts across all matched
pairs: 23 hours; a quarter within 10 minutes). Among the inexact (1-4 bit) pairs only 4% are the
same account; those are mostly the same template re-captioned by different people.

**Match audit (checked by eye by me).** The earlier agent's 30 random matched pairs, which I
regenerated and re-judged with the same result: 20 same meme, 9 same template with a different
caption, 1 near-black GIF frame. All 17 exact matches among them are the same meme. Because the
doubt sits in the inexact matches, I drew 40 more posts at random from those whose closest
other-community image is 1-4 bits away: 19 same meme, 16 same template with a different caption,
5 unrelated (dark or plain-white text screenshots, which hash alike). So roughly half of the
inexact matches are not the same meme, and the 2019 control (0.4%) understates false matches,
because it cannot contain 2025 templates. The corrected same-meme figure (6.5%) combines both
audits and their sampling uncertainty (`m2_audit_estimate.json`). Contact sheets:
`m2_contact_sheet.png`, `m2_match_audit_sheet1.png`, `m2_match_audit_sheet2.png`.

**Dead links.** Only 63.6% of the 25,495 image posts could be downloaded a year later: 8,777
(34.4%) return 404, 477 (1.9%) exceed the 10 MB cap (421 of them r/whenthe GIFs, 12% of that
community), 5 other failures. Downloadable share by community: r/196 86.9%, r/whenthe 66.4%,
r/shitposting 65.4%, r/memes 61.3%, r/comedyheaven 60.6%, r/dankmemes 60.1%, r/meirl 52.2%.
Pre-registered bounds on the all-posts overlap (no dead post is a duplicate / every dead post is
one): pooled 5.3% to 41.6%, which is too wide to be useful. Under the weaker assumption that a dead
post duplicates as often as a live one in the same community with the same moderation status,
the all-posts figure is 8.5% (8.0-9.0%). Among downloadable posts, those removed by moderators
match more often (11.4%) than unremoved ones (7.9%), so the missing posts more likely push the
true figure up than down. The image-free check (identical URL in two communities) finds almost
nothing (0.03% in the window), because native image uploads get a new URL each time; it cannot
substitute for hashing.

**Against the pre-registered criteria.**
- "Overlap above about 10% in any community means naive pooling double-counts": met for r/dankmemes
  (27.6%, and 19.5% on exact matches alone) and, narrowly, r/memes (10.7%; 6.0% exact). This
  supports the ADR's own rejection of naive pooling.
- "Fewer than about 100 cross-community duplicate pairs per month among the long-running set would
  make the 'same image in two communities' identification idea anecdotal": 554 pairs in 28 days
  at radius 4 (323 distinct images), 285 exact. That clears the line in aggregate, but 382 of the
  554 are r/memes-r/dankmemes; r/comedyheaven has 4-16 pairs with each other long-running
  community. And most exact pairs are one person posting twice (only 48 exact pairs in 28 days
  are by different accounts), which is a much weaker design than independent adoption of the same
  content by two populations.
- "If indistinguishable from the control, report no measurable cross-posting": it is clearly
  distinguishable (8.3% vs 0.4%; 4.9% exact vs 0.03% exact in the control).

**What it implies for the decision.** Pooling must de-duplicate across communities, and the
duplication is concentrated: r/dankmemes would contribute a quarter of its images twice. The
ADR's "cross-posting as an identification strategy" consequence survives only in a narrow form:
enough pairs exist for r/memes-r/dankmemes (and perhaps r/memes-r/meirl), not for r/comedyheaven,
and most pairs are a single account seeding two communities at once rather than the same content
spreading independently in two populations.

**Confidence.** Medium-high on the size of the effect among downloadable posts (the plots, the
exact-match subset and two by-eye audits agree). Lower on the all-posts figure because a third of
posts are dead and removal is not random. One four-week window in late 2025, so it says nothing
about overlap in 2019-2022, when most of the corpus was posted (the URL-sharing history, which
undercounts, was higher in 2019-2021 for r/meirl, r/shitposting and r/whenthe at 1-2.6%). What
would undermine it: a different 4-week window giving a very different r/dankmemes figure (one
heavy cross-poster can move it), or a better image matcher showing the template matches matter.
The by-eye judgements are mine and the previous agent's, not an independent reviewer's.

Checked by: author only (third agent re-checked the earlier agent's analysis outputs, code for the
day bootstrap and nearest-distance computation, both plots, and judged the contact sheets).

Plots: `experiments/WP-18/m2_crosspost_overlap.png` (title and caption updated by me to include the
audit), `experiments/WP-18/m2_overlap_vs_radius.png`. Data: `m2_summary.json`,
`m2_audit_estimate.json`, `m2_match_audit_judgement.csv`, `m2_contact_sheet_judgement.csv`,
`m2_deadlink_reweight.json`, `m2_url_shared_summary.json`, `m2_per_day.csv`.

---

## Measurement 3: can a classifier tell which community a meme came from?

Design (fixed in advance): 987 images per community (6,909 in all), drawn at random (seed 18016)
from the downloadable images of measurement 2's four-week window (3-30 Nov 2025), so every
community is sampled from the same weeks and calendar time cannot stand in for community. One post
per byte-identical image. The input is the project's embedding (OpenCLIP ViT-B/16, laion2b),
either image plus the CLIP embedding of text read off the image by OCR (EasyOCR, the
`p_img_ocr` composition), or the image alone. Classifier: logistic regression on standardised
features, regularisation chosen inside each training fold only. Held-out split: 5-fold
cross-validation in which every author, and every group of near-duplicate images (hash within 4
bits), sits on one side only (4,157 such groups). Primary metric: held-out top-1 accuracy over
the seven balanced classes, with a 95% interval from resampling those groups (2,000 resamples).

**The "[empty]" bug.** The project's `embed_pipeline` encodes an image with no extracted text as
the literal string "[empty]", so every text-free image shares one text vector. I set the text half
to **zero** for those images (323 of 6,909, 4.7%) and left the image half unchanged. It made no
difference here: 50.7% with the fix, 50.8% with the bug, and the "no text found" flag alone
predicts community at 15.3% (chance 14.3%). For a linear classifier both versions amount to the
same constant point, so this was expected; the bug matters more for distance-based uses (nearest
neighbours, clustering), where all text-free images look identical in the text half.

**Finding.** Yes, well above chance, but it is not uniform across communities.

| Input | held-out accuracy | 95% interval | share of the chance-to-perfect gap closed |
|---|---|---|---|
| **Image + extracted text (primary)** | **50.7%** | 49.1-52.4% | 42.5% |
| Image only | 49.5% | 47.7-51.3% | 41.1% |
| Extracted-text embedding only | 37.1% | 35.6-38.7% | |
| Baseline: file size, shape, format, animated or not | 33.2% | 30.8-35.7% | |
| Baseline: posting time (day in window, hour, weekday) | 14.9% | 13.9-15.9% | |
| Control: same pipeline, labels shuffled | 14.1% | 13.3-14.9% | |
| Chance = majority class | 14.3% | | |

The five fold accuracies for the primary are 49.7-52.5%. The main deliverable is the confusion
matrix, `m3_confusion_matrix.png`: if community could not be read from the image, every cell would
be about 14%. Instead, per-community recall (primary input): r/whenthe 91%, r/comedyheaven 64%,
r/meirl 63%, r/196 44%, r/dankmemes 34%, r/memes 30%, r/shitposting 28%. r/memes and r/dankmemes
are confused with each other almost as often as they are recognised (29% of r/memes images are
called r/dankmemes; 25% the other way round). r/shitposting and r/196 scatter across several
communities, mostly r/comedyheaven and each other.

Secondary results, pre-registered:
- Long-running set versus newer communities (two-way, balanced): 69.2% (67.6-70.9%) against 50%.
- r/memes versus r/dankmemes (two-way): 54.3% (52.0-56.7%) against 50%, barely separable.

Sensitivities (declared before results, lower standing):
- r/whenthe is 97% animated GIFs, so its 91% could be "is it a GIF" rather than content. Dropping
  it, six-way accuracy is 44.8% (43.2-46.4%) against 16.7% chance; on still images only (six
  communities, 936 each): 44.9% (43.3-46.6%). So most of the effect survives without the GIF
  community: about a third of the chance-to-perfect gap is closed on content alone.
- Folds that ignore authors and near-duplicates: 48.2%, no higher than grouped folds, so the
  result is not driven by one person's posts or reposts landing on both sides.
- OCR text naming a candidate community ("r/memes" etc.): 2 of 6,909 images (0.03%). No literal
  label leak.
- The regularisation search picked the strongest setting in the grid every time (C = 0.001, the
  grid's edge). Checked afterwards with fixed C (not pre-registered): 48.7% at 0.0001, 50.5% at
  0.0003, 50.7% at 0.001, so a wider grid would not have raised the figure.

**Against the pre-registered reading.** The interval (49.1-52.4%) excludes 14.3% and exceeds both
the metadata baseline (33.2%) and the time baseline (14.9%). By the rule written in advance, the
embedding carries community information as content. It closes 42.5% of the gap from chance to
perfect, below the "more than half" line I gave for orientation as "strong"; the expectation was
45-65%, and the result sits inside it. The user is the one to judge whether 51% is strong enough to
matter.

**What it implies for the decision.** The ADR names this risk itself: if community can be read off
the meme, a spread model given the embedding can infer community (and with it size, moderation and
house style) without being told, and the "new community" cells of its proposed factorial split
would read optimistically. This result says that risk is real for most of the corpus. It is
strongest for exactly the split the ADR leans on: long-running set versus newer communities is 69%
separable, and r/whenthe is nearly perfectly identifiable. It is weakest for r/memes versus
r/dankmemes, which look nearly alike (and, from measurement 2, share a quarter of r/dankmemes'
images). Practical consequence if the ADR goes ahead: the community-generalisation test needs a
control, e.g. reporting results with community given explicitly to the model as well as without,
or residualising the embedding on community, so that the effect of community can be separated from
content.

**Confidence.** High that the embedding separates communities well above chance (narrow interval,
stable folds, permutation control at chance, robust to grouping and to removing the GIF
community). Medium on the size as a guide to what a spread model would exploit: a linear probe is a
lower bound on what a flexible model could extract, and the probe says nothing about whether a
spread model would actually use it. Limits: only downloadable images (a third of posts were dead
links, likely the least on-style ones), one four-week window in late 2025 (does not test 2019-2021,
where most of the corpus lies, and where community styles may have been more or less distinct), GIFs
seen by first frame only, and r/dankmemes had only 987 usable images, which set the per-class size.

Checked by: author only (third agent ran the embedding and every probe, and looked at both plots).

Plots: `experiments/WP-18/m3_confusion_matrix.png` (main deliverable),
`experiments/WP-18/m3_accuracy_vs_baselines.png`. Data: `m3_summary.json`, `m3_sensitivity.json`,
`m3_c_edge.json`, `m3_probe_early.json`, `m3_oof_predictions.parquet`, `m3_embeddings.npz`,
`m3_embed_meta.json`, `m3_ocr*.jsonl`, `m3_sample.parquet`, `m3_sample_meta.json`.

---

## Archive completeness check (requested by the coordinator after a truncation risk was found in the ingester)

The ingester's zstd reader stops at the end of the first compressed frame without error. I checked
all six extra archives (all still on disk under `/home/linus/reddit_dl_wp18/reddit/subreddits25/`)
with an independent full decompression (pyzstd, restarted at every frame boundary) plus the
reviewer's header-level frame walk (`experiments/wp16/review/zstd_frames.py`).

| Community | frames | non-blank lines in archive | rows in parquet | last / max timestamp match |
|---|---|---|---|---|
| r/dankmemes | 1 | 6,076,698 | 6,076,698 | yes (1767224800) |
| r/meirl | 1 | 619,586 | 619,586 | yes (1767225410) |
| r/comedyheaven | 1 | 597,629 | 597,629 | yes (1767224155) |
| r/whenthe | 1 | 199,960 | 199,960 | yes (1767225587) |
| r/196 | 1 | 1,699,704 | 1,699,704 | yes (1767224918) |
| r/shitposting | 1 | 1,349,071 | 1,349,071 | yes (1767225454) |

Every archive is a single frame that ends exactly at end of file, no line failed to parse, and both
the last record's and the maximum `created_utc` equal the parquet's (all 31 Dec 2025, UTC). The six
tables are complete; measurements 1-3 are unaffected. Output: `verify_ingest_frames.json`, script
`experiments/wp18/verify_ingest_frames.py`.

---

## Surprises

- The share of posts with an uncountable author collapses from about a third to about 1% in one
  month (mid-2023), simultaneously in all seven communities: an archive artefact large enough to
  move any per-author denominator by up to 2x.
- r/comedyheaven had 96-99% of posts flagged removed from 2020 to early 2024.
- A quarter of r/dankmemes' image posts also appear in r/memes, mostly the same account posting the
  same file to both within minutes.
- Roughly half of the inexact (1-4 bit) hash matches are the same template with a different caption,
  not the same meme; the 2019 chance control does not catch this.
- About a third of image links from November 2025 were already dead.

## What I could not do or did not do

- No independent reviewer has checked any of this (the project's rules ask for one before results
  enter the record). All three measurements are UNCERTIFIED. The by-eye match judgements are mine
  and the previous agent's.
- `docs/` (work-package log, state, changelog) not updated, as instructed.
- The all-posts overlap in measurement 2 is bounded only loosely (5.3-41.6%) because of dead links.
- Measurement 3 covers one four-week window in late 2025; separability in earlier years is untested.

## Process notes

- Two files were truncated by the 18:17 crash: `m2_contact_sheet.png` (regenerated, same seed) and
  `m3_embed.log` (the embedding had not started). Everything else checked out.
- The embedding was rewritten as a resumable job (`experiments/wp18/m3_embed_resumable.py`) and the
  OCR split across three workers because the machine was heavily loaded; total GPU time about 2h10m
  (21:11-23:24). The GPU was released at 23:24 (no WP-18 process on it afterwards).
- Seeds: 18016 everywhere (sampling, folds, bootstraps, audit draws); label permutation 18017.
