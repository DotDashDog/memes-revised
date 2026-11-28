# WP-18 — expectations, written before each measurement

Purpose: run the three precondition measurements of ADR 0016 (multi-subreddit corpus). These notes
are written **before** the corresponding result is looked at. Nothing below is edited after the
fact; corrections and deviations are appended under "Deviations" with a timestamp.

Candidate communities: backbone = r/memes, r/dankmemes, r/meirl, r/comedyheaven;
extension = r/whenthe, r/196, r/shitposting.

Global seed for everything random in WP-18: **18016** (sub-seeds stated per script).

---

## Measurement 1 — unique authors per community-month (written 2026-11-27, before the other six
subreddits were ingested; only the r/memes parquet existed and had not been queried for authors)

**Question.** ADR 0016 mitigation 2 divides post counts by N(c,t) = unique authors active in
community c during time t. Is that denominator well behaved?

**Definitions (fixed now).**
- Month = UTC calendar month of `created_utc`.
- `N_named(c,t)` = number of distinct `author` values in the month, excluding null, `[deleted]`
  and `[removed]`. This is the primary quantity.
- `deleted_share(c,t)` = fraction of the month's posts whose author is null/`[deleted]`.
- "Active life" of a community = from the first month with >= 1,000 posts to 2025-12. Months before
  that are reported but not judged (a community that has not started is not a bad denominator).
- Near-zero month: `N_named < 100` inside the active life. Why 100: N is itself a count, so its own
  relative sampling noise is about 1/sqrt(N); below 100 the denominator alone carries >10% noise,
  comparable to the effect sizes a spread model would be asked to resolve, and a single meme with
  10 posters is already >10% of the "population".
- Discontinuity: a month-on-month change in `N_named` by a factor >= 2 (up or down) inside the
  active life. Why 2: organic change in r/memes, the only community measured so far, is a smooth
  decline of roughly 3-8% per month (WP-16: -50%/yr at the fastest); a factor of 2 in one month is
  an order of magnitude faster than that and cannot be absorbed as trend. I also report a robust
  z-score of each month's log-change against that community's own distribution of log-changes,
  because a fixed factor is blind to steps of 1.3-1.9x; both are reported with equal standing.
- Named events checked explicitly whatever the flags say: May 2023 (Pushshift loses API access),
  June 2023 (API-pricing blackout, many subs private 12-14 June and longer), and the last months of
  the dump (2025-10..12, where late captures might be incomplete).

**What I expect.**
- r/memes and r/dankmemes: N rises to a 2020 peak (order 10^5 authors/month for r/memes), then a
  smooth decline; no near-zero months after 2018.
- r/comedyheaven: the research note's day counts (662 -> 13 -> 65 posts/day in 2020/2023/2026)
  suggest a collapse and partial recovery; I expect at least one >=2x step and possibly months
  under 1,000 authors around 2023. This is the backbone member I most expect to misbehave.
- r/196 and others: a visible hole in June 2023 (r/196 stayed dark longer than most).
- deleted_share: I expect it to be higher for older months (accounts deleted before the dump's
  capture/refresh) and to differ by era, probably 5-30%. I do not know how the 2025-12 dump treats
  authors of posts captured live versus refreshed later; a step in deleted_share at 2023-04/05
  (collection-route change) is plausible.
- N will be strongly correlated with post count (most authors post once a month). If
  log N vs log posts is nearly a straight line, unique authors carries almost no information that
  post count does not, which is not a defect of the denominator but bears on the ADR's open
  sub-question about which N to use.

**What would count against the ADR.**
1. Any near-zero month (N_named < 100) in the active life of a backbone community: the exposure
   term would blow up there.
2. Any >=2x month-on-month step that is not a community's organic start-up: a normalised count
   would jump at that month for reasons unrelated to memes.
3. deleted_share moving by more than 10 percentage points between eras within a community: the
   named-author count is then biased by a factor that changes over time (each deleted post hides an
   unknown author), so a trend in N partly measures account deletion, not population. Why 10 points:
   the bias in N from deleted authors is bounded by deleted_share itself (N_true is between N_named
   and N_named + number of deleted-author posts), so a 10-point swing is a potential 10% swing in
   the denominator, the same noise level used for criterion 1.

**Why these test the question.** The ADR's claim is that N(c,t) is a usable exposure term. A usable
exposure term must be (a) bounded away from zero, (b) change no faster than the population it
stands for, and (c) be measured with a bias that does not itself drift. The three criteria map onto
(a), (b), (c).

**Blind spots.** Monthly resolution hides within-month outages (a 3-day blackout barely moves a
monthly unique count); I therefore also compute daily post counts for June 2023. Unique *posters*
says nothing about lurkers or commenters (ADR open sub-question; not testable from submissions).
A count that is smooth can still be the wrong population.

---

## Measurement 2 — cross-posting overlap (written 2026-11-27, before any image for this task was
fetched and before the other subreddits' data existed locally)

**Question.** What fraction of image posts in one candidate community are near-duplicates of an
image posted in another candidate community? Large overlap means naive pooling double-counts;
some overlap is needed for the ADR's "same image in two communities" identification idea.

**Design (fixed now).**
- Window: **28 consecutive days, 2025-11-03 00:00 UTC to 2025-11-30 23:59 UTC** (four whole
  Mon-Sun weeks). Why this window: it is recent (image links rot with age; 43% of 2019 links are
  dead), it avoids the Christmas/New Year fortnight and the dump's final days, and four whole weeks
  balances day-of-week effects. Why 28 days and not 1: the smallest community (r/comedyheaven) posts
  about 65 images/day. If the true overlap is ~1%, a binomial 95% interval of +/-0.5 points needs
  about 1,500 posts (1.96^2 x 0.01 x 0.99 / 0.005^2 = 1,521), i.e. ~23 days of r/comedyheaven; and
  because cross-posted memes arrive in bursts, uncertainty has to come from resampling whole days,
  which needs a few dozen days to be meaningful. 28 is the smallest whole-week count meeting both.
  If volumes turn out so different that this fails (e.g. <1,000 fetchable posts in a community), I
  will say so rather than extend the window after seeing overlap numbers.
- Population: every submission in the seven communities in the window whose URL the project's
  fetcher regards as a direct image (`experiments.wp04.fetch.normalize_url`). Reddit galleries,
  videos and text posts are counted and reported but are not "image posts" here.
- Every image URL is fetched once. Every outcome (ok, 404, removed, ...) is counted per community.
- Near-duplicate: 64-bit dHash (`experiments/wp01/dhash.py`, first frame for GIFs),
  **Hamming distance <= 4** as the primary radius. Radii 0, 2, 6, 8, 10 are reported alongside with
  equal standing. Why 4: it is the middle of the radius scan used in the project's earlier family
  work (WP-04, T in 0..10), no radius was ever certified there, so I fix one now instead of picking
  after seeing overlap. Its false-match rate is measured, not assumed (see controls).
- **Primary metric:** for each community A, the fraction of A's successfully fetched image posts in
  the window that have at least one near-duplicate among the fetched image posts of *any other*
  candidate community anywhere in the same 28-day window. Headline = that fraction pooled over the
  seven communities, and the 7x7 matrix (row A, column B = share of A's posts with a near-duplicate
  in B).
- Uncertainty: cluster bootstrap over the 28 days (resample days of the query post with
  replacement, 2,000 resamples, seed 18016), 95% percentile interval. Per-day values are plotted so
  the reader sees the spread directly.
- Dead links: reported per community. Bounds on the all-posts overlap are given under the two
  extreme assumptions (no dead-link post is a duplicate / every dead-link post is one). In addition,
  an image-free measure that includes dead links: the share of posts whose exact URL also appears in
  another candidate community (native Reddit crossposts share the URL), computed for the window and
  for every month of the archive.
- Controls:
  1. Background false-match rate: the same matching run against 26k dHashes of 2019 r/memes images
     (`experiments/WP-04/dhash.parquet`). Six years apart, so matches are either hash collisions or
     evergreen templates; this bounds the false-positive rate per ~26k candidates from above.
  2. Same-community repost rate at the same radius (near-duplicate inside A itself), as a scale
     reference: is cross-community overlap bigger or smaller than ordinary reposting?
  3. A contact sheet of randomly drawn matched pairs at the primary radius, inspected by eye, with
     the number of pairs that are not actually the same image stated.

**What I expect.** Small. Reddit communities enforce repost rules and these seven have distinct
house styles. Guess: 0.5-3% of posts at radius 4, highest between r/memes and r/dankmemes, and
between r/whenthe / r/shitposting / r/196. Background false-match rate well under 0.5%.

**What would count against the ADR, and what would count for it.**
- Against naive pooling (which the ADR already rejects, so this supports the ADR's caution): overlap
  above roughly 10% in any community would mean one in ten "independent" posts is a copy.
- Against the ADR's own proposal: the "cross-posting as identification strategy" consequence needs
  enough cross-posted images to estimate a community effect. If the pooled overlap implies fewer
  than about 100 cross-community duplicate pairs per month on the backbone, that strategy has too
  few samples to be more than an anecdote, and that consequence should be struck. Why 100: a
  within-meme comparison is a paired difference; with fewer than ~100 pairs spread over six
  community pairs there are under ~20 per pair, too few to estimate even a mean shift with a useful
  interval. This is an order-of-magnitude criterion, not a sharp one, and I will report the count.
- If the overlap estimate is not distinguishable from the background false-match rate, the honest
  reading is "no measurable cross-posting at this radius", not a small positive number.

**Blind spots.** dHash misses crops, added captions/watermarks, re-encodes with borders, and the
same joke redrawn, so this is a lower bound on "same meme" and close to a measure of "same file,
lightly re-encoded". It only sees these seven communities. A 28-day window misses reposts with a
lag longer than the window and truncates lags at the edges. Dead links are not missing at random
(reposts are disproportionately removed by moderators), which biases the fetched-image estimate
downward; the bounds and the URL measure exist for that reason.

---

## Measurement 3 — community-shortcut audit (written 2026-11-27, before any embedding existed for
any community other than r/memes, and before any image for this task was fetched)

**Question.** Can a classifier tell which community a meme came from using only the meme's
embedding? If so, a later spread model given the embedding can infer community (and with it
community size, moderation regime, era) without being told, and the ADR's "new community" test
cells may read optimistically.

**Design (fixed now).**
- Data: the images fetched for measurement 2 — all seven communities, the same 28-day window
  (2025-11-03..2025-11-30). Using one common window is the control for the time confound: every
  class is sampled from the same four weeks, so "when" cannot stand in for "where" at any scale
  coarser than days. A timestamp-only classifier (day index within the window, hour of day, weekday)
  is run as a baseline to show how much time alone still gives.
- Class-balanced: the same number n of images per community, drawn at random (seed 18016) from the
  successfully fetched, decodable images after removing exact-URL and byte-identical duplicates.
  n = min(1,500, smallest community's available count). Balanced classes make chance = majority =
  1/7 = 14.3%, and keep the confusion matrix readable.
- Embeddings: OpenCLIP ViT-B/16 (laion2b_s34b_b88k), the project's primary encoder.
  (a) `p_img_ocr`: image vector concatenated with the CLIP text vector of the EasyOCR-extracted text
  (`easyocr_filtered`, no preprocessing — the settings in `experiments/wp04/ocr_embed.py`);
  (b) `p_vlm_only`: image vector only. Both reported with equal standing.
- Classifier: multinomial logistic regression on standardised embeddings, L2 strength chosen by
  inner cross-validation on training folds only. A linear probe is deliberately simple: it gives a
  *lower bound* on what a flexible model could extract.
- Held-out split: 5-fold cross-validation with **groups**. A group is a connected component of
  "same author" or "dHash within Hamming 4" — so no author and no near-duplicate image is on both
  sides of a split. Without this, an author who posts only in one community, or a repost, would let
  the classifier memorise instead of generalise. Deleted authors are singletons.
- **Primary metric:** out-of-fold top-1 accuracy over the seven balanced classes, for `p_img_ocr`.
  Uncertainty: 95% interval from a bootstrap over groups of the out-of-fold predictions (2,000
  resamples, seed 18016), plus the five per-fold accuracies.
- Baselines, same folds, same metric: chance (1/7); majority class (1/7 by construction);
  label-permutation (same pipeline on shuffled labels — must return ~1/7, otherwise the pipeline
  leaks); timestamp-only; file-metadata-only (width, height, aspect ratio, file size, format,
  animated or not) — tells whether any signal is trivial formatting instead of content; OCR-text
  vector only.
- Secondary, reported whatever they show: the 7x7 confusion matrix (main deliverable); a binary
  backbone-vs-extension probe; r/memes-vs-r/dankmemes pairwise accuracy.

**What I expect.** Clearly above chance: 45-65% seven-way for `p_img_ocr`, a little lower for
image-only. r/memes and r/dankmemes mutually confused; r/196, r/whenthe and r/comedyheaven the most
identifiable. Timestamp-only near chance (maybe 16-20%, from time-zone differences). Metadata-only
above chance (20-30%) because r/whenthe and r/shitposting carry more GIFs.

**What would count against the ADR.** The ADR itself states the reading: strong performance means
community is available to the model as a shortcut and the factorial split's off-diagonal cells
("new community") will read optimistically. I cannot derive a sharp number at which "above chance"
becomes "strong", and I will not invent one. What I commit to in advance:
- If the 95% interval for the primary metric includes 14.3%, community is *not* recoverable by a
  linear probe (precondition satisfied in the favourable direction, subject to the lower-bound
  caveat).
- If the interval excludes 14.3% **and** exceeds the metadata-only and timestamp-only baselines, the
  embedding carries community information as content. The size is reported as accuracy and as the
  fraction of the gap from chance to perfect that is closed, (acc - 1/7)/(1 - 1/7), and the user
  judges. For orientation only: closing more than half the gap would mean a model can name the
  community more often than not from a single image; I would describe that as strong.
- Separately from the headline: if extension communities are rarely confused with backbone
  communities (binary probe far above its 50%-balanced chance), the embedding separates exactly the
  two halves the ADR's split design contrasts.

**Why this tests the question.** The concern is whether community is *recoverable from the model's
input*. A held-out, group-separated probe on the exact embedding the project uses measures
recoverability directly; the time-matched window and timestamp baseline remove "era" as the
explanation; the metadata baseline separates "content/style" from "file format".

**Blind spots.** Accuracy from a linear probe understates what a non-linear model could extract and
says nothing about whether a spread model *would* use it. Only successfully fetched images are
included, so deleted/removed posts (which may be the least on-style ones) are missing. One 28-day
window in late 2025: community styles may have been more or less distinct in other years; this does
not test separability in 2019-2021, where most of the corpus lies.

---

## Deviations and corrections (appended, never rewritten)

- 2026-11-27 12:08 PST — **Machine restarted mid-ingest** (WSL came back with uptime 1 min; cause
  unknown, not something this task did deliberately). The torrent download had already reported
  COMPLETE; the first subreddit ingest (r/dankmemes) was killed partway. Action: deleted the partial
  staging parquet, re-ran the torrent client against the downloaded files (it re-verified and
  reported complete with nothing to fetch; log in `fetch_archive_recheck.log`), restarted the
  ingest from scratch. No measurement had been run or looked at before the restart.
- 2026-11-27 12:30 PST — **Measurement 2/3, before the full fetch, after a 300-image pilot** (pilot
  outcome looked at: fetch statuses and file sizes only; no hashes, no overlap). Two things the
  pilot and the URL table showed that the design above did not anticipate:
  1. **r/whenthe is 97% animated GIF** (3,393 of 3,501 direct-image URLs in the window end in
     `.gif`; every other community is 0-6% GIF). GIFs average ~7 MB against ~0.5 MB for stills, so
     the full 28-day fetch would be ~29 GB, of which ~18 GB is r/whenthe GIFs.
  2. Dead links are ~30% (90/300) even for a 12-month-old window, not the few percent I assumed.
  **Decision (mine, cost-driven):** the user set ~25 GB as the budget for this task's archive
  download and the project rule reserves "tens of GB" for the user, so I do not fetch 29 GB
  unasked. I lower the fetcher's per-file size cap from 30 MB to **10 MB** (the fetcher's own
  `too_large` status, decided from Content-Length before the body is read). Expected: ~16 GB;
  no still image is affected (largest still in the pilot: 7.9 MB); about a quarter of GIFs
  (so about a quarter of r/whenthe) are recorded `too_large`. They are **counted as not usable**,
  exactly like dead links, and enter the dead-link bounds. The cap is applied uniformly in the
  analysis (pilot images over 10 MB that were already stored are treated as `too_large`).
  Consequence to keep in view: r/whenthe's usable sample is biased toward smaller GIFs; for GIFs
  the hash and the embedding see the **first frame only**. The 28-day window, radius, primary
  metrics and all criteria are unchanged.
- 2026-11-27 ~17:30 PST — **Resumed by a second agent after the 12:38 shutdown.** Written before
  any measurement-2 overlap number or any measurement-3 embedding/probe result exists (no
  `m2_dhash.parquet`, `m2_summary.json` or `m3_*` file is on disk).
  1. State verified, not assumed: the fetch had in fact finished (25,181/25,181 URLs, run summary
     and compacted index written 12:37:34); shards and index are checked byte-for-byte against the
     recorded sha256 by `experiments/wp18/m2_verify_shards.py` -> `m2_shard_verify.json` before use.
     The fetch ran with the WP-04 limits unmodified (i.redd.it ramp 4->16 workers, reached 16; no
     429/403 full stops); only the size cap was lowered, as recorded above.
  2. The 12:30 note says "the user set ~25 GB as the budget". I cannot find that in `docs/`; it may
     have come from the previous agent's instructions. Recorded as unverified. No further download
     is made.
  3. Measurement 1: the robust-z flag uses |z| > 5, a number that is not in the pre-registered text
     above (which says only that robust z is "reported"). I treat it as descriptive, not as a
     criterion. Pre-registered criteria 1-3 are judged as written.
  4. Primary metrics, radius, windows, criteria for measurements 2 and 3 are **unchanged** from the
     text above. Added, NOT pre-registered, reported as sensitivities with lower standing than the
     primaries, decided now before seeing results:
     - M2: share of cross-community pairs by the same named author (is "overlap" one person
       posting to several places, or independent spread?); count of distinct cross-community
       duplicate *groups* per 28 days on the backbone next to the pre-registered *pair* count,
       because one image posted in 3 places makes 3 pairs.
     - M3: (a) 6-way probe without r/whenthe, and a stills-only probe (n_frames == 1), because
       r/whenthe is 97% GIF and the image embedding of a GIF first frame may encode palette
       quantisation rather than content; (b) share of OCR texts that contain a subreddit name or
       "r/<name>" (a literal label leak inside the image); (c) the same ungrouped-fold sensitivity
       already in the code.
- 2026-11-27 ~21:20 PST — **Third agent, after the 18:17 shutdown (disk full).** Written before
  any measurement-3 embedding of the full sample or any probe result exists (only the 64-image
  smoke-test embedding, which was never fed to a probe).
  1. Files found truncated by the crash: `m2_contact_sheet.png` (0 bytes; regenerated with the
     same seed and the earlier by-eye judgement re-checked) and `m3_embed.log` (0 bytes; the full
     embedding had not started). `m3_sample.parquet` is intact (6,909 rows = 987 per community,
     unique post ids).
  2. **Measurement 3, "[empty]" text bug.** `experiments/wp01/pipelines.embed_pipeline` encodes
     an image whose extracted text is empty as the literal string "[empty]", so every text-free
     image gets the same, semantically arbitrary text vector. For this measurement the text half
     of `p_img_ocr` is set to **zero** for every image whose OCR text is empty after `.strip()`
     (the same emptiness test `embed_pipeline` uses). The image half is unchanged. Implemented in
     `experiments/wp18/m3_embed_resumable.py` (which also checks on the first chunk that its
     direct `encode_images` / `encode_texts` calls reproduce `embed_pipeline` exactly). Note for
     interpretation, stated before seeing results: for a *linear* probe on standardised features,
     zero and the "[empty]" vector are both one constant point, i.e. both amount to a
     "no text found" indicator; the fix changes only where that point sits. The bugged variant is
     therefore also run, as a comparison with lower standing, and the per-community share of
     text-free images is reported because that indicator can itself carry community information.
  3. **Measurement 3, primary metric kept as pre-registered**: out-of-fold top-1 accuracy, seven
     balanced classes (987 each; chance = majority = 1/7 = 14.3%), `p_img_ocr` (fixed as above),
     5-fold grouped CV (groups = same named author or dHash within 4), 95% interval by bootstrap
     over groups (2,000 resamples, seed 18016). Expectation unchanged: 45-65%. What counts
     against the ADR unchanged: interval excluding 14.3% and exceeding the metadata-only and
     timestamp-only baselines means the embedding carries community information as content;
     closing more than half the chance-to-perfect gap (accuracy > 57.1%) is what I would call
     strong. I see no reason to call the metric unsound.
  4. **Engineering changes, not design changes:** embedding is split into resumable phases (image
     vectors; OCR checkpointed per 64 images; text vectors) because two shutdowns have killed this
     task and the machine is heavily shared. Same encoder, OCR settings, sample and seed.
  5. **Measurement 2, added match audit (not pre-registered; decided after reading the previous
     agent's 30-pair judgement, before looking at any new pair):** 40 posts drawn at random (seed
     18016) from those whose nearest other-community image is at Hamming 1-4, judged by eye
     against that nearest image, to estimate how much of the primary radius-4 overlap is the same
     meme rather than the same template with a new caption. Reported next to the pre-registered
     primary, which is not replaced.
