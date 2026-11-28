# Independent review: clustering audit (WP-04, sample-001)

Reviewer: fresh agent (Opus 5.5), 2026-11-27. I was not told the author's conclusion. I read
it afterwards in the title of `plots/precision_recall.png`: "the embedding recovers templates
about as well as a perceptual hash, and recovers meaning-level meme families only slightly
better than not clustering at all".
Scripts and outputs are in this folder: `recompute.py` (+ `recompute.json`), `diagnose.py`
(+ `.json`), `empty_fix.py` (+ `.json`), `bg_only.py` (+ `.json`), and contact sheets
`empty_ocr_cluster.png` and `cyanide_cluster.png`. None of these import the audit's code.
I changed no existing file.

## 1. What I recomputed, and whether it matched

- **Alignment.** `ids.json` holds the same 797 ids as `labels.parquet` but in a different
  order. `score_audit.py` maps rows by id, which is correct. The image half of
  `embeddings_p_img_ocr` equals `embeddings_p_vlm_only` row for row (max difference 0.0).
  The text half is identical for all 29 items with no extracted text. `labels.parquet`
  equals `stage2/labels_A.json` on both layers. Separately, exact-hash duplicates (radius
  0) show up as each other's nearest neighbours, which is a further sign that rows match
  images.
- **B-cubed (my own Counter-based implementation).** Recall at precision 0.90 matches
  `scores.json` to 4 decimals for every arm × stratum × layer (24 numbers). Precision curves
  agree to 5e-5. Headline family-layer numbers, pooled / background:
  - image + text: 0.370 / 0.282
  - image only: 0.370 / 0.287
  - perceptual hash: 0.294 / 0.242
  - every item alone: 0.242 / 0.210
- A finer threshold grid (step 0.005 instead of 0.02–0.04) changes these by at most 0.012.
  The grid is not driving the result.

## 2. Problems, ranked by how much they could change the conclusion

### P1. The arm comparison depends on the operating point, and the plot disagrees with the number (method-discipline §5)
On the family layer, image + text and image only tie at precision 0.90. Below that, the
image + text curve clearly dominates (visible in `precision_recall.png`, top row). Paired
bootstrap of image + text minus image only, done the same way as the registered one
(`diagnose.json`):

| precision setting | pooled | background |
|---|---|---|
| 0.90 | −0.003 [−0.034, 0.021] | −0.011 [−0.060, 0.035] |
| 0.80 | +0.027 [−0.019, 0.062] | +0.038 [−0.015, 0.092] |
| 0.70 | **+0.059 [0.015, 0.103]** | **+0.073 [0.017, 0.139]** |
| 0.50 | **+0.117 [0.052, 0.185]** | **+0.179 [0.074, 0.280]** |

The UMAP shows the same thing. With image + text, area-51, team-trees, keanu-reeves,
minecraft-vs-fortnite and spooktober form tight clumps. With image only they are scattered.
At 0.90, 78% of the precision loss falls on images the labels call unique (`diagnose.json`),
so the registered point is set by a handful of false merges, not by how well families are
recovered. The registered number is what it is. But "adding text makes no difference" holds
only at that operating point. Stated without that qualifier it is not supported, and the
disagreement with the plot is an open finding.

### P2. Pipeline bug: every image without extracted text gets the same text vector
`embed_pipeline` (experiments/wp01/pipelines.py) replaces empty OCR with the string
`"[empty]"`. All 29 text-free images therefore share an identical text half, so any two of
them have image + text cosine similarity of at least 0.5. At the 0.90 setting, image + text
puts 17 of them into one cluster. They are unrelated pictures: crying cat, Barney, Kirby,
blurry faces (`empty_ocr_cluster.png`). That one cluster costs about 0.02 of the 0.10
precision budget.

Zeroing the text half for these items (`empty_fix.json`) changes the image + text numbers:
- recall at precision 0.90 rises from 0.367 to 0.392 (pooled) and from 0.278 to 0.318
  (background);
- minus image only, at 0.90: +0.022 [−0.012, 0.047] pooled, +0.029 [−0.033, 0.081]
  background;
- minus image only, at 0.80: +0.043 [−0.005, 0.079] pooled, **+0.060 [0.010, 0.118]**
  background.

The fix does not flip the primary number on its own. It does reverse the sign of the
"text vs no text" difference, and it hands an artefactual handicap to the arm the project
has accepted. At corpus scale the bug would make one giant cluster out of every image
without text, so it matters for ADR 0009 regardless of this audit.

### P3. The agent labels were used for scoring before the user validation the protocol requires
The protocol says agent labels "are used for scoring only if they agree with the user"
(≥85% on agent-same pairs and on hard negatives). `user_check/pairs.json` has 100 pairs
drawn, but I found no user answers. Every score is therefore on unvalidated labels and
should be labelled as such.

The two-labeller agreement (92–98%) is weaker evidence than it looks:
- 717 of 797 stage-1 descriptions given to A and B are byte-identical.
- Both labellers are the same model family.
- So the agreement bounds noise in the grouping step, not bias in the descriptions or in
  the model's idea of "same meme".

Item-level agreement is in fact slightly lower on the shared descriptions (0.88) than on the
differing ones (0.91, n = 80), so shared input did not obviously drive it. I found one
systematic labelling ambiguity that lands directly in precision. "Fill-in-the-slot"
templates are labelled inconsistently:
- Cyanide & Happiness "talk to me like you're a fucking idiot" (`cyanide_cluster.png`):
  5 posts, labelled unique, area-51-raid and me-and-the-boys.
- "Ew, I stepped in shit": 6 posts, labelled unique, mematic-watermark and
  minecraft-vs-fortnite.

Protocol rule 3 would plausibly make each of these one family ("me-and-the-boys" was
treated that way). The clustering groups them and is charged with false merges. Only the
user's pairs can settle this.

### P4. The metric's floor, and a mismatch between its name and its formula
The registration describes the primary as "the share of same-family pairs kept together".
B-cubed recall counts each item as kept together with itself, so leaving every image alone
already scores 0.242 pooled and 0.210 in the background (the mean of 1/family size). With
self-pairs removed, which matches the registered wording (`recompute.json`):

| | pooled | background |
|---|---|---|
| image + text | 0.213 | 0.107 |
| image only | 0.211 | 0.110 |
| hash | 0.090 | 0.040 |
| every item alone | 0 | 0 |

The ranking is the same, but a reader shown "0.37" will overrate it. As a share of the gap
between leaving everything alone and perfect recall, image + text recovers 17% (pooled) and
9% (background).

Standard B-cubed F1 over all items, the other "equally defensible" choice, is uninformative
here: about 0.89–0.91 for every arm, because 72% of items are singletons. That is exactly the
failure the 2026-11-27 log warned about.

### P5. The "background" number is not a clean measure of a uniform draw
Background items are scored against families that include family-stratum members: 79 of
the 103 background items in families of two or more have a relative in the family-seeded
stratum. I clustered the 500 background images alone and scored them against labels
restricted to the background (`bg_only.json`). That leaves only 23 families of two or more
(91 items). Family layer, recall at precision 0.90:

| | recall |
|---|---|
| image + text | 0.323 (0.351 with the P2 fix) |
| image only | 0.328 |
| text only | 0.330 |
| hash | 0.278 |
| every item alone | 0.253 |

With 23 families this cannot separate the arms. That is the registered "not enough families
to tell" outcome and should be reported that way, not as a tie. Also, the hash sweep ends at
radius 10. In the background template layer, hash precision never drops below 0.90
(0.904 at radius 10), so the hash value there is cut off by the sweep range and slightly
understates the hash.

### P6. The plot title overstates one claim
"Recovers templates about as well as a perceptual hash" is true for image only: −0.058
[−0.116, 0.005] pooled, +0.015 [−0.081, 0.129] background. It is false for image + text,
the accepted composition: 0.555 vs 0.824, difference −0.269 [−0.333, −0.199] pooled, and
−0.097 [−0.176, −0.016] in the background. Adding the text half hurts template recovery a
lot. That is expected, but the title hides it.

### P7. The bootstrap is approximate (minor)
Resampling gold groups with per-item precision and recall held fixed does not re-cluster.
It also does not update the sizes of predicted clusters when a group is drawn twice, and it
carries no label uncertainty. Labeller B against A moves the headline numbers by 0.01–0.03,
which is within the interval. The paired differences are internally consistent because each
draw applies the same weights to every arm, and re-reading the operating point inside each
draw is right. I expect the intervals to be somewhat too narrow. I do not think this
changes any conclusion; P1 and P2 matter more.

## 3. Things I checked and found sound
- **Clustering method.** I tried average, complete and single linkage, mutual 1- and 3-NN
  graph components, and image : text weightings of 0.25 to 1.5. None changes family-layer
  recall at precision 0.90 materially: 0.34 to 0.39 pooled for every variant
  (`recompute.json → alternative_methods`). HDBSCAN points in the plot agree. Within
  single-threshold flat clustering, the result does not come from method choice.
- **Nearest-neighbour check** (`neighbours.json`, not registered). It reproduces, and
  reposts do not inflate it:
  - Exact-hash duplicates are about 10% of hits.
  - Excluding the two largest families gives 0.63 (image + text) and 0.54 (image only);
    the macro average over families gives 0.60 and 0.55.
  - For items whose family spans more than one template, with same-template candidates
    excluded, the nearest neighbour is a family member for 0.43 (image + text), 0.33
    (image only) and 0.45 (text only). Chance is 0.01.

  This is consistent with the PR curves. The embedding does hold meaning-level neighbour
  information, and image + text holds more than image only. A single global threshold
  cannot use it at 90% purity, because unrelated-but-similar pairs (same subject, fill-in
  templates, textless images) sit at the same distances.
- **Family vs template structure.** Only 84 of the 745 same-family pairs share a template,
  and 171 of the 223 family-grouped items have a cross-template family member. The family
  layer really is about meaning, not relabelled templates, so the template-vs-meaning
  comparison is meaningful.

## 4. What the evidence does and does not support
Supported (on unvalidated agent labels):
- At the registered 90%-purity setting, flat clustering of either embedding keeps few
  meaning-level families together: about 21% of same-family pairs pooled and about 11% in
  the background, against 9% and 4% for the perceptual hash.
- Pooled, the embedding beats the hash at that setting (+0.08 [0.04, 0.12]). In the
  background the image + text advantage is not established (+0.04 [−0.01, 0.10]).
- For templates, image only is about as good as the hash, and image + text is clearly worse.

Not supported:
- "Adding text makes no difference." It depends on the operating point (P1) and is partly
  caused by a pipeline bug (P2).
- "The embedding does not capture meaning." The nearest-neighbour and UMAP evidence say it
  does, and the failure is in turning that into high-purity flat clusters.
- Any background-stratum verdict between arms. 23 families in the background alone is
  "not enough families to tell".

## 5. What I could not check
- Whether the labels are right. I looked at about 25 images only. User validation (P3) is
  the missing piece.
- Performance on the 42% of 2019 posts whose images are dead, on other years or
  subreddits, or for families the labellers missed entirely. Dead links probably remove old
  and large families unevenly, and the family stratum's cap of 6 per hash group shortens
  large families.
- A full bootstrap that re-clusters on each draw. I argued P7 qualitatively instead.
- Whether a non-flat method (for example per-cluster adaptive thresholds or a supervised
  similarity) would turn the neighbour signal into pure clusters. That is beyond a review.
