# Gold-set labelling protocol (clustering audit, WP-04)

Frozen 2026-11-27, before any item of the gold sample was looked at for labelling. Changes
after this date are appended at the bottom with a date and a reason, never edited in place.

## What is being labelled, and why

The audit asks whether the project's embedding groups meme posts the way a person would. To
score that we need a reference grouping of a sample of posts. Each post gets two labels:

- **Template**: the base image or visual format the post is built on, whatever its caption.
  All "Drake approves / disapproves" two-panel posts share one template.
- **Family**: the meme as something that spreads, judged by meaning. Two posts are in the same
  family when they are instances of the same specific joke, trend, event or catchphrase, so
  that seeing one makes the other recognisably "that meme again". The family layer is the one
  the project cares about; the template layer is there to measure how much of any apparent
  success is only picture matching.

The two layers are independent. One template carrying unrelated jokes is one template and
several families. One trend told through different templates (the 2019 "storm Area 51" jokes
on a dozen formats) is several templates and one family.

## Who labels

Agents label every item. The user labels a subsample as pairwise same/different judgments,
and agent labels are used for scoring only if they agree with the user well enough (criterion
at the bottom). This replaces the original plan of the user hand-labelling everything, by the
user's decision on 2026-11-27.

## Blindness

Labellers see the image only. They do not see the post title, the subreddit metadata, any
embedding, any cluster assignment, any similarity score from the models under audit, or the
labels another labeller gave the same item. Perceptual-hash (dhash) groups may be shown as
candidates for the template layer only; they are never shown for the family layer.

## Stage 1: describe each image independently

For each image, one agent that has seen no grouping information writes:

- `text`: the text in the image, transcribed.
- `template`: a short conventional name for the base image or format if it has one
  ("Drake Hotline Bling", "Lisa Simpson presentation", "Surprised Pikachu"), else a
  five-to-ten-word description of the base image; `none` for plain text, screenshots, or
  one-off original images with no reused base.
- `referents`: the specific things the joke is about or depends on: named events, trends,
  catchphrases, releases, in-jokes ("Area 51 raid", "No Nut November", "cake day", "OK
  boomer", "Minecraft's comeback versus Fortnite"). Empty if the joke is self-contained.
- `premise`: one sentence stating the joke in plain words.
- `notes`: anything that makes the item hard to classify.

## Stage 2: resolve labels across the sample

Template layer. Candidate groups come from identical or near-identical `template` names and
from dhash groups. Each candidate group is confirmed by looking at its images side by side.
Same base image with crops, recolours, added panels or different captions is the same
template. A redrawn or parodied version that a viewer would still call the same format is
the same template. Items with `template: none` are each their own template.

Family layer. Candidate groups come from shared `referents` and near-identical `premise`
only. Each candidate group is confirmed by looking at its images side by side. Rules:

1. Same specific referent or same specific joke premise: same family, whatever the template.
2. A broad subject area is not a family. "School", "gaming", "relationships", "Reddit",
   "reposts are bad" are subjects. "Storm Area 51", "No Nut November", "the Minecraft versus
   Fortnite reversal of 2019" are families.
3. A template by itself is not a family, with one exception: when the template was itself the
   trend and its posts tell the same joke with small substitutions (the caption is a fill-in
   slot, as in "Me and the boys ..."), posts using it in that standard way are one family.
   When in doubt, ask whether the captions would be interchangeable to someone scrolling past.
4. An exact or near-exact repost is in the family of the original.
5. Everything else is `unique`: a family of one. `unique` is a decision, not a default; it
   is given only after the item has been checked against the candidate groups.
6. When an item could reasonably go either way, assign the best guess and set `uncertain`.
   Uncertain items are scored, and results are also reported with them removed.

## Self-consistency check

A seeded 10% of items are described and resolved a second time by a fresh agent that has not
seen the first pass. Disagreement between passes is reported per layer as a bound on label
noise.

## User validation, and the criterion for using agent labels

The user is shown pairs of images from the sample and answers two yes/no questions per pair:
same template? same family (same specific meme by meaning)? The user sees no agent labels.
Pairs are drawn in three strata so that the informative cases are not swamped by trivially
different ones:

- pairs the agents put in the same family;
- hard negatives: same template but different family, or shared referent words but different
  family;
- random pairs.

Criterion, fixed here before any labels exist. Agent labels are used for scoring a layer if,
on that layer, (a) of the pairs agents call "same", the user agrees on at least 85%, and (b)
of the hard-negative pairs agents call "different", the user agrees on at least 85%, each
with the lower end of a 95% interval above 75%. Reasoning: the audit's planned resolution on
the clustering score is about ±0.05, and the comparison of interest (embedding versus a plain
perceptual hash) is only meaningful if it exceeds the reference labels' own error. At 85%
pairwise agreement in the informative strata, label error is comparable to that resolution;
below 75% it would dominate, and the labels would measure the labeller, not the embedding.
If a layer fails, its scores are reported as unvalidated and the disagreeing pairs are shown
to the user as the finding.

Known limit, stated up front: an agent labeller and the embedding model under audit were both
trained on large image-text corpora and may share blind spots, so agreement between them is
weaker evidence than agreement with a person. The user's pairs are the only guard, and the
random-pair stratum exists to catch families the agents failed to see at all.
