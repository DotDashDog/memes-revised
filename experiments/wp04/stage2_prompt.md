You are building reference labels for a research audit: grouping 797 meme images from
r/memes (2019) by template and, separately, by meaning. The labels will be used to judge
whether a model's clusters match human judgment, so they must come from your own reading of
the images and must not be contaminated by any model output.

Read `/home/linus/memes-revised/experiments/wp04/labeling_protocol.md` first; the sections
"What is being labelled", "Blindness" and "Stage 2" are your rules. Stage 1 is done: each
image has an independent written description in the descriptions file named at the end of
this message (fields: id, text, template, referents, premise, notes). The descriptions were
written by several people working separately, so the same template or referent will appear
under different wordings ("Eric Andre let me in" / "let me in Eric Andre"), and some
descriptions will be wrong. They are a search aid. The images are the evidence.

Files you may open: the protocol, your descriptions file, the images at
`/home/linus/memes-revised/experiments/WP-04/sample-001/label_images/<id>.jpg`, and
`/home/linus/memes-revised/experiments/WP-04/sample-001/stage2/hash_groups_template_candidates.json`
(groups of visually near-identical images found by a perceptual hash; candidates for the
template layer only, and they can be wrong). Open nothing else in the repository: no other
file in the sample folder, no docs, no other labeller's output. No web search.

To compare images side by side, make a contact sheet and view it:
`cd /home/linus/memes-revised && uv run python -m experiments.wp04.sheet <out.png> <id> <id> ...`
(up to about 15 ids per sheet stays legible). Put sheets and any working files in the
working folder named at the end of this message.

What to produce, for every one of the 797 ids:
- `template_id`: items built on the same base image or format share an id. Give groups a
  short readable id (`drake-hotline-bling`); items with no reused base get a unique id
  (`none-<id>`).
- `family_id`: items that are the same meme by meaning share an id (`area-51-raid`), following
  the protocol's rules, including that a broad subject is not a family and a template alone
  is not a family except where the template itself is the joke. Items in no family get
  `unique-<id>`.
- `uncertain_template`, `uncertain_family`: true where the call could reasonably go the other
  way.
- `why`: for any item placed in a family of two or more, a few words on what makes it that
  family.

How to work. Build candidate groups from the descriptions (normalise wordings; look at shared
referents, similar premises, shared distinctive phrases in the text), then confirm every
candidate group of two or more by looking at its images together, splitting or merging as the
images warrant. The referent vocabulary is uneven in grain: "Avengers Endgame" as a referent
covers both a specific running joke (say, spoiler-avoidance) and unrelated jokes that merely
mention the film; the protocol's test is whether the posts are the same specific joke or
trend, so look and decide. Then make a second pass in the other direction: for items you have
left as unique, scan their text and premise against the families you formed, since the easy
error is a family member whose description used different words. Do not stop at the
description-level grouping; a label set that was never checked against images is not
acceptable. Expect most items to be unique in the family layer.

Save progress to disk as you go. Write the final labels as a JSON array of objects
`{id, template_id, family_id, uncertain_template, uncertain_family, why}` to the output path
named below, check that it parses and has exactly 797 distinct ids, and reply with: the
number of multi-member templates and families and the items in them, the five largest
families with sizes, the kinds of case you found hardest to call and how you called them, and
anything in the protocol that was ambiguous in practice.
