# 0018 — Separate image and text channels; sentence model for the text

**Status:** accepted
**Date:** 2026-11-28
**Accepted:** 2026-11-28 — user ("I'm good with your recommendation #1"), on the evidence below
**Amends:** [0006](0006-ocr-vs-vlm.md) (which text encoder; how the two signals are held) and
[0017](0017-no-title-in-clustering-embedding.md) (its "image ⊕ OCR" concatenation)
**Informs:** WP-04 (audit), WP-05 (pipeline build), 0009 (clustering), WP-10 (predictor inputs)

## Context

ADR 0006 chose a hybrid: an OpenCLIP image embedding plus an embedding of the text that
EasyOCR extracts from the image. ADR 0017 fixed the composition for meme identity as the two
vectors concatenated, both produced by the same OpenCLIP model (its image tower and its text
tower). The clustering audit (WP-04) and its diagnostic, on 797 hand-checked images from
2019, found two things wrong with that composition:

- **The CLIP text tower is a weak encoder of meme text.** The same extracted text embedded
  with a general sentence model (`BAAI/bge-base-en-v1.5`) ranks same-meme pairs better:
  0.86 against 0.77 on agent labels (probability that a same-meme pair is more similar than
  another pair), and 0.77 against 0.68 for the whole concatenated pipeline on the user's own
  99 same-meme answers.
- **Concatenating the image vector lets the picture outvote the meaning.** With the image
  vector appended, a different-meme pair on the same template is more similar than a
  same-meme pair on different templates 72% of the time. Appending the image vector to
  written descriptions lowers every meaning score measured (for example 0.92 to 0.75 on the
  user's answers for the best descriptions). For recognising a *template* the image vector
  alone is the best of the three (0.77 of each template kept together at 90% purity, against
  0.56 for the concatenation).

The image answers "which template is this"; the text answers "what is it saying". Forcing
them into one vector makes each worse at its own job.

## Decision

1. **Two channels, kept separate.** Each meme image has an *image channel* (OpenCLIP
   ViT-B-16 `laion2b_s34b_b88k` image embedding, unchanged) and a *text channel*. They are
   stored and passed around as two vectors. Nothing downstream may assume a single
   concatenated vector.
2. **The text channel is embedded with a sentence model,** currently
   `BAAI/bge-base-en-v1.5`, not with the CLIP text tower. Its input today is the EasyOCR
   transcript, as in ADR 0006.
3. **An image with no text has no text vector.** It gets a zero vector and a `has_text`
   flag set to false. (The old code embedded the literal string "[empty]" for every
   text-free image, which made all of them look alike on the text side.)
4. ADR 0017 stands as to *what* goes in: post title and body stay out of meme identity.

Implemented as `embed_channels` in `experiments/wp01/pipelines.py`. The old
`embed_pipeline` compositions stay in the file only so that recorded runs can be reproduced.

## What this does not decide

- **How the two channels are used to group memes.** That depends on what counts as "one
  meme" (open, with the user) and on ADR 0009. Template identity can use the image channel
  alone today.
- **What the text channel's input should be in the end.** Descriptions written by a
  vision-language model score higher than extracted text and may replace or join it; that
  is under test (WP-04 log, local description models) and would be a further decision.
- **Which sentence model.** `bge-base-en-v1.5` is the one that was measured. Swapping it
  for another general sentence model is an implementation choice, provided the same
  comparison on the audit sample is rerun and recorded.
- **How the predictor combines the channels** (WP-10).

## Alternatives considered

- **Keep the concatenated CLIP pipeline.** Rejected on the measurements above.
- **Swap the text encoder but keep concatenating.** Keeps the second problem: any fixed
  concatenation sets an arbitrary exchange rate between picture and meaning.
- **Drop the image channel.** Rejected: it is the best available signal for template
  identity, and a meme's template may itself matter for how it spreads.
- **Wait for the description-model question to be settled.** Rejected: the swap is free,
  uses text already being extracted, and nothing about it is undone by adding descriptions
  later.

## Consequences

- **No recorded result is invalidated.** The audit's scores for the old composition stand
  as measurements of the old composition. New work uses the two channels.
- **Contamination window (ADR 0004).** The evaluation period must start after the training
  data of every encoder in use. `bge-base-en-v1.5` was released in September 2023, later
  than the OpenCLIP checkpoint (data collected through about 2022), so this moves the
  earliest safe start of the evaluation period later by about a year. The archive runs to
  December 2025, so there is still room, but ADR 0011 (the split) must take the later date.
- **Cost.** The sentence model adds a second small model at embedding time (CPU is enough).
  Text extraction itself (EasyOCR, 1-2 images per second) is unchanged and remains the slow
  step.
- The measured bug in (3) is fixed for new work; runs that used the old function keep the
  old behaviour and are labelled as such.

## Evidence

- WP-04 log: scored run, independent review, diagnostic (2026-11-27), and local description
  models (2026-11-28). Numbers above are from `experiments/WP-04/sample-001/diagnose_failure.json`,
  `experiments/WP-19/description_eval.json` and `description_followup.json`.
- Confidence: the encoder comparison is author-computed and has not been independently
  reviewed; the user-answer comparison rests on 99 pairs (0.77, interval 0.67-0.86, against
  0.68, interval 0.57-0.78). The direction is consistent across every reference used.
