# 0006 — OCR pipeline vs. VLM-native embedding

**Status:** accepted
**Date:** 2026-05-19
**Gates:** [WP-01](../work-packages/WP-01-ocr-vs-vlm-benchmark.md)

## Context

Memes carry meaning in (a) the image, (b) the post text, and (c) text rendered onto the image. The prior pipeline ran a dedicated OCR step to extract (c) and then embedded image and OCR-text separately. OCR on memes has proved hard (uneven backgrounds, stylized fonts, varying layouts) and is the current roadblock.

Modern vision-language models may handle some in-image text natively, removing the need for an explicit OCR step. WP-01 tested this directly with OpenAI CLIP ViT-B/32 and OpenCLIP ViT-B-16 (`laion2b_s34b_b88k`) on a cached r/memes pilot with manual in-image transcripts.

## Decision

Adopt **Option B — Hybrid** for the project embedding pipeline:

- Image signal: VLM image embedding, with OpenCLIP ViT-B-16 (`laion2b_s34b_b88k`) as the current primary local encoder.
- Text-in-image signal: filtered EasyOCR (`easyocr_filtered:none`) transcript embedding.
- Post text signal: embed post title/body alongside OCR text.

VLM-native-only is not sufficient for this project phase because CLIP/OpenCLIP image embeddings did not faithfully retrieve the exact in-image caption against hard text negatives. OCR-only is also not sufficient because meme meaning is inherently multimodal and the image signal remains important.

## Alternatives considered

(See WP-01 for full experimental design.)

## WP-01 evidence

2026-05-19 final WP-01 pilot:

- Implemented the WP-01 benchmark harness for OpenAI `CLIP ViT-B/32` and OpenCLIP `ViT-B-16` (`laion2b_s34b_b88k`).
- Ran a fully local synthetic Impact-caption probe in `experiments/WP-01/report.md` to validate image-to-text retrieval, text-to-image retrieval, hard negatives, and the three pipeline clustering path.
- Replaced the synthetic top-level artifacts with a cached real r/memes pilot: 24 downloaded images, 8 per coarse stratum, with manual visual in-image transcripts.
- Real pilot result: OpenCLIP B/16 retrieved exact captions weakly in image-to-text ranking (top-1 16.7%, minimal-pair accuracy 64.8%), while text-to-image retrieval was stronger (top-1 91.7%).
- OCR comparison selected `easyocr_filtered:none`: CER 0.291, WER 0.481, token F1 0.802 on the 24 manual transcripts. PaddleOCR had higher token F1 (0.859) but was ~3-4x slower and visually noisier in the inspection gallery.
- Pipeline clustering on 24 items remained inconclusive (all noise), so WP-01 decides the architecture from text-fidelity and OCR-quality evidence, not from cluster-quality evidence.

## Consequences

- The embedding pipeline should retain a dedicated OCR transcript channel.
- Production embedding should use `easyocr_filtered:none` until a larger audit shows a better OCR option.
- The exact modality combination strategy (concatenation, weighting, learned projection) is deferred to a follow-up decision after larger-scale embeddings exist.
- A larger cached-image audit should be run before making strong claims about cluster quality or OCR error rates across all r/memes.

## Open sub-questions

- How exactly should the VLM image, OCR text, and post text embeddings be combined for the first predictor run?
- Does OpenCLIP's approximate early/mid-2022 training-pool cutoff satisfy the final ADR 0004/0011 contamination-control split, or should a stricter post-cutoff eval window be used?
