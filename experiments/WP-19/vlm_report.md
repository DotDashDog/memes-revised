# Local vision-language models for meme descriptions: speed and setup (WP-19)

Status: UNCERTIFIED (author only). Quality has not been scored; that is the next step, against
the reference labels, which I did not open.

## What was found

Both models run on the RTX 3090 with vLLM (a serving engine that batches many requests
continuously). All 797 images were described. **Every speed below is a lower bound.** Two
CPU-heavy jobs from another work package (`experiments.wp18.m3_sensitivity` and `m3_probe`)
used about 15 of the 16 cores the whole time, and the load average was 37–41 for every
timing run. The load never fell below 16 during the 45 minutes after the first measurement,
so no clean repeat was possible. Apart from those two jobs, nothing else was on the GPU
during the timing runs (checked with `nvidia-smi` before each run).

| | Qwen3-VL-8B, bf16 (full precision) | Qwen3-VL-8B, FP8 weights | Qwen3-VL-4B, bf16 |
|---|---|---|---|
| Speed over all 797 images, wall clock | **2.26 images/s** (concurrency 32) | **3.48 images/s** (concurrency 64) | **5.01 images/s** (concurrency 128) |
| Steady-state rate (10th to 90th percentile of completions) | 2.28 | 3.52 | 5.52 |
| Best steady-state rate seen in the sweep on 192 images | 2.48 | 4.66 | 8.5–9.0 |
| Requests actually running at once | about 13–14 (memory for cached attention state is full) | up to about 64 | about 95–100 |
| Input tokens per image (mean) | 686 (about 400 image + about 270 prompt) | 686 | 686 |
| Output tokens per image (mean / median / 99th percentile) | 94 / 84 / 273 | 93 | 90 / 79 / 254 |
| Output tokens per second | 212 | 322 | 450 |
| Peak GPU memory, whole card | 22.2 GB (vLLM reserves 90% of the card by design) | 21.4 GB | 21.9 GB |
| Valid on the first pass | 780 / 797 | 785 / 797 | 771 / 797 |
| Valid after one retry | 790 / 797 | 791 / 797 | 775 / 797 |
| 10 million images, at the 797-image wall-clock rate | **51 GPU-days** | **33 GPU-days** | **23 GPU-days** (14 at the best sweep rate) |

Start-up cost is separate from these rates:

- The first load of the 8B weights from the USB disk with a cold cache took 115 s (16.3 GiB,
  about 145 MB/s). That launch then failed for another reason; see "What was hard".
- Measured launches took 6–6.3 minutes to become ready, with weights already in RAM (4–8 s
  to load). Most of that time was a one-off compilation step: about 2 minutes, now cached on
  disk.

### Concurrency sweep

Each point is 192 images, after a warm-up of 24 images. Values are images/s, wall clock and
then steady-state. "Schema" means structured output was on.

| concurrency | 4 | 16 | 32 | 64 | 128 | 32 + schema | 128 + schema |
|---|---|---|---|---|---|---|---|
| 8B bf16 | 1.07 / 1.17 | 2.06 / 2.37 | 2.33 / 2.42 | 2.32 / 2.39 | 2.34 / 2.48 | 2.06 / 2.08 | |
| 8B FP8 | | 2.75 / 2.87 | 3.25 / 3.49 | 2.63 / 4.66 | 2.86 / 3.83 | | |
| 4B | 1.88 / 1.93 | 3.64 / 3.81 | 4.16 / 5.07 | 4.43 / 7.08 | 4.10 / 8.53 | | 3.86 / 8.99 |

How to read it:

- **8B bf16 is flat above 16.** It cannot run more than about 14 requests at once, so
  extra concurrency only adds queueing.
- **On 192 images the wall-clock rate is dragged down by one runaway output.** In almost
  every sweep run, one image produces 1,024 tokens of repetition and finishes alone at the
  end. That is why the 4B's wall-clock and steady-state rates differ so much in the sweep.
  Over all 797 images the gap is smaller.
- **The 4B's steady rate on the full run (5.5) is well below its sweep peak (8.5–9).** I
  cannot separate CPU contention, the different image set and run-to-run noise from these
  data. Treat the 4B figure as somewhere between 5 and 9 images/s under this load.
- **Structured output (JSON enforced by the server) cost about 11% on the 8B** (2.06 against
  2.33 at concurrency 32). On the 4B no cost was measurable. Structured output was not used
  for the main runs, only for the retry.

### Failures, all counted per model

On the first pass, decoding was greedy (deterministic) with no enforced JSON. Every failure
was retried once with the model card's sampling settings and server-enforced JSON.

| | 8B bf16 | 8B FP8 | 4B |
|---|---|---|---|
| Valid JSON with every field empty (text, premise and referents all blank) | 5 → 5 after retry | 6 → 6 | 21 → 21 |
| Invalid JSON (closing brace missing) | 7 → 0 | 3 → 0 | 0 |
| Truncated (runaway repetition up to 1,024 tokens) | 5 → 2 | 3 → 0 | 5 → 1 |
| Refusals | 0 | 0 | 0 |
| JSON wrapped in a code fence | 0 | 0 | 0 |
| **Final errors** | **7** | **6** | **22** |

- **The "everything empty" outputs are not refusals.** The five images that do this on both
  models are screenshots that are almost all text. For example, `ar4l63` is a nested chain of
  "look at this meme" / "I saw that earlier" messages. The model answers
  `{"text": "", "template": "none", "referents": [], "premise": ""}`, and sampling does not
  change that answer.
- **The 8B's invalid JSON is all one kind of slip.** It ends the output right after the
  premise and leaves out the final `}`. Appending one character would recover all of these
  without a retry.
- **The truncations are repetition loops**, such as "Biiiiii…" or "I I I I …" when
  transcribing stretched-out text.
- Records that still failed after the retry carry an `error` field such as
  `first:empty_fields; retry:empty_fields`. Counts are in `descriptions_<tag>.failures.json`.

## 10-million-image estimate and its assumptions

GPU-days = 10,000,000 / (images per second × 86,400 seconds per day).

- 8B bf16: 51 days
- 8B FP8: 33 days
- 4B: 23 days, or about 14 at its best sweep rate

The estimate assumes:

- one RTX 3090 with the same settings;
- images of the same size (longest side at most 768 px, about 400 image tokens);
- outputs of the same length (about 90 tokens);
- retries for about 1–3% of images, which adds about 2% (ignored);
- the same CPU contention as here. Without it the rates should be higher, by an amount I did
  not measure. GPU utilisation averaged about 80% while busy, so the GPU was not fully fed.

What would most plausibly speed it up, roughly in order of expected gain:

1. **Run with the CPU free.** The image decoding and scheduling share the 16 cores with
   whatever else is running. This is the cheapest test and the first thing to repeat.
2. **Use smaller weights for the 8B.** FP8 already gave +54%, because it frees memory for
   more requests in flight. A 4-bit checkpoint (AWQ or GPTQ) would free more but needs a new
   download, which I did not do. An FP8 cache for the attention state would roughly double
   the number of requests in flight. I have not tested it.
3. **Cap the output length at about 400 tokens instead of 1,024.** The longest legitimate
   output was 327 tokens. Runaway repetitions are what make the wall-clock rate fall below
   the steady-state rate.
4. **Fewer image tokens.** Capping images at about 256 tokens instead of about 400 would cut
   processing of the input, at a cost to legibility of small text. This has to be checked
   against quality first.
5. **Put the fixed prompt text before the image.** The roughly 270 identical prompt tokens
   per image could then be cached once and reused (the cache hit rate is about 0% now). This
   changes the input the model sees, so it needs a quality check too.
6. **More or bigger GPUs.** The work splits by image, so time scales roughly linearly with
   the number of GPUs.

## How the image size was chosen

No limit was set, so every image is processed at native resolution. Qwen3-VL turns each
32×32-pixel patch into one token, and the sample images are already at most 768 px on the
longest side. That gives at most 576 image tokens, with a median of about 414. The model's
default minimum of 256×256 pixels enlarges only 8 tiny images.

Reducing resolution would make small meme text harder to read. Input processing is not the
bottleneck here: the server handled 1,700–5,800 input tokens/s against 200–650 output
tokens/s. So I kept native size. A lower cap is the fourth speed-up idea above and needs a
quality comparison first.

## What looked wrong in about 20 outputs per model

I read 20 random outputs per model, plus counts over all of them. This is my reading only;
it has not been checked against the labels.

**8B (bf16):**
- **Templates are mostly descriptions rather than the conventional names.**
  - Expanding Brain became "Four-panel meme with escalating brain imagery".
  - Monkey Puppet became "two panels of a monkey puppet looking sideways".
  - The "LET ME IN" meme (Eric Andre) became "Man in suit yelling at gate".
  - Some names are right, e.g. "Rick Astley".
- **Invented or wrong referents.**
  - "Drake Hotline Bling" for a two-panel "ME AT 6 / ME A TEENAGER" image that does not look
    like Drake (`b25fnj`).
  - "Will Smith" for a "No, I don't think I will" meme.
  - A run of literal referents for `e3oyfo` ("Yoda's green robe", "Yoda's pointed ears", …)
    before it truncated.
- **Transcription errors.** "Magin Buu" for "Majin Buu"; stretched words become repetition
  loops.
- **Premises are often generic or mislabel the joke.** About 45 of 790 premises start "The
  meme…/The joke…" and 80 contain "humor". The name meme `cb3sg4` is explained as a correction
  of a mispronunciation.
- **A few popularity remarks despite the instruction.**
  - `dl6az7`: "…which is surprising because they are popular".
  - `b8rkgm`: "referencing his famous…".
  - `ccwaq1`: "posting a popular meme".
  - Other keyword hits are about memes whose subject is old or dead memes, which is
    legitimate content.

**4B:** much the same problems, more often.
- **It gives up on the template far more often.** "none" appears 118 times against the 8B's
  4.
- **Premises are often a literal description of the picture rather than the joke.** For
  example: "A person pours coffee into a mug while being labeled as 'Quality Memes' and
  'Reddit'".
- **More wrong referents and template names.**
  - "Scooby-Doo, Thor, Bender" for `akxwm4`.
  - "Avengers: Endgame Thanos vs. Iron Man" as a template.
  - "Linda pouring coffee".
- **It has 4× as many empty outputs as the 8B** (21 against 5).
- Popularity keyword hits: 6, mostly legitimate subject matter.

**8B FP8 against bf16:**
- The transcribed text is identical for 92% of images.
- Template wording is identical for 58%, and the premise for 28%.
- So quantisation changes the wording a lot. Whether it changes the quality has to be
  scored; it should not be assumed.

## What was hard about setup

- **The previous agent's state was much less complete than described.**
  - The serving environment held only Python and pip; vLLM had never been installed.
  - The 8B download had one complete shard out of four; the 4B had none.
  - `raw/` was empty.
- **Both downloads resumed quickly.** All six weight shards match the Hub's sha256 checksums.
- **The machine has no C compiler and no root.** vLLM's kernel compiler (Triton) needs one
  to build a small helper at start-up, and the first launch died with
  `Failed to find C compiler`. I installed zig's bundled clang (the `ziglang` package) in the
  isolated environment and pointed `CC`/`CXX` at it.
- **One kernel library could not be used.** FlashInfer is disabled because there is no CUDA
  compiler. vLLM uses FlashAttention instead, and all numbers were taken that way.
- **The full-precision 8B leaves only about 2.8 GiB for the attention cache in 24 GB.** That
  limits it to about 14 requests in flight. I therefore also ran an FP8-weight version of
  the same 8B checkpoint. It was quantised by vLLM at load time; nothing new was downloaded.
  You allowed this if batching was poor; I report it alongside the full-precision run rather
  than instead of it.
- **The GPU was fully occupied by another job for about two hours before measurements could
  start.** After that, the CPU jobs described above ran throughout.

## Files

All under `/home/linus/memes-revised/experiments/WP-19/`:

- **Descriptions** (797 records each):
  - `descriptions_qwen3vl8b.json`
  - `descriptions_qwen3vl4b.json`
  - `descriptions_qwen3vl8bfp8.json` (extra, FP8 weights)
- **Failure counts:** `descriptions_*.failures.json`.
- **Timing:** `timing_qwen3vl8b.json`, `timing_qwen3vl4b.json`, `timing_qwen3vl8bfp8.json`.
  Each holds the settings, start-up, sweep, full run, retry, failures, and load and GPU state
  before and after every run.
- **`setup.md`:** exact commands and versions.
- **`raw/`:** every raw response, server logs, the 1 Hz GPU and load logs, and the id lists
  for warm-up and sweep.
- **Code in `experiments/wp19/`:**
  - `describe_vlm.py`: added the steady-state rate, latency percentiles and the `--sample`
    retry mode.
  - `collect_descriptions.py`: "all fields empty" is now separated from "refusal".
  - `make_timing.py`: new.
- **Outside the repo:** `~/tools/vlm-serve/` holds the environment, plus `scripts/serve.sh`,
  `timed_run.sh` and `verify_weights.py`, and the zig compiler wrappers in `bin/`.
