# WP-19 local VLM descriptions: environment and run commands

Machine: RTX 3090 24 GB, WSL2 Ubuntu (kernel 6.18 microsoft-standard), NVIDIA driver 610.88
(Windows side; `nvidia-smi` reports CUDA UMD 13.3), 16 logical CPUs, 62 GB RAM. No root, no
system CUDA toolkit (not needed: the wheels bring their own CUDA runtime).

## 1. Isolated serving environment (not the project's uv environment)

```bash
mkdir -p ~/tools/vlm-serve && cd ~/tools/vlm-serve
uv venv --python 3.12 .venv            # uv 0.11.32, CPython 3.12.13
uv pip install --python .venv/bin/python "vllm==0.34.0" --torch-backend=auto
```

Resolved versions (199 packages): vllm 0.34.0, torch 2.14.0+cu132, transformers 5.22.2,
xgrammar 0.2.8, flashinfer-python 0.7.3, httpx 0.29.0, huggingface_hub 2.5.0.
The project's `pyproject.toml` / `uv.lock` were not touched.

```bash
# Triton (used by vLLM at start-up) must compile a small C module, and the machine has no
# gcc/clang and no root. Use zig's bundled clang as the C compiler instead:
uv pip install --python .venv/bin/python ziglang          # ziglang 0.17.1
mkdir -p bin
printf '#!/bin/sh\nexec %s/.venv/bin/python -m ziglang cc "$@"\n'  "$PWD" > bin/cc
printf '#!/bin/sh\nexec %s/.venv/bin/python -m ziglang c++ "$@"\n' "$PWD" > bin/c++
chmod +x bin/cc bin/c++
# serve.sh exports CC=~/tools/vlm-serve/bin/cc CXX=~/tools/vlm-serve/bin/c++
```

Without this the engine dies after loading weights with
`RuntimeError: Failed to find C compiler` (log kept: `raw/qwen3vl8b_server_fail1_nocc.log`).
FlashInfer kernels are disabled for the same reason (no nvcc); vLLM falls back to the
FLASH_ATTN backend, which is what all numbers here used.

Run Python from this env with a neutral working directory (the scripts `cd /tmp`).
The helper scripts live in `~/tools/vlm-serve/scripts/` (`serve.sh`, `timed_run.sh`,
`verify_weights.py`); they are outside the repo, so copy them along if the environment moves.

## 2. Weights

```bash
uvx --from "huggingface_hub[hf_xet]" hf download Qwen/Qwen3-VL-8B-Instruct   # rev 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
uvx --from "huggingface_hub[hf_xet]" hf download Qwen/Qwen3-VL-4B-Instruct   # rev ebb281ec70b05090aa6165b016eac8ec08e71b17
.venv/bin/python -I scripts/verify_weights.py Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-4B-Instruct
```

`verify_weights.py` compares every safetensors shard's sha256 and size with the Hub's LFS
metadata; all 6 shards matched (log: `~/tools/vlm-serve/logs/verify.log`).

## 3. Server

`~/tools/vlm-serve/scripts/serve.sh <model> <log> [extra args]` runs:

```bash
CC=~/tools/vlm-serve/bin/cc CXX=~/tools/vlm-serve/bin/c++ OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
VLLM_NO_USAGE_STATS=1 vllm serve <model> --port 8000 --seed 0 \
  --max-model-len 4096 --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --mm-processor-cache-gb 0 --generation-config vllm  <extra args>
```

and waits for `/health`, logging time-to-ready (start-up cost) to `<log>.meta`.
Extra args used:

| tag | model | extra args |
|---|---|---|
| `qwen3vl8b` | Qwen/Qwen3-VL-8B-Instruct | `--dtype bfloat16 --gpu-memory-utilization 0.90 --max-num-seqs 128` |
| `qwen3vl8bfp8` | Qwen/Qwen3-VL-8B-Instruct | as above plus `--quantization fp8` (online weight-only FP8, Marlin kernels on Ampere) |
| `qwen3vl4b` | Qwen/Qwen3-VL-4B-Instruct | `--dtype bfloat16 --gpu-memory-utilization 0.90 --max-num-seqs 128` |

Image size: no pixel limit passed; the processor defaults (min 65,536 px, max 16.7 M px)
leave these images (longest side at most 768 px) at native resolution, at most 576 image
tokens, median about 414. The first start of each model spends about 2 minutes in
torch.compile; the result is cached in `~/.cache/vllm/torch_compile_cache/`.

## 4. Client and post-processing (run with the serving env's Python)

```bash
PY=~/tools/vlm-serve/.venv/bin/python
R=~/memes-revised/experiments/WP-19/raw
# GPU/load logger, 1 Hz, whole-GPU memory (WSL2 cannot attribute memory per process)
experiments/wp19/gpu_log.sh $R/<tag>_gpu.csv &
# warm-up (discarded), sweep, full run, retry
# ~/tools/vlm-serve/scripts/timed_run.sh <outbase> <args> wraps each call below and writes
# load average, nvidia-smi and top processes before/after the run to <outbase>.cond.txt
$PY experiments/wp19/describe_vlm.py --model <model> --ids-file $R/ids_warmup.txt --concurrency 16 --out $R/<tag>_warmup.jsonl --no-resume
$PY experiments/wp19/describe_vlm.py --model <model> --ids-file $R/ids_sweep.txt --concurrency <N> --out $R/<tag>_sweep_c<N>.jsonl --no-resume
$PY experiments/wp19/describe_vlm.py --model <model> --concurrency <best> --out $R/<tag>_full.jsonl
$PY experiments/wp19/collect_descriptions.py --raw $R/<tag>_full.jsonl --write-retry-ids $R/<tag>_retry_ids.txt
$PY experiments/wp19/describe_vlm.py --model <model> --ids-file $R/<tag>_retry_ids.txt --concurrency <best> --sample --schema --out $R/<tag>_retry.jsonl
$PY experiments/wp19/collect_descriptions.py --raw $R/<tag>_full.jsonl --retry $R/<tag>_retry.jsonl --out experiments/WP-19/descriptions_<tag>.json
```

Ids for warm-up (24) and sweep (192) are disjoint random subsets (Python `random.Random(19)`
shuffle of the sorted ids) in `raw/ids_warmup.txt`, `raw/ids_sweep.txt`.

Sweep concurrencies used: 4, 16, 32, 64, 128 (8B bf16 and 4B), 16, 32, 64, 128 (8B FP8), plus
one structured-output point (8B at 32, 4B at 128). Full runs: 8B bf16 at 32, 8B FP8 at 64,
4B at 128. Then:

```bash
python experiments/wp19/make_timing.py <tag> experiments/WP-19/raw/<tag>_settings.json
```

builds `timing_<tag>.json` from the raw files. (`collect_descriptions.py` and
`make_timing.py` use only the standard library and were run with system `python3`.)

## 5. Files

- `descriptions_<tag>.json`: 797 records `{id, text, template, referents, premise}` or `{id, error}`.
- `descriptions_<tag>.failures.json`: failure counts, first pass and retry.
- `timing_<tag>.json`: settings, start-up, sweep, full run, retry, conditions (load, GPU) per run.
- `raw/<tag>_*.jsonl`: every raw response with token counts and latencies; `raw/<tag>_server.log`;
  `raw/<tag>_gpu.csv` (1 Hz whole-GPU memory and utilisation, 1-minute load average).
