"""Describe meme images with a locally served VLM (vLLM OpenAI-compatible server).

Runs in the isolated serving environment (~/tools/vlm-serve/.venv), not the project env.

Each image is sent as a base64 JPEG data URL together with the prompt in
experiments/wp19/describe_prompt.md (verbatim). Results are appended to a JSONL file
as they arrive (one line per request), so an interruption loses nothing; a rerun skips
ids already present in the JSONL.

Usage:
  python describe_vlm.py --model Qwen/Qwen3-VL-8B-Instruct --out run.jsonl \
      --concurrency 32 [--limit N] [--ids-file ids.txt] [--schema] [--max-tokens 1024]

Timing: wall clock from first request sent to last response received is written to
<out>.timing.json, together with per-request latency and token usage.
"""

import argparse
import asyncio
import base64
import json
import os
import pathlib
import time

import httpx

REPO = pathlib.Path(__file__).resolve().parents[2]
IMG_DIR = REPO / "experiments/WP-04/sample-001/label_images"
PROMPT = (REPO / "experiments/wp19/describe_prompt.md").read_text()
# Per-image context for prompts that contain "{when}" (the posting month); set in main().
WHEN: dict[str, str] = {}

SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "template": {"type": "string"},
        "referents": {"type": "array", "items": {"type": "string"}},
        "premise": {"type": "string"},
    },
    "required": ["text", "template", "referents", "premise"],
    "additionalProperties": False,
}


def build_body(model, img_b64, max_tokens, schema, sampling=None, mid=None):
    prompt = PROMPT.replace("{when}", WHEN[mid]) if "{when}" in PROMPT else PROMPT
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 0,
    }
    if sampling:  # used for the retry pass: greedy decoding would just repeat a failure
        body.update(sampling)
    if schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "meme_description", "schema": SCHEMA},
        }
    return body


async def one(client, url, sem, model, mid, max_tokens, schema, fh, lock, sampling=None):
    img_b64 = base64.b64encode((IMG_DIR / f"{mid}.jpg").read_bytes()).decode()
    body = build_body(model, img_b64, max_tokens, schema, sampling, mid)
    async with sem:
        t0 = time.perf_counter()
        rec = {"id": mid, "t_send": t0}
        try:
            r = await client.post(url, json=body, timeout=1800)
            r.raise_for_status()
            j = r.json()
            ch = j["choices"][0]
            rec.update(
                content=ch["message"]["content"],
                finish_reason=ch.get("finish_reason"),
                prompt_tokens=j["usage"]["prompt_tokens"],
                completion_tokens=j["usage"]["completion_tokens"],
            )
        except Exception as e:  # recorded, counted as a failure, retried later
            rec["http_error"] = repr(e)[:500]
        rec["t_recv"] = time.perf_counter()
        rec["latency_s"] = rec["t_recv"] - t0
    async with lock:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
    return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--sample", action="store_true",
                    help="Qwen3-VL model-card VL sampling (T=0.7, top_p=0.8, top_k=20, presence 1.5) instead of greedy")
    ap.add_argument("--prompt", default=None, help="prompt file (default describe_prompt.md)")
    ap.add_argument("--when-file", default=None,
                    help='JSON {id: "Month YYYY"}; required if the prompt contains "{when}"')
    a = ap.parse_args()

    global PROMPT
    if a.prompt:
        PROMPT = pathlib.Path(a.prompt).read_text()
    if "{when}" in PROMPT:
        WHEN.update(json.loads(pathlib.Path(a.when_file).read_text()))

    if a.ids_file:
        ids = [l.strip() for l in open(a.ids_file) if l.strip()]
    else:
        ids = sorted(p.stem for p in IMG_DIR.glob("*.jpg"))
    if a.limit:
        ids = ids[: a.limit]
    out = pathlib.Path(a.out)
    done = set()
    if out.exists() and not a.no_resume:
        done = {json.loads(l)["id"] for l in out.open() if l.strip()}
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} ids, {len(done)} already done, {len(todo)} to do", flush=True)

    sampling = ({"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}
                if a.sample else None)
    url = f"http://localhost:{a.port}/v1/chat/completions"
    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    limits = httpx.Limits(max_connections=a.concurrency + 8, max_keepalive_connections=a.concurrency + 8)
    with out.open("a") as fh:
        async with httpx.AsyncClient(limits=limits) as client:
            load_start = os.getloadavg()
            t0 = time.perf_counter()
            recs = await asyncio.gather(
                *(one(client, url, sem, a.model, i, a.max_tokens, a.schema, fh, lock, sampling) for i in todo)
            )
            wall = time.perf_counter() - t0
            load_end = os.getloadavg()

    ok = [r for r in recs if "content" in r]
    # Steady-state rate: completions between the 10th and 90th percentile of finish
    # times, which excludes ramp-up and the tail where the batch drains.
    fin = sorted(r["t_recv"] - t0 for r in recs)
    steady = None
    if len(fin) >= 20:
        lo, hi = int(0.1 * len(fin)), int(0.9 * len(fin))
        steady = (hi - lo) / (fin[hi] - fin[lo]) if fin[hi] > fin[lo] else None
    lat = sorted(r["latency_s"] for r in recs)
    timing = {
        "images_per_s_steady_p10_p90": steady,
        "latency_p50_s": lat[len(lat) // 2] if lat else None,
        "latency_p95_s": lat[int(0.95 * (len(lat) - 1))] if lat else None,
        "model": a.model,
        "concurrency": a.concurrency,
        "schema": a.schema,
        "sampling": sampling or "greedy",
        "max_tokens": a.max_tokens,
        "n_requests": len(recs),
        "n_http_ok": len(ok),
        "wall_s": wall,
        "images_per_s": len(recs) / wall if wall else None,
        "prompt_tokens_total": sum(r["prompt_tokens"] for r in ok),
        "completion_tokens_total": sum(r["completion_tokens"] for r in ok),
        "mean_prompt_tokens": sum(r["prompt_tokens"] for r in ok) / max(len(ok), 1),
        "mean_completion_tokens": sum(r["completion_tokens"] for r in ok) / max(len(ok), 1),
        "output_tok_per_s": sum(r["completion_tokens"] for r in ok) / wall if wall else None,
        "mean_latency_s": sum(r["latency_s"] for r in recs) / max(len(recs), 1),
        "finish_length": sum(r.get("finish_reason") == "length" for r in ok),
        "loadavg_1_5_15_start": load_start,
        "loadavg_1_5_15_end": load_end,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - wall)),
    }
    pathlib.Path(str(out) + ".timing.json").write_text(json.dumps(timing, indent=1))
    print(json.dumps(timing, indent=1), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
