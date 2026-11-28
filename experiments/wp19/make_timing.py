"""Assemble experiments/WP-19/timing_<tag>.json from the raw run files.

Usage: python make_timing.py <tag> <settings.json> [--variant-tags t1,t2]
Reads raw/<tag>_{warmup,sweep_c*,full,retry}.jsonl.timing.json, the *.cond.txt files
(load average and GPU state before/after each run), raw/<tag>_server.log(.meta), the
1 Hz GPU log raw/<tag>_gpu.csv, and descriptions_<tag>.failures.json.
"""

import glob
import json
import pathlib
import re
import sys

WP = pathlib.Path(__file__).resolve().parents[1] / "WP-19"
RAW = WP / "raw"


def tj(path):
    t = json.loads(pathlib.Path(path).read_text())
    keep = ["concurrency", "schema", "sampling", "n_requests", "n_http_ok", "wall_s", "images_per_s",
            "images_per_s_steady_p10_p90", "mean_prompt_tokens", "mean_completion_tokens",
            "output_tok_per_s", "latency_p50_s", "latency_p95_s", "finish_length",
            "loadavg_1_5_15_start", "loadavg_1_5_15_end", "started_at"]
    out = {k: t.get(k) for k in keep}
    cond = pathlib.Path(str(path).replace(".jsonl.timing.json", ".cond.txt"))
    if cond.exists():
        out["conditions_before_after"] = cond.read_text().strip().splitlines()
    return out


def main():
    tag, settings_path = sys.argv[1], sys.argv[2]
    settings = json.loads(pathlib.Path(settings_path).read_text())
    res = {"model_tag": tag, "settings": settings}

    meta = (RAW / f"{tag}_server.log.meta").read_text()
    log = (RAW / f"{tag}_server.log").read_text()
    m = re.search(r"startup_s ([\d.]+)", meta)
    res["startup"] = {
        "launch_to_ready_s": float(m.group(1)) if m else None,
        "meta": meta.strip().splitlines(),
        "server_log_lines": [l.split("] ", 1)[-1] for l in log.splitlines()
                             if re.search(r"Loading weights took|Model loading took|torch.compile took|"
                                          r"Available KV cache|GPU KV cache size|init engine|Graph capturing", l)],
    }

    gpu = [l.split(",") for l in (RAW / f"{tag}_gpu.csv").read_text().splitlines()[1:] if l.count(",") == 3]
    res["peak_gpu_memory_mib_whole_card"] = max(int(g[1]) for g in gpu) if gpu else None
    res["peak_gpu_memory_note"] = ("whole-GPU nvidia-smi reading at 1 Hz over the server's life (WSL2 cannot "
                                   "attribute memory per process); includes ~0.7 GB used by the Windows desktop. "
                                   "vLLM pre-allocates gpu_memory_utilization x 24 GB, so this reflects the setting, "
                                   "not the minimum needed.")

    res["warmup"] = tj(RAW / f"{tag}_warmup.jsonl.timing.json")
    res["sweep_192_images"] = [tj(p) for p in sorted(glob.glob(str(RAW / f"{tag}_sweep_c*.jsonl.timing.json")),
                                                      key=lambda p: (("schema" in p), int(re.search(r"_c(\d+)", p).group(1))))]
    res["full_797"] = tj(RAW / f"{tag}_full.jsonl.timing.json")
    if (RAW / f"{tag}_retry.jsonl.timing.json").exists():
        res["retry"] = tj(RAW / f"{tag}_retry.jsonl.timing.json")
    fp = WP / f"descriptions_{tag}.failures.json"
    if fp.exists():
        res["failures"] = json.loads(fp.read_text())
    for extra in sorted(glob.glob(str(RAW / f"{tag}_rerun*.jsonl.timing.json"))):
        res.setdefault("repeat_timing_runs", []).append(tj(extra))
    (WP / f"timing_{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("startup", "peak_gpu_memory_mib_whole_card")}, indent=1))


if __name__ == "__main__":
    main()
