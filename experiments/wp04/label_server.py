"""WP-04 §4: blind two-layer labeling server (localhost, keyboard-driven).

Blindness protocol (frozen before labeling starts — see WP-04 §4):
- The labeler sees the image and its OCR text ONLY. No titles, no cluster
  assignments, no embedding-similarity suggestions.
- Two label layers per unit:
    template_id — appearance lineage (same base image/format). The UI shows a
      dhash-component *suggestion* for this layer only (sanctioned scaffolding).
    family_id — meaning ("same meme as a spread phenomenon"). NO suggestions;
      recall is labeler-driven via the searchable catalog (thumbnails + OCR
      substring). `unique` is an explicit key, not a default.
- Byte-identical images (same sha256) are collapsed to one labeling unit.
- Items arrive in seeded random order. Every decision is appended to a JSONL
  on the NVMe (crash-safe; restart resumes). After the main pass, a seeded 10%
  subsample is served again as a relabel pass for self-agreement, with prior
  labels hidden.

Usage:
    uv run python -m experiments.wp04.label_server \
        --sample experiments/WP-04/sample-001 --port 8765
    # expects <sample>/images/{post_id}.{ext}, <sample>/ocr.json,
    # <sample>/units.parquet (unit_id | post_id | dhash_component)
    # labels appended to <sample>/labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ORDER_SEED = 42
RELABEL_SEED = 43
RELABEL_FRACTION = 0.10
LAYERS = ("template", "family")


class Store:
    def __init__(self, sample_dir: Path) -> None:
        self.dir = sample_dir
        self.images = {p.stem: p for p in (sample_dir / "images").iterdir()}
        self.ocr = json.loads((sample_dir / "ocr.json").read_text())
        import pandas as pd

        units = pd.read_parquet(sample_dir / "units.parquet")
        # one labeling unit per sha256; representative post_id must have image+ocr
        self.units = [
            {
                "unit_id": r.unit_id,
                "post_id": r.post_id,
                "dhash_component": int(r.dhash_component),
            }
            for r in units.itertuples()
            if r.post_id in self.images
        ]
        rng = random.Random(ORDER_SEED)
        rng.shuffle(self.units)
        self.labels_path = sample_dir / "labels.jsonl"
        self.records: list[dict] = []
        if self.labels_path.exists():
            with open(self.labels_path) as f:
                self.records = [json.loads(line) for line in f if line.strip()]

    # ---- label state -------------------------------------------------------
    def labeled(self, pass_name: str) -> dict[str, dict[str, str]]:
        """unit_id -> {layer: label} for a pass (later records win)."""
        out: dict[str, dict[str, str]] = {}
        for r in self.records:
            if r["pass"] == pass_name:
                out.setdefault(r["unit_id"], {})[r["layer"]] = r["label"]
        return out

    def next_unit(self) -> tuple[dict | None, str, dict]:
        main = self.labeled("main")
        for u in self.units:
            got = main.get(u["unit_id"], {})
            if not all(layer in got for layer in LAYERS):
                return u, "main", got
        # main pass done -> relabel pass on a seeded 10% subsample
        rng = random.Random(RELABEL_SEED)
        n = max(1, int(len(self.units) * RELABEL_FRACTION))
        subsample = rng.sample(self.units, n)
        re = self.labeled("relabel")
        for u in subsample:
            got = re.get(u["unit_id"], {})
            if not all(layer in got for layer in LAYERS):
                return u, "relabel", got
        return None, "done", {}

    def catalog(self, layer: str) -> dict[str, list[dict]]:
        """label -> members [{unit_id, post_id, ocr_text}] from the main pass."""
        main = self.labeled("main")
        by_pid = {u["unit_id"]: u for u in self.units}
        out: dict[str, list[dict]] = {}
        for uid, layers in main.items():
            label = layers.get(layer)
            if label and label != "unique" and uid in by_pid:
                pid = by_pid[uid]["post_id"]
                out.setdefault(label, []).append(
                    {"unit_id": uid, "post_id": pid,
                     "ocr_text": self.ocr.get(pid, {}).get("text", "")}
                )
        return out

    def template_suggestion(self, unit: dict) -> list[dict]:
        """Already-labeled units in the same dhash component (template layer only)."""
        main = self.labeled("main")
        sugg = []
        for u in self.units:
            if (u["dhash_component"] == unit["dhash_component"]
                    and u["unit_id"] != unit["unit_id"]
                    and "template" in main.get(u["unit_id"], {})):
                sugg.append({"post_id": u["post_id"],
                             "template_id": main[u["unit_id"]]["template"]})
        return sugg[:8]

    def append(self, rec: dict) -> None:
        with open(self.labels_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
        self.records.append(rec)

    def progress(self) -> dict:
        main = self.labeled("main")
        done = sum(1 for u in self.units
                   if all(l in main.get(u["unit_id"], {}) for l in LAYERS))
        return {"total": len(self.units), "main_done": done}


STORE: Store | None = None
HTML = (Path(__file__).parent / "label_ui.html").read_text()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quiet
        pass

    def _json(self, obj, code=200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/next":
            unit, pass_name, got = STORE.next_unit()
            if unit is None:
                self._json({"done": True, "progress": STORE.progress()})
                return
            pid = unit["post_id"]
            payload = {
                "done": False,
                "pass": pass_name,
                "unit": unit,
                "ocr_text": STORE.ocr.get(pid, {}).get("text", ""),
                "already": got,
                "progress": STORE.progress(),
                "template_suggestion": (
                    STORE.template_suggestion(unit) if pass_name == "main" else []
                ),
            }
            self._json(payload)
        elif url.path == "/api/catalog":
            layer = parse_qs(url.query).get("layer", ["family"])[0]
            self._json(STORE.catalog(layer))
        elif url.path.startswith("/img/"):
            pid = url.path[5:]
            p = STORE.images.get(pid)
            if p is None:
                self.send_response(404)
                self.end_headers()
                return
            body = p.read_bytes()
            ctype = {"jpg": "image/jpeg", "png": "image/png", "gif": "image/gif",
                     "webp": "image/webp"}.get(p.suffix[1:], "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/label":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers["Content-Length"])
        rec = json.loads(self.rfile.read(length))
        assert rec["layer"] in LAYERS and rec["pass"] in ("main", "relabel")
        assert rec["label"], "empty label"
        from datetime import datetime, timezone

        rec["ts"] = datetime.now(timezone.utc).isoformat()
        STORE.append(rec)
        self._json({"ok": True})


def main() -> None:
    global STORE
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    STORE = Store(Path(args.sample))
    print(f"{len(STORE.units)} labeling units; labels -> {STORE.labels_path}")
    print(f"open http://localhost:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
