"""Effect of the '[empty]' OCR placeholder on the image+text arm.

Items with no extracted text all get the CLIP text embedding of the literal string
'[empty]', so their text halves are identical and their image+text cosine similarity
is at least 0.5. Variant: zero the text half for those items (image-only for them).
Reports R@P90 (registered metric) and R@P80 with the registered-style bootstrap of the
paired difference to image only.

Usage (repo root): uv run python experiments/WP-04/sample-001/review/empty_fix.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

S = Path("/home/linus/memes-revised/experiments/WP-04/sample-001")
TH = np.round(np.arange(0.01, 0.9, 0.01), 3)
lab = pd.read_parquet(S / "labels.parquet").reset_index(drop=True)
ids = json.loads((S / "ids.json").read_text()); pos = {p: i for i, p in enumerate(ids)}
rows = np.array([pos[p] for p in lab.post_id])
E = np.load(S / "embeddings_p_img_ocr.npy")[rows].astype(np.float64)
ocr = json.loads((S / "ocr.json").read_text())
empty = np.array([not ocr[p]["text"].strip() for p in lab.post_id])
norm = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
Ez = E.copy(); Ez[empty, 512:] = 0
X = {"image + text (as audited)": norm(E), "image + text, empty-OCR text half zeroed": norm(Ez),
     "image only": norm(E[:, :512])}
bg = (lab.stratum == "background").to_numpy()
rng = np.random.default_rng(11)
out = {"n_empty_ocr": int(empty.sum())}
for layer in ("family_id", "template_id"):
    _, g = np.unique(lab[layer].to_numpy(), return_inverse=True)
    gs = np.bincount(g)[g]; multi = gs >= 2
    ng = g.max() + 1
    W = rng.multinomial(ng, np.ones(ng) / ng, size=1000)[:, g].astype(float)

    def pri(pred):
        _, p = np.unique(pred, return_inverse=True)
        cj = Counter(zip(g, p)); j = np.array([cj[(a, b)] for a, b in zip(g, p)])
        return j / np.bincount(p)[p], j / gs

    def r_at(Ps, Rs, t):
        ok = np.where(Ps >= t)[0]
        if len(ok) == 0: return np.nan
        i = ok.max()
        if i == len(Ps) - 1 or Ps[i] == Ps[i + 1]: return Rs[i]
        w = (Ps[i] - t) / (Ps[i] - Ps[i + 1]); return Rs[i] + w * (Rs[i + 1] - Rs[i])

    boots = {}
    for name, x in X.items():
        Z = linkage(x, "average", metric="cosine")
        prs = [pri(fcluster(Z, t=t, criterion="distance")) for t in TH]
        for sname, mask in (("pooled", np.ones(len(g), bool)), ("background", bg)):
            Wm = W * mask
            bP = np.stack([(Wm * a).sum(1) / Wm.sum(1) for a, _ in prs], 1)
            bR = np.stack([(Wm * multi * b).sum(1) / (Wm * multi).sum(1) for _, b in prs], 1)
            P0 = np.array([a[mask].mean() for a, _ in prs]); R0 = np.array([b[mask & multi].mean() for _, b in prs])
            for t in (0.9, 0.8):
                pt = r_at(P0, R0, t); bt = np.array([r_at(bP[b], bR[b], t) for b in range(1000)])
                boots[(name, sname, t)] = (pt, bt)
                out[f"{layer} | {name} | {sname} | R@P{int(t*100)}"] = round(float(pt), 3)
    for sname in ("pooled", "background"):
        for t in (0.9, 0.8):
            a = boots[("image + text, empty-OCR text half zeroed", sname, t)]
            b = boots[("image only", sname, t)]
            d = a[1] - b[1]
            out[f"{layer} | zeroed minus image only | {sname} | R@P{int(t*100)}"] = {
                "point": round(float(a[0] - b[0]), 3), "ci95": [round(float(v), 3) for v in np.nanpercentile(d, [2.5, 97.5])]}
(S / "review/empty_fix.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, v)
