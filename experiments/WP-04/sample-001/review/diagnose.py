"""Reviewer diagnostics: what costs precision at the P=0.90 setting, operating-point
dependence of the arm comparison (bootstrap as registered, reimplemented), and
a list of false-merge clusters to look at.

Usage (repo root): uv run python experiments/WP-04/sample-001/review/diagnose.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

S = Path("/home/linus/memes-revised/experiments/WP-04/sample-001")
OUT = S / "review"
TH = np.round(np.arange(0.01, 0.9, 0.01), 3)
lab = pd.read_parquet(S / "labels.parquet").reset_index(drop=True)
ids = json.loads((S / "ids.json").read_text()); pos = {p: i for i, p in enumerate(ids)}
rows = np.array([pos[p] for p in lab.post_id])
E_io = np.load(S / "embeddings_p_img_ocr.npy")[rows].astype(np.float64)
norm = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
X = {"image + text": norm(E_io), "image only": norm(np.load(S / "embeddings_p_vlm_only.npy")[rows].astype(np.float64)),
     "text only": norm(E_io[:, 512:])}
gold = lab.family_id.to_numpy()
_, g = np.unique(gold, return_inverse=True)
gs = np.bincount(g)[g]; multi = gs >= 2
bg = (lab.stratum == "background").to_numpy()
out = {}


def pr_items(pred):
    _, p = np.unique(pred, return_inverse=True)
    cj = Counter(zip(g, p)); cp = np.bincount(p)
    j = np.array([cj[(a, b)] for a, b in zip(g, p)])
    return j / cp[p], j / gs


def r_at(Ps, Rs, t):
    ok = np.where(Ps >= t)[0]
    if len(ok) == 0: return np.nan
    i = ok.max()
    if i == len(Ps) - 1: return Rs[i]
    if Ps[i] == Ps[i + 1]: return Rs[i]
    w = (Ps[i] - t) / (Ps[i] - Ps[i + 1]); return Rs[i] + w * (Rs[i + 1] - Rs[i])


PRI = {}
for name, x in X.items():
    Z = linkage(x, "average", metric="cosine")
    preds = [fcluster(Z, t=t, criterion="distance") for t in TH]
    PRI[name] = [pr_items(p) for p in preds]
    Ps = np.array([pi.mean() for pi, _ in PRI[name]])
    k = np.where(Ps >= 0.9)[0].max()
    pi, _ = PRI[name][k]
    loss = 1 - pi
    pred = preds[k]
    # decomposition of precision loss at the P>=0.90 setting
    out[f"{name} precision loss at t={TH[k]}"] = {
        "P": round(Ps[k], 4),
        "share of loss on gold-singleton items": round(float(loss[~multi].sum() / loss.sum()), 3),
        "n gold-singleton items placed in a cluster of 2+": int(((np.bincount(pred)[pred] >= 2) & ~multi).sum()),
        "n clusters of 2+": int((np.bincount(pred) >= 2).sum()),
        "n clusters of 2+ with no two items sharing a gold family": int(sum(
            1 for c in np.unique(pred) if (pred == c).sum() >= 2 and len(set(gold[pred == c])) == (pred == c).sum())),
    }
    if name == "image + text":
        # list pure-false-merge clusters for visual inspection
        fm = []
        for c in np.unique(pred):
            m = pred == c
            if m.sum() >= 2 and len(set(gold[m])) == m.sum():
                fm.append(lab.post_id[m].tolist())
        out["image + text false-merge clusters (all members in different gold families)"] = fm

# bootstrap, as registered (group resampling, no reclustering), at several operating points
rng = np.random.default_rng(7)
ng = g.max() + 1
W = rng.multinomial(ng, np.ones(ng) / ng, size=1000)[:, g].astype(float)
for sname, mask in (("pooled", np.ones(len(g), bool)), ("background", bg)):
    for t in (0.9, 0.8, 0.7, 0.5):
        vals = {}
        for name in X:
            Wm = W * mask
            bP = np.stack([(Wm * pi).sum(1) / Wm.sum(1) for pi, _ in PRI[name]], 1)
            bR = np.stack([(Wm * multi * ri).sum(1) / (Wm * multi).sum(1) for _, ri in PRI[name]], 1)
            P0 = np.array([pi[mask].mean() for pi, _ in PRI[name]]); R0 = np.array([ri[mask & multi].mean() for _, ri in PRI[name]])
            vals[name] = (r_at(P0, R0, t), np.array([r_at(bP[b], bR[b], t) for b in range(1000)]))
        for a, b in (("image + text", "image only"), ("text only", "image only")):
            d = vals[a][1] - vals[b][1]
            out[f"{a} minus {b} R@P{int(t*100)} [{sname}]"] = {
                "point": round(vals[a][0] - vals[b][0], 3),
                "ci95": [round(v, 3) for v in np.nanpercentile(d, [2.5, 97.5])]}
        out[f"R@P{int(t*100)} [{sname}] per arm"] = {n: (round(v[0], 3), [round(q, 3) for q in np.nanpercentile(v[1], [2.5, 97.5])]) for n, v in vals.items()}

(OUT / "diagnose.json").write_text(json.dumps(out, indent=1, default=float))
for k, v in out.items():
    if "false-merge" in k:
        print(k, len(v), v[:15])
    else:
        print(k, v)
