"""Background stratum scored on its own: cluster only the 500 background images and
score against gold groups restricted to background images (reviewer check).
Usage (repo root): uv run python experiments/WP-04/sample-001/review/bg_only.py"""
import json
from collections import Counter
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
S = "/home/linus/memes-revised/experiments/WP-04/sample-001/"
lab = pd.read_parquet(S + "labels.parquet"); lab = lab[lab.stratum == "background"].reset_index(drop=True)
ids = json.load(open(S + "ids.json")); pos = {p: i for i, p in enumerate(ids)}; r = [pos[p] for p in lab.post_id]
E = np.load(S + "embeddings_p_img_ocr.npy")[r].astype(float); n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
ocr = json.load(open(S + "ocr.json")); empty = np.array([not ocr[p]["text"].strip() for p in lab.post_id])
Ez = E.copy(); Ez[empty, 512:] = 0
fam = pd.read_parquet(S + "../families.parquet").drop_duplicates("post_id").set_index("post_id")
TH = np.round(np.arange(0.01, 0.9, 0.01), 3)
out = {}
for layer in ("family_id", "template_id"):
    g = lab[layer].to_numpy(dtype=object); cg = Counter(g); gs = np.array([cg[x] for x in g]); m = gs >= 2
    def pr(pred):
        cp = Counter(pred); cj = Counter(zip(g, pred))
        j = np.array([cj[(a, b)] for a, b in zip(g, pred)])
        return j / np.array([cp[b] for b in pred]), j / gs
    def r_at(P, R, t=0.9):
        ok = np.where(P >= t)[0]; i = ok.max()
        if i == len(P) - 1 or P[i] == P[i + 1]: return R[i]
        w = (P[i] - t) / (P[i] - P[i + 1]); return R[i] + w * (R[i + 1] - R[i])
    res = {"items in multi": int(m.sum()), "multi groups": int(sum(v >= 2 for v in cg.values())),
           "every item alone": round(float((1 / gs[m]).mean()), 3)}
    for name, x in (("image + text", n(E)), ("image + text, empty zeroed", n(Ez)), ("image only", n(E[:, :512])), ("text only", n(E[:, 512:]))):
        Z = linkage(x, "average", metric="cosine")
        c = [pr(fcluster(Z, t=t, criterion="distance")) for t in TH]
        P = np.array([a.mean() for a, _ in c]); R = np.array([b[m].mean() for _, b in c])
        res[name] = {"R@P90": round(float(r_at(P, R)), 3), "R@P80": round(float(r_at(P, R, 0.8)), 3)}
    c = [pr(fam.loc[lab.post_id, f"fam_t{k}"].to_numpy()) for k in (0, 2, 4, 6, 8, 10)]
    P = np.array([a.mean() for a, _ in c]); R = np.array([b[m].mean() for _, b in c])
    res["perceptual hash"] = {"R@P90": round(float(r_at(P, R)), 3), "P_curve": P.round(3).tolist(), "R_curve": R.round(3).tolist()}
    out[layer] = res
json.dump(out, open(S + "review/bg_only.json", "w"), indent=1)
print(json.dumps(out, indent=1))
