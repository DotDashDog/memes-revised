"""Independent reviewer recomputation of the WP-04 clustering audit.

Does not import experiments.wp04.*. Writes review/recompute.json and prints a summary.
Usage (repo root): uv run python experiments/WP-04/sample-001/review/recompute.py
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

S = Path("/home/linus/memes-revised/experiments/WP-04/sample-001")
OUT = S / "review"
TH = np.round(np.concatenate([np.arange(0.02, 0.30, 0.02), np.arange(0.30, 0.82, 0.04)]), 3)
FINE_TH = np.round(np.arange(0.005, 0.9, 0.005), 3)
res: dict = {}

# ---------------- load and align ----------------
lab = pd.read_parquet(S / "labels.parquet").reset_index(drop=True)
labB = pd.read_parquet(S / "labels_B.parquet").set_index("post_id").loc[lab.post_id].reset_index()
ids = json.loads((S / "ids.json").read_text())
pos = {p: i for i, p in enumerate(ids)}
rows = np.array([pos[p] for p in lab.post_id])
E_io = np.load(S / "embeddings_p_img_ocr.npy")[rows].astype(np.float64)
E_v = np.load(S / "embeddings_p_vlm_only.npy")[rows].astype(np.float64)
d_img = E_v.shape[1]
align = {
    "n_ids": len(ids), "n_labels": len(lab), "same_set": set(ids) == set(lab.post_id),
    "emb_shapes": [list(E_io.shape), list(E_v.shape)],
    # image half of image+text should equal image-only embedding row for row
    "max_abs_diff_image_half_vs_image_only": float(np.abs(E_io[:, :d_img] - E_v).max()),
}
# check labels.parquet matches stage2/labels_A.json
A_json = pd.DataFrame(json.loads((S / "stage2/labels_A.json").read_text())).set_index("id").loc[lab.post_id]
align["labels_parquet_equals_labels_A_json_family"] = bool((A_json.family_id.values == lab.family_id.values).all())
align["labels_parquet_equals_labels_A_json_template"] = bool((A_json.template_id.values == lab.template_id.values).all())
# image-file check: post id of label row k really is the image embedded at row k; spot-check
# by re-hashing is not possible without re-embedding; instead verify OCR text of empty-ocr
# items is identical in the text half (all "[empty]").
ocr = json.loads((S / "ocr.json").read_text())
empty = np.array([not ocr[p]["text"].strip() for p in lab.post_id])
T = E_io[:, d_img:]
if empty.sum() > 1:
    te = T[empty]
    align["n_empty_ocr"] = int(empty.sum())
    align["text_half_identical_for_empty_ocr"] = float(np.abs(te - te[0]).max())
res["alignment"] = align

norm = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
X = {"image + text": norm(E_io.copy()), "image only": norm(E_v.copy()), "text only": norm(T.copy())}

fam = pd.read_parquet(S.parent / "families.parquet").drop_duplicates("post_id").set_index("post_id")
HASH = {r: fam.loc[lab.post_id, f"fam_t{r}"].to_numpy() for r in [0, 2, 4, 6, 8, 10]}
strata = {"pooled": np.ones(len(lab), bool), "background": (lab.stratum == "background").to_numpy(),
          "family-seeded": (lab.stratum == "family").to_numpy()}


# ---------------- own B-cubed ----------------
def bcubed(gold, pred):
    """Per-item B-cubed precision and recall, plain dictionary counting."""
    gold = list(gold); pred = list(pred)
    cg, cp = Counter(gold), Counter(pred)
    cj = Counter(zip(gold, pred))
    p = np.array([cj[(g, q)] / cp[q] for g, q in zip(gold, pred)])
    r = np.array([cj[(g, q)] / cg[g] for g, q in zip(gold, pred)])
    pairr = np.array([(cj[(g, q)] - 1) / (cg[g] - 1) if cg[g] > 1 else np.nan for g, q in zip(gold, pred)])
    return p, r, pairr


def pair_counts(gold, pred):
    """Pair-counting precision/recall (each unordered pair once)."""
    cg, cp, cj = Counter(gold), Counter(pred), Counter(zip(gold, pred))
    c2 = lambda n: n * (n - 1) / 2
    tp = sum(c2(v) for v in cj.values())
    sp = sum(c2(v) for v in cp.values()); sg = sum(c2(v) for v in cg.values())
    return (tp / sp if sp else 1.0), (tp / sg if sg else np.nan)


def r_at_p(Ps, Rs, target=0.90):
    ok = np.where(np.asarray(Ps) >= target)[0]
    if len(ok) == 0:
        return float("nan")
    i = ok.max()
    if i == len(Ps) - 1:
        return float(Rs[i])
    p0, p1, r0, r1 = Ps[i], Ps[i + 1], Rs[i], Rs[i + 1]
    if p0 == p1:
        return float(r0)
    t = (p0 - target) / (p0 - p1)
    return float(r0 + t * (r1 - r0))


def curve(gold, preds, mask):
    gs = pd.Series(gold).map(pd.Series(gold).value_counts()).to_numpy()
    multi = gs >= 2
    P, R, PR, PP, PPR = [], [], [], [], []
    for pr in preds:
        p, r, pairr = bcubed(gold, pr)
        P.append(p[mask].mean()); R.append(r[mask & multi].mean()); PR.append(np.nanmean(pairr[mask & multi]))
        pp, ppr = pair_counts([g for g, m in zip(gold, mask) if m] if False else list(gold), list(pr))
        PP.append(pp); PPR.append(ppr)
    return np.array(P), np.array(R), np.array(PR), np.array(PP), np.array(PPR)


def agglo(x, method="average", ths=TH):
    Z = linkage(x, method=method, metric="cosine") if method != "ward" else linkage(x, "ward")
    return [fcluster(Z, t=t, criterion="distance") for t in ths], Z


preds = {}
for name in ("image + text", "image only"):
    preds[name] = agglo(X[name])[0]
preds["perceptual hash"] = [HASH[r] for r in [0, 2, 4, 6, 8, 10]]
preds["every item alone"] = [np.arange(len(lab))]

sc = json.loads((S / "scores.json").read_text())
recomp = {}
for layer in ("family_id", "template_id"):
    gold = lab[layer].to_numpy()
    recomp[layer] = {}
    for m, pl in preds.items():
        for sname, mask in strata.items():
            P, R, PR, _, _ = curve(gold, pl, mask)
            mine = r_at_p(P, R)
            theirs = sc["layers"][layer]["methods"][m][sname]["recall_at_p90"]
            pair_mine = r_at_p(P, PR)
            # floor-normalised: share of the gap between all-alone and perfect recall
            recomp[layer][f"{m} [{sname}]"] = {
                "R@P90_mine": round(mine, 4), "R@P90_theirs": round(theirs, 4),
                "pairwise_recall@P90 (self excluded)": round(pair_mine, 4),
                "max_curve_abs_diff_P": float(np.abs(P - np.array(sc["layers"][layer]["methods"][m][sname]["precision_curve"])).max()),
            }
res["recompute"] = recomp

# floor analysis
floor = {}
for layer in ("family_id", "template_id"):
    gold = lab[layer].to_numpy()
    gs = pd.Series(gold).map(pd.Series(gold).value_counts()).to_numpy()
    for sname, mask in strata.items():
        m = mask & (gs >= 2)
        floor[f"{layer} [{sname}]"] = {
            "mean 1/|g| over multi items (= all-alone recall)": round(float((1 / gs[m]).mean()), 4),
            "share of items in multi groups": round(float((gs >= 2)[mask].mean()), 3),
            "multi group size distribution": dict(Counter(gs[m].tolist())),
        }
res["floor"] = floor

# ---------------- alternative metrics / operating points on the registered methods ----------------
alt = {}
for layer in ("family_id", "template_id"):
    gold = lab[layer].to_numpy()
    for m in ("image + text", "image only", "text only"):
        pl = preds[m] if m in preds else agglo(X[m])[0]
        preds.setdefault(m, pl)
        for sname in ("pooled", "background"):
            P, R, PR, PP, PPR = curve(gold, pl, strata[sname])
            F = 2 * P * R / (P + R)
            # B-cubed with recall over all items (standard): need recompute
            Rall = []
            for pr in pl:
                p, r, _ = bcubed(gold, pr); Rall.append(r[strata[sname]].mean())
            Rall = np.array(Rall); Fall = 2 * P * Rall / (P + Rall)
            alt[f"{layer} | {m} [{sname}]"] = {
                "R@P90": round(r_at_p(P, R), 3), "R@P80": round(r_at_p(P, R, 0.8), 3),
                "R@P70": round(r_at_p(P, R, 0.7), 3), "R@P50": round(r_at_p(P, R, 0.5), 3),
                "pairR@P90": round(r_at_p(P, PR), 3), "pairR@P80": round(r_at_p(P, PR, 0.8), 3),
                "max_F1(P all, R multi)": round(float(F.max()), 3),
                "max_F1 standard bcubed (all items)": round(float(Fall.max()), 3),
                "pair-counting max F1": round(float(np.nanmax(2 * PP * PPR / (PP + PPR))), 3) if sname == "pooled" else None,
            }
res["alternative_metrics"] = alt

# ---------------- alternative clustering methods ----------------
methods = {}
gold_f = lab.family_id.to_numpy(); gold_t = lab.template_id.to_numpy()


def report(name, pl):
    out = {}
    for layer, gold in (("family", gold_f), ("template", gold_t)):
        for sname in ("pooled", "background"):
            P, R, PR, _, _ = curve(gold, pl, strata[sname])
            out[f"{layer} [{sname}] R@P90"] = round(r_at_p(P, R), 3)
            out[f"{layer} [{sname}] R@P80"] = round(r_at_p(P, R, 0.8), 3)
    methods[name] = out


for m in ("image + text", "image only", "text only"):
    for link in ("average", "complete", "single"):
        report(f"{m} | {link} linkage (fine grid)", agglo(X[m], link, FINE_TH)[0])
# reweighted concatenation: alpha weight on text half
for a in (0.25, 0.5, 0.75, 1.0, 1.5):
    x = norm(np.concatenate([X["image only"], a * X["text only"]], 1))
    report(f"concat image + {a}*text | average (fine grid)", agglo(x, "average", FINE_TH)[0])


# kNN-graph clustering: connect i-j if mutual top-k and cos >= s; connected components
def knn_cc(x, k, s):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    Sm = x @ x.T
    np.fill_diagonal(Sm, -np.inf)
    top = np.argsort(-Sm, 1)[:, :k]
    n = len(x)
    A = np.zeros((n, n), bool)
    for i in range(n):
        A[i, top[i]] = True
    A = A & A.T & (Sm >= s)
    return connected_components(csr_matrix(A), directed=False)[1]


for m in ("image + text", "image only"):
    for k in (1, 3):
        pl = [knn_cc(X[m], k, s) for s in np.arange(0.99, 0.3, -0.01)]
        report(f"{m} | mutual {k}-NN graph components, sim threshold swept", pl)
res["alternative_methods"] = methods

# ---------------- nearest-neighbour check, with exclusions ----------------
nn = {}
h4 = HASH[4]; h0 = HASH[0]
for layer, gold in (("family_id", gold_f), ("template_id", gold_t)):
    gs = pd.Series(gold).map(pd.Series(gold).value_counts()).to_numpy()
    multi = gs >= 2
    # items whose gold group contains a member that is NOT in their hash-r4 group and NOT same template
    for m in ("image + text", "image only", "text only"):
        Sm = X[m] @ X[m].T
        np.fill_diagonal(Sm, -np.inf)
        nn1 = Sm.argmax(1)
        hit = gold[nn1] == gold
        # exclusion variants: remove from candidate set items that share hash r4 group (near-dup)
        Sx = Sm.copy(); Sx[h4[:, None] == h4[None, :]] = -np.inf
        nn1x = Sx.argmax(1)
        eligible = np.array([((gold == gold[i]) & (h4 != h4[i])).any() for i in range(len(gold))])
        hitx = gold[nn1x] == gold
        St = Sm.copy(); St[gold_t[:, None] == gold_t[None, :]] = -np.inf
        nn1t = St.argmax(1)
        elig_t = np.array([((gold == gold[i]) & (gold_t != gold_t[i])).any() for i in range(len(gold))])
        hitt = gold[nn1t] == gold
        nn[f"{layer} | {m}"] = {
            "nn1 hit (as reported)": round(float(hit[multi].mean()), 3),
            "nn1 hit among background items": round(float(hit[multi & strata['background']].mean()), 3),
            "share of nn1 hits that are same hash-r4 group": round(float((h4[nn1] == h4)[multi & hit].mean()), 3),
            "share of nn1 hits that are exact hash (r0) dup": round(float((h0[nn1] == h0)[multi & hit].mean()), 3),
            "nn1 hit excluding same-hash-r4 candidates (items with an eligible member)": round(float(hitx[eligible].mean()), 3),
            "n eligible (hash)": int(eligible.sum()),
            "nn1 hit excluding same-template candidates (items with a cross-template member)": round(float(hitt[elig_t].mean()), 3) if elig_t.any() else None,
            "n eligible (cross-template)": int(elig_t.sum()),
            "chance for cross-template eligible": round(float(np.mean([((gold == gold[i]) & (gold_t != gold_t[i])).sum() / (gold_t != gold_t[i]).sum() for i in np.where(elig_t)[0]])), 4) if elig_t.any() else None,
        }
res["neighbours"] = nn

# ---------------- family vs template structure ----------------
def same_pairs(lbl):
    g = {}
    for i, l in enumerate(lbl):
        g.setdefault(l, []).append(i)
    return {p for v in g.values() if len(v) > 1 for p in combinations(v, 2)}


fp, tp = same_pairs(gold_f), same_pairs(gold_t)
hp = same_pairs(h4)
fams = pd.Series(gold_f).value_counts(); fams = fams[fams >= 2]
one_template = sum(len(set(gold_t[gold_f == f])) == 1 for f in fams.index)
res["family_vs_template"] = {
    "family same-pairs": len(fp), "of which also same template": len(fp & tp),
    "of which also same hash r4 group": len(fp & hp),
    "multi families": len(fams), "multi families entirely within one template": int(one_template),
    "items in multi families": int(fams.sum()),
    "items in multi families whose family has a cross-template partner": int(sum(
        ((gold_f == gold_f[i]) & (gold_t != gold_t[i])).any() for i in range(len(gold_f)) if fams.get(gold_f[i], 0) >= 2)),
    "template same-pairs": len(tp), "template pairs that are same family": len(tp & fp),
    "largest families": fams.head(12).to_dict(),
}

# ---------------- background stratum: cross-stratum families ----------------
bg = strata["background"]
x = {}
for layer, gold in (("family_id", gold_f), ("template_id", gold_t)):
    gs = pd.Series(gold).map(pd.Series(gold).value_counts()).to_numpy()
    bgm = bg & (gs >= 2)
    has_fs = np.array([((gold == gold[i]) & ~bg).any() for i in range(len(gold))])
    # background-only gold sizes
    gb = pd.Series(gold[bg]).value_counts()
    x[layer] = {"bg items in multi groups": int(bgm.sum()),
                "of which group has a family-stratum member": int((bgm & has_fs).sum()),
                "bg items in multi groups counting bg members only": int(gb[gb >= 2].sum()),
                "multi groups within bg only": int((gb >= 2).sum())}
res["background_cross_stratum"] = x

# ---------------- labeller-A vs B description overlap ----------------
dA = {d["id"]: d for d in json.loads((S / "stage2/descriptions_A.json").read_text())}
dB = {d["id"]: d for d in json.loads((S / "stage2/descriptions_B.json").read_text())}
same = [k for k in dA if dA[k] == dB.get(k)]
fA, fB = lab.family_id.to_numpy(), labB.family_id.to_numpy()
diff_desc = np.array([p not in set(same) for p in lab.post_id])
agree_item = np.array([set(np.where(fA == fA[i])[0]) == set(np.where(fB == fB[i])[0]) for i in range(len(fA))])
res["labellers"] = {"descriptions identical A vs B": len(same), "of": len(dA),
                    "item family-neighbourhood identical A vs B, shared descriptions": round(float(agree_item[~diff_desc].mean()), 3),
                    "same, differing descriptions": round(float(agree_item[diff_desc].mean()), 3),
                    "n differing descriptions": int(diff_desc.sum())}

(OUT / "recompute.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
