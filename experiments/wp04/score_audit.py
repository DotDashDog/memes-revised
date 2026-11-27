"""Score clusterings of the gold sample against the agent/user reference labels.

Implements the scoring plan registered in the WP-04 log on 2026-11-27.

For each arm (embedding composition) and each reference point (hash groups,
all-singletons), sweeps granularity and computes B-cubed precision over all
items and B-cubed recall over items whose gold group has >= 2 members, against
both label layers, pooled and per stratum. The primary number is recall at the
sweep setting where precision crosses 0.90 (linear interpolation), with a 95%
interval from resampling gold groups (singletons included) with replacement.

Usage:
    uv run python -m experiments.wp04.score_audit --sample experiments/WP-04/sample-001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

WP04 = Path("/home/linus/memes-revised/experiments/WP-04")
ARMS = {"image + text": "embeddings_p_img_ocr.npy", "image only": "embeddings_p_vlm_only.npy"}
THRESHOLDS = np.round(np.concatenate([np.arange(0.02, 0.30, 0.02), np.arange(0.30, 0.82, 0.04)]), 3)
RADII = [0, 2, 4, 6, 8, 10]
P_TARGET = 0.90
N_BOOT = 1000
SEED = 42


def per_item_pr(gold: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """B-cubed per-item precision and recall."""
    _, g = np.unique(gold, return_inverse=True)
    _, p = np.unique(pred, return_inverse=True)
    key = g.astype(np.int64) * (p.max() + 1) + p
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    joint = cnt[inv]
    return joint / np.bincount(p)[p], joint / np.bincount(g)[g]


def summarise(prec_i, rec_i, multi, w=None):
    """Mean precision over all items; mean recall over items in multi-member gold groups."""
    if w is None:
        w = np.ones(len(prec_i))
    P = float((prec_i * w).sum() / w.sum())
    wm = w * multi
    R = float((rec_i * wm).sum() / wm.sum()) if wm.sum() > 0 else float("nan")
    return P, R


def recall_at_precision(Ps: np.ndarray, Rs: np.ndarray, target: float = P_TARGET) -> float:
    """Recall where the fine-to-coarse sweep's precision falls through `target`.

    Settings are ordered fine -> coarse; precision is (near) non-increasing.
    Returns recall at the coarsest setting still >= target, interpolated towards
    the next one. NaN if precision never reaches the target.
    """
    ok = np.where(Ps >= target)[0]
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


def sweeps(sample: Path, ids: list[str]) -> dict[str, list[np.ndarray]]:
    """name -> list of predicted-label arrays ordered fine -> coarse."""
    out: dict[str, list[np.ndarray]] = {}
    emb_ids = json.loads((sample / "ids.json").read_text())
    pos = {pid: i for i, pid in enumerate(emb_ids)}
    rows = [pos[i] for i in ids]
    for name, fn in ARMS.items():
        X = np.load(sample / fn)[rows].astype(np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        Z = linkage(X, method="average", metric="cosine")
        out[name] = [fcluster(Z, t=t, criterion="distance") for t in THRESHOLDS]
    fam = pd.read_parquet(WP04 / "families.parquet").drop_duplicates("post_id").set_index("post_id")
    out["perceptual hash"] = [fam.loc[ids, f"fam_t{r}"].to_numpy() for r in RADII]
    out["every item alone"] = [np.arange(len(ids))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--labels", default="labels.parquet")
    ap.add_argument("--drop-uncertain", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sample = Path(args.sample)
    lab = pd.read_parquet(sample / args.labels)
    if args.drop_uncertain:
        lab = lab[~lab.uncertain]
    ids = lab.post_id.tolist()
    preds = sweeps(sample, ids)
    rng = np.random.default_rng(SEED)
    strata = {"pooled": np.ones(len(lab), bool),
              "background": (lab.stratum == "background").to_numpy(),
              "family-seeded": (lab.stratum == "family").to_numpy()}
    result: dict = {"n_items": len(lab), "p_target": P_TARGET, "layers": {}}
    for layer in ("family_id", "template_id"):
        gold = lab[layer].to_numpy()
        _, g = np.unique(gold, return_inverse=True)
        gsize = np.bincount(g)[g]
        multi = (gsize >= 2).astype(float)
        layer_res = {"n_gold_groups": int(g.max() + 1),
                     "n_multi_groups": int((np.bincount(g) >= 2).sum()),
                     "n_items_in_multi": int(multi.sum()), "methods": {}}
        # bootstrap weights: resample gold groups; item weight = times its group was drawn
        ngroups = g.max() + 1
        draws = rng.multinomial(ngroups, np.ones(ngroups) / ngroups, size=N_BOOT)
        W = draws[:, g].astype(float)  # (N_BOOT, n_items)
        per_method_boot: dict[str, dict[str, np.ndarray]] = {}
        for name, plist in preds.items():
            pr = [per_item_pr(gold, p) for p in plist]
            mres = {}
            per_method_boot[name] = {}
            for sname, mask in strata.items():
                m = mask.astype(float)
                curve = [summarise(pi, ri, multi, m) for pi, ri in pr]
                Ps = np.array([c[0] for c in curve]); Rs = np.array([c[1] for c in curve])
                Wm = W * m
                bP = np.stack([(Wm * pi).sum(1) / Wm.sum(1) for pi, _ in pr], 1)
                den = (Wm * multi).sum(1)
                bR = np.stack([(Wm * multi * ri).sum(1) / np.where(den > 0, den, np.nan)
                               for _, ri in pr], 1)
                boot = np.array([recall_at_precision(bP[b], bR[b]) for b in range(N_BOOT)])
                per_method_boot[name][sname] = boot
                lo, hi = np.nanpercentile(boot, [2.5, 97.5]) if np.isfinite(boot).any() else (np.nan, np.nan)
                mres[sname] = {
                    "precision_curve": Ps.round(4).tolist(), "recall_curve": Rs.round(4).tolist(),
                    "recall_at_p90": recall_at_precision(Ps, Rs),
                    "recall_at_p90_ci95": [float(lo), float(hi)],
                    "boot_nan_frac": float(np.isnan(boot).mean()),
                    "n_items": int(mask.sum()),
                    "n_items_in_multi": int((multi * m).sum()),
                }
            layer_res["methods"][name] = mres
        # paired differences against the hash baseline and between arms
        diffs = {}
        for a, b in [("image + text", "perceptual hash"), ("image only", "perceptual hash"),
                     ("image + text", "image only")]:
            for sname in strata:
                d = per_method_boot[a][sname] - per_method_boot[b][sname]
                if np.isfinite(d).any():
                    lo, hi = np.nanpercentile(d, [2.5, 97.5])
                    diffs[f"{a} minus {b} [{sname}]"] = {
                        "point": layer_res["methods"][a][sname]["recall_at_p90"]
                        - layer_res["methods"][b][sname]["recall_at_p90"],
                        "ci95": [float(lo), float(hi)]}
        layer_res["differences_recall_at_p90"] = diffs
        result["layers"][layer] = layer_res
    result["sweep"] = {"cosine_thresholds": THRESHOLDS.tolist(), "hash_radii": RADII}
    out = sample / f"scores{args.tag}.json"
    out.write_text(json.dumps(result, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
