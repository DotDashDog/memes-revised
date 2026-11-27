"""WP-04 Method §2 (stage 3): simulation-based sample-size derivation.

The question: how many labeled posts n does the audit need so that the 95%
bootstrap CI half-width on B-cubed F1 (scored against the meaning layer) is at
most W_TARGET — for the family stratum and pooled, at plausible clusterer
quality levels? "The derivation, not a feel-picked n, is the deliverable."

Approach (all seeds fixed):
1. Take the empirical dhash family-size distribution at the chosen radius as
   the stand-in truth (scaffolding; declared in the WP).
2. For each candidate design (n_seed seeds uniform over posts, family members
   included up to CAP; n_bg uniform background), simulate the labeled set.
3. Simulate clusterers at three quality levels by perturbing the truth:
   each member splits off its family with prob p_split; each family merges
   into a random other with prob p_merge. Levels calibrated to B3 F1 about
   0.85 / 0.70 / 0.55.
4. Bootstrap by *family* (clustered resampling — items within a family are
   dependent) and record the 95% CI half-width of B3 F1.
5. Emit the width-vs-n table; the chosen design is the smallest meeting
   W_TARGET at every quality level.

W_TARGET is proposed in the output but is NOT registered by this script —
the registration happens in the WP-04 Log after user approval.

Usage:
    uv run python -m experiments.wp04.derive_n --radius 4
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

WP04 = Path("/home/linus/memes-revised/experiments/WP-04")
W_TARGET = 0.05  # proposed 95% CI half-width on B-cubed F1
CAP = 12  # max labeled members per seeded family
QUALITY = {"good": (0.05, 0.02), "mid": (0.15, 0.08), "poor": (0.30, 0.15)}
N_SIM = 40  # simulated label sets per design
N_BOOT = 400  # bootstrap resamples per simulated set


def bcubed_f1(truth: np.ndarray, pred: np.ndarray) -> float:
    n = len(truth)
    _, t = np.unique(truth, return_inverse=True)
    _, p = np.unique(pred, return_inverse=True)
    t_sizes = np.bincount(t)
    p_sizes = np.bincount(p)
    pairs, c = np.unique(t.astype(np.int64) * len(p_sizes) + p, return_counts=True)
    jt, jp = pairs // len(p_sizes), pairs % len(p_sizes)
    prec = float((c * c / p_sizes[jp]).sum()) / n
    rec = float((c * c / t_sizes[jt]).sum()) / n
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def simulate_design(sizes: np.ndarray, n_seed: int, n_bg: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (family_id, stratum) arrays for one simulated labeled set."""
    post_family = np.repeat(np.arange(len(sizes)), sizes)
    fam_ids: list[int] = []
    strata: list[int] = []
    seeds = rng.choice(len(post_family), size=n_seed, replace=False)
    for fam in np.unique(post_family[seeds]):
        take = min(int(sizes[fam]), CAP)
        fam_ids += [fam] * take
        strata += [0] * take
    bg = rng.choice(len(post_family), size=n_bg, replace=False)
    fam_ids += list(post_family[bg])
    strata += [1] * n_bg
    return np.array(fam_ids), np.array(strata)


def perturb(truth: np.ndarray, p_split: float, p_merge: float,
            rng: np.random.Generator) -> np.ndarray:
    pred = truth.copy()
    next_id = truth.max() + 1
    for i in range(len(pred)):
        if rng.random() < p_split:
            pred[i] = next_id
            next_id += 1
    fams = np.unique(pred)
    remap = {}
    for f in fams:
        if rng.random() < p_merge:
            remap[f] = int(rng.choice(fams))
    if remap:
        pred = np.array([remap.get(f, f) for f in pred])
    return pred


def ci_half_width(truth: np.ndarray, pred: np.ndarray,
                  rng: np.random.Generator) -> float:
    fams = np.unique(truth)
    idx_by_fam = {f: np.where(truth == f)[0] for f in fams}
    stats = []
    for _ in range(N_BOOT):
        pick = rng.choice(fams, size=len(fams), replace=True)
        idx = np.concatenate([idx_by_fam[f] for f in pick])
        stats.append(bcubed_f1(truth[idx], pred[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (hi - lo) / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--designs", default="150:150,250:250,400:300,600:400")
    args = ap.parse_args()

    fam = pd.read_parquet(WP04 / "families.parquet")
    comp = fam[f"fam_t{args.radius}"].to_numpy()
    sizes = np.array(sorted(Counter(comp).values(), reverse=True))
    print(f"empirical dist (T={args.radius}): {len(sizes):,} families, "
          f"{(sizes >= 2).sum():,} multi, largest {sizes.max()}")

    rng = np.random.default_rng(42)
    report = {"radius": args.radius, "w_target_proposed": W_TARGET, "cap": CAP,
              "designs": []}
    for spec in args.designs.split(","):
        n_seed, n_bg = (int(x) for x in spec.split(":"))
        row = {"n_seed": n_seed, "n_bg": n_bg}
        for qname, (ps, pm) in QUALITY.items():
            widths, f1s, tot = [], [], []
            for _ in range(N_SIM):
                truth, strata = simulate_design(sizes, n_seed, n_bg, rng)
                pred = perturb(truth, ps, pm, rng)
                fam_mask = strata == 0
                widths.append({
                    "pooled": ci_half_width(truth, pred, rng),
                    "family_stratum": ci_half_width(truth[fam_mask],
                                                    pred[fam_mask], rng),
                })
                f1s.append(bcubed_f1(truth, pred))
                tot.append(len(truth))
            row[qname] = {
                "mean_f1": round(float(np.mean(f1s)), 3),
                "half_width_pooled": round(float(np.mean(
                    [w["pooled"] for w in widths])), 4),
                "half_width_family_stratum": round(float(np.mean(
                    [w["family_stratum"] for w in widths])), 4),
            }
        row["mean_n_labeled"] = round(float(np.mean(tot)), 1)
        report["designs"].append(row)
        print(json.dumps(row))

    (WP04 / "sample_size_derivation.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {WP04 / 'sample_size_derivation.json'}")


if __name__ == "__main__":
    main()
