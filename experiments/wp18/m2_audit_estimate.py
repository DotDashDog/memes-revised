"""WP-18 measurement 2: overlap corrected by the by-eye match audits (NOT pre-registered; lower
standing than the primary radius-4 metric, which it does not replace).

Post-level cross-community overlap = share(nearest other-community image at distance 0) * p0
                                    + share(nearest at distance 1-4) * p14
p0  = share of distance-0 matches judged the same meme (contact sheet, m2_contact_sheet_judgement.csv)
p14 = share of distance-1-4 posts judged the same meme (m2_match_audit_judgement.csv)
Uncertainty: day-cluster bootstrap of the two shares (as in m2_overlap.py) times independent
Jeffreys Beta draws for p0 and p14; 2,000 draws, seed 18016.

    uv run python -m experiments.wp18.m2_audit_estimate
"""
import json

import numpy as np
import pandas as pd

from experiments.wp18.common import OUT, SEED, SUBS

u = pd.read_parquet(OUT / "m2_usable_posts.parquet")
mind = np.load(OUT / "m2_min_dist.npy").astype(int)
lab = u.community.map({s: i for i, s in enumerate(SUBS)}).to_numpy()
mind[np.arange(len(u)), lab] = 999
cm = mind.min(axis=1)
d0, d14 = (cm == 0), (cm >= 1) & (cm <= 4)
cs = pd.read_csv(OUT / "m2_contact_sheet_judgement.csv")
au = pd.read_csv(OUT / "m2_match_audit_judgement.csv")
k0, n0 = int((cs[cs.dist == 0].judgement == "same meme").sum()), int((cs.dist == 0).sum())
k14, n14 = int((au.judgement == "same meme").sum()), len(au)
kt14 = int(au.judgement.isin(["same meme", "same template/base image, different caption"]).sum())
rng = np.random.default_rng(SEED)
days = np.unique(u.day)
dd = u.day.to_numpy()
n_d = np.array([(dd == d).sum() for d in days], float)
a_d = np.array([d0[dd == d].sum() for d in days], float)
b_d = np.array([d14[dd == d].sum() for d in days], float)
B = 2000
idx = rng.integers(0, len(days), (B, len(days)))
s0 = a_d[idx].sum(1) / n_d[idx].sum(1)
s14 = b_d[idx].sum(1) / n_d[idx].sum(1)
p0 = rng.beta(k0 + .5, n0 - k0 + .5, B)
p14 = rng.beta(k14 + .5, n14 - k14 + .5, B)
pt = rng.beta(kt14 + .5, n14 - kt14 + .5, B)
same = s0 * p0 + s14 * p14
same_or_tmpl = s0 * p0 + s14 * pt
pct = lambda x: [round(100 * float(np.percentile(x, 2.5)), 2), round(100 * float(np.percentile(x, 97.5)), 2)]  # noqa
out = {
    "share_nearest_cross_dist0_pct": round(100 * d0.mean(), 2), "share_nearest_cross_dist1to4_pct": round(100 * d14.mean(), 2),
    "audit_dist0_same_meme": f"{k0}/{n0}", "audit_dist1to4_same_meme": f"{k14}/{n14}",
    "audit_dist1to4_same_meme_or_same_template": f"{kt14}/{n14}",
    "same_meme_overlap_pct": round(100 * float(d0.mean() * k0 / n0 + d14.mean() * k14 / n14), 2),
    "same_meme_overlap_ci95_pct": pct(same),
    "same_meme_or_template_overlap_pct": round(100 * float(d0.mean() * k0 / n0 + d14.mean() * kt14 / n14), 2),
    "same_meme_or_template_overlap_ci95_pct": pct(same_or_tmpl),
    "seed": SEED, "draws": B,
}
(OUT / "m2_audit_estimate.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
