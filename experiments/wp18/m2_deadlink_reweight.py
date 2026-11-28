"""WP-18 m2, NOT pre-registered (added after seeing that dead links are enriched among
moderator-removed posts): does the duplicate rate differ by removal status among fetched posts,
and what does the pooled overlap become if fetched posts are reweighted to the removal-status mix
of ALL image posts (dead ones included)? Assumes that within a (community, removal status) cell,
dead and live posts are duplicates equally often -- an untestable assumption, stated.

    uv run python -m experiments.wp18.m2_deadlink_reweight
"""
import json

import numpy as np
import pandas as pd

from experiments.wp18.common import OUT, SEED, SUBS

u = pd.read_parquet(OUT / "m2_usable_posts.parquet")
mind = np.load(OUT / "m2_min_dist.npy")
lab = u.community.map({s: i for i, s in enumerate(SUBS)}).to_numpy()
o = mind.copy()
o[np.arange(len(u)), lab] = 65
u["dup"] = o.min(axis=1) <= 4
w = pd.read_parquet(OUT / "m2_window_posts.parquet")
w = w[w.url_class == "image"]
cat = lambda s: s.fillna("none").replace({"automod_filtered": "moderator"})  # noqa: E731
w["rbc"] = cat(w.removed_by_category)
u = u.merge(w[["post_id", "rbc"]], on="post_id")
rate = u.groupby(["community", "rbc"]).dup.agg(["mean", "size"])
pop = w.groupby(["community", "rbc"]).size().rename("all_posts")
t = rate.join(pop, how="outer")
print(t.round(3).to_string())


def reweighted(uu):
    r = uu.groupby(["community", "rbc"]).dup.mean()
    p = pop.reindex(r.index)
    return float((r * p).sum() / p.sum())


est = reweighted(u)
rng = np.random.default_rng(SEED)
days = u.day.unique()
bs = []
for _ in range(1000):
    pick = rng.choice(days, len(days))
    bs.append(reweighted(pd.concat([u[u.day == d] for d in pick])))
res = {"assumption": "within community x removal status, dead posts duplicate as often as live ones",
       "dup_rate_by_removal_status_pooled": u.groupby("rbc").dup.mean().to_dict(),
       "n_by_removal_status_fetched": u.rbc.value_counts().to_dict(),
       "pooled_unweighted": float(u.dup.mean()),
       "pooled_reweighted_to_all_image_posts": est,
       "pooled_reweighted_ci95_day_bootstrap": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
       "note": "'deleted' (author deleted) posts are ~100% dead and have almost no fetched examples; they "
               "are excluded from the reweighting cells that have no fetched posts"}
(OUT / "m2_deadlink_reweight.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
