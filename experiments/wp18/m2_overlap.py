"""WP-18 measurement 2, step 4: cross-community near-duplicate overlap (ADR 0016 precondition 2).

Design and primary metric are fixed in experiments/WP-18/expectations.md:
fraction of a community's successfully fetched image posts (28-day window) that have a dHash
near-duplicate (Hamming <= 4) among another candidate community's posts in the same window.

    uv run python -m experiments.wp18.m2_overlap
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.wp18.common import BACKBONE, OUT, REPO, SEED, SUBS

RADII = [0, 2, 4, 6, 8, 10]
PRIMARY = 4
N_BOOT = 2000
CAP_BYTES = 10 * 1024 * 1024


def min_dist_to_sets(h: np.ndarray, cand: np.ndarray, cand_label: np.ndarray, n_labels: int,
                     self_index: np.ndarray | None, chunk: int = 1024) -> np.ndarray:
    """mind[i, b] = min Hamming distance from h[i] to any candidate with label b
    (excluding candidate self_index[i], i.e. the post itself). 65 if label b is empty."""
    out = np.full((len(h), n_labels), 65, dtype=np.int16)
    order = np.argsort(cand_label, kind="stable")
    cand, lab = cand[order], cand_label[order]
    bounds = np.searchsorted(lab, np.arange(n_labels + 1))
    inv = np.empty(len(order), dtype=np.int64)
    inv[order] = np.arange(len(order))
    for s in range(0, len(h), chunk):
        d = np.bitwise_count(h[s:s + chunk, None] ^ cand[None, :]).astype(np.int16)
        if self_index is not None:
            rows = np.arange(min(chunk, len(h) - s))
            d[rows, inv[self_index[s:s + chunk]]] = 65
        for b in range(n_labels):
            if bounds[b + 1] > bounds[b]:
                out[s:s + chunk, b] = d[:, bounds[b]:bounds[b + 1]].min(axis=1)
    return out


def boot_ci(flag: np.ndarray, day: np.ndarray, rng: np.random.Generator) -> list[float]:
    """Cluster bootstrap over days: resample whole days with replacement."""
    days = np.unique(day)
    num = np.array([flag[day == d].sum() for d in days], dtype=float)
    den = np.array([(day == d).sum() for d in days], dtype=float)
    idx = rng.integers(0, len(days), size=(N_BOOT, len(days)))
    stat = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return [float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))]


def main() -> None:
    posts = pd.read_parquet(OUT / "m2_window_posts.parquet")
    idx = pd.read_parquet(OUT / "m2_shard_index.parquet")
    dh = pd.read_parquet(OUT / "m2_dhash.parquet")
    img = posts[posts.url_class == "image"].merge(
        idx[["post_id", "fetch_status", "http_status", "sha256", "length"]], on="post_id", how="left")
    assert img.fetch_status.notna().all(), "some pool posts have no fetch outcome"
    # the 10 MiB cap is applied uniformly: pilot images fetched under the old 30 MB cap that
    # exceed it are treated as too_large (expectations.md, deviation 2026-11-27 12:30)
    over = (img.fetch_status == "ok") & (img.length > CAP_BYTES)
    img.loc[over, "fetch_status"] = "too_large"
    res_over = int(over.sum())
    by_sha = dh.drop_duplicates("sha256").set_index("sha256")[["dhash", "decode_ok", "n_frames"]]
    img = img.join(by_sha, on="sha256")
    img["usable"] = (img.fetch_status == "ok") & (img.decode_ok == True)  # noqa: E712

    res: dict = {"pilot_images_reclassified_too_large": res_over, "size_cap_bytes": CAP_BYTES, "window": "2025-11-03..2025-11-30 UTC", "primary_radius": PRIMARY,
                 "seed": SEED, "n_boot": N_BOOT}

    # ---- fetch outcome accounting (dead links are counted, not ignored)
    acct = img.groupby(["community", "fetch_status"]).size().unstack(fill_value=0).reindex(SUBS)
    acct["image_posts"] = acct.sum(axis=1)
    acct["usable"] = img.groupby("community").usable.sum().reindex(SUBS)
    acct["usable_pct"] = 100 * acct.usable / acct.image_posts
    print(acct.to_string())
    res["fetch_accounting"] = acct.to_dict(orient="index")
    res["removed_by_category_vs_status"] = (
        img.assign(rbc=img.removed_by_category.fillna("<none>"))
        .groupby(["rbc", "fetch_status"]).size().unstack(fill_value=0).to_dict(orient="index"))

    u = img[img.usable].reset_index(drop=True)
    h = u.dhash.to_numpy(dtype=np.uint64)
    lab = u.community.map({s: i for i, s in enumerate(SUBS)}).to_numpy()
    day = u.day.to_numpy()
    n = len(u)
    print("usable posts:", n)

    mind = min_dist_to_sets(h, h, lab, len(SUBS), np.arange(n))
    np.save(OUT / "m2_min_dist.npy", mind)
    u[["post_id", "community", "day", "created_utc", "author", "sha256", "dhash"]].to_parquet(
        OUT / "m2_usable_posts.parquet", index=False)

    other = mind.copy()
    other[np.arange(n), lab] = 65
    d_other = other.min(axis=1)            # nearest image in any OTHER community
    d_same = mind[np.arange(n), lab]       # nearest other post in the SAME community
    is_bb = np.isin(lab, [SUBS.index(s) for s in BACKBONE])
    bb_cols = [SUBS.index(s) for s in BACKBONE]
    d_other_bb = other[:, bb_cols].min(axis=1)   # nearest in another BACKBONE community

    # ---- background control: 2019 r/memes pool (WP-04), six years earlier
    bg = pd.read_parquet(REPO / "experiments/WP-04/dhash.parquet")
    bgh = bg[bg.decode_ok].dhash.to_numpy(dtype=np.uint64)
    d_bg = min_dist_to_sets(h, bgh, np.zeros(len(bgh), dtype=int), 1, None)[:, 0]
    res["background_pool_size"] = int(len(bgh))

    rng = np.random.default_rng(SEED)
    by_radius = {}
    for T in RADII:
        row = {"pooled": {
            "n": n,
            "cross_any_other": float((d_other <= T).mean()),
            "cross_any_other_ci95": boot_ci(d_other <= T, day, rng),
            "same_community_repost": float((d_same <= T).mean()),
            "same_community_repost_ci95": boot_ci(d_same <= T, day, rng),
            "background_2019_pool": float((d_bg <= T).mean()),
            "background_2019_pool_ci95": boot_ci(d_bg <= T, day, rng),
        }, "per_community": {}, "matrix_row_has_dup_in_col": {}}
        for a, sa in enumerate(SUBS):
            m = lab == a
            n_other = int((~m).sum())
            row["per_community"][sa] = {
                "n": int(m.sum()),
                "n_candidates_in_other_communities": n_other,
                "cross_any_other": float((d_other[m] <= T).mean()),
                "cross_any_other_ci95": boot_ci((d_other <= T)[m], day[m], rng),
                "cross_count": int((d_other[m] <= T).sum()),
                "same_community_repost": float((d_same[m] <= T).mean()),
                "background_2019_pool": float((d_bg[m] <= T).mean()),
            }
            row["matrix_row_has_dup_in_col"][sa] = {
                sb: float((mind[m, b] <= T).mean()) for b, sb in enumerate(SUBS)}
        m = is_bb
        row["backbone_only"] = {
            "n": int(m.sum()),
            "cross_other_backbone": float((d_other_bb[m] <= T).mean()),
            "cross_other_backbone_ci95": boot_ci((d_other_bb <= T)[m], day[m], rng),
            "cross_count": int((d_other_bb[m] <= T).sum()),
        }
        by_radius[str(T)] = row
    res["by_radius"] = by_radius

    # ---- bounds on the all-image-posts rate under extreme assumptions about dead links
    T = PRIMARY
    bounds = {}
    for a, sa in enumerate(SUBS + ["pooled"]):
        sel = img if sa == "pooled" else img[img.community == sa]
        k = int((d_other[lab == a] <= T).sum()) if sa != "pooled" else int((d_other <= T).sum())
        n_all, n_dead = len(sel), int((~sel.usable).sum())
        bounds[sa] = {"image_posts": n_all, "not_usable": n_dead,
                      "lower_if_no_dead_is_dup": k / n_all,
                      "upper_if_all_dead_are_dup": (k + n_dead) / n_all}
    res["dead_link_bounds_at_primary"] = bounds

    # ---- duplicate groups spanning communities (connected components at the primary radius)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    for s in range(0, n, 1024):
        d = np.bitwise_count(h[s:s + 1024, None] ^ h[None, :])
        ii, jj = np.nonzero(d <= T)
        for i, j in zip(ii + s, jj):
            if i < j:
                pairs.append((int(i), int(j), int(d[i - s, j])))
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
    comp = np.array([find(i) for i in range(n)])
    u["component"] = comp
    g = u.groupby("component")
    span_all = g.community.nunique()
    bbu = u[u.community.isin(BACKBONE)]
    span_bb = bbu.groupby("component").community.nunique()
    cross_pairs = [(i, j, d) for i, j, d in pairs if lab[i] != lab[j]]
    cross_pairs_bb = [(i, j, d) for i, j, d in cross_pairs if is_bb[i] and is_bb[j]]
    res["groups_at_primary"] = {
        "near_dup_pairs_total": len(pairs),
        "cross_community_pairs": len(cross_pairs),
        "cross_community_pairs_backbone_only": len(cross_pairs_bb),
        "groups_spanning_2plus_communities": int((span_all >= 2).sum()),
        "groups_spanning_2plus_backbone_communities": int((span_bb >= 2).sum()),
        "largest_component_size": int(g.size().max()),
        "posts_in_largest_cross_groups": g.size()[span_all >= 2].sort_values(ascending=False)
        .head(10).astype(int).tolist(),
    }
    cp = pd.DataFrame(cross_pairs, columns=["i", "j", "dist"])
    cp["post_i"] = u.post_id.to_numpy()[cp.i]
    cp["post_j"] = u.post_id.to_numpy()[cp.j]
    cp["comm_i"] = u.community.to_numpy()[cp.i]
    cp["comm_j"] = u.community.to_numpy()[cp.j]
    cp["same_author"] = (u.author.to_numpy()[cp.i] == u.author.to_numpy()[cp.j]) & \
        ~pd.Series(u.author.to_numpy()[cp.i]).isin(["[deleted]"]).to_numpy()
    cp["same_sha"] = u.sha256.to_numpy()[cp.i] == u.sha256.to_numpy()[cp.j]
    cp["lag_hours"] = np.abs(u.created_utc.to_numpy()[cp.i] - u.created_utc.to_numpy()[cp.j]) / 3600
    cp.to_parquet(OUT / "m2_cross_pairs.parquet", index=False)
    if len(cp):
        res["cross_pairs_at_primary"] = {
            "n": len(cp), "same_author_share": float(cp.same_author.mean()),
            "byte_identical_share": float(cp.same_sha.mean()),
            "lag_hours_quantiles": {q: float(cp.lag_hours.quantile(float(q)))
                                    for q in ["0.1", "0.25", "0.5", "0.75", "0.9"]},
            "dist_counts": cp.dist.value_counts().sort_index().to_dict(),
            "by_pair": cp.assign(pair=cp.comm_i + "|" + cp.comm_j).pair.value_counts().to_dict(),
        }
    # degenerate hashes (flat / near-flat images collide by construction)
    pc = np.bitwise_count(h)
    res["degenerate_hash_share"] = {"popcount_lt8_or_gt56": float(((pc < 8) | (pc > 56)).mean()),
                                    "most_common_hash_count": int(pd.Series(h).value_counts().iloc[0])}
    keep = (pc >= 8) & (pc <= 56)
    res["sensitivity_excluding_degenerate_hashes"] = {
        "cross_any_other_at_primary": float((d_other[keep] <= T).mean()), "n": int(keep.sum())}

    # ---- per-day series for the plot
    perday = pd.DataFrame({"day": day, "community": u.community, "dup": d_other <= T}) \
        .groupby(["community", "day"]).dup.agg(["sum", "size"]).reset_index()
    perday.to_csv(OUT / "m2_per_day.csv", index=False)

    # ---- image-free measure: same exact URL in two communities, inside the window
    w = img.copy()
    ncomm = w.groupby("fetch_url").community.nunique()
    w["url_shared"] = w.fetch_url.map(ncomm) >= 2
    res["exact_url_shared_in_window"] = {
        "pooled": float(w.url_shared.mean()),
        "per_community": w.groupby("community").url_shared.mean().reindex(SUBS).to_dict(),
        "among_usable": float(w[w.usable].url_shared.mean()),
        "among_not_usable": float(w[~w.usable].url_shared.mean()) if (~w.usable).any() else None,
    }
    (OUT / "m2_summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("by_radius", "removed_by_category_vs_status")},
                     indent=1, default=float))
    for T in RADII:
        p = by_radius[str(T)]["pooled"]
        print(T, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()})


if __name__ == "__main__":
    main()
