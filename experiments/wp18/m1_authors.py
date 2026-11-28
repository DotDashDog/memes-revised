"""WP-18 measurement 1: unique authors per community-month (ADR 0016 precondition 1).

Definitions are fixed in experiments/WP-18/expectations.md. Deterministic; no randomness.

    uv run python -m experiments.wp18.m1_authors
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from experiments.wp18.common import DATA, OUT, SUBS, parquet_path

DELETED = {"[deleted]", "[removed]"}
ACTIVE_POSTS = 1000          # first month with >= this many posts starts the "active life"
NEAR_ZERO = 100              # N_named below this inside the active life is flagged
STEP = 2.0                   # month-on-month factor flagged as a discontinuity


def monthly(sub: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pq.read_table(parquet_path(sub),
                      columns=["created_utc", "author", "removed_by_category"]).to_pandas()
    ts = pd.to_datetime(t.created_utc, unit="s", utc=True)
    t["month"] = ts.dt.strftime("%Y-%m")
    t["day"] = ts.dt.strftime("%Y-%m-%d")
    t["is_deleted"] = t.author.isna() | t.author.isin(DELETED)
    t["is_removed"] = t.removed_by_category.notna()
    named = t[~t.is_deleted]
    g = t.groupby("month")
    m = pd.DataFrame({
        "posts": g.size(),
        "deleted_author_posts": g.is_deleted.sum(),
        "removed_posts": g.is_removed.sum(),
        "n_named": named.groupby("month").author.nunique(),
    }).fillna({"n_named": 0}).astype(int)
    # fill calendar gaps so a month with no rows at all shows up as zero, not as absence
    full = pd.period_range(m.index.min(), "2025-12", freq="M").strftime("%Y-%m")
    m = m.reindex(full, fill_value=0)
    m.index.name = "month"
    m["deleted_share"] = m.deleted_author_posts / m.posts.replace(0, np.nan)
    m["removed_share"] = m.removed_posts / m.posts.replace(0, np.nan)
    m["posts_per_named_author"] = (m.posts - m.deleted_author_posts) / m.n_named.replace(0, np.nan)
    m.insert(0, "subreddit", sub)
    daily = t.groupby("day").agg(posts=("author", "size"),
                                 n_named=("author", lambda s: s[~s.isin(DELETED)].nunique()))
    daily.insert(0, "subreddit", sub)
    return m.reset_index(), daily.reset_index()


def judge(m: pd.DataFrame) -> dict:
    sub = m.subreddit.iloc[0]
    big = m.index[m.posts >= ACTIVE_POSTS]
    if len(big) == 0:
        return {"subreddit": sub, "active_from": None}
    a = m.loc[big[0]:].copy()
    logn = np.log(a.n_named.replace(0, np.nan))
    d = logn.diff()
    mad = np.nanmedian(np.abs(d - np.nanmedian(d)))
    a["factor"] = np.exp(d)
    a["robust_z"] = (d - np.nanmedian(d)) / (1.4826 * mad)
    steps = a[(a.factor >= STEP) | (a.factor <= 1 / STEP) | a.factor.isna().shift(-0) & (a.index != a.index[0])]
    return {
        "subreddit": sub,
        "active_from": a.month.iloc[0],
        "active_months": int(len(a)),
        "min_n_named": int(a.n_named.min()),
        "min_n_named_month": a.month.loc[a.n_named.idxmin()],
        "max_n_named": int(a.n_named.max()),
        "max_n_named_month": a.month.loc[a.n_named.idxmax()],
        "n_named_2025_12": int(a.n_named.iloc[-1]),
        "near_zero_months": a.month[a.n_named < NEAR_ZERO].tolist(),
        "months_under_1000": int((a.n_named < 1000).sum()),
        "steps_ge_2x": [
            {"month": r.month, "factor": None if pd.isna(r.factor) else round(float(r.factor), 3),
             "n_prev": int(a.n_named.shift(1).loc[i]) if not pd.isna(a.n_named.shift(1).loc[i]) else None,
             "n": int(r.n_named)}
            for i, r in steps.iterrows()],
        "median_abs_monthly_change_pct": round(float(100 * (np.exp(np.nanmedian(np.abs(d))) - 1)), 2),
        "months_abs_robust_z_gt_5": [
            {"month": r.month, "factor": round(float(r.factor), 3), "z": round(float(r.robust_z), 1)}
            for _, r in a[a.robust_z.abs() > 5].iterrows()],
        "deleted_share_min_pct": round(float(100 * a.deleted_share.min()), 2),
        "deleted_share_max_pct": round(float(100 * a.deleted_share.max()), 2),
        "deleted_share_by_year_pct": {
            y: round(float(100 * g.deleted_author_posts.sum() / g.posts.sum()), 2)
            for y, g in a.groupby(a.month.str[:4])},
        "removed_share_by_year_pct": {
            y: round(float(100 * g.removed_posts.sum() / g.posts.sum()), 2)
            for y, g in a.groupby(a.month.str[:4])},
        "loglog_slope_n_vs_posts": round(float(np.polyfit(np.log(a.posts[a.n_named > 0]),
                                                           np.log(a.n_named[a.n_named > 0]), 1)[0]), 3),
        "loglog_r2_n_vs_posts": round(float(np.corrcoef(np.log(a.posts[a.n_named > 0]),
                                                        np.log(a.n_named[a.n_named > 0]))[0, 1] ** 2), 4),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    months, days, verdicts = [], [], []
    for sub in SUBS:
        m, d = monthly(sub)
        months.append(m)
        days.append(d)
        verdicts.append(judge(m))
        print(sub, len(m), "months")
    pd.concat(months).to_csv(OUT / "m1_authors_monthly.csv", index=False)
    pd.concat(days).to_parquet(OUT / "m1_daily.parquet", index=False)
    (OUT / "m1_summary.json").write_text(json.dumps(
        {"definitions": {"active_posts": ACTIVE_POSTS, "near_zero": NEAR_ZERO, "step": STEP,
                         "deleted_tokens": sorted(DELETED) + ["<null>"]},
         "data_dir": str(DATA), "communities": verdicts}, indent=1))
    print(json.dumps(verdicts, indent=1))


if __name__ == "__main__":
    main()
