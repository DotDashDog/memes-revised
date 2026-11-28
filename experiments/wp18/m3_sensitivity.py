"""WP-18 measurement 3, sensitivities NOT pre-registered (declared in expectations.md, resumption
note of 2026-11-27 ~17:30, before any result existed). Lower standing than m3_probe.py's primaries.

(a) 6-way probe without r/whenthe (97% GIF) and a stills-only probe (n_frames == 1, all
    communities, classes rebalanced by downsampling to the smallest), same grouped CV;
(b) share of OCR texts naming a candidate subreddit (a literal label leak inside the image).

    uv run python -m experiments.wp18.m3_sensitivity
"""
from __future__ import annotations

import json
import re
import warnings

import numpy as np
import pandas as pd

from experiments.wp18.common import OUT, SEED, SUBS
from experiments.wp18.m3_probe import boot, make_groups, oof


def run(name, X, y, groups, rng, k):
    pred, fa, Cs = oof(X, y, groups)
    acc = float((pred == y).mean())
    return {"accuracy": acc, "ci95": boot(y, pred, groups, rng), "fold_accuracies": fa,
            "chosen_C": Cs, "n": int(len(y)), "chance": 1 / k,
            "per_class_n": np.bincount(y).tolist()}


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    z = np.load(OUT / "m3_embeddings.npz", allow_pickle=False)
    sample = pd.read_parquet(OUT / "m3_sample.parquet").set_index("post_id")
    s = sample.loc[z["post_id"]].reset_index()
    groups_all = make_groups(s)
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED}

    # (a1) without r/whenthe
    keep = (s.community != "whenthe").to_numpy()
    subs6 = [c for c in SUBS if c != "whenthe"]
    y6 = s.community[keep].map({c: i for i, c in enumerate(subs6)}).to_numpy()
    for key in ["p_img_ocr", "p_vlm_only"]:
        X = np.asarray(z[key][keep], dtype=np.float64)
        out[f"6-way without whenthe, {key}"] = run(key, X, y6, groups_all[keep], rng, 6)
        print(key, "6-way", out[f"6-way without whenthe, {key}"]["accuracy"], flush=True)

    # (a2) stills only, rebalanced to the smallest class
    still = (s.n_frames.fillna(1) <= 1).to_numpy()
    cnt = s[still].community.value_counts().reindex(SUBS).fillna(0).astype(int)
    out["stills_available_per_class"] = cnt.to_dict()
    classes = [c for c in SUBS if cnt[c] >= 100]   # a class with <100 stills is dropped, and said so
    m = int(cnt[classes].min())
    r2 = np.random.default_rng(SEED)
    pick = np.concatenate([r2.choice(np.nonzero(still & (s.community == c).to_numpy())[0], m, replace=False)
                           for c in classes])
    ys = s.community.to_numpy()[pick]
    ys = pd.Series(ys).map({c: i for i, c in enumerate(classes)}).to_numpy()
    for key in ["p_img_ocr", "p_vlm_only"]:
        X = np.asarray(z[key][pick], dtype=np.float64)
        out[f"stills only ({len(classes)} classes x {m}), {key}"] = run(key, X, ys, groups_all[pick], rng,
                                                                         len(classes))
        print(key, "stills", out[f"stills only ({len(classes)} classes x {m}), {key}"]["accuracy"], flush=True)
    out["stills_classes"] = classes

    # (b) literal subreddit names in the OCR text
    ocr = {r["post_id"]: r["text"] for r in map(json.loads, (OUT / "m3_ocr.jsonl").read_text().splitlines())}
    pat = re.compile(r"\b(r\s*/\s*)?(" + "|".join(re.escape(c) for c in SUBS + ["meirl", "me_irl"]) + r")\b",
                     re.I)
    txt = s.post_id.map(lambda p: ocr.get(p, "") or "")
    hit = txt.map(lambda t: bool(pat.search(t)))
    own = [bool(re.search(r"\b" + re.escape(c) + r"\b", t, re.I)) for c, t in zip(s.community, txt)]
    out["ocr_mentions_any_candidate_name_share"] = float(hit.mean())
    out["ocr_mentions_own_community_name_share_by_class"] = \
        pd.Series(own).groupby(s.community.to_numpy()).mean().reindex(SUBS).to_dict()
    # (c) third agent: the "[empty]" bug. Share of text-free images per class, and the primary
    # probe re-run on the bugged embedding (text-free images get the vector of "[empty]").
    out["ocr_empty_share_by_class"] = pd.Series(z["ocr_empty"]).groupby(s.community.to_numpy()) \
        .mean().reindex(SUBS).to_dict()
    y7 = s.community.map({c: i for i, c in enumerate(SUBS)}).to_numpy()
    out["7-way p_img_ocr_bugged ('[empty]' vector kept)"] = run(
        "bugged", np.asarray(z["p_img_ocr_bugged"], dtype=np.float64), y7, groups_all, rng, 7)
    out["7-way 'no text found' indicator only"] = run(
        "empty", z["ocr_empty"].astype(np.float64)[:, None], y7, groups_all, rng, 7)
    (OUT / "m3_sensitivity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
