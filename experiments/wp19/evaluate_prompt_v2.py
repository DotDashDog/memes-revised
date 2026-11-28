"""Compare the revised description prompt with the first one, same model and precision.

Plan and expectations: WP-04 log, entry of 2026-11-28 "Revised prompt for the local
description model", committed before the revised descriptions existed.

Paired intervals:
  user score            resample the user's pairs (as in evaluate_followup.py)
  agent-label scores    resample whole families (item weight = times its family was drawn),
                        the same scheme as experiments/wp04/score_audit.py. Pair weights are
                        the product of the two item weights.

Usage: uv run python -m experiments.wp19.evaluate_prompt_v2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import roc_auc_score

from experiments.wp04.score_audit import per_item_pr, recall_at_precision
from experiments.wp19.evaluate_descriptions import OUT, POPULARITY, S, THR, as_text, evaluate, unit

N_BOOT_USER, N_BOOT_FAM = 4000, 400


def curves(X, fam):
    """Per threshold: per-item precision, and per-item share of other family members kept."""
    Xn = unit(X); Sm = Xn @ Xn.T
    iu = np.triu_indices(len(fam), 1)
    Z = linkage(np.clip(1.0 - Sm, 0.0, 2.0)[iu], method="average")
    fsize = pd.Series(fam).map(pd.Series(fam).value_counts()).to_numpy()
    P, R = [], []
    for t in THR:
        cl = fcluster(Z, t=t, criterion="distance")
        pi, ri = per_item_pr(fam, cl)
        P.append(pi); R.append((ri * fsize - 1) / np.maximum(fsize - 1, 1))
    return np.array(P), np.array(R), Sm[iu]


def main() -> None:
    lab = pd.read_parquet(S / "labels.parquet"); ids = lab.post_id.tolist(); ix = {p: i for i, p in enumerate(ids)}
    fam = lab.family_id.to_numpy()
    pairs = pd.read_csv(S / "user_check/pairs_scored.csv"); pairs = pairs[pairs.family != "unsure"]
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")

    def enc(texts):
        e = st.encode([t if t else " " for t in texts], batch_size=64, normalize_embeddings=True,
                      show_progress_bar=False).astype(np.float64)
        e[np.array([t == "" for t in texts])] = 0.0
        return e

    def load(path): return {d["id"]: d for d in json.loads(Path(path).read_text())}
    def desc(dd): return [as_text(dd.get(i, {})) if not dd.get(i, {}).get("error") else "" for i in ids]
    def text(dd): return [str(dd.get(i, {}).get("text") or "").strip() for i in ids]
    def both(dd): return [(a + " " + b).strip() for a, b in zip(desc(dd), text(dd))]

    ocr = json.loads((S / "ocr.json").read_text())
    v1 = load(OUT / "descriptions_qwen3vl8bfp8.json"); v2 = load(OUT / "descriptions_qwen3vl8bfp8_v2.json")
    opus = load(S / "stage1/descriptions.json")
    reps = {
        "extracted text (EasyOCR)": enc([(ocr[i].get("text") or "").strip() for i in ids]),
        "first prompt": enc(desc(v1)),
        "revised prompt": enc(desc(v2)),
        "first prompt + its transcription": enc(both(v1)),
        "revised prompt + its transcription": enc(both(v2)),
        "Opus description": enc(desc(opus)),
    }
    rng = np.random.default_rng(2)
    res = {k: evaluate(k, X, lab, pairs, rng) for k, X in reps.items()}
    cols = {"user AUC": "user_auc", "pair AUC": "pair_auc", "NN x-tpl": "nn_same_family_cross_template",
            "link AUC": "link_auc", "P(tpl>meme)": "p_template_beats_meme",
            "kept@P90": "family_members_kept_at_p90", "kept@P70": "family_members_kept_at_p70"}
    print(pd.DataFrame({k: {c: v[f] for c, f in cols.items()} for k, v in res.items()}).T.round(3).to_string())

    # screen and formatting counts for the two prompts
    screen = {}
    for name, dd in (("first prompt", v1), ("revised prompt", v2)):
        hits = [i for i in ids if i in dd and POPULARITY.search(json.dumps({k: dd[i].get(k) for k in ("template", "referents", "premise")}))]
        date_hits = [i for i in ids if i in dd and __import__("re").search(r"\b(2019|r/memes)\b", json.dumps({k: dd[i].get(k) for k in ("template", "referents", "premise")}))]
        refs = [r.strip().lower() for i in ids if i in dd for r in (dd[i].get("referents") or []) if isinstance(r, str)]
        screen[name] = {"missing_or_error": sum(1 for i in ids if i not in dd or dd[i].get("error")),
                        "mentions_popularity_words": len(hits), "popularity_examples": hits[:8],
                        "mentions_2019_or_subreddit": len(date_hits),
                        "mean_referents": float(np.mean([len(dd[i].get("referents") or []) for i in ids if i in dd])),
                        "images_with_no_referent": int(sum(1 for i in ids if i in dd and not dd[i].get("referents"))),
                        "distinct_referent_strings": len(set(refs)), "referent_strings": len(refs)}
    print(json.dumps(screen, indent=1))

    # paired differences
    y = (pairs.family == "yes").to_numpy()
    usim = {k: np.array([unit(X)[ix[a]] @ unit(X)[ix[b]] for a, b in zip(pairs.a, pairs.b)]) for k, X in reps.items()}
    cv = {k: curves(X, fam) for k, X in reps.items()}
    _, g = np.unique(fam, return_inverse=True); ng = g.max() + 1
    in_fam = (np.bincount(g)[g] >= 2).astype(float)
    W = rng.multinomial(ng, np.ones(ng) / ng, size=N_BOOT_FAM)[:, g].astype(float)
    iu = np.triu_indices(len(fam), 1)
    sf = (fam[:, None] == fam[None, :])[iu]

    def kept(k, w, target):
        P, R, _ = cv[k]
        Ps = (P * w).sum(1) / w.sum(); den = (w * in_fam).sum()
        return recall_at_precision(Ps, (R * w * in_fam).sum(1) / den, target)

    def diff(a, b):
        out = {}
        d = []
        for _ in range(N_BOOT_USER):
            k = rng.integers(0, len(y), len(y))
            if 0 < y[k].sum() < len(k):
                d.append(roc_auc_score(y[k], usim[a][k]) - roc_auc_score(y[k], usim[b][k]))
        out["user AUC"] = (roc_auc_score(y, usim[a]) - roc_auc_score(y, usim[b]), np.percentile(d, [2.5, 97.5]))
        dp, d90, d70 = [], [], []
        for w in W:
            pw = (w[:, None] * w[None, :])[iu]
            if pw[sf].sum() > 0:
                m = pw > 0
                dp.append(roc_auc_score(sf[m], cv[a][2][m], sample_weight=pw[m]) - roc_auc_score(sf[m], cv[b][2][m], sample_weight=pw[m]))
            d90.append(kept(a, w, 0.90) - kept(b, w, 0.90)); d70.append(kept(a, w, 0.70) - kept(b, w, 0.70))
        one = np.ones(len(fam))
        out["pair AUC"] = (res[a]["pair_auc"] - res[b]["pair_auc"], np.percentile(dp, [2.5, 97.5]))
        out["kept@P90"] = (kept(a, one, 0.90) - kept(b, one, 0.90), np.nanpercentile(d90, [2.5, 97.5]))
        out["kept@P70"] = (kept(a, one, 0.70) - kept(b, one, 0.70), np.nanpercentile(d70, [2.5, 97.5]))
        return {m: [round(float(v), 3), [round(float(x), 3) for x in ci]] for m, (v, ci) in out.items()}

    diffs = {f"{a} minus {b}": diff(a, b) for a, b in [
        ("revised prompt", "first prompt"),
        ("revised prompt + its transcription", "first prompt + its transcription"),
        ("revised prompt", "extracted text (EasyOCR)"),
        ("Opus description", "revised prompt"),
    ]}
    for k, v in diffs.items():
        print(k); [print(f"   {m:10s} {d:+.3f}  95% interval {ci}") for m, (d, ci) in v.items()]
    (OUT / "prompt_v2_eval.json").write_text(json.dumps(
        {"representations": {k: {a: b for a, b in v.items() if a != "curve"} for k, v in res.items()},
         "screen": screen, "paired_differences": diffs,
         "n_boot": {"user_pairs": N_BOOT_USER, "families": N_BOOT_FAM}}, indent=1))


if __name__ == "__main__":
    main()
