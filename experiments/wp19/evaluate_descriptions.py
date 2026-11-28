"""Score a set of written meme descriptions as a representation of "which meme is this".

Written 2026-11-27 before any local-model description existed. For each
descriptions file (JSON array of {id, text, template, referents, premise}) the
text "referents. premise" is embedded with a sentence model (bge-base-en-v1.5)
and compared, with the same statistics used to diagnose the image embeddings:

  user AUC      how well pair similarity predicts the user's own "same meme"
                answers on the 99 validation pairs. The only reference here
                that owes nothing to any description. PRIMARY.
  pair AUC      same-meme pairs versus all other pairs, agent labels.
  NN x-tpl      share of images with a family whose nearest neighbour on a
                DIFFERENT template is from the same family.
  link AUC      can one similarity cut-off tell an image's closest family
                member from a unique image's closest neighbour (0.5 = no).
  P(tpl>meme)   how often a different-meme/same-template pair is more similar
                than a same-meme/different-template pair (lower is better).
  kept@P90      for an image with a family, the share of its other family members
                that land in its cluster, averaged over images, with agglomerative
                clustering at the setting where B-cubed precision is 0.90 (no
                clustering scores 0). This is the figure the independent review
                reported as 0.21 for the accepted pipeline. pairs@P90 is the same
                idea counted over pairs, where the largest families dominate.

Expectations. The agent-written descriptions (Opus) are the ceiling, and are
favoured on everything except user AUC because the labels were built from them.
A local model is "good enough to pursue" if its user AUC is clearly above the
sentence-model-on-extracted-text baseline (0.77, interval 0.67-0.86) and close
to the ceiling (0.94), and its pairs@P90 is clearly above the image pipeline's
0.21. It is "not good enough" if it lands at or below extracted text, since
then it adds cost without adding information. What this cannot show: quality on
memes newer than the model's training data (all 797 are from 2019, which every
model here has seen plenty of), and whether descriptions leak how popular a
meme became (screened only by a word list).

Usage: uv run python -m experiments.wp19.evaluate_descriptions name=path [name=path ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import roc_auc_score

from experiments.wp04.score_audit import per_item_pr, recall_at_precision

S = Path("/home/linus/memes-revised/experiments/WP-04/sample-001")
OUT = Path("/home/linus/memes-revised/experiments/WP-19")
THR = np.round(np.arange(0.04, 0.90, 0.02), 3)
POPULARITY = re.compile(r"\b(viral|popular|famous|well[- ]known|iconic|trending|trended|classic|overused|"
                        r"widely (used|shared|known)|went viral|hit meme|successful|beloved|dead meme|old meme)\b", re.I)


def unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


def as_text(d: dict) -> str:
    refs = d.get("referents") or []
    if isinstance(refs, str):
        refs = [refs]
    return ("; ".join(str(r) for r in refs) + ". " + str(d.get("premise") or "")).strip(". ").strip()


def evaluate(name: str, X: np.ndarray, lab: pd.DataFrame, pairs: pd.DataFrame, rng) -> dict:
    ids = lab.post_id.tolist(); ix = {p: i for i, p in enumerate(ids)}
    fam, tpl = lab.family_id.to_numpy(), lab.template_id.to_numpy()
    Xn = unit(X); Sm = Xn @ Xn.T
    n = len(ids); iu = np.triu_indices(n, 1)
    same_f = fam[:, None] == fam[None, :]; same_t = tpl[:, None] == tpl[None, :]
    s, sf, stp = Sm[iu], same_f[iu], same_t[iu]
    in_fam = pd.Series(fam).map(pd.Series(fam).value_counts()).to_numpy() >= 2
    Sd = Sm.copy(); np.fill_diagonal(Sd, -np.inf)
    best_same = np.where(same_f, Sd, -np.inf).max(1); nn_sim = Sd.max(1)
    nnx = np.argmax(np.where(same_t, -np.inf, Sd), axis=1)
    # clustering
    # condensed cosine distance by hand: an all-zero row (no description) is at distance 1 from everything
    D = np.clip(1.0 - Sm, 0.0, 2.0)
    Z = linkage(D[iu], method="average")
    Ps, Rb, Rp, Ri = [], [], [], []
    n_same = sf.sum()
    fsize = pd.Series(fam).map(pd.Series(fam).value_counts()).to_numpy()
    for t in THR:
        cl = fcluster(Z, t=t, criterion="distance")
        pi, ri = per_item_pr(fam, cl)
        Ps.append(pi.mean()); Rb.append(ri[in_fam].mean())
        Rp.append(((cl[:, None] == cl[None, :])[iu] & sf).sum() / n_same)
        # per image: share of its OTHER family members that share its cluster (the figure
        # reported from the independent review; every family member counts equally, so
        # large families do not dominate as they do in the pair count above)
        Ri.append(((ri * fsize - 1) / np.maximum(fsize - 1, 1))[in_fam].mean())
    Ps, Rb, Rp, Ri = map(np.array, (Ps, Rb, Rp, Ri))
    # user pairs
    y = (pairs.family == "yes").to_numpy(); near = pairs.near_identical.to_numpy()
    us = np.array([Xn[ix[a]] @ Xn[ix[b]] for a, b in zip(pairs.a, pairs.b)])
    bs = []
    for _ in range(2000):
        k = rng.integers(0, len(y), len(y))
        if 0 < y[k].sum() < len(k):
            bs.append(roc_auc_score(y[k], us[k]))
    return {
        "user_auc": float(roc_auc_score(y, us)), "user_auc_ci95": [float(v) for v in np.percentile(bs, [2.5, 97.5])],
        "user_auc_without_near_identical": float(roc_auc_score(y[~near], us[~near])),
        "pair_auc": float(roc_auc_score(sf, s)),
        "nn_same_family": float((fam[np.argmax(Sd, 1)] == fam)[in_fam].mean()),
        "nn_same_family_cross_template": float((fam[nnx] == fam)[in_fam].mean()),
        "link_auc": float(roc_auc_score(np.r_[np.ones(in_fam.sum()), np.zeros((~in_fam).sum())],
                                        np.r_[best_same[in_fam], nn_sim[~in_fam]])),
        "p_template_beats_meme": float(roc_auc_score(
            np.r_[np.ones((~sf & stp).sum()), np.zeros((sf & ~stp).sum())], np.r_[s[~sf & stp], s[sf & ~stp]])),
        "pairs_kept_at_p90": recall_at_precision(Ps, Rp), "bcubed_recall_at_p90": recall_at_precision(Ps, Rb),
        "pairs_kept_at_p70": recall_at_precision(Ps, Rp, 0.70),
        "family_members_kept_at_p90": recall_at_precision(Ps, Ri),
        "family_members_kept_at_p70": recall_at_precision(Ps, Ri, 0.70),
        "curve": {"precision": Ps.round(4).tolist(), "pairs_kept": Rp.round(4).tolist()},
    }


def main() -> None:
    lab = pd.read_parquet(S / "labels.parquet")
    ids = lab.post_id.tolist()
    pairs = pd.read_csv(S / "user_check/pairs_scored.csv")
    pairs = pairs[pairs.family != "unsure"]
    rng = np.random.default_rng(0)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")

    def enc(texts):
        e = st.encode([t if t else " " for t in texts], batch_size=64, normalize_embeddings=True,
                      show_progress_bar=False).astype(np.float64)
        e[np.array([t == "" for t in texts])] = 0.0
        return e

    emb_ids = json.loads((S / "ids.json").read_text()); pos = {p: i for i, p in enumerate(emb_ids)}
    rows = [pos[i] for i in ids]
    img = unit(np.load(S / "embeddings_p_vlm_only.npy")[rows].astype(np.float64))
    ocr = json.loads((S / "ocr.json").read_text())
    reps = {"image only (CLIP)": img,
            "image + extracted text (accepted pipeline)": unit(np.load(S / "embeddings_p_img_ocr.npy")[rows].astype(np.float64)),
            "extracted text, sentence model": enc([(ocr[i].get("text") or "").strip() for i in ids])}
    screen = {}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        dd = {d["id"]: d for d in json.loads(Path(path).read_text())}
        texts = [as_text(dd.get(i, {})) if not dd.get(i, {}).get("error") else "" for i in ids]
        missing = sum(1 for i in ids if i not in dd or dd[i].get("error"))
        hits = [i for i in ids if i in dd and POPULARITY.search(json.dumps({k: dd[i].get(k) for k in ("template", "referents", "premise")}))]
        screen[name] = {"missing_or_error": missing, "empty_description": int(sum(t == "" for t in texts)),
                        "mentions_popularity_words": len(hits), "examples": hits[:8],
                        "mean_referents": float(np.mean([len(dd[i].get("referents") or []) for i in ids if i in dd]))}
        reps[f"description by {name}"] = enc(texts)
        reps[f"description by {name} + image"] = np.hstack([enc(texts), img])
    res = {k: evaluate(k, X, lab, pairs, rng) for k, X in reps.items()}
    OUT.mkdir(exist_ok=True)
    (OUT / "description_eval.json").write_text(json.dumps({"representations": res, "screen": screen}, indent=1))
    df = pd.DataFrame({k: {"user AUC": v["user_auc"], "lo": v["user_auc_ci95"][0], "hi": v["user_auc_ci95"][1],
                           "pair AUC": v["pair_auc"], "NN x-tpl": v["nn_same_family_cross_template"],
                           "link AUC": v["link_auc"], "P(tpl>meme)": v["p_template_beats_meme"],
                           "kept@P90": v["family_members_kept_at_p90"], "kept@P70": v["family_members_kept_at_p70"],
                           "pairs@P90": v["pairs_kept_at_p90"]} for k, v in res.items()}).T.round(3)
    print(df.to_string())
    print(json.dumps(screen, indent=1))


if __name__ == "__main__":
    main()
