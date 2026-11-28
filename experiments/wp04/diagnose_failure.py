"""Why does clustering not recover meaning-level families? (exploratory diagnostic)

Three explanations and what each predicts, written before running:

H1 the information is absent from the embedding.
   Predicts: same-family pairs are barely more similar than other pairs
   (pair AUC near 0.5) and nearest neighbours are rarely same-family.
H2 the clustering algorithm fails on information that is there and well-behaved.
   Predicts: high pair AUC AND a single similarity threshold separates true
   links from spurious ones (link AUC high), yet clusters are poor. Already
   made unlikely by the review (five methods agree).
H3 the information is there but distance does not mean "same meme" on one scale.
   Predicts: neighbours are good (ranking works) but (a) same-template,
   different-family pairs are MORE similar than same-family, different-template
   pairs, so distance orders by picture before meaning; and/or (b) unique images
   have nearest neighbours as close as real family links, so no global threshold
   separates real links from spurious ones (link AUC well below pair AUC).

Also asked: is the text channel limited by text-extraction errors, by the text
encoder, or by the text itself? Compared: CLIP text encoder on extracted text
(what the pipeline uses), a sentence-embedding model on the same extracted text,
and the sentence model on clean transcriptions written by the labelling agents.
Caution: the family labels were built partly from those transcriptions, so the
last one is favoured; and the family labels failed the user's hard-pair check.

Usage: uv run python -m experiments.wp04.diagnose_failure
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

S = Path("/home/linus/memes-revised/experiments/WP-04/sample-001")


def unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


def main() -> None:
    lab = pd.read_parquet(S / "labels.parquet")
    ids = lab.post_id.tolist()
    emb_ids = json.loads((S / "ids.json").read_text())
    pos = {p: i for i, p in enumerate(emb_ids)}
    rows = [pos[i] for i in ids]
    img = np.load(S / "embeddings_p_vlm_only.npy")[rows].astype(np.float64)
    both = np.load(S / "embeddings_p_img_ocr.npy")[rows].astype(np.float64)
    d = img.shape[1]
    # locate the image half of the concatenation
    if np.allclose(both[:, :d], img, atol=1e-4):
        clip_txt = both[:, d:]
    else:
        assert np.allclose(both[:, d:], img, atol=1e-4)
        clip_txt = both[:, :d]
    ocr = json.loads((S / "ocr.json").read_text())
    ocr_text = [(ocr[i].get("text") or "").strip() for i in ids]
    desc = {x["id"]: x for x in json.loads((S / "stage1/descriptions.json").read_text())}
    clean_text = [(desc[i]["text"] or "").strip() for i in ids]
    empty_ocr = np.array([t == "" for t in ocr_text])
    clip_txt = clip_txt.copy()
    clip_txt[empty_ocr] = 0.0  # undo the "[empty]" artefact

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    def enc(texts):
        e = st.encode([t if t else " " for t in texts], batch_size=64, normalize_embeddings=True,
                      show_progress_bar=False).astype(np.float64)
        e[np.array([t == "" for t in texts])] = 0.0
        return e
    bge_ocr, bge_clean = enc(ocr_text), enc(clean_text)

    img, clip_txt = unit(img), unit(clip_txt)
    reps = {
        "image only (CLIP)": img,
        "extracted text, CLIP text encoder": clip_txt,
        "image + extracted text (pipeline, bug fixed)": np.hstack([img, clip_txt]),
        "extracted text, sentence model": bge_ocr,
        "image + extracted text, sentence model": np.hstack([img, bge_ocr]),
        "clean transcription, sentence model": bge_clean,
        "image + clean transcription, sentence model": np.hstack([img, bge_clean]),
    }

    fam = lab.family_id.to_numpy(); tpl = lab.template_id.to_numpy()
    n = len(lab)
    iu = np.triu_indices(n, 1)
    same_f = (fam[:, None] == fam[None, :]); same_t = (tpl[:, None] == tpl[None, :])
    sf, stp = same_f[iu], same_t[iu]
    in_fam = pd.Series(fam).map(pd.Series(fam).value_counts()).to_numpy() >= 2
    types = {"same meme, same template": sf & stp, "same meme, different template": sf & ~stp,
             "different meme, same template": ~sf & stp, "different meme, different template": ~sf & ~stp}
    out = {"n_items": n, "pair_counts": {k: int(v.sum()) for k, v in types.items()},
           "note_empty_text": {"extracted": int(empty_ocr.sum()), "clean": int(sum(t == "" for t in clean_text))},
           "representations": {}}
    sims_for_plot = {}
    for name, X in reps.items():
        Xn = unit(X)
        Sm = Xn @ Xn.T
        s = Sm[iu]
        np.fill_diagonal(Sm, -np.inf)
        order = np.argsort(-Sm, axis=1)
        nn = order[:, 0]
        # nearest same-family member's similarity and rank, for items that have a family
        Sf = np.where(same_f, Sm, -np.inf)
        best_same = Sf.max(1)
        rank_same = (Sm > best_same[:, None]).sum(1) + 1
        # spurious-link score for unique items: similarity to their nearest neighbour
        nn_sim = Sm.max(1)
        link_auc = roc_auc_score(np.r_[np.ones(in_fam.sum()), np.zeros((~in_fam).sum())],
                                 np.r_[best_same[in_fam], nn_sim[~in_fam]])
        # cross-template family signal
        Sx = np.where(same_t, -np.inf, Sm)
        nnx = np.argmax(Sx, axis=1)
        r = {
            "pair_auc_same_meme_vs_rest": float(roc_auc_score(sf, s)),
            "pair_auc_cross_template_only": float(roc_auc_score(sf[~stp], s[~stp])),
            "median_similarity_by_pair_type": {k: float(np.median(s[v])) for k, v in types.items()},
            "prob_diffmeme_sametemplate_more_similar_than_samememe_difftemplate":
                float(roc_auc_score(np.r_[np.ones(types["different meme, same template"].sum()),
                                          np.zeros(types["same meme, different template"].sum())],
                                    np.r_[s[types["different meme, same template"]],
                                          s[types["same meme, different template"]]])),
            "nearest_neighbour_same_meme": float((fam[nn] == fam)[in_fam].mean()),
            "nearest_neighbour_same_meme_excluding_same_template": float((fam[nnx] == fam)[in_fam].mean()),
            "same_meme_member_within_top5": float((rank_same[in_fam] <= 5).mean()),
            "link_auc_true_link_vs_unique_items_nearest": float(link_auc),
            "median_sim_true_nearest_link": float(np.median(best_same[in_fam])),
            "median_sim_unique_item_nearest_neighbour": float(np.median(nn_sim[~in_fam])),
        }
        out["representations"][name] = r
        sims_for_plot[name] = (s, best_same[in_fam], nn_sim[~in_fam])
    (S / "diagnose_failure.json").write_text(json.dumps(out, indent=1))
    df = pd.DataFrame({k: {
        "pair AUC": v["pair_auc_same_meme_vs_rest"], "pair AUC x-tpl": v["pair_auc_cross_template_only"],
        "NN same": v["nearest_neighbour_same_meme"], "NN same x-tpl": v["nearest_neighbour_same_meme_excluding_same_template"],
        "top5": v["same_meme_member_within_top5"], "link AUC": v["link_auc_true_link_vs_unique_items_nearest"],
        "P(tpl>meme)": v["prob_diffmeme_sametemplate_more_similar_than_samememe_difftemplate"],
    } for k, v in out["representations"].items()}).T.round(3)
    print(out["pair_counts"], out["note_empty_text"])
    print(df.to_string())
    for k, v in out["representations"].items():
        print(k, {a: round(b, 3) for a, b in v["median_similarity_by_pair_type"].items()},
              round(v["median_sim_true_nearest_link"], 3), round(v["median_sim_unique_item_nearest_neighbour"], 3))

    # plot: pair-type similarity distributions and the link-threshold problem
    show = ["image only (CLIP)", "image + extracted text (pipeline, bug fixed)", "clean transcription, sentence model"]
    col = {"same meme, same template": "#2ca02c", "same meme, different template": "#1f77b4",
           "different meme, same template": "#d62728", "different meme, different template": "#7f7f7f"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for j, name in enumerate(show):
        s, true_link, uniq_nn = sims_for_plot[name]
        ax = axes[0, j]
        bins = np.linspace(-0.1, 1, 56)
        for k, m in types.items():
            ax.hist(s[m], bins=bins, density=True, histtype="step", lw=1.8, color=col[k], label=f"{k} ({int(m.sum()):,} pairs)")
        ax.set_title(name, fontsize=10); ax.set_xlabel("cosine similarity of a pair of images"); ax.set_yticks([])
        if j == 0: ax.legend(fontsize=7, loc="upper right"); ax.set_ylabel("share of pairs (each curve sums to 1)")
        ax = axes[1, j]
        ax.hist(true_link, bins=bins, density=True, alpha=0.55, color="#1f77b4", label="image with a family: similarity to its closest family member")
        ax.hist(uniq_nn, bins=bins, density=True, alpha=0.55, color="#7f7f7f", label="unique image: similarity to its closest neighbour (always a wrong link)")
        ax.set_xlabel("cosine similarity of the closest candidate link"); ax.set_yticks([])
        ax.set_title(f"a threshold tells a real link from a wrong one {out['representations'][name]['link_auc_true_link_vs_unique_items_nearest']:.0%} of the time (50% = useless)", fontsize=9)
        if j == 0: ax.legend(fontsize=7, loc="upper left"); ax.set_ylabel("share of images")
    fig.suptitle("Why clustering by distance fails for meaning-level families.\n"
                 "Top: in image-based embeddings, sharing a TEMPLATE (red) makes a pair far more similar than sharing a MEME on different templates (blue).\n"
                 "Bottom: unique images have neighbours almost as close as real family links, so no single cut-off separates them.", fontsize=11)
    fig.tight_layout()
    fig.savefig(S / "plots/diagnose_failure.png", dpi=120)


if __name__ == "__main__":
    main()
