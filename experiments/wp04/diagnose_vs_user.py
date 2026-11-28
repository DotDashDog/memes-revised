"""Exploratory: which representation's similarity best predicts the USER's
"same meme by meaning" answers on the 100 validation pairs?

The user's answers do not depend on any agent description, so this is the one
test here where a description-based representation is not favoured by how the
labels were made. Caveats: 100 pairs, chosen in strata defined by agent labels;
26 are near-identical images (reported separately).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

S = Path("/home/linus/memes-revised/experiments/WP-04/sample-001")
pairs = pd.read_csv(S / "user_check/pairs_scored.csv")
pairs = pairs[pairs.family != "unsure"]
emb_ids = json.loads((S / "ids.json").read_text()); pos = {p: i for i, p in enumerate(emb_ids)}
desc = {x["id"]: x for x in json.loads((S / "stage1/descriptions.json").read_text())}
ocr = json.loads((S / "ocr.json").read_text())
ids = sorted(set(pairs.a) | set(pairs.b)); ix = {p: i for i, p in enumerate(ids)}
rows = [pos[i] for i in ids]
unit = lambda x: x / np.where(np.linalg.norm(x, axis=1, keepdims=True) > 0, np.linalg.norm(x, axis=1, keepdims=True), 1)
img = unit(np.load(S / "embeddings_p_vlm_only.npy")[rows].astype(float))
both = np.load(S / "embeddings_p_img_ocr.npy")[rows].astype(float)

from sentence_transformers import SentenceTransformer
st = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
def enc(texts):
    e = st.encode([t if t else " " for t in texts], normalize_embeddings=True, show_progress_bar=False).astype(float)
    e[np.array([t == "" for t in texts])] = 0
    return e
ocr_t = [(ocr[i].get("text") or "").strip() for i in ids]
about = ["; ".join(desc[i]["referents"]) + ". " + desc[i]["premise"] for i in ids]
refs = ["; ".join(desc[i]["referents"]) for i in ids]
reps = {
    "image only (CLIP)": img,
    "image + extracted text (pipeline as built)": unit(both),
    "extracted text, sentence model": enc(ocr_t),
    "agent description: what the joke refers to + one-line premise, sentence model": enc(about),
    "agent description: referents only, sentence model": enc(refs),
}
y = (pairs.family == "yes").to_numpy()
near = pairs.near_identical.to_numpy()
print("pairs", len(pairs), "user says same meme:", int(y.sum()), "| without near-identical:", int((~near).sum()), int(y[~near].sum()))
res = {}
rng = np.random.default_rng(0)
for name, X in reps.items():
    s = np.array([X[ix[a]] @ X[ix[b]] for a, b in zip(pairs.a, pairs.b)])
    def auc_ci(mask):
        yy, ss = y[mask], s[mask]
        a = roc_auc_score(yy, ss)
        bs = []
        for _ in range(2000):
            k = rng.integers(0, len(yy), len(yy))
            if 0 < yy[k].sum() < len(k): bs.append(roc_auc_score(yy[k], ss[k]))
        return round(float(a), 3), [round(float(v), 3) for v in np.percentile(bs, [2.5, 97.5])]
    res[name] = {"all_pairs": auc_ci(np.ones(len(y), bool)), "without_near_identical": auc_ci(~near)}
    print(f"{name:80s} all {res[name]['all_pairs']}  no-dups {res[name]['without_near_identical']}")
(S / "user_check/representation_vs_user.json").write_text(json.dumps(res, indent=1))
