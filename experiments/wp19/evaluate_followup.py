"""Exploratory follow-up to evaluate_descriptions.py (written after seeing its results).

1. Paired bootstrap on the user's 99 pairs: is the 8B model's description really better
   than plain extracted text, and really worse than the Opus description?
2. Does using the local model's own transcription, or description plus transcription,
   change the picture?
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.wp19.evaluate_descriptions import S, OUT, as_text, evaluate, unit

lab = pd.read_parquet(S / "labels.parquet"); ids = lab.post_id.tolist(); ix = {p: i for i, p in enumerate(ids)}
pairs = pd.read_csv(S / "user_check/pairs_scored.csv"); pairs = pairs[pairs.family != "unsure"]
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
def enc(texts):
    e = st.encode([t if t else " " for t in texts], batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float64)
    e[np.array([t == "" for t in texts])] = 0.0
    return e
def load(path):
    return {d["id"]: d for d in json.loads(Path(path).read_text())}
ocr = json.loads((S / "ocr.json").read_text())
opus = load(S / "stage1/descriptions.json"); q8 = load(OUT / "descriptions_qwen3vl8b.json"); q4 = load(OUT / "descriptions_qwen3vl4b.json")
def desc(dd): return [as_text(dd.get(i, {})) if not dd.get(i, {}).get("error") else "" for i in ids]
def text(dd): return [str(dd.get(i, {}).get("text") or "").strip() for i in ids]
def both(dd): return [(a + " " + b).strip() for a, b in zip(desc(dd), text(dd))]
reps = {
    "extracted text (EasyOCR)": enc([(ocr[i].get("text") or "").strip() for i in ids]),
    "8B transcription only": enc(text(q8)),
    "8B description": enc(desc(q8)),
    "8B description + its transcription": enc(both(q8)),
    "4B description + its transcription": enc(both(q4)),
    "Opus transcription only": enc(text(opus)),
    "Opus description": enc(desc(opus)),
    "Opus description + its transcription": enc(both(opus)),
}
rng = np.random.default_rng(1)
res = {k: evaluate(k, X, lab, pairs, rng) for k, X in reps.items()}
df = pd.DataFrame({k: {"user AUC": v["user_auc"], "pair AUC": v["pair_auc"], "NN x-tpl": v["nn_same_family_cross_template"],
                       "link AUC": v["link_auc"], "kept@P90": v["family_members_kept_at_p90"], "kept@P70": v["family_members_kept_at_p70"]}
                   for k, v in res.items()}).T.round(3)
print(df.to_string())
y = (pairs.family == "yes").to_numpy()
sims = {k: np.array([unit(X)[ix[a]] @ unit(X)[ix[b]] for a, b in zip(pairs.a, pairs.b)]) for k, X in reps.items()}
def paired(a, b):
    d = []
    for _ in range(4000):
        k = rng.integers(0, len(y), len(y))
        if 0 < y[k].sum() < len(k):
            d.append(roc_auc_score(y[k], sims[a][k]) - roc_auc_score(y[k], sims[b][k]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return round(float(roc_auc_score(y, sims[a]) - roc_auc_score(y, sims[b])), 3), [round(float(lo), 3), round(float(hi), 3)]
diffs = {f"{a} minus {b}": paired(a, b) for a, b in [
    ("8B description", "extracted text (EasyOCR)"), ("Opus description", "8B description"),
    ("8B description + its transcription", "extracted text (EasyOCR)"), ("Opus description", "8B description + its transcription"),
    ("8B transcription only", "extracted text (EasyOCR)")]}
for k, v in diffs.items(): print(k, v)
(OUT / "description_followup.json").write_text(json.dumps({"representations": {k: {a: b for a, b in v.items() if a != "curve"} for k, v in res.items()}, "user_auc_paired_differences": diffs}, indent=1))
