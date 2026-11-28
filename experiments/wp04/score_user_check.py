"""Compare the user's pair judgments with the agent labels (labelling protocol,
"User validation, and the criterion for using agent labels").

Per layer, by stratum: of the pairs agents call "same", the share the user also
calls same; of the pairs agents call "different", the share the user also calls
different. Wilson 95% intervals. "Unsure" answers are reported and excluded from
the rates. Near-identical image pairs (perceptual-hash distance <= 4) are also
broken out, since agreement on reposts is trivial.

Usage: uv run python -m experiments.wp04.score_user_check
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

S = Path("/home/linus/memes-revised/experiments/WP-04/sample-001")
WP04 = S.parent


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def main() -> None:
    pairs = pd.DataFrame(json.loads((S / "user_check/pairs.json").read_text()))
    ans = pd.DataFrame([{**json.loads(p.read_text()), "pair_id": p.stem}
                        for p in sorted((S / "user_check/db/answers").glob("p*.json"))])
    df = pairs.merge(ans[["pair_id", "template", "family"]], on="pair_id", validate="1:1")
    dh = pd.read_parquet(WP04 / "dhash.parquet").drop_duplicates("post_id").set_index("post_id").dhash
    df["hash_dist"] = [bin(int(dh[a]) ^ int(dh[b])).count("1") for a, b in zip(df.a, df.b)]
    df["near_identical"] = df.hash_dist <= 4
    for L, fn in (("A", "labels.parquet"), ("B", "labels_B.parquet")):
        lab = pd.read_parquet(S / fn).set_index("post_id")
        for layer, col in (("template", "template_id"), ("family", "family_id")):
            df[f"agent{L}_{layer}"] = [lab.at[a, col] == lab.at[b, col] for a, b in zip(df.a, df.b)]
    df.to_csv(S / "user_check/pairs_scored.csv", index=False)

    out: dict = {"n_pairs": len(df), "near_identical_pairs": int(df.near_identical.sum()),
                 "unsure": {l: int((df[l] == "unsure").sum()) for l in ("template", "family")}}

    def rate(sub: pd.DataFrame, layer: str, agent: str, want_same: bool) -> dict:
        s = sub[(sub[f"agent{agent}_{layer}"] == want_same) & (sub[layer] != "unsure")]
        k = int(((s[layer] == "yes") == want_same).sum())
        lo, hi = wilson(k, len(s))
        return {"n": len(s), "user_agrees": k, "rate": round(k / len(s), 3) if len(s) else None,
                "ci95": [round(lo, 3), round(hi, 3)]}

    for agent in ("A", "B"):
        res = {}
        for layer in ("template", "family"):
            hard = df[df.stratum.isin(["hard_tpl", "hard_ref"])]
            res[layer] = {
                "agents_same__all": rate(df, layer, agent, True),
                "agents_same__excluding_near_identical": rate(df[~df.near_identical], layer, agent, True),
                "agents_different__hard_negatives": rate(hard, layer, agent, False),
                "agents_different__random": rate(df[df.stratum == "random"], layer, agent, False),
                "agents_different__all": rate(df, layer, agent, False),
            }
        out[f"labeller_{agent}"] = res
    (S / "user_check/user_agreement.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("\nDisagreements with labeller A:")
    for layer in ("template", "family"):
        d = df[(df[layer] != "unsure") & ((df[layer] == "yes") != df[f"agentA_{layer}"])]
        print(layer, len(d))
        print(d[["pair_id", "a", "b", "stratum", layer, f"agentA_{layer}", f"agentB_{layer}", "hash_dist"]].to_string(index=False))
    print("\nUnsure:")
    print(df[(df.template == "unsure") | (df.family == "unsure")][["pair_id", "a", "b", "stratum", "template", "family", "agentA_template", "agentA_family"]].to_string(index=False))


if __name__ == "__main__":
    main()
