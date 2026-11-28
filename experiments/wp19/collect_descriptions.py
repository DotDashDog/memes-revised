"""Turn raw JSONL responses into the descriptions file, and count failures.

Usage:
  python collect_descriptions.py --raw first.jsonl [--retry retry.jsonl] --out descriptions.json
      [--write-retry-ids ids.txt]

Classification of each raw response (first pass and, if given, the retry pass):
  http_error   request failed
  empty        content is empty/whitespace
  truncated    finish_reason == "length"
  invalid_json content is not a JSON object even after removing a ```json fence
  bad_fields   JSON object missing a field or with a wrong type
  refusal      premise starts with a refusal phrase, or unparseable content matching one
  empty_fields valid JSON whose text, premise and referents are all empty
  ok
Strict JSON (no fence, no surrounding text) is counted separately as `strict_json_ok`,
since a fenced reply is a formatting deviation even when recoverable.
The final file takes the first-pass result if ok, otherwise the retry result if ok,
otherwise an {id, error} record.
"""

import argparse
import json
import pathlib
import re

REFUSAL = re.compile(r"\b(I can(no|')t|I'm (sorry|unable)|I am unable|cannot (help|assist|describe))\b", re.I)
FIELDS = {"text": str, "template": str, "referents": list, "premise": str}


def parse(content):
    s = content.strip()
    strict = True
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", s, re.S)
    if m:
        s, strict = m.group(1), False
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        strict = False
        i, j = s.find("{"), s.rfind("}")
        try:
            if i < 0 or j <= i:
                raise json.JSONDecodeError("no object", s, 0)
            obj = json.loads(s[i : j + 1])
        except json.JSONDecodeError:
            # The model sometimes stops one character early, after the last field and
            # before the closing brace (common with the revised prompt). Appending the
            # brace changes no content; anything else stays a failure.
            try:
                obj = json.loads(s[max(i, 0):].rstrip().rstrip(",") + "}")
            except json.JSONDecodeError:
                return None, False
    return obj, strict


def classify(rec):
    if "http_error" in rec:
        return "http_error", None, False
    c = rec.get("content") or ""
    if not c.strip():
        return "empty", None, False
    if rec.get("finish_reason") == "length":
        return "truncated", None, False
    obj, strict = parse(c)
    if not isinstance(obj, dict):
        return ("refusal" if REFUSAL.search(c) else "invalid_json"), None, False
    for k, t in FIELDS.items():
        if not isinstance(obj.get(k), t):
            return "bad_fields", None, strict
    if isinstance(obj["referents"], list) and not all(isinstance(x, str) for x in obj["referents"]):
        return "bad_fields", None, strict
    if REFUSAL.match(obj["premise"]):
        return "refusal", obj, strict
    if not any([obj["text"].strip(), obj["premise"].strip(), obj["referents"]]):
        return "empty_fields", obj, strict  # valid JSON but nothing described
    return "ok", {k: obj[k] for k in FIELDS}, strict


def load(path):
    recs = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            recs[r["id"]] = r  # last record per id wins
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--retry")
    ap.add_argument("--out")
    ap.add_argument("--ids", default=None, help="file of all expected ids (default: image dir)")
    ap.add_argument("--write-retry-ids")
    a = ap.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[2]
    ids = sorted(p.stem for p in (repo / "experiments/WP-04/sample-001/label_images").glob("*.jpg"))
    first = load(a.raw)
    retry = load(a.retry) if a.retry else {}

    counts1, counts2, strict1 = {}, {}, 0
    final, need_retry = [], []
    for i in ids:
        r = first.get(i)
        st, obj, strict = classify(r) if r else ("missing", None, False)
        counts1[st] = counts1.get(st, 0) + 1
        strict1 += strict and st == "ok"
        if st != "ok":
            need_retry.append(i)
            if i in retry:
                st2, obj2, _ = classify(retry[i])
                counts2[st2] = counts2.get(st2, 0) + 1
                if st2 == "ok":
                    obj, st = obj2, "ok"
                else:
                    st = f"first:{st}; retry:{st2}"
        final.append({"id": i, **obj} if st == "ok" else {"id": i, "error": st})

    summary = {
        "n_ids": len(ids),
        "first_pass": counts1,
        "first_pass_strict_json_ok": strict1,
        "retry_pass": counts2,
        "final_ok": sum("error" not in f for f in final),
        "final_failed": sum("error" in f for f in final),
    }
    print(json.dumps(summary, indent=1))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(final, ensure_ascii=False, indent=1))
        pathlib.Path(a.out).with_suffix(".failures.json").write_text(json.dumps(summary, indent=1))
    if a.write_retry_ids:
        pathlib.Path(a.write_retry_ids).write_text("\n".join(need_retry) + ("\n" if need_retry else ""))


if __name__ == "__main__":
    main()
