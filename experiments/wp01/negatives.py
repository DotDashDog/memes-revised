"""Hard-negative candidate generation for text-fidelity retrieval."""

from __future__ import annotations

import random
import re
from typing import Any

from experiments.wp01.dhash import hamming


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def shuffle_words(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 3:
        return text[::-1]
    rng.shuffle(words)
    return " ".join(words)


def minimal_negation_flip(text: str) -> str | None:
    lower = text.lower()
    for a, b in (("not ", ""), ("never ", "always "), ("no ", "yes ")):
        if a in lower:
            return lower.replace(a, b, 1)
    words = text.split()
    if len(words) >= 2:
        return "not " + text
    return None


def number_entity_swap(text: str, rng: random.Random) -> str | None:
    def repl(m: re.Match[str]) -> str:
        n = int(m.group(0))
        return str(n + rng.choice([-1, 1, 2]))

    swapped = re.sub(r"\b\d+\b", repl, text, count=1)
    return swapped if swapped != text else None


def ocr_corrupt(text: str, rng: random.Random) -> str:
    if len(text) < 4:
        return text + "x"
    idx = rng.randrange(len(text))
    chars = list(text)
    chars[idx] = rng.choice("l1O0|" if chars[idx].isalpha() else "x")
    return "".join(chars)


def one_word_swap(text: str, rng: random.Random, vocabulary: list[str]) -> str | None:
    words = text.split()
    if len(words) < 2:
        return None
    idx = rng.randrange(len(words))
    alt = rng.choice(vocabulary)
    if alt.lower() == words[idx].lower():
            return None
    words[idx] = alt
    return " ".join(words)


def build_candidates_for_item(
    item_id: str,
    true_text: str,
    all_texts: dict[str, str],
    template_groups: dict[str, list[str]],
    rng: random.Random,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = [{"text": true_text, "label": "true"}]
    seen = {true_text.strip().lower()}

    def add(text: str, label: str) -> None:
        key = text.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        candidates.append({"text": text, "label": label})

    # Same-template different caption
    group = template_groups.get(item_id, [])
    for other_id in group:
        if other_id == item_id:
            continue
        other = all_texts.get(other_id, "")
        if other:
            add(other, "same_template_caption")
            break

    # Other captions from corpus
    others = [t for tid, t in all_texts.items() if tid != item_id and t]
    rng.shuffle(others)
    for other in others[:3]:
        add(other, "other_caption")

    vocab = []
    for t in all_texts.values():
        vocab.extend(_tokenize(t))
    vocab = list({w for w in vocab if len(w) > 2})

    corrupted = ocr_corrupt(true_text, rng)
    add(corrupted, "ocr_corrupted")

    shuffled = shuffle_words(true_text, rng)
    add(shuffled, "shuffled_words")

    neg = minimal_negation_flip(true_text)
    if neg:
        add(neg, "negation_flip")

    num = number_entity_swap(true_text, rng)
    if num:
        add(num, "number_swap")

    if vocab:
        swap = one_word_swap(true_text, rng, vocab)
        if swap:
            add(swap, "one_word_swap")

    return candidates


def assign_template_groups(
    sample_items: list[dict[str, Any]],
    *,
    max_hamming: int = 10,
) -> dict[str, list[str]]:
    """Group items with similar dHash as proxy meme templates."""
    ids = [it["id"] for it in sample_items]
    hashes = {it["id"]: int(it.get("dhash", 0)) for it in sample_items}
    groups: dict[str, list[str]] = {i: [i] for i in ids}

    for i, hi in hashes.items():
        for j, hj in hashes.items():
            if i >= j:
                continue
            if hamming(hi, hj) <= max_hamming:
                groups.setdefault(i, []).append(j)
                groups.setdefault(j, []).append(i)

    return groups


def build_all_candidates(
    sample: dict[str, Any],
    transcripts: dict[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    items = sample["items"]
    all_texts = {
        iid: data.get("in_image_text", "")
        for iid, data in transcripts["items"].items()
        if data.get("in_image_text", "").strip()
    }
    template_groups = assign_template_groups(items)

    out_items: dict[str, Any] = {}
    for entry in items:
        item_id = entry["id"]
        true_text = all_texts.get(item_id, "")
        if not true_text.strip():
            continue
        out_items[item_id] = {
            "true_text": true_text,
            "candidates": build_candidates_for_item(
                item_id, true_text, all_texts, template_groups, rng
            ),
        }

    return {
        "version": 1,
        "seed": seed,
        "items": out_items,
        "n_items": len(out_items),
    }
