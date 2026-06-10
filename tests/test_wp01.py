"""Unit tests for WP-01 helpers (no GPU / network)."""

from __future__ import annotations

import random

from experiments.wp01.negatives import build_candidates_for_item, shuffle_words


def test_shuffle_words_changes_order():
    rng = random.Random(0)
    out = shuffle_words("one two three four", rng)
    assert out != "one two three four"


def test_build_candidates_has_true_label():
    rng = random.Random(1)
    all_texts = {"a": "hello world", "b": "goodbye moon"}
    groups = {"a": ["a", "b"], "b": ["a", "b"]}
    cands = build_candidates_for_item("a", "hello world", all_texts, groups, rng)
    labels = [c["label"] for c in cands]
    assert "true" in labels
    assert len(cands) >= 3
