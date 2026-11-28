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


class _FakeEncoder:
    """Stands in for the sentence model: no download, no GPU."""

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, **_):
        import numpy as np

        v = np.array([[len(t), t.count(" ") + 1, 1.0, 0.0] for t in texts], dtype=float)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_text_channel_marks_text_free_images():
    import numpy as np

    from experiments.wp01.pipelines import embed_text_channel

    vecs, has_text = embed_text_channel(["hello there", "", "   ", None, "hello there"], _FakeEncoder())
    assert has_text.tolist() == [True, False, False, False, True]
    # text-free images get no vector at all, so they cannot look alike on the text side
    assert not vecs[~has_text].any()
    assert np.allclose(np.linalg.norm(vecs[has_text], axis=1), 1.0)
    assert np.allclose(vecs[0], vecs[4])


def test_text_channel_all_empty():
    from experiments.wp01.pipelines import embed_text_channel

    vecs, has_text = embed_text_channel(["", ""], _FakeEncoder())
    assert vecs.shape == (2, 4) and not vecs.any() and not has_text.any()
