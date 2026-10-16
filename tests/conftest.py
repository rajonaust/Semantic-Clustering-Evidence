"""
Shared test fixtures.

The real pipeline depends on `sentence-transformers` and `keybert`, which
pull in a multi-GB torch install. Unit tests shouldn't need a real model to
verify the pipeline's *logic* (clustering, thresholding, scoring, merging,
grid building) -- so before any pipeline module is imported, we register
lightweight stand-ins for those two packages in sys.modules. Every stage
module does `from sentence_transformers import SentenceTransformer` /
`from keybert import KeyBERT` at import time, so this has to happen before
the first `import semantic_clustering_evidence...` anywhere in the suite.

Tests that actually need encoder behavior inject a `FakeEncoder` (below)
through the `model=` parameter every stage function already accepts for
exactly this kind of reuse/injection -- they never rely on the stub's own
(intentionally non-functional) encode().
"""
import sys
import types

import numpy as np
import pytest


def _install_ml_stubs():
    if "sentence_transformers" not in sys.modules:
        fake_st = types.ModuleType("sentence_transformers")

        class _StubSentenceTransformer:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "SentenceTransformer stub was instantiated directly -- "
                    "tests should inject a FakeEncoder via the model= parameter "
                    "instead of letting pipeline code load a real model."
                )

        fake_st.SentenceTransformer = _StubSentenceTransformer
        sys.modules["sentence_transformers"] = fake_st

    if "keybert" not in sys.modules:
        fake_kb = types.ModuleType("keybert")

        class _StubKeyBERT:
            def __init__(self, *args, **kwargs):
                pass

            def extract_keywords(self, *args, **kwargs):
                raise RuntimeError(
                    "KeyBERT stub was called directly -- inject a fake kw_model "
                    "in tests instead."
                )

        fake_kb.KeyBERT = _StubKeyBERT
        sys.modules["keybert"] = fake_kb


_install_ml_stubs()


class FakeEncoder:
    """Deterministic drop-in for a SentenceTransformer.encode() call.

    Maps each text to a bag-of-words embedding over a fixed vocabulary
    (word presence, L2-normalized), so texts sharing vocabulary score higher
    on cosine similarity than texts that don't -- enough structure for
    segmentation/clustering/threshold tests to exercise real decision logic
    without a real model.
    """

    def __init__(self, vocab=None, dim=16):
        self.vocab = vocab or [
            "nap", "sleep", "tired", "bed",
            "sketch", "design", "deadline", "freelance",
            "photo", "camera", "wedding", "utah",
            "religion", "buddhist", "christian", "agnostic",
        ]
        self.dim = dim

    def encode(self, text):
        if isinstance(text, list):
            return np.stack([self._embed_one(t) for t in text])
        return self._embed_one(text)

    def _embed_one(self, text):
        words = set(str(text).lower().split())
        vec = np.array([1.0 if w in words else 0.0 for w in self.vocab], dtype=float)
        if vec.sum() == 0:
            # Give every unmatched text the same tiny, distinct direction
            # rather than an all-zero vector (which breaks cosine similarity).
            vec = np.full(len(self.vocab), 1e-6)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


@pytest.fixture
def fake_encoder():
    return FakeEncoder()
