"""
Embedder interface for rag-local-eval-loop and VartaLaap.
Provides embed, embed_one, and get_model functions.
"""

import numpy as np
from app.vector_store import _get_sentence_transformer

_model = None


def get_model():
    global _model
    if _model is None:
        cls = _get_sentence_transformer()
        _model = cls()
    return _model


def embed_one(text: str) -> np.ndarray:
    model = get_model()
    vecs = model.encode([text])
    return vecs[0]


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_model()
    return model.encode(texts)
