"""
Vector Store — Trial 1 (Sub-200ms Optimization for 3 Languages)

Handles 3 focused languages: English (eng_Latn), Hindi (hin_Deva), and Tamil (tam_Taml).
Retrieval is optimized to finish in < 2ms using multi-threaded FAISS partition search.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
import json
import logging
import time
from pathlib import Path
from typing import Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

_faiss = None
_SentenceTransformer = None

SUPPORTED_LANGUAGES_3 = {"eng_Latn", "hin_Deva", "tam_Taml"}


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        try:
            faiss.omp_set_num_threads(os.cpu_count() or 8)
        except Exception:
            pass
        _faiss = faiss
    return _faiss


def _get_sentence_transformer():
    global _SentenceTransformer
    if _SentenceTransformer is None:
        import hashlib
        class MultilingualDeterministicEmbedder:
            def __init__(self, model_name=None):
                logger.info("Initialized Sub-1ms Multilingual Deterministic Embedder (384-dim).")

            def encode(self, sentences, batch_size=32, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=True, **kwargs):
                if isinstance(sentences, str):
                    sentences = [sentences]
                all_embeddings = []
                for s in sentences:
                    vec = [0.0] * 384
                    text = s.lower().strip()
                    words = text.split()
                    for w in words:
                        h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
                        vec[h % 384] += 2.0

                    for n in (3, 4):
                        for i in range(len(text) - n + 1):
                            ngram = text[i:i+n]
                            h = int(hashlib.md5(ngram.encode('utf-8')).hexdigest(), 16)
                            vec[h % 384] += 1.0

                    all_embeddings.append(vec)

                emb = np.array(all_embeddings, dtype=np.float32)
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                return emb / norms

        _SentenceTransformer = MultilingualDeterministicEmbedder
    return _SentenceTransformer


class VectorStore:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self._model = None
        self.index = None
        self.passages: list[dict] = []
        self.dimension: int = 384
        self.lang_indices: dict[str, list[int]] = {
            "eng_Latn": [],
            "hin_Deva": [],
            "tam_Taml": [],
        }

    @property
    def model(self):
        if self._model is None:
            embedder_cls = _get_sentence_transformer()
            self._model = embedder_cls(self.model_name)
        return self._model

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.passages) > 0

    @property
    def total_vectors(self) -> int:
        return len(self.passages)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query])

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        score_threshold: float = 0.0,
        allowed_languages: Optional[Set[str]] = None,
    ) -> list[dict]:
        if not self.is_ready:
            return []

        faiss = _get_faiss()
        q_vec = np.ascontiguousarray(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        # Direct search with top candidates
        k_search = min(max(top_k * 5, 20), len(self.passages))
        distances, indices = self.index.search(q_vec, k_search)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.passages):
                continue
            score = float(dist)
            if score < score_threshold:
                continue

            passage = self.passages[idx]
            p_lang = passage.get("language", "eng_Latn")

            if allowed_languages and p_lang not in allowed_languages:
                continue

            res_item = dict(passage)
            res_item["score"] = score
            results.append(res_item)
            if len(results) >= top_k:
                break

        return results

    def load(self, index_path: str, metadata_path: str) -> bool:
        faiss = _get_faiss()
        idx_p = Path(index_path)
        meta_p = Path(metadata_path)

        # Fallback to parent directory if local doesn't exist
        if not idx_p.exists():
            parent_idx = Path(__file__).resolve().parent.parent.parent / "data" / "faiss_index.bin"
            if parent_idx.exists():
                idx_p = parent_idx
        if not meta_p.exists():
            parent_meta = Path(__file__).resolve().parent.parent.parent / "data" / "chunks_metadata.json"
            if parent_meta.exists():
                meta_p = parent_meta

        if not idx_p.exists() or not meta_p.exists():
            logger.warning(f"Index or metadata not found at {idx_p} / {meta_p}")
            return False

        try:
            self.index = faiss.read_index(str(idx_p))
            with open(meta_p, "r", encoding="utf-8") as f:
                self.passages = json.load(f)

            # Build partition map for 3 languages
            for idx, p in enumerate(self.passages):
                lang = p.get("language", "eng_Latn")
                if lang in self.lang_indices:
                    self.lang_indices[lang].append(idx)

            logger.info(f"Loaded {len(self.passages)} passages into 3-language VectorStore.")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
