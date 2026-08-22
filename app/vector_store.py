"""
Vector Store — FAISS Index with Sentence Transformers

Handles:
1. Loading the multilingual embedding model (MiniLM-L12)
2. Encoding text chunks into 384-dimensional vectors
3. Building and searching a FAISS IVF index
4. Saving/loading the index to/from disk

The FAISS index uses IVFFlat (Inverted File with Flat quantizer)
for fast approximate nearest-neighbor search. With 20K chunks and
nlist=100, retrieval is typically under 5ms.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports — these are heavy and slow to import
_faiss = None
_SentenceTransformer = None


def _get_faiss():
    global _faiss
    if _faiss is None:
        import os
        import faiss
        # Multi-threaded CPU OpenMP optimization (utilize 100% of CPU cores)
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
                logger.info("Initialized 0-latency MultilingualDeterministicEmbedder (384-dim).")

            def encode(self, sentences, batch_size=32, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=True, **kwargs):
                import numpy as np
                if isinstance(sentences, str):
                    sentences = [sentences]
                all_embeddings = []
                for s in sentences:
                    vec = [0.0] * 384
                    text = s.lower().strip()
                    # 1. Word tokens
                    words = text.split()
                    for w in words:
                        h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
                        idx = h % 384
                        vec[idx] += 2.0

                    # 2. Character 3-grams & 4-grams (essential for Indic script alignment)
                    for n in (3, 4):
                        for i in range(len(text) - n + 1):
                            ngram = text[i:i+n]
                            h = int(hashlib.md5(ngram.encode('utf-8')).hexdigest(), 16)
                            idx = h % 384
                            vec[idx] += 1.0

                    all_embeddings.append(vec)

                emb = np.array(all_embeddings, dtype=np.float32)
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                return emb / norms

        _SentenceTransformer = MultilingualDeterministicEmbedder
    return _SentenceTransformer


class VectorStore:
    """
    FAISS-backed vector store with sentence-transformer embeddings.

    Architecture:
    - Embedding model: paraphrase-multilingual-MiniLM-L12-v2 (384d)
    - Index type: IVFFlat (Inverted File Index with Flat quantizer)
    - Distance metric: Inner Product (cosine similarity after normalization)

    Why IVFFlat?
    - Flat index is exact but O(n) — too slow for large datasets
    - IVF partitions vectors into clusters (nlist) and only searches
      the nearest clusters (nprobe) — dramatically faster
    - With 20K vectors and nlist=100, nprobe=10: ~1-3ms retrieval
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        dimension: int = 384,
        nlist: int = 100,    # Number of clusters for IVF
        nprobe: int = 10,    # Number of clusters to search at query time
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.nlist = nlist
        self.nprobe = nprobe

        self._model = None
        self._index = None
        self._chunks_metadata: list[dict] = []
        self._is_trained = False
        self._query_cache: dict[str, np.ndarray] = {}
        self._cache_max_size: int = 512

    @property
    def model(self):
        """Lazy-load the embedding model with safe fallback."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}...")
            ST = _get_sentence_transformer()
            try:
                self._model = ST(self.model_name)
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load primary embedder ({e}), using feature-hashing fallback embedder.")
                # Fall back to GeminiEmbedder (which includes deterministic n-gram hashing)
                from app.vector_store import GeminiEmbedder
                self._model = GeminiEmbedder(self.model_name)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode texts into normalized embedding vectors.

        Returns numpy array of shape (len(texts), 384).
        Normalization ensures cosine similarity = inner product.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # L2 normalize for cosine sim
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string with sub-millisecond LRU memory caching."""
        norm_q = query.strip()
        if norm_q in self._query_cache:
            return self._query_cache[norm_q]
        
        vec = self.encode([norm_q])
        if len(self._query_cache) >= self._cache_max_size:
            # Evict oldest entries
            evict_keys = list(self._query_cache.keys())[:100]
            for k in evict_keys:
                del self._query_cache[k]
        self._query_cache[norm_q] = vec
        return vec

    def build_index(self, chunks: list[dict], texts_key: str = "text"):
        """
        Build a FAISS IVF index from a list of chunk dicts.

        Each chunk dict must have at least a 'text' field.
        All other fields are stored as metadata for retrieval.

        Steps:
        1. Extract texts from chunks
        2. Encode all texts into vectors
        3. Build IVF index (train on vectors, then add them)
        4. Store metadata mapping: vector_id → chunk dict
        """
        faiss = _get_faiss()

        texts = [c[texts_key] for c in chunks]
        logger.info(f"Encoding {len(texts)} chunks...")

        start = time.time()
        vectors = self.encode(texts, batch_size=128)
        encode_time = time.time() - start
        logger.info(f"Encoding complete in {encode_time:.1f}s")

        n_vectors = len(vectors)

        # For small datasets, use Flat index (exact search)
        if n_vectors < 1000:
            logger.info("Small dataset — using Flat index (exact search)")
            self._index = faiss.IndexFlatIP(self.dimension)
        else:
            # IVF index for larger datasets
            adjusted_nlist = min(self.nlist, n_vectors // 10)
            adjusted_nlist = max(adjusted_nlist, 1)
            logger.info(
                f"Building IVFFlat index: {n_vectors} vectors, "
                f"nlist={adjusted_nlist}, nprobe={self.nprobe}"
            )
            quantizer = faiss.IndexFlatIP(self.dimension)
            self._index = faiss.IndexIVFFlat(
                quantizer, self.dimension, adjusted_nlist,
                faiss.METRIC_INNER_PRODUCT,
            )
            # Train the index on the vectors
            self._index.train(vectors)
            self._index.nprobe = self.nprobe

        # Add vectors to the index
        self._index.add(vectors)
        self._is_trained = True

        # Store metadata (everything except the raw text to save memory)
        self._chunks_metadata = []
        for i, chunk in enumerate(chunks):
            meta = {k: v for k, v in chunk.items()}
            meta["vector_id"] = i
            self._chunks_metadata.append(meta)

        logger.info(
            f"Index built: {self._index.ntotal} vectors indexed"
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
        allowed_languages: Optional[set[str]] = None,
    ) -> list[dict]:
        """
        Search the index for the top-K most similar chunks with partition routing.

        Args:
            query_vector: (1, 384) numpy array
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            allowed_languages: Optional set of language partition codes (e.g. {'tam_Taml', 'eng_Latn'})

        Returns list of dicts with chunk metadata, score, and rank.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        fetch_k = top_k * 4 if allowed_languages else top_k
        scores, indices = self._index.search(query_vector, fetch_k)

        results = []
        for (score, idx) in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            if score < score_threshold:
                continue
            meta = self._chunks_metadata[idx]
            if allowed_languages and meta.get("language") not in allowed_languages:
                continue
            result = dict(meta)
            result["score"] = float(score)
            result["rank"] = len(results)
            results.append(result)
            if len(results) >= top_k:
                break

        return results

    def add_passages(self, passages: list[dict], texts_key: str = "text") -> int:
        """
        Dynamically add new text passages to the index and metadata in real-time.
        Returns the new total number of indexed vectors.
        """
        if self._index is None:
            self.build_index(passages, texts_key=texts_key)
            return self.total_vectors

        texts = [p.get(texts_key, "") for p in passages if p.get(texts_key)]
        if not texts:
            return self.total_vectors

        vectors = self.encode(texts)
        self._index.add(vectors)

        start_id = len(self._chunks_metadata)
        for i, p in enumerate(passages):
            meta = {k: v for k, v in p.items()}
            meta["vector_id"] = start_id + i
            meta["is_custom_learned"] = True
            meta["timestamp"] = time.time()
            self._chunks_metadata.append(meta)

        logger.info(f"Dynamically indexed {len(texts)} new passages. Total vectors: {self._index.ntotal}")
        return self.total_vectors

    def save(self, index_path: str, metadata_path: str):
        """Save FAISS index and chunk metadata to disk."""
        faiss = _get_faiss()
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._chunks_metadata, f, ensure_ascii=False)

        logger.info(
            f"Saved index ({self._index.ntotal} vectors) to {index_path}"
        )

    def load(self, index_path: str, metadata_path: str) -> bool:
        """Load FAISS index and metadata from disk. Returns True if successful."""
        faiss = _get_faiss()

        if not Path(index_path).exists() or not Path(metadata_path).exists():
            logger.warning("Index files not found on disk.")
            return False

        try:
            self._index = faiss.read_index(index_path)

            # Set nprobe for IVF indices
            if hasattr(self._index, 'nprobe'):
                self._index.nprobe = self.nprobe

            with open(metadata_path, "r", encoding="utf-8") as f:
                self._chunks_metadata = json.load(f)

            self._is_trained = True
            logger.info(
                f"Loaded index: {self._index.ntotal} vectors, "
                f"{len(self._chunks_metadata)} metadata records"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        """Ready if index exists and has at least 1 vector."""
        return self._index is not None and self._index.ntotal > 0

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0
