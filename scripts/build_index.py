"""
Build Index Script

One-time script to:
1. Download MSMARCO-XI dataset from HuggingFace
2. Extract passages for 4 Indic languages
3. Apply multi-strategy chunking
4. Encode chunks into vectors
5. Build FAISS IVF index
6. Save index + metadata to disk

Usage:
    python scripts/build_index.py

This takes ~10-30 minutes depending on network speed and CPU.
The output files (data/faiss_index.bin, data/chunks_metadata.json)
should be committed or deployed with the application.
"""

import sys
import os
import json
import logging
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.dataset_loader import load_dataset_from_huggingface, save_passages_to_disk, load_passages_from_disk
from app.chunking import ChunkEngine
from app.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Config
LANGUAGES = ["hin_Deva", "ben_Beng", "tam_Taml", "tel_Telu"]
MAX_PER_LANGUAGE = int(os.getenv("PASSAGES_PER_LANGUAGE", "5000"))
DATA_DIR = Path(__file__).parent.parent / "data"
PASSAGES_CACHE = str(DATA_DIR / "raw_passages.json")
INDEX_PATH = str(DATA_DIR / "faiss_index.bin")
METADATA_PATH = str(DATA_DIR / "chunks_metadata.json")


def main():
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("MSMARCO-XI RAG Index Builder")
    logger.info(f"Languages: {LANGUAGES}")
    logger.info(f"Max passages per language: {MAX_PER_LANGUAGE}")
    logger.info("=" * 60)

    # ── Step 1: Load Dataset ──
    passages = load_passages_from_disk(PASSAGES_CACHE)
    if passages is None:
        logger.info("\n📥 Downloading dataset from HuggingFace...")
        passages = load_dataset_from_huggingface(
            languages=LANGUAGES,
            max_per_language=MAX_PER_LANGUAGE,
        )
        save_passages_to_disk(passages, PASSAGES_CACHE)
    else:
        logger.info(f"📦 Using cached passages ({len(passages)} total)")

    if not passages:
        logger.error("No passages loaded! Check dataset access.")
        sys.exit(1)

    # ── Step 2: Chunk Passages ──
    logger.info(f"\n✂️ Chunking {len(passages)} passages with multi-strategy engine...")
    chunk_engine = ChunkEngine(
        sentences_per_chunk=3,
        sentence_overlap=1,
        fixed_chunk_size=256,
        fixed_overlap=64,
        paragraph_max_chars=512,
    )

    chunks = chunk_engine.chunk_passages(passages)
    stats = chunk_engine.get_strategy_stats(chunks)

    logger.info(f"\n📊 Chunking Statistics:")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    logger.info(f"  By strategy: {json.dumps(stats['by_strategy'], indent=2)}")
    logger.info(f"  Avg char count: {stats['avg_char_count']:.0f}")
    logger.info(f"  Avg token estimate: {stats['avg_token_estimate']:.0f}")

    # Convert chunks to dicts for vector store
    chunk_dicts = [c.to_dict() for c in chunks]

    # ── Step 3: Build Vector Index ──
    logger.info(f"\n🔨 Building FAISS index...")
    vector_store = VectorStore(
        model_name=os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
    )

    vector_store.build_index(chunk_dicts)

    # ── Step 4: Save to Disk ──
    logger.info(f"\n💾 Saving index to disk...")
    vector_store.save(INDEX_PATH, METADATA_PATH)

    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info(f"✅ Index build complete in {elapsed:.1f}s")
    logger.info(f"  Index: {INDEX_PATH}")
    logger.info(f"  Metadata: {METADATA_PATH}")
    logger.info(f"  Total vectors: {vector_store.total_vectors}")
    logger.info(f"{'=' * 60}")

    # ── Step 5: Quick Sanity Test ──
    logger.info("\n🧪 Running sanity test...")
    test_queries = [
        "What is the capital of India?",
        "भारत की राजधानी क्या है?",
        "இந்தியாவின் தலைநகரம் என்ன?",
    ]
    for query in test_queries:
        query_vec = vector_store.encode_query(query)
        results = vector_store.search(query_vec, top_k=3)
        logger.info(f"\n  Query: {query}")
        for r in results:
            logger.info(f"    [{r['score']:.3f}] {r['text'][:80]}...")


if __name__ == "__main__":
    main()
