"""
Build 3-Language (English + Hindi + Tamil) Vector Index
Extracts ~45,000 - 50,000 deduplicated passages with rich metadata (Passage IDs, Query Types, Queries)
from local MSMARCO-XI parquets and builds a high-speed FAISS index using VectorStore.
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
import pyarrow.parquet as pq

# Fix stdout encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.vector_store import VectorStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HIN_PARQUET = DATA_DIR / "hintrain.parquet"
TAM_PARQUET = DATA_DIR / "tamtrain.parquet"

OUTPUT_INDEX = DATA_DIR / "faiss_index.bin"
OUTPUT_METADATA = DATA_DIR / "chunks_metadata.json"


def extract_3lang_passages() -> list[dict]:
    """
    Extracts balanced, deduplicated passages from English, Hindi, and Tamil.
    ~18k English, ~16k Hindi, ~14k Tamil
    """
    logger.info("Extracting 3-language passages from local parquets...")
    passages = []
    seen_hashes = set()

    # 1. Hindi Parquet (provides Hindi + English paired passages)
    if HIN_PARQUET.exists():
        logger.info(f"Reading Hindi parquet: {HIN_PARQUET.name}")
        pf = pq.ParquetFile(str(HIN_PARQUET))
        
        hin_count = 0
        eng_count = 0
        
        for batch in pf.iter_batches(batch_size=2000):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                qid = str(row.get("query_id", ""))
                qtype = str(row.get("query_type", "DESCRIPTION"))
                hi_q = str(row.get("query", ""))
                en_q = str(row.get("Eng_Query", ""))
                passages_dict = row.get("passages", {})
                
                # Hindi passages
                tr_passages = passages_dict.get("Translated_passages", []) if isinstance(passages_dict, dict) else []
                for p_text in tr_passages:
                    if p_text and len(p_text.strip()) > 30:
                        p_hash = hashlib.md5(p_text.strip().encode("utf-8")).hexdigest()[:16]
                        if p_hash not in seen_hashes and hin_count < 16000:
                            seen_hashes.add(p_hash)
                            passages.append({
                                "passage_id": p_hash,
                                "text": p_text.strip(),
                                "language": "hin_Deva",
                                "query": hi_q,
                                "query_type": qtype,
                                "query_id": qid,
                                "source": f"MSMARCO-XI/hin/{qtype.lower()}"
                            })
                            hin_count += 1
                
                # English passages
                en_passages = passages_dict.get("English_passages", []) if isinstance(passages_dict, dict) else []
                for p_text in en_passages:
                    if p_text and len(p_text.strip()) > 30:
                        p_hash = hashlib.md5(p_text.strip().encode("utf-8")).hexdigest()[:16]
                        if p_hash not in seen_hashes and eng_count < 18000:
                            seen_hashes.add(p_hash)
                            passages.append({
                                "passage_id": p_hash,
                                "text": p_text.strip(),
                                "language": "eng_Latn",
                                "query": en_q,
                                "query_type": qtype,
                                "query_id": qid,
                                "source": f"MSMARCO-XI/eng/{qtype.lower()}"
                            })
                            eng_count += 1
                            
            if hin_count >= 16000 and eng_count >= 18000:
                break
                
        logger.info(f"Loaded {hin_count} Hindi passages, {eng_count} English passages.")

    # 2. Tamil Parquet (provides Tamil passages)
    if TAM_PARQUET.exists():
        logger.info(f"Reading Tamil parquet: {TAM_PARQUET.name}")
        pf = pq.ParquetFile(str(TAM_PARQUET))
        
        tam_count = 0
        for batch in pf.iter_batches(batch_size=2000):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                qid = str(row.get("query_id", ""))
                qtype = str(row.get("query_type", "DESCRIPTION"))
                ta_q = str(row.get("query", ""))
                passages_dict = row.get("passages", {})
                
                tr_passages = passages_dict.get("Translated_passages", []) if isinstance(passages_dict, dict) else []
                for p_text in tr_passages:
                    if p_text and len(p_text.strip()) > 30:
                        p_hash = hashlib.md5(p_text.strip().encode("utf-8")).hexdigest()[:16]
                        if p_hash not in seen_hashes and tam_count < 14000:
                            seen_hashes.add(p_hash)
                            passages.append({
                                "passage_id": p_hash,
                                "text": p_text.strip(),
                                "language": "tam_Taml",
                                "query": ta_q,
                                "query_type": qtype,
                                "query_id": qid,
                                "source": f"MSMARCO-XI/tam/{qtype.lower()}"
                            })
                            tam_count += 1
                            
            if tam_count >= 14000:
                break
                
        logger.info(f"Loaded {tam_count} Tamil passages.")

    logger.info(f"Total deduplicated 3-language passages: {len(passages)}")
    return passages


if __name__ == "__main__":
    passages = extract_3lang_passages()
    
    vs = VectorStore(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    logger.info("Building FAISS index with VectorStore...")
    vs.build_index(passages, texts_key="text")
    
    logger.info(f"Saving FAISS index to {OUTPUT_INDEX}...")
    vs.save(str(OUTPUT_INDEX), str(OUTPUT_METADATA))
    logger.info("✅ 3-Language Index Build & Save Complete!")
