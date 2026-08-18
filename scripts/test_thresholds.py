import sys
import json
import os
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")
from app.vector_store import VectorStore

def main():
    vs = VectorStore()
    vs.load("data/faiss_index.bin", "data/chunks_metadata.json")

    queries = [
        ("Who voiced the Optimus Prime in Transformers Prime?", "UNINDEXED / OUT-OF-DOMAIN"),
        ("Who was the first governor of Goa?", "UNINDEXED / OUT-OF-DOMAIN"),
        ("Who was the first governor of Uttar Pradesh?", "UNINDEXED / OUT-OF-DOMAIN"),
        ("What is the oldest language of India?", "INDEXED IN DATASET"),
        ("இந்தியாவின் மிகப் பழமையான மொழி எது?", "INDEXED IN DATASET"),
        ("விக்ரம் திரைப்படம் இயக்குநர் யார்?", "UNINDEXED / OUT-OF-DOMAIN"),
    ]

    print("=" * 80)
    print("EMPIRICAL COSINE SIMILARITY SCORE DISTRIBUTION:")
    print("=" * 80)

    for q, category in queries:
        vec = np.ascontiguousarray(vs.encode([q]), dtype=np.float32)
        results = vs.search(vec, top_k=1)
        top_score = round(results[0]["score"], 3) if results else 0.0
        status = "REJECT (Refusal Fallback)" if top_score < 0.55 else "ACCEPT (Grounded Generation)"
        print(f"[{category}]")
        print(f'Query: "{q}"')
        print(f"Top Cosine Score: {top_score}")
        print(f"Action with 0.55 Threshold: {status}")
        print("-" * 80)

if __name__ == "__main__":
    main()
