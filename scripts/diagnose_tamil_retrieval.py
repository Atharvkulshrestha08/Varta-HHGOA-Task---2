import sys
import json
import os
import torch
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")
from app.vector_store import VectorStore
from app.harness import detect_language

def main():
    vs = VectorStore()
    vs.load("data/faiss_index.bin", "data/chunks_metadata.json")

    test_queries = [
        "MS Dhoni cricket career record என்ன?",
        "IPL 2024 final match winner யார்?",
        "CSK team captain யார்?",
        "இந்தியாவின் மிகப் பழமையான மொழி எது?",
        "What is the oldest language of India?",
        "விக்ரம் திரைப்படம் இயக்குநர் யார்?"
    ]

    print("=" * 80)
    print("TAMIL CODE-MIXED RETRIEVAL LOGS & PARTITION ANALYSIS:")
    print("=" * 80)

    for q in test_queries:
        det_lang = detect_language(q)
        has_tamil = any(0x0B80 <= ord(c) <= 0x0BFF for c in q)
        has_latin = any(0x0041 <= ord(c) <= 0x005A or 0x0061 <= ord(c) <= 0x007A for c in q)
        composition = f"Tamil_Script={has_tamil}, Latin_Script={has_latin} (Code-Mixed: {has_tamil and has_latin})"

        q_vec = np.ascontiguousarray(vs.encode([q]), dtype=np.float32)
        results = vs.search(q_vec, top_k=3)

        print(f'QUERY: "{q}"')
        print(f"DETECTED LANG: {det_lang} | COMPOSITION: {composition}")
        print("TOP-3 RETRIEVED PASSAGES:")
        for i, r in enumerate(results):
            text_snippet = r["text"][:140].replace("\n", " ")
            score = round(r["score"], 3)
            partition = r.get("language", "N/A")
            p_id = r.get("passage_id", "N/A")
            print(f'  [{i+1}] Score: {score} | Partition: {partition} | DocID: {p_id}')
            print(f'      Text: "{text_snippet}..."')
        print("-" * 80)

if __name__ == "__main__":
    main()
