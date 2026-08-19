import sys
import time
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")
from app.vector_store import VectorStore
from app.non_llm_synthesizer import NonLLMSynthesizer

def main():
    vs = VectorStore()
    vs.load("data/faiss_index.bin", "data/chunks_metadata.json")
    synth = NonLLMSynthesizer()

    queries = [
        "what direction does phloem flow",
        "what was the immediate impact of the success of the manhattan project?",
        "how to use sysdate in sql",
        "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?",
        "மன்ஹாட்டன் திட்டத்தின் வெற்றியின் உடனடி விளைவு என்ன?",
        "Who was the first governor of Uttar Pradesh?"
    ]

    print("=" * 80)
    print("NON-LLM CONTINUOUS TEXTRANK + SVD SYNTHESIS TEST:")
    print("=" * 80)

    for q in queries:
        t0 = time.perf_counter()
        q_vec = np.ascontiguousarray(vs.encode([q]), dtype=np.float32)
        results = vs.search(q_vec, top_k=3)
        res = synth.synthesize(q, q_vec, results, vs)
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000

        print(f"Q: \"{q}\"")
        print(f"⚡ End-to-End Latency: {total_ms:.2f} ms | Confidence: {res['confidence']} | Method: {res['method']}")
        print(f"Answer:\n{res['answer']}")
        print("-" * 80)

if __name__ == "__main__":
    main()
