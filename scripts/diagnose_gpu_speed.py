import time
import sys
import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")
from app.vector_store import VectorStore

def main():
    print("=" * 80)
    print("GPU EMBEDDING & PIPELINE LATENCY DIAGNOSTIC")
    print("=" * 80)

    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")
        print(f"VRAM Allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")

    vs = VectorStore()
    vs.load("data/faiss_index.bin", "data/chunks_metadata.json")

    test_queries = [
        "Warmup Query 1",
        "Warmup Query 2",
        "What is the capital of India?",
        "Explain partial integration in calculus",
        "सौरमंडल में कितने ग्रह हैं?",
        "இந்தியாவின் மிகப் பழமையான மொழி எது?"
    ]

    print("\n⏱️ SINGLE-QUERY EMBEDDING LATENCY MEASUREMENTS:")
    for i, q in enumerate(test_queries):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        vec = vs.encode([q])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        print(f"  [{i+1}] {ms:6.2f} ms | Query: \"{q}\"")

if __name__ == "__main__":
    main()
