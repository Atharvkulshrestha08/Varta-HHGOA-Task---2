"""Measure end-to-end vector retrieval latency (embed + FAISS search) against the
200ms budget defined for VartaLaap.

Usage:
    python benchmark.py [n_queries]
"""
import statistics
import sys
import time
from dataclasses import dataclass
from app.vector_store import VectorStore

LATENCY_BUDGET_MS = 200.0

QUERIES = [
    # English Queries
    "What is the capital of India?",
    "Who is the prime minister of India?",
    "What is quantum gravity?",
    "What is the integral of e^x?",
    "Calculate the derivative of sin(x)...",
    "What is Einstein's mass energy equation?",
    "Explain the second law of thermodynamics...",
    # Hindi (hin_Deva)
    "भारत की राजधानी क्या है?",
    "गंगा नदी कहाँ से निकलती है?",
    "सर्वम एआई क्या है?",
    "ज़र्ब को ज़र्ब क्यों कहा जाता है?",
    "भारत की राजधानी का नाम बताओ...",
    # Tamil (tam_Taml)
    "இந்தியாவின் தலைநகரம் எது?",
    "தமிழ்நாட்டின் தலைநகரம் எது?",
    "மலைகளில் ஏன் இவ்வளவு குளிராக இருக்கிறது?",
    # Bengali (ben_Beng)
    "ভারতের রাজধানী কি?",
    "পশ্চিমবঙ্গের রাজধানী কোনটি?",
    "কলকাতা শহর কিসের জন্য বিখ্যাত?",
    # Telugu (tel_Telu)
    "భారతదేశ రాజధాని ఏమిటి?",
    "హైదరాబాద్ నగరం గురించి చెప్పండి?",
    # RAG Architecture & AI Queries
    "What is FAISS used for?",
    "How does HNSW indexing work?",
    "Which embedding model is fast on CPU?",
    "How do you reduce RAG latency?",
    "What are the stages of a RAG pipeline?",
]

# Singleton VectorStore instance
_vs: VectorStore = None


def warmup():
    global _vs
    if _vs is None:
        _vs = VectorStore()
        _vs.load("data/faiss_index.bin", "data/chunks_metadata.json")
        _ = _vs.model.encode(["warmup query"])


@dataclass
class RetrievalBenchmarkResponse:
    embed_ms: float
    search_ms: float
    total_ms: float
    passages: list


def search(query: str, top_k: int = 5) -> RetrievalBenchmarkResponse:
    global _vs
    if _vs is None:
        warmup()

    # 1. Measure Embedding Latency
    t0 = time.perf_counter()
    query_vector = _vs.model.encode([query])
    t1 = time.perf_counter()
    embed_ms = (t1 - t0) * 1000.0

    # 2. Measure FAISS Search Latency
    t2 = time.perf_counter()
    passages = _vs.search(query_vector, top_k=top_k)
    t3 = time.perf_counter()
    search_ms = (t3 - t2) * 1000.0

    total_ms = embed_ms + search_ms
    return RetrievalBenchmarkResponse(
        embed_ms=embed_ms,
        search_ms=search_ms,
        total_ms=total_ms,
        passages=passages,
    )


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print("Warming up (model load + first inference)...")
    warmup()

    total_ms, embed_ms, search_ms = [], [], []
    for i in range(n):
        query = QUERIES[i % len(QUERIES)]
        resp = search(query, top_k=5)
        total_ms.append(resp.total_ms)
        embed_ms.append(resp.embed_ms)
        search_ms.append(resp.search_ms)

    print(f"\nRan {n} queries\n")
    print(f"{'stage':<12}{'avg':>8}{'p50':>8}{'p95':>8}{'p99':>8}   (ms)")
    for name, values in [("embed", embed_ms), ("search", search_ms), ("total", total_ms)]:
        print(
            f"{name:<12}"
            f"{statistics.mean(values):>8.2f}"
            f"{percentile(values, 50):>8.2f}"
            f"{percentile(values, 95):>8.2f}"
            f"{percentile(values, 99):>8.2f}"
        )

    p95_total = percentile(total_ms, 95)
    print(f"\nLatency budget: {LATENCY_BUDGET_MS}ms | p95 total: {p95_total:.2f}ms")
    if p95_total <= LATENCY_BUDGET_MS:
        print("PASS: within budget")
    else:
        print("FAIL: over budget -- see README 'Tuning latency' section")
        sys.exit(1)


if __name__ == "__main__":
    main()
