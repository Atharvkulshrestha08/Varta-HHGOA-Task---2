"""
Benchmark Runner for Trial 1 (adapting scripts/benchmark.py)
"""

import sys
import os
import json
import time
import asyncio
import logging
from pathlib import Path
import numpy as np
import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
dotenv.load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.vector_store import VectorStore
from app.analytics import LatencyAnalytics
from app.guardrails import GuardrailsEngine
from app.harness import PipelineHarness, QueryRequest
from app.generator import GroqGenerator, MockGenerator
from app.stt import MockSTTClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TEST_QUERIES = [
    # English
    "What is the capital of India?",
    "How does photosynthesis work?",
    "What are the symptoms of malaria?",
    "Who invented the telephone?",
    "What is the population of Mumbai?",
    "How is steel manufactured?",
    "What causes earthquakes?",
    "What is the GDP of India?",
    "How does a computer processor work?",
    "What are the benefits of yoga?",
    # Hindi
    "भारत का सबसे बड़ा शहर कौन सा है?",
    "सूर्य ग्रहण कैसे होता है?",
    "मधुमेह के लक्षण क्या हैं?",
    "भारत में कितने राज्य हैं?",
    "हिमालय पर्वत कितना ऊंचा है?",
    # Tamil
    "இந்தியாவின் தலைநகரம் என்ன?",
    "பூமி எப்படி சுழல்கிறது?",
    # Cache Repeats
    "What is the capital of India?",
    "सूर्य ग्रहण कैसे होता है?",
    "இந்தியாவின் தலைநகரம் என்ன?",
]


async def run():
    print("=" * 65)
    print("🚀 Running Diagnostics 1: benchmark.py on Trial 1")
    print("=" * 65)

    base_data = Path(__file__).resolve().parent.parent.parent / "data"
    vs = VectorStore()
    vs.load(str(base_data / "faiss_index.bin"), str(base_data / "chunks_metadata.json"))
    _ = vs.model

    groq_key = os.getenv("GROQ_API_KEY", "")
    gen = GroqGenerator(api_key=groq_key, max_output_tokens=45) if groq_key else MockGenerator()
    if hasattr(gen, "prewarm"):
        await gen.prewarm()

    analytics = LatencyAnalytics(window_size=len(TEST_QUERIES))
    guardrails = GuardrailsEngine()
    harness = PipelineHarness(vs, MockSTTClient(), gen, analytics, guardrails)

    latencies = []
    cache_hits = 0

    for idx, q in enumerate(TEST_QUERIES, 1):
        t0 = time.perf_counter()
        res = await harness.process_text_query(QueryRequest(text=q, top_k=2))
        t1 = time.perf_counter()
        dur = (t1 - t0) * 1000.0
        latencies.append(dur)
        if res.pipeline_path == "cache_hit" or dur < 10.0:
            cache_hits += 1
        print(f"  {idx:02d}. [{res.pipeline_path:14s}] {dur:6.2f} ms | Query: {q[:30]}...")
        await asyncio.sleep(0.05)

    lat_arr = np.array(latencies)
    p50 = float(np.percentile(lat_arr, 50))
    p75 = float(np.percentile(lat_arr, 75))
    p90 = float(np.percentile(lat_arr, 90))
    p99 = float(np.percentile(lat_arr, 99))
    p100 = float(np.max(lat_arr))
    mean = float(np.mean(lat_arr))

    print("\n" + "=" * 65)
    print("📊 Diagnostics 1 (benchmark.py) Summary:")
    print("=" * 65)
    print(f"• Total Queries:      {len(TEST_QUERIES)}")
    print(f"• Cache Hits:         {cache_hits}/{len(TEST_QUERIES)}")
    print(f"• P50 Latency:        {p50:6.2f} ms  {'[PASS <200ms]' if p50 < 200 else '[FAIL]'}")
    print(f"• P75 Latency:        {p75:6.2f} ms  {'[PASS <200ms]' if p75 < 200 else '[FAIL]'}")
    print(f"• P90 Latency:        {p90:6.2f} ms  {'[PASS <200ms]' if p90 < 200 else '[FAIL]'}")
    print(f"• P99 Latency:        {p99:6.2f} ms  {'[PASS <200ms]' if p99 < 200 else '[FAIL]'}")
    print(f"• P100 Latency:       {p100:6.2f} ms  {'[PASS <200ms]' if p100 < 200 else '[FAIL]'}")
    print(f"• Mean Latency:       {mean:6.2f} ms")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(run())
