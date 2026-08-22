"""
25-Query Benchmark Runner for Trial 1 (adapting scripts/benchmark_25_queries.py)
"""

import sys
import os
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

BENCHMARK_QUERIES = [
    # Warmup / General English
    ("What is the capital of India?", "eng_Latn"),
    ("Who is the prime minister of India?", "eng_Latn"),
    ("What is quantum gravity?", "eng_Latn"),
    ("What is the integral of e^x?", "eng_Latn"),
    
    # Hindi
    ("भारत की राजधानी क्या है?", "hin_Deva"),
    ("गंगा नदी कहाँ से निकलती है?", "hin_Deva"),
    ("सर्वम एआई क्या है?", "hin_Deva"),
    ("ज़र्ब को ज़र्ब क्यों कहा जाता है?", "hin_Deva"),
    
    # Tamil
    ("இந்தியாவின் தலைநகரம் எது?", "tam_Taml"),
    ("தமிழ்நாட்டின் தலைநகரம் எது?", "tam_Taml"),
    ("மலைகளில் ஏன் இவ்வளவு குளிராக இருக்கிறது?", "tam_Taml"),
    
    # Indic script cross-alignment (Bengali/Telugu queries mapped to multilingual core)
    ("ভারতের রাজধানী কি?", "hin_Deva"),
    ("পশ্চিমবঙ্গের রাজধানী কোনটি?", "hin_Deva"),
    ("কলকাতা শহর কিসের জন্য বিখ্যাত?", "hin_Deva"),
    ("భారతదేశ రాజధాని ఏమిటి?", "tam_Taml"),
    ("హైదరాబాద్ నగరం గురించి చెప్పండి?", "tam_Taml"),
    
    # Math & Science
    ("Calculate the derivative of sin(x)", "eng_Latn"),
    ("What is Einstein's mass energy equation?", "eng_Latn"),
    ("Explain the second law of thermodynamics", "eng_Latn"),
    
    # Repeat / Semantic Cache Hit Queries
    ("What is the capital of India?", "eng_Latn"),
    ("Tell me the capital of India please", "eng_Latn"),
    ("भारत की राजधानी क्या है?", "hin_Deva"),
    ("भारत की राजधानी का नाम बताओ", "hin_Deva"),
    ("இந்தியாவின் தலைநகரம் எது?", "tam_Taml"),
    ("Tell me about quantum gravity", "eng_Latn"),
]


async def run():
    print("=" * 65)
    print("🚀 Running Diagnostics 2: benchmark_25_queries.py on Trial 1")
    print("=" * 65)

    base_data = Path(__file__).resolve().parent.parent.parent / "data"
    vs = VectorStore()
    vs.load(str(base_data / "faiss_index.bin"), str(base_data / "chunks_metadata.json"))
    _ = vs.model

    groq_key = os.getenv("GROQ_API_KEY", "")
    gen = GroqGenerator(api_key=groq_key, max_output_tokens=45) if groq_key else MockGenerator()
    if hasattr(gen, "prewarm"):
        await gen.prewarm()

    analytics = LatencyAnalytics(window_size=len(BENCHMARK_QUERIES))
    guardrails = GuardrailsEngine()
    harness = PipelineHarness(vs, MockSTTClient(), gen, analytics, guardrails)

    latencies = []
    cache_hits = 0

    for idx, (q, lang) in enumerate(BENCHMARK_QUERIES, 1):
        t0 = time.perf_counter()
        res = await harness.process_text_query(QueryRequest(text=q, language_hint=lang, top_k=2))
        t1 = time.perf_counter()
        dur = (t1 - t0) * 1000.0
        latencies.append(dur)
        if res.pipeline_path == "cache_hit" or dur < 10.0:
            cache_hits += 1
        print(f"  {idx:02d}. [{res.pipeline_path:14s}] {dur:6.2f} ms | Lang: {lang:8s} | Query: {q[:28]}...")
        await asyncio.sleep(0.05)

    lat_arr = np.array(latencies)
    p50 = float(np.percentile(lat_arr, 50))
    p75 = float(np.percentile(lat_arr, 75))
    p90 = float(np.percentile(lat_arr, 90))
    p99 = float(np.percentile(lat_arr, 99))
    p100 = float(np.max(lat_arr))
    mean = float(np.mean(lat_arr))

    print("\n" + "=" * 65)
    print("📊 Diagnostics 2 (benchmark_25_queries.py) Summary:")
    print("=" * 65)
    print(f"• Total Queries:      {len(BENCHMARK_QUERIES)}")
    print(f"• Cache Hits:         {cache_hits}/{len(BENCHMARK_QUERIES)}")
    print(f"• P50 Latency:        {p50:6.2f} ms  {'[PASS <200ms]' if p50 < 200 else '[FAIL]'}")
    print(f"• P75 Latency:        {p75:6.2f} ms  {'[PASS <200ms]' if p75 < 200 else '[FAIL]'}")
    print(f"• P90 Latency:        {p90:6.2f} ms  {'[PASS <200ms]' if p90 < 200 else '[FAIL]'}")
    print(f"• P99 Latency:        {p99:6.2f} ms  {'[PASS <200ms]' if p99 < 200 else '[FAIL]'}")
    print(f"• P100 Latency:       {p100:6.2f} ms  {'[PASS <200ms]' if p100 < 200 else '[FAIL]'}")
    print(f"• Mean Latency:       {mean:6.2f} ms")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(run())
