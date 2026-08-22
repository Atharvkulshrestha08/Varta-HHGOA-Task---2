"""
Comprehensive Benchmark Runner — Trial 1 (Sub-200ms Across All Percentiles)

Evaluates:
- P50, P75, P90, P99, P100 latencies
- English, Hindi, and Tamil queries
- Target: < 200ms in ALL cases
"""

import asyncio
import os
import sys
import time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

# Fix stdout encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add trial_1 to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env from trial_1
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

from app.analytics import LatencyAnalytics
from app.guardrails import GuardrailsEngine
from app.harness import PipelineHarness, QueryRequest
from app.generator import GroqGenerator, MockGenerator
from app.stt import MockSTTClient
from app.vector_store import VectorStore

BENCHMARK_QUERIES = [
    # English (eng_Latn)
    ("What is the capital of India?", "eng_Latn"),
    ("Who is the prime minister of India?", "eng_Latn"),
    ("What is quantum gravity?", "eng_Latn"),
    ("Explain the second law of thermodynamics.", "eng_Latn"),
    ("What is the integral of e^x?", "eng_Latn"),
    ("What is the speed of light in vacuum?", "eng_Latn"),
    ("What is FAISS used for?", "eng_Latn"),
    ("What is artificial intelligence?", "eng_Latn"),
    
    # Hindi (hin_Deva)
    ("भारत की राजधानी क्या है?", "hin_Deva"),
    ("गंगा नदी कहाँ से निकलती है?", "hin_Deva"),
    ("भारत के प्रधानमंत्री कौन हैं?", "hin_Deva"),
    ("सर्वम एआई क्या है?", "hin_Deva"),
    ("सौर मंडल में कितने ग्रह हैं?", "hin_Deva"),
    ("पानी का रासायनिक सूत्र क्या है?", "hin_Deva"),
    ("ताजमहल कहाँ स्थित है?", "hin_Deva"),
    
    # Tamil (tam_Taml)
    ("இந்தியாவின் தலைநகரம் எது?", "tam_Taml"),
    ("தமிழ்நாட்டின் தலைநகரம் எது?", "tam_Taml"),
    ("சென்னை எதற்கு பிரபலமானது?", "tam_Taml"),
    ("சூரியன் எந்த திசையில் உதிக்கிறது?", "tam_Taml"),
    ("பூமியின் ஒரே இயற்கை துணைக்கோள் எது?", "tam_Taml"),
    ("கணிதத்தின் தந்தை யார்?", "tam_Taml"),
    
    # Repeat / Semantic Cache Hit Queries (Production Simulation)
    ("What is the capital of India?", "eng_Latn"),
    ("What is the capital of India please?", "eng_Latn"),
    ("भारत की राजधानी क्या है?", "hin_Deva"),
    ("भारत की राजधानी का नाम बताओ", "hin_Deva"),
    ("இந்தியாவின் தலைநகரம் எது?", "tam_Taml"),
    ("தமிழ்நாட்டின் தலைநகரம் எது?", "tam_Taml"),
    ("What is quantum gravity?", "eng_Latn"),
]


async def run_benchmark():
    print("=" * 70)
    print("🚀 Initializing Trial 1 Sub-200ms Pipeline...")
    print("=" * 70)

    # 1. Initialize Vector Store
    vs = VectorStore()
    base_data = Path(__file__).resolve().parent.parent.parent / "data"
    idx_path = str(base_data / "faiss_index.bin")
    meta_path = str(base_data / "chunks_metadata.json")
    vs.load(idx_path, meta_path)
    _ = vs.model

    # 2. Generator
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        gen = GroqGenerator(api_key=groq_key, model_name="allam-2-7b", max_output_tokens=35)
        print("🟢 Engine: Groq LPU (allam-2-7b @ ~1000 tok/s)")
        await gen.prewarm()
    else:
        gen = MockGenerator()
        print("⚪ Engine: Mock")

    analytics = LatencyAnalytics(window_size=1000)
    guardrails = GuardrailsEngine()
    stt = MockSTTClient()

    harness = PipelineHarness(
        vector_store=vs,
        stt_client=stt,
        generator=gen,
        analytics=analytics,
        guardrails=guardrails,
    )

    print("\n⚡ Running Benchmark Queries (Target: < 200ms across ALL percentiles)...")
    latencies = []
    cache_latencies = []
    cold_latencies = []

    for idx, (query, lang) in enumerate(BENCHMARK_QUERIES, 1):
        req = QueryRequest(text=query, language_hint=lang, top_k=2)
        t0 = time.perf_counter()
        resp = await harness.process_text_query(req)
        t1 = time.perf_counter()
        dur_ms = (t1 - t0) * 1000.0

        is_cache = resp.pipeline_path == "cache_hit" or dur_ms < 10.0
        if is_cache:
            cache_latencies.append(dur_ms)
        else:
            cold_latencies.append(dur_ms)

        latencies.append(dur_ms)
        status_lbl = "⚡ [CACHE HIT]" if is_cache else "🟢 [COLD RAG] "
        print(f"  {idx:02d}. {status_lbl} {dur_ms:6.2f} ms | Lang: {lang:8s} | Query: {query[:32]}...")
        if not is_cache:
            await asyncio.sleep(0.1)

    lat_arr = np.array(latencies)
    p50 = float(np.percentile(lat_arr, 50))
    p75 = float(np.percentile(lat_arr, 75))
    p90 = float(np.percentile(lat_arr, 90))
    p99 = float(np.percentile(lat_arr, 99))
    p100 = float(np.max(lat_arr))
    mean_lat = float(np.mean(lat_arr))

    print("\n" + "=" * 70)
    print("📊 Trial 1 Latency Performance Report (Measured across queries):")
    print("=" * 70)
    
    all_pass = p100 < 200.0
    status_str = "✅ [ALL METRICS UNDER 200MS]" if all_pass else "⚠️ [SOME METRICS > 200MS]"
    print(f"Overall Status:          {status_str}")
    print(f"• P50 Latency (Median):  {p50:6.2f} ms   {'[PASS <200ms]' if p50 < 200 else '[FAIL]'}")
    print(f"• P75 Latency:           {p75:6.2f} ms   {'[PASS <200ms]' if p75 < 200 else '[FAIL]'}")
    print(f"• P90 Latency:           {p90:6.2f} ms   {'[PASS <200ms]' if p90 < 200 else '[FAIL]'}")
    print(f"• P99 Latency:           {p99:6.2f} ms   {'[PASS <200ms]' if p99 < 200 else '[FAIL]'}")
    print(f"• P100 Latency (Worst):  {p100:6.2f} ms   {'[PASS <200ms]' if p100 < 200 else '[FAIL]'}")
    print(f"• Mean (Average):        {mean_lat:6.2f} ms")
    if cache_latencies:
        print(f"• Avg Cache Hit Latency: {np.mean(cache_latencies):6.2f} ms")
    if cold_latencies:
        print(f"• Avg Cold RAG Latency:  {np.mean(cold_latencies):6.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
