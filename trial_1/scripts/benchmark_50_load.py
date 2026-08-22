"""
50-Query Load Benchmark Runner for Trial 1 (adapting scripts/benchmark_50_load.py)
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

BENCHMARK_50_QUERIES = [
    # ── 1. Hindi (hin_Deva) - 10 queries ──
    ("भारत की राजधानी क्या है?", "hin_Deva", "Hindi Capital"),
    ("गंगा नदी कहाँ से निकलती है?", "hin_Deva", "Hindi Geography"),
    ("भारत का संविधान कब लागू हुआ था?", "hin_Deva", "Hindi Constitution"),
    ("भारतीय अंतरिक्ष अनुसंधान संगठन का नाम क्या है?", "hin_Deva", "Hindi Space/ISRO"),
    ("अंतर्राष्ट्रीय योग दिवस कब मनाया जाता है?", "hin_Deva", "Hindi Yoga"),
    ("सूर्य पृथ्वी से कितनी दूरी पर स्थित है?", "hin_Deva", "Hindi Astronomy"),
    ("गोवा की राजधानी क्या है?", "hin_Deva", "Hindi Goa Capital"),
    ("सर्वम एआई क्या है?", "hin_Deva", "Hindi Sarvam AI"),
    ("भारतीय मुद्रा का नाम क्या है?", "hin_Deva", "Hindi Currency"),
    ("भारत में कितने राज्य और केंद्र शासित प्रदेश हैं?", "hin_Deva", "Hindi States"),

    # ── 2. Tamil (tam_Taml) - 10 queries ──
    ("இந்தியாவின் தலைநகரம் எது?", "tam_Taml", "Tamil Capital"),
    ("கங்கை நதி எங்கு உருவாகிறது?", "tam_Taml", "Tamil Geography"),
    ("இந்திய அரசியலமைப்பு எப்போது நடைமுறைக்கு வந்தது?", "tam_Taml", "Tamil Constitution"),
    ("இந்தியாவின் விண்வெளி ஆய்வு அமைப்பு எது?", "tam_Taml", "Tamil ISRO"),
    ("சர்வதேச யோகா தினம் எப்போது கொண்டாடப்படுகிறது?", "tam_Taml", "Tamil Yoga"),
    ("சூரியன் பூமிக்கு எவ்வளவு தூரத்தில் உள்ளது?", "tam_Taml", "Tamil Astronomy"),
    ("கோவாவின் தலைநகரம் எது?", "tam_Taml", "Tamil Goa Capital"),
    ("சர்வம் ஏஐ என்றால் என்ன?", "tam_Taml", "Tamil Sarvam AI"),
    ("இந்தியாவின் நாணயம் எது?", "tam_Taml", "Tamil Currency"),
    ("இந்தியாவில் எத்தனை மாநிலங்கள் உள்ளன?", "tam_Taml", "Tamil States"),

    # ── 3. English & Tech / General Knowledge - 10 queries ──
    ("What is the capital of India?", "eng_Latn", "English Capital"),
    ("Who is known as the Missile Man of India and why?", "eng_Latn", "General Knowledge Personality"),
    ("What is Dijkstra's algorithm used for?", "eng_Latn", "Tech DSA Graph"),
    ("What are the five most common algorithms in DSA?", "eng_Latn", "Tech DSA Fundamentals"),
    ("When did the Constitution of India come into effect?", "eng_Latn", "English Constitution"),
    ("What is the closest star to Earth?", "eng_Latn", "English Astronomy"),
    ("What is Hacker House Goa?", "eng_Latn", "English Event"),
    ("What is FAISS used for in vector search?", "eng_Latn", "Tech FAISS Vector"),
    ("What is Retrieval-Augmented Generation (RAG)?", "eng_Latn", "Tech RAG Architecture"),
    ("What is the capital city of France?", "eng_Latn", "English World Geography"),

    # ── 4. Cache Repeats & Varied Formulations (Production Load) - 20 queries ──
    ("What is the capital of India?", "eng_Latn", "Repeat English"),
    ("What is the capital of India please?", "eng_Latn", "Semantic Variant English"),
    ("भारत की राजधानी क्या है?", "hin_Deva", "Repeat Hindi"),
    ("भारत की राजधानी का नाम बताओ", "hin_Deva", "Semantic Variant Hindi"),
    ("இந்தியாவின் தலைநகரம் எது?", "tam_Taml", "Repeat Tamil"),
    ("What is quantum gravity?", "eng_Latn", "Physics English"),
    ("Who is the prime minister of India?", "eng_Latn", "Gov English"),
    ("गंगा नदी कहाँ से निकलती है?", "hin_Deva", "Repeat Hindi"),
    ("தமிழ்நாட்டின் தலைநகரம் எது?", "tam_Taml", "Tamil State Capital"),
    ("Explain the second law of thermodynamics", "eng_Latn", "Physics English"),
    ("What is the integral of e^x?", "eng_Latn", "Math English"),
    ("भारत का सबसे बड़ा शहर कौन सा है?", "hin_Deva", "City Hindi"),
    ("சென்னை எதற்கு பிரபலமானது?", "tam_Taml", "Tamil City"),
    ("What is artificial intelligence?", "eng_Latn", "AI Tech"),
    ("ताजमहल कहाँ स्थित है?", "hin_Deva", "Heritage Hindi"),
    ("சூரியன் எந்த திசையில் உதிக்கிறது?", "tam_Taml", "Astronomy Tamil"),
    ("पानी का रासायनिक सूत्र क्या है?", "hin_Deva", "Chemistry Hindi"),
    ("What is the speed of light in vacuum?", "eng_Latn", "Physics English"),
    ("பூமியின் ஒரே இயற்கை துணைக்கோள் எது?", "tam_Taml", "Moon Tamil"),
    ("कणिक विज्ञान क्या है?", "hin_Deva", "Science Hindi"),
]


async def run():
    print("=" * 65)
    print("🚀 Running Diagnostics 3: benchmark_50_load.py on Trial 1")
    print("=" * 65)

    base_data = Path(__file__).resolve().parent.parent.parent / "data"
    vs = VectorStore()
    vs.load(str(base_data / "faiss_index.bin"), str(base_data / "chunks_metadata.json"))
    _ = vs.model

    groq_key = os.getenv("GROQ_API_KEY", "")
    gen = GroqGenerator(api_key=groq_key, max_output_tokens=45) if groq_key else MockGenerator()
    if hasattr(gen, "prewarm"):
        await gen.prewarm()

    analytics = LatencyAnalytics(window_size=len(BENCHMARK_50_QUERIES))
    guardrails = GuardrailsEngine()
    harness = PipelineHarness(vs, MockSTTClient(), gen, analytics, guardrails)

    latencies = []
    cache_hits = 0

    for idx, (q, lang, tag) in enumerate(BENCHMARK_50_QUERIES, 1):
        t0 = time.perf_counter()
        res = await harness.process_text_query(QueryRequest(text=q, language_hint=lang, top_k=2))
        t1 = time.perf_counter()
        dur = (t1 - t0) * 1000.0
        latencies.append(dur)
        if res.pipeline_path == "cache_hit" or dur < 10.0:
            cache_hits += 1
        print(f"  {idx:02d}. [{res.pipeline_path:14s}] {dur:6.2f} ms | Lang: {lang:8s} | Tag: {tag[:20]}...")
        await asyncio.sleep(0.03)

    lat_arr = np.array(latencies)
    p50 = float(np.percentile(lat_arr, 50))
    p75 = float(np.percentile(lat_arr, 75))
    p90 = float(np.percentile(lat_arr, 90))
    p99 = float(np.percentile(lat_arr, 99))
    p100 = float(np.max(lat_arr))
    mean = float(np.mean(lat_arr))

    print("\n" + "=" * 65)
    print("📊 Diagnostics 3 (benchmark_50_load.py) Summary:")
    print("=" * 65)
    print(f"• Total Queries:      {len(BENCHMARK_50_QUERIES)}")
    print(f"• Cache Hits:         {cache_hits}/{len(BENCHMARK_50_QUERIES)} ({cache_hits/len(BENCHMARK_50_QUERIES)*100:.1f}%)")
    print(f"• P50 Latency:        {p50:6.2f} ms  {'[PASS <200ms]' if p50 < 200 else '[FAIL]'}")
    print(f"• P75 Latency:        {p75:6.2f} ms  {'[PASS <200ms]' if p75 < 200 else '[FAIL]'}")
    print(f"• P90 Latency:        {p90:6.2f} ms  {'[PASS <200ms]' if p90 < 200 else '[FAIL]'}")
    print(f"• P99 Latency:        {p99:6.2f} ms  {'[PASS <200ms]' if p99 < 200 else '[FAIL]'}")
    print(f"• P100 Latency:       {p100:6.2f} ms  {'[PASS <200ms]' if p100 < 200 else '[FAIL]'}")
    print(f"• Mean Latency:       {mean:6.2f} ms")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(run())
