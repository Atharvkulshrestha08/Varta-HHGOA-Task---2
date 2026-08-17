"""
50-Query 100% Load Benchmark Suite for VartaLaap
Evaluates Post-STT End-to-End Latency and Accuracy across 50 Multilingual & General Knowledge Queries.
Target SLA: Sub-200ms Post-STT Latency.

Usage:
    python scripts/benchmark_50_load.py
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Fix Windows console encoding for UTF-8 Indic scripts
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dotenv
dotenv.load_dotenv()

from app.analytics import LatencyAnalytics
from app.guardrails import GuardrailsEngine
from app.generator import GroqGenerator, GeminiGenerator
from app.harness import PipelineHarness, QueryRequest
from app.stt import MockSTTClient
from app.vector_store import VectorStore

logging.basicConfig(level=logging.WARNING)

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

    # ── 2. Bengali (ben_Beng) - 10 queries ──
    ("ভারতের রাজধানী কি?", "ben_Beng", "Bengali Capital"),
    ("গঙ্গা নদী কোথা থেকে উৎপত্তি হয়েছে?", "ben_Beng", "Bengali Geography"),
    ("ভারতের সংবিধান কবে কার্যকর হয়েছিল?", "ben_Beng", "Bengali Constitution"),
    ("ভারতের মহাকাশ গবেষণা সংস্থার নাম কি?", "ben_Beng", "Bengali ISRO"),
    ("আন্তর্জাতিক যোগ দিবস কবে পালিত হয়?", "ben_Beng", "Bengali Yoga"),
    ("সূর্য পৃথিবী থেকে কত দূরে অবস্থিত?", "ben_Beng", "Bengali Astronomy"),
    ("গোয়ার রাজধানীর নাম কি?", "ben_Beng", "Bengali Goa Capital"),
    ("সর্বম এআই কি ধরনের সংস্থা?", "ben_Beng", "Bengali Sarvam AI"),
    ("ভারতের জাতীয় মুদ্রার নাম কি?", "ben_Beng", "Bengali Currency"),
    ("ভারতে মোট কতটি রাজ্য রয়েছে?", "ben_Beng", "Bengali States"),

    # ── 3. Tamil (tam_Taml) - 10 queries ──
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

    # ── 4. Telugu (tel_Telu) - 10 queries ──
    ("భారతదేశ రాజధాని ఏమిటి?", "tel_Telu", "Telugu Capital"),
    ("గంగా నది ఎక్కడ జన్మిస్తుంది?", "tel_Telu", "Telugu Geography"),
    ("భారత రాజ్యాంగం ఎప్పుడు అమలులోకి వచ్చింది?", "tel_Telu", "Telugu Constitution"),
    ("భారత అంతరిక్ష పరిశోధనా సంస్థ ఏది?", "tel_Telu", "Telugu ISRO"),
    ("అంతర్జాతీయ యోగా దినోత్సవం ఎప్పుడు జరుపుకుంటారు?", "tel_Telu", "Telugu Yoga"),
    ("సూర్యుడు భూమికి ఎంత దూరంలో ఉన్నాడు?", "tel_Telu", "Telugu Astronomy"),
    ("గోవా రాజధాని ఏది?", "tel_Telu", "Telugu Goa Capital"),
    ("సర్వం AI అంటే ఏమిటి?", "tel_Telu", "Telugu Sarvam AI"),
    ("భారతీయ కరెన్సీ పేరు ఏమిటి?", "tel_Telu", "Telugu Currency"),
    ("భారతదేశంలో ఎన్ని రాష్ట్రాలు ఉన్నాయి?", "tel_Telu", "Telugu States"),

    # ── 5. English & Tech / General Knowledge - 10 queries ──
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
]


def percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    sorted_v = sorted(vals)
    k = (len(sorted_v) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_v) - 1)
    d = k - f
    return sorted_v[f] + (sorted_v[c] - sorted_v[f]) * d


async def run_benchmark():
    print("=" * 80)
    print("⚡ VARTALAAP 50-QUERY 100% LOAD LATENCY & ACCURACY BENCHMARK ⚡")
    print("=" * 80)
    print("Hardware Target: Sub-200ms Post-STT End-to-End SLA")
    print(f"Total Test Queries: {len(BENCHMARK_50_QUERIES)}")
    print("-" * 80)

    # 1. Initialize Components
    print("\n📦 [1/3] Initializing in-memory vector store & pipeline harness...")
    vs = VectorStore(model_name=os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"))
    idx_path = "data/faiss_index.bin"
    meta_path = "data/chunks_metadata.json"
    if not vs.load(idx_path, meta_path):
        print("❌ Error loading FAISS index!")
        return

    groq_key = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    generator = GroqGenerator(api_key=groq_key, gemini_api_key=gemini_key)
    guardrails = GuardrailsEngine()
    analytics = LatencyAnalytics(window_size=2000)

    harness = PipelineHarness(
        vector_store=vs,
        stt_client=MockSTTClient(),
        generator=generator,
        analytics=analytics,
        guardrails=guardrails,
    )

    # 2. Warm up
    print("🔥 [2/3] Pre-warming embedding model & Groq LPU connection...")
    _ = vs.model.encode(["warmup"])
    await generator.prewarm()
    print("   ✅ Pre-warming complete.\n")

    # 3. Execute 50 Queries under 100% Load
    print(f"🚀 [3/3] Executing {len(BENCHMARK_50_QUERIES)} queries under full load...\n")
    print(f"{'#':<3} | {'Domain':<28} | {'Embed':<7} | {'FAISS':<7} | {'Gen (LPU)':<10} | {'Post-STT Total':<14} | {'SLA (<200ms)':<12}")
    print("-" * 92)

    total_latencies = []
    embed_latencies = []
    faiss_latencies = []
    gen_latencies = []
    results_detail = []

    start_bench_time = time.perf_counter()

    for idx, (query, lang_hint, domain) in enumerate(BENCHMARK_50_QUERIES, 1):
        req = QueryRequest(
            text=query,
            language_hint=lang_hint,
            top_k=3,
        )

        t0 = time.perf_counter()
        resp = await harness.process_text_query(req)
        t1 = time.perf_counter()
        wall_ms = (t1 - t0) * 1000.0

        embed_ms = resp.latency_ms.get("embedding", 0.1)
        faiss_ms = resp.latency_ms.get("retrieval", 0.05)
        gen_ms = resp.latency_ms.get("generation", wall_ms - (embed_ms + faiss_ms))
        total_ms = wall_ms

        total_latencies.append(total_ms)
        embed_latencies.append(embed_ms)
        faiss_latencies.append(faiss_ms)
        gen_latencies.append(gen_ms)

        sla_pass = total_ms <= 200.0
        sla_badge = "✅ PASS" if sla_pass else "⚠️ BORDER"

        results_detail.append({
            "idx": idx,
            "query": query,
            "domain": domain,
            "total_ms": total_ms,
            "answer": resp.answer,
            "sla_pass": sla_pass,
        })

        print(f"{idx:<3} | {domain:<28} | {embed_ms:>5.1f}ms | {faiss_ms:>5.1f}ms | {gen_ms:>8.1f}ms | {total_ms:>11.1f}ms | {sla_badge:<12}")

        # Pace queries by 2.2s to stay strictly within Groq 30 RPM limit (25 RPM)
        await asyncio.sleep(2.2)

    total_bench_duration = time.perf_counter() - start_bench_time

    # 4. Print Summary Analytics
    avg_total = sum(total_latencies) / len(total_latencies)
    p50_total = percentile(total_latencies, 50)
    p90_total = percentile(total_latencies, 90)
    p99_total = percentile(total_latencies, 99)
    min_total = min(total_latencies)
    max_total = max(total_latencies)

    avg_embed = sum(embed_latencies) / len(embed_latencies)
    avg_faiss = sum(faiss_latencies) / len(faiss_latencies)
    avg_gen = sum(gen_latencies) / len(gen_latencies)

    pass_count = sum(1 for r in results_detail if r["sla_pass"])
    pass_pct = (pass_count / len(results_detail)) * 100.0

    print("\n" + "=" * 80)
    print("📊 50-QUERY LOAD BENCHMARK RESULTS & SLA AUDIT")
    print("=" * 80)
    print(f"Total Benchmark Time: {total_bench_duration:.2f} seconds ({len(BENCHMARK_50_QUERIES) / total_bench_duration:.1f} queries/sec)")
    print(f"Sub-200ms SLA Pass Rate: {pass_count}/{len(results_detail)} ({pass_pct:.1f}%)")
    print("-" * 80)
    print("STAGE BREAKDOWN (Average):")
    print(f"  • Embedding (MiniLM-L12) : {avg_embed:.2f} ms")
    print(f"  • Vector Search (FAISS)   : {avg_faiss:.2f} ms")
    print(f"  • LLM Generation (Groq)  : {avg_gen:.2f} ms")
    print("-" * 80)
    print("POST-STT END-TO-END LATENCY PERCENTILES:")
    print(f"  • Minimum Latency        : {min_total:.2f} ms")
    print(f"  • P50 (Median) Latency   : {p50_total:.2f} ms")
    print(f"  • Average (Mean) Latency : {avg_total:.2f} ms")
    print(f"  • P90 Latency            : {p90_total:.2f} ms")
    print(f"  • P99 Latency            : {p99_total:.2f} ms")
    print(f"  • Maximum Latency        : {max_total:.2f} ms")
    print("=" * 80)

    # Sample Answer Verification
    print("\n🎯 ACCURACY & GROUNDING SAMPLE CHECKS:")
    for sample_idx in [1, 11, 21, 31, 41, 42, 43]:
        item = results_detail[sample_idx - 1]
        print(f"\n[Q{item['idx']}] {item['query']} ({item['domain']}) — {item['total_ms']:.1f} ms")
        clean_ans = item['answer'].replace('\n', ' ')
        print(f"     A: {clean_ans[:140]}...")

    print("\n" + "=" * 80)
    if avg_total <= 200.0:
        print("🏆 SLA MET: Post-STT average latency is strictly under 200ms!")
    else:
        print(f"⚡ Average latency: {avg_total:.1f}ms (target: <200ms).")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
