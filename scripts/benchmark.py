"""
Latency Benchmark Script

Runs N test queries through the pipeline and reports
P50 / P70 / P100 latency statistics.

Usage:
    python scripts/benchmark.py --queries 100
    python scripts/benchmark.py --queries 50 --output results/report.json
"""

import sys
import os
import json
import time
import asyncio
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vector_store import VectorStore
from app.analytics import LatencyAnalytics
from app.guardrails import GuardrailsEngine
from app.harness import PipelineHarness, QueryRequest
from app.generator import MockGenerator
from app.stt import MockSTTClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Test queries across languages
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
    # Bengali
    "ভারতের রাজধানী কি?",
    "পৃথিবী কিভাবে ঘোরে?",
    "ডেঙ্গু জ্বরের লক্ষণ কি?",
    # Tamil
    "இந்தியாவின் தலைநகரம் என்ன?",
    "பூமி எப்படி சுழல்கிறது?",
    # Telugu
    "భారతదేశ రాజధాని ఏమిటి?",
    "భూమి ఎలా తిరుగుతుంది?",
]


async def run_benchmark(num_queries: int, output_path: str = None):
    """Run benchmark queries and report latency stats."""
    logger.info("=" * 60)
    logger.info(f"VoiceRAG Latency Benchmark — {num_queries} queries")
    logger.info("=" * 60)

    # Load index
    data_dir = Path(__file__).parent.parent / "data"
    index_path = str(data_dir / "faiss_index.bin")
    metadata_path = str(data_dir / "chunks_metadata.json")

    vector_store = VectorStore()
    if not vector_store.load(index_path, metadata_path):
        logger.error("Failed to load FAISS index. Run build_index.py first.")
        sys.exit(1)

    # Initialize pipeline with mock external services (to measure retrieval latency only)
    analytics = LatencyAnalytics(window_size=num_queries)
    guardrails = GuardrailsEngine()

    # Use mock generator to isolate retrieval latency
    # For full pipeline benchmark, set real API keys
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        from app.generator import GeminiGenerator
        generator = GeminiGenerator(api_key=gemini_key)
        logger.info("Using real Gemini generator")
    else:
        generator = MockGenerator()
        logger.info("Using mock generator (set GEMINI_API_KEY for full benchmark)")

    harness = PipelineHarness(
        vector_store=vector_store,
        stt_client=MockSTTClient(),
        generator=generator,
        analytics=analytics,
        guardrails=guardrails,
    )

    # Run queries
    logger.info(f"\nRunning {num_queries} queries...\n")
    results = []

    for i in range(num_queries):
        query = TEST_QUERIES[i % len(TEST_QUERIES)]
        request = QueryRequest(text=query, top_k=5)

        start = time.perf_counter_ns()
        response = await harness.process_text_query(request)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

        results.append({
            "query": query,
            "total_ms": elapsed_ms,
            "latency_breakdown": response.latency_ms,
            "pipeline_latency_ms": response.total_latency_ms,
            "success": response.success,
            "confidence": response.confidence,
        })

        if (i + 1) % 10 == 0:
            logger.info(f"  Completed {i + 1}/{num_queries} queries")

    # Report
    stats = analytics.get_stats()
    pipeline = stats.get("total_pipeline", {})

    logger.info("\n" + "=" * 60)
    logger.info("📊 BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"\nTotal queries: {num_queries}")
    logger.info(f"\n{'─' * 40}")
    logger.info(f"  P50  (median):  {pipeline.get('p50', 'N/A')} ms")
    logger.info(f"  P70  (70th %%):  {pipeline.get('p70', 'N/A')} ms")
    logger.info(f"  P100 (worst):   {pipeline.get('p100', 'N/A')} ms")
    logger.info(f"  Mean:           {pipeline.get('mean', 'N/A')} ms")
    logger.info(f"  Min:            {pipeline.get('min', 'N/A')} ms")
    logger.info(f"{'─' * 40}")

    target_met = pipeline.get('p50', float('inf')) < 200
    logger.info(f"\n  {'✅' if target_met else '❌'} P50 < 200ms target: {'MET' if target_met else 'NOT MET'}")

    logger.info(f"\nPer-stage breakdown:")
    for stage, s in stats.get("per_stage", {}).items():
        logger.info(f"  {stage:25s} P50={s['p50']:8.2f}ms  P70={s['p70']:8.2f}ms  P100={s['p100']:8.2f}ms")

    # Save report
    if output_path:
        report = {
            "config": {
                "num_queries": num_queries,
                "total_vectors": vector_store.total_vectors,
            },
            "summary": {
                "p50_ms": pipeline.get("p50"),
                "p70_ms": pipeline.get("p70"),
                "p100_ms": pipeline.get("p100"),
                "mean_ms": pipeline.get("mean"),
                "min_ms": pipeline.get("min"),
                "target_200ms_met": target_met,
            },
            "per_stage": stats.get("per_stage", {}),
            "individual_results": results,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"\n📄 Full report saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceRAG Latency Benchmark")
    parser.add_argument("--queries", type=int, default=100, help="Number of test queries")
    parser.add_argument("--output", type=str, default="results/latency_report.json", help="Output report path")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.queries, args.output))
