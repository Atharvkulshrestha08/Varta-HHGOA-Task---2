"""
Tests for Latency Analytics Engine & SLA Target Verification
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.analytics import LatencyAnalytics, PipelineLatencyRecord


def test_latency_analytics_record_and_percentiles():
    """Test percentile calculations (P50, P70, P100) across stage timers."""
    analytics = LatencyAnalytics(window_size=100)

    # Record 10 simulated query runs with known latencies: 10, 20, ..., 100 ms
    for i in range(1, 11):
        rec = analytics.start_record(f"q_{i}")
        analytics.record_stage(rec, "embedding", i * 2.0)
        analytics.record_stage(rec, "retrieval", i * 3.0)
        analytics.record_stage(rec, "generation", i * 5.0)
        analytics.finish_record(rec)

    stats = analytics.get_stats()

    assert stats["total_queries"] == 10
    total_stats = stats["total_pipeline"]

    # Total latencies are 10, 20, 30, ..., 100
    assert total_stats["min"] == 10.0
    assert total_stats["p100"] == 100.0
    assert 50.0 <= total_stats["p50"] <= 60.0
    assert stats["meets_target_200ms"], "P50 under 200ms should satisfy SLA badge"
    print("  [OK] Analytics percentile calculation tests passed")


def test_stage_context_timer():
    """Test stage context manager timing accuracy."""
    analytics = LatencyAnalytics()
    rec = analytics.start_record("q_timer")

    with analytics.time_stage(rec, "dummy_stage"):
        time.sleep(0.05)  # 50ms sleep

    analytics.finish_record(rec)

    assert "dummy_stage" in rec.stages
    duration = rec.stages["dummy_stage"]
    assert 40.0 <= duration <= 100.0, f"Expected ~50ms duration, got {duration:.1f}ms"
    print("  [OK] Stage context manager timing tests passed")


if __name__ == "__main__":
    print("\nRunning Latency Analytics & SLA Tests...\n")
    test_latency_analytics_record_and_percentiles()
    test_stage_context_timer()
    print("\nAll latency analytics tests passed!\n")
