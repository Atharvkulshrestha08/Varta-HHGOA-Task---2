"""
Latency Analytics Engine

Tracks per-stage latency for every pipeline invocation and computes
P50 / P70 / P100 percentile metrics across a rolling window of queries.

Stages tracked:
- stt: Speech-to-text (Sarvam API)
- embedding: Query → vector conversion
- retrieval: FAISS similarity search
- reranking: Post-retrieval re-scoring
- guardrails: Safety/relevance checks
- generation: Gemini Flash answer generation
- hallucination_check: Post-generation verification
- total_pipeline: End-to-end (excluding STT)
"""

import time
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StageLatency:
    """Latency measurement for a single pipeline stage."""
    stage: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PipelineLatencyRecord:
    """Complete latency record for one pipeline invocation."""
    query_id: str
    stages: dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0
    total_without_stt_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class LatencyTimer:
    """Context manager for timing a pipeline stage."""

    def __init__(self):
        self.start_ns: int = 0
        self.end_ns: int = 0
        self.duration_ms: float = 0.0

    def __enter__(self):
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        self.end_ns = time.perf_counter_ns()
        self.duration_ms = (self.end_ns - self.start_ns) / 1_000_000  # ns → ms


class LatencyAnalytics:
    """
    Rolling-window latency analytics with percentile computation.

    Usage:
        analytics = LatencyAnalytics(window_size=1000)

        # During a pipeline run:
        record = analytics.start_record("query_123")
        with analytics.time_stage(record, "embedding"):
            embedding = model.encode(query)
        with analytics.time_stage(record, "retrieval"):
            results = index.search(embedding)
        analytics.finish_record(record)

        # Get stats:
        stats = analytics.get_stats()
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        # Per-stage latency histories (rolling window)
        self._stage_latencies: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        # Full pipeline records
        self._records: deque[PipelineLatencyRecord] = deque(maxlen=window_size)
        self._total_queries: int = 0

    def start_record(self, query_id: str) -> PipelineLatencyRecord:
        """Create a new latency record for a pipeline run."""
        return PipelineLatencyRecord(query_id=query_id)

    def time_stage(self, record: PipelineLatencyRecord, stage: str) -> LatencyTimer:
        """
        Context manager that times a stage and records it.

        Usage:
            with analytics.time_stage(record, "retrieval") as timer:
                results = search(query)
            # timer.duration_ms is now available
        """
        timer = _RecordingTimer(record, stage, self._stage_latencies)
        return timer

    def record_stage(
        self, record: PipelineLatencyRecord, stage: str, duration_ms: float
    ):
        """Manually record a stage duration (for pre-measured stages)."""
        record.stages[stage] = duration_ms
        self._stage_latencies[stage].append(duration_ms)

    def finish_record(self, record: PipelineLatencyRecord):
        """Finalize a pipeline record and compute totals."""
        record.total_ms = sum(record.stages.values())
        record.total_without_stt_ms = sum(
            v for k, v in record.stages.items() if k != "stt"
        )
        self._records.append(record)
        self._stage_latencies["total_pipeline"].append(record.total_without_stt_ms)
        self._total_queries += 1

    def _percentile(self, data: list[float], p: int) -> float:
        """Compute the p-th percentile of a sorted list."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100)
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[f]
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    def get_stats(self) -> dict:
        """
        Return comprehensive latency statistics.

        Returns dict with:
        - per_stage: { stage_name: { p50, p70, p100, mean, count } }
        - total_pipeline: { p50, p70, p100, mean }
        - total_queries: int
        - meets_target: bool (P50 < 200ms)
        """
        result = {
            "per_stage": {},
            "total_pipeline": {},
            "total_queries": self._total_queries,
            "window_size": self.window_size,
        }

        for stage, latencies in self._stage_latencies.items():
            data = list(latencies)
            if not data:
                continue
            stage_stats = {
                "p50": round(self._percentile(data, 50), 2),
                "p70": round(self._percentile(data, 70), 2),
                "p100": round(max(data), 2),
                "mean": round(statistics.mean(data), 2),
                "min": round(min(data), 2),
                "count": len(data),
            }
            if stage == "total_pipeline":
                result["total_pipeline"] = stage_stats
                result["meets_target_200ms"] = stage_stats["p50"] < 200
            else:
                result["per_stage"][stage] = stage_stats

        return result

    def get_recent_records(self, n: int = 20) -> list[dict]:
        """Get the N most recent pipeline records."""
        records = list(self._records)[-n:]
        return [
            {
                "query_id": r.query_id,
                "stages": r.stages,
                "total_ms": round(r.total_ms, 2),
                "total_without_stt_ms": round(r.total_without_stt_ms, 2),
                "timestamp": r.timestamp,
            }
            for r in records
        ]


class _RecordingTimer(LatencyTimer):
    """Timer that auto-records to a PipelineLatencyRecord on exit."""

    def __init__(
        self,
        record: PipelineLatencyRecord,
        stage: str,
        stage_latencies: dict[str, deque],
    ):
        super().__init__()
        self._record = record
        self._stage = stage
        self._stage_latencies = stage_latencies

    def __exit__(self, *args):
        super().__exit__(*args)
        self._record.stages[self._stage] = self.duration_ms
        self._stage_latencies[self._stage].append(self.duration_ms)
