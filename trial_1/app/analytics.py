"""
Latency Analytics Engine — Trial 1 (Sub-200ms Optimization)

Tracks per-stage latency for every pipeline invocation and computes
P50 / P75 / P90 / P99 / P100 percentile metrics across a rolling window of queries.

Stages tracked:
- stt: Speech-to-text (Sarvam API)
- embedding: Query → vector conversion (< 1ms)
- retrieval: FAISS similarity search (< 1ms)
- guardrails: Fast regex safety checks (< 0.1ms)
- generation: Ultra-low token Groq generation (< 120ms)
- total_pipeline: Post-STT End-to-End latency (Target < 200ms across P50/P75/P90/P99/P100)
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
    Rolling-window latency analytics with full P50/P75/P90/P99/P100 percentile computation.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._stage_latencies: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._records: deque[PipelineLatencyRecord] = deque(maxlen=window_size)
        self._total_queries: int = 0

    def start_record(self, query_id: str) -> PipelineLatencyRecord:
        """Create a new latency record for a pipeline run."""
        return PipelineLatencyRecord(query_id=query_id)

    def time_stage(self, record: PipelineLatencyRecord, stage: str) -> LatencyTimer:
        return _RecordingTimer(record, stage, self._stage_latencies)

    def record_stage(
        self, record: PipelineLatencyRecord, stage: str, duration_ms: float
    ):
        record.stages[stage] = duration_ms
        self._stage_latencies[stage].append(duration_ms)

    def finish_record(self, record: PipelineLatencyRecord):
        record.total_ms = sum(record.stages.values())
        record.total_without_stt_ms = sum(
            v for k, v in record.stages.items() if k != "stt"
        )
        self._records.append(record)
        self._stage_latencies["total_pipeline"].append(record.total_without_stt_ms)
        self._total_queries += 1

    def _percentile(self, data: list[float], p: float) -> float:
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
        """Return comprehensive latency statistics across all percentiles."""
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
                "p75": round(self._percentile(data, 75), 2),
                "p90": round(self._percentile(data, 90), 2),
                "p99": round(self._percentile(data, 99), 2),
                "p100": round(max(data), 2),
                "mean": round(statistics.mean(data), 2),
                "min": round(min(data), 2),
                "count": len(data),
            }
            if stage == "total_pipeline":
                result["total_pipeline"] = stage_stats
                result["meets_all_200ms"] = stage_stats["p100"] < 200.0
                result["meets_p50_200ms"] = stage_stats["p50"] < 200.0
            else:
                result["per_stage"][stage] = stage_stats

        return result

    def get_recent_records(self, n: int = 20) -> list[dict]:
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
