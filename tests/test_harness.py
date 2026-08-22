"""
Tests for Pipeline Harness & Dead-End Fallback Architecture
"""
import pytest
import asyncio
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.harness import PipelineHarness, CircuitBreaker, QueryRequest
from app.analytics import LatencyAnalytics
from app.guardrails import GuardrailsEngine
from app.stt import MockSTTClient
from app.generator import MockGenerator


class MockVectorStore:
    """Fast mock vector store for unit testing harness fallbacks."""

    def __init__(self, mock_results: list[dict] = None):
        self._mock_results = mock_results or []
        self.total_vectors = len(self._mock_results)

    @property
    def is_ready(self) -> bool:
        return True

    def encode_query(self, query: str) -> np.ndarray:
        return np.zeros((1, 384), dtype=np.float32)

    def search(self, query_vector: np.ndarray, top_k: int = 5, score_threshold: float = 0.0, allowed_languages: set = None, **kwargs) -> list[dict]:
        return self._mock_results[:top_k]


def test_circuit_breaker_flow():
    """Test circuit breaker state transitions (closed -> open -> half_open)."""
    cb = CircuitBreaker(threshold=3, reset_timeout=0.2)
    assert not cb.is_open, "Initial state should be closed"

    cb.record_failure()
    cb.record_failure()
    assert not cb.is_open, "Should stay closed under threshold"

    cb.record_failure()
    assert cb.is_open, "Should trip open after 3 failures"

    # Wait for reset timeout
    import time
    time.sleep(0.25)

    assert not cb.is_open, "Should transition to half_open after timeout"
    cb.record_success()
    assert not cb.is_open, "Should close after success"
    print("  [OK] Circuit breaker transition tests passed")


def test_harness_low_confidence_fallback():
    """Test that queries with retrieval confidence < 0.45 trigger low_confidence fallback."""
    async def _run():
        vector_store = MockVectorStore([
            {"text": "Photosynthesis process.", "score": 0.10, "rank": 0}
        ])

        harness = PipelineHarness(
            vector_store=vector_store,
            stt_client=MockSTTClient(),
            generator=MockGenerator(),
            analytics=LatencyAnalytics(),
            guardrails=GuardrailsEngine(min_retrieval_score=0.35),
        )

        req = QueryRequest(text="What is quantum gravity in string theory?")
        response = await harness.process_text_query(req)

        assert response.answer and len(response.answer) > 5
        print("  [OK] Low confidence / open-domain response test passed")

    asyncio.run(_run())


def test_harness_llm_circuit_fallback():
    """Test raw context card fallback when LLM circuit breaker trips."""
    async def _run():
        vector_store = MockVectorStore([
            {"text": "New Delhi is the capital of India.", "score": 0.90, "rank": 0}
        ])

        harness = PipelineHarness(
            vector_store=vector_store,
            stt_client=MockSTTClient(),
            generator=MockGenerator(),
            analytics=LatencyAnalytics(),
            guardrails=GuardrailsEngine(min_retrieval_score=0.10),
        )

        # Trip LLM circuit breaker
        harness.llm_circuit.record_failure()
        harness.llm_circuit.record_failure()
        harness.llm_circuit.record_failure()
        assert harness.llm_circuit.is_open

        req = QueryRequest(text="What is the capital of India?")
        response = await harness.process_text_query(req)

        assert response.is_fallback
        assert response.fallback_tier == "llm_circuit_breaker"
        assert "temporarily unavailable" in response.answer or "Passage" in response.answer
        print("  [OK] LLM circuit breaker raw context fallback test passed")

    asyncio.run(_run())


if __name__ == "__main__":
    print("\nRunning Pipeline Harness & Fallback Tests...\n")
    test_circuit_breaker_flow()
    asyncio.run(test_harness_low_confidence_fallback())
    asyncio.run(test_harness_llm_circuit_fallback())
    print("\nAll harness & fallback tests passed!\n")
