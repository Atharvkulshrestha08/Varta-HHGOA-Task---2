"""
VartaLaap — Async Supabase Database Client

Handles secure, non-blocking background logging of:
1. User Voice & Text Queries
2. AI Answers & Source Citations
3. Latency Metrics & Stage Breakdowns
4. HITL User Feedback (Thumbs Up / Thumbs Down)
"""

import os
import logging
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger(__name__)


class SupabaseDBClient:
    """
    Ultra-lightweight, non-blocking Supabase database client powered by HTTPX.
    Executes background writes to Supabase REST endpoints without adding latency to user queries.
    """

    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.publishable_key = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
        self.secret_key = os.getenv("SUPABASE_SECRET_KEY", "")
        self._client: Optional[httpx.AsyncClient] = None

        if self.supabase_url and self.secret_key and not self.supabase_url.startswith("https://your-project-id"):
            logger.info(f"[OK] Supabase Secure DB Initialized -> {self.supabase_url}")
        else:
            logger.warning("[NOTICE] Supabase URL not configured or set to placeholder. Local logging active.")

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {
                "apikey": self.secret_key or self.publishable_key,
                "Authorization": f"Bearer {self.secret_key or self.publishable_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            }
            self._client = httpx.AsyncClient(timeout=3.0, headers=headers)
        return self._client

    async def log_user_query(
        self,
        query_id: str,
        original_query: str,
        answer: str,
        language: str,
        zone: str,
        confidence: float,
        latency_ms: Dict[str, float],
        total_latency_ms: float,
        session_id: Optional[str] = None,
    ) -> bool:
        """Background log query interaction into Supabase 'user_queries' table."""
        if not self.supabase_url or self.supabase_url.startswith("https://your-project-id"):
            return False

        try:
            client = self._get_client()
            endpoint = f"{self.supabase_url}/rest/v1/user_queries"
            payload = {
                "query_id": query_id,
                "session_id": session_id or "anonymous",
                "original_query": original_query,
                "answer": answer,
                "language": language,
                "zone": zone,
                "confidence": confidence,
                "retrieval_latency_ms": round(
                    (latency_ms.get("guardrails_input", 0) + 
                     latency_ms.get("embedding", 0) + 
                     latency_ms.get("retrieval", 0) + 
                     latency_ms.get("guardrails_retrieval", 0)), 2
                ),
                "generation_latency_ms": round(latency_ms.get("generation", 0), 2),
                "total_latency_ms": round(total_latency_ms, 2),
            }
            res = await client.post(endpoint, json=payload)
            return res.status_code in (200, 201)
        except Exception as e:
            logger.debug(f"Supabase background query log skipped: {e}")
            return False

    async def log_user_feedback(self, query_id: str, is_correct: bool, feedback_text: str = "") -> bool:
        """Background log HITL user feedback into Supabase 'user_feedback' table."""
        if not self.supabase_url or self.supabase_url.startswith("https://your-project-id"):
            return False

        try:
            client = self._get_client()
            endpoint = f"{self.supabase_url}/rest/v1/user_feedback"
            payload = {
                "query_id": query_id,
                "is_correct": is_correct,
                "feedback_text": feedback_text,
            }
            res = await client.post(endpoint, json=payload)
            return res.status_code in (200, 201)
        except Exception as e:
            logger.debug(f"Supabase feedback log skipped: {e}")
            return False

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Global singleton instance
supabase_db = SupabaseDBClient()
