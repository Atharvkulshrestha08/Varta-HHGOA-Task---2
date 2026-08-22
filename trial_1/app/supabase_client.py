"""
VartaLaap — Async Supabase Database Client for trial_1
"""

import os
import logging
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger(__name__)


class SupabaseDBClient:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.publishable_key = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
        self.secret_key = os.getenv("SUPABASE_SECRET_KEY", "")
        self._client: Optional[httpx.AsyncClient] = None

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
        if not self.supabase_url or self.supabase_url.startswith("https://your-project-id"):
            return False
        try:
            client = self._get_client()
            payload = {
                "query_id": query_id,
                "query_text": original_query,
                "answer_text": answer,
                "language": language,
                "confidence": confidence,
                "total_latency_ms": total_latency_ms,
            }
            endpoint = f"{self.supabase_url}/rest/v1/user_queries"
            resp = await client.post(endpoint, json=payload)
            return resp.status_code in (200, 201)
        except Exception:
            return False


supabase_db = SupabaseDBClient()
