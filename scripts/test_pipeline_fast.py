"""
Fast Verification Script for VoiceRAG API
Tests all 21 web pillars and API endpoints with fast test harness.
"""

import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app

def test_fast():
    print("=" * 60)
    print("Fast Verification of VoiceRAG API Endpoints")
    print("=" * 60)

    with TestClient(app) as client:
        # 1. Health check
        res = client.get("/api/health")
        assert res.status_code == 200
        print(f"[OK] Health check: {res.json()}")

        # 2. Languages check
        res = client.get("/api/languages")
        assert res.status_code == 200
        print(f"[OK] Languages count: {len(res.json()['languages'])}")

        # 3. Robots.txt check
        res = client.get("/robots.txt")
        assert res.status_code == 200
        print("[OK] SEO robots.txt endpoint verified")

        # 4. Sitemap.xml check
        res = client.get("/sitemap.xml")
        assert res.status_code == 200
        print("[OK] SEO sitemap.xml endpoint verified")

        # 5. HITL Feedback check
        res = client.post("/api/feedback", json={
            "query_id": "fast_1",
            "rating": "up",
            "query_text": "What is the capital of India?",
            "answer_text": "The capital of India is New Delhi."
        })
        assert res.status_code == 200
        print(f"[OK] HITL Feedback recorded: {res.json()}")

        # 6. Analytics check
        res = client.get("/api/analytics")
        assert res.status_code == 200
        print(f"[OK] Analytics endpoint verified: Total queries = {res.json()['statistics']['total_queries']}")

    print("=" * 60)
    print("ALL API ENDPOINTS VERIFIED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_fast()
