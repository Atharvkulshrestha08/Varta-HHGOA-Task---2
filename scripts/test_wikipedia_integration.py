"""
Multilingual Wikipedia Intelligence Verification Suite
Tests:
1. Live Wikipedia retriever across English, Hindi, Tamil, Telugu, and Bengali
2. Direct API endpoint /api/wiki/search
3. Pipeline text query with live Wikipedia grounding & FAISS dynamic caching
"""

import sys
import os
import asyncio
from pathlib import Path

# Force UTF-8 encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.wikipedia_retriever import WikipediaRetriever
from app.main import app
from fastapi.testclient import TestClient


async def test_retriever():
    print("\n" + "=" * 60)
    print("1. Testing WikipediaRetriever across Indic Languages")
    print("=" * 60)

    retriever = WikipediaRetriever(timeout=5.0)

    test_queries = [
        ("APJ Abdul Kalam", "eng_Latn", "en"),
        ("ताजमहल", "hin_Deva", "hi"),
        ("रवीन्द्रनाथ ठाकुर", "ben_Beng", "bn"),
        ("தமிழ்நாடு", "tam_Taml", "ta"),
        ("హైదరాబాదు", "tel_Telu", "te"),
        ("What is Quantum Computing?", "eng_Latn", "en"),
    ]

    for q, lang_script, expected_lang in test_queries:
        print(f"\nSearching for '{q}' in {lang_script}...")
        res = await retriever.fetch_topic_summary(q, language=lang_script)
        if res:
            print(f"   [OK] Title: {res.get('title')}")
            print(f"   [OK] URL: {res.get('url')}")
            print(f"   [OK] Language: {res.get('language')}")
            print(f"   [OK] Extract ({len(res.get('extract', ''))} chars): {res.get('extract', '')[:120]}...")
        else:
            print(f"   [WARN] No result returned for '{q}'")

    await retriever.close()


def test_api_pipeline():
    print("\n" + "=" * 60)
    print("2. Testing FastAPI Wikipedia & Pipeline Endpoints")
    print("=" * 60)

    with TestClient(app) as client:
        # Test 1: Direct Wikipedia API
        print("\n POST /api/wiki/search ('ISRO')")
        resp = client.post("/api/wiki/search", json={"query": "ISRO", "language": "eng_Latn"})
        print(f"   Status: {resp.status_code}")
        data = resp.json()
        print(f"   Response: {data.get('success')}, Title: {data.get('data', {}).get('title')}")
        assert resp.status_code == 200
        assert data.get("success") is True

        # Test 2: Text Query with Live Wikipedia Grounding
        print("\n POST /api/query/text ('Who was APJ Abdul Kalam?')")
        resp = client.post("/api/query/text", json={"text": "Who was APJ Abdul Kalam?", "top_k": 3})
        print(f"   Status: {resp.status_code}")
        q_data = resp.json()
        print(f"   Answer: {q_data.get('answer')[:180]}...")
        print(f"   Sources: {len(q_data.get('sources', []))}")
        for s in q_data.get("sources", []):
            print(f"     * [{s.get('strategy')}] {s.get('source')}: {s.get('url')}")
        assert resp.status_code == 200
        assert len(q_data.get("sources", [])) > 0

        # Test 3: Indic Text Query with Wikipedia Grounding
        print("\n POST /api/query/text ('ताजमहल किसने बनवाया था?')")
        resp = client.post("/api/query/text", json={"text": "ताजमहल किसने बनवाया था?", "top_k": 3})
        print(f"   Status: {resp.status_code}")
        h_data = resp.json()
        print(f"   Answer: {h_data.get('answer')[:200]}...")
        print(f"   Sources: {len(h_data.get('sources', []))}")
        assert resp.status_code == 200

    print("\n[OK] All Wikipedia Integration Tests Completed Successfully!")


if __name__ == "__main__":
    asyncio.run(test_retriever())
    test_api_pipeline()
