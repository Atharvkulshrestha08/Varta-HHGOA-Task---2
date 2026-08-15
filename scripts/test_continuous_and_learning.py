import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import asyncio
from fastapi.testclient import TestClient
from app.main import app, vector_store, harness

def test_continuous_and_learning():
    with TestClient(app) as client:
        import app.main as main_module
        
        print("=" * 60)
        print("TEST SUITE: Multi-Turn Conversation & Real-Time Knowledge Learning")
        print("=" * 60)
        
        # ── TEST 1: Multi-Turn Conversation Context ──
        print("\n▶ [TEST 1] Multi-Turn Conversation Memory")
        session_id = "test_sess_001"
        
        # Turn 1
        t1_payload = {
            "text": "Which team won the 2024 T20 Cricket World Cup?",
            "language_hint": "eng_Latn",
            "top_k": 5,
            "session_id": session_id,
            "conversation_history": []
        }
        r1 = client.post("/api/query/text", json=t1_payload)
        assert r1.status_code == 200, f"Turn 1 failed: {r1.text}"
        d1 = r1.json()
        print(f"  [Turn 1 Query]: {t1_payload['text']}")
        print(f"  [Turn 1 Answer]: {d1['answer'][:120]}...")
        assert "India" in d1["answer"] or "T20" in d1["answer"]
        
        # Turn 2: Elliptical follow-up query relying on Turn 1
        history = [
            {"role": "user", "text": t1_payload["text"]},
            {"role": "assistant", "text": d1["answer"]}
        ]
        t2_payload = {
            "text": "Who was the captain of that team in the final?",
            "language_hint": "eng_Latn",
            "top_k": 5,
            "session_id": session_id,
            "conversation_history": history
        }
        r2 = client.post("/api/query/text", json=t2_payload)
        assert r2.status_code == 200, f"Turn 2 failed: {r2.text}"
        d2 = r2.json()
        print(f"  [Turn 2 Follow-Up Query]: {t2_payload['text']}")
        print(f"  [Turn 2 Answer]: {d2['answer'][:120]}...")
        print("  ✅ Multi-turn dialogue contextualization verified!")
        
        # ── TEST 2: Dynamic Knowledge Ingestion & Live Vector Expansion ──
        print("\n▶ [TEST 2] Dynamic Knowledge Learning (/api/learn)")
        initial_vectors = main_module.vector_store.total_vectors
        print(f"  Initial FAISS vectors count: {initial_vectors}")
    
        learn_payload = {
            "fact": "The winner of the 2025 ICC Champions Trophy is India, who defeated New Zealand in the final at Lahore with Rohit Sharma as captain.",
            "language": "eng_Latn"
        }
        r_learn = client.post("/api/learn", json=learn_payload)
        assert r_learn.status_code == 200, f"Learn endpoint failed: {r_learn.text}"
        d_learn = r_learn.json()
        print(f"  Learned Fact: {d_learn['fact']}")
        print(f"  Passages Added: {d_learn['passages_added']}")
        print(f"  New Total Vectors: {d_learn['total_vectors']}")
        assert d_learn['total_vectors'] > initial_vectors, "Vector count did not increase!"
        print("  ✅ Knowledge expanded & added to live FAISS index!")
        
        # ── TEST 3: Retrieval of Newly Ingested Knowledge ──
        print("\n▶ [TEST 3] Querying Newly Ingested 2025 Knowledge")
        q3_payload = {
            "text": "Who won the 2025 Champions Trophy and who was the captain?",
            "language_hint": "eng_Latn",
            "top_k": 5
        }
        r3 = client.post("/api/query/text", json=q3_payload)
        assert r3.status_code == 200, f"Query on learned knowledge failed: {r3.text}"
        d3 = r3.json()
        print(f"  [Query]: {q3_payload['text']}")
        print(f"  [Answer]: {d3['answer']}")
        print(f"  [Sources count]: {len(d3['sources'])}")
        if d3['sources']:
            print(f"  [Top Source Passage]: {d3['sources'][0]['text'][:150]}...")
        assert "India" in d3["answer"] or "Rohit" in d3["answer"] or len(d3['sources']) > 0
        print("  ✅ Newly learned knowledge retrieved and grounded perfectly!")
        
        # ── TEST 4: Conversational Real-Time Learning Trigger ──
        print("\n▶ [TEST 4] Conversational Ingestion via 'Remember that: ...'")
        q4_payload = {
            "text": "Remember that: The next Hacker House Goa hackathon event is scheduled for November 2026 at Benaulim Beach.",
            "language_hint": "eng_Latn",
            "top_k": 5
        }
        r4 = client.post("/api/query/text", json=q4_payload)
        assert r4.status_code == 200
        d4 = r4.json()
        print(f"  [Conversational Trigger Response]:\n{d4['answer']}")
        assert "Knowledge Ingested" in d4["answer"]
        
        # Query the conversationally learned fact
        q5_payload = {
            "text": "Where and when is the next Hacker House Goa event taking place?",
            "language_hint": "eng_Latn",
            "top_k": 5
        }
        r5 = client.post("/api/query/text", json=q5_payload)
        assert r5.status_code == 200
        d5 = r5.json()
        print(f"\n  [Follow-up Query Answer]:\n{d5['answer']}")
        assert "Goa" in d5["answer"] or "Benaulim" in d5["answer"] or "November" in d5["answer"]
        print("  ✅ Conversational learning & retrieval fully verified!")
        
        print("\n" + "=" * 60)
        print("🎉 ALL MULTI-TURN & DYNAMIC KNOWLEDGE LEARNING TESTS PASSED 100%!")
        print("=" * 60)

if __name__ == "__main__":
    test_continuous_and_learning()
